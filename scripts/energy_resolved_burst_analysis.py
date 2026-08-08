#!/usr/bin/env python3
"""
energy_resolved_burst_analysis.py

Runs an energy-resolved analysis of confirmed Type-I X-ray bursts across
five LAXPC energy bands: for each burst already confirmed in the
broadband analysis (see burst_timing_parameters.py), this script
independently measures each band's own burst start/peak/decay timing,
then produces per-burst energy-resolved light curve plots and a
Persistent/Peak/Net-peak summary table.

Method
------
Per-band timing (Section "Per-band burst timing" below):
  For each confirmed broadband burst, a search window is anchored on
  that burst's broadband start_time_met (with a small buffer to allow
  a band-specific start slightly earlier than the broadband one). Within
  that window, this band's own preburst mean/std is computed, and the
  first time the rate exceeds preburst_mean + search_sigma * preburst_std
  is taken as this band's start_time. From there, peak_time (first time
  reaching 98% of peak), peak_rate, rise_time, efold_time, and decay_time
  are measured using the same logic as burst_timing_parameters.py's
  burst search, just windowed around a known burst rather than scanning
  the whole observation.

  This anchored approach (rather than a blind re-scan of each band) is
  used because a blind search would rediscover bursts already known,
  and would very plausibly fail the significance threshold in faint
  bands where the burst excess is a small fraction of the persistent
  rate. If a band never crosses its own threshold near the anchor burst
  (common for the faintest band on a weak burst), its record is kept
  with "detected": false and only preburst stats filled in — nothing is
  silently dropped.

  RAW (non background-subtracted) band rates are used throughout, since
  only the burst-free persistent light curve is typically background-
  subtracted in this style of analysis, not the per-band burst curves.

Plots:
  One figure per confirmed burst, showing all energy bands on a shared
  time axis (zeroed to each band's own start time), plus zoomed-in inset
  panels for the faintest bands where the burst signal is small relative
  to the main axis scale. A combined grid figure tiles all bursts
  together. Insets for the faint bands are rebinned to a coarser
  resolution before plotting, purely to reduce visual scatter.

Summary table:
  For each burst x band, reports Persistent (mean rate in the preburst
  window), PeakRate (max rate within the search window), and NetPeak
  (PeakRate - Persistent) — a simple pre-burst vs. in-burst comparison,
  independent of any background file.

Input
-----
--json     : broadband confirmed-burst JSON produced by
             burst_timing_parameters.py --json
             (default: results/burst_analysis_confirmed.json)
--lc-dir   : directory containing the per-band light curve files
             (default: results/lc). Expected filenames follow the
             pattern lightcurve_all_<lo>_<hi>keV.lc for each band in
             BAND_FILES below — edit that dict if your band edges or
             filenames differ.

Output (all under --outdir, default: results/energy_resolved/)
----------------------------------------------------------------
  energy_resolved_burst_analysis.json   per-burst, per-band timing parameters
  energy_resolved_burst_analysis.csv    Persistent/PeakRate/NetPeak summary table
  energy_resolved_burst_analysis.tex    same table, as a LaTeX table
  plots/<burst>_energy_resolved.png/.pdf   one figure per burst
  plots/all_bursts_energy_resolved.png/.pdf   combined grid figure

Requires burst_timing_parameters.py to be in the same directory
(scripts/) so it can be imported directly.

Usage
-----
    python3 energy_resolved_burst_analysis.py \\
        --json results/burst_analysis_confirmed.json \\
        --lc-dir results/lc \\
        --outdir results/energy_resolved
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from burst_timing_parameters import load_lightcurve, rebin_lightcurve

# --------------------------------------------------------------------------
# Band definitions — edit these if your energy bands or filenames differ.
# --------------------------------------------------------------------------
BAND_SUFFIXES = {
    "3.0-9.0":   "lightcurve_all_3_9keV.lc",
    "9.0-15.0":  "lightcurve_all_9_15keV.lc",
    "15.0-21.0": "lightcurve_all_15_21keV.lc",
    "21.0-30.0": "lightcurve_all_21_30keV.lc",
    "30.0-40.0": "lightcurve_all_30_40keV.lc",
}

# Bands shown as zoomed-in insets on the main plot (too faint for the
# main axis scale) and rebinned to a coarser resolution before plotting.
INSET_BANDS = ["21.0-30.0", "30.0-40.0"]

COLORS = {
    "3.0-9.0":   "tab:blue",
    "9.0-15.0":  "tab:orange",
    "15.0-21.0": "tab:green",
    "21.0-30.0": "tab:red",
    "30.0-40.0": "tab:purple",
}

# Inset positions in axes-fraction coordinates [x0, y0, width, height].
INSET_BBOXES = {
    "21.0-30.0": [0.42, 0.74, 0.30, 0.24],
    "30.0-40.0": [0.66, 0.38, 0.30, 0.24],
}

# --------------------------------------------------------------------------
# Defaults (match burst_timing_parameters.py's own defaults)
# --------------------------------------------------------------------------
REBIN_S = 0.5                 # rebin resolution applied to INSET_BANDS only
SEARCH_SIGMA = 4.0
PREBURST_WINDOW_S = 100.0
MAX_SEARCH_S = 60.0
PRE_SEARCH_BUFFER_S = 5.0      # how far before the broadband start a band's
                                # own start is allowed to be found
PEAK_SEARCH_S = 60.0            # window after start used for peak search
PLOT_PRE_S = 3.0                 # seconds of pre-burst baseline shown on plots
PLOT_POST_S = 24.0                # seconds after start shown on plots


def band_label(band):
    """'3.0-9.0' -> '3.0 - 9.0 keV', for legend labels."""
    return band.replace("-", " - ") + r" $keV$"


def _finite_or_none(x):
    return float(x) if (x is not None and np.isfinite(x)) else None


# --------------------------------------------------------------------------
# Per-band burst timing (anchored on the broadband burst)
# --------------------------------------------------------------------------

def band_preburst_stats(t, r, t_ref, window_s):
    """Mean, std, and SEM of rate in [t_ref - window_s, t_ref)."""
    sel = (t >= t_ref - window_s) & (t < t_ref)
    if sel.sum() < 3:
        return np.nan, np.nan, np.nan
    vals = r[sel]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    sem = std / np.sqrt(len(vals))
    return mean, std, sem


def analyze_band_burst(t, r, e, t_broadband_start,
                        preburst_window=PREBURST_WINDOW_S,
                        search_sigma=SEARCH_SIGMA,
                        max_search=MAX_SEARCH_S,
                        pre_buffer=PRE_SEARCH_BUFFER_S):
    """
    Measure this band's own burst start/peak/decay timing, anchored to
    an already-confirmed broadband burst. Returns a dict of parameters;
    "detected" is False (with only preburst stats filled in) if this
    band never crosses its own significance threshold near the anchor.
    """
    pre_mean, pre_std, pre_sem = band_preburst_stats(
        t, r, t_broadband_start, preburst_window)

    result = {
        "detected": False,
        "preburst_rate": _finite_or_none(pre_mean),
        "preburst_rate_err": _finite_or_none(pre_sem),
        "preburst_std": _finite_or_none(pre_std),
        "start_time_met": None,
        "peak_time_met": None,
        "peak_rate": None,
        "peak_rate_err": None,
        "rise_time_s": None,
        "efold_time_s": None,
        "decay_time_s": None,
        "amplitude_ratio": None,
    }
    if not np.isfinite(pre_mean) or not np.isfinite(pre_std) or pre_std == 0:
        return result

    threshold = pre_mean + search_sigma * pre_std
    search_lo = t_broadband_start - pre_buffer
    search_hi = t_broadband_start + max_search
    win = (t >= search_lo) & (t <= search_hi)
    if win.sum() < 2:
        return result

    t_win, r_win, e_win = t[win], r[win], e[win]
    above = np.where(r_win > threshold)[0]
    if len(above) == 0:
        return result

    start_time = t_win[above[0]]

    fwd = t_win >= start_time
    t_fwd = t_win[fwd]
    r_fwd_sub = r_win[fwd] - pre_mean
    e_fwd = e_win[fwd]

    peak_idx_local = np.argmax(r_fwd_sub)
    peak_rate = r_fwd_sub[peak_idx_local]
    peak_rate_err = np.sqrt(np.nan_to_num(e_fwd[peak_idx_local]) ** 2 + pre_sem ** 2)

    if peak_rate <= search_sigma * pre_std:
        return result  # threshold crossing didn't lead to a significant peak

    reach98 = np.where(r_fwd_sub[:peak_idx_local + 1] >= 0.98 * peak_rate)[0]
    peak_idx = reach98[0] if len(reach98) else peak_idx_local
    peak_time = t_fwd[peak_idx]
    rise_time = peak_time - start_time

    after = t_fwd >= peak_time
    t_after = t_fwd[after]
    r_after = r_fwd_sub[after]

    efold_time = np.nan
    below_e = np.where(r_after <= peak_rate / np.e)[0]
    if len(below_e):
        efold_time = t_after[below_e[0]] - peak_time

    decay_time = np.nan
    below_10 = np.where(r_after <= 0.10 * peak_rate)[0]
    if len(below_10):
        decay_time = t_after[below_10[0]] - peak_time

    ratio = (peak_rate + pre_mean) / pre_mean if pre_mean else np.nan

    result.update({
        "detected": True,
        "start_time_met": float(start_time),
        "peak_time_met": float(peak_time),
        "peak_rate": float(peak_rate),
        "peak_rate_err": float(peak_rate_err),
        "rise_time_s": float(rise_time),
        "efold_time_s": _finite_or_none(efold_time),
        "decay_time_s": _finite_or_none(decay_time),
        "amplitude_ratio": _finite_or_none(ratio),
    })
    return result


def band_onset(band_starts_for_burst, band, broadband_start):
    """This band's own start_time_met if detected, else the broadband start."""
    band_result = band_starts_for_burst.get(band)
    if band_result and band_result.get("detected") and band_result.get("start_time_met") is not None:
        return band_result["start_time_met"]
    return broadband_start


def band_stats(t, r, t_onset, preburst_window_s, peak_search_s):
    """Persistent (preburst mean), PeakRate, NetPeak for one band/burst."""
    pre_mask = (t >= t_onset - preburst_window_s) & (t < t_onset)
    persistent = np.mean(r[pre_mask]) if pre_mask.sum() > 0 else np.nan
    peak_mask = (t >= t_onset) & (t <= t_onset + peak_search_s)
    peak_rate = np.max(r[peak_mask]) if peak_mask.sum() > 0 else np.nan
    net_peak = peak_rate - persistent
    return persistent, peak_rate, net_peak


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def make_burst_insets(ax_main):
    """
    Build opaque zoom-in insets for the faint bands at fixed axes-fraction
    positions. Solid white faces + high zorder keep them fully readable
    over whatever part of the main curves sits underneath.
    """
    insets = {}
    for band, bbox in INSET_BBOXES.items():
        iax = ax_main.inset_axes(bbox)
        iax.set_facecolor("white")
        iax.set_zorder(10)
        insets[band] = iax
    return insets


def plot_burst_panel(ax_main, band_data, band_data_inset_rebinned,
                      band_starts_for_burst, broadband_start):
    """Draw the multi-band panel (main axes + inset zoom-ins) for one burst."""
    insets = make_burst_insets(ax_main)

    for band, (t, r, e) in band_data.items():
        t_onset = band_onset(band_starts_for_burst, band, broadband_start)
        mask = (t >= t_onset - PLOT_PRE_S) & (t <= t_onset + PLOT_POST_S)
        t_seg, r_seg, e_seg = t[mask] - t_onset, r[mask], e[mask]

        ax_main.errorbar(t_seg, r_seg, yerr=e_seg, fmt="o", ms=3, ls="--",
                          lw=0.7, elinewidth=0.6, capsize=0, alpha=0.9,
                          color=COLORS[band], label=band_label(band))

        if band in insets:
            iax = insets[band]
            t05, r05, e05 = band_data_inset_rebinned[band]
            mask05 = (t05 >= t_onset - PLOT_PRE_S) & (t05 <= t_onset + PLOT_POST_S)
            t_seg05, r_seg05, e_seg05 = t05[mask05] - t_onset, r05[mask05], e05[mask05]
            iax.errorbar(t_seg05, r_seg05, yerr=e_seg05, fmt="o", ms=2.5, ls="--",
                         lw=0.6, elinewidth=0.5, capsize=0, alpha=0.9,
                         color=COLORS[band], label=band_label(band))
            iax.set_xlabel("Time since burst (s)", fontsize=8)
            iax.set_ylabel("Rate (counts/s)", fontsize=8)
            iax.tick_params(labelsize=7)
            iax.legend(fontsize=7.5, loc="upper right", frameon=True,
                       framealpha=1, facecolor="white", edgecolor="black",
                       handlelength=1.0, borderpad=0.3)

    ax_main.set_xlabel("Time since burst (s)")
    ax_main.set_ylabel("Counts/s LAXPC")
    ax_main.legend(fontsize=9.5, loc="upper right", frameon=True,
                   framealpha=1, facecolor="white", edgecolor="black",
                   handlelength=1.4, borderpad=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="results/burst_analysis_confirmed.json",
                     help="broadband confirmed-burst JSON from burst_timing_parameters.py")
    ap.add_argument("--lc-dir", default="results/lc",
                     help="directory containing the per-band light curve files")
    ap.add_argument("--outdir", default="results/energy_resolved",
                     help="directory to write the output JSON, tables, and plots/ into")
    ap.add_argument("--rebin", type=float, default=REBIN_S,
                     help=f"rebin resolution (s) applied only to INSET_BANDS before "
                          f"their burst search and plotting (default {REBIN_S})")
    ap.add_argument("--search-sigma", type=float, default=SEARCH_SIGMA,
                     help=f"sigma threshold above a band's own preburst rate to "
                          f"flag its start (default {SEARCH_SIGMA})")
    ap.add_argument("--preburst-window", type=float, default=PREBURST_WINDOW_S,
                     help=f"seconds before the search window used for preburst "
                          f"stats (default {PREBURST_WINDOW_S})")
    ap.add_argument("--max-search", type=float, default=MAX_SEARCH_S,
                     help=f"max seconds after the broadband start to search for "
                          f"a band's peak/decay (default {MAX_SEARCH_S})")
    ap.add_argument("--pre-buffer", type=float, default=PRE_SEARCH_BUFFER_S,
                     help=f"seconds before the broadband start a band's own start "
                          f"is allowed to be found (default {PRE_SEARCH_BUFFER_S})")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(args.json) as f:
        payload = json.load(f)
    bursts = payload["confirmed_bursts"]
    preburst_window_s = payload.get("preburst_window_s", PREBURST_WINDOW_S)
    print(f"Loaded {len(bursts)} confirmed broadband burst(s) from {args.json}")

    for i, b in enumerate(bursts, start=1):
        b["bid_label"] = f"B{i}"

    band_files = {band: os.path.join(args.lc_dir, fname)
                  for band, fname in BAND_SUFFIXES.items()}

    # ---- Load all band light curves once (native resolution) ----
    band_data = {}
    for band, path in band_files.items():
        t, r, e = load_lightcurve(path)
        band_data[band] = (t, r, e)
        print(f"Loaded {path}: {len(t)} points, band {band} keV")

    # ---- Rebinned versions of the faint (inset) bands, for search + plotting ----
    band_data_rebinned = {}
    for band in INSET_BANDS:
        t, r, e = band_data[band]
        band_data_rebinned[band] = rebin_lightcurve(t, r, e, args.rebin)
        print(f"Rebinned {band} keV to {args.rebin} s: {len(band_data_rebinned[band][0])} points")

    # ---- Per-burst, per-band timing ----
    band_starts_by_bid = {}
    for b in bursts:
        bid = b["bid"]
        t_broadband_start = b["start_time_met"]
        band_starts_by_bid[bid] = {}
        for band, (t, r, e) in band_data.items():
            t_use, r_use, e_use = (band_data_rebinned[band] if band in INSET_BANDS
                                    else (t, r, e))
            result = analyze_band_burst(
                t_use, r_use, e_use, t_broadband_start,
                preburst_window=args.preburst_window,
                search_sigma=args.search_sigma,
                max_search=args.max_search,
                pre_buffer=args.pre_buffer,
            )
            band_starts_by_bid[bid][band] = result

    # ---- Console summary ----
    print(f"\n{'BID':>3} {'Band (keV)':>11} {'Detected':>8} {'Start(MET,s)':>14} "
          f"{'PeakRate':>10} {'PreRate':>10} {'Rise(s)':>8} {'Decay(s)':>9}")
    print("-" * 78)
    for b in bursts:
        for band, res in band_starts_by_bid[b["bid"]].items():
            start_disp = f"{res['start_time_met']:.3f}" if res["start_time_met"] is not None else "n/a"
            peak_disp = f"{res['peak_rate']:.2f}" if res["peak_rate"] is not None else "n/a"
            pre_disp = f"{res['preburst_rate']:.2f}" if res["preburst_rate"] is not None else "n/a"
            rise_disp = f"{res['rise_time_s']:.3f}" if res["rise_time_s"] is not None else "n/a"
            decay_disp = f"{res['decay_time_s']:.3f}" if res["decay_time_s"] is not None else "n/a"
            print(f"{b['bid']:>3} {band:>11} {str(res['detected']):>8} "
                  f"{start_disp:>14} {peak_disp:>10} {pre_disp:>10} "
                  f"{rise_disp:>8} {decay_disp:>9}")

    # ---- Individual burst plots ----
    for b in bursts:
        fig, ax_main = plt.subplots(figsize=(9, 7))
        plot_burst_panel(ax_main, band_data, band_data_rebinned,
                          band_starts_by_bid[b["bid"]], b["start_time_met"])
        outpng = os.path.join(plots_dir, f"{b['bid_label']}_energy_resolved.png")
        outpdf = os.path.join(plots_dir, f"{b['bid_label']}_energy_resolved.pdf")
        fig.savefig(outpng, dpi=300, bbox_inches="tight")
        fig.savefig(outpdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outpng} and {outpdf}")

    # ---- Combined grid figure ----
    n = len(bursts)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 6.5 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, b in zip(axes.flat, bursts):
        plot_burst_panel(ax, band_data, band_data_rebinned,
                          band_starts_by_bid[b["bid"]], b["start_time_met"])
    for ax in axes.flat[n:]:
        ax.axis("off")
    outpng = os.path.join(plots_dir, "all_bursts_energy_resolved.png")
    outpdf = os.path.join(plots_dir, "all_bursts_energy_resolved.pdf")
    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpng} and {outpdf}")

    # ---- Persistent / PeakRate / NetPeak summary table ----
    rows = []
    for b in bursts:
        for band, (t, r, e) in band_data.items():
            t_onset = band_onset(band_starts_by_bid[b["bid"]], band, b["start_time_met"])
            persistent, peak_rate, net_peak = band_stats(
                t, r, t_onset, preburst_window_s, PEAK_SEARCH_S)
            rows.append({
                "Burst": b["bid_label"],
                "Band (keV)": band,
                "Persistent": persistent,
                "PeakRate": peak_rate,
                "NetPeak": net_peak,
            })

    csv_path = os.path.join(args.outdir, "energy_resolved_burst_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Burst", "Band (keV)", "Persistent",
                                                "PeakRate", "NetPeak"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v)
                              for k, v in row.items()})
    print(f"Saved {csv_path}")

    tex_path = os.path.join(args.outdir, "energy_resolved_burst_analysis.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write("\\renewcommand{\\arraystretch}{1.15}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\\hline\n")
        f.write("Burst & Band (keV) & Persistent (cts~s$^{-1}$) & "
                "Peak Rate (cts~s$^{-1}$) & Net Peak (cts~s$^{-1}$) \\\\\n")
        f.write("\\hline\n")
        for b in bursts:
            for band, (t, r, e) in band_data.items():
                t_onset = band_onset(band_starts_by_bid[b["bid"]], band, b["start_time_met"])
                persistent, peak_rate, net_peak = band_stats(
                    t, r, t_onset, preburst_window_s, PEAK_SEARCH_S)
                f.write(f"{b['bid_label']} & {band} & {persistent:.1f} & "
                        f"{peak_rate:.1f} & {net_peak:.1f} \\\\\n")
            f.write("\\hline\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Persistent, peak, and net-peak count rates in "
                "each energy band for the confirmed bursts. Persistent is "
                "the mean raw rate in the preburst window; Peak Rate is "
                "the maximum raw rate within the search window after burst "
                "start; Net Peak = Peak Rate $-$ Persistent. No background "
                "subtraction is applied to the per-band rates.}\n")
        f.write("\\label{tab:energy_resolved}\n\\end{table}\n")
    print(f"Saved {tex_path}")

    # ---- JSON export: per-burst, per-band timing parameters ----
    out_payload = {
        "broadband_json": args.json,
        "band_files": band_files,
        "inset_bands": INSET_BANDS,
        "rebin_s_for_inset_bands": args.rebin,
        "search_sigma": args.search_sigma,
        "preburst_window_s": args.preburst_window,
        "max_search_s": args.max_search,
        "pre_search_buffer_s": args.pre_buffer,
        "bursts": [
            {
                "bid": b["bid"],
                "broadband_start_time_met": b["start_time_met"],
                "broadband_start_time_mjd": b.get("start_time_mjd"),
                "bands": band_starts_by_bid[b["bid"]],
            }
            for b in bursts
        ],
    }
    json_out_path = os.path.join(args.outdir, "energy_resolved_burst_analysis.json")
    with open(json_out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved per-band, per-burst timing parameters -> {json_out_path}")


if __name__ == "__main__":
    main()
