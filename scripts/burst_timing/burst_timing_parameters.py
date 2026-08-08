#!/usr/bin/env python3
"""
burst_timing_parameters.py

Identifies Type-I X-ray bursts in an AstroSat/LAXPC light curve and
measures their timing parameters, following the burst-identification
and parameter-measurement convention used in Bostanci et al. 2023,
ApJ 958, 55 (Section 2 / Table 1), adapted here for LAXPC data.

Parameter definitions (all computed on the preburst-rate-subtracted
count rate unless noted otherwise):

    preburst_rate   Mean count rate over a configurable window
                    (default 100 s) immediately preceding burst start.
                    Uncertainty is the standard error of that mean.
    start_time      First time (on a 0.5 s binned light curve, by
                    default) the rate exceeds
                    preburst_rate + 4 * sigma_preburst.
    peak_time       First time after start_time the rate reaches
                    98% of the peak rate.
    peak_rate       Maximum preburst-subtracted rate in the burst window.
    rise_time       peak_time - start_time.
    efold_time      Time after peak_time for the rate to fall to peak / e.
    decay_time      Time after peak_time for the rate to fall to 10% of peak.

Usage
-----
    python3 burst_timing_parameters.py --lc lightcurve.lc 

Input
-----
--lc  : light curve file to analyze (FITS or ASCII). In this repo's
        layout, this is typically one of the light curves produced by
        the reduction pipeline (see docs/laxpc_data_reduction.md),
        e.g. results/lc/lightcurve_all_3_80keV_0.1s.lc
--bkg : (optional) matching background light curve, e.g.
        results/lc/Back_lightcurve_3.0_80.0keV.lc

Auto-detects both light curve formats:
  - FITS light curves (XSELECT/HEASoft-style, RATE extension with
    TIME/RATE/ERROR columns)
  - ASCII light curves (LAXPC laxpc_make_lightcurve output: whitespace-
    separated columns, '!' or '#' comment lines, typically Time Rate Error)

Output
------
--json  : (optional) writes confirmed-burst parameters to the given
          path, e.g. results/burst_timing_parameters.json. This is
          the file plot_burst_lightcurves.py expects as its --json
          input, so keep the filename consistent between the two
          scripts if you customize it.
--latex : (optional) prints a publication-style LaTeX table of
          confirmed bursts to stdout.

If neither flag is given, results print to stdout only (candidate
table + confirmed-burst table) and nothing is written to disk.

Notes
-----
- Background subtraction (if --bkg is given) linearly interpolates the
  background rate onto the source time grid and subtracts it BEFORE the
  burst search, giving a net (persistent + burst) light curve. The
  "preburst rate" reported is therefore the persistent accretion
  emission, matching the reference paper's convention — it does not
  include the instrumental background.
- This script performs light-curve-level burst identification only
  (paper Section 2, Table 1). It does not perform time-resolved
  spectroscopy or oscillation searches — those require photon event
  files and GTIs rather than a light curve, and belong in a separate
  pipeline stage.
"""

import argparse
import json
import numpy as np
from scipy.ndimage import uniform_filter1d


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def get_mjd_ref(path):
    """
    Read MJDREFI/MJDREFF/TIMESYS from a FITS light-curve header.

    Returns (mjdref, timesys) as (float, str), or (None, None) if the
    file isn't FITS or the keywords are absent (e.g. an ASCII LAXPC
    .lc file). In that case, pull MJDREFI/MJDREFF from the parent
    event file's header instead and pass them in via --mjdref.
    """
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            for hdu in hdul:
                h = hdu.header
                if "MJDREFI" in h:
                    mjdrefi = h["MJDREFI"]
                    mjdreff = h.get("MJDREFF", 0.0)
                    timesys = h.get("TIMESYS", "UNKNOWN")
                    return float(mjdrefi) + float(mjdreff), timesys
    except Exception:
        pass
    return None, None


def load_lightcurve(path):
    """
    Load a light curve from either a FITS or ASCII file.

    Returns (time, rate, error) as numpy arrays.
    """
    # Try FITS first
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hasattr(hdu.data, "names"):
                    names = [n.upper() for n in hdu.data.names]
                    if "TIME" in names and ("RATE" in names or "COUNTS" in names):
                        data = hdu.data
                        time = np.array(data["TIME"], dtype=float)
                        if "RATE" in names:
                            rate = np.array(data["RATE"], dtype=float)
                        else:
                            rate = np.array(data["COUNTS"], dtype=float)
                        if "ERROR" in names:
                            err = np.array(data["ERROR"], dtype=float)
                        else:
                            err = np.sqrt(np.clip(rate, 0, None))
                        return time, rate, err
        raise ValueError("No TIME/RATE table found in FITS file")
    except OSError:
        pass  # not a FITS file, fall through to ASCII

    # ASCII fallback (typical LAXPC lc output: Time Rate Error, '!'/'#' comments)
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals[:3] if len(vals) >= 3 else vals + [np.nan])
    if not rows:
        raise ValueError(f"Could not parse any numeric rows from {path}")
    arr = np.array(rows, dtype=float)
    time, rate = arr[:, 0], arr[:, 1]
    err = arr[:, 2] if arr.shape[1] > 2 else np.sqrt(np.clip(rate, 0, None))
    return time, rate, err


def rebin_lightcurve(time, rate, err, dt_new):
    """
    Rebin a (possibly unevenly sampled) light curve onto uniform bins
    of width dt_new, using inverse-variance weighting.

    Vectorized with np.bincount for efficiency on long observations
    with fine native time resolution (e.g. a full ~40 hr LAXPC
    observation at 0.1 s sampling).
    """
    t0 = time.min()
    idx = np.floor((time - t0) / dt_new).astype(np.int64)
    n_bins = idx.max() + 1

    good = np.isfinite(rate) & np.isfinite(err)
    idx_g = idx[good]
    rate_g = rate[good]
    err_g = np.clip(err[good], 1e-6, None)

    w = 1.0 / err_g ** 2
    sum_w = np.bincount(idx_g, weights=w, minlength=n_bins)
    sum_wr = np.bincount(idx_g, weights=w * rate_g, minlength=n_bins)
    counts = np.bincount(idx_g, minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        new_rate = sum_wr / sum_w
        new_err = np.sqrt(1.0 / sum_w)

    new_time = t0 + (np.arange(n_bins) + 0.5) * dt_new

    valid = (counts > 0) & np.isfinite(new_rate)
    return new_time[valid], new_rate[valid], new_err[valid]


def background_subtract(time, rate, bkg_time, bkg_rate):
    """Interpolate background rate onto the source time grid and subtract."""
    bkg_interp = np.interp(time, bkg_time, bkg_rate)
    return rate - bkg_interp


# --------------------------------------------------------------------------
# Burst finding
# --------------------------------------------------------------------------

def find_persistent_baseline(time, rate, window_s, smooth_s=None):
    """
    Compute a rolling-median baseline, used only to seed candidate
    burst regions. This is not the final "preburst rate" reported per
    burst — that value is recomputed from a clean window immediately
    before each confirmed burst start.
    """
    dt = np.median(np.diff(time))
    win_pts = max(1, int(round(window_s / dt)))
    baseline = uniform_filter1d(rate, size=win_pts, mode="nearest")
    resid = rate - baseline
    sigma = np.nanstd(resid[np.abs(resid) < 5 * np.nanstd(resid)])
    return baseline, sigma


def preburst_stats(time, rate, start_time, window_s=100.0):
    """Mean and standard error of the rate in [start_time - window_s, start_time)."""
    sel = (time >= start_time - window_s) & (time < start_time)
    if sel.sum() < 3:
        return np.nan, np.nan
    vals = rate[sel]
    mean = np.mean(vals)
    sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return mean, sem


def _sliding_preburst_stats(time, rate, window_s):
    """
    Vectorized rolling mean/std/SEM of `rate` over the trailing
    window_s seconds before each point (exclusive of the point
    itself), using prefix sums and searchsorted for O(n log n)
    performance on long light curves.
    """
    n = len(time)
    cs_r = np.concatenate(([0.0], np.cumsum(rate)))
    cs_r2 = np.concatenate(([0.0], np.cumsum(rate.astype(np.float64) ** 2)))

    lo_idx = np.searchsorted(time, time - window_s, side="left")
    hi_idx = np.arange(n)  # window is [lo_idx, i), i.e. strictly before point i

    counts = hi_idx - lo_idx
    sum_r = cs_r[hi_idx] - cs_r[lo_idx]
    sum_r2 = cs_r2[hi_idx] - cs_r2[lo_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sum_r / counts
        var = (sum_r2 - counts * mean ** 2) / np.clip(counts - 1, 1, None)
        std = np.sqrt(np.clip(var, 0, None))

    valid = counts > 3
    mean = np.where(valid, mean, np.nan)
    std = np.where(valid, std, np.nan)
    sem = np.where(valid, std / np.sqrt(np.clip(counts, 1, None)), np.nan)
    return mean, std, sem


def find_bursts(time, rate, err=None, search_sigma=4.0, preburst_window=100.0,
                 min_separation=20.0, max_burst_search=60.0):
    """
    Scan a light curve for FRED-shaped (fast-rise, exponential-decay)
    excursions above preburst_rate + search_sigma * sigma_preburst.

    `err` (per-bin rate uncertainty) is used to propagate an
    uncertainty on peak_rate: sigma_peak = sqrt(err_at_peak^2 +
    preburst_sem^2), combining the statistical error on the peak bin
    with the uncertainty on the subtracted preburst baseline.

    Returns a list of dicts, one per candidate burst, with Table-1-style
    parameters (unfiltered — see build_confirmed_bursts for filtering).
    """
    n = len(time)
    if err is None:
        err = np.full(n, np.nan)
    bursts = []
    last_burst_end_time = -np.inf

    pre_mean_arr, pre_std_arr, pre_sem_arr = _sliding_preburst_stats(
        time, rate, preburst_window)

    thresh_arr = pre_mean_arr + search_sigma * pre_std_arr
    candidate_mask = np.isfinite(thresh_arr) & (rate > thresh_arr)
    candidate_idx = np.flatnonzero(candidate_mask)

    ci = 0
    while ci < len(candidate_idx):
        i = candidate_idx[ci]
        t = time[i]
        if t - last_burst_end_time < min_separation:
            ci += 1
            continue

        pre_mean = pre_mean_arr[i]
        pre_std = pre_std_arr[i]
        pre_sem = pre_sem_arr[i]
        if not np.isfinite(pre_mean) or not np.isfinite(pre_std) or pre_std == 0:
            ci += 1
            continue

        thresh = thresh_arr[i]
        if rate[i] > thresh:
            start_idx = i
            start_time = time[start_idx]

            # Search forward for peak within max_burst_search seconds
            end_search = start_time + max_burst_search
            win = (time >= start_time) & (time <= end_search)
            if win.sum() < 2:
                ci += 1
                continue
            t_win = time[win]
            r_win = rate[win] - pre_mean  # preburst-subtracted
            e_win = err[win]
            peak_idx_local = np.argmax(r_win)
            peak_rate = r_win[peak_idx_local]
            peak_rate_err = np.sqrt(
                np.nan_to_num(e_win[peak_idx_local]) ** 2 + pre_sem ** 2)

            if peak_rate <= search_sigma * pre_std:
                # Spurious spike, not a real burst
                ci += 1
                continue

            # peak_time = first time reaching 98% of peak
            reach98 = np.where(r_win[:peak_idx_local + 1] >= 0.98 * peak_rate)[0]
            peak_idx = reach98[0] if len(reach98) else peak_idx_local
            peak_time = t_win[peak_idx]
            rise_time = peak_time - start_time

            # Decay: search after peak
            after = t_win >= peak_time
            t_after = t_win[after]
            r_after = r_win[after]

            efold_time = np.nan
            below_e = np.where(r_after <= peak_rate / np.e)[0]
            if len(below_e):
                efold_time = t_after[below_e[0]] - peak_time

            below_10 = np.where(r_after <= 0.10 * peak_rate)[0]
            decay_time = t_after[below_10[0]] - peak_time if len(below_10) else np.nan

            bursts.append({
                "start_time": start_time,
                "peak_time_abs": peak_time,
                "peak_rate": peak_rate,
                "peak_rate_err": peak_rate_err,
                "preburst_rate": pre_mean,
                "preburst_rate_err": pre_sem,
                "preburst_std": pre_std,
                "rise_time_s": rise_time,
                "efold_time_s": efold_time,
                "decay_time_s": decay_time,
            })

            last_burst_end_time = peak_time + (decay_time if np.isfinite(decay_time) else max_burst_search)
            next_time_idx = np.searchsorted(time, last_burst_end_time)
            ci = np.searchsorted(candidate_idx, next_time_idx)
            continue

        ci += 1

    return bursts


# --------------------------------------------------------------------------
# Confirmed-burst filtering and output
# --------------------------------------------------------------------------

def build_confirmed_bursts(bursts, mjdref, min_ratio_for_inclusion=3.0,
                            require_decay=True):
    """
    Filter candidate bursts down to confirmed ones using two checks:

    1. Amplitude ratio: peak_rate / preburst_rate must clear
       min_ratio_for_inclusion. This weeds out chance sigma-threshold
       crossings from noise — genuine bursts from these sources
       typically rise by a factor of ~10.
    2. Decay confirmation (if require_decay=True): efold_time_s must
       be finite, i.e. the rate actually declined to 1/e of peak
       within the search window. This distinguishes a real burst
       (rises AND decays) from a persistent-state step or
       instrumental/GTI artifact (rises and stays flat indefinitely).

    Returns (confirmed, skipped), where confirmed is a list of dicts
    with full-precision burst parameters (used for both JSON export
    and LaTeX table generation, so the two outputs can never
    disagree), and skipped is a list of (burst, reasons) tuples for
    candidates that failed one or both checks — nothing is silently
    dropped.

    start_time_mjd is reported to 8 decimal places.
    """
    confirmed = []
    skipped = []
    bid = 0
    for b in bursts:
        ratio = (b["peak_rate"] + b["preburst_rate"]) / b["preburst_rate"] \
            if b["preburst_rate"] else np.nan
        reasons = []
        if not np.isfinite(ratio) or ratio < min_ratio_for_inclusion:
            reasons.append(f"ratio {ratio:.1f} < {min_ratio_for_inclusion}")
        if require_decay and not np.isfinite(b["efold_time_s"]):
            reasons.append("no confirmed decay (step-like / never reached peak/e)")
        if reasons:
            skipped.append((b, reasons))
            continue

        bid += 1
        t_start_mjd = mjdref + b["start_time"] / 86400.0 if mjdref is not None else None

        confirmed.append({
            "bid": bid,
            "start_time_met": float(b["start_time"]),
            "start_time_mjd": (round(float(t_start_mjd), 8)
                                if t_start_mjd is not None else None),
            "peak_time_met": float(b["peak_time_abs"]),
            "peak_rate": float(b["peak_rate"]),
            "peak_rate_err": float(b["peak_rate_err"]),
            "preburst_rate": float(b["preburst_rate"]),
            "preburst_rate_err": float(b["preburst_rate_err"]),
            "preburst_std": float(b["preburst_std"]),
            "rise_time_s": float(b["rise_time_s"]),
            "efold_time_s": (float(b["efold_time_s"])
                              if np.isfinite(b["efold_time_s"]) else None),
            "decay_time_s": (float(b["decay_time_s"])
                              if np.isfinite(b["decay_time_s"]) else None),
            "amplitude_ratio": float(ratio),
        })

    return confirmed, skipped


def make_latex_table(bursts, mjdref, min_ratio_for_inclusion=3.0, require_decay=True):
    """
    Build a publication-style LaTeX table of confirmed burst timing
    parameters. Delegates filtering to build_confirmed_bursts so the
    table and JSON export always agree on which bursts are included.
    """
    confirmed, skipped = build_confirmed_bursts(
        bursts, mjdref, min_ratio_for_inclusion, require_decay
    )

    rows = []
    for c in confirmed:
        bid = c["bid"]
        if c["start_time_mjd"] is not None:
            t_str = f"{c['start_time_mjd']:.8f}"
        else:
            t_str = f"{c['start_time_met']:.2f} (MET, no MJDREF found)"

        peak_str = f"${c['peak_rate']:.1f} \\pm {c['peak_rate_err']:.1f}$"
        pre_str = f"${c['preburst_rate']:.1f} \\pm {c['preburst_rate_err']:.1f}$"
        rise_str = f"{c['rise_time_s']:.2f}"
        efold_str = f"{c['efold_time_s']:.2f}" if c['efold_time_s'] is not None else "L"
        decay_str = f"{c['decay_time_s']:.2f}" if c['decay_time_s'] is not None else "L"

        rows.append(f"{bid} & {t_str} & {peak_str} & {pre_str} & "
                     f"{rise_str} & {efold_str} & {decay_str} \\\\")

    table = r"""\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccccc}
\hline\hline
BID & $t_{\text{start}}$ (MJD, TDB) & Peak Rate\textsuperscript{a} (cts s$^{-1}$) & Preburst Rate\textsuperscript{b} (cts s$^{-1}$) & Rise Time (s) & $e$-folding Time (s) & Decay Time\textsuperscript{c} (s) \\
\hline
""" + "\n".join(rows) + r"""
\hline\hline
\end{tabular}
\caption{Timing Properties of Thermonuclear X-Ray Bursts Observed with AstroSat/LAXPC}
\label{tab:burst_timing}
\vspace{4pt}
{\raggedright \footnotesize \textbf{Notes.} Parameters are derived from 3.0--80.0 keV AstroSat/LAXPC light curves with a time resolution of 0.1 s; therefore, the uncertainties in the rise and decay times are 0.1 s. BID shows the observed burst number.\\
\textsuperscript{a} Preburst count rates are subtracted (net peak rate), with uncertainties incorporating instrumental and mean propagation.\\
\textsuperscript{b} Calculated as the average count rate 100 s prior to the burst start time, with uncertainties reflecting the standard error of the mean.\\
\textsuperscript{c} The time for the net count rate to reach 10\% of the peak value.\par}
\end{table}"""
    return table, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc", required=True,
                     help="source broadband light curve file (FITS or ASCII)")
    ap.add_argument("--bkg", default="",
                     help="background light curve file, or omit to disable background subtraction")
    ap.add_argument("--rebin", type=float, default=0.5,
                     help="rebin to this resolution in seconds before burst search (default 0.5)")
    ap.add_argument("--search-sigma", type=float, default=4.0,
                     help="sigma threshold above preburst rate to flag a burst start (default 4)")
    ap.add_argument("--preburst-window", type=float, default=100.0,
                     help="seconds before burst start used to compute preburst rate (default 100)")
    ap.add_argument("--min-sep", type=float, default=20.0,
                     help="minimum seconds between distinct bursts")
    ap.add_argument("--max-search", type=float, default=60.0,
                     help="max seconds after start to search for peak/decay")
    ap.add_argument("--mjdref", type=float, default=None,
                     help="MJD of MET=0 (MJDREFI+MJDREFF), if not auto-detectable from the FITS "
                          "header (needed for ASCII .lc files — get it from the parent event file's header)")
    ap.add_argument("--latex", action="store_true",
                     help="also print a LaTeX table of confirmed bursts to stdout")
    ap.add_argument("--min-amp-ratio", type=float, default=3.0,
                     help="minimum peak/preburst rate ratio for a candidate to count as confirmed (default 3.0)")
    ap.add_argument("--no-require-decay", action="store_true",
                     help="don't require a confirmed e-folding decay for a candidate to count as confirmed")
    ap.add_argument("--json", default="",
                     help="path to save confirmed-burst parameters as JSON")
    args = ap.parse_args()

    time, rate, err = load_lightcurve(args.lc)
    print(f"Loaded {args.lc}: {len(time)} points, "
          f"span {time.max()-time.min():.1f} s, median dt {np.median(np.diff(time)):.3f} s")

    mjdref, timesys = get_mjd_ref(args.lc)
    if args.mjdref is not None:
        mjdref, timesys = args.mjdref, "user-supplied"
    if mjdref is not None:
        print(f"MJD reference: {mjdref} (TIMESYS={timesys})")
    else:
        print("No MJDREFI/MJDREFF found in file header — MET-to-MJD conversion "
              "unavailable unless you pass --mjdref (check the parent event file's header).")

    if args.bkg:
        bt, br, _ = load_lightcurve(args.bkg)
        rate = background_subtract(time, rate, bt, br)
        print(f"Background-subtracted using {args.bkg}")
    else:
        print("No background subtraction (--bkg not given).")

    t_bin, r_bin, e_bin = rebin_lightcurve(time, rate, err, args.rebin)
    print(f"Rebinned to {args.rebin} s: {len(t_bin)} points")

    bursts = find_bursts(t_bin, r_bin, err=e_bin,
                          search_sigma=args.search_sigma,
                          preburst_window=args.preburst_window,
                          min_separation=args.min_sep,
                          max_burst_search=args.max_search)

    if not bursts:
        print("\nNo bursts found. Try lowering --search-sigma or check the input band/file.")
        return

    # ---- Full-precision candidate table (unfiltered) ----
    print(f"\nFound {len(bursts)} candidate burst(s):\n")
    header = (f"{'BID':>3} {'Start(s)':>16} {'PeakRate':>12} {'PeakErr':>10} "
              f"{'PreRate':>12} {'PreErr':>10} {'Rise(s)':>9} {'eFold(s)':>10} "
              f"{'Decay(s)':>10} {'Ratio':>9}")
    print(header)
    print("-" * len(header))
    for bid, b in enumerate(bursts, start=1):
        ratio = (b["peak_rate"] + b["preburst_rate"]) / b["preburst_rate"]
        efold_disp = b['efold_time_s'] if np.isfinite(b['efold_time_s']) else float('nan')
        decay_disp = b['decay_time_s'] if np.isfinite(b['decay_time_s']) else float('nan')
        print(f"{bid:>3} {b['start_time']:>16.4f} {b['peak_rate']:>12.4f} "
              f"{b['peak_rate_err']:>10.4f} {b['preburst_rate']:>12.4f} "
              f"{b['preburst_rate_err']:>10.4f} {b['rise_time_s']:>9.4f} "
              f"{efold_disp:>10.4f} {decay_disp:>10.4f} {ratio:>9.4f}")

    # ---- Confirmed bursts (single source of truth for LaTeX + JSON) ----
    confirmed, skipped = build_confirmed_bursts(
        bursts, mjdref, args.min_amp_ratio, require_decay=not args.no_require_decay
    )

    print(f"\nConfirmed burst(s) (ratio >= {args.min_amp_ratio}"
          f"{', decay confirmed' if not args.no_require_decay else ''}): "
          f"{len(confirmed)} of {len(bursts)}\n")
    chdr = (f"{'BID':>3} {'MJD (TDB)':>16} {'Start(MET,s)':>16} {'PeakRate':>12} "
            f"{'PeakErr':>10} {'PreRate':>12} {'PreErr':>10} {'Rise(s)':>9} "
            f"{'eFold(s)':>10} {'Decay(s)':>10} {'Ratio':>9}")
    print(chdr)
    print("-" * len(chdr))
    for c in confirmed:
        mjd_disp = f"{c['start_time_mjd']:.8f}" if c["start_time_mjd"] is not None else "n/a"
        efold_disp = f"{c['efold_time_s']:.4f}" if c["efold_time_s"] is not None else "nan"
        decay_disp = f"{c['decay_time_s']:.4f}" if c["decay_time_s"] is not None else "nan"
        print(f"{c['bid']:>3} {mjd_disp:>16} {c['start_time_met']:>16.4f} "
              f"{c['peak_rate']:>12.4f} {c['peak_rate_err']:>10.4f} "
              f"{c['preburst_rate']:>12.4f} {c['preburst_rate_err']:>10.4f} "
              f"{c['rise_time_s']:>9.4f} {efold_disp:>10} {decay_disp:>10} "
              f"{c['amplitude_ratio']:>9.4f}")

    if skipped:
        print(f"\n% Note: {len(skipped)} candidate(s) excluded from the confirmed set:")
        for s, reasons in skipped:
            print(f"%   t={s['start_time']:.4f}s : " + "; ".join(reasons))

    # ---- JSON export ----
    if args.json:
        payload = {
            "source_lc": args.lc,
            "background_lc": args.bkg if args.bkg else None,
            "mjdref": mjdref,
            "timesys": timesys,
            "rebin_s": args.rebin,
            "search_sigma": args.search_sigma,
            "preburst_window_s": args.preburst_window,
            "min_amp_ratio": args.min_amp_ratio,
            "require_decay": not args.no_require_decay,
            "n_candidates_total": len(bursts),
            "n_confirmed": len(confirmed),
            "confirmed_bursts": confirmed,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved confirmed-burst parameters -> {args.json}")

    if args.latex:
        table, _ = make_latex_table(bursts, mjdref, args.min_amp_ratio,
                                     require_decay=not args.no_require_decay)
        print(f"\n--- LaTeX table (ratio >= {args.min_amp_ratio}"
              f"{', decay confirmed' if not args.no_require_decay else ''}) ---\n")
        print(table)


if __name__ == "__main__":
    main()
