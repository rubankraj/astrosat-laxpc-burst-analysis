"""
plot_burst_lightcurves.py
--------------------------
Publication-style burst light curve plots for confirmed bursts produced
by burst_timing_parameters.py (i.e. its --json output), following the
Bostanci et al. 2023 visual convention:

    Persistent Rate    black, solid, thin horizontal line
    Threshold          green, thin horizontal line
                       (preburst_rate + search_sigma * preburst_std --
                        the actual sigma-based start-time criterion
                        used by burst_timing_parameters.py, not a fixed
                        1.5x or 10% line)
    Start Time         black, long-dash vertical line
    Peak Time          red, thin solid vertical line
    e-folding Time     black, short-dash vertical line
    Decay Time (10%)   black, fine-dotted vertical line

Produces one plot per confirmed burst, plus a combined panel figure
showing all bursts together.

Units note
----------
In the JSON output, peak_rate is the NET (preburst-subtracted) value —
see the LaTeX table footnote "Preburst count rates are subtracted."
This script adds preburst_rate back on wherever it plots against the
raw light curve, so the peak marker lines up visually with the actual
data points.

Data consistency
-----------------
The light curve is loaded using burst_timing_parameters.py's own
load_lightcurve() / background_subtract() functions (imported
directly, not reimplemented), and the source/background file paths
are read from the JSON's own "source_lc" / "background_lc" fields.
This guarantees the plotted curve is exactly what was analyzed --
same file, same background subtraction, no risk of the plot and the
analysis silently drifting apart.

Requires burst_timing_parameters.py to be in the same directory
(scripts/) so it can be imported directly.

Input
-----
--json : path to the confirmed-burst JSON produced by
         burst_timing_parameters.py --json, e.g.
         results/burst_analysis_confirmed.json
         (the JSON itself points back to the original light curve
         file(s), so nothing else needs to be passed in)

Output
------
--outdir : directory to save plots into (default: plots/). For each
           confirmed burst this writes <outdir>/burst<N>.png and
           <outdir>/burst<N>.pdf, plus a combined panel figure at
           <outdir>/all_bursts.png and <outdir>/all_bursts.pdf.

Usage
-----
    python3 plot_burst_lightcurves.py \\
        --json results/burst_analysis_confirmed.json \\
        --outdir plots/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from burst_timing_parameters import load_lightcurve, background_subtract

# ---------- Plot style ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "figure.dpi": 300,
})

POINT_COLOR = "0.2"


def draw_markers(ax, t_onset, t_efold, t_decay, preburst_rate,
                  threshold_rate, lw_scale=1.0):
    """Draw the standard burst-parameter marker set onto an axis."""
    ax.axhline(preburst_rate, color="black", ls="-", lw=0.7 * lw_scale,
               label="Persistent Rate")
    ax.axhline(threshold_rate, color="green", ls="-", lw=0.7 * lw_scale,
               label="Threshold")

    ax.axvline(t_onset, color="black", lw=0.9 * lw_scale,
               dashes=(8, 3), label="Start Time")              # long dash
    ax.axvline(0.0, color="red", ls="-", lw=0.7 * lw_scale,
               label="Peak Time")                               # thin solid
    if t_efold is not None:
        ax.axvline(t_efold, color="black", lw=0.9 * lw_scale,
                   dashes=(3, 2), label="e-folding Time")      # short dash
    if t_decay is not None:
        ax.axvline(t_decay, color="black", ls=":", lw=0.9 * lw_scale,
                   label="Decay Time")                           # fine dotted


def get_window(burst, pad_before=6.0, pad_after=8.0, default_tail=20.0):
    """Time window (relative to observation start) to plot around a burst."""
    t_peak = burst["peak_time_met"]
    t_onset = burst["start_time_met"]
    decay = burst.get("decay_time_s")
    tail = decay if decay is not None else default_tail
    return t_onset - pad_before, t_peak + tail + pad_after


def plot_single_burst(ax, burst, t_abs, rate, err, search_sigma):
    """Plot one burst's light curve segment with markers onto an axis."""
    t_peak = burst["peak_time_met"]
    t_onset = burst["start_time_met"]
    preburst_rate = burst["preburst_rate"]
    preburst_std = burst["preburst_std"]
    threshold_rate = preburst_rate + search_sigma * preburst_std
    efold = burst.get("efold_time_s")
    decay = burst.get("decay_time_s")

    t_lo, t_hi = get_window(burst)
    mask = (t_abs >= t_lo) & (t_abs <= t_hi)
    t_seg = t_abs[mask] - t_peak
    r_seg = rate[mask]
    e_seg = err[mask]

    ax.errorbar(
        t_seg, r_seg, yerr=e_seg,
        fmt=".", color=POINT_COLOR, ecolor=POINT_COLOR,
        markersize=1.5, elinewidth=0.4, capsize=0,
    )
    draw_markers(ax, t_onset - t_peak, efold, decay, preburst_rate, threshold_rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True,
                     help="path to the confirmed-burst JSON produced by "
                          "burst_timing_parameters.py --json")
    ap.add_argument("--outdir", default="plots",
                     help="directory to save output plots into (default: plots/)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.json) as f:
        payload = json.load(f)

    data = payload["confirmed_bursts"]
    search_sigma = payload["search_sigma"]
    source_lc_path = payload["source_lc"]
    bkg_lc_path = payload.get("background_lc")

    if not data:
        print("No confirmed bursts found in the JSON file — nothing to plot.")
        return

    for i, b in enumerate(data, start=1):
        b["name"] = f"burst{i}"
        b["label"] = f"Burst {i}"

    # Load the exact light curve that was analyzed (same source + background)
    t_abs, rate, err = load_lightcurve(source_lc_path)
    if bkg_lc_path:
        bt, br, _ = load_lightcurve(bkg_lc_path)
        rate = background_subtract(t_abs, rate, bt, br)
        print(f"Background-subtracted using {bkg_lc_path}")
    else:
        print("No background subtraction (JSON has no background_lc).")

    # ---------------- Individual burst plots ----------------
    for b in data:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        plot_single_burst(ax, b, t_abs, rate, err, search_sigma)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count/s")
        ax.set_title(b["label"])
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()

        outpng = os.path.join(args.outdir, f"{b['name']}.png")
        outpdf = os.path.join(args.outdir, f"{b['name']}.pdf")
        fig.savefig(outpng, dpi=300)
        fig.savefig(outpdf)
        plt.close(fig)
        print(f"Saved {outpng} and {outpdf}")

    # ---------------- Combined panel figure ----------------
    n = len(data)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 4.25 * nrows),
                              sharex=False, sharey=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, b in zip(axes.flat, data):
        plot_single_burst(ax, b, t_abs, rate, err, search_sigma)
        ax.set_title(b["label"])

    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count/s")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    outpng = os.path.join(args.outdir, "all_bursts.png")
    outpdf = os.path.join(args.outdir, "all_bursts.pdf")
    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpng} and {outpdf}")


if __name__ == "__main__":
    main()
