"""
energy_resolved_plot_lc.py
------------------------------
Publication-style burst light curve plots, in the exact visual style
and marker set as plot_lc.py (Bostanci et al. 2023 convention), but
for each of the five energy-resolved LAXPC bands (E1-E5) separately,
using the per-band parameters in energy_resolved_burst_starts.json
(produced by energy_resolved_burst_start_analysis.py) instead of the
single broadband (3-80 keV) burst_analysis_confirmed.json.

Markers (identical meaning/style to plot_lc.py, now per band):
  Persistent Rate   : black solid thin horizontal line
                       (THIS band's own preburst_rate)
  Threshold         : green thin horizontal line
                       (THIS band's own preburst_rate + search_sigma*preburst_std
                        -- the actual 4-sigma start-time criterion used by
                        energy_resolved_burst_start_analysis.py for THIS band,
                        not a fixed 1.5x or 10% line)
  Start Time        : black, LONG dash vertical line (this band's start_time_met)
  Peak Time         : red thin solid vertical line   (this band's peak_time_met)
  e-folding Time    : black, SHORT dash vertical line
  Decay Time (10%)  : black, fine dotted vertical line

Not-detected bands
-------------------
If a band was never significantly detected for a given burst
(band_result["detected"] is False -- typically the faintest 30-40 keV
band on a weak burst; see the console output of
energy_resolved_burst_start_analysis.py), the light curve and this
band's Persistent Rate / Threshold lines are still drawn -- so you can
see there really was nothing there -- but the Start/Peak/e-fold/Decay
vlines are omitted (none of those exist for that band), the panel is
centered on the broadband start instead of a (nonexistent) band peak,
and the panel title is suffixed "(not detected)".

IMPORTANT re: units -- exactly as in plot_lc.py, peak_rate stored in
the JSON is NET (preburst-subtracted). This script does not need to
add anything back on for the vlines (they're time markers, not rate
markers), but the horizontal Persistent Rate / Threshold lines are
already in raw-rate units and will line up directly with the plotted
raw light curve.

Binning matches what energy_resolved_burst_start_analysis.py actually
used for each band (native ~0.1 s for E1-E3, rebinned to
rebin_s_for_rebin_bands for the bands listed in rebin_bands, i.e.
E4/E5) -- read directly from the JSON's own metadata, not re-guessed,
so the plotted curve is exactly what was analyzed.

Run from the same directory as burst_analysis.py (or add it to
PYTHONPATH), with JSON_PATH pointing at your
energy_resolved_burst_starts.json.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# burst_analysis.py lives in the sibling "parameters check" folder
# (pipelines/parameters check/), not here (pipelines/energy resolved/).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BURST_ANALYSIS_DIR = os.path.join(_SCRIPT_DIR, "..", "parameters check")
sys.path.insert(0, _BURST_ANALYSIS_DIR)

from burst_analysis import load_lightcurve, rebin_lightcurve

# ---------- Style matching reference script ----------
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

BAND_ORDER = ["3.0-9.0", "9.0-15.0", "15.0-21.0", "21.0-30.0", "30.0-40.0"]

# ---------- Paths ----------
JSON_PATH = f"{_SCRIPT_DIR}/energy_resolved_burst_starts.json"

OUT_DIR = ("/home/ruban/AstroSat/observations/4U1728-34/"
           "T01_041T01_9000000362/results/energy_resolved_bostanci")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load JSON (self-describing: has its own band_files/rebin info) ----------
with open(JSON_PATH) as f:
    payload = json.load(f)

burst_entries = payload["bursts"]
band_files = payload["band_files"]
rebin_bands = set(payload["rebin_bands"])
rebin_s = payload["rebin_s_for_rebin_bands"]
search_sigma = payload["search_sigma"]

for i, entry in enumerate(burst_entries, start=1):
    entry["label"] = f"Burst {i}"
    entry["name"] = f"B{i}"

# ---------- Load + bin each band lightcurve, exactly as the start analysis did ----------
band_data = {}
for band, path in band_files.items():
    t, r, e = load_lightcurve(path)
    if band in rebin_bands:
        t, r, e = rebin_lightcurve(t, r, e, rebin_s)
        print(f"Loaded {path}: rebinned to {rebin_s} s, band {band} keV")
    else:
        print(f"Loaded {path}: native ~0.1 s binning, band {band} keV")
    band_data[band] = (t, r, e)


def band_label(band):
    """'3.0-9.0' -> '3.0 - 9.0 keV' for titles/filenames."""
    return band.replace("-", " - ") + " keV"


def draw_markers(ax, res, t_center, lw_scale=1.0):
    """
    Draw the Persistent Rate / Threshold / Start / Peak / e-fold / Decay
    markers for one band's result dict `res`, in a time frame already
    centered on `t_center` (0.0 in the plotted x-axis == t_center).

    If res["detected"] is False, only the two horizontal rate lines are
    drawn -- there is no Start/Peak/e-fold/Decay to mark.
    """
    preburst_rate = res["preburst_rate"]
    preburst_std = res["preburst_std"]
    threshold_rate = (preburst_rate + search_sigma * preburst_std
                       if preburst_rate is not None and preburst_std is not None
                       else None)

    if preburst_rate is not None:
        ax.axhline(preburst_rate, color="black", ls="-", lw=0.7 * lw_scale,
                   label="Persistent Rate")
    if threshold_rate is not None:
        ax.axhline(threshold_rate, color="green", ls="-", lw=0.7 * lw_scale,
                   label="Threshold")

    if not res["detected"]:
        return

    t_onset = res["start_time_met"] - t_center
    t_peak = res["peak_time_met"] - t_center
    t_efold = res["efold_time_s"]   # already a duration after peak
    t_decay = res["decay_time_s"]   # already a duration after peak

    ax.axvline(t_onset, color="black", lw=0.9 * lw_scale,
               dashes=(8, 3), label="Start Time")             # long dash
    ax.axvline(t_peak, color="red", ls="-", lw=0.7 * lw_scale,
               label="Peak Time")                              # thin solid
    if t_efold is not None:
        ax.axvline(t_peak + t_efold, color="black", lw=0.9 * lw_scale,
                   dashes=(3, 2), label="e-folding Time")     # short dash
    if t_decay is not None:
        ax.axvline(t_peak + t_decay, color="black", ls=":", lw=0.9 * lw_scale,
                   label="Decay Time")                          # fine dotted


def get_window(res, broadband_start):
    """Absolute-MET (t_lo, t_hi) plot window, and the center time (0.0
    on the plotted x-axis) to use for this band's panel."""
    if res["detected"]:
        t_peak = res["peak_time_met"]
        t_onset = res["start_time_met"]
        decay = res.get("decay_time_s")
        tail = decay if decay is not None else 20.0
        return t_onset - 6, t_peak + tail + 8, t_peak
    else:
        # No detection in this band -- fall back to a fixed window
        # around the broadband start, just for context.
        return broadband_start - 6, broadband_start + 40, broadband_start


def plot_one_panel(ax, band, res, broadband_start, ms=2.0, elw=0.5, lw_scale=1.0):
    """Draw one band's light curve + markers into `ax`. Returns whether
    this band was detected (so callers can prefer a 'full' panel for
    the shared legend)."""
    t, r, e = band_data[band]
    t_lo, t_hi, t_center = get_window(res, broadband_start)
    mask = (t >= t_lo) & (t <= t_hi)
    t_seg = t[mask] - t_center
    r_seg = r[mask]
    e_seg = e[mask]

    ax.errorbar(
        t_seg, r_seg, yerr=e_seg,
        fmt=".", color=POINT_COLOR, ecolor=POINT_COLOR,
        markersize=ms, elinewidth=elw, capsize=0,
    )
    draw_markers(ax, res, t_center, lw_scale=lw_scale)
    title = band_label(band)
    if not res["detected"]:
        title += " (not detected)"
    ax.set_title(title, fontsize=10)
    return res["detected"]


# ================= Individual (burst x band) plots =================
for entry in burst_entries:
    bid_name = entry["name"]
    broadband_start = entry["broadband_start_time_met"]

    for band in BAND_ORDER:
        res = entry["bands"][band]

        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        plot_one_panel(ax, band, res, broadband_start, ms=2.0, elw=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count/s")
        ax.set_title(f"{entry['label']}: {band_label(band)}"
                     + ("" if res["detected"] else " (not detected)"))
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()

        band_tag = band.replace(".", "p").replace("-", "_")
        outpng = f"{OUT_DIR}/{bid_name}_{band_tag}keV_bostanci_markers.png"
        outpdf = f"{OUT_DIR}/{bid_name}_{band_tag}keV_bostanci_markers.pdf"
        fig.savefig(outpng, dpi=300)
        fig.savefig(outpdf)
        plt.close(fig)
        print(f"Saved {outpng} and {outpdf}")

# ================= Per-burst combined panel (5 bands in a grid) =================
ncols = 2
nrows = int(np.ceil(len(BAND_ORDER) / ncols))

for entry in burst_entries:
    bid_name = entry["name"]
    broadband_start = entry["broadband_start_time_met"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 4.25 * nrows),
                              sharex=False, sharey=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    detected_ax = None
    for ax, band in zip(axes.flat, BAND_ORDER):
        res = entry["bands"][band]
        was_detected = plot_one_panel(ax, band, res, broadband_start,
                                       ms=1.5, elw=0.4, lw_scale=1.0)
        if was_detected and detected_ax is None:
            detected_ax = ax

    for ax in axes.flat[len(BAND_ORDER):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Count/s")

    # Prefer the legend from a detected panel (has all 6 marker types);
    # fall back to whatever the first panel has if none were detected.
    legend_src = detected_ax if detected_ax is not None else axes.flat[0]
    handles, labels = legend_src.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(entry["label"])
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    outpng = f"{OUT_DIR}/{bid_name}_all_bands_bostanci_markers.png"
    outpdf = f"{OUT_DIR}/{bid_name}_all_bands_bostanci_markers.pdf"
    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpng} and {outpdf}")
