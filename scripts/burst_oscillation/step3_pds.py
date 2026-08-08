#!/usr/bin/env python3
"""
step3_pds.py
============
Plot the Leahy-normalised power density spectrum (PDS) for the best
refined-search candidate segment.

Recomputes the PDS for the segment identified as the best candidate in
step2_refined_search.py, then produces a log-log plot showing:

- The full Leahy power spectrum over the 10–1000 Hz search band.
- A horizontal dotted line at the Poisson noise level (Leahy = 2).
- A dashed horizontal threshold line at the chosen sigma level
  (default 3 sigma), computed using the same number of trials as
  the refined search for direct comparison.
- A text annotation at the detected peak giving its frequency,
  Leahy power, and trial-corrected significance.

Input
-----
The same barycentric-corrected 3–30 keV LAXPC event FITS file used in
the coarse and refined searches. The segment boundaries (BEST_T_START,
BEST_T_END) and trial count (N_TRIALS) are copied from the refined
search output.

Time convention
---------------
BEST_T_START and BEST_T_END are entered as seconds since burst start,
matching the refined search output. Internal photon selection converts
to absolute barycentric MET.

Output
------
- <OUTPUT_DIR>/pds_plot.png

Usage
-----
Fill in the CONFIGURATION block from the refined search output, then:

    python step3_pds.py

Dependencies: numpy, scipy, matplotlib, astropy, stingray
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfinv, erf
from astropy.io import fits
from stingray import Lightcurve, Powerspectrum

# =============================================================================
# CONFIGURATION — fill in from the refined search output
# =============================================================================

EVENT_FILE = "/path/to/bary_level2_3_30keV.event.fits"
OUTPUT_DIR = "/path/to/results/BO_Burst_1"

BURST_START = 195101672.035   # Burst start, barycentric MET (seconds)

# Best candidate segment from step2 output (seconds since burst start)
BEST_T_START = 1.200
BEST_T_END   = 3.200

# Number of trials from the refined search (for threshold consistency)
N_TRIALS = 72

SIGMA_LEVEL      = 3.0      # Significance threshold line to draw (sigma)
FREQ_LO          = 10.0     # Search band lower bound (Hz)
FREQ_HI          = 1000.0   # Search band upper bound (Hz)
DT               = 1.0 / 4096.0
NORM             = "leahy"
LEAHY_NOISE_LEVEL = 2.0

PLOT_PATH = os.path.join(OUTPUT_DIR, "pds_plot.png")

# =============================================================================


def read_times(event_file):
    """
    Read photon arrival times from a barycentric LAXPC event FITS file.

    Parameters
    ----------
    event_file : str
        Path to the barycentric event FITS file.

    Returns
    -------
    times : numpy.ndarray of float64
        Sorted photon arrival times in barycentric MET seconds.

    Raises
    ------
    RuntimeError
        If no TIME column is found in any extension of the file.
    """
    preferred = {"EVENTS", "EVENT", "EVENTS1"}
    with fits.open(event_file, memmap=True) as hdul:
        ext = None
        for hdu in hdul:
            name = (getattr(hdu, "name", "") or "").upper()
            if name in preferred:
                cols = [c.upper() for c in (hdu.columns.names or [])]
                if "TIME" in cols:
                    ext = hdu
                    break
        if ext is None:
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    cols = [c.upper() for c in (hdu.columns.names or [])]
                    if "TIME" in cols:
                        ext = hdu
                        break
        if ext is None:
            raise RuntimeError(f"No TIME column found in {event_file}")
        times = np.asarray(ext.data["TIME"], dtype=np.float64)
    if not np.all(np.diff(times) >= 0):
        times = np.sort(times)
    return times


def sigma_to_power_threshold(sigma, n_trials):
    """
    Convert a detection significance to a Leahy power threshold.

    Inverts the paper's detection significance formula:
        x = 1 − erf(sigma / sqrt(2))
        P_threshold = −2 × ln(x / n_trials)

    Parameters
    ----------
    sigma : float
        Desired significance level (e.g. 3.0 for 3 sigma).
    n_trials : int
        Number of independent trials in the search.

    Returns
    -------
    float
        Leahy power threshold corresponding to the given sigma level.
    """
    x = 1.0 - erf(sigma / np.sqrt(2.0))
    return -2.0 * np.log(x / n_trials)


def power_to_sigma(p_max, n_trials):
    """
    Convert a peak Leahy power to a trial-corrected detection significance.

    Uses the paper's relations:
        x = exp(−Pmax / 2) × n_trials
        sigma = sqrt(2) × erfinv(1 − x)

    Parameters
    ----------
    p_max : float
        Peak Leahy power.
    n_trials : int
        Number of independent trials in the search.

    Returns
    -------
    float
        Detection significance in sigma.
    """
    x = np.exp(-p_max / 2.0) * n_trials
    x_clipped = min(x, 1.0 - 1e-15)
    return np.sqrt(2.0) * erfinv(1.0 - x_clipped)


def main():
    if not os.path.isfile(EVENT_FILE):
        raise RuntimeError(f"Event file not found: {EVENT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PDS PLOT — best refined-search candidate segment")
    print("=" * 70)
    print(f"Burst start       : {BURST_START:.6f} s (absolute MET; t = 0 below)")
    print(f"Segment interval  : [{BEST_T_START:.3f}, {BEST_T_END:.3f}] s "
          f"(since burst start, span = {BEST_T_END - BEST_T_START:.3f} s)")
    print(f"N trials          : {N_TRIALS}")

    t_start_abs = BURST_START + BEST_T_START
    t_end_abs   = BURST_START + BEST_T_END
    seg_len     = BEST_T_END - BEST_T_START

    times = read_times(EVENT_FILE)
    mask  = (times >= t_start_abs) & (times < t_end_abs)
    seg_times = times[mask]
    print(f"Photons in segment: {seg_times.size}")

    if seg_times.size < 2:
        raise RuntimeError("Fewer than 2 photons in the specified segment.")

    lc = Lightcurve.make_lightcurve(
        seg_times, dt=DT, tstart=t_start_abs, tseg=seg_len
    )
    ps = Powerspectrum(lc, norm=NORM)

    fmask  = (ps.freq >= FREQ_LO) & (ps.freq <= FREQ_HI)
    freqs  = ps.freq[fmask]
    powers = ps.power[fmask]

    peak_idx  = int(np.argmax(powers))
    peak_freq = float(freqs[peak_idx])
    p_max     = float(powers[peak_idx])
    sigma_det = power_to_sigma(p_max, N_TRIALS)
    p_thr     = sigma_to_power_threshold(SIGMA_LEVEL, N_TRIALS)

    print(f"Peak frequency    : {peak_freq:.3f} Hz")
    print(f"Peak Leahy power  : {p_max:.4f}")
    print(f"Detection sigma   : {sigma_det:.2f}  (n = {N_TRIALS})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.plot(freqs, powers, color="#2b6cb0", lw=1.0, alpha=0.9)

    ax.axhline(LEAHY_NOISE_LEVEL, color="gray", ls=":", lw=1.2)
    ax.text(FREQ_HI * 0.97, LEAHY_NOISE_LEVEL * 1.08, "Poisson noise",
            ha="right", va="bottom", fontsize=9, color="gray")

    ax.axhline(p_thr, color="#c53030", ls="--", lw=1.3)
    ax.text(FREQ_LO * 1.3, p_thr * 1.08,
            f"{SIGMA_LEVEL:.0f}$\\sigma$ ($P$ = {p_thr:.1f})",
            ha="left", va="bottom", fontsize=9, color="#c53030")

    ax.text(peak_freq * 0.85, p_max * 1.05,
            f"{peak_freq:.1f} Hz,  $P$ = {p_max:.1f},  {sigma_det:.2f}$\\sigma$",
            ha="right", va="bottom", fontsize=10, color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylim(bottom=max(0.5, powers.min() * 0.7), top=p_max * 1.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Leahy Power", fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {PLOT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
