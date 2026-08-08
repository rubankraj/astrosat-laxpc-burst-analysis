#!/usr/bin/env python3
"""
step1_coarse_search.py
======================
Coarse burst oscillation search on AstroSat/LAXPC event data.

Replicates the sliding-window FFT method described in the paper
(Section 3.6, "Search for Burst Oscillations", coarse pass):

    A 2 s window is slid across the burst interval in steps of 0.5 s.
    For each window, a Leahy-normalised power spectrum is computed using
    Stingray, and the peak power in the 10–1000 Hz band is recorded.
    The window with the highest peak power is reported as the coarse
    candidate for the refined search.

Input
-----
A barycentric-corrected LAXPC event FITS file pre-filtered to the
3–30 keV energy band (e.g. bary_level2_3_30keV.event.fits).
Energy filtering is NOT repeated here; the input file is assumed to
already contain only photons in the desired band.

Time convention
---------------
Photon selection uses absolute barycentric mission elapsed time (MET),
as stored in the FITS TIME column. All printed and saved output
(window intervals, best candidate) is expressed in seconds since burst
start (t = 0 at BURST_START) for readability.

Output
------
- Console: per-window progress and final best-candidate summary.
- <OUTPUT_DIR>/coarse_search_results.txt: full window table and
  best-candidate block.

Usage
-----
Edit the CONFIGURATION block below to match your observation and burst,
then run:

    python step1_coarse_search.py

Dependencies: numpy, astropy, stingray
"""

import os
import numpy as np
from astropy.io import fits
from stingray import Lightcurve, Powerspectrum

# =============================================================================
# CONFIGURATION — edit these to match your observation and burst parameters
# =============================================================================

EVENT_FILE = "/path/to/bary_level2_3_30keV.event.fits"
OUTPUT_DIR = "/path/to/results/BO_Burst_1"

BURST_START    = 195101672.035   # Burst start, barycentric MET (seconds)
BURST_DURATION = 16.106          # Burst duration (seconds)
BURST_END      = BURST_START + BURST_DURATION

WINDOW = 2.0      # Sliding window length (seconds)
STEP   = 0.5      # Slide step size (seconds)
FREQ_LO = 10.0    # Search band lower bound (Hz)
FREQ_HI = 1000.0  # Search band upper bound (Hz)

# Light-curve bin size. Nyquist frequency must exceed FREQ_HI.
# 1/4096 s gives Nyquist = 2048 Hz, safely above 1000 Hz.
DT   = 1.0 / 4096.0
NORM = "leahy"    # Leahy normalisation (expected Poisson noise level = 2)

# =============================================================================


def read_times(event_file):
    """
    Read photon arrival times from a barycentric LAXPC event FITS file.

    Searches for a binary table extension named EVENTS, EVENT, or EVENTS1
    containing a TIME column. Falls back to the first binary table with a
    TIME column if none of the preferred names are found.

    Parameters
    ----------
    event_file : str
        Path to the barycentric event FITS file.

    Returns
    -------
    times : numpy.ndarray of float64
        Sorted array of photon arrival times in barycentric MET seconds.

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


def main():
    if not os.path.isfile(EVENT_FILE):
        raise RuntimeError(f"Event file not found: {EVENT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("COARSE BURST OSCILLATION SEARCH")
    print("=" * 70)
    print(f"Event file     : {EVENT_FILE}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"Burst start    : {BURST_START:.6f} s (absolute MET; printed as t = 0 below)")
    print(f"Burst end      : {BURST_END:.6f} s  (duration = {BURST_DURATION:.3f} s)")
    print(f"Window / step  : {WINDOW} s / {STEP} s")
    print(f"Search band    : {FREQ_LO}–{FREQ_HI} Hz")
    print(f"Normalisation  : {NORM}")
    print(f"LC bin size    : {DT:.8f} s  (Nyquist = {0.5 / DT:.1f} Hz)")
    print("=" * 70)

    times = read_times(EVENT_FILE)
    print(f"Total photons in file         : {times.size:,}")

    burst_mask  = (times >= BURST_START) & (times < BURST_END)
    burst_times = times[burst_mask]
    print(f"Photons within burst interval : {burst_times.size:,}")

    if burst_times.size < 2:
        raise RuntimeError("Fewer than 2 photons in the burst interval.")

    n_windows = int(np.floor((BURST_DURATION - WINDOW) / STEP)) + 1
    if n_windows < 1:
        raise RuntimeError(
            f"Burst duration ({BURST_DURATION:.3f} s) is shorter than the "
            f"window ({WINDOW} s)."
        )
    print(f"Number of sliding windows     : {n_windows}\n")

    results = []

    for widx in range(n_windows):
        w_start_abs = BURST_START + widx * STEP
        w_end_abs   = w_start_abs + WINDOW
        w_start_rel = widx * STEP
        w_end_rel   = w_start_rel + WINDOW

        wmask     = (burst_times >= w_start_abs) & (burst_times < w_end_abs)
        win_times = burst_times[wmask]
        n_gamma   = win_times.size

        if n_gamma < 2:
            print(f"  Window {widx:3d} [{w_start_rel:6.3f}, {w_end_rel:6.3f}] s : "
                  f"only {n_gamma} photon(s) — skipped")
            results.append({
                "window": widx, "t_start": w_start_rel, "t_end": w_end_rel,
                "n_photons": n_gamma, "peak_freq": np.nan, "peak_power": np.nan,
                "status": "skipped_low_counts",
            })
            continue

        try:
            lc = Lightcurve.make_lightcurve(
                win_times, dt=DT, tstart=w_start_abs, tseg=WINDOW
            )
            ps = Powerspectrum(lc, norm=NORM)
        except Exception as exc:
            print(f"  Window {widx:3d} [{w_start_rel:6.3f}, {w_end_rel:6.3f}] s : "
                  f"FAILED ({exc})")
            results.append({
                "window": widx, "t_start": w_start_rel, "t_end": w_end_rel,
                "n_photons": n_gamma, "peak_freq": np.nan, "peak_power": np.nan,
                "status": f"error: {exc}",
            })
            continue

        fmask = (ps.freq >= FREQ_LO) & (ps.freq <= FREQ_HI)
        if not np.any(fmask):
            results.append({
                "window": widx, "t_start": w_start_rel, "t_end": w_end_rel,
                "n_photons": n_gamma, "peak_freq": np.nan, "peak_power": np.nan,
                "status": "no_freq_bins",
            })
            continue

        peak_idx   = int(np.argmax(ps.power[fmask]))
        peak_freq  = float(ps.freq[fmask][peak_idx])
        peak_power = float(ps.power[fmask][peak_idx])

        print(f"  Window {widx:3d} [{w_start_rel:6.3f}, {w_end_rel:6.3f}] s  "
              f"N = {n_gamma:5d}  peak = {peak_power:6.3f} at {peak_freq:8.2f} Hz")

        results.append({
            "window": widx, "t_start": w_start_rel, "t_end": w_end_rel,
            "n_photons": n_gamma, "peak_freq": peak_freq, "peak_power": peak_power,
            "status": "ok",
        })

    # Best candidate
    valid = [r for r in results if r["status"] == "ok"]
    print("\n" + "=" * 70)
    best = None
    if not valid:
        print("No valid windows produced a power spectrum.")
    else:
        best = max(valid, key=lambda r: r["peak_power"])
        print("STRONGEST COARSE CANDIDATE")
        print(f"  Window index       : {best['window']}")
        print(f"  Window interval    : [{best['t_start']:.3f}, {best['t_end']:.3f}] s "
              f"(since burst start)")
        print(f"  Peak frequency     : {best['peak_freq']:.3f} Hz")
        print(f"  Peak Leahy power   : {best['peak_power']:.4f}")
        print(f"  Photons in window  : {best['n_photons']}")
    print("=" * 70)

    # Save to file
    out_path = os.path.join(OUTPUT_DIR, "coarse_search_results.txt")
    with open(out_path, "w") as f:
        f.write("Coarse Burst Oscillation Search — Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Event file     : {EVENT_FILE}\n")
        f.write(f"Burst start    : {BURST_START:.6f} s (absolute MET; t = 0 below)\n")
        f.write(f"Burst end      : {BURST_END:.6f} s (duration = {BURST_DURATION:.3f} s)\n")
        f.write(f"Window / step  : {WINDOW} s / {STEP} s\n")
        f.write(f"Search band    : {FREQ_LO}–{FREQ_HI} Hz\n")
        f.write(f"Normalisation  : {NORM}\n")
        f.write(f"LC bin size    : {DT:.8f} s (Nyquist = {0.5 / DT:.1f} Hz)\n")
        f.write(f"Total windows  : {n_windows}\n")
        f.write("All time values below are seconds since burst start.\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'idx':>4} {'t_start':>10} {'t_end':>10} {'n_photons':>10} "
                f"{'peak_freq_Hz':>13} {'peak_power':>11} {'status'}\n")
        for r in results:
            f.write(
                f"{r['window']:>4d} {r['t_start']:>10.4f} {r['t_end']:>10.4f} "
                f"{r['n_photons']:>10d} {r['peak_freq']:>13.3f} "
                f"{r['peak_power']:>11.4f} {r['status']}\n"
            )
        f.write("\n" + "=" * 70 + "\n")
        if best is not None:
            f.write("STRONGEST COARSE CANDIDATE\n")
            f.write(f"  Window index      : {best['window']}\n")
            f.write(f"  Window interval   : [{best['t_start']:.3f}, {best['t_end']:.3f}] s\n")
            f.write(f"  Peak frequency    : {best['peak_freq']:.3f} Hz\n")
            f.write(f"  Peak Leahy power  : {best['peak_power']:.4f}\n")
            f.write(f"  Photons in window : {best['n_photons']}\n")
        else:
            f.write("No valid candidate found.\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
