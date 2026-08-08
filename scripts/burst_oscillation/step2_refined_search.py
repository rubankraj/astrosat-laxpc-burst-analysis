#!/usr/bin/env python3
"""
step2_refined_search.py
=======================
Refined burst oscillation search on AstroSat/LAXPC event data.

Replicates the multi-segment sliding-window method described in the paper
(Section 3.6, "Search for Burst Oscillations", refined pass):

    A 4 s refined window is constructed around the coarse candidate by
    padding the coarse window by 1 s on each side. Three segment sizes
    (1 s, 2 s, 3 s) are slid across this window in steps of 0.076 s,
    0.1 s, and 0.1 s respectively, yielding 40 + 20 + 10 = 70 overlapping
    trial segments. For each segment, a Leahy-normalised power spectrum is
    computed (Stingray), and the peak power in the 10–1000 Hz band is
    recorded.

    The single highest peak power across all trials (Pmax) is taken as the
    refined candidate. Detection significance is computed as:

        x = exp(−Pmax / 2) × n          (trial-corrected chance probability)
        X = sqrt(2) × erfinv(1 − x)     (significance in sigma)

    The fractional rms amplitude is computed as:

        A_rms = sqrt((Ps − 2) / Nm) × sqrt(Nm / (Nm − Nbkg))

    where Ps is the Leahy power, Nm is the total photon count, and Nbkg
    is the background count (set to 0 if unknown).

Input
-----
The same barycentric-corrected 3–30 keV LAXPC event FITS file used in
the coarse search, plus the coarse candidate window (COARSE_START,
COARSE_END) read from the coarse search output.

Time convention
---------------
COARSE_START and COARSE_END are entered as seconds since burst start,
matching the coarse search output. Internal photon selection uses
absolute barycentric MET. All printed and saved output uses relative
time (seconds since burst start).

Output
------
- Console: per-trial progress and refined candidate summary.
- <OUTPUT_DIR>/refined_search_results.txt: full trial table and
  candidate block.

Usage
-----
Edit the CONFIGURATION block below (especially COARSE_START and
COARSE_END from the coarse search output), then run:

    python step2_refined_search.py

Dependencies: numpy, scipy, astropy, stingray
"""

import os
import numpy as np
from scipy.special import erfinv
from astropy.io import fits
from stingray import Lightcurve, Powerspectrum

# =============================================================================
# CONFIGURATION — edit these to match your observation and coarse result
# =============================================================================

EVENT_FILE = "/path/to/bary_level2_3_30keV.event.fits"
OUTPUT_DIR = "/path/to/results/BO_Burst_1"

BURST_START    = 195101672.035   # Burst start, barycentric MET (seconds)
BURST_DURATION = 16.106          # Burst duration (seconds)
BURST_END      = BURST_START + BURST_DURATION

# Coarse candidate window from step1 output (seconds since burst start)
COARSE_START = 1.0
COARSE_END   = 3.0

# Refined window: pad coarse window by 1 s each side (nominally 4 s total)
REFINED_START = COARSE_START - 1.0
REFINED_END   = COARSE_END   + 1.0

# Clamp to burst boundaries by shifting, not truncating, to preserve the
# full window span needed by the largest segment size.
_wlen = REFINED_END - REFINED_START
if REFINED_START < 0.0:
    REFINED_START = 0.0
    REFINED_END   = REFINED_START + _wlen
if REFINED_END > BURST_DURATION:
    REFINED_END   = BURST_DURATION
    REFINED_START = REFINED_END - _wlen
REFINED_START = max(REFINED_START, 0.0)
REFINED_END   = min(REFINED_END, BURST_DURATION)

SEGMENT_SIZES    = [1.0, 2.0, 3.0]   # Segment sizes (seconds)
STEP_BY_SEGMENT  = {1.0: 0.076, 2.0: 0.1, 3.0: 0.1}  # Step per segment size

FREQ_LO = 10.0     # Search band lower bound (Hz)
FREQ_HI = 1000.0   # Search band upper bound (Hz)

DT   = 1.0 / 4096.0   # Light-curve bin size (s); Nyquist = 2048 Hz
NORM = "leahy"

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


def chance_probability_and_sigma(p_max, n_trials):
    """
    Compute the trial-corrected detection significance.

    Uses the paper's relations:
        x = exp(−Pmax / 2) × n
        X = sqrt(2) × erfinv(1 − x)

    Parameters
    ----------
    p_max : float
        Peak Leahy power across all trials.
    n_trials : int
        Total number of independent trials searched.

    Returns
    -------
    x : float
        Trial-corrected single-trial chance probability.
    sigma : float
        Detection significance in units of sigma.
    """
    x = np.exp(-p_max / 2.0) * n_trials
    x_clipped = min(x, 1.0 - 1e-15)
    sigma = np.sqrt(2.0) * erfinv(1.0 - x_clipped) if x_clipped > -1.0 else np.nan
    return x, sigma


def fractional_rms(p_max, n_counts, n_bkg=0.0):
    """
    Compute the fractional rms amplitude of the detected oscillation.

    Uses the paper's relation:
        A_rms = sqrt((Ps − 2) / Nm) × sqrt(Nm / (Nm − Nbkg))

    Parameters
    ----------
    p_max : float
        Peak Leahy power (Ps).
    n_counts : int
        Total photon count in the segment (Nm).
    n_bkg : float, optional
        Background photon count (Nbkg). Default is 0.

    Returns
    -------
    float
        Fractional rms amplitude (dimensionless; multiply by 100 for %).
    """
    if n_counts <= n_bkg or n_counts == 0:
        return np.nan
    signal_power = max(p_max - 2.0, 0.0)
    arms = np.sqrt(signal_power / n_counts) * np.sqrt(n_counts / (n_counts - n_bkg))
    return arms


def main():
    if not os.path.isfile(EVENT_FILE):
        raise RuntimeError(f"Event file not found: {EVENT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("REFINED BURST OSCILLATION SEARCH")
    print("=" * 70)
    print(f"Event file      : {EVENT_FILE}")
    print(f"Burst start     : {BURST_START:.6f} s (absolute MET; t = 0 below)")
    print(f"Burst duration  : {BURST_DURATION:.3f} s")
    print(f"Coarse window   : [{COARSE_START:.3f}, {COARSE_END:.3f}] s (since burst start)")
    print(f"Refined window  : [{REFINED_START:.3f}, {REFINED_END:.3f}] s "
          f"({REFINED_END - REFINED_START:.3f} s)")
    print(f"Segment sizes   : {SEGMENT_SIZES} s")
    print(f"Step sizes      : {STEP_BY_SEGMENT}")
    print(f"Search band     : {FREQ_LO}–{FREQ_HI} Hz")
    print(f"Normalisation   : {NORM}")
    print(f"LC bin size     : {DT:.8f} s  (Nyquist = {0.5 / DT:.1f} Hz)")
    print("=" * 70)

    times = read_times(EVENT_FILE)
    print(f"Total photons in file          : {times.size:,}")

    refined_start_abs = BURST_START + REFINED_START
    refined_end_abs   = BURST_START + REFINED_END
    mask = (times >= refined_start_abs) & (times < refined_end_abs)
    win_pool_times = times[mask]
    print(f"Photons within refined window  : {win_pool_times.size:,}")

    if win_pool_times.size < 2:
        raise RuntimeError("Fewer than 2 photons in the refined window.")

    results = []

    for seg in SEGMENT_SIZES:
        step = STEP_BY_SEGMENT[seg]
        n_positions = int(np.floor((REFINED_END - REFINED_START - seg) / step + 1e-9)) + 1
        if n_positions < 1:
            print(f"\n  Segment {seg} s: refined window too short — skipped")
            continue

        print(f"\n  Segment size {seg:.1f} s  (step {step:.3f} s)  "
              f"→ {n_positions} overlapping positions")

        for pidx in range(n_positions):
            s_start_rel = REFINED_START + pidx * step
            s_end_rel   = s_start_rel + seg
            s_start_abs = BURST_START + s_start_rel
            s_end_abs   = BURST_START + s_end_rel

            smask     = (win_pool_times >= s_start_abs) & (win_pool_times < s_end_abs)
            seg_times = win_pool_times[smask]
            n_gamma   = seg_times.size

            base = {
                "seg_size": seg, "step": step, "pos_idx": pidx,
                "t_start": s_start_rel, "t_end": s_end_rel, "n_photons": n_gamma,
            }

            if n_gamma < 2:
                results.append({**base, "peak_freq": np.nan,
                                 "peak_power": np.nan, "status": "skipped_low_counts"})
                continue

            try:
                lc = Lightcurve.make_lightcurve(
                    seg_times, dt=DT, tstart=s_start_abs, tseg=seg
                )
                ps = Powerspectrum(lc, norm=NORM)
            except Exception as exc:
                results.append({**base, "peak_freq": np.nan,
                                 "peak_power": np.nan, "status": f"error: {exc}"})
                continue

            fmask = (ps.freq >= FREQ_LO) & (ps.freq <= FREQ_HI)
            if not np.any(fmask):
                results.append({**base, "peak_freq": np.nan,
                                 "peak_power": np.nan, "status": "no_freq_bins"})
                continue

            peak_idx   = int(np.argmax(ps.power[fmask]))
            peak_freq  = float(ps.freq[fmask][peak_idx])
            peak_power = float(ps.power[fmask][peak_idx])

            print(f"    pos {pidx:3d} [{s_start_rel:7.3f}, {s_end_rel:7.3f}] s  "
                  f"N = {n_gamma:5d}  peak = {peak_power:6.3f} at {peak_freq:8.2f} Hz")

            results.append({**base, "peak_freq": peak_freq,
                             "peak_power": peak_power, "status": "ok"})

    valid    = [r for r in results if r["status"] == "ok"]
    n_trials = len(valid)

    print("\n" + "=" * 70)
    best = None
    if not valid:
        print("No valid trials produced a power spectrum.")
    else:
        best    = max(valid, key=lambda r: r["peak_power"])
        p_max   = best["peak_power"]
        n_counts = best["n_photons"]
        x, sigma = chance_probability_and_sigma(p_max, n_trials)
        arms     = fractional_rms(p_max, n_counts)

        print("REFINED CANDIDATE")
        print(f"  Total trials (n)        : {n_trials}  "
              f"(paper: 40 + 20 + 10 = 70)")
        print(f"  Best segment size       : {best['seg_size']:.1f} s "
              f"(step {best['step']:.3f} s)")
        print(f"  Best segment interval   : [{best['t_start']:.3f}, {best['t_end']:.3f}] s "
              f"(since burst start)")
        print(f"  Peak frequency          : {best['peak_freq']:.3f} Hz")
        print(f"  Peak Leahy power (Pmax) : {p_max:.4f}")
        print(f"  Photons in segment (Nm) : {n_counts}")
        print(f"  Single trial probability: {x:.3e}")
        print(f"  Detection significance  : {sigma:.2f} sigma")
        print(f"  Fractional rms          : {arms * 100:.2f} %")
    print("=" * 70)

    out_path = os.path.join(OUTPUT_DIR, "refined_search_results.txt")
    with open(out_path, "w") as f:
        f.write("Refined Burst Oscillation Search — Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Event file      : {EVENT_FILE}\n")
        f.write(f"Burst start     : {BURST_START:.6f} s (absolute MET; t = 0 below)\n")
        f.write(f"Coarse window   : [{COARSE_START:.3f}, {COARSE_END:.3f}] s\n")
        f.write(f"Refined window  : [{REFINED_START:.3f}, {REFINED_END:.3f}] s\n")
        f.write(f"Segment sizes   : {SEGMENT_SIZES} s\n")
        f.write(f"Step sizes      : {STEP_BY_SEGMENT}\n")
        f.write(f"Search band     : {FREQ_LO}–{FREQ_HI} Hz\n")
        f.write(f"Normalisation   : {NORM}\n")
        f.write(f"LC bin size     : {DT:.8f} s (Nyquist = {0.5 / DT:.1f} Hz)\n")
        f.write(f"Total trials    : {n_trials}\n")
        f.write("All time values below are seconds since burst start.\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'seg':>5} {'step':>6} {'pos':>4} {'t_start':>10} {'t_end':>10} "
                f"{'n_photons':>10} {'peak_freq_Hz':>13} {'peak_power':>11} {'status'}\n")
        for r in results:
            f.write(
                f"{r['seg_size']:>5.1f} {r['step']:>6.3f} {r['pos_idx']:>4d} "
                f"{r['t_start']:>10.4f} {r['t_end']:>10.4f} {r['n_photons']:>10d} "
                f"{r['peak_freq']:>13.3f} {r['peak_power']:>11.4f} {r['status']}\n"
            )
        f.write("\n" + "=" * 70 + "\n")
        if best is not None:
            f.write("REFINED CANDIDATE\n")
            f.write(f"  Total trials (n)        : {n_trials}\n")
            f.write(f"  Best segment size       : {best['seg_size']:.1f} s\n")
            f.write(f"  Best segment interval   : [{best['t_start']:.3f}, {best['t_end']:.3f}] s\n")
            f.write(f"  Peak frequency          : {best['peak_freq']:.3f} Hz\n")
            f.write(f"  Peak Leahy power (Pmax) : {p_max:.4f}\n")
            f.write(f"  Photons in segment (Nm) : {n_counts}\n")
            f.write(f"  Single trial probability: {x:.3e}\n")
            f.write(f"  Detection significance  : {sigma:.2f} sigma\n")
            f.write(f"  Fractional rms          : {arms * 100:.2f} %\n")
        else:
            f.write("No valid refined candidate found.\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
