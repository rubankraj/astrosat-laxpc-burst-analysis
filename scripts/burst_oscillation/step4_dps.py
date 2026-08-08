#!/usr/bin/env python3
"""
step9_dynamic_power_spectrum.py
================================
Dynamic power spectrum (DPS) around the burst oscillation frequency,
built directly from the barycentric event FITS file (no intermediate
CSV step) -- consistent with the coarse/refined search scripts.

Method:
    - Read photon arrival times directly from EVENT_FILE.
    - Define the burst interval from BURST_START / BURST_DURATION.
    - Slide a WINDOW_SIZE-second window across [DPS_T_START, DPS_T_END]
      (seconds relative to burst start) in steps of STEP_SIZE seconds.
    - For each window position, build a Leahy-normalized Powerspectrum
      and keep the power values restricted to the [FREQ_LO, FREQ_HI]
      band (typically bracketing the detected oscillation frequency).
    - Stack these into a (frequency x time) matrix and display it as a
      smooth mesh (pcolormesh + gouraud shading) rather than a blocky
      pixel grid, so the time-frequency evolution of the oscillation
      reads as a continuous surface.

Output:
    - <OUTPUT_DIR>/b1_dps_mesh.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from stingray import EventList, Powerspectrum

# Configure strict publication layout typography
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# =============================================================================
# CONFIGURATION
# =============================================================================

EVENT_FILE = "/home/ruban/AstroSat/observations/4U1728-34/T01_041T01_9000000362/analysis/bary_level2_3_30keV.event.fits"
OUTPUT_DIR = "/home/ruban/AstroSat/observations/4U1728-34/T01_041T01_9000000362/results/BO_Burst_1"

BURST_START = 195101672.035      # t_start, barycentric mission seconds
BURST_DURATION = 16.106          # seconds
BURST_END = BURST_START + BURST_DURATION

# DPS time range, in seconds SINCE BURST_START
DPS_T_START = -0.5
DPS_T_END = 16.0

WINDOW_SIZE = 2.0     # s -- sliding window length
STEP_SIZE = 0.1       # s -- slide step

# Frequency band to display (bracketing the detected oscillation)
FREQ_LO = 200
FREQ_HI = 700

DT = 1.0 / 4096.0     # light-curve bin size, same as coarse/refined search
NORM = "leahy"
MIN_COUNTS = 5        # minimum photons required to attempt a PDS for a window

VMIN, VMAX = 2.0, 40.0
CMAP = "Purples"

PLOT_PATH = os.path.join(OUTPUT_DIR, "b1_dps_mesh.png")


# =============================================================================
# Helpers
# =============================================================================

def read_times(event_file):
    """Read the TIME column from the barycentric event FITS file, sorted ascending."""
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


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.isfile(EVENT_FILE):
        raise RuntimeError(f"Event file not found: {EVENT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("    DYNAMIC POWER SPECTRUM (DPS) -- smooth mesh plot")
    print("=" * 60)
    print(f"Event file      : {EVENT_FILE}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Burst start     : {BURST_START:.6f} s")
    print(f"Burst end       : {BURST_END:.6f} s  (duration = {BURST_DURATION:.3f} s)")
    print(f"DPS range       : [{DPS_T_START:.2f}, {DPS_T_END:.2f}] s since burst start")
    print(f"Window / step   : {WINDOW_SIZE:.1f} s / {STEP_SIZE:.2f} s")
    print(f"Freq band       : {FREQ_LO:.1f}-{FREQ_HI:.1f} Hz")
    print(f"Light-curve dt  : {DT:.8f} s")
    print("=" * 60)

    all_times = read_times(EVENT_FILE)
    print(f"Total photons in file : {all_times.size:,}")

    # Only need photons spanning the padded DPS range (window extends
    # WINDOW_SIZE/2 beyond either end of the requested time grid).
    pad = WINDOW_SIZE / 2.0
    abs_lo = BURST_START + DPS_T_START - pad
    abs_hi = BURST_START + DPS_T_END + pad
    mask = (all_times >= abs_lo) & (all_times <= abs_hi)
    relative_times = all_times[mask] - BURST_START
    print(f"Photons in padded DPS range : {relative_times.size:,}")

    if relative_times.size < MIN_COUNTS:
        raise RuntimeError("Not enough photons in the DPS time range.")

    # -------------------- Frequency axis (reference PDS) --------------------
    sample_ev = EventList(time=np.linspace(0, WINDOW_SIZE, 200))
    sample_pds = Powerspectrum(sample_ev, dt=DT, norm=NORM)
    freq_mask = (sample_pds.freq >= FREQ_LO) & (sample_pds.freq <= FREQ_HI)
    freq_axis = sample_pds.freq[freq_mask]

    # -------------------- Sliding window loop --------------------
    time_grid = np.arange(DPS_T_START, DPS_T_END, STEP_SIZE)
    dps_matrix = np.zeros((len(freq_axis), len(time_grid)))

    for j, t_center in enumerate(time_grid):
        w_start = t_center - (WINDOW_SIZE / 2.0)
        w_end = t_center + (WINDOW_SIZE / 2.0)

        win_mask = (relative_times >= w_start) & (relative_times < w_end)
        seg_times = relative_times[win_mask]

        if len(seg_times) > MIN_COUNTS:
            try:
                ev = EventList(time=seg_times)
                pds = Powerspectrum(ev, dt=DT, norm=NORM)
                if len(pds.freq) == len(sample_pds.freq):
                    dps_matrix[:, j] = pds.power[freq_mask]
                else:
                    dps_matrix[:, j] = np.interp(freq_axis, pds.freq, pds.power)
            except Exception as exc:
                print(f"  [warn] window t={t_center:.2f}s failed: {exc}")

    # -------------------- Plot: smooth mesh (pcolormesh + gouraud) --------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    T, F = np.meshgrid(time_grid, freq_axis)  # shapes match dps_matrix (freq, time)
    mesh = ax.pcolormesh(
        T, F, dps_matrix,
        cmap=CMAP,
        shading='gouraud',
        vmin=VMIN, vmax=VMAX,
    )

    ax.set_xlim(DPS_T_START, DPS_T_END)
    ax.set_ylim(FREQ_LO, FREQ_HI)
    ax.set_xlabel('Time Since Burst (s)', fontsize=16)
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03, shrink=0.98)
    cbar.set_label('Leahy Power', fontsize=15)

    fig.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[SUCCESS] Smooth-mesh DPS plot saved to: {PLOT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
