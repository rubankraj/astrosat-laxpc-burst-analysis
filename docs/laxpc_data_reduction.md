# AstroSat/LAXPC Data Reduction Pipeline

A complete walkthrough for installing the LAXPC software, downloading data, and reducing it into science-ready light curves and spectra. Based on the standard LAXPC analysis workflow.

Replace any placeholder path, filename, or ID (`<...>`) with your actual values.

---

## 1. Install the LAXPC Software

LAXPCsoft is the core package used to extract and analyze LAXPC data — intensity and spectral properties of X-ray sources. It's distributed as a zip file from the [AstroSat Science Support Cell](http://astrosat-ssc.iucaa.in/laxpcData).

**Create a directory for the software:**
```bash
mkdir LAXPCsoftware
```

**Unzip the downloaded package inside it:**
```bash
unzip LAXPCsoftware_Aug4.zip
```
This produces a folder named `LAXPCsoftware_Aug4`.

**Add the following to your `.bashrc`:**
```bash
export LAXPCSOFT="/home/path/LAXPCsoftware/LAXPCsoftware_Aug4/"
export LAXPCPATH="$LAXPCSOFT/LAXPC_CAL/"
export PATH="$PATH:$LAXPCSOFT/laxpc_bin"
```

**Open a new terminal and configure:**
```bash
./Configure
```
If this throws an error, use `./Configure_curl` instead.

---

## 2. Download and Access LAXPC Data

LAXPC (Large Area X-ray Proportional Counter) is an instrument aboard AstroSat that studies the temporal and spectral properties of X-ray sources in the 3–80 keV range.

1. **Download the observation** from the [AstroBrowse archive](https://webapps.issdc.gov.in/astro_archive/archive/Home.jsp).
2. Copy the downloaded dataset into a new folder and open it.
3. **Unzip the Level-1 data:**
   ```bash
   unzip LEVL1AS1LXP<...>.zip
   ```
   This creates a folder containing the observation data.
4. **Create an analysis directory:**
   ```bash
   mkdir Analysis
   cd /home/path/Analysis
   ```
5. **Point to your Level-1 data:**
   ```bash
   export LAXPCDATAPATH="/home/path/data/laxpc/"
   ```
6. **Initialize HEASoft** ([install guide](https://heasarc.gsfc.nasa.gov/lheasoft/install.html)):
   ```bash
   heainit
   ```

> This assumes LAXPCsoft is already installed and you have a basic familiarity with the LAXPC data format.

---

## 3. Create the Level-2 Event File and GTI

**Generate a file list from the Level-1 data:**
```bash
laxpc_make_filelist
```
This produces three text files: `eventfiles`, `filterfiles`, and `orb_filelist`.

**Create the Level-2 event FITS file:**
```bash
laxpc_make_event eventfiles
```
Output: `level2.event.fits`. The tool reports events read per LAXPC unit (10/20/30) and confirms the calibration response files used for each.

**Create Good Time Intervals (GTI)**, removing Earth occultation and SAA passages:
```bash
laxpc_make_stdgti filterfiles
```
Output: `usergti.fits`

**Custom GTI (optional):** you can supply your own start/stop times instead, either as plain ASCII:
```
500.3 3000.1
3500.0 4550.34
```
or as a FITS GTI file with `START`/`STOP` columns.

**Define an energy range (optional):** create an `eneinput` file with min/max energy in keV, e.g.:
```
3 50
```
If this file isn't present, the default range is 3–80 keV.

---

## 4. Extract Light Curves

```bash
laxpc_make_lightcurve -p pcu -t timebin -u user_gti_file -e energy_define_file -l layer_no level2.event.fits
```

| Flag | Meaning | Default |
|------|---------|---------|
| `-p` | PCU/LAXPC unit selection | all units |
| `-t` | Time bin (seconds) | 1.0 |
| `-u` | User GTI file | optional |
| `-o` | Output filename root | `lightcurve` |
| `-e` | Energy range file | `eneinput` (3–80 keV if absent) |
| `-l` | Detector layer | all layers |

**Output files** (for the default root name):
- `lightcurve.txt` — ASCII counts/sec vs. time for each energy band
- `lightcurve_Emin_EmaxkeV.lc` — FITS light curve file

**Example** — light curve using the default energy band and a 1.0 s time bin:
```bash
laxpc_make_lightcurve -u usergti.fits level2.event.fits
```
The tool's own printout confirms what it actually used — event file, time bin, GTI file, energy file, PCUs, and layers — which is worth checking against what you intended before trusting the output.

---

## 5. Barycentric Correction

Needed before timing analysis (e.g. burst oscillation searches).

```bash
laxpc_make_merged_orbit orb_filelist
as1bary -i merged_orbits.fits -f level2.event.fits -o bary_level2.event.fits
```
Confirm success by checking that `TIMESYS` changes from `UTC` to `TDB` in the output FITS header.

---

## 6. Spectra and Background Products

```bash
laxpc_make_spectra -u usergti.fits -o spectrum bary_level2.event.fits
```
Output: `spectrum_10.pha`, `spectrum_20.pha`, `spectrum_30.pha` (per LAXPC unit)

```bash
laxpc_make_backspectra -u usergti.fits filterfiles
```
Output: `backlxp10.pha`, `backlxp20.pha`, `backlxp30.pha`

```bash
laxpc_make_backlightcurve -t 50.0 -u usergti.fits -e eneinput filterfiles
```
Output: `Back_lightcurve_<Elo>_<Ehi>keV.lc`

---

## 7. Practical Notes

- **Long file paths:** `-o`, `-u`, and `-e` arguments longer than ~50 characters can get truncated by some LAXPC tools without any error message, silently producing bad output. It's safest to symlink your input files into a short-path working directory and pass bare filenames rather than full paths.
- **Verification:** before trusting a light curve for timing work, cross-check its pre-burst count rate against a previously validated value — the ratio should come out close to 1.000.
- **LAXPC10:** has a documented gain instability from a gas leak since March 28, 2018 — not relevant for pre-2018 observations.
- **LAXPC30:** also has known gain instability. Depending on the observation and analysis type, some studies exclude it from spectral fitting (`-p` flag) while retaining it for timing/light-curve work — check what's appropriate for your specific source and epoch.
- Duplicate `EXTNAME` warnings during file creation steps are harmless and can be ignored.

---

**Reference:** [Antia et al. 2017, arXiv:1805.05393](https://arxiv.org/abs/1805.05393) — LAXPC calibration and analysis software.
