# Live filter-pipeline audit — 2026-10-23

Scope: installed SED_Tools, remote acquisition, local catalogue/metadata, original-source fallback, detector headers, component recipes, viewing, combined MESA sets, and an isolated `custom_colors` run. No implementation fixes were made. Spectral-grid acquisition was deliberately left for the next stage.

## Environment and evidence

- Mac checkout: clean `main` at `5df36fa7ca560057453033302c5726996dd75a2c`, equal to fetched `origin/main`. Reinstalled with `python -m pip install -e . --no-deps`; `sed-tools --help` exposes the current filter commands, and both the CLI and Python import resolve to this checkout. Package version is `0.3.0`; Python is 3.14.
- Scratch data, scripts, machine-readable results and logs: `/Users/njm/.cache/SED_Tools/filter_pipeline_audit_20261023/`. The [curve preview](/Users/njm/.cache/SED_Tools/filter_pipeline_audit_20261023/filter_preview.png) plots downloaded published responses and two component products. The existing SED_Tools data tree and original MESA test-suite files were not changed.
- The live NJM index reported 3,963 filters in 75 facilities. The MESA run used a copy of `star/test_suite/custom_colors` under the scratch directory, with its instrument path set to the test product, AB magnitudes, PGstar off and `max_model_number=2`. Its existing `build/bin/star` was dated 2026-09-30; the checked MESA colors source was dated 2026-10-13, so the runtime findings are about the installed executable, not a newly rebuilt MESA binary.

## What passed

- Downloaded and independently parsed **637 expected root curves in 34 instrument sets**: all 28 registered component-source sets plus GAIA, 2MASS, WISE, SDSS, TESS and Kepler. After retries, all sets had the expected file count, a matching MESA index and every manifest-declared component file. The first four-at-once pass was 26/34 clean; a serial retry and explicit recovery of four UVIS2 component files completed the set. Full per-set results are in `consolidated.json`; first-pass failures are in `downloads.json` and `downloads.log` under the scratch directory.
- CLI and Python both built 13 real products: full filter+optics+QE for LSST, Liverpool IOO, VISTA and HSC; LSST with its 1.2-airmass atmosphere; ZTF filter+QE; band-selected DECam, NIRCam and ACS/WFC products; filter-only GSAOI and NIFS; GAIA's published response; and an energy-counting IOO copy. Product MESA indices matched their files, selected bands were respected, and output curves and headers passed the audit. See `products.json` and `products.log`.
- `filters-info --offline --json`, text `filters-info`, offline inventory, and `filters-detector --dry-run` worked. Detector dry-run reported zero proposed changes. Direct CLI `filters` and `filters-compose` worked. The interactive facility prompt accepted `59,63` and downloaded IRTF/NSFCam and Liverpool/IOO without further instrument or component prompts; see `bulk.log`.
- Real CLI `filters-combine` produced a 13-band GAIA+2MASS+WISE set and a 75-band ACS/WFC+WFPC2-PC set with conflict renaming and correct MESA index files.
- With NJM availability deliberately disabled, the original-source path fetched all 7 IOO, 7 PANIC and 26 NSFCam curves without SVO; see `fallback.log`. Expected rejection cases (unavailable GAIA QE, NIRCam's incomplete all-band recipe, overlapping published+filter factors, and atmosphere on space/no-site instruments) produced explanatory `ComponentError`s.

## Findings — no fixes made

### 1. Installed MESA `custom_colors` cannot use metadata-tagged SED_Tools files (high)

The installed MESA executable reached model 2 with a 13-band combined set but printed zero/negative zero-point errors for every band, plus zero integrated-transmission errors for some bands. With the seven-band composed IOO set it printed 14 zero-point and 14 integrated-transmission errors over two models. Both runs exited **0**, so the process exit status alone reports a false success. The stock, untagged Johnson set had no such errors. Removing **only** leading `#` metadata lines from scratch copies of the same combined and IOO curves made both runs finish without these photometry errors. The physical wavelength/transmission samples and MESA index were unchanged. See `mesa_optical_ir.log`, `mesa_ioo_raw.log`, `mesa_johnson_baseline.log`, `mesa_stripped.log`, and `mesa_ioo_stripped.log`.

This establishes a format-compatibility failure in the installed MESA path for SED_Tools' tagged `.dat` output. It does not isolate whether the MESA parser or SED_Tools' claimed MESA-ready format should change. The current MESA colors source also has a newer timestamp than the executable, so rebuild and retest before attributing the exact parser behavior to current source.

The inspected MESA colors source computes synthetic flux and zero points with wavelength (photon) weighting and does not inspect `DetectorType` headers. Consequently the SED_Tools energy/photon override is not conveyed to this MESA path by the current file format; this is a source-level interoperability concern, not an independently measured difference from the installed MESA executable.

### 2. Transient public-mirror HTTP 403 can leave component downloads incomplete (high)

Four concurrent instrument downloads produced intermittent `403 Forbidden` for otherwise public HST and JWST files. Serial retries fetched MIRI, NIRISS, WFPC2-WF and WFPC2-PC completely. A further serial UVIS2 download obtained all 63 root curves but got 403 on its last four QE components; immediately calling `NJMFilterGrabber.download_components('HST','WFC3_UVIS2')` fetched those four. The same URLs later returned HTTP 200; Cirrus files had normal 644 permissions. This pattern is consistent with temporary access throttling, but the server's exact 403 cause was not established. The downloader reports each failure and continues, without automatic retry/backoff. See `downloads.log`, `retry.log`, `uvis_retry.log`.

### 3. Partial filter directory can be accepted as complete when both index and manifest are unavailable (high)

During the 403 burst, the first WFPC2-WF attempt left 24 of 52 root curves and neither the MESA index nor manifest in the scratch directory. A later `Filters.fetch('HST','WFPC2-WF', ...)` returned that directory even though it was still incomplete. `FilterLibrary.published_complete` can only compare against expected bands when an index or manifest is present; with neither, the existence of any `.dat` suffices. When SVO is also unavailable, this can silently hand MESA a truncated set. A subsequent clean serial fetch recovered all 52. Reproduction evidence is the first-pass scratch directory state and `downloads.log`; the final serial copy is complete.

### 4. Two served published curves contain invalid samples (medium)

`CAHA/PANIC/Y.dat` has three non-increasing wavelength steps (duplicate wavelengths); `Gemini/GSAOI/Klong_cont.dat` has one negative transmission sample, minimum `-3.4e-05`. They are SVO-sourced published roots on NJM and downloaded without validation error. Their separately sourced `products/filter/` versions have strictly increasing wavelengths and nonnegative transmissions. The 637-curve consolidated scan found no other non-increasing axes, negative/nonfinite response values, missing detector or wavelength-unit headers, or root-index mismatches. See `consolidated.json`.

### 5. Availability text is contradictory for partial instruments (low)

`sed-tools filters-info HST/ACS_WFC --offline` says `Separate components : none identified`, then lists filter, optics and QE as available for 16 specific bands. JSON exposes the partial coverage correctly. The text line should be read as “no role available for every band,” but it currently sounds like no components exist. See `info_acs.txt`.

### 6. Product-folder discovery differs from the README claim (low)

`find_filter_sets()` included composed `products/` directories in the interactive combination list. The README says automatic filter discovery skips `components/` **and** `products/`. This may be intentional for combining a derived product, but the current behavior and documentation disagree. Detector dry-run also traversed components/products as designed for header tagging. No source files were altered by the dry-run.

## External limitation

The SVO Filter Profile Service endpoint refused connections during this audit. A simulated NJM outage on a set with no original-source builder (Generic/Johnson) therefore raised `Failed to fetch SVO index`. The original-source fallback passed as noted above, but a successful **live SVO fallback** could not be validated on this date. This is an external availability limit, not evidence of a SED_Tools parser defect.

## Reproduction entry points

Run from `/Users/njm/SED_Tools` with the editable install and a fresh scratch root. Use `python .../audit_downloads.py` for bulk download checks and `python .../audit_products.py` for component recipes; the precise scripts and their JSON/log outputs are preserved in the scratch directory above. For a single CLI route, use `sed-tools filters Liverpool/IOO --base <scratch>/filters_cli --components filter,optics,qe --bands g`. The MESA comparison logs and scratch `inlist_colors` are in `<scratch>/mesa_custom_colors/`; only that copied inlist was edited.
