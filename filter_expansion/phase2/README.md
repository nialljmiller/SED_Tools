# Phase 2 (2026-10-25): NJM curve fixes

BRIEF-2.md step 2. Every curve the 2026-10-25 audit flagged (105), every filename with more
than one dot (30) and every published curve without DetectorType (119) was fixed on NJM.
Mirror after the fix: 5,039 curves, 0 audit rows, 0 untagged, 0 dotted names, 0 misread by
released MESA (`released_mesa_reads`).

- `fixes_manifest.json`: one entry per changed file (250), following
  `source_evidence/filters/2026-10-23/manifest.json`: path, raw copy of the old file,
  operation, old and new SHA-256, plus `source_url` for re-sourced curves and `renamed_from`.
  The old files are on Cirrus in `source_evidence/filters/2026-10-25-fixes/<F>/<I>/`.
- `../renamed_filters_20261025.csv`: old -> new names (Paranal/NACO 28, ERIS/NIX 2).
  Inner dots became `p` (`IB_2.00.dat` -> `IB_2p00.dat`) because MESA names history columns by
  the text before the first dot.
- `raw_downloads_manifest.json`: raw files fetched for this phase (CFHT filter scans, NOIRLab
  DECam tables, the manuals cited for DetectorType), archived in
  `source_evidence/filters/2026-10-25-fixes/raw_downloads/`.
- `notes/`: what each worker did, per instrument, with the scripts used.
- `tools/`: `deploy_fix.sh` (backup, raw copies, rename, rsync, audit, migrate dry-run,
  generate_index, HTTPS check), `gate.py` (no new `fx_validate` errors vs the live copy),
  `verify_fix.py`, `survey.py`, and the raw-archive manifest builders.

Summary: 40 curves re-sourced from the original teams (CFHT/CFHT 36 and CFHT/Wircam 4 from
cfht.hawaii.edu scans); DECam curves confirmed point-for-point against the NOIRLab tables
they came from; the rest minimally curated with `fxlib.clean_curve` (negative tails clipped
to 0, exact duplicate wavelengths averaged, descending CFHT scans reversed); 119 curves tagged
from team documentation (`DetectorTypeSource=original`). Backups: Cirrus
`backups/phase2-fix-*`.
