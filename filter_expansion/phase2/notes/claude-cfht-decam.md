# CFHT + DECam fixes (Claude worker, BRIEF-2 step 2)

Script: phase2/fix/scripts/claude_cfht_decam.py (run with `.venv/bin/python -E -P`, since numpy lives in user site, so `-I` cannot import it).
Outputs: stage/filters/{CFHT/CFHT,CFHT/Megaprime,CFHT/Wircam,DECam/Broad,DECam/Medium,DECam/Narrow,DECam/all},
changes/<F>__<I>.json, stage/provenance/<F>__<I>.json. Raw downloads: downloads/CFHT/cfh*.dat (36), downloads/DECam/scidoc0472.txt, scidoc3138.xls (plus scidoc0473.xlsx, fetched but not used).
None of these dirs had a manifest.json, so each now gets one from InstrumentManifest.from_published (fx_validate requires it).

- CFHT/CFHT (38): 36 cfhNNNN curves re-sourced from https://www.cfht.hawaii.edu/Instruments/Filters/curves/cfhNNNN.dat.
  Each original is one strictly descending scan (nm, %), so reversing it is safe (no spliced scans). The SVO copies had wavelengths
  rounded to whole Å, which caused the 13 non-increasing flags. Negatives in the originals were clipped to 0.
  I.dat and R.dat have no CFHT filter number (404 on CFHT), so they got clean_curve clipping only. Pivot shifts are all <= 0.031%.
- CFHT/Megaprime (6): clean_curve clips 1 negative per curve. These are SVO-built totals and filter-only siblings with
  non-SVO FilterIDs, and the totals cannot be rebuilt from CFHT filter files alone. Pivot shift <= 0.0005%.
- CFHT/Wircam (4): H2, W, LowOH1 and LowOH2 re-sourced from CFHT cfh8304, 8105, 8104 and 8102 (WIRCam page wircam.html). Pivot shift <= 0.0001%.
- DECam (Broad 7, Medium 5, Narrow 9, all 21): every current curve is sample-for-sample identical to the NOIRLab originals
  (scidoc0472.txt "DECam Filter transmission" for ugrizY+VR; scidoc3138.xls for M/N). The negatives are in the originals,
  so g, r, i, z, Y and N662 were clipped with clean_curve (pivot shift <= 0.030%). All curves were tagged
  DetectorType=photon (DECam is a CCD camera), DetectorTypeSource=original, with FilterID, WavelengthUnit,
  Components=filter, ComponentsSource=original and Sources. The unflagged curves were changed with write_filter_header (samples byte-identical).
  Note: scidoc3138's VR sheet (1 nm, 791 rows) differs from scidoc0472's VR. The mirror keeps scidoc0472's.

Held / not fixed: CFHT/Megaprime/Halpha_off_old.dat lacks Components/ComponentsSource. This predates the fix and was
not flagged; SVO is down, so it was left unchanged (fx_validate reports 2 errors for it).

Checks: fx_validate (root phase2/fix/stage): 6/7 ok, Megaprime 2 errors (above). Audit of staged CFHT+DECam: 452 curves, 0 rows.
released_mesa_reads True for all; 0 untagged; all unchanged files byte-identical to current/.
