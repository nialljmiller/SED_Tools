# Claude worker: BRIEF-2 step 2, audit flags (13 NJM instruments, 45 curves)

Action for all 45 flagged curves: minimal curation with fxlib.clean_curve. Header line kept as-is, untouched rows kept byte-identical, other files in each dir byte-identical (asserted). Script: claude-auditflags-stage.py. Change records: changes/<F>__<I>.json (45 entries). No downloads used, so no provenance JSON.

Non-increasing wavelengths: every case is an exact repeated wavelength, with no decreasing steps and no spliced or reversed scans. Duplicates were averaged. Nothing held.
- Astrosat/UVIT *_old (one duplicate each). F172M_old's duplicate is the band edge (1620 Å: 0.0 and 0.7607), so averaging gives 0.38 and the pivot moves +0.031%.
- BOK/90prime B/U/V (4/7/2 duplicates; the U pairs have identical values).
- CTIO/CTIO 6130_590 and 8123_1313 (150 and 65). Wavelengths look rounded to 10 Å.
- CTIO/ISPI FeII/Pa (280/286, rounded to 1 Å) and H/J/K (4/2/3).
- CTIO/SMARTS V1 (4 identical pairs).

Negatives were clipped to 0. All lie outside the half-maximum range.
- AAO 13 files, AlSat1 2, CMP_SAI 3, CSTAR 2, Spartan 2, TIRCAM2 3: tiny (|min| <= 0.0046).
- CAHA 385_8/395_6/402_9/408_4: 129-176 negatives each in the blue tail. The minimum is -0.068, which is 24% of peak for 385_8. Positive noise in the same tail is untouched. Flag: this scan baseline is noisy.
- Gemini/AcqCam_S I: 63 negatives at 11254-11987 Å, minimum -0.094.

Pivot shifts: all < 0.1%. The largest is CAHA 385_8 at -0.095%.

Re-sourcing: tried briefly. The NOIRLab filter list has ISPI J/H pages but no numeric files for them, and it gave no 6130/8123 match. No numeric file for Bok 90prime was found. Nothing was re-sourced.

Checks:
- fx_validate (13 instruments): 97 errors on current -> 52 on stage. All 45 flagged errors are gone. The 52 that remain are already present in current: 13 "no manifest.json", 38 "header lacks Components/ComponentsSource" on unflagged UVIT/Spartan/etc. files, and 1 BOK/90prime case collision (R/r).
- SED_Tools audit_filter_curves: 582 curves, 0 rows.
- released_mesa_reads: 45/45 True.
