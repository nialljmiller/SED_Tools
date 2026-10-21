# Claude peer review: Grok VISTA (C07) and Gemini Cassini (C01)

Task 305a606ae4, 2026-10-20. Original files re-fetched into scratch dirs; no owner outputs edited.

## C07 — Grok VISTA mixed units / atmosphere vs hardware: **confirmed, with two additions**

Source: https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst/Filters_QE_Atm_curves.tar.gz
(re-downloaded and extracted; 19 members incl. `f02.sm.gz`).

Confirmed against the bytes/script:
- `f02.sm` divides filter, M1, M2 and `qe.tab` wavelengths by 1000 (nm -> um) but does **not** divide the `trans_*` wavelengths (already um, 0.9-5.6).
- `f02.sm` multiplies atmosphere transmission by 100 for the plot; atmosphere column 2 maxes at 1.0 (0-1 fraction). `qe.tab` max is 94.762 (percent-like); first rows `806 68`. So hardware = nm + percent, atmosphere = um + fraction. Grok's correction is right.
- Atmosphere headers: `trans_30_20.dat` is `#3.0mm H2O, airmass=2.0`.

Additions Grok/Codex should apply:
1. **Header format differs between the two atmosphere files.** `trans_10_10.dat` line 1 is `1.0mm H2O, airmass=1.0` with **no `#`**; `trans_30_20.dat` has `#`-prefixed lines. `f02.sm` skips two lines (`lines 3 999999`). A comment-char-based reader would mis-parse the first file; skip the first 2 lines for both.
2. The files give no model/version, and no explicit water/airmass pairing beyond those two (PWV 1/X=1 and 3/X=2 only). Do not interpolate across PWV/airmass as if it were a grid.

Remaining unknown (agree with Codex): atmosphere model provenance; whether QE is per detector.

## C01 — Gemini historical Cassini paired-filter interpretation: **Codex correction confirmed**

Sources (PDS Rings, COISS_0011):
- https://pds-rings.seti.org/holdings/volumes/COISS_0xxx/COISS_0011/calib/efficiency/efficiency_info.txt
- .../efficiency/systrans/issnabl1cl2_systrans.tab and issnacl1cl2_systrans.tab
- .../efficiency/ccdqe.tab (e-/photon, NAC CCD 220 / WAC CCD 361), na_optics.tab

Verified:
- `efficiency_info.txt`: systrans = "non-wavelength-integrated system transmission values (QE*optics*filter1*filter2)"; col 2 "Transmitted Flux" (no unit stated), col 3 times solar flux. So a systrans file is a **product**, not a single-filter curve.
- NAC BL1+CL2 (6,491 rows, 200.0-849.0 nm) vs CL1+CL2 (8,571 rows, 200.0-1057.0 nm); BL1/CL2 peak 0.1107, CL1/CL2 peak 0.2212. Every BL1CL2 wavelength has a nonzero CL1CL2 partner (same 0.1 nm grid, no zero denominators), so the quotient is computable on 200-849 nm only.
- **Two distinct quotients (do not conflate):**
  1. *Codex C01 quotient (per its issue text):* `systrans / (QE*optics) = filter1*filter2`. This recovers the filter **pair** (including the clear-filter CL factor), not a single filter. Gemini's historical notes proposed this division per Codex C01; I did not re-derive it numerically.
  2. *My numeric check:* `systrans(BL1CL2) / systrans(CL1CL2) = BL1/CL1` (CL2 and QE*optics cancel). Peak 0.955. This is a filter **ratio relative to clear**, not BL1 alone; CL1 is neither unity nor independently published here.
- **Attribution:** quotient 2 is **mine** (round 2 claude, representative NAC BL1 only). I read Gemini's round-2 source records (`round2/gemini/sources.jsonl`, Cassini NAC/WAC qe, optics and systrans-index records): they define systrans as the product and contain no quotient, so I do not attribute quotient 2 to Gemini. Quotient 1 is attributed to Gemini only as cited by Codex C01 (historical notes not re-read).
- CCD QE here is in e-/photon, so QE is not a dimensionless <=1 factor in all regimes; do not mix with fractional transmission conventions.

Conclusion: either quotient is a **filter-pair or filter-ratio** quantity, not an individual filter transmission. Actionable corrections:
1. Label quotient 1 "filter-pair transmission" and quotient 2 `BL1/CL1 (relative to clear)`; absolute BL1 requires an independent CL1 curve.
2. Restrict to the intersection of wavelength ranges (here <= 849 nm for BL1 in quotient 2).
3. Match **camera** (issna vs isswa) and record the **release applicability** of every file used (COISS_0011 volume and file version). The 2004-12-07 (ccdqe, optics) and 2014-04-25 (systrans) file publication dates are **recorded, but they do not by themselves prove incompatible calibration epochs** (M02): files of one calibration volume are meant to be used together, and the systrans files are defined as already containing QE*optics. Do not forbid mixing across dates; require a provenance check (efficiency_info.txt and PDS calibration documentation) and flag it when no such statement is found.
4. Keep the unit/convention of column 2 flagged as unstated in the label.

Corrected 2026-10-20 (manager M01/M02, task 7a29b39159); the earlier wording "do not combine across publication dates" and the single-quotient attribution to Gemini are **superseded** by the text above.

Scope: BL1 only checked; other filters not tested. Gemini was not run; its historical notes were judged via Codex's cited definition files.
