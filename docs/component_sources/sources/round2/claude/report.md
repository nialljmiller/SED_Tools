# Claude space partition, round 2 (sed-components-quarter-20261020)

Checked 2026-10-20 (UTC). Cumulative: **24 of 96 baseline SVO keys characterized** (13 from round 1 re-examined, 11 new). SPHEREx is an additional, non-baseline carry-over. Files: `sources.jsonl` (63 records: 30 round-1 preserved with added fields, 33 new), `reviews.csv` (24 baseline + 1 additional), `progress.json`. No code, production data or other partitions touched; `results/` preserved.

**Depth varies and is stated per instrument.** "Reproduced" means I computed it; "representative" means one filter only. Siblings were not individually checked. The WebSearch tool was not permitted in this session, so discovery was by crawling known pages; literature (Jordi+2010, Morrissey+2007, Cropper+2016, Hora+2008) was **not** opened.

## Corrections to round 1 / existing builder
1. **SVO NIRCam vs NIRCam2025 (Codex C09, resolved for identity).** F200W: SVO `NIRCam2025` equals the JDox v7.0 individual file exactly (ratio 1.0000). The old SVO `NIRCam` curve matches *neither* the v7.0 file (ratio 0.81-1.06) *nor* the `nircam_throughputs_4Nov2022_v5.zip` (0.78-1.02). Round 1's implied "v5 zip = older SVO key" is withdrawn; the original for the old SVO release is not located (`not_found`, SVO-only).
2. **Pandeia reference data is ~20 MiB, not multi-GB.** STScI Outerspace lists JWST `pandeia-data-v2026p7-jwst` = 20 MiB and PSFs separately (4 GiB). The builder's hard-coded v4.0 bundle is superseded (2025.7, 2026.2, 2026.7 exist). All Box `v/` links gave HTTP 404 to anonymous GET from this host (including the builder's), so content is unread.
3. **SVO provenance facts** (tested on one filter each): HST SVO curves are `stsynphot.band(...)` products of TRDS components; SVO WISE = IRSA per-photon RSR x lambda, peak-normalised (already weighted); SVO GALEX = effective area (equal to GALEX archive tables, ratio 1.000), not a throughput; SVO Herschel PACS = HSA filter transmission only (despite SVO's "x bolometer" comment); SVO SPIRE PLW = HSA RSRF; SVO TESS is peak-normalised and **not reducible** to HEASARC v2.0 (ratio 0.16-4.6); SVO Roman F129 peak 2.87 m^2 exceeds every v8 SCA peak (2.42-2.81), so a different release.

## Common retrieval procedures
* **STScI Box/JDox files**: hrefs carry time-stamped ids; re-scrape the page rather than hard-coding. Browser User-Agent was blocked by one team site (jwst.astro.umontreal.ca: 403 for "Mozilla", page loads with "curl/8", files still 403).
* **TRDS synphot (HST)**: see HST section; graph/comp-map files `99c1930om_tmg.fits`, `aa621124m_tmc.fits` in https://ssb.stsci.edu/trds/mtab/. My resolver reproduces what stsynphot picks (all obsmode keywords consumed in every test). Throughput tables: BINTABLE WAVELENGTH [Angstrom], THROUGHPUT [dimensionless]; photon-counting. Some "components" are scale factors (`*_cor` up to 1.22) or time-dependent (`*_mjd`, `*_tds_def`).
* **HEASARC CALDB** (Swift): `https://heasarc.gsfc.nasa.gov/FTP/caldb/data/swift/uvota/bcf/<dir>/`; use astropy.
* Atmosphere: not applicable for every instrument in this partition.

## Dossiers
Status words follow the brief (available/documented/derived/not_found/blocked/unknown). "Combined" = a single published product including several roles.

### JWST/NIRCam2025 and JWST/NIRCam
* Identity: SVO `NIRCam2025` = JDox v7.0 (Jan 28 2025) detector-averaged system throughput per filter (29 files `<F>_mean_system_throughput.txt`); SVO `NIRCam` = an older release, original not located.
* Source: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters ; example https://jwst-docs.stsci.edu/files/216457506/216457520/1/1762454205059/F200W_mean_system_throughput.txt (ids change; re-scrape).
* Format: 2 columns, header `Microns Throughput`, 618 rows for F200W. Unit: micron; response = photon-to-electron efficiency (e-/photon, per the page). Includes OTE, NIRCam optics, dichroics, filters, QE (combined; page statement). Per-detector curves exist only in the older v5 zip (`detector_based_throughputs/`, 209 members). v5 and v7.0 differ numerically (1.6874 um: 8.34e-5 vs 8.02e-5).
* Filter/optics/QE separately: expected in Pandeia refdata `jwst/nircam/{filter,optical,qe}` (JDox) - **blocked** here.
* Pitfall: a `P`-wheel filter's curve already includes its F150W2/F322W2/F444W partner (page text).
* Recipe: scrape page, download v7.0 files, um->A x1e4, treat as photon-counting. For separable parts download Pandeia 2026.7 data in a browser, untar, read FITS (units unverified).

### JWST/MIRI
* Source of record: Box folder https://stsci.box.com/s/f1939jcr8i2um4kqcgehnnaad3ot9aje ("Instrument throughputs" link on JDox MIRI Filters page): `MIRI_transmission_v4.0` (27 files, 1.18 MB, updated 2024-09-12) and `v3.0`. Folder page loads; listing/download failed anonymously. Contents unread (filter vs system, OTE inclusion, units all unknown). SVO comment: OTE not included.
* Status: filter documented, optics/QE unknown, published documented. Next: browser download, or Pandeia `jwst/miri`.

### JWST/NIRISS
* Source: https://jwst-docs.stsci.edu/files/216454094/216454098/1/1762450948325/niriss_filter_tarball.tar (3.3 MB, 30 members). `DATA/NIRISS_<F>.txt`: cols wavelength [micron], filter throughput [dimensionless], PCE [electrons/photon]; 0.5-5.5 um step 0.001, 5001 rows. ReadMe (Nov 2022): filter measured cold by vendor, set to 0 below 1e-8 of peak; PCE = OTE x pick-off mirror x collimator TMA x camera TMA x detector x CLEAR/CLEARP, with contamination model; double-stacked short-wave filters include the blocker; filter-wheel filters have a 16% CLEARP loss. Numerically PCE contains the filter (F200W: filter 0.977, PCE 0.691 at 2.154 um; both ~1e-7 at 1.5 um).
* Derived: system without filter = PCE/filter where filter > ~1e-3 (a derivation, valid only on the filter's passband).
* Team scans `NIRISS_<F>_Flight.txt` at https://jwst.astro.umontreal.ca/?page_id=116: 403, unread.
* Pitfall: pupil/AMI losses are not in these curves; this tarball is older than Pandeia 2026.7.

### JWST/NIRSpec
* Filter wheel/gratings: https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters lists `jwst_nirspec_{f070lp,f100lp,f110w,f140x,f170lp,f290lp,clear}_trans.fits` and `_disp.fits` (prism, g140m/h, g235m/h, g395m/h). F110W opened: WAVELENGTH [MICRONS], THROUGHPUT [UNITLESS], 1019 rows; header `PEDIGREE='PLACEHOLDER'`, LITREF NTN-2015-017 (P. Ferruit, 2015-09-08), "none to be used as final ETC reference files". Only F110W opened.
* System throughput: only via ETC/Pandeia (blocked). SVO key has 5 grating+filter combinations; their derivation is not verified.

### HST WFC3_UVIS1, WFC3_UVIS2, WFC3_IR
* Published: https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs -> `.../_documents/UVIS.zip` (20.4 MB, 208 files, Oct 2020) and `IR.zip` (0.55 MB, 63 files). txt: `# WAVELENGTH THROUGHPUT`, Angstrom, fraction; FITS twin. Combined lab filter x QE x optics (page).
* Components (TRDS): UVIS F606W path = `hst_ota_007, wfc3_pom_001, wfc3_uvis_mir1_002, mir2_002, wfc3uvis{1,2}_f606w_008, wfc3_uvis_owin_002, iwin_002, wfc3_uvis_ccd{1,2}_003, wfc3uvis{1,2}_cor_005`; IR F160W path = `wfc3_ir_primary, secondary, pom, csm, fold, mir1, mir2, mask, rcp, wfc3_ir_f160w_008, win, qe_003, cor_004` (IR has no `hst_ota` entry). Fully separable: OTE, mirrors, windows, filter, detector QE.
* Reproduction (representative filters): UVIS1 F606W product vs zip: max abs diff 0.0; IR F160W vs zip: 0.32% (achromatic). UVIS2 resolved but not numerically reproduced. SVO ratio to product: UVIS1 1.0000, IR 1.0032. SVO reference epoch 2009-06-26.
* Pitfalls: `*_cor` > 1 is a scale; `mjd#` variants for time dependence; UVIS1 vs UVIS2 are different chips (SVO has both keys). QE is a component; no double application.

### HST ACS_WFC, ACS_HRC, ACS_SBC
* No STScI-published ACS throughput zip found (guessed URLs 404); the original is TRDS components https://ssb.stsci.edu/trds/comp/acs/ via the graph. F606W paths: WFC1 `hst_ota, acs_wfc_im123_004, acs_f606w_wfc_008, acs_wfc_ebe_win12f_005, acs_wfc_ccd1_031` (WFC2 uses ccd2_031); HRC `hst_ota, acs_hrc_m12_006, acs_hrc_m3_005, acs_f606w_hrc_008, acs_hrc_win_005, acs_hrc_ccd_018`; SBC F122M `hst_ota, acs_hrc_m12_006, acs_f122m_007, acs_f122m_tds_def_010, acs_sbc_mama_013`.
* SVO ratios to my products (F606W / F122M only): WFC 0.9971, HRC 1.0000, SBC 0.9625; offsets are constant (achromatic), likely version/epoch (unproven).
* HRC failed 2007; WFC chips differ by QE; SBC is a MAMA (no thermal-noise-free QE as CCD).

### GAIA/GAIA3
* https://www.cosmos.esa.int/documents/29201/4226701/GaiaEDR3_passbands_zeropoints.zip/b03ab6ac-8b02-9850-7586-7dd7cdbc84c9?t=1603980987171 (27.9 kB): `passband.dat` 781 rows, lambda F7.2 nm, then G, e_G, BP, e_BP, RP, e_RP (E14.9, 99.99 = undefined; ReadMe mislabels unit "mag"); `zeropt.dat` VEGAMAG/AB. G_RVS: `.../GAIADR3_GRVSFILTER.zip` (`grvsfilter.csv`: nm, transmissivity, error; zero point 21.317).
* Convention (Riello+2021 eqs 7-8): n_p = P * integral n_p(lambda) S(lambda) dlambda, S photon->e- response, so photon-counting; S = R* x exp(Legendre series), R* = pre-launch nominal response (Jordi+2010) for G. Passband = filter/prisms, optics, QE plus processing features, per release (do not mix DR2/EDR3). SVO G peak 0.7186 equals the file.
* Components: only pre-launch nominal curves (Jordi+2010) and DR2 sets (`GaiaDR2_Passbands_ZeroPoints.zip`, `..._Revised_...`) are published; separate tables not obtained (documented).

### TESS/TESS
* https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv : nm, transmission; 300 nm step 2 nm; peak 0.785 at 722 nm; v2.0 2020-07-30. Includes lens AR coatings, lens elements, CCD QE and AR, long-pass ~600 nm (Handbook v0.1 sec 3.1.2). On-axis; effective area 86.6 cm^2 (field dependent). Components: Handbook figures only. SVO curve differs (see corrections).

### Kepler/Kepler
* https://keplergo.github.io/KeplerScienceWebsite/data/kepler_response_hires1.txt : nm, absolute fraction, peak 0.725 at 589 nm; equals Handbook (KSCI-19033-001) Table 5 "Absolute". Multiply by obscuration/vignetting for effective area. Response = Schmidt corrector, primary mirror assembly, field flattener with bandpass coating, CCD QE; per-element curves only in Handbook Fig 13.

### WISE/WISE
* https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/figures/RSR-W{1..4}.txt : micron, RSR per photon (peak 0.88 for W1), uncertainty (ppt). Only W1 opened. SVO W1 = RSR x lambda normalised to peak 1 (ratio 1.0000): do not weight twice. Components not searched.

### Spitzer/IRAC, Spitzer/MIPS
* IRAC: https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/201125ch1trans_full.txt (similar for ch2-4, subarray and per-pixel cubes). micron, electrons/photon; page lists included components. SVO I1 matches (median ratio 1.0003).
* MIPS: https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt (31 kB, 21 March 2000; 24 um, 70 um NB, 160 um blocks of lambda[micron]/response, noisy, negatives). Convention unstated; not numerically compared with SVO.

### GALEX/GALEX
* https://asd.gsfc.nasa.gov/archive/galex/Documents/PostLaunchFUVImagingResponseCurves.htm (and NUV): inline `Wavelength | Effective_Area` 1200-3099 A, 1 A step; FUV peak 36.806 at 1480 A, NUV 61.66 at 2200 A. Identical to ground tables `.../tools/Resolution_Response/EA-{fuv,nuv}_im.tbl`. SVO = these (ratio 1.000). Unit cm^2 per SVO listing, not printed in source. Archival site only; components unknown.

### Swift/UVOT
* HEASARC CALDB `.../uvota/bcf/`: `ftrans/swuftrans20041120v102.fits` (extensions TRANSMISSION<U,B,V,UVW1,UVW2,UVM2,WHITE>, 641 rows, ENERGY eV / WAVELENGTH A / TRANSMIS), `effarea/swuinstea20041120v001.fits` (6601 rows, EA cm^2), `swueffarea20010101v014.fits`, `swumirrorea20010101v015.fits` (44 rows each), `senscorr/swusenscorr20041120v007.fits` (time dependence).
* SVO `_fil` = ftrans (exact). SVO plain curve = filter effective area (cm^2): INSTEA x ftrans reproduces its peak (~22.3), point-wise equality not tested. SVO `_trn` ("full transmission") not reproduced. Mission-start values; apply senscorr for epoch.

### Euclid/NISP, Euclid/VIS
* NISP: https://euclid.esac.esa.int/msp/refdata/data/NISP-PHOTO-PASSBANDS-V1-{Y,J,H}_throughput.{dat,fits} (page resolves `../data` to /msp/refdata/data). Columns WAVE, T_TOTAL, T_FILTER, T_TELESCOPE, T_NI-OA, T_QE; no units (nm by range: Y 905-1245, J 1005-1695, H 1405-2145). Product of four equals T_TOTAL to <5e-6 for Y, J and H (all opened). Centre of focal plane; telescope includes mirrors and dichroic. T_FILTER has tiny negatives; clamp.
* VIS: ESA refdata lists none, Q1 explanatory supplement (all 52 pages fetched) has no passband table, SVO source (EPDB) returns HTTP 401. Status svo_only; absence not proven.

### Herschel/Pacs, Herschel/SPIRE
* PACS: http://archives.esac.esa.int/hsa/legacy/cal/PACS/user/PCalPhotometer_FilterTransmission_FM_v1.fits : extensions red/green/blue, columns (transmission, wavelength [micrometer]) in that order; 2.5-500 um; peaks 0.517 (blue, 65.4 um), 0.583 (green, 97.8), 0.547 (red, 150.9). Filter only; SVO identical.
* SPIRE: http://archives.esac.esa.int/hsa/legacy/ADP/SPIRE/SPIRE-P_filter_curves/SPIRE_phot_RSRF_14_3.fits : ext `rsrf` 512 rows, frequency GHz, wavenumber cm-1, psw/pmw/plw. lambda[um] = 299792.458/f[GHz]. SVO PLW identical (peak 0.6124 at 444 um). Components not separated.

### Roman/WFI
* https://github.com/RomanSpaceTelescope/roman-technical-information (path `roman_technical_information/data/WideFieldInstrument/Imaging/EffectiveAreas/Roman_effarea_v8_SCA01..18_20240301.ecsv`): Wave [micron], effective area [m^2] for F062 F087 F106 F129 F158 F184 F146 F213 Prism Grism orders. Combined; includes aperture, so not a throughput. No component curves in the repo tree (159 paths). SVO F129 release differs. Pandeia Roman data blocked.

## Open items
Pandeia data (JWST, Roman) and MIRI Box folder need a browser download; UdeM NIRISS scans; ACS published throughput page; VIS original; MIPS convention; GALEX/WISE/PACS/SPIRE component curves; per-filter checks of every HST filter; licences (none evaluated this round).

## Reconciliation addendum (Task f91328df1a, Codex audit incorporated)
* **Counts.** 24/96 baseline keys is a *breadth* milestone (identity and original source located), not 24 fully characterized. Component-level original files verified only for HST WFC3/ACS (6), Euclid NISP, Swift UVOT. The old "14 rows / 82 remaining" counted SPHEREx; correct: 13 baseline reviewed in round 1, 83 others, SPHEREx excluded.
* **C09 (NIRCam).** Accepted. `reviews.csv` now has a `release_mapping` column: NIRCam2025 = JDox v7.0 (F200W exact); the v5 zip (2022-11-04) is a different release and must not be used for either key; the original for SVO `NIRCam` is unresolved (open).
* **Euclid.** Codex independently confirmed NISP Y separability (max residual 4.98e-7); J/H were mine. Header has no units (nm inferred). SVO NISP release mapping unverified. VIS: searched, not found, limited (ESA refdata, Q1 supplement, SVO source 401); not searched: Cropper+2016, ESA papers; absence not proven.
* **Searched-not-found vs not-searched** is in `reviews.csv` column `component_search_scope`. Not searched at all (incomplete depth, not "absent"): WISE, MIPS, GALEX, PACS, SPIRE components; literature for Gaia/TESS/Kepler/Roman. Blocked: Pandeia/Box (JWST, Roman), MIRI contents.
* **No new original searches**: WebSearch was denied this session, so no depth was added.

## Round 3 addendum (Task 3fe7267656): component searches and literature
Method: direct HTTP (curl) of official pages, papers (arXiv), and archive files; WebSearch not used. 12 new source records in `sources.jsonl`; `reviews.csv` status and `component_search_scope` updated for nine rows. These are the results; "searched" is limited to the sources named.
* **WISE**: IPAC RSR-W1..W4.txt are combined per-photon RSRs. Wright+2010 says the RSR is a weighted mean of measured, component-predicted and design responses, so no standalone component curves exist in these sources (searched, not published).
* **MIPS**: only `MIPSfiltsumm.txt` (2000-03-21, combined 24/70NB/160 um). Handbook PDF and Rieke+2004 unopened; component absence not proven.
* **GALEX**: Morrissey+2007 gives mean/peak effective area and materials, no curves. Archive effective-area files: see Round 4 (located, URLs verified).
* **PACS / SPIRE**: Poglitsch+2010 Fig 6 is filter x bolometer combined; Griffin+2010 describes edge filters/dichroics and cites the RSRF. Observers Manuals unopened.
* **Gaia**: Jordi+2010 and Riello+2021 opened; passband = Ag mirrors x attenuation x CCD QE (three CCD QE types, prism coatings); the ESA EDR3 `passband.dat` is combined. Component tables not found there.
* **TESS**: HEASARC v2.0 CSV (2020-07-30) combined; Ricker+2015 plots lens transmission and CCD QE only as a figure.
* **Kepler**: Instrument Handbook sec 3.3 describes the element-wise response (Fig 13) and a combined channel-11.2 table; data not extracted; MAST per-channel model not retrieved.
* **Roman**: **correction** - detector QE per SCA exists (`FPSPerformance/WFI_Quantum_efficiency.ecsv`), plus band edges; effective areas combined. Mirror/filter tables not found. Roman papers unopened.
* **Still incomplete depth**: MIRI, NIRSpec system part, Euclid/VIS, MIPS, GALEX, PACS, SPIRE, Kepler, Roman optics/filter. C09 (NIRCam original release) remains open. `progress.json` now has `complete: false` and the stale completion note replaced.
* **Not done**: Observers Manuals, Kepler Fig 13 digitisation, Pandeia Box downloads. Candidates for Codex verification: Pandeia JWST/Roman data (Box, anonymous GET fails), MIRI Box folder, Euclid VIS passband original (SVO EPDB 401).


## Round 4 reconciliation (task 305a606ae4) — current vs superseded
Current claims; superseded ones are marked in `sources.jsonl` (`claim_status`) and `reviews.csv` (`superseded_claims`).
* **WISE**: IRSA RSR-W1..W4 are per-photon and **not** peak-normalised (peaks 0.8798, 0.750, 0.420, 0.300). The earlier "peak 1" note is superseded. SVO W1 = RSR x lambda renormalised.
* **MIPS**: `MIPSfiltsumm.txt` (21 Mar 2000) is the digitized Handbook Figs 2.3-2.5; Handbook sec 2.2.4 states R is e- per unit energy, detector+filter. "Convention unstated" applied to the file only.
* **GALEX**: effective-area tables located and verified: post-launch pages `.../Documents/PostLaunch{FUV,NUV}ImagingResponseCurves.htm` and `.../tools/Resolution_Response/EA-{fuv,nuv}_im.tbl`. The Round 3 "not located" line is superseded.
* **PACS**: manager full-array comparison (blue, 405 rows, max abs diff 5.0e-7) replaces peak-only reasoning. Equality does not show whether detector response is in the product.
* **Roman**: detector QE is published separately (`FPSPerformance/`); old "no QE" superseded; mirror/filter tables still not found.
Peer review of Grok VISTA (C07) and Gemini Cassini (C01): `peer-review.md`.


## Round 5 schema and correction pass (task 7a29b39159) — current, edited 2026-10-20T19:50:31Z
Edit time is actual UTC; source-inspection times are per-record `checked_utc`. Record numbers below are 1-based JSONL line numbers.
* **Schema**: all 79 `sources.jsonl` lines parse and all now carry `source_basis`, `retrieval_instructions`, `wavelength_unit`, `response_unit` (49 records, 98 fields were empty before). A new `unit_provenance` field says whether each unit is stated in the source, assumed/inferred (labelled) or unknown. Counts: 16 documentation/index/paper records are `not_applicable`; 10 records have an `unknown` unit (lines 7, 14-17, 23, 24, 32, 34, 60); 7 carry a labelled inference (lines 8-10 WFC3 headers give no units, Angstrom/fraction assumed; lines 14-17 NISP nm inferred from values). I did not invent any unit.
* **Small re-reads this pass** (headers only): WFC3 UVIS2 and IR `.txt` (`# WAVELENGTH THROUGHPUT`, no units); Kepler low-res (`#(nm)`); Gaia G_RVS README (nm, transmissivity); Roman `WFI_Quantum_efficiency_mean_median.ecsv` (QE unit `''`, wavelength bins in nm); MIPSfiltsumm.txt head (columns lambda/response, no unit keyword). Everything else was taken from earlier records.
* **Links**: landing/data URLs of lines 63-79 returned HTTP 200/206 with plausible content types (status and type only; content parsed only where stated).
* **Vocabulary**: `combined_only` (GALEX filter/optics/qe) was outside the shared status vocabulary. Replaced by `not_found`; the distinct meaning is kept in new column `combined_product_only` (yes for WISE, MIPS, GALEX, Pacs, SPIRE) plus `status_vocab_note`. Brief not edited. 25 rows parse; all status values are in the vocabulary.
* **M01/M02** (`peer-review.md` C01): two distinct Cassini quotients, `systrans/(QE*optics)=filter1*filter2` (Codex C01) and `systrans(BL1CL2)/systrans(CL1CL2)=BL1/CL1` (mine; Gemini's records contain no quotient). File publication dates do not prove incompatible epochs; camera and release applicability must still be checked.
* **M03**: `progress.json` timestamp replaced with actual UTC; the guessed 15:30Z value is kept under `superseded_timestamps`.
* **Counts unchanged**: 24 baseline keys characterized (breadth), 25 review rows incl. SPHEREx, 79 source records. Full component depth only for HST(6), NISP and UVOT. Remaining unknowns: items under `schema_audit_task_7a29b39159.remaining_unknown_units` in `progress.json`, plus the incomplete-depth keys already listed there.
