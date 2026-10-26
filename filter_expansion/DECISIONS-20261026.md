# Nill's decisions, 2026-10-26 (supersede earlier holds in BRIEF-4)

**1. Hosting policy: ask forgiveness, not permission.** Maximise completeness.
- **Host everything:** publish every curve we can source and verify, whatever its licence class, including the held builds:
  - Sentinel-2 MSI 2A/2B/2C (`FX_MSI_PUBLISH=1`);
  - Suomi-NPP VIIRS;
  - restricted, permission and unknown sources.
- **Remove nothing:** never delete or withdraw a file. If an owner asks for removal, Nill decides then. The website already gives owners a contact.
- **Credit and terms are still mandatory:**
  - `licence.terms` must be the **complete, verbatim** statement, with its URL. About 47 curated terms end mid-sentence: re-quote them from the live page. Keck is fixed in `ac07578`.
  - Add `licence.scope` when the statement is a site-wide footer rather than a data licence.
  - Keep the swarm's `redistribution` class as found.
  - SVO-derived curves stay credited to SVO.

**2. Newer official curves replace older ones under the standard name. The older ones are kept as `<band>_old`.** This applies to JWST NIRCam (the Nov 2025 JDox files), Roman, and any later case.

**3. Original source versus SVO copy: release both.** The original-source curve takes the band name, and SVO's copy becomes `<band>_svo`. This applies wherever both exist, including:
- **Generic/Johnson:** research and document, with citations, the best original-source Johnson realization, including the LI/LII choice. Post it in the room for review before it replaces `Generic/Johnson/<band>.dat`. Then the current SVO-derived files become `<band>_svo.dat`.
- **The Internet Archive harvest:**
  - IDs NJM lacks are added credited as SVO-derived (stage and deploy, `ia/scripts/stage.py` and `deploy_ia.sh`);
  - IDs NJM already holds from an original source are added as `<band>_svo`.

**Exceptions to "never overwrite", granted for items 2 and 3:**
- **Renames:** rename and replace only as described above.
- **Records:** write a rename map CSV per change, like `filter_expansion/renamed_filters_20261025.csv`, and archive the old raw files in `source_evidence/`.
- **Index:** regenerate `index.json`.
- **Names:** no dots, and no case-insensitive collisions within an instrument.
- **MESA index files:** until Nill says otherwise, `_old` and `_svo` files are hosted and listed in `index.json` but **not** listed in the MESA instrument index file (`<Facility>/<Instrument>/<Instrument>`), so MESA's default bands don't change.

**4. Scope is wide (not guaranteed).**
- **Include:** about 120 commercial Earth-observation IDs (SkySat, WorldView, Pléiades, …) and the microwave/GHz channels (ATMS, Planck LFI, BICEP, …).
- **Also go ahead with:** the block C inventory gaps (large telescopes, ISO CVF, LICK, Landsat, MODIS, Sentinel-3, METOP, ULTRASAT, legacy GOES, SEVIRI and AVHRR).

**5. Codex:** take what it produces, accept partial results, give the rest to Gemini or Claude workers, and don't wait on its budget questions. **Grok:** no longer used. Its existing output stays.

**6. Solar flux and extinction:** the Mac session is implementing these. Solar flux from the CALSPEC Sun spectra; A_f/A_V from Fitzpatrick (1999) via `dust_extinction`, cited. Don't duplicate this work.

**7. Engineering for you (server branch):**
- **The SVO-compatible service:** commit `phase3/svc` and deploy it at `/sed_tools/svo_fps/`. Document the astroquery override (`SvoFps.SVO_MAIN_URL`). Accept SVO-form IDs (e.g. `Paranal/NACO.H`) through an alias map. The Mac session will test it with astroquery.
- **Tooling:** commit the phase 3–4 tooling (ia, svc, dispatch, verify_a, integrate_a, metrics, c4_*.py) under `filter_expansion/`, and archive block C's raw NOAA/JMA downloads in `source_evidence/filters/`.
- **SVO alias map:** WFIRST = Roman/WFI, SPIRE PSW/PMW/PLW = 250/350/500, MIPS 24mu = 24, GAIA2r = DR2r, and so on. Use it in the service and the coverage metrics. (`phase4/svo_missing.txt` is cleaned to 5,876 IDs; dropped entries are in `svo_missing_dropped.txt`.)

**8. Data fixes.** Every rename gets a map; nothing is deleted.
- **BOK/90prime `R`/`r`:** these collide case-insensitively. Rename one by what it is, with a map.
- **Stale DetectorType notes:** fix the 41, including the 16 CFHT/Megaprime split into `X` (full curve) and `X_fil` (filter only). Use the new curated band fields `svo_filterID` and `DetectorType_note` (coming from the Mac session today).
- **Small checks:** annotate HAWK-I `Knew`/`Kold` as identical; check NIFS `ZJ_298K` against the source headers; check the Spitzer MIPS 70 µm label.
- **Detector types:** research GOES/JPSS/Himawari conventions for the 156 `energy;assumed` bands, or explain them in notes.
- **Credit gaps:** SDSS (description and credits), GALEX (acknowledgement), and the UKIRT/WFCAM template placeholders.
- **Legacy header gaps:** fill FilterID, Components and Sources on older curves.

Report milestones in the room with the metrics table. Authorship Niall Miller <niall.j.miller@gmail.com>, no Claude trailers, server branch only, token-efficient.
