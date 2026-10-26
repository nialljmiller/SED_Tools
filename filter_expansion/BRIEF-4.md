# Filter expansion phase 4: beat SVO, professionally

**Goal (Nill, 2026-10-26).** NJM becomes the filter service people cite instead of SVO:
- **More filters,** including ones SVO lacks.
- **Every band traceable and cited.**
- **Every data provider credited** the way it asks.
- **Nothing a data owner could reasonably ask us to take down.**

Being complete matters less than being professional. A missing filter is fine; an uncredited or unlicensed one is not.

## Where we are (2026-10-26)
| Measure | Now |
|---|---|
| Instruments / bands on NJM | 315 / 4,165 |
| Instruments with a description | 39 |
| Bands citing a paper | 423 |
| Instruments with recorded redistribution terms | 0 |
| Bands with an *assumed* DetectorType | 127 (NACO 33, NSFCam 26, SPHERE 24, INGRID 20, HAWK-I 11, IO:O 7, NIFS 6) |
| SVO IDs seen in Internet Archive captures | 8,974 (NJM holds 2,974) |

- **SVO IDs NJM lacks:** `phase4/svo_missing.txt` (6,000), ranked by facility in `phase4/svo_missing_by_facility.txt`.
- **Pipeline:** `filter_info/README.md` on the server branch. `filter_info/refresh.sh` rebuilds every info.json.

## Standards for every curve, new or existing
1. **Provenance.**
   - Archive the raw download on NJM under `source_evidence/` with its URL, retrieval time and SHA-256.
   - Give the curve header a `Sources=` field and a `FilterID`.
   - Never change a curve without a record of what changed and why.
2. **Permission to redistribute.** Record each instrument's terms in its curated file:
   `"licence": {"redistribution": "public|attribution|permission|restricted|unknown", "terms": "...", "url": "..."}`
   - **Agency and observatory data** (ESO, NOIRLab, STScI/MAST, IPAC/IRSA, ESA, NASA, NOAA, CFHT, ING, ...) is normally redistributable with credit. Still link the terms page.
   - **Manufacturer catalogue curves and commercial satellites** (Asahi, Chroma, Semrock, Omega, Custom Scientific, Baader, Astrodon, Planet/SkySat, Maxar, ...): check the terms.
     - If redistribution isn't clearly allowed, don't publish the curve.
     - Instead record a source-builder recipe (users fetch it from the owner) and list it in your report.
   - **No published curve may stay `unknown`.**
3. **Credit.** Every instrument's info.json must cite:
   - **(a)** the curve's origin (paper, or the data page);
   - **(b)** the instrument paper;
   - **(c)** the calibration paper, where one exists;
   - **(d)** the provider's requested acknowledgement, as curated `"acknowledgements": [{"for", "text", "url", "references", "applies_to"}]`. Many publish one (2MASS, SDSS, Gaia/DPAC, WISE, Spitzer, HST/MAST, ESO, Pan-STARRS, GALEX, DES, UKIDSS/VISTA, Kepler/TESS, ...). Copy their text verbatim, with its URL.

   SVO-derived curves credit SVO automatically. Never present an SVO copy as an original.
4. **Truthfulness.**
   - **References:** bibcode, DOI or arXiv keys that resolve (`python -m filter_info.references resolve`, then check `unresolved.txt`).
   - **Copied text** gets attributed in `notes`.
   - **Nothing from memory,** and no invented values.
   - **DetectorType:** needs documented evidence, or stays `assumed` with an explanation.
5. **Quality gates (Claude, before deploying).**
   - **Format:**
     - released-MESA `load_filter` reads the curve identically;
     - wavelengths in Å, strictly increasing;
     - transmission non-negative;
     - `Components` and `DetectorType` fields present, each with its source.
   - **Duplicates:** no duplicate of a band already on NJM.
   - **Cross-checks:**
     - compare against SVO's archived curve where one exists, and report differences;
     - check the pivot against published values.

## Work, in priority order
- **A. References, descriptions and terms for everything already on NJM.** This is the most visible win. Write curated `filter_info/curated/instruments/<Facility>__<Instrument>.json` for the ~276 instruments without a description, most-used first:
  - **Sets to start with:** HST, JWST, Spitzer, WISE, Gaia, SDSS, Pan-STARRS, DECam/DES, Rubin, Euclid, 2MASS, UKIDSS/VISTA, GALEX, Swift, Kepler/TESS, Johnson/Bessell/Cousins, and the large-telescope imagers.
  - **Done means:** description, facility description, links, at least one paper reference, the provider's acknowledgement where it asks for one, and licence terms.
  - **Roles:** Codex and Gemini research, split by facility (Grok is removed from all workflows). Claude verifies (`check_refs`, link checks, band names matching info.json) and integrates.
- **B. Detector types.** Resolve the 127 assumed tags and the 21 SVO disagreements (search info.json for `DetectorType_note`), with evidence per instrument. Report the evidence; don't change headers without it.
- **C. More filters.**
  - **SVO's missing IDs:** for the ~6,000 in `svo_missing.txt`, original sources come first. Archived SVO copies from the IA harvest are the fallback, credited as SVO-derived.
  - **Order:** major surveys and space missions first, then large telescopes (ESO/La Silla/Paranal, Gemini, Keck, Subaru, GTC, LBT, SALT, MMT), then Earth-observation and weather satellites (GOES, JPSS, Meteosat, Himawari, Sentinel), then the rest.
  - **Beyond SVO:** where official curves are public, add filters SVO lacks or has outdated: Rubin/LSST as built, Euclid, Roman, full JWST NIRCam/NIRISS/MIRI sets, SPHEREx, ULTRASAT, CSST, and so on. **This is how we beat SVO.**
- **D. SVO harvest.** Keep the IA harvest going, NJM's IDs first.
  - **After each batch:** archive it on NJM, list its manifest in `filter_info/svo_manifests.txt`, run `filter_info/refresh.sh`, and commit any new `references.json` entries.
- **E. SVO's remaining quantities.** SVO also lists F_sun and A_f/A_V.
  - **Research first:** find which solar spectrum and extinction law SVO uses, citing SVO's pages.
  - **Before implementing:** propose the choice to Nill in the room.

## Rules (unchanged)
- **Authorship:** commits authored by Niall Miller <niall.j.miller@gmail.com>, with no Claude trailers.
- **Branches:** data and tooling go to the `server` branch. SED_Tools code changes go as PRs, never to `main`.
- **No deleting:** never delete anything. Never print or store credentials.
- **Rate limits:**
  - the Internet Archive and arXiv throttle the shared home IP, so run one crawler at a time, with backoff;
  - NJM blocks over 500 requests per 5 seconds.
- **Token-efficient:** sample files, don't dump them.

## Reporting
Report in the room at each milestone, giving the table above refreshed, plus:
- licences still `unknown`;
- curves withheld for licence reasons, with their recipes;
- new filters not in SVO.

**Targets:**
- every instrument has a description and terms;
- at least 90 % of bands cite a paper;
- no unexplained assumed detector types;
- no published curve with unknown terms.
