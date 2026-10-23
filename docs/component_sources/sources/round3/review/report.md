# Round 3 independent review

Request: 9253fd2ba9540be0d9191916. Reviewer: Codex delegated crosscheck. Claims: round3/review only.

## Baseline

Across original results and all round2 reviews.csv files: 56 facility/group keys and 165 instrument pairs already reviewed. This includes Claude supplemental ground and ground2. Unreviewed means absent from those review records, not absent from all literature. Full keys and source paths: existing-coverage.json.

Novelty must use physical-facility identity and aliases: WFIRST=Roman, P200=Palomar; ESO/NOAO are umbrellas, MKO a filter standard. Additional instruments at already reviewed observatories improve completeness but cannot count as new facilities.

NOIRLab supplemental evidence directory currently contains only manifest_noirlab.json; no numeric curves are available there to credit. Existing inventory.csv has 633 entries and is historical, not round3 output.

## New artifacts

Completed independent review of final packets: **238 numeric ground files (229 distinct contents), 171 normalized space CSVs (171 distinct contents)**. All 409 listed hashes and independent row/range checks pass; all space raw download hashes also pass. Space CSVs independently reconstructed from raw text/HTML/FITS/tar members: 171/171 agree, including XMM area scaling by 0.001. Sample re-fetches across 12 ground groups (10 physical facilities) and 8 space download groups all return HTTP 200 with identical hashes. MOST/CoRoT/PLATO share CHEOPS software FITS, so no separate source retrieval was possible or claimed. Evidence: independent-audit.json, independent-ground-audit.json, space-source-derivation.json, source-refetch.json, ground-source-refetch.json.

**Novelty passes:** 10 newly reviewed physical ground facilities: NOT, LBT, Liverpool, FLWO, WHT, IRTF, MMT, CAHA, JCMT, LICK. CAHA-BUSCA and CAHA-LAICA are instruments at CAHA, not extra facilities. 10 newly reviewed flown space facilities: AKARI, COBE, IRAS, WMAP, PLANCK, MSX, CHEOPS, MOST, Corot, XMM. PLATO is an extra planned-facility software curve and excluded from flown count. These keys do not overlap the 56-key baseline. "New" means new in this research inventory, not newly discovered observatories.

### Corrections and practical limitations

- Original NOT blanket micron/percent metadata was wrong for mixed formats. Per-file suggestions sent to owner and applied: cold measured NOTCam .txt wavelength is Angstrom inferred; warm IAC/NDC headers specify nm; legacy .dat scales imply micron. Response scales differ, and inference is explicitly labeled. No normalization/absolute calibration certified.
- 17 LBT files contain one source NaN row each. Manifest rows are finite rows; source nonfinite values remain preserved. Hash/stat checks pass under that stated convention. Retain negative scan noise rather than silently clipping it.
- Liverpool IO:O model table contains wavelength in Angstrom and nine percentage-response columns: u,g,r,i,z,B,V,CCD QE,cryostat window. Header explicitly documents missing QE below 2900 Angstrom (NaN). A two-column inventory counts only u, so 238 is a file count, not a number of curves or complete optical paths.
- MMT manufacturer tables contain multiple radial scans and sparse tab-separated cells; z has variable numeric widths 6/7/9. Preserve tab slots and headers; whitespace-collapsing parsing cannot safely attribute each radial sample. Modeled MMT tables have 10 columns and need their own labels.
- JCMT .pdf-named downloads are actually numeric ASCII, as the primary page explicitly explains. Predicted models have 15/16 columns with no in-file column labels. Do not assume the second column is total throughput or certify component decomposition. Measured 3-column profiles have arbitrary vertical scaling per primary page; retain uncertainty/variance columns and known fringing caveat.
- CAHA BUSCA is a 5-column channel-response table (four channels); file count understates curves. WHT warm/cold scans have descending coordinates and must be sorted with responses when interpolation requires it.
- Space software curves: CHEOPS is mission-team pycheops; MOST/CoRoT/PLATO are secondary curves in that software with unresolved original calibration provenance and response units. They are not SVO downloads. Ground FLWO is sncosmo author-data hosting and LICK is a pinned Pantheon calibration dataset, not telescope-hosted scans.
- Planck native spectral coordinates include GHz/inverse cm; do not relabel everything wavelength. Nonmonotonic native grids, negative values and source flags are retained. IRAS 25 micron source typo suspicion remains unresolved. These packets increase coverage but do not establish a complete optical path for every instrument.

Review claims released. Existing researcher files unchanged by reviewer.
