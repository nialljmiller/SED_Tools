# Ground harvest, Codex delegated session ground-9253 (2026-10-23 UTC)

Claims released: only this ground-agent directory edited. Six new research facilities, 134 numeric source files (tables can contain multiple curves). Existing discovery roster already listed these facilities; novelty means new downloaded component research, not new observatory names. No prior reviews found for CAHA, IRTF, JCMT, MMT, WHT or LICK in results/*reviews.csv and round2/*/reviews.csv. Parent performs independent comparison against supplemental harvests.

| Facility | Files | Numeric products and interpretation |
|---|---:|---|
| CAHA | 24 | PANIC 7 in-band filter scans at 100 K, nm/%; 6 out-of-band optical density scans (OD, **not transmission**). LAICA 10 headerless filter tables (nm/percent inferred from page and ranges, retain unresolved label). BUSCA 1 table has four beam splitter channel curves excluding detector QE; wavelength unit not explicit in table (nm inferred), fractions. |
| IRTF | 39 | NSFCam filters, filter+PK50 blocker combinations, and separate PK50 blocker. Primary page explicitly defines microns and absolute transmission 0–1. Some files extend to ~27 microns despite detector-sensitive 1–6 micron range. Historical NSFCam data linked by NSFCam2 page; not automatically current installed filters. |
| JCMT | 4 | SCUBA-2 pre-2017 predicted full stack files model450/model850; **15/16 columns without headers**, retain unresolved column mapping and frequency units (values suggest GHz). Do not read the second column as total response. Measured s8c/s8d are explicitly wavenumber cm^-1, arbitrary transmission and standard deviation. All four files are ASCII despite .pdf extension. Predicted stack includes window/dichroic; excludes mirrors/scattering; measurements at room temperature, not operational temperature. |
| MMT | 10 | Megacam five radial filter maps (nm, percent; 5 radii per file), plus five calculated throughput/component tables with header WAVE FLUX T QE NU STARToverNU EXTINCT THROUGHPUT R SOLGEL WINDOW TOTAL. These contain empty tab fields: preserve tabs/column alignment; inventory numeric-row scan collapses blanks and must not be used as an ingestion parser. Component tables wavelength units need confirmation (values suggest Angstrom), retain as research evidence. |
| WHT | 20 | INGRID measured warm filters and extrapolated cold filters, numeric HTML PRE blocks explicitly wavelength nm and %T. Cold values are estimates for 5 degrees and 77 K, warm scans ~293 K and 0 degrees. Negative scan noise preserved. |
| LICK | 37 | KAIT/Nickel telescope-system response curves from PantheonPlusSH0ES research-team release pinned at c447f0fea703fcd0fff57de5000947b5ca81286b. README gives Angstrom/transmission. Raw, shifted, cut curves retained as separate variants; AAA_README says shifted is old. No independent component separation. Excluded editor/temp/backup files. README's count of 16 describes an older subset, not all 37 files in this directory. |

## Source-link faults and exclusions

- CAHA PANIC `ascii_out_Ks` links to `ascii_out_Z.txt`: deduplicated; **Ks blocking curve not obtained**.
- CAHA LAICA `John_U` links to `john_B`: deduplicated; **U curve not obtained**.
- WHT J-continuum cold link points to H-continuum cold: deduplicated, **J-cont cold not obtained**. H-Barr and Pa-beta cold source links return HTTP 404 and are not counted.
- JCMT predicted columns have no inline names; identification requires consultation with the source authors/documentation. Do not claim all columns as independently identified components.
- Quick AAO/IRIS2 source searches found description/papers but no numeric files. APEX primary host timed out; SCUBA/LABOCA array-parameter RCP tables are spatial calibration, and were not counted. SVO host refused connections in both HTTP/HTTPS and web fetch; alternate svo.cab hostname returned 404. These failed facilities were replaced rather than counted.

## Artifacts and verification

`inventory.json` records every numeric file's original URL, relative path, SHA256, parsed numeric rows, column counts and endpoint samples. `SHA256SUMS` covers all 134 files. `source.html` snapshots preserve facility page definitions and links. No normalization or product manufacture performed. Number of source files is not number of independently validated curves. `harvest.py` regenerates the first five facilities; `lick.py` appends LICK and uses saved tree snapshot for pinned retrieval. Re-running harvest resets inventory to first five, so run lick.py afterward. All 134 hash checks passed locally, all files have nonzero numeric rows.
