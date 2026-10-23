# Four new ground facilities — Codex, 2026-10-23 UTC

Request 9253fd2. Relative to all earlier review CSVs, NOT, LBT, Liverpool and FLWO are new facility keys. Prior global SVO discovery is not counted as a completed source harvest. Actual numeric files, source URLs, SHA256, first-pair statistics and contexts are in manifest.json; original page snapshots retained. No previous researcher files changed.

- NOT / NOTCam: 32 measured transmission files. Mix of manufacturer warm scans, cryogenic grism measurements and IAC 2021 warm lab scans. Units vary per file: early .dat scans micron/percent inferred; cryogenic .txt scans Angstrom/relative fraction inferred; IAC lab header nm, percent scale inferred. Per-file review corrections applied in manifest. Mounted/spare and warm/cold versions are retained, not interchangeable. Three duplicate byte contents across the whole packet are not additional unique curves.
- LBT / LBC: 54 numeric files, including current 2026-posted comma-delimited full/bandpass measurements and older 2022/2023 scans. Angstrom and percent. Full/bandpass are versions of a physical filter, not separate filters. Negative measurement noise retained.
- Liverpool / IO:O: 12 individual measured filter scans (nm, percent) plus official-page-linked author dataset on Figshare. The latter is Angstrom, percent and has NINE response columns: seven filters, CCD QE and cryostat window. QE below 2900 A is NaN. It is a component table, not a first-two-column-only curve.
- FLWO / KeplerCam: five JLA relative passbands in Angstrom from the maintained sncosmo distribution. Secondary research-library copy explicitly labeled. Old CfA link redirects to facility home, JLA origin returns 403, SVO connections refused. Component inclusion and epochs need further original documentation; no separate optics/QE/atmosphere claim.

104 numeric files/tables, 101 unique byte contents; these are neither a count of physical filters nor complete optical paths. No SVO bytes downloaded. Primary search failures recorded in research notes above. Existing source.html snapshots anchor attribution. Independent review lives in ../review/.

Reproducibility: harvest.py retrieves the three primary pages and linked files, finish.py adds comma-delimited LBT scans, FLWO and Liverpool component data. Manifest metadata corrections following header inspection take precedence over generated defaults. Run only for intentional refresh, since live files may change.

Independent review: 104/104 hashes pass. Manifest row counts are finite first-pair rows; 17 LBT files also contain NaN rows preserved in original bytes and counted explicitly in nonfinite_first_pair_rows. Sample source re-fetches match saved bytes.
