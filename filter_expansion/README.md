# Filter expansion, October 2026

Record of the 2026-10-25 expansion of the NJM filter mirror (`data/filters/`), where
new instruments were built from the original teams' published curves instead of SVO.

- `BRIEF.md`, `BRIEF-2.md`: Nill's briefs (format contract, phases, rules).
- `research/`: the source surveys (`codex*.csv/md` block A ground and C1 generic systems,
  `grok.csv/md` block B space, `gemini.csv/md` block C2 large telescopes). Each row gives
  the instrument, band, original source URL and the reasons for any held band.
- `tools/`: the build tooling. `fxlib.py` (download, unit conversion, `clean_curve`,
  `build_instrument`), `recipes/*.py` (one recipe per source), `build.py` (runs recipes),
  `fx_validate.py` (format contract and pivot checks), `deploy.sh` (backup, add-only rsync,
  `migrate_filters.py --dry-run`, `generate_index.py`), `verify_https.py` (byte comparison
  against the public mirror). See `tools/README.md`. The scripts expect the original
  workspace layout (`downloads/`, `stage/`, `.venv/` next to `tools/`).
- `provenance/<Facility>__<Instrument>.json`: per-band source URLs, DetectorType and its
  source, cleaning applied, reference and measured pivot wavelengths, raw-file SHA-256.
- `raw_downloads_manifest.json`: every raw download (path, source URL, SHA-256, bytes,
  retrieval time). The files themselves are archived on Cirrus under
  `source_evidence/filters/2026-10-25-expansion/` (not served publicly).
- `svo-diagnostics-a8c7ef0f.md`: why SVO FPS (`svo2.cab.inta-csic.es`) timed out.
