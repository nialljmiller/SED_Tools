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

## Phases 3-4 (2026-10-25/26)

- `BRIEF-4.md`, `DECISIONS-20261026.md`: the phase 4 brief and Nill's decisions of 2026-10-26.
- `svo_aliases.json`: SVO filter IDs that NJM holds under another path (SVO ID -> NJM
  `Facility/Instrument/band`, with the basis of each), built by
  `phase4/tools/build_svo_aliases.py`; used by the SVO-compatible service and `metrics.py`.
- `svc/`: the SVO-compatible Filter Profile Service (`fps.php`, `getdata.php`, `_lib.php`,
  `build_catalog.py`), deployed to `data/svo_fps/` =
  https://nillmill.ddns.net/sed_tools/svo_fps/fps.php. See `svc/README.md` (astroquery
  override `SvoFps.SVO_MAIN_URL`, catalog rebuild).
- `ia/scripts/`: the Internet Archive harvest of SVO FPS captures (CDX enumeration,
  selection, throttled downloads, parse, staging, batch export/publish). Versions of
  2026-10-25 (the W2 worker of 2026-10-26 keeps editing its own copies).
- `phase4/`: `tools/` (`dispatch.py` model-free worker dispatcher, `verify_a.py`,
  `integrate_a.py`, `apply_b.py`, `patch_star.py`, `metrics.py`, `build_svo_aliases.py`),
  `INTEGRATE.md`, `BRIEF-CB.md`, `WORKERS-20261026.md`, `svo_missing.txt` (the 5,876 SVO
  IDs NJM lacked on 2026-10-26).
- `tools/recipes/c4_*.py`: block C4 recipes (GOES ABI, Himawari AHI, JPSS VIIRS, Sentinel-2 MSI).
  Their raw downloads are archived in `source_evidence/filters/2026-10-26-blockC-raw/`.
