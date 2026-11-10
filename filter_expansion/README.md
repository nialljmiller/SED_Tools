# filter_expansion

What remains of the October 2026 expansion of the NJM filter collection (`data/filters/`), when
instruments were built from the original teams' published curves instead of SVO. The build
tooling, briefs and research surveys of that work are kept at tag `filter-expansion-record-2026-10`.

- `provenance/<Facility>__<Instrument>.json`: per-band source URLs, DetectorType and its source,
  cleaning applied, reference and measured pivot wavelengths, raw-file SHA-256.
  `filter_info/compose.py` reads them into each `info.json`.
- `raw_downloads_manifest.json`: every raw download (path, source URL, SHA-256, bytes, retrieval
  time). The files themselves are archived on NJM under
  `source_evidence/filters/2026-10-25-expansion/` (not served).
- `renamed_filters_20261025.csv`: the band renames of 25 Oct 2026. `filter_info/compose.py` reads
  it to match a renamed band to its SVO record by its old name.
- `renamed_filters_20261026_bok.csv`, `renamed_filters_20261026_cassini.csv`,
  `renamed_filters_20261026_euclid.csv`, `renamed_filters_20261102_njm_dedupe.csv`: the renames
  that the curated notes of BOK/90prime, Cassini/ISS_NAC, Cassini/ISS_WAC and Euclid/NISP cite.
- `svo_aliases.json`: SVO filter IDs that NJM holds under another path (SVO ID -> NJM
  `Facility/Instrument/band`, with the basis of each). `svc/build_catalog.py` and
  `filter_info/svo_instruments.py` read it. The map is frozen; its builder is kept at the tag.
- `svc/`: the SVO-compatible Filter Profile Service (`fps.php`, `getdata.php`, `_lib.php`,
  `build_catalog.py`), copied to `data/svo_fps/` and served at
  https://data.nialljmiller.com/sed_tools/svo_fps/fps.php. See `svc/README.md` (astroquery
  override `SvoFps.SVO_MAIN_URL`, catalogue rebuild). `svc/tests/astroquery_live.py` is the live
  astroquery check of 26 Oct 2026, with its output beside it.
