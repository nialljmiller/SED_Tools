# Filter-expansion tooling (Claude, request 34dfcd2c38629862ce9f74c6)

Baseline: SED_Tools 0.4.1 (e5464af), detached worktree `../SED_Tools`, venv `../.venv`
(`.venv/bin/python -m pytest` in SED_Tools: 476 passed, 4 xfailed on 2026-10-25).
Brief: agent-share/projects/sed-tools-work/filter-expansion/BRIEF.md.

- `fxlib.py`: fetch (downloads/<Facility>/, HTML rejected), unit conversion to Å
  (Å/nm/µm/mm/GHz/Hz/cm⁻¹), `clean_curve` (sort, merge duplicates, negatives -> 0,
  optional peak normalisation; every change reported), `make_band`, `build_instrument`
  (curves via `write_filter_file`, index file, `manifest.json`, `components/<role>/`,
  `stage/provenance/<Facility>__<Instrument>.json` with raw-file SHA-256). A re-run moves
  the earlier build to `stage/_superseded/`.
- `recipes/<block>_<name>.py`: one `run()` per source; `build.py NAME ...` runs them.
- `fx_validate.py [ROOT] [--svo] [--json OUT]`: the format-contract checks plus collision
  with `mirror_index.json` and pivot checks (published `ref_pivot`, or SVO with `--svo`).

Run: `.venv/bin/python tools/build.py b_hipparcos && .venv/bin/python tools/fx_validate.py`
Deploy tree: `stage/filters/` only (`_superseded/`, `provenance/` stay local; provenance is
copied to the server under `source_evidence/filter-expansion/`).
