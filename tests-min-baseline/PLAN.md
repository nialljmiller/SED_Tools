# Plan: SED_Tools' tests as 9 table-driven tests (10 Nov 2026)

**The plain picture.** Each of the 9 areas becomes one test function, `test_<area>(rows)`. It walks
tables of named rows, and pytest's built-in `subtests` reports each row by name. Four packages convert
8 of the areas in parallel; packaging and version is already done, as the worked example. Each package
proves its own areas against the baseline, line for line and id for id, before it hands back.

- **Branch:** `tests/minimal`, worktree `/Users/njm/.claude/jobs/6c6fde46/tmp/agents/tests-min`, off `origin/declutter` (f0103c76).
- **Brief:** `BRIEF.md`, next to this file.
- **Harness:** `tests/support.py` (the row runner and every shared fake), `tests/conftest.py` (the `rows` fixture).
- **Worked example:** `tests/test_packaging.py`. It replaces `test_version.py` and `test_api_dependencies.py`, which are deleted.
- **Baseline and tools:** `baseline/` and `tools/`, next to this file. A gzipped copy of `baseline/` is committed as `tests-min-baseline/` at the repo top. Remove it with one `git rm -r` before any PR.

## 1. Baseline (old suite, f0103c76)

| | |
|---|---|
| test ids | 951 (`baseline/ids.txt`), from 740 test functions in 83 files, 16,130 lines in `tests/*.py` |
| 0.4.1 or fixed-bug checks | 537 ids: 94 always kept, 443 0.4.1/bug (53 of them from declutter fixes); 414 ids are kept for lines only (`baseline/checks.tsv`) |
| run time, pytest 9.1.1 | 951 passed in 22.8 s (wall 23.6 s). It varies from 23 to 31 s with this machine's load, so compare runs made back to back. The old tree and this branch, timed alternately: 25.2 vs 24.8 s, then 30.6 vs 30.9 s |
| package lines run, all contexts | 17,984 (17,982 without the 2 nondeterministic lines) |
| package lines run inside tests | 16,401 |
| package lines run only at collection | 1,583 |

**How it was measured.** `tools/run_cov.sh` uses the trim's setup: the `ctrace` core, `patch = subprocess`,
one coverage context per test phase (`ctxplugin`), a scratch `HOME`, and the trim's `--deselect` of the
distribution-metadata test. It adds `-p noconnect`, which blocks the network at the socket. `tools/lines.py`
turns `.coverage` into `{all, per-context}` JSON.

**Baseline files.**

| file | what it holds |
|---|---|
| `lines.json.gz` | per-context lines, network blocked. This is the reference. |
| `lines.txt` | the sorted `file:line` list |
| `lines_network_on.json.gz` | the same run with the network up, the trim's way |
| `ids.txt` | the 951 old ids |
| `checks.tsv` | each id's kind (always, 0.4.1/bug, lines) and why |
| `asserts.json.gz` | every old test function's asserts, raises and warns, normalised by `trim-work/checks.py` |
| `collected_modules.txt` | the 111 `sed_tools` modules imported by the end of collection |
| `declutter_touched.json` | the tests each declutter commit added or changed |

**Why the reference blocks the network.**
- One old test reaches the real network: `test_filters_njm_down.py::test_svo_that_does_not_answer_is_waited_for_once_per_process` asks NJM for real.
- With the network up it misses 4 lines of `njm_mirror.plain_reason` (64, 65, 67, 69). With the network down it runs them. Nothing is lost either way.
- No other old test reaches a socket. No test trips any of the old network guards: both were checked with an instrumented copy.

**Nondeterministic lines** (`tools/prove.py` and `tools/area_check.py` leave them out):
- `sed_tools/ml/common.py:261` and `:266`: optuna's unseeded sampler in the auto-train test.
- At the per-test level only, `ml/common.py:60` in `test_auto_trainer_stops_on_a_trial_with_no_usable_epoch`. The union is stable.

Four baseline runs (two with the network up, two with it blocked) gave identical sets, apart from these lines.

**Which ids are checks** (`tools/checks_list.py`):
- the trim's `kept_because` for each id (`TEST_TRIM_ids.tsv`), wherever it is not `lines`;
- plus every test that a declutter fix commit (277f358..origin/declutter) added or changed. These match DECISIONS.md items B1–B5, C, W1, W3, W4 and c, and the earlier `svo_*`, ML-batch and radius fixes;
- plus the new `svo_*` cases those commits added to `test_cli_mains.py`'s tables.

The detector-edit feature commits (1e459b7b, 4d581f88) are not counted as fixes. Their 14 new tests are `lines`.

## 2. The harness

### `rows`
Every area test takes the `rows` fixture and nothing else from pytest:

```python
TABLE = {"row_name": data, ...}        # check(t, *data) when data is a tuple, else check(t, data)
SCENARIOS = {"row_name": function}     # function(t): a short scenario

def test_area(rows):
    rows(TABLE, check)
    rows(SCENARIOS)
```

What `rows` does:
- **Reporting.** Each row is `subtests.test(name)`. A failing row is reported as `SUBFAILED[name]` and the next row still runs. `-v` lists `SUBPASSED[name]` for each row.
- **Checked under pytest 9.1.1 and 9.2.0.** The two versions behave identically for:
  - a failing row;
  - an exception raised in a row;
  - a `BaseException` raised in a row;
  - a skipped row;
  - the exit status.

  CI's `pip install pytest` installs 9.2.0 today. pytest 9 supports Python 3.10.
- **Self-check.** `tools/selfcheck_rows.py`: copy it into `tests/` and run it. One row fails on purpose and one skips. Every other check passes:
  - output, logs, environment, attributes and the working directory don't leak between rows;
  - tmp folders and config folders are distinct;
  - random, numpy, torch and MPS draws are identical in each row;
  - the network guard fires;
  - a duplicate row name is refused.

### Isolation
Each row gets a `support.Row` called `t`, with:

| field | what it is |
|---|---|
| `t.tmp` | a fresh folder |
| `t.mp` | a `MonkeyPatch.context()`, undone after the row (chdir, env, setattr, setitem) |
| `t.capsys` | cleared before the row |
| `t.caplog` | cleared before the row |
| `t.stack` | an `ExitStack` closed after the row |
| `t.name` | the row's name |

Before each row, what the old autouse fixtures did per test is redone:
- a private config folder (`config._CONFIG_DIR` and `_CONFIG_FILE`);
- NJM's and SVO's `_STATE` reset, before and after;
- the random state saved and restored: `random`, numpy, and torch with MPS once torch is imported. Every row therefore starts from the same random state;
- the network guard. `requests.Session.send` raises `support.NetworkUsed`, a `BaseException`, so no `except Exception` can swallow it. The package uses only `requests`.

### `support.py` parts
**Core (frozen):**
- `Rows`, `Row` and `NetworkUsed`;
- `answer(mp, *replies, echo=False)`, which returns the prompts;
- `narrow_terminal(t)` and `wide_lines`;
- `run_cli(mp, *argv)` and `run_module(mp, module, *argv)` (`python -m` in process);
- `Resp`;
- `Session(files, base="", **resp)`: URL → body, status or exception. `calls` lists what was asked.

**The other parts:**

| part | owner | what is in it |
|---|---|---|
| cubes and grids | C | `GRID`, `GOLDEN_MODELS`, `KURUCZ`, `NJM`, `write_flux_cube`, `AffineCube`, `affine_cube`, `GridServer`, `njm_grid`, `njm_grids`, `digests`, `copy_without_times`, `write_lookup`, `flat_filter` |
| filters | A | `CURVE`, `njm_index`, `write_set`, `njm_up`, `njm_filters`, `NJM_DOWN`, `bundled`, `GAIA`, `svo_stub`, `svo_browser`, `atmosphere_site`, `STALE` |
| builders | B | `FIX`, `WL`, `SWIFT_GRID`, `saved`, `curve`, `check_published`, `tar_gz`, `gauss`, `published`, `published_set`, `text_table`, `ipac_page`, `twomass_files`, `lsst_files`, `fits_file`, `fits_table`, `uvot_files`, `pds_table`, `cassini_files`, `derived`, `team_site`, `skytable`, `SkyCalcForm` |
| ML | B | `cpu_torch(t)`, `tiny_library(t)`, `fixed_completer`, `fixed_generator` |
| CLI and API | D | `GRABBERS`, `fake_grabber`, `filters_api(t)`, `CLEAN`, `tagged`, `Stdin`, `availability`, `summary_inputs`, `GAIA_INSTRUMENTS`, `gaia(t)`, `catalogue(t)` |

**Old names to support names.** Fixtures became factories that take the row.

| old | now |
|---|---|
| fixture `filters` (helpers_api) | `filters_api(t)` |
| fixture `bundled` (helpers_filters) | `bundled(t, catalogue)` |
| fixture `njm` (helpers_filters) | `njm_filters(t, root, files)`; `njm_up(t)` for the patches alone |
| fixture `njm` (helpers_grids) | `njm_grids(t)` |
| fixture `njm_grid` | `njm_grid(t)`, built once per session |
| fixture `gaia` | `gaia(t)` |
| fixture `catalogue` | `catalogue(t)` |
| fixture `cpu_torch` | `cpu_torch(t)` |
| fixture `tiny_library` | `tiny_library(t)` |
| fixture `cube_writer` | `write_flux_cube` |
| fixture `make_affine_cube` | `affine_cube(t.tmp, ...)` |
| fixture `narrow_terminal` | `narrow_terminal(t)` |
| fixture `keep_random_state` | built in (every row) |
| fixture `no_network` (helpers_api), `offline` network part (cli_mains), `no_network` (svo_interactive, api_untested) | built in (every row) |
| `Served(files)` | `Session(files, base=NJM)` |
| `FakeNJM(root)` | `GridServer(root)` |
| `.requested` | `.calls` |
| `Reply(status, body)`, builder `Resp`, `_Resp` (all variants) | `Resp(body, status, encoding=..., error=...)` |
| builder `Session`, `_Session` | `Session(files)`; `Session(saved())` for the saved pages |
| `answer` (helpers_cli, helpers_cli_menus, cli_mains, svo_interactive), `_feed_input` (helpers_cube; ml_engines with echo) | `answer(mp, *replies, echo=...)` |
| `run_cli` (5 copies) | `run_cli(mp, *argv)` |
| `run_m` | `run_module(mp, module, *argv)` |
| `_wide_lines` | `wide_lines` |
| `_tar` | `tar_gz` |
| `_gauss` (ground) | `gauss(c, w)` |
| `_gauss` (swift) | `gauss(c, w, SWIFT_GRID)` |
| `_published` (cfht_lsst) | `published` |
| `_published` (ground) | `published_set` |
| `_table` (cfht_lsst) | `text_table` |
| `_table` (swift) | `fits_table` |
| `_hdul` | `fits_file` |
| `_pre` | `ipac_page` |
| `_pds` | `pds_table` |
| `_curve` (sites) | `derived` |
| `_site` | `team_site` |
| `_skytable` | `skytable` |
| `_FormSession` | `SkyCalcForm` |
| `_twomass_files`, `_lsst_files`, `_uvot_files`, `_cassini_files` | the same names without `_` |
| `_fixed_completer` | `fixed_completer` |
| `_fixed_generator` | `fixed_generator` |
| `_availability` | `availability` |
| `_summary_inputs` | `summary_inputs` |
| `_Stdin` | `Stdin` |

**Differences between the merged fakes.** Keep each one as a parameter where it matters:
- **What a 4xx or 5xx raises.** `Resp` raises `requests.HTTPError("<status> for <url>", response=self)`. The old builder `_Resp` in cfht_lsst and ground raised `RuntimeError`; pass `error=RuntimeError` there if a row or the coverage differs. `swift.py` and `skycalc.py` fall back only on a `RequestException`.
- **Text encoding.** `.text` decodes as utf-8 with `replace`. The old SkyCalc, swift and cassini fakes used latin-1; pass `encoding="latin-1"` for non-ASCII bytes.
- **Running out of answers.** `answer` raises `AssertionError` when the replies run out. The old `_feed_input` raised `StopIteration`.

## 3. The 9 areas

Each area has one test, `tests/test_<area>.py::test_<area>`, which replaces the old files below. Counts are pytest ids and file lines. "Fakes from" names the old helpers or conftest fixtures each file used; all are now in `support.py`.

### 1. filters (package A): `tests/test_filters.py::test_filters`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_filter_api.py | 5 | 3 | 89 | - |
| test_filter_bulk_selection.py | 4 | 2 | 57 | - |
| test_filter_catalogue.py | 10 | 1 | 356 | - |
| test_filter_colors.py | 1 | 0 | 24 | - |
| test_filter_components.py | 9 | 0 | 182 | - |
| test_filter_detector_edit.py | 15 | 1 | 297 | helpers_cli_menus |
| test_filter_detector_type.py | 7 | 0 | 150 | - |
| test_filter_library.py | 14 | 1 | 279 | - (autouse `offline`: NJM down) |
| test_filter_properties.py | 6 | 0 | 103 | - |
| test_filters_atmosphere_sites.py | 16 | 11 | 271 | helpers_builders_sites |
| test_filters_catalogue_gaps.py | 1 | 0 | 60 | helpers_api |
| test_filters_component_variants.py | 5 | 1 | 97 | helpers_builders_ground |
| test_filters_failures.py | 25 | 14 | 525 | - (autouse: socket guard, `SVO_FPS_URL_ENV` unset) |
| test_filters_njm_down.py | 28 | 11 | 509 | helpers_filters, helpers_grids |
| test_filters_sources.py | 6 | 0 | 164 | helpers_filters |
| test_filters_spelling_and_archives.py | 35 | 28 | 516 | helpers_filters |
| test_filters_svo_interactive.py | 5 | 0 | 107 | helpers_filters, helpers_random |
| test_svo_fail_fast.py | 2 | 0 | 58 | - (autouse: socket guard, `SVO_FPS_URL_ENV` unset) |
| test_mesa_legacy_format.py | 28 | 28 | 373 | - |
| test_wavelength_units.py | 12 | 12 | 60 | - |
| **total** | **234** | **113** | **4,277** | |

### 2. grids (package C): `tests/test_grids.py::test_grids`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_grids_cleaning.py | 17 | 9 | 333 | - |
| test_grids_flux_units.py | 45 | 45 | 388 | helpers_grids |
| test_grids_mast_msg_downloads.py | 1 | 0 | 100 | helpers_grids |
| test_grids_njm_outages.py | 3 | 1 | 61 | helpers_filters |
| test_grids_out_of_date.py | 8 | 8 | 117 | helpers_grids |
| test_grids_selection.py | 11 | 8 | 117 | helpers_grids, cube_writer |
| test_grids_source_failures.py | 38 | 4 | 679 | - (autouse guard at `HTTPAdapter.send`) |
| test_mast_spectra_grabber.py | 1 | 0 | 45 | - |
| test_mast_wave_grid_lookup.py | 1 | 1 | 22 | - |
| test_spectra_cleaner_units.py | 4 | 0 | 27 | - |
| test_spectra_cleaner_workflow.py | 3 | 1 | 74 | - |
| test_shared_utilities.py | 1 | 0 | 21 | - |
| **total** | **133** | **77** | **1,984** | |

### 3. flux cubes (package C): `tests/test_cubes.py::test_cubes`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_cube_binary_io.py | 3 | 0 | 55 | make_affine_cube |
| test_cube_build.py | 24 | 9 | 379 | helpers_grids, cube_writer |
| test_cube_combine.py | 6 | 6 | 111 | - |
| test_cube_densifier.py | 9 | 8 | 179 | helpers_cube, cube_writer (a module-scoped fixture of its own) |
| test_cube_menus.py | 28 | 20 | 457 | conftest, helpers_api, helpers_cli, helpers_cube, cube_writer, make_affine_cube |
| test_cube_photometry_specs.py | 4 | 4 | 76 | helpers_grids, cube_writer |
| test_cube_tool_prompts.py | 8 | 2 | 161 | helpers_random, cube_writer (autouse `searched_only_here`) |
| test_flux_cube_collisions.py | 2 | 0 | 68 | - |
| test_collision_config.py | 10 | 0 | 104 | - |
| test_interpolation_known_answers.py | 2 | 0 | 34 | make_affine_cube |
| test_mesa_prepare_variant_gap.py | 1 | 1 | 56 | - |
| test_photometry.py | 10 | 10 | 102 | helpers_api |
| test_photometry_formulas.py | 5 | 5 | 45 | helpers_grids |
| test_resample.py | 1 | 1 | 19 | - |
| test_terminal_plots.py | 2 | 1 | 38 | - |
| test_models.py | 1 | 0 | 61 | - |
| test_models_path_inputs.py | 2 | 0 | 46 | - |
| **total** | **118** | **67** | **1,991** | |

### 4. ML (package B): `tests/test_ml.py::test_ml`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_ml_api.py | 2 | 0 | 32 | - |
| test_ml_completer_data_prep.py | 2 | 1 | 30 | - |
| test_ml_engines_device_and_auto.py | 7 | 6 | 242 | cube_writer (MPS, optuna, `skipif`) |
| test_ml_engines.py | 33 | 26 | 528 | helpers_ml, cube_writer |
| test_ml_generator_data_prep.py | 1 | 0 | 27 | cube_writer |
| test_ml_ranges_and_menus.py | 10 | 10 | 262 | helpers_api, helpers_cube, cube_writer |
| **total** | **55** | **43** | **1,121** | |

### 5. builders, `component_sources` (package B): `tests/test_builders.py::test_builders`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_argus.py | 11 | 8 | 247 | - |
| test_builders_cfht_eris_2mass_lsst.py | 6 | 2 | 161 | helpers_builders_cfht_lsst |
| test_builders_grabbers.py | 10 | 3 | 93 | builder_fixtures, helpers_cli_menus |
| test_builders_ground.py | 6 | 2 | 202 | helpers_builders_ground |
| test_builders_no_silent_failures.py | 6 | 1 | 189 | - |
| test_builders_no_silent_skips.py | 9 | 8 | 229 | - |
| test_builders_observatories.py | 23 | 15 | 129 | builder_fixtures (autouse `xlrd` stand-in) |
| test_builders_space.py | 24 | 18 | 319 | builder_fixtures |
| test_builders_surveys.py | 12 | 10 | 77 | builder_fixtures (autouse LCO sha256) |
| test_builders_swift_csp_cassini.py | 6 | 3 | 180 | helpers_builders_swift_cassini |
| test_builders_teams.py | 24 | 2 | 384 | builder_fixtures, helpers_builders_sites |
| test_component_sources.py | 14 | 10 | 134 | - |
| test_original_sources.py | 20 | 3 | 339 | builder_fixtures |
| **total** | **171** | **85** | **2,683** | |

### 6. CLI and `python -m` entry points (package D): `tests/test_cli.py::test_cli`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_cli.py | 3 | 1 | 59 | - |
| test_cli_errors.py | 26 | 7 | 408 | helpers_api, helpers_cli |
| test_cli_mains.py | 39 | 39 | 331 | helpers_filters, helpers_grids, helpers_ml, helpers_random (autouse `offline`: chdir, `STELLAR_DIR_DEFAULT`) |
| test_cli_menus_and_options.py | 48 | 10 | 596 | helpers_cli_menus |
| test_cli_optional_packages.py | 24 | 21 | 182 | conftest, helpers_cli |
| test_ui_picker_widths.py | 3 | 3 | 50 | helpers_api, helpers_cli |
| test_ui_utils.py | 8 | 6 | 120 | helpers_cli_menus |
| **total** | **151** | **87** | **1,746** | |

The new `tests/test_cli.py` replaces the old file of the same name.

### 7. Public API, top level and the 0.4.1/0.5.x legacy surface (package D): `tests/test_api.py::test_api`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_api_errors.py | 22 | 3 | 321 | helpers_api |
| test_api_untested.py | 19 | 19 | 282 | helpers_api, helpers_filters, helpers_grids, helpers_ml, helpers_random, cube_writer |
| test_old_paths.py | 4 | 4 | 51 | - |
| test_programmatic_usage.py | 10 | 10 | 235 | - |
| test_config.py | 10 | 8 | 101 | helpers_cli |
| **total** | **65** | **44** | **990** | |

### 8. golden (package D): `tests/test_golden.py::test_golden`
| old file | tests | 0.4.1/bug | lines | fakes from |
|---|--:|--:|--:|---|
| test_golden.py | 17 | 17 | 66 | - (`tests/golden/cases.py`, `record.py`) |

`tests/golden/` stays as it is: `cases.py`, `record.py`, `expected/` and `inputs/` import nothing from the helpers or conftest. Check the recorded environment once, before the rows. If it differs, skip the whole test once (CI's one skip), not 16 rows.

### 9. Packaging and version (design step): `tests/test_packaging.py::test_packaging`. **Done.**
| old file | tests | 0.4.1/bug | lines | now |
|---|--:|--:|--:|---|
| test_version.py | 2 | 2 | 20 | rows `version_matches_pyproject`, `version_matches_distribution_metadata` |
| test_api_dependencies.py | 5 | 2 | 115 | 7 rows: 5 pyproject facts, 4 fresh-interpreter scripts (2 of them from the old optional-package loop) |
| **total** | **7** | **4** | **135** | **95 lines, 11 rows** |

The inventory is in `inventory/design.tsv`. It runs in 2.2 s, as the old 7 did.

## 4. Conversion rules (every package)

1. **One test per area:** `def test_<area>(rows)` in `tests/test_<area>.py`.
   - No `parametrize`, no other test function, and no fixture of your own.
   - The test takes only `rows`. Rows can't request fixtures; use the support factories that take `t`.
2. **Rows.**
   - A row is data plus at most a short check. Same-shape cases share one check and differ only in data.
   - A scenario that doesn't fit a table is a short function in a `{name: function}` table.
   - No copy-paste rows.
   - Run `ruff check --select F601`: a repeated literal key in a table dict silently drops a row.
3. **Row names.**
   - snake_case, unique within the area (`rows` refuses a duplicate), and saying what is checked.
   - For a 0.4.1 or bug check, keep the old name's words so it can still be found.
4. **Isolation.**
   - Use only `t.tmp`, `t.mp`, `t.capsys`, `t.caplog` and `t.stack`.
   - Log levels: `with t.caplog.at_level(...)`, or `t.stack.enter_context(t.caplog.at_level(...))`. Never `set_level`: it leaks to later rows.
   - A file-local autouse fixture becomes setup that only that file's rows get: one line in their check, or a helper in the area file. The tables in section 3 list them.
   - `narrow_terminal(t)` is for rows from files that imported helpers_cli_menus.
   - `cpu_torch(t)` is for rows that requested it.
5. **Network.** Every row refuses a real request with `NetworkUsed`.
   - The one old test that reached the network (section 1) must fake NJM unreachable. Raise `requests.ConnectionError` with a long message that contains "(", so `plain_reason` runs `njm_mirror.py` 64–69.
   - Rows from files whose old guard was an `AssertionError` (helpers_grids, test_grids_source_failures, the socket guards) get the stricter guard. No old test tripped a guard.
6. **Skips.**
   - `pytest.importorskip` or `pytest.skip` goes inside the row, so the other rows run.
   - Golden checks its environment once and skips the whole test.
   - Note every row that skips on Linux CI in the inventory: golden, MPS, optuna, and torch-less paths.
7. **Assertions.**
   - Every assert, `raises` and `warns` of an old 0.4.1/bug id (`baseline/checks.tsv`, `asserts.json.gz`) stays in its mapped row or rows, as the same expression. Names may change.
   - Never weaken a check: no `in` where `==` was, no dropped `match=`, no looser tolerance.
   - A lines-only id keeps its asserts too, unless another row already asserts the same fact. Say so in the note.
8. **Imports.**
   - A `sed_tools` import keeps its level. An old file's top-level import may be top-level in the area file; an import inside a test stays inside the row or its check.
   - A top-level import of a module that is not in `baseline/collected_modules.txt` moves lines out of the tests, and the proof fails.
9. **Order.**
   - Keep rows in the old collection order: the file order in `tools/areas.json`, then the order within each file.
   - Grouping rows into tables may reorder only rows that share no state. `test_packaging` groups its pyproject facts before its scripts.
   - Each row starts from the same random state. A row that used the state an earlier test left must seed itself, and its coverage must be re-checked. This applies to `test_train_complete_and_generate_run_on_mps`.
10. **Fakes.**
    - Use `support.py`. A fake only your area uses lives in your area file.
    - A new shared fake goes only in your package's part of `support.py`. Use other parts, but never edit them.
11. **Deleting the old files.** Delete each old file once its rows pass and the area check passes.
12. **Files not to touch:**
    - `sed_tools/`;
    - `pyproject.toml`;
    - `tests/conftest.py`;
    - the core part of `support.py`;
    - `helpers_*.py` and `builder_fixtures.py`;
    - `tests/fixtures/`;
    - `tests/golden/`;
    - another package's areas or `support.py` parts.

    The merge step deletes the helpers and prunes the fixtures.
13. **Style.**
    - Lines of 120 columns at most.
    - Plain docstrings and comments, saying what is checked.
    - No new dependencies and no warnings filters added to make a row pass.

## 5. Checking an area against the baseline

Run these in your package's worktree (`<wt>`), with your scratch folder (`<scratch>`):

```sh
T=/Users/njm/SED_Tools-reports-20261026/tests-min/tools
PY=/Users/njm/.claude/jobs/6c6fde46/tmp/golden-venv/bin/python
echo tests/test_<area>.py > <scratch>/nodes.txt
$T/run_cov.sh <wt> <scratch>/cov-<area> <scratch>/nodes.txt -p noconnect
$PY $T/lines.py <wt> <scratch>/cov-<area>/.coverage <scratch>/cov-<area>.json
$PY $T/area_check.py <area> <scratch>/cov-<area>.json ../inventory/<package>.tsv
```

`area_check.py` fails when any of these happen:
- **Lost.** A line the old area's tests ran isn't run by the new test, outside the two nondeterministic lines.
- **Moved to collection.** A line they ran inside a test now runs only at collection.
- **Inventory unmapped.** An old id of the area isn't in the inventory.
- **Row not run.** The inventory names a row that didn't run.

It also notes, without failing, each old id whose lines its own rows don't all run. Look at each one: the row may need that step back. `rowctx.py` gives each row its own coverage context, `<test>|row=<name>`, and subprocesses a row starts carry the label too.

**Before you hand back:**
- Each of your areas prints `AREA <area> OK`.
- The whole suite passes:
  - under pytest 9.1: `HOME=<scratch>/home PYTHONPATH=/Users/njm/Library/Python/3.14/lib/python/site-packages $PY -m pytest -q -p no:cacheprovider tests`;
  - under 9.2: the same, with `PYTHONPATH` starting with `/Users/njm/.claude/jobs/6c6fde46/tmp/pt92/lib:`.
- The whole-suite proof gives `PROOF OK`: `$T/run_cov.sh <wt> <scratch>/cov-all -p noconnect`, then `lines.py`, then `$PY $T/prove.py $T/../baseline/lines.json.gz <scratch>/cov-all.json`.
- `git diff origin/declutter -- sed_tools` is empty.

**Inventory.** One fragment per package: `inventory/<package>.tsv`, with the columns `old_id`, `new_test`, `row` and `note`. Write one line per old id and row. An id checked by three rows gets three lines. The note says:
- how a 0.4.1/bug id's asserts were kept;
- `skips on CI: <why>` where that applies.

`inventory/design.tsv` is the example.

## 6. The four packages

Each package works in its own worktree, on its own branch off the design commit:

```sh
git -C /Users/njm/SED_Tools worktree add -b tests/minimal-<p> /Users/njm/.claude/jobs/6c6fde46/tmp/agents/tests-min-<p> tests/minimal
```

Commit as Niall Miller, with no AI attribution, and don't push. The manager merges the four branches into `tests/minimal`. No two packages touch the same file, except their own parts of `support.py`. Each part is a contiguous block under its banner, so git merges them cleanly.

| package | areas | old ids | 0.4.1/bug | old lines | new files | `support.py` parts it may extend |
|---|---|--:|--:|--:|---|---|
| **A** | filters | 234 | 113 | 4,277 | `tests/test_filters.py` | filters |
| **B** | builders, ML | 226 | 128 | 3,804 | `tests/test_builders.py`, `tests/test_ml.py` | builders, ML |
| **C** | grids, flux cubes | 251 | 144 | 3,975 | `tests/test_grids.py`, `tests/test_cubes.py` | cubes and grids |
| **D** | CLI, API and legacy, golden | 233 | 148 | 2,802 | `tests/test_cli.py`, `tests/test_api.py`, `tests/test_golden.py` | CLI and API |

Total: 944 ids plus the 7 already converted, which makes 951.

**Shared reads to be careful with:**
- **A** uses the builders part (`team_site`, `derived`, `gauss`, `published_set`, `WL`) for the atmosphere-site and component-variant rows. It also uses the grids part (`KURUCZ`, `GridServer`) for the NJM-down rows.
- **D** uses every part. `test_cli_mains` and `test_api_untested` reach filters, grids and ML.
- If you need a change in another package's part, don't make it. Write the fake in your area file and report it, and the merge step moves it.

## 7. After the four packages (merge and proof step)

1. Merge A–D into `tests/minimal`.
2. Delete `helpers_*.py`, `builder_fixtures.py` and conftest's old-file fixtures, then rerun.
3. Prune `tests/fixtures/` files that nothing opens. Golden's `inputs/` and `expected/` stay.
4. Concatenate `inventory/*.tsv` into `inventory.tsv` and check every one of the 951 ids is there.
5. Full proof with `prove.py`, against `baseline/lines.json.gz`. For its Linux-CI line set, replace the old `CI_SKIPPED` ids with the rows the inventories mark `skips on CI`.
6. Mutation sample: about 60–100 mutants on lines the old suite covers. Every mutant the old suite kills, the new suite must kill.
7. Time the suite under 9.1 and 9.2 against the 22.8 s baseline.
8. `git rm -r tests-min-baseline` before the PR.
