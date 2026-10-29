# Contributing to SED_Tools

Thank you for your interest in contributing to SED_Tools.

## Reporting Bugs & Requesting Features

Open an issue on GitHub. For bugs, include:

- A clear description of the problem
- Steps to reproduce it
- What you expected vs. what happened
- Relevant environment details (OS, Python version, relevant file sizes/grid names if applicable)

For feature requests, describe the use case and why it would be useful.

## Contribution Workflow

1. Fork the repository
2. Create a branch for your change
3. Make your changes
4. Open a pull request against `main` with a clear description of what you changed and why

Keep PRs focused — one logical change per PR. Large or sweeping changes should be discussed in an issue first.

## Code Style

No strict requirements. Match the style of the surrounding code you are editing.

## Package layout

The code in `sed_tools/` is grouped by the data it handles:

- `api/`: the Python API (`SED`, `SpectralGrid`, `Spectrum`, `Filters` and the rest).
- `cli/`: the `sed-tools` command and its menus.
- `filters/`: filter curves: the `.dat` format, where a set comes from (NJM, the original sources, SVO), instrument folders and composing their components.
- `component_sources/`: builders that make instruments' curves and components from their teams' own files.
- `grids/`: spectral grids: the NJM, SVO, MSG and MAST download sources, cleaning and unit conversion, lookup tables and coverage.
- `cube/`: flux cubes: building, checking and reading `flux_cube.bin`, and combining, densifying and exporting grids for MESA.
- `ml/`: the ML SED completer and generator.
- Top-level modules: settings (`config.py`), photometry on curves and spectra (`photometry.py`), the NJM mirror (`njm_mirror.py`), and prompts and terminal plots (`ui_utils.py`, `terminal_plots.py`).

The old module paths (`sed_tools.filter_library`, `sed_tools.spectra_cleaner` and the rest) still work: each forwards to the module that now holds its code. New code inside `sed_tools/` should import the new paths.

## What to Contribute

Areas where contributions are most useful:

- Bug fixes with a clear root cause
- New stellar spectral grid support
- Performance improvements to processing or I/O (especially at large data scales)
- Documentation corrections

## What Not to Do

- Do not open PRs that add unrelated changes bundled together
- Do not submit changes that have not been tested against real data
- Do not modify the unit standardization logic in `sed_tools/grids/units.py` (old path `spectra_cleaner`) without opening an issue and discussing it first — it is the single authority for unit conversion across the entire pipeline

## Contact

For questions not suited to a public issue: **niall.j.miller@gmail.com**
