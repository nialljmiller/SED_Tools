# Developing

[CONTRIBUTING.md](https://github.com/nialljmiller/SED_Tools/blob/main/CONTRIBUTING.md)
says how to report a bug and open a pull request, and describes the package
layout (`api/`, `cli/`, `filters/`, `component_sources/`, `grids/`, `cube/`,
`ml/` and the top-level modules).

## Setting up

```bash
git clone https://github.com/nialljmiller/SED_Tools.git
cd SED_Tools
pip install -e ".[full]" "pytest>=9"
```

The tests use pytest's built-in subtests, hence pytest 9 or newer.

## Running the tests

```bash
python -m pytest tests/
python -m pytest tests/test_filters.py -q
```

The tests run offline: the files they need are in `tests/fixtures/`, and
`tests/fixtures/builders/README.md` says where each comes from. Most tests are
table-driven: one test function runs many rows, and a new case is a new row in
an existing table.

GitHub Actions runs the suite on Python 3.10 and 3.12 for every push to
`main` and every pull request
([`.github/workflows/tests.yml`](https://github.com/nialljmiller/SED_Tools/blob/main/.github/workflows/tests.yml)).

## Golden tests

`tests/test_golden.py` reruns every case in `tests/golden/cases.py` on fixed
inputs and requires the outputs to match `tests/golden/expected/` byte for
byte, so any change that alters a result, even in the last digit, fails. The
outputs depend on the platform and on the numpy and scipy versions recorded in
`expected/environment.json`; on any other environment the cases skip and say
why. A failure means the code now gives a different answer: find out why, and
do not edit `expected/`. Re-recording is the maintainer's decision.
[tests/golden/README.md](https://github.com/nialljmiller/SED_Tools/blob/main/tests/golden/README.md)
has the details.

## Filter labels

`python -m sed_tools.filters.label_check FILTERS_DIR [--against OLD_DIR]`
checks the label rules every served filter tree must keep
([Filters](filters.md)). The NJM mirror runs it on every refresh.

## The documentation

These pages are plain Markdown under `docs/`, one page per topic, linked to
each other by relative links. The README links to them with absolute GitHub
URLs, because PyPI shows the README and cannot follow relative links. Every
`bash` and `python` block is meant to run as written, under a fresh `HOME`,
in page order.

`docs/source/` holds a Sphinx project with the API reference (autodoc) and the
changelog:

```bash
pip install sphinx
sphinx-build -b html docs/source docs/build/html
```

## Releases

`scripts/bump_version.py major|minor|patch` sets the version in
`pyproject.toml`, `sed_tools/__init__.py` and `CITATION.cff`, dates the
changelog's heading, and turns the README's version DOI line into a
placeholder until Zenodo archives the release. `scripts/bump_version.py
--check` confirms the versions agree; the publish workflow runs it.
