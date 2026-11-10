#!/bin/bash
# Per-test (and per-row) line coverage of sed_tools, as the trim measured it: ctrace core, subprocesses
# patched, one context per test phase and per subtest row.
# usage: run_cov.sh SRC OUTDIR [nodeid-file] [extra pytest args...]   then: lines.py SRC OUTDIR/.coverage OUT.json
# Environment: PYTEST_PATH (prepended to PYTHONPATH, e.g. the pytest 9.2 lib).
set -u
TOOLS=$(cd "$(dirname "$0")" && pwd)
SRC=$1; OUT=$2; shift 2
NODES=""
if [ $# -gt 0 ] && [ -f "$1" ]; then NODES=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift; fi
rm -rf "$OUT"; mkdir -p "$OUT/home"
cat > "$OUT/coveragerc" <<RC
[run]
source =
    sed_tools
parallel = true
core = ctrace
patch = subprocess
data_file = $OUT/.coverage
RC
cd "$SRC"
ARGS=()
if [ -n "$NODES" ]; then while IFS= read -r l; do [ -n "$l" ] && ARGS+=("$l"); done < "$NODES"; else ARGS=(tests); fi
HOME="$OUT/home" COVERAGE_RCFILE="$OUT/coveragerc" \
  PYTHONPATH="${PYTEST_PATH:+$PYTEST_PATH:}$TOOLS:/Users/njm/SED_Tools-reports-20261026/chisel/scripts:/Users/njm/Library/Python/3.14/lib/python/site-packages" \
  PYTHONDONTWRITEBYTECODE=1 \
  /Users/njm/.claude/jobs/6c6fde46/tmp/golden-venv/bin/python -m coverage run --rcfile="$OUT/coveragerc" -m pytest -q \
  -p no:cacheprovider -p ctxplugin -p rowctx --basetemp="$OUT/pt" \
  --deselect tests/test_version.py::test_package_version_matches_distribution_metadata "$@" "${ARGS[@]}" > "$OUT/pytest.log" 2>&1
echo "pytest exit $?"
tail -1 "$OUT/pytest.log"
HOME="$OUT/home" /Users/njm/.claude/jobs/6c6fde46/tmp/golden-venv/bin/python -m coverage combine --rcfile="$OUT/coveragerc" -q
