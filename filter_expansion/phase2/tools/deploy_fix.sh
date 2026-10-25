#!/usr/bin/env bash
# Phase 2: replace corrected instrument dirs on NJM (BRIEF-2.md step 2).
# Usage: phase2/fix/deploy_fix.sh Facility/Instrument ...
# Inputs: stage/filters/<F>/<I>/ (complete corrected dir), changes/<F>__<I>.json
# (list of {path, raw_copy, operation, raw_sha256, curated_sha256[, renamed_from]}),
# optional stage/provenance/<F>__<I>.json.
# Backs up each dir (tar) + index.json; copies every old file to source_evidence;
# moves renamed old files into the backup dir (never deletes); rsyncs the stage over;
# then audit, migrate --dry-run, generate_index and HTTPS check.
set -euo pipefail
FIX=$(cd "$(dirname "$0")" && pwd); TOP=$(cd "$FIX/../.." && pwd)
RSH="ssh -F $HOME/.ssh/config.agent-network"; SSH="$RSH cirrus"
REPO=/media/data3/MESA/SED_Tools; EV=source_evidence/filters/2026-10-25-fixes
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$REPO/backups/phase2-fix-$TS; LOG=$FIX/logs/deploy-$TS.log
mkdir -p "$FIX/logs"; exec > >(tee -a "$LOG") 2>&1
[ $# -gt 0 ] || { echo "usage: $0 F/I ..."; exit 2; }
echo "deploy-fix $TS: $*"
"$TOP/.venv/bin/python" "$FIX/gate.py" "$@"
for fi in "$@"; do
  [ -d "$FIX/stage/filters/$fi" ] && [ -f "$FIX/changes/${fi/\//__}.json" ] || { echo "missing stage/changes: $fi"; exit 1; }
  $SSH "test -d '$REPO/data/filters/$fi'" 2>/dev/null || { echo "not on mirror: $fi"; exit 1; }
done
$SSH "mkdir -p '$BK' && cp -p '$REPO/data/index.json' '$BK/index.json'" 2>/dev/null
for fi in "$@"; do
  key=${fi/\//__}
  $SSH "cd '$REPO/data/filters' && mkdir -p '$BK/$(dirname "$fi")' && tar -czf '$BK/$key.tar.gz' '$fi' && mkdir -p '$REPO/$EV/$fi' && cp -an '$fi/.' '$REPO/$EV/$fi/'" 2>/dev/null
  # Renamed files: move the old name out of the served tree into the backup dir.
  "$TOP/.venv/bin/python" -I -c 'import json,sys; [print(e["renamed_from"]) for e in json.load(open(sys.argv[1])) if e.get("renamed_from")]' "$FIX/changes/$key.json" | while read -r old; do
    $SSH "mkdir -p '$BK/renamed/$(dirname "$old")' && mv -n '$REPO/$old' '$BK/renamed/$old'" </dev/null 2>/dev/null && echo "moved aside $old"
  done
  rsync -a --checksum -e "$RSH" "$FIX/stage/filters/$fi/" "cirrus:$REPO/data/filters/$fi/" 2>/dev/null
  [ -f "$FIX/stage/provenance/$key.json" ] && rsync -a -e "$RSH" "$FIX/stage/provenance/$key.json" "cirrus:$REPO/source_evidence/filter-expansion/phase2-$key.json" 2>/dev/null || true
  rsync -a -e "$RSH" "$FIX/changes/$key.json" "cirrus:$REPO/$EV/changes-$key-$TS.json" 2>/dev/null
done
$SSH "cd '$REPO' && python3 scripts/audit_filter_curves.py data/filters --output '$BK/audit_after.csv' | tail -2; echo audit rows: \$(( \$(wc -l < '$BK/audit_after.csv') - 1 )); python3 migrate_filters.py --dry-run | tail -3 && python3 generate_index.py data | tail -2" 2>/dev/null
"$TOP/.venv/bin/python" "$FIX/verify_fix.py" "$@"
echo "backup: $BK"
