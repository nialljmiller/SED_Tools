#!/usr/bin/env bash
# Deploy staged filter instruments to the NJM mirror on Cirrus (BRIEF.md "Deploying to NJM").
# Usage: tools/deploy.sh Facility/Instrument [Facility/Instrument ...]
# Adds new instrument directories only; never overwrites or deletes. Backs up every
# affected existing facility and index.json first. Log: logs/deploy-<ts>.log
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
RSH="ssh -F $HOME/.ssh/config.agent-network"
SSH="$RSH cirrus"
REPO=/media/data3/MESA/SED_Tools
TS=$(date -u +%Y%m%dT%H%M%SZ)
BK=$REPO/backups/filter-expansion-$TS
LOG=$HERE/logs/deploy-$TS.log
[ $# -gt 0 ] || { echo "usage: $0 Facility/Instrument ..."; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "deploy $TS: $*"

# Local gate: everything requested must validate.
"$HERE/.venv/bin/python" "$HERE/tools/fx_validate.py" --only "$@" --json "$HERE/logs/validate-$TS.json"

# Remote preconditions: no target instrument may exist yet.
for fi in "$@"; do
  [ -d "$HERE/stage/filters/$fi" ] || { echo "not staged: $fi"; exit 1; }
  if $SSH "test -e '$REPO/data/filters/$fi'" 2>/dev/null; then echo "already on mirror: $fi"; exit 1; fi
done

# Backups: index.json and each affected facility that already exists.
facs=$(printf '%s\n' "$@" | cut -d/ -f1 | sort -u)
$SSH "mkdir -p '$BK' && cp -p '$REPO/data/index.json' '$BK/index.json'" 2>/dev/null
for f in $facs; do
  $SSH "if [ -d '$REPO/data/filters/$f' ]; then tar -C '$REPO/data/filters' -czf '$BK/$f.tar.gz' '$f' && echo 'backed up $f'; else echo 'new facility $f'; fi" 2>/dev/null
done

# Copy: new instrument directories (and their provenance) only.
for fi in "$@"; do
  f=${fi%%/*}; i=${fi#*/}
  $SSH "mkdir -p '$REPO/data/filters/$f' '$REPO/source_evidence/filter-expansion'" 2>/dev/null
  rsync -a --ignore-existing -e "$RSH" "$HERE/stage/filters/$fi/" "cirrus:$REPO/data/filters/$fi/" 2>/dev/null
  rsync -a --ignore-existing -e "$RSH" "$HERE/stage/provenance/${f}__${i}.json" "cirrus:$REPO/source_evidence/filter-expansion/" 2>/dev/null
done

# Server checks and index.
$SSH "cd '$REPO' && python3 migrate_filters.py --dry-run && python3 generate_index.py data | tail -3" 2>/dev/null
echo "backup: $BK"
