#!/usr/bin/env bash
# New-files-only deploy of phase3/ia staged instruments to NJM (Cirrus).
# Usage: deploy_ia.sh Facility/Instrument ...   (reads phase3/ia/stage)
# New instruments: the remote dir must not exist. Existing instruments: every staged .dat must be
# absent remotely; the index file is the only file replaced, after a backup to backups/phase3-ia-<ts>/.
set -euo pipefail
IA=$(cd "$(dirname "$0")/.." && pwd)
RSH="ssh -q -F $HOME/.ssh/config.agent-network"; SSH="$RSH cirrus"
REPO=/media/data3/MESA/SED_Tools
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}; BK=$REPO/backups/phase3-ia-$TS
EV=$REPO/source_evidence/filters/2026-10-25-svo-archive
LOG=$IA/logs/deploy-$TS.log; mkdir -p "$IA/logs"
exec > >(tee -a "$LOG") 2>&1
echo "deploy $TS: $# instruments"
$SSH "mkdir -p '$BK' '$EV/provenance'"
for fi in "$@"; do
  f=${fi%%/*}; i=${fi#*/}; L="$IA/stage/filters/$fi"
  [ -d "$L" ] || { echo "not staged: $fi"; exit 1; }
  if [ -f "$L/manifest.json" ]; then           # new instrument
    if $SSH "test -e '$REPO/data/filters/$fi'"; then echo "SKIP new instrument already on NJM: $fi"; continue; fi
    $SSH "mkdir -p '$REPO/data/filters/$f'"
    rsync -a --ignore-existing -e "$RSH" "$L/" "cirrus:$REPO/data/filters/$fi/"
  else                                          # existing instrument: new bands + index
    names=$(cd "$L" && ls *.dat | tr '\n' ' ')
    clash=$($SSH "cd '$REPO/data/filters/$fi' && for n in $names; do [ -e \"\$n\" ] && echo \$n; done; true")
    [ -z "$clash" ] || { echo "SKIP $fi: already exists remotely: $clash"; continue; }
    $SSH "mkdir -p '$BK/$f/$i' && cp -p '$REPO/data/filters/$fi/$i' '$BK/$f/$i/$i'"
    rsync -a --ignore-existing -e "$RSH" --include='*.dat' --exclude='*' "$L/" "cirrus:$REPO/data/filters/$fi/"
    rsync -a -e "$RSH" "$L/$i" "cirrus:$REPO/data/filters/$fi/$i"
  fi
  rsync -a --ignore-existing -e "$RSH" "$IA/stage/provenance/${f}__${i}.json" "cirrus:$EV/provenance/"
  echo "deployed $fi"
done
$SSH "cp -p '$REPO/data/index.json' '$BK/index.json' && cd '$REPO' && python3 migrate_filters.py --dry-run | tail -5 && python3 generate_index.py data | tail -3"
echo "backup: $BK"
