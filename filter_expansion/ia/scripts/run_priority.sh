#!/usr/bin/env bash
# Room request ddf8e20d: download cdx/chunks/* in order (NJM-served IDs first, VOTables then browse
# pages, then the rest), publishing each chunk to NJM as one batch. One crawler at a time, 4.3 s pace.
set -u
IA=/home/nill/SED_Workspace/filter-expansion/phase3/ia
PY=/home/nill/SED_Workspace/filter-expansion/.venv/bin/python
cd /tmp
for q in "$IA"/cdx/chunks/*.json; do
  b=$(basename "$q" .json)
  name="ia-2026-10-26-b$b"; name=${name//_/-}
  [ -d "$IA/batches/$name" ] && continue
  echo "$(date -u +%FT%TZ) start $b" >> "$IA/run_priority.log"
  case $b in *votable*) s=download.py ;; *) s=download_browse.py ;; esac
  nice -n 10 "$PY" -I "$IA/scripts/$s" "$q" 4.3 >> "$IA/run_priority.log" 2>&1
  bash "$IA/scripts/publish_batch.sh" "$name"
done
echo "$(date -u +%FT%TZ) all chunks done" >> "$IA/run_priority.log"
