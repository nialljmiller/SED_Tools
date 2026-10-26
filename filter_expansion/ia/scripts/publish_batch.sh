#!/usr/bin/env bash
# Publish one IA batch to NJM (room request ddf8e20d): archive raw + manifest under
# source_evidence/svo-archive/<NAME>/, list it in filter_info/svo_manifests.txt, commit, run
# filter_info/refresh.sh (build, generate, resolve if unresolved, generate), commit references.json, push.
set -u
NAME=$1
IA=/home/nill/SED_Workspace/filter-expansion/phase3/ia
SSH="ssh -q -F /home/nill/.ssh/config.agent-network"
REPO=/media/data3/MESA/SED_Tools
LOG=$IA/publish.log
log() { echo "$(date -u +%FT%TZ) $NAME $*" >> "$LOG"; }
cd /tmp
n=$(/usr/bin/python3 -I "$IA/scripts/export_batch.py" "$NAME"); rc=$?
if [ $rc -eq 3 ]; then log "no new records"; exit 0; fi
[ $rc -eq 0 ] || { log "export failed rc=$rc"; exit 1; }
log "exported $n records"
rsync -a -e "$SSH" "$IA/batches/$NAME/" "cirrus:$REPO/source_evidence/svo-archive/$NAME/" || { log "rsync failed"; exit 1; }
$SSH cirrus bash -s "$NAME" "$n" >> "$LOG" 2>&1 <<'R'
set -u
NAME=$1; N=$2; cd /media/data3/MESA/SED_Tools
D=source_evidence/svo-archive/$NAME
python3 -I - "$D" <<'P' || { echo "SHA-256 check failed for $D"; exit 1; }
import json, hashlib, sys, pathlib
d = pathlib.Path(sys.argv[1]); bad = 0; n = 0
for l in open(d / "manifest.jsonl"):
    r = json.loads(l)
    if not r.get("file"): continue
    n += 1; bad += hashlib.sha256((d / r["file"]).read_bytes()).hexdigest() != r["sha256"]
print(f"{d}: {n} files, {bad} SHA-256 mismatches"); sys.exit(1 if bad else 0)
P
exec 9>/tmp/sed-filter-info.lock; flock 9
M=$D/manifest.jsonl
grep -qxF "$M" filter_info/svo_manifests.txt || echo "$M" >> filter_info/svo_manifests.txt
git -c "user.name=Niall Miller" -c user.email=niall.j.miller@gmail.com commit -q -m "List SVO archive batch $NAME" -m "$N Internet Archive captures of SVO Filter Profile Service records (VOTables and browse pages), archived raw with SHA-256 in $D." -- filter_info/svo_manifests.txt && echo "committed manifest list"
bash filter_info/refresh.sh 2>&1 | tail -40
if ! git diff --quiet -- filter_info/references.json; then
  git -c "user.name=Niall Miller" -c user.email=niall.j.miller@gmail.com commit -q -m "Resolve references from SVO archive batch $NAME" -- filter_info/references.json && echo "committed references.json"
fi
echo "unresolved: $(wc -l < filter_info/unresolved.txt)"
git push -q origin HEAD:server 2>&1 && echo pushed || echo "PUSH FAILED"
R
rc=$?; log "remote rc=$rc"; exit $rc
