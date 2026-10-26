#!/usr/bin/env bash
# Rebuild every info.json from the archived SVO harvests and the committed inputs.
# Run on NJM from the server checkout. The network is used only when new references turn up.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m filter_info.svo_archive build source_evidence/svo-archive/parsed $(grep -v '^#' filter_info/svo_manifests.txt)
python3 generate_index.py data
if [ -s filter_info/unresolved.txt ]; then
  python3 -m filter_info.references resolve
  python3 generate_index.py data
  echo "filter_info/references.json gained entries: commit it (author Niall Miller)."
fi
