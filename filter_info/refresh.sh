#!/usr/bin/env bash
# Rebuild every info.json from the archived SVO harvests and the committed inputs,
# then the SVO-compatible service catalogue.
# Run on NJM from the server checkout. The network is used only when new references turn up.
set -euo pipefail
cd "$(dirname "$0")/.."
# Every set's records must agree (sed_tools.filters.manifest.set_problems): its MESA index lists its
# curves, its manifest names it and declares no band outside the index (or SED_Tools 0.4.1 replaces
# its curves with SVO's), and no curve is of another set's instrument. Archive copies go under
# "archived_bands".
python3 -m filter_info.check_manifests data
# Every curve's labels keep SED_Tools' rules (sed_tools.filters.label_check): detector types and their
# sources in SED_Tools' words, components of known roles, an SVO filter-only curve labelled "filter", a
# curve equal to a registered component carrying that component's role, Sources one " | " list of
# distinct references, and one DetectorType for every served copy of the same curve.
python3 -m sed_tools.filters.label_check data/filters
python3 -m filter_info.svo_archive build source_evidence/svo-archive/parsed $(grep -v '^#' filter_info/svo_manifests.txt)
python3 generate_index.py data
if [ -s filter_info/unresolved.txt ]; then
  python3 -m filter_info.references resolve
  python3 generate_index.py data
  echo "filter_info/references.json gained entries: commit it (author Niall Miller)."
fi
# The SVO-compatible service at /sed_tools/svo_fps/ reads only data/svo_fps/catalog.json.
python3 filter_expansion/svc/build_catalog.py
