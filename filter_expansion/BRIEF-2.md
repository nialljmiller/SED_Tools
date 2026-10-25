# Filter expansion, phase 2: preserve, fix, and give users the tooling

Nill's decisions, 2026-10-25. Posted by the SED_Tools Claude session on his Mac. BRIEF.md still applies: format contract, no deleting, and Nill as sole author.

**Commits:** author must be `Niall Miller <niall.j.miller@gmail.com>`; use `git -c user.name="Niall Miller" -c user.email=niall.j.miller@gmail.com` (vega's global name is "Niall Miller - Vega", and Cirrus has none set). No Claude trailers, no "Generated with" lines, nothing pushed to `main`, no release.

## 1. Commit the swarm's findings to the SED_Tools `server` branch, and archive every raw download on NJM
- **What to commit** (`server` branch only), under e.g. `filter_expansion/` with a README:
  - the tooling in `~/SED_Workspace/filter-expansion/tools`: `fxlib.py`, `fx_validate.py`, `build.py`, `deploy.sh`, `verify_https.py`, `recipes/`, README;
  - the research files `research/*.csv` and `*.md`, plus BRIEF.md and BRIEF-2.md;
  - the provenance JSON files.
- **Also commit the Cirrus checkout's loose ends:** the three untracked `prepare_*_filters.py` scripts and the regenerated `data/index.json`.
- **Push:** push `server`, then fast-forward the Cirrus checkout (`/media/data3/MESA/SED_Tools`).
- **Archive the raw downloads:**
  - copy all 169 raw files in `~/SED_Workspace/filter-expansion/downloads` to Cirrus `/media/data3/MESA/SED_Tools/source_evidence/filters/2026-10-25-expansion/`, keeping the per-facility layout;
  - add a `manifest.json` (path, source URL, SHA-256, retrieved date);
  - follow the precedent in `source_evidence/filters/2026-10-23/manifest.json`;
  - source_evidence is not served publicly.

## 2. Fix every problem curve on NJM, however we can
- **Scope:**
  - 105 published curves the audit flags (75 with negative transmission, 30 with non-increasing wavelengths), listed in `/media/data3/MESA/SED_Tools-metadata-row-backup-20261025/audit_20261025T161503Z.csv`;
  - about 30 filenames with more than one dot (for example Paranal/NACO `IB_2.00.dat`). MESA names history columns by the text before the first dot, so these collide;
  - 119 curves with no DetectorType.
- **Approach for each curve, in order of preference:**
  1. re-source it from the original team's publication and rebuild with fxlib and a recipe;
  2. re-fetch it from SVO when SVO is reachable (svo2 is down; see Codex's diagnostic note);
  3. minimal curation with `fxlib.clean_curve` (sort, merge duplicate wavelengths, clip negative tails to 0), recording every change.

  Never invent data. Anything that can't be fixed honestly gets listed.
- **Dotted filenames:** rename to one-dot names (e.g. `IB_2.00.dat` → `IB_2p00.dat`). Update the index file and `manifest.json`, keep a raw copy of the old file in source_evidence, and commit an old→new rename map to `server` so users can be told.
- **Untagged curves:** tag DetectorType from the original documentation, or from SVO once it's back, with a matching DetectorTypeSource. Otherwise leave them and list them.
- **Every change must:**
  - back up each affected instrument directory first;
  - keep raw copies plus SHA-256 provenance;
  - pass `fx_validate.py`, `scripts/audit_filter_curves.py` (0 flags apart from listed held items) and `python3 migrate_filters.py --dry-run` (clean);
  - be followed by `python3 generate_index.py data` and an HTTPS check.

## 3. Tooling for users: rebuild from aggregated, vetted sources (SED_Tools feature branch + PR, not merged)
- **Port the recipes:** move the vetted original-source recipes into `sed_tools/component_sources` and register them in `catalog.BUILDERS_FOR`.
  - Then, when NJM is unavailable, `Filters.fetch` rebuilds those instruments from their original sources. `FilterLibrary.published` already tries NJM, then a registered builder, then SVO; 28 instruments are registered today.
  - Reuse the existing patterns, such as `component_sources/tabular.py` and `_common.Table` with `wl_scale`.
  - Add offline tests that use small fixtures.
- **For 0.4.2, make SVO fail fast:**
  - a short connect timeout instead of today's 120 s;
  - a clear error message;
  - a configurable FPS URL (it is hard-coded to svo2 today).
- **Hand-off:** open a PR from a feature branch for Nill to review.

**Order:** 1 first, so everything is preserved. Steps 2 and 3 can then run in parallel, since they touch different files. As manager you may hand research parts to Codex, Grok or Gemini (Gemini's default model is out of quota). Keep room messages short. Anything heavy on vega must be mentioned first: the MESA_ARGUS session schedules compute there.
