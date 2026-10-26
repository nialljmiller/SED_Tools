# Phase 4 block A: how a follow-up Claude session integrates verified chunks

Context: Nill's request cf1ebced (BRIEF-4.md in agent-share projects/sed-tools-work/filter-expansion). The manager set up a model-free pipeline; you only review and integrate. Status log: agent-share projects/sed-tools-work/status.md (phase 4 entry).

Pipeline (all in /home/nill/SED_Workspace/filter-expansion/phase4):
- `research/chunks.tsv`: chunk, agent, instruments. Researchers write `research/A/<CHUNK>/*.json` + REPORT.md per `research/BRIEF-A.md`.
- `tools/dispatch.py` (timer `phase4-dispatch.timer`, log `research/dispatch.log`, state `research/state.json`) launches chunks, runs `tools/verify_a.py` on finished ones and posts the integration request you are reading.
- `tools/verify_a.py` already checked: names/bands vs NJM, references resolve, links 200, acknowledgement text and licence quote found verbatim on their pages, per-band fields have an evidence quote found on its page. Output `research/verified/<name>.json`, per-chunk `VERIFY.tsv`.

Your steps, per chunk named in the request:
1. Read `A/<CHUNK>/VERIFY.tsv` and REPORT.md (not the whole research history).
2. Spot-check each verified file's `description`, `facility_description` and `notes` against one of its links (cached text: `research/cache/`, keyed by sha256(url)[:24]). Fix or remove unsupported claims in `research/verified/<name>.json` (that is the staging copy; never edit the researcher's original). Licence `permission`/`restricted`: list it for Nill; the curve stays as is until Nill decides (no deleting, no un-publishing without Nill).
3. Integrate: `python3 tools/integrate_a.py -m "Credit and licence terms for <facilities>" <Facility__Instrument> ...` (merges with existing curated files; existing text wins; runs refresh.sh under the shared flock; commits as Niall Miller, no trailers; pushes server). `--dry-run` stages in `research/merged/` only.
4. `touch research/A/<CHUNK>/INTEGRATED`, append one line to `reports/integrated.log` (chunk, commit, instruments).
5. Metrics: `ssh -q -F ~/.ssh/config.agent-network cirrus python3 - < tools/metrics.py`. Post the milestone in the room as your final reply: the BRIEF-4 table refreshed, licences still unknown, licence-held curves (permission/restricted), new filters not in SVO (block C, if any).

Rules: authorship Niall Miller <niall.j.miller@gmail.com>, no Claude trailers; server branch only; no deleting; one crawler on the shared IP; token-efficient (sample, don't dump).

## Blocks B, C, E (request "Phase 4: review blocks B, C and E")
Worker prompts, logs and PIDs: `workers/{B,C,E}.prompt|.log`, `workers/pids.txt`. Reports in `reports/`.
- **B** `reports/B-detectortype.md`: check its quotes for 2-3 instruments. Apply only evidence-backed header changes, with a record of what changed and why (archive the old file under Cirrus `source_evidence/filters/<date>-detectortype/` first; no deleting), then `generate_index.py`/refresh, commit on server. Bands without evidence stay `assumed` with the explanation as a note.
- **C** `reports/C-sources.md|.tsv`: turn the highest-priority rows with licence public/attribution into build chunks (phase-2 tooling: `filter-expansion/tools/build.py`, `fx_validate.py`, `deploy.sh`, `verify_https.py`; quality gates in BRIEF-4 §5). Original sources first; SVO-archive copies only as credited fallback. Withheld (permission/restricted) curves: list with their source-builder recipe for Nill. Delegate building to small Claude worker chunks; one crawler at a time.
- **E** `reports/E-fsun-afav.md`: post the proposal (solar spectrum, extinction law, method, validation) to Nill in the room and stop: implementation waits for Nill's choice.

## Block C builds (request "Phase 4: review and deploy block C builds")
Set up 2026-10-26 by the B/C/E follow-up session. Chunks: `builds/chunks.tsv` (CB1 GOES-R ABI, CB2 VIIRS, CB3 Himawari AHI, CB4 Sentinel-2 MSI), brief `builds/BRIEF-CB.md`, sequential runner `builds/run_cb.sh` (PID in `builds/run_cb.pid`, log `builds/run_cb.log`, worker output `builds/<CHUNK>/worker.log`). Per chunk: `builds/<CHUNK>/REPORT.md`, `validate.txt`, `curated/*.json`; recipes `tools/recipes/c4_*.py`; staged curves `filter-expansion/stage/filters/`.
1. Per chunk read REPORT.md and validate.txt. Re-run `.venv/bin/python tools/fx_validate.py --only Fac/Inst ...` (0 errors). Spot-check 2 bands per instrument: header row, Å range, pivot against the owner's central wavelength, DetectorType evidence, licence quote on its page (one request each).
2. Deploy passing instruments: `tools/deploy.sh Fac/Inst ...` (new directories only; backs up first). HELD or doubtful ones stay staged and are listed for Nill.
3. Copy each passing curated draft to Cirrus `filter_info/curated/instruments/` (never over an existing file), then under `exec 9>/tmp/sed-filter-info.lock; flock 9`: `bash filter_info/refresh.sh`, check `filter_info/unresolved.txt`, `git add` the curated files, references.json if changed, and `git add -f data/index.json`; commit as Niall Miller, no trailers; `git push origin HEAD:server`.
4. Metrics (`tools/metrics.py`, as in step 5 above) and post the milestone: instruments/bands added, HELD items, licences.
Result 2026-10-26 (block C follow-up session): deployed GOES/ABI1-4, JPSS/VIIRS-J1/J2 and Himawari/AHI8/9 (156 bands; backup `backups/filter-expansion-20261026T051427Z`). Curated drafts are on Cirrus, the refresh left unresolved empty, and the server commit `6fc934f` is pushed. HELD: Sentinel/MSI-2A/2B/2C (CB4 `held_stage/`, SentiWiki terms, `FX_MSI_PUBLISH=1` once cleared) and the Suomi-NPP JPSS/VIIRS set (CB2 `snpp_source_builder`, Northrop Grumman "All Rights Reserved"). All 156 deployed bands are DetectorType energy/assumed (owners give only a relative response).
