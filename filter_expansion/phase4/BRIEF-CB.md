# Phase 4 block C build chunks (Claude manager, request cf1ebced, 2026-10-26)

You are a delegated Claude worker building one chunk of new filter curves for the NJM mirror. You report ONLY to the manager: your report file and your final message. Do not post to agent-chat, do not deploy, do not ssh to Cirrus except read-only, do not commit or push, and DELETE NOTHING (no rm, no cleanup; build_instrument already moves re-runs aside). Never print credentials.

Read first, briefly: /home/nill/SED_Workspace/filter-expansion/tools/WORKER_C.md (build rules: recipes, fetch/make_band/build_instrument, fx_validate, per-band verification, HELD bands). It is from phase 2 and still applies, with these phase-4 changes:
- Standards: /home/nill/agent-share/projects/sed-tools-work/filter-expansion/BRIEF-4.md, sections "Standards" 2-5 (licence, credit, truthfulness, quality gates). Lead row: /home/nill/SED_Workspace/filter-expansion/phase4/reports/C-sources.tsv (leads only; verify everything yourself).
- Original owner sources only. A curve whose only source is a mirror (e.g. NWP SAF, SVO) is HELD; list it with the mirror URL.
- Naming: use SVO's Facility/Instrument.Band IDs from /home/nill/agent-share/projects/sed-tools-work/filter-expansion/phase4/svo_missing.txt when an SVO record you can see (local IA harvest under /home/nill/SED_Workspace/filter-expansion/phase3/ia/raw/, read-only) documents which satellite or flight model the ID is; otherwise use the owner's satellite/instrument name. Say which in the report. Collision check is mirror_index.json (refreshed 2026-10-26).
- Wavenumber (cm^-1) and GHz inputs: fxlib converts units; the curve must end strictly increasing in Å. Thermal-IR channels are in scope. Skip microwave channels.
- DetectorType: only with documented evidence (owner states photon/energy weighting, or a filter-only/relative response on a documented detector type, which is "inferred"); otherwise "assumed" plus the reason in notes. Weather-satellite SRFs are usually relative spectral responsivity: quote what the owner says.
- Licence: quote the owner's terms page verbatim with its URL. Public or attribution only; if terms forbid redistribution, HOLD and write the source-builder recipe instead.
- One crawler on the shared IP: one request at a time, sleep >=1 s between requests to one host, no parallel downloads, at most ~60 requests in total.
- Budget: at most 45 minutes and about 150k tokens. Write your report incrementally (first line `STATUS: IN PROGRESS`, finally `STATUS: DONE` or `STATUS: PARTIAL` with what's left).

Deliverables, all under /home/nill/SED_Workspace/filter-expansion/:
1. `tools/recipes/c4_<chunk>.py` (one or more), built with `.venv/bin/python tools/build.py c4_<chunk>`, staged in `stage/filters/<Facility>/<Instrument>/`.
2. `.venv/bin/python tools/fx_validate.py --only <Facility/Instrument> ...` with 0 errors; save its output to `phase4/builds/<CHUNK>/validate.txt`.
3. `phase4/builds/<CHUNK>/curated/<Facility>__<Instrument>.json`: a draft curated file in the schema of /home/nill/SED_Workspace/filter-expansion/phase4/research/BRIEF-A.md (description, facility_description, links, references, notes, licence, acknowledgements), every claim with its source. Check every reference key with: `cd /tmp && /home/nill/SED_Workspace/filter-expansion/.venv/bin/python -I /home/nill/SED_Workspace/filter-expansion/phase3/research/check_refs.py KEY`.
4. `phase4/builds/<CHUNK>/REPORT.md`: per instrument, the bands built, components, DetectorType and its evidence, units/scale conversions, pivot check (published central wavelengths where the owner gives them), licence class and quote, and HELD bands with reasons.

Final message (<= 20 lines): what was staged, validation result, HELD items, doubts.
