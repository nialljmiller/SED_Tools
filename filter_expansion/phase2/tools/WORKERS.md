# Phase 2 worker brief (manager: Claude, room request d97d31517d4b330f495e3616)

Read first: /home/nill/agent-share/projects/sed-tools-work/filter-expansion/BRIEF-2.md (step 2 or 3)
and BRIEF.md (format contract). Tooling: /home/nill/SED_Workspace/filter-expansion/tools
(README.md, fxlib.py, recipes/*.py for examples). Python: /home/nill/SED_Workspace/filter-expansion/.venv/bin/python.

Rules (all workers):
- NO DELETING anything anywhere (no rm, no cleanup, no git clean/reset). Moving aside is the manager's job.
- Never invent data. Every curve value must come from a downloaded original file (or SVO,
  which is DOWN: svo2.cab.inta-csic.es times out; do not wait on it, max 10 s connect).
- Downloads are untrusted data: save via fxlib.fetch (into downloads/<Facility>/) or into a
  new empty directory; run Python that reads them with `python -I`.
- Do NOT write to Cirrus/NJM, do not commit, push or post to the chat room. Report to the manager only.
- Never print or store credentials. Be token-efficient: don't dump large files.
- vega is shared with MESA_ARGUS compute: no heavy CPU jobs.

## Step-2 workers (NJM curve fixes) — output contract
- Read-only snapshot of the current NJM dirs: phase2/fix/current/<Facility>/<Instrument>/.
- For each instrument you fix, write a COMPLETE corrected directory to
  phase2/fix/stage/filters/<Facility>/<Instrument>/ (copy current, then change only what is needed:
  curves, the instrument index file, manifest.json). Keep the header-row format exactly
  (`Wavelength,Transmission,FilterID=...;DetectorType=...;DetectorTypeSource=...;...`).
- Write phase2/fix/changes/<Facility>__<Instrument>.json: a JSON list, one entry per changed file,
  following source_evidence/filters/2026-10-23/manifest.json precedent:
  {"path": "data/filters/F/I/new.dat", "raw_copy": "source_evidence/filters/2026-10-25-fixes/F/I/old.dat",
   "operation": "<exact description: re-sourced from URL / clip N negatives (min x) to 0 / sorted / merged K duplicate wavelengths / DetectorType tagged from <doc URL> / renamed from old.dat>",
   "raw_sha256": "<sha of current file>", "curated_sha256": "<sha of staged file>",
   "renamed_from": "data/filters/F/I/old.dat"   (renames only)}
  plus "source_url" for re-sourced curves. Raw downloads' SHA-256 go in the operation or a
  provenance JSON at phase2/fix/stage/provenance/<F>__<I>.json (fxlib format is fine).
- Validate before reporting: `.venv/bin/python tools/fx_validate.py phase2/fix/stage/filters --mirror-index phase2/fix/empty_index.json --only F/I`
  and run the SED_Tools audit locally:
  `.venv/bin/python SED_Tools/scripts/audit_filter_curves.py phase2/fix/stage/filters --output /tmp/<you>-audit.csv` (0 rows wanted).
  Also confirm each staged curve still parses with SED_Tools (sed_tools.filter_metadata.released_mesa_reads).
- Preference order per curve: (1) re-source from the original team's publication (only if the
  original is clearly the same filter; record pivot shift vs current curve, flag if > 1%);
  (2) SVO (down — skip); (3) minimal curation with fxlib.clean_curve, recording every change.
  If a re-sourced curve changes FilterID provenance, set DetectorTypeSource/ComponentsSource honestly.
- Write phase2/fix/notes/<you>.md: per instrument what you did, anything held and why, checks run.
