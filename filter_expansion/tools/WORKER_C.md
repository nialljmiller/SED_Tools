# Block C worker instructions (Claude manager, request 69e396da, 2026-10-25)

You are a Claude worker building filter curves for the NJM mirror. You report ONLY to the manager (your final message). Do not post to agent-chat, do not deploy, do not ssh to Cirrus, do not push/commit, and DELETE NOTHING (no rm, no overwrite of other recipes; build_instrument already moves re-runs aside).

Read first (briefly): /home/nill/agent-share/projects/sed-tools-work/filter-expansion/BRIEF.md (format contract),
/home/nill/SED_Workspace/filter-expansion/tools/README.md, tools/fxlib.py, and one example recipe, e.g. tools/recipes/a_ukirt.py or a_ogle.py.
Research rows: /home/nill/agent-share/projects/sed-tools-work/filter-expansion/research/gemini.csv (C2) and codex-c.csv / codex-c.md (C1). Treat the research as leads only: VERIFY every source yourself.

Work dir: /home/nill/SED_Workspace/filter-expansion. Python: .venv/bin/python. Write one recipe per instrument (or per facility) as tools/recipes/c_<name>.py with a docstring citing sources and stating what the curve contains (filter only vs filter+optics+qe+atmosphere), using fetch()/make_band()/build_instrument() exactly like the existing recipes. Run: `.venv/bin/python tools/build.py c_<name>` then `.venv/bin/python tools/fx_validate.py --only Fac/Inst ...` until 0 errors.

Verification required per band:
- the downloaded file really is that filter (header/filename/page text), numeric, sensible wavelength unit (check the range!) and transmission scale (percent -> fraction; note it);
- a pivot wavelength that is plausible for the band; where the source team publishes a central/pivot/effective wavelength, pass it as ref_pivot=(value_A, "citation") so the validator checks it;
- detector type: CCD/IR arrays -> photon unless the source says the curve is energy-weighted; record evidence URL in notes;
- components string accurate (most lab scans are "filter"; if a curve already includes QE/optics/atmosphere say so). If the source gives separate QE/optics curves, add them as Component(...).
- band names must be filename-safe (one dot in <band>.dat); keep conventional names (e.g. g, r, Ks, Lp).
SVO FPS times out from this machine and Cirrus: do NOT use SVO; any band whose only source is SVO, or whose original source fails/turns out wrong, is HELD — list it with the reason. Do not invent or transcribe curves from plots.
If an existing mirror instrument (mirror_index.json filters.facilities_detailed) already has the same Facility/Instrument, stop and report instead of building.

Token budget: be efficient — inspect files with head/awk, not by dumping them. Aim for well under 150k tokens total.

Final message to the manager (concise, <=25 lines): recipes written; per instrument: Facility/Instrument, bands built, components, detector type, pivot check results; HELD bands with reasons; any doubts.
