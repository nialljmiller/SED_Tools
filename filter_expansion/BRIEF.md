# Filter expansion brief (sed-tools-work)

Posted 2026-10-25 by the SED_Tools Claude session on Nill's Mac, at Nill's request. Read this once; room messages only link here.

## Goal
Add the popular filter systems users expect to the NJM mirror. That means many instruments, including the standard systems, surveys and missions people actually use, not obscure ones.

## Roles (Nill's instruction)
- **Codex, Grok, Gemini: research only.** For your block, find the source links, components and detector types. No code, and no writes to the mirror.
- **Claude: manager and coder.** Claude checks and tests everything, writes the code, and adds the filters to the NJM server via ssh to Cirrus.
- Be token-efficient: compact tables, links rather than pasted text, no re-research of what already exists, and short room messages.

## Baseline (the workspace's 0.1.2 / 015fbaa source is obsolete)
- **SED_Tools 0.4.1:** GitHub nialljmiller/SED_Tools `main` e5464af, also on PyPI as `sed-tools` 0.4.1.
- **Mirror:** Cirrus `/media/data3/MESA/SED_Tools/data`, on the `server` branch at fbbfafa, served at https://nillmill.ddns.net/sed_tools/.
- **Server tools:** `generate_index.py` and `migrate_filters.py`. The `prepare_*_filters.py` scripts are examples; they write plain files without metadata.
- **From Vega:** use `ssh -F ~/.ssh/config.agent-network cirrus`.
- **Already on the mirror:** 3,963 filters in 75 facilities, listed in https://nillmill.ddns.net/sed_tools/index.json. Check it before researching anything.
- **Prior research to reuse, not redo:** `projects/sed-tools-components/` (`facilities.csv`, `instruments.csv`, `priority_queue.csv`, `results/`, `round2`, `round3`).

## Research blocks (gaps found against index.json; add obvious omissions within your block)
- **@codex, A. Ground surveys and networks:**
  - Pan-STARRS1 (grizy, w);
  - SkyMapper (uvgriz);
  - UKIRT WFCAM/UKIDSS (ZYJHK);
  - ATLAS (c, o), ASAS-SN (V, g), OGLE-III/IV (V, I), APASS;
  - J-PLUS and J-PAS (OAJ);
  - LCO (Sinistro, MuSCAT3, Spectral);
  - INT WFC, KMTNet.
- **@grok, B. Space missions:**
  - Spitzer (IRAC, MIPS, IRS peak-up);
  - Swift UVOT, XMM-Newton OM;
  - IRAS, Herschel (PACS, SPIRE), MSX, ISO (ISOCAM, ISOPHOT main bands);
  - Hipparcos Hp and Tycho BT/VT;
  - Planck (HFI, LFI), SPHEREx, PLATO;
  - Gaia passband versions not yet on the mirror (DR2 revised, EDR3/DR3), checked against the existing GAIA/GAIA set.
- **@gemini, C. Standard systems and large telescopes:**
  - a `Generic` facility with Bessell UBVRI, Johnson UBVRIJHKLMN, Cousins RI, Kron, Strömgren uvby-β, MKO JHKL'M', Thuan-Gunn and USNO u'g'r'i'z'. MESA's default instrument is `Generic/Johnson`; compare with the GCPD sets already present.
  - Keck (NIRC2, LRIS, MOSFIRE, DEIMOS);
  - VLT FORS1/FORS2, ISAAC, VISIR; VST OmegaCAM;
  - Subaru (Suprime-Cam, MOIRCS, FOCAS);
  - Magellan (FourStar, IMACS, MegaCam);
  - NOT ALFOSC, WIYN ODI, KPNO Mosaic.

## Research output (one file each)
- **Location:** `filter-expansion/research/<agent>.csv`, plus a short `<agent>.md` whose first line is `STATUS: IN PROGRESS` or `STATUS: DONE`.
- **CSV columns:**
  `facility,instrument,band,priority(1-3),source_kind(original|svo),source_url,format(columns,delimiter),wavelength_unit,detector_type(photon|energy),detector_evidence_url,components(total|filter|optics|qe|atmosphere),component_urls,svo_id,license_note,notes`
- **Sources:** prefer the original team's published curves and components, with SVO as the fallback. Detector type needs evidence: CCDs and IR arrays are photon-counting unless the source says otherwise. Note whether a curve already includes the atmosphere or QE.
- **Finishing:** set `STATUS: DONE`, then post one short room message linking your files.

## Format contract (Claude must enforce it for every file)
Released MESA colors (25.10.1rc1, 25.12.1, 26.4.1) discards line 1 of a filter file and counts every other line as a sample.
- **Writing:** write every curve with `sed_tools.filter_metadata.write_filter_file`. The metadata goes in the third field of the `Wavelength,Transmission,FilterID=...;DetectorType=...` row: FilterID, DetectorType, DetectorTypeSource, WavelengthUnit=Angstrom, Components, ComponentsSource. Never write `#` lines or blank lines.
- **Samples:** wavelengths in Å and strictly increasing, transmission ≥ 0, at least 2 samples.
- **Filenames:** `<band>.dat` with exactly one dot. MESA names history columns by the text before the first dot, so a name like `IB_2.00.dat` collides with its neighbours.
- **Per instrument:**
  - an index file named after the instrument, listing every `.dat`;
  - a `manifest.json` (`InstrumentManifest`);
  - separate component curves under `components/<role>/`.
- **Checks:**
  - `sed_tools.filter_metadata.released_mesa_reads(path)` returns True for every file;
  - `read_filter_header` returns the intended fields;
  - the pytest suite passes if any code changes;
  - spot-check synthetic photometry or zero points against SVO where possible.

## Deploying to NJM (Claude)
1. Back up each affected `data/filters/<Facility>` before writing. Delete nothing.
2. Add the new sets under `data/filters/`.
3. Run `python3 migrate_filters.py --dry-run`. It should report 0 curves and no misreads.
4. Run `python3 generate_index.py data`.
5. Verify over HTTPS: fetch the files, check row 1 and the index.

## Code and authorship
- Nill must be the sole author of everything on GitHub: no Claude co-author trailers and no "Generated with" lines.
- Server-only scripts belong on the `server` branch.
- Don't release SED_Tools or push `main` without Nill's approval.

## Sequencing
Research runs concurrently. Claude starts with stage 1 now: baseline, tooling and validation harness. Claude then integrates each block once its `.md` says `STATUS: DONE`. If research is still running when Claude's session ends, Claude records progress in `status.md`; the Mac session or Nill will then post the integration request to @claude.
