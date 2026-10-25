# Claude worker: dotted filenames + missing DetectorType (BRIEF-2 step 2)

Script: /tmp/clw2work/fix.py (copies current/ -> stage/filters/, renames, tags headers with
sed_tools write_filter_header so sample rows stay byte-identical, writes changes/ + provenance).
Docs saved in downloads/_docs_phase2_claude/ (SHA-256 in each changes operation and stage/provenance/).

## Renames (30) — phase2/fix/rename_map.csv
- Paranal/NACO: 28 (IB_2.00..IB_2.48 -> IB_2pNN, NB_1.04..NB_3.74 -> NB_NpNN). Index file NACO and
  manifest.json updated (bands, components.filter, detector_types, detector_type_sources, published_bands;
  140 changed lines, nothing else). Curve files byte-identical (raw == curated sha).
- ERIS/NIX: 2 (IB-2.42 -> IB-2p42, IB-2.48 -> IB-2p48). Index NIX updated; no manifest.json exists.
- FilterID: none of the renamed files carries a FilterID, so nothing to change. If one did, it would be
  the SVO ID (upstream identifier, independent of our filename) and should stay unchanged. The Sources URLs
  keep the upstream ESO filenames (provenance). No components/ subdirs exist in these dirs.

## DetectorType tags (DetectorTypeSource=original, recipe vocabulary for team documentation)
- ERIS/NIX 7 photon: ERIS User Manual v118.0 sect. 2.4.1.1 (Teledyne Hawaii 2RG).
- HAWK-I/HAWK-I 12 photon: eso.org hawki.html (four HAWAII 2RG detectors).
- SPHERE/IRDIS 30 photon: SPHERE User Manual P114 p.11 (Hawaii II RG).
- SPHERE/ZIMPOL 23 photon: same manual, Table 3 (e2v CCD 44-82 frame-transfer CCDs).
- Teide/COSMO15 3 energy: Gallegos et al. astro-ph/0108020 (total-power receivers, HEMT + square-law detectors).
- Teide/COSMO11 2 energy: Hildebrandt et al. arXiv:0706.1873 (2 total-power channels, matched square-law
  detectors). Energy because square-law output is proportional to power (same reasoning as Planck LFI recipe).
- Untagged leftovers: none.

## Checks
- SED_Tools audit_filter_curves.py over the 7 staged dirs: 120 curves, 0 rows.
- released_mesa_reads True for all 120; all one-dot names; all have DetectorType.
- fx_validate (root must be phase2/fix/stage, not stage/filters): remaining errors are pre-existing
  legacy-format gaps outside this task: no FilterID (all 7 dirs, except ERIS svo-matched ones), no
  manifest.json (all but NACO), SPHERE/Teide headers lack WavelengthUnit/Components/ComponentsSource,
  Teide range flagged "implausible" (microwave, 1.8e8-3.3e8 Angstrom, real). Not changed: no evidence-backed
  values supplied beyond DetectorType. NACO DetectorTypeSource stays "assumed" (not in scope; CONICA uses
  an Aladdin InSb array, could be upgraded to original later).
