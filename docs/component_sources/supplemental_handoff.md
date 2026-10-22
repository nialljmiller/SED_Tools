# Vega component-source follow-up (2026-10-22)

This handoff collects the later ground-source work on Vega and Codex's independent
review. It supplements the [main research report](report.md); it does not raise
the verified coverage count or enable new runtime sources. The 163 reviewed SVO
browse keys (24 Claude, 70 Grok, 69 Gemini) overlap the supplemental reviews.
They are not 163 complete optical-path reconstructions.

At the Vega task boundary, Claude's assigned space milestone and Gemini's
other-system milestone are marked done. Grok saved the 70-key ground checkpoint,
but its later research and cross-review tasks are marked blocked by provider
access. Codex completed the saved-checkpoint audits and sampled selected
primary files; no agent has certified every component or band. The later
Claude ground passes were extra work outside the original partition and
overlap keys already in the baseline matrix.

## What was collected

- [Claude's first ground follow-up](sources/round2/claude/ground/report.md)
  covers CSP-I/II, Las Cumbres filter tables, DECam, and unsuccessful original
  source searches. Its `reviews.csv` is a historical checkpoint.
- [Claude's larger harvest](sources/round2/claude/ground2/report.md) and
  [numeric-file inventory](sources/round2/claude/ground2/inventory.csv) cover
  ESO, Gemini, Subaru, Rubin, STScI, IRSA and other leads. The inventory has
  633 rows, but only 594 distinct file contents. It includes sky backgrounds,
  examples, spatial calibration and duplicate copies; it is **not** a count of
  usable filter curves. Its per-file summaries are a first-two-column parser,
  so multi-column products need separate inspection.
- [Provenance for the 190 inventory rows without a direct URL](sources/round2/claude/ground2/provenance_for_unlinked.csv)
  maps each to a pinned Rubin repository file or an exact member of an ESO
  archive. Files under Rubin's `2MASS`, `WISE`, `megacam` and similar folders
  are mirrors, not those missions' original publications. These mappings
  identify the saved bytes; they do not establish physical meaning or licence.
- [Codex's current-work audit](sources/round2/codex/current_work_audit.md)
  and [measurements](sources/round2/codex/current_work_audit.json) check source
  hashes, inventory structure and selected numerical claims. The
  [audit script](sources/round2/codex/current_work_audit.py) records how those
  checks were made. The raw evidence is deliberately outside Git: on Vega at
  `/home/nill/agent-share/projects/sed-tools-components/round2/claude/{ground,ground2}/evidence/`
  and in the Mac cache at
  `/Users/njm/.cache/SED_Tools/vega-component-research-20261021/`.
  Both [ground](sources/round2/claude/ground/evidence/SHA256SUMS) and
  [ground2](sources/round2/claude/ground2/evidence/SHA256SUMS) hash manifests
  were checked against the Mac copy: 14 + 704 files match. No redistribution
  rights have been cleared for publishing those original bytes.

The supplemental `reviews.csv` files are preserved as worker checkpoints,
**not merged into the main component matrix**. Several ground2 statuses
overstate separability: NIRCam and IRAC system curves do not by themselves
provide separate filter, optics and QE files; some Gemini entries mark a
published system response when only filter scans were opened. Follow the audit
and original source definitions before changing any production metadata.

## Sources with the most implementation value

| Source | Established contents | Remaining interpretation |
|---|---|---|
| [ESO VIRCAM/VISTA bundle](https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst/Filters_QE_Atm_curves.tar.gz) | Eight filter files, M1/M2 reflectivity, detector QE and two supplied atmospheres. An adapter already exists. | Hardware is nm/percent-like; atmosphere is µm/fraction. The two models are 1 mm PWV at X=1 and 3 mm at X=2, not a general grid. Do not assume these files account for every window or lens. |
| [Rubin throughput repository](https://github.com/lsst/throughputs/tree/0bcec2a6bec62e442a094df738ec2a9fdff844c5/baseline) | Separate baseline filters, mirrors, lenses, detector and atmosphere; saved hardware × standard atmosphere equals saved total in ugrizy. An adapter already exists. | The saved baseline is v1.9 and does not automatically reproduce older SVO LSST v1.5. Other-mission folders are mirrors. |
| [ESO ERIS/NIX archive](https://www.eso.org/sci/facilities/paranal/instruments/eris/img/nix-filters.zip) | Filter scans, camera paths, dichroic and ND files. A NIX preparation script exists on Cirrus. | Files in one archive have different units and meanings; ND exceeds one in some samples. Establish a recipe per observing mode. |
| ESO [HAWK-I](https://www.eso.org/sci/facilities/paranal/instruments/hawki/inst/filters/hawki_J.dat), [NACO](https://www.eso.org/sci/facilities/paranal/instruments/naco/inst/filters/J_CONICA.dat), [SPHERE](https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_Y.dat) and [FORS](https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/M_BESS_B.txt) example files | Many original filter files; some ND and collimator data. | Filter generation, temperature, field position and percent/fraction conventions vary. No full detector/optics decomposition is established from these downloads. |
| [Subaru HSC sensitivity tables](https://www.naoj.org/Observing/Instruments/HSC/sensitivity.html) | Filter generations and separate detector/optics tables. An adapter already exists. | `vignetting.txt` is a spatial function of radius in degrees, not a wavelength response. Keep old and replacement filters distinct. |
| [Gemini component pages](https://www.gemini.edu/instrumentation) | GMOS filters and detector QE; NIRI, GSAOI, Flamingos-2, GNIRS and NIFS filter files. | Detector epochs, cold/warm scans, system versus filter-only products and units differ. A component page is not proof of a complete optical path. |
| [CSP passbands](https://csp.obs.carnegiescience.edu/data/CSPCSPII_passbands.tgz) and [component package](https://csp.obs.carnegiescience.edu/data/CSP_filter_package.tgz) | CSP-I/II response FITS and telescope/camera/atmosphere products. | Published passbands already include atmosphere at X=1.2. Map filter replacement epochs; telescope+camera is combined, not isolated QE. |
| [Las Cumbres filter archive](https://lco.global/documents/2/lco_filter_curves.zip) | 18 filter tables. | Their exact included factors are not established by the file headers alone. |
| [CFHT MegaCam technical page](https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/specsinformation.html) | Official filter, mirror/corrector and measured/simulated QE links, missed by the first harvest. | Retrieve the tables, distinguish the pre/post-2015 filter sets and verify how published throughput is defined. This is a high-value next source. |
| [STScI SYNPHOT tables](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/synphot-throughput-tables) | HST component and graph-table route for several cameras. An HST adapter already uses this family. | Preserve the applicable graph-table epoch, chip and calibration version for each band. |

The [NOIRLab filter service](https://noirlab.edu/science/filters) yielded 226
candidate IDs in the saved harvest manifest but **zero downloaded numeric
curves**. Its final log says `done` after an HTTP 202 challenge and later HTTP
200 probe. A successful probe is not a filter file. Recover document links
slowly or use SVO for now; do not count the 226 IDs as acquired bands.

## Checks before adding a source to SED_Tools

1. Start with the original page or archive and enumerate bands, detector
   variants, modes and epochs without querying SVO. Join the inventory to the
   provenance sidecar by `path` when `source_url` is blank.
2. Inspect each table's axes, response unit and included factors. In particular,
   HSC vignetting is spatial; SPHERE ND has three response columns and NaNs;
   [MIPS `MIPSfiltsumm`](https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt)
   has three wavelength/response pairs (24, 70 NB and 160 µm), not one curve.
3. Keep measured factors, combined products and derived quotients separate.
   For [DECam](https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt),
   dividing the published table by its atmosphere column yields i/z values
   above one. Preserve the source values and investigate normalization; do not
   silently clip the quotient or call it a measured filter.
4. Compare against an official total only at the matching release and
   observing conditions. Combined NIRCam/IRAC responses do not prove that
   separate QE/optics files exist. WISE and WFC3 downloads still need review.
5. Record source URLs, archive members, checksums, release, licence and failure
   behavior before enabling an NJM mirror. Keep SVO as a labelled fallback.
   Reuse common table/archive parsing for simple sources; add special-case code
   only where interpretation truly differs.

These files are an implementation queue and reproducible research record, not
an executable registry. The existing [implementation notes](implementation_notes.md)
describe the CLI/Python API and fallback acceptance checks for later coding.
