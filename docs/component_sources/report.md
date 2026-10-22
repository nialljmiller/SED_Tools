# Independent component-source research

The inventory contains **649 SVO facility/instrument browse keys** across 206 groups. The current research milestone has dossiers for **163 keys** (25.1% of the inventory), plus SPHEREx outside the SVO baseline. A key is a browse entry, not necessarily a distinct physical instrument. The full [priority queue](priority_queue.csv) preserves the remaining entries, ordered qualitatively from widely used filter sets to narrower systems.

## Coverage and limits

| Owner | Assigned | Target | Reviewed keys | Original files | Original documentation | Mixed | SVO only |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude | 96 | 24 | 24 | 20 | 1 | 1 | 2 |
| Grok | 280 | 70 | 70 | 53 | 16 | 0 | 1 |
| Gemini | 273 | 69 | 69 | 33 | 0 | 14 | 22 |

Across the 163 reviewed keys, researchers found some original numeric files for 106, original documentation for 17, mixed sources for 15, and SVO-only data for 25. These are researcher classifications. One verified representative file does not establish every band, component, observing mode or detector variant. A combined passband cannot generally be split into physical parts. Many dossiers remain incomplete in that depth.

1 ground source record still lack both a data URL and a saved landing URL. Other documentation leads may point to a paper or instrument page without an opened numeric curve. See the [manager follow-up](sources/round2/manager/ground_source_followup.md) and [reconciliation](sources/round2/manager/reconciliation.md).

The status `combined_only` means the factor is present in a combined curve but no separate usable curve was established. `not_found` records the documented search, not proof of absence. `unknown` includes work not yet done. `file_verified` on an SVO fallback means its numeric response was opened; it does not make the source independent. Redistribution terms have not been cleared for NJM hosting.

## Source dossiers and independent checks

- [Claude's space dossiers](sources/round2/claude/report.md) and [source records](sources/round2/claude/sources.jsonl): JWST, HST, Gaia, TESS, Kepler, WISE, Spitzer, Euclid, Herschel, Roman and others. [Peer review](sources/round2/claude/peer-review.md).
- [Grok's ground dossiers](sources/round2/grok/report.md) and [source records](sources/round2/grok/sources.jsonl): major surveys, VLT, Gemini, Subaru, Keck and others. Its attempted later peer review hit a provider balance limit.
- [Gemini's other-system dossiers](sources/round2/gemini/report.md) and [source records](sources/round2/gemini/sources.jsonl): planetary, Earth-observation, generic systems and cameras. Its report includes a VISTA/ZTF peer check.
- [Codex audit](sources/round2/codex/report.md), [issues](sources/round2/codex/issues.jsonl) and [focused review](sources/round2/codex/focused-review.md). Each states its actual input hashes and verification boundary; structural checks do not certify all numeric claims.
- [Vega supplemental handoff](supplemental_handoff.md) combines Claude's later ground harvest, per-file provenance and [Codex's current-work audit](sources/round2/codex/current_work_audit.md). Its additional worker reviews overlap the baseline and are not promoted into the main matrix.
- [Mac manager checks](sources/round2/manager/report.md) and [exact retrieval log](sources/round2/manager/checks.json): independent WISE W1 and PACS blue numerical checks plus MIPS documentation. [Post-audit reconciliation](sources/round2/manager/reconciliation.md) identifies later corrections and remaining retrieval-link gaps. [Ground follow-up](sources/round2/manager/ground_source_followup.md) adds official landing pages without upgrading unverified numeric curves.

Source records carry landing/data URLs, response contents, retrieval instructions, units, version, detector convention and search notes where established. Read the dossier and linked original definition before coding a parser; bare URLs are insufficient. The [matrix](review_matrix.csv) provides all five component-role statuses per entry. The [source-link table](source_links.csv) is a sortable list of original and fallback URLs, component roles, units, versions and retrieval instructions.

## Scientific and sourcing decisions for later coding

- Preserve the source release, detector, filter generation and observing mode. Rubin v1.9 cannot silently stand in for SVO's older LSST v1.5; JWST v5 and v7 curves differ.
- Preserve each curve's physical quantity. A WISE per-photon response, a MIPS electrons-per-energy response, a GALEX effective area and a dimensionless filter transmission are not interchangeable. The [manager check](sources/round2/manager/report.md) documents a corrected WISE normalization claim.
- Cassini's published system curve contains two filter wheels with QE and optics. Dividing by QE and optics yields a **filter pair**; dividing paired system curves can yield a ratio relative to a clear filter. Neither operation automatically isolates a single absolute filter curve. Keep calibration and camera applicability explicit.
- VISTA's hardware tables use nanometres and percent-like response, while its supplied atmosphere files use micrometres and fractions. Keep site and observing conditions attached to each atmosphere product. ZTF's detector coatings are distinct variants.
- Original-source discovery must enumerate bands and components without SVO. SVO aliases and fallback remain useful. The code paths and acceptance checks for this later work are in [implementation notes](implementation_notes.md).

## Retrieval and implementation handoff

1. Choose an instrument below; follow the dossier's original archive index or manifest to enumerate every band, mode and component. Record exact source version and checksum.
2. Download original files, parse native units and definitions, and retain measured, combined and derived quantities separately. Retrieve official totals for comparisons only where the matching factors are supplied.
3. Record gaps at the component level, including the search trail and access restrictions. Use SVO as an explicit fallback with its provenance. Do not treat SVO data as independent.
4. Resolve the relevant [audit issues](sources/round2/codex/issues.jsonl), release mappings and redistribution rights before enabling an NJM-hosted downloader. Then test discovery with SVO disabled and fallback with NJM offline.

Research has not changed the application, deployed the website or downloaded a bulk mirror. The report is a handoff for the later coding task, not an executable source registry.

## Instrument index

Each link opens the owner's dossier; the full machine-readable statuses are in the matrix.

| Instrument | Example source | Filter | Optics | QE | Atmosphere | Published | Source basis |
|---|---|---|---|---|---|---|---|
| [2MASS/2MASS](sources/round2/grok/report.md) | [source](https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec3_1b1.tbl1.html) | available | available | available | available | available | original_files |
| [APO/SDSS](sources/round2/grok/report.md) | [SVO browse](https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=APO&gname2=SDSS) | available | unknown | unknown | unknown | unknown | original_files |
| [Aqua/MODIS](sources/round2/gemini/report.md) | [source](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Aqua_RSR_in-band.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [BOK/90prime](sources/round2/grok/report.md) | [source](https://www.legacysurvey.org/dr10/files/bass-g.txt) | available | not_found | available | available | available | original_files |
| [BOK/BASS](sources/round2/grok/report.md) | [source](https://www.legacysurvey.org/dr10/files/bass-g.txt) | available | not_found | available | available | available | original_files |
| [BOK/SCUSS](sources/round2/grok/report.md) | [source](https://doi.org/10.3847/0004-6256/151/2/37) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [CFHT/CFH12k](sources/round2/grok/report.md) | [source](https://www.cfht.hawaii.edu/Instruments/Filters/curves/cfh7403.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [CFHT/MegaCam](sources/round2/grok/report.md) | [source](https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/data.MegaPrime/MegaCam_QE_data.txt) | available | not_found | available | not_found | documented | original_files |
| [CFHT/Megaprime](sources/round2/grok/report.md) | [source](https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/data.MegaPrime/CFHT_MegaPrime_Transmission.txt) | documented | available | unknown | not_found | available | original_files |
| [CFHT/Wircam](sources/round2/grok/report.md) | [source](https://www.cfht.hawaii.edu/Instruments/Filters/curves/cfh8101.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [CTIO/DECam](sources/round2/grok/report.md) | [source](https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt) | documented | unknown | unknown | documented | available | original_files |
| [CTIO/S-PLUS](sources/round2/grok/report.md) | [source](https://raw.githubusercontent.com/splus-collab/splus_filters/master/gSDSS.csv) | available | unknown | unknown | unknown | unknown | original_files |
| [Cameras/Arriflex](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Arriflex.D21_B) | not_found | not_found | not_found | not_applicable | available | svo_only |
| [Cameras/Canon](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Canon.10D_Block) | available | not_found | not_found | not_applicable | available | mixed |
| [Cameras/Fujifilm](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Fujifilm.S3_Block) | available | not_found | not_found | not_applicable | available | svo_only |
| [Cameras/Hasselblad](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Hasselblad.H3D_B) | not_found | not_found | not_found | not_applicable | available | mixed |
| [Cameras/Kodak](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Kodak.DCS200_B) | not_found | not_found | not_found | not_applicable | available | svo_only |
| [Cameras/Leica](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Leica.M8_B) | not_found | not_found | not_found | not_applicable | available | svo_only |
| [Cameras/Lumix](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Lumix.G10_Block) | available | not_found | not_found | not_applicable | not_found | svo_only |
| [Cameras/Nikon](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Nikon.D100_Block) | available | not_found | not_found | not_applicable | available | mixed |
| [Cameras/Nokia](sources/round2/gemini/report.md) | [SVO browse](https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Cameras&gname2=Nokia) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Cameras/Olympus](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Olympus.EP1_Block) | available | not_found | not_found | not_applicable | available | mixed |
| [Cameras/Panasonic](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Panasonic.DML-LX3_B) | not_found | not_found | not_found | not_applicable | available | svo_only |
| [Cameras/Pentax](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Pentax.K100D_Block) | available | not_found | not_found | not_applicable | available | mixed |
| [Cameras/PhaseOne](sources/round2/gemini/report.md) | [SVO browse](https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Cameras&gname2=PhaseOne) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Cameras/PointGrey](sources/round2/gemini/report.md) | [SVO browse](https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=Cameras&gname2=PointGrey) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Cameras/SQM](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/SQM.CM500_fil) | available | not_found | documented | not_applicable | available | mixed |
| [Cameras/Samsung](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Samsung.NX100_Block) | available | not_found | not_found | not_applicable | not_found | svo_only |
| [Cameras/Sony](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Cameras/Sony.A330_Block) | available | not_found | not_found | not_applicable | available | mixed |
| [Cassini/ISS_NAC](sources/round2/gemini/report.md) | [source](https://pds-rings.seti.org/holdings/volumes/COISS_0xxx/COISS_0011/calib/efficiency/ccdqe.tab) | derived | available | available | not_applicable | available | original_files |
| [Cassini/ISS_WAC](sources/round2/gemini/report.md) | [source](https://pds-rings.seti.org/holdings/volumes/COISS_0xxx/COISS_0011/calib/efficiency/ccdqe.tab) | derived | available | available | not_applicable | available | original_files |
| [DENIS/DENIS](sources/round2/grok/report.md) | [SVO browse](https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=DENIS&gname2=DENIS) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [Envisat/MERIS](sources/round2/gemini/report.md) | [source](https://earth.esa.int/eogateway/documents/20142/37627/MERIS_NominalSRF_Model2004.xlsx/6be37b1e-8fae-d376-0933-d1195b5228a0) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Euclid/NISP](sources/round2/claude/report.md) | [source](https://euclid.esac.esa.int/msp/refdata/data/NISP-PHOTO-PASSBANDS-V1-Y_throughput.dat) | available | available | available | not_applicable | available | original_files |
| [Euclid/VIS](sources/round2/claude/report.md) | [source](https://euclid.esac.esa.int/msp/refdata/MISSION-REFERENCE-DATA-V1) | unknown | unknown | unknown | not_applicable | blocked | svo_only |
| [GAIA/GAIA3](sources/round2/claude/report.md) | [source](https://www.cosmos.esa.int/documents/29201/4226701/GaiaEDR3_passbands_zeropoints.zip/b03ab6ac-8b02-9850-7586-7dd7cdbc84c9?t=1603980987171) | documented | documented | documented | not_applicable | available | original_files |
| [GALEX/GALEX](sources/round2/claude/report.md) | [source](https://asd.gsfc.nasa.gov/archive/galex/Documents/PostLaunchFUVImagingResponseCurves.htm) | not_found | not_found | not_found | not_applicable | available | original_files |
| [GCPD/Cousins](sources/round2/gemini/report.md) | [source](https://arxiv.org/src/1412.1474/anc/Filters.ascii) | documented | documented | documented | not_applicable | available | mixed |
| [GCPD/Johnson](sources/round2/gemini/report.md) | [source](https://arxiv.org/src/1412.1474/anc/Filters.ascii) | documented | documented | documented | not_applicable | available | mixed |
| [GCPD/Stromgren](sources/round2/gemini/report.md) | [source](https://arxiv.org/src/1412.1474/anc/Filters.ascii) | documented | documented | documented | not_applicable | available | mixed |
| [GOES/ABI1](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/GOESR/docs/ABI-11-468A_PFM_Spectral_Response_Curves.xlsx) | available | available | derived | not_applicable | available | original_files |
| [GOES/ABI2](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/GOESR/docs/vendor_original_ABI_SRF_PFM_FM2_FM3_FM4.zip) | available | available | derived | not_applicable | available | original_files |
| [GOES/ABI3](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/GOESR/docs/vendor_original_ABI_SRF_PFM_FM2_FM3_FM4.zip) | available | available | derived | not_applicable | available | original_files |
| [GOES/ABI4](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/GOESR/docs/vendor_original_ABI_SRF_PFM_FM2_FM3_FM4.zip) | available | available | derived | not_applicable | available | original_files |
| [GTC/HIPERCAM](sources/round2/grok/report.md) | [source](https://raw.githubusercontent.com/StuartLittlefair/ucam_thruput/master/ucam_thruput/data/hcam_g.txt) | documented | documented | documented | documented | documented | original_files |
| [GTC/OSIRIS](sources/round2/grok/report.md) | [source](https://www.gtc.iac.es/instruments/osiris/media/tables/Sloang.txt) | available | not_found | documented | not_found | not_found | original_files |
| [Gemini/Flamingos2](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/flamingos2/) | available | documented | not_found | not_found | available | original_files |
| [Gemini/GMOS-N](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/gmos/filters/) | available | not_found | available | documented | not_found | original_files |
| [Gemini/GMOS-S](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/gmos/) | available | not_found | available | documented | not_found | original_files |
| [Gemini/GNIRS](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/nirs/filters/j_mk.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [Gemini/GPI](sources/round2/grok/report.md) | [source](https://github.com/geminiplanetimager/gpi_pipeline/blob/master/config/filters/GPI-filter-Y.fits) | available | not_found | not_found | available | not_found | original_files |
| [Gemini/GSAOI](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/gsaoi/) | available | not_found | not_found | documented | not_found | original_files |
| [Gemini/NIFS](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/nifs/) | available | not_found | not_found | not_found | not_found | original_files |
| [Gemini/NIRI](sources/round2/grok/report.md) | [source](https://www.gemini.edu/sciops/instruments/niri/) | available | not_found | not_found | documented | not_found | original_files |
| [Generic/Bessell](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Bessell.U) | documented | not_applicable | not_applicable | not_applicable | available | svo_only |
| [Generic/Bessell_JHKLM](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Bessell_JHKLM.J) | documented | not_applicable | not_applicable | documented | available | svo_only |
| [Generic/Cousins](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Cousins.R) | documented | not_applicable | not_applicable | not_applicable | available | svo_only |
| [Generic/Johnson](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson.U) | documented | not_applicable | not_applicable | not_applicable | available | svo_only |
| [Generic/Johnson_13col](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson_13col.33) | documented | not_applicable | not_applicable | not_applicable | available | svo_only |
| [Generic/Johnson_UBVRIJHKL](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Johnson_UBVRIJHKL.U) | documented | not_applicable | not_applicable | documented | available | svo_only |
| [Generic/Stromgren](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Generic/Stromgren.u) | documented | not_applicable | not_applicable | not_applicable | available | svo_only |
| [Generic/UCM](sources/round2/gemini/report.md) | [source](https://arxiv.org/e-print/2103.17009) | documented | not_applicable | not_applicable | not_applicable | available | original_files |
| [HST/ACS_HRC](sources/round2/claude/report.md) | [source](https://ssb.stsci.edu/trds/comp/acs/acs_f606w_hrc_008_syn.fits) | available | available | available | not_applicable | derived | original_files |
| [HST/ACS_SBC](sources/round2/claude/report.md) | [source](https://ssb.stsci.edu/trds/comp/acs/acs_f122m_007_syn.fits) | available | available | available | not_applicable | derived | original_files |
| [HST/ACS_WFC](sources/round2/claude/report.md) | [source](https://ssb.stsci.edu/trds/comp/acs/acs_f606w_wfc_008_syn.fits) | available | available | available | not_applicable | derived | original_files |
| [HST/WFC3_IR](sources/round2/claude/report.md) | [source](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/performance/throughputs/_documents/IR.zip) | available | available | available | not_applicable | available | original_files |
| [HST/WFC3_UVIS1](sources/round2/claude/report.md) | [source](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/performance/throughputs/_documents/UVIS.zip) | available | available | available | not_applicable | available | original_files |
| [HST/WFC3_UVIS2](sources/round2/claude/report.md) | [source](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/performance/throughputs/_documents/UVIS.zip) | available | available | available | not_applicable | available | original_files |
| [Herschel/Pacs](sources/round2/claude/report.md) | [source](https://arxiv.org/abs/1005.1487) | available | not_found | not_found | not_applicable | available | original_files |
| [Herschel/SPIRE](sources/round2/claude/report.md) | [source](https://arxiv.org/abs/1005.5123) | documented | not_found | not_found | not_applicable | available | original_files |
| [Himawari/AHI8](sources/round2/gemini/report.md) | [source](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/srf_201309/AHI-08_SpectralResponsivity.zip) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Himawari/AHI9](sources/round2/gemini/report.md) | [source](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/srf_201310/AHI-09_SpectralResponsivity.zip) | not_found | not_found | not_found | not_applicable | available | original_files |
| [INT/IPHAS](sources/round2/grok/report.md) | [source](https://astro.ing.iac.es/filter/list.php?instrument=WFC&sort=system) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [INT/UVEX](sources/round2/grok/report.md) | [source](https://astro.ing.iac.es/filter/list.php?instrument=WFC&sort=system) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [INT/WFC](sources/round2/grok/report.md) | [source](https://astro.ing.iac.es/filter/list.php?instrument=WFC&sort=system) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [JPSS/VIIRS](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/documents/documentation/NG_VIIRS_NPP_RSR_filtered_Oct2011_BA.zip) | not_found | not_found | not_found | not_applicable | available | original_files |
| [JPSS/VIIRS-J1](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/NOAA-20/docs/J1_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2.1_Nov2016.zip) | not_found | not_found | not_found | not_applicable | available | original_files |
| [JPSS/VIIRS-J2](sources/round2/gemini/report.md) | [source](https://ncc.nesdis.noaa.gov/NOAA-21/docs/J2_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2_Jul2019.zip) | not_found | not_found | not_found | not_applicable | available | original_files |
| [JWST/MIRI](sources/round2/claude/report.md) | [source](https://stsci.box.com/s/f1939jcr8i2um4kqcgehnnaad3ot9aje) | documented | unknown | unknown | not_applicable | documented | original_documentation |
| [JWST/NIRCam](sources/round2/claude/report.md) | [source](https://jwst-docs.stsci.edu/files/216457506/216457520/1/1762454205059/F200W_mean_system_throughput.txt) | blocked | blocked | blocked | not_applicable | not_found | svo_only |
| [JWST/NIRCam2025](sources/round2/claude/report.md) | [source](https://jwst-docs.stsci.edu/files/216457506/216457520/1/1762454205059/F200W_mean_system_throughput.txt) | blocked | blocked | blocked | not_applicable | available | original_files |
| [JWST/NIRISS](sources/round2/claude/report.md) | [source](https://jwst-docs.stsci.edu/files/216454094/216454098/1/1762450948325/niriss_filter_tarball.tar) | available | derived | derived | not_applicable | available | original_files |
| [JWST/NIRSpec](sources/round2/claude/report.md) | [source](https://jwst-docs.stsci.edu/files/216455634/216455641/1/1762452061447/jwst_nirspec_f110w_trans.fits) | available | blocked | blocked | not_applicable | blocked | mixed |
| [KPNO/DESI](sources/round2/grok/report.md) | [source](https://github.com/desihub/desimodel) | documented | documented | documented | not_found | documented | original_documentation |
| [KPNO/Mosaic](sources/round2/grok/report.md) | [source](https://noirlab.edu/science/programs/kpno/filters) | documented | not_found | not_found | not_found | not_found | original_documentation |
| [KPNO/MzLS](sources/round2/grok/report.md) | [source](https://www.legacysurvey.org/dr10/files/kpzd.txt) | available | documented | documented | documented | available | original_files |
| [Keck/LRIS](sources/round2/grok/report.md) | [source](https://www2.keck.hawaii.edu/inst/lris/data_files/g_blue.dat) | available | documented | not_found | not_found | not_found | original_files |
| [Keck/NIRC2](sources/round2/grok/report.md) | [source](https://www2.keck.hawaii.edu/inst/nirc2/) | documented | not_found | not_found | not_found | not_found | original_documentation |
| [Keck/NIRSPEC](sources/round2/grok/report.md) | [source](https://www2.keck.hawaii.edu/inst/nirspec/) | available | not_found | not_found | not_found | not_found | original_files |
| [Keck/OSIRIS](sources/round2/grok/report.md) | [source](https://www2.keck.hawaii.edu/inst/osiris/) | available | not_found | not_found | not_found | not_found | original_files |
| [Kepler/Kepler](sources/round2/claude/report.md) | [source](https://keplergo.github.io/KeplerScienceWebsite/data/kepler_response_hires1.txt) | documented | documented | documented | not_applicable | available | original_files |
| [LCO/CSP](sources/round2/grok/report.md) | [source](https://csp.obs.carnegiescience.edu/data/g_tel_ccd_atm_ext_1.2.dat) | documented | documented | documented | documented | available | original_files |
| [LCO/CSP2](sources/round2/grok/report.md) | [source](https://csp.obs.carnegiescience.edu/data/filters) | documented | documented | documented | documented | documented | original_documentation |
| [LCO/FourStar](sources/round2/grok/report.md) | [source](https://instrumentation.obs.carnegiescience.edu/FourStar/OPTICS/curves/J_cam_optics.txt) | not_found | available | not_found | not_found | not_found | original_files |
| [LCO/IMACS](sources/round2/grok/report.md) | [source](https://instrumentation.obs.carnegiescience.edu/imacs/) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [LCO/OGLE-IV](sources/round2/grok/report.md) | [source](https://www.astrouw.edu.pl/~jskowron/pub/ogle/OGLE-IV/OGLE-IV-I-filter-transmission.dat) | available | not_found | available | not_found | not_found | original_files |
| [LRO/Diviner](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=LRO&Instrument=Diviner) | documented | documented | documented | not_applicable | available | svo_only |
| [LRO/LROC_NAC](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=LRO/LROC_NAC.L) | not_applicable | documented | documented | not_applicable | available | svo_only |
| [LRO/LROC_WAC](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=LRO/LROC_WAC.320nm) | documented | documented | documented | not_applicable | available | svo_only |
| [LSST/LSST](sources/round2/grok/report.md) | [source](https://raw.githubusercontent.com/lsst/throughputs/main/baseline/version_info) | documented | documented | documented | documented | documented | svo_only |
| [LSST/LSSTv19](sources/round2/grok/report.md) | [source](https://raw.githubusercontent.com/lsst/throughputs/main/baseline/filter_g.dat) | available | available | available | available | available | original_files |
| [LaSilla/EFOSC](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/lasilla/instruments/efosc/inst/Efosc2Filters.html) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [LaSilla/GROND](sources/round2/grok/report.md) | [source](https://www.mpe.mpg.de/~jcg/GROND/GROND_filtercurves.txt) | available | unknown | unknown | unknown | unknown | original_files |
| [LaSilla/SOFI](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/lasilla/instruments/sofi.html) | available | not_found | not_found | not_found | not_found | original_files |
| [LaSilla/WFI](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/lasilla/instruments/wfi/inst/filters/842.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [Landsat/MSS-1](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L1_MSS_RSR.xls) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/MSS-2](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L2_MSS_RSR.xls) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/MSS-3](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L3_MSS_RSR.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/MSS-4](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L4_MSS_RSR.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/MSS-5](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L5_MSS_RSR.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/OLI](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_OLI_RSR.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Landsat/TIRS](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_TIRS_RSR.xlsx) | documented | documented | documented | not_applicable | available | original_files |
| [LasCumbres/LasCumbres](sources/round2/grok/report.md) | [source](https://lco.global/documents/2/lco_filter_curves.zip) | documented | not_found | not_found | not_found | not_found | original_documentation |
| [METOP/AVHRR-A](sources/round2/gemini/report.md) | [source](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_metop_2_avhrr_srf/rtcoef_metop_2_avhrr_srf.tar.gz) | not_found | not_found | not_found | not_applicable | available | original_files |
| [MRO/CTX](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=MRO/CTX.600nm_fil) | available | not_found | not_found | not_applicable | available | svo_only |
| [MRO/HiRISE](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=MRO&Instrument=HiRISE) | documented | documented | documented | not_applicable | available | svo_only |
| [Mars2020/MastCamZ](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=Mars2020&Instrument=MastCamZ) | documented | documented | documented | not_applicable | available | mixed |
| [Mars2020/WATSON](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Mars2020/WATSON.cutoff) | available | not_found | available | not_applicable | available | svo_only |
| [Misc/APASS](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=Misc/APASS.B) | documented | documented | documented | documented | available | svo_only |
| [Misc/Atlas](sources/round2/gemini/report.md) | [source](https://arxiv.org/e-print/1802.00879) | documented | documented | documented | documented | available | original_files |
| [OAJ/JPAS](sources/round2/grok/report.md) | [source](https://www.j-pas.org/survey/jpcam) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [OAJ/JPLUS](sources/round2/grok/report.md) | [source](https://www.j-plus.es/doc/general_page/10/JPLUS_gSDSS.dat) | available | documented | documented | documented | documented | original_files |
| [OSIRIS-REX/MapCam](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=OSIRIS-REX&Instrument=MapCam) | documented | documented | documented | not_applicable | available | mixed |
| [OSIRIS-REX/PolyCam](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=OSIRIS-REX&Instrument=PolyCam) | not_applicable | documented | documented | not_applicable | available | mixed |
| [OSIRIS-REX/SamCam](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=OSIRIS-REX&Instrument=SamCam) | documented | documented | documented | not_applicable | available | mixed |
| [PAN-STARRS/PS1](sources/round2/grok/report.md) | [source](https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/99/table3.dat) | documented | documented | documented | available | available | original_files |
| [Palomar/ZTF](sources/round2/grok/report.md) | [source](https://raw.githubusercontent.com/ZwickyTransientFacility/ztf_information/7c1d1ed1adeea47ee024272e797e77d7e65160d5/filter_transmission/ZTF_g_band_test_data_pos1_0AOI.csv) | available | not_found | available | not_found | documented | original_files |
| [Paranal/ERIS](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/eris/img/nix-filters.zip) | available | available | not_found | not_found | not_found | original_files |
| [Paranal/FORS1](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/u_HIGH.txt) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/FORS2](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/M_BESS_B.txt) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/HAWKI](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/hawki/inst/filters/hawki_Y.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/ISAAC](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/decommissioned/isaac/inst/filters/FJ_BB.ASCII) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/NACO](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/decommissioned/naco/inst.html) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/NGTS](sources/round2/grok/report.md) | [source](https://academic.oup.com/mnras/article/475/4/4476/4597999) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [Paranal/OmegaCAM](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/omegacam/tools/OmegaCAM_transmission.tar.gz) | available | not_found | documented | not_found | documented | original_files |
| [Paranal/SPHERE](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_Y.dat) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/VIMOS](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/vimos/inst/filters_2007-04-04/U_only_transmission_q1_to_q4.dat) | available | documented | documented | not_found | available | original_files |
| [Paranal/VISIR](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/instruments/visir/) | available | not_found | not_found | not_found | not_found | original_files |
| [Paranal/VISTA](sources/round2/grok/report.md) | [source](https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst/Filters_QE_Atm_curves.tar.gz) | available | available | available | available | documented | original_files |
| [Roman/WFI](sources/round2/claude/report.md) | [source](https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/main/roman_technical_information/data/WideFieldInstrument/Imaging/EffectiveAreas/Roman_effarea_v8_SCA01_20240301.ecsv) | documented | not_found | available | not_applicable | available | original_files |
| [SLOAN/SDSS](sources/round2/grok/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/index.php?id=SLOAN/SDSS.g) | documented | documented | documented | available | available | original_files |
| [SPHEREx/SPHEREx](sources/round2/claude/report.md) | [source](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/spectral_files_20260812/SPHERExSpectralFiles_202607.tar) | unknown | unknown | unknown | not_applicable | documented | original_documentation |
| [SeaStar/SeaWiFS](sources/round2/gemini/report.md) | [source](https://oceandata.sci.gsfc.nasa.gov/rsr_tables/images/rsr/seawifs_rsrs.txt) | documented | documented | documented | not_applicable | available | original_files |
| [Sentinel/MSI](sources/round2/gemini/report.md) | [source](https://landsat.usgs.gov/landsat/spectral_viewer/bands/Sentinel-2A%20MSI%20Spectral%20Responses.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Sentinel/OLCI-A](sources/round2/gemini/report.md) | [source](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_sentinel3_1_olci_srf/rtcoef_sentinel3_1_olci_srf.tar.gz) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Sentinel/OLCI-B](sources/round2/gemini/report.md) | [source](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_sentinel3_2_olci_srf/rtcoef_sentinel3_2_olci_srf.tar.gz) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Sentinel/SLSTR-A](sources/round2/gemini/report.md) | [source](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_sentinel3_1_slstr_srf/rtcoef_sentinel3_1_slstr_srf.tar.gz) | not_found | not_found | not_found | not_applicable | available | original_files |
| [Sentinel/SLSTR-B](sources/round2/gemini/report.md) | [source](https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_sentinel3_2_slstr_srf/rtcoef_sentinel3_2_slstr_srf.tar.gz) | not_found | not_found | not_found | not_applicable | available | original_files |
| [SkyMapper/SkyMapper](sources/round2/grok/report.md) | [source](https://skymapper.anu.edu.au/data-release/) | not_found | not_found | not_found | documented | documented | original_documentation |
| [Spitzer/IRAC](sources/round2/claude/report.md) | [source](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/201125ch1trans_full.txt) | documented | documented | documented | not_applicable | available | original_files |
| [Spitzer/MIPS](sources/round2/claude/report.md) | [source](https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt) | available | not_found | not_found | not_applicable | available | original_files |
| [Subaru/FOCAS](sources/round2/grok/report.md) | [source](https://subarutelescope.org/Instruments/FOCAS/camera/txt/B.txt) | available | not_found | not_found | not_found | not_found | original_files |
| [Subaru/HSC](sources/round2/grok/report.md) | [source](https://subarutelescope.org/Instruments/HSC/txt/wHSC-g.txt) | available | available | available | available | documented | original_files |
| [Subaru/IRCS](sources/round2/grok/report.md) | [source](https://subarutelescope.org/Instruments/IRCS/camera/txt/j.txt) | available | not_found | not_found | not_found | not_found | original_files |
| [Subaru/MOIRCS](sources/round2/grok/report.md) | [source](https://subarutelescope.org/Instruments/MOIRCS/IMGS/J1_convolved_1_aoi5_t120.txt) | available | not_found | not_found | not_found | not_found | original_files |
| [Subaru/Suprime](sources/round2/grok/report.md) | [source](https://subarutelescope.org/Archive/Instruments/SCam/txt/g.txt) | available | available | not_found | not_found | not_found | original_files |
| [Swift/UVOT](sources/round2/claude/report.md) | [source](https://heasarc.gsfc.nasa.gov/FTP/caldb/data/swift/uvota/bcf/ftrans/swuftrans20041120v102.fits) | available | available | documented | not_applicable | available | original_files |
| [TESS/TESS](sources/round2/claude/report.md) | [source](https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv) | documented | documented | documented | not_applicable | available | original_files |
| [Terra/MISR](sources/round2/gemini/report.md) | [source](https://svo2.cab.inta-csic.es/theory/fps/fps.php?Facility=Terra&Instrument=MISR) | documented | documented | documented | not_applicable | available | svo_only |
| [Terra/MODIS](sources/round2/gemini/report.md) | [source](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Terra_RSR_in-band.xlsx) | not_found | not_found | not_found | not_applicable | available | original_files |
| [UKIRT/UFTI](sources/round2/grok/report.md) | [source](https://about.ifa.hawaii.edu/ukirt/instruments/ukirt-fast-track-imager/) | not_found | not_found | not_found | not_found | not_found | original_documentation |
| [UKIRT/UKIDSS](sources/round2/grok/report.md) | [source](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/379/1599) | documented | documented | documented | documented | available | original_files |
| [UKIRT/WFCAM](sources/round2/grok/report.md) | [source](https://about.ifa.hawaii.edu/ukirt/instruments/wfcam/description-of-wfcam/) | documented | not_found | not_found | documented | available | original_files |
| [WISE/WISE](sources/round2/claude/report.md) | [source](https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/figures/RSR-W1.txt) | available | not_found | not_found | not_applicable | available | original_files |
