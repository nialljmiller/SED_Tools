# Grok round-2 component dossiers

Reference `sed-components-quarter-20261020`. Checked 2026-10-20T14:30:41Z.
Partition: ground observatories and surveys. Output is research only. No application code, production tables, gateway or other agent's files were changed. `results/grok.*` is preserved and converted here.

## How to read the statuses

A component is `available` only when numeric values for that role were read. `documented` means a page, filename or header states the role and the separate numbers were not read, or the role is included inside a combined product. `not_found` means this search did not retrieve it. It does not prove the curve does not exist. `file_verified` in `sources.jsonl` means the bytes were read. A representative file does not verify its siblings. Public HTTP access was not treated as a licence. `redistribution` is `not_reviewed` on every record. SVO `DetectorType` is a service label, not a detector convention for an original file. QE alone was not used to assign photon or energy weighting.

## Codex corrections accepted

C05 LSST: SVO `LSST/LSST` is version 1.5. The opened baseline files are version 1.9 and are recorded as `LSST/LSSTv19`. `atmos_std` is not reduced to one of the two phrases Codex found in the baseline README. `atmos_10` has no stated airmass, PWV or site. Builder tag v1.9.2 must not be written back onto the SVO 1.5 key.

C06 HSC: `r` and `r2`, and `i` and `i2`, stay separate. Negative measured samples stay. The pinned README URL that returned 404 is not cited as a retrieved file.

C07 VISTA: atmosphere wavelength is micrometres; hardware tables are nm and percent-like. `f02.sm` scales component wavelength for plotting. Supplied 1 mm and 3 mm atmospheres stay available. SkyCalc does not replace them.

C08 ZTF: the two e2v coating curves stay separate. Their average is derived. Unit headers are absent.

C10 2MASS: the total-only classification stays rejected. The VI.4a page was re-fetched and quotes the wavelength multiplication, the peak renormalization and the direct F_lambda integral.

C01-C04 and C09 belong to other partitions and were not edited.

## Shared retrieval rules

Rubin baseline files live in `lsst/throughputs` `baseline/`. Read the version line. Do not map `LSST/LSST` to that directory.

SDSS response files from the round-1 DR7 and Doi pages are the Sloan camera. APO `filters.apo.nmsu.edu` is a separate server and was not byte-compared.

2MASS component tables are the III.1.b HTML PRE tables. The RSR is the VI.4a product and is already wavelength-weighted.

PS1 numbers are CDS `J/ApJ/750/99` table3, fixed width, as the ReadMe describes. They include 1.2 airmasses.

ESO filter directories do not share one unit convention. Read each file. FORS, HAWK-I, SPHERE, VISIR, ERIS, VIMOS, ISAAC, NACO, SOFI and WFI were sampled, not exhausted. ERIS members inside one zip do not share units.

Gemini GMOS 2001/2005 transmission files are not on the same scale as the 2017 percent file. NIRI warm scans are not cold scans. F2 JH must not be read as microns. GPI Y has no TUNIT. The Gemini ATRAN filename `mktrans_zm_16_15` means Maunakea, 1.6 mm, airmass 1.5, and the local save of that file is truncated at 0.977 um.

UKIDSS totals are `http://www.ukidss.org/technical/photom/hewett-ukidss-tables.tar`. WFCAM narrow bands are not in tables 2-6.

Legacy Survey DR10 `bass-g`, `bass-r`, `kpzd`, `kpzdccdcorr3` and `decam.*.am1p4` are different products from NOIRLab Y3A2 and from each other. DR9 copies of the BASS and MzLS paths returned 404.

J-PLUS, J-PAS and S-PLUS are three cameras. J-PLUS full-efficiency atmosphere is an Apache Point model. S-PLUS CSV role is unlabeled. CSP-I Site2 g is not CSP-II and not V-LC3014.

Keck OSIRIS is not GTC OSIRIS. HiPERCAM in `ucam_thruput` is `hcam` plus `gtc`, not ULTRACAM or ULTRASPEC. The saved `hcam_g.txt` is short of the git blob.

## Instruments

### SLOAN/SDSS

SVO browse `SLOAN/SDSS` returned 10 ids, HTTP 200. First ids: SLOAN/SDSS.uprime_filter, SLOAN/SDSS.u, SLOAN/SDSS.g, SLOAN/SDSS.gprime_filter, SLOAN/SDSS.r, SLOAN/SDSS.rprime_filter, SLOAN/SDSS.i, SLOAN/SDSS.iprime_filter....
Statuses: filter documented; optics documented; QE documented; atmosphere available; published available; independent source original_files.
DR7 u.dat has airmass-1.3 point-source and extended columns; the Doi response file is CCD+filter+corrector+mirrors and excludes atmosphere. Separate filter, mirror and QE files were not opened. Do not equate these bytes with APO/SDSS.
Sources read for this key:
- published / file_verified / original: SDSS DR7 imager response tables. u.dat is 2291 bytes and has a header for five columns: wavelength; sensitivity through airmass 1.3 for a point source; through airmass 1.3 for an extended source; through airmass 0.0 for an extended source; atmospheric extinction through 1.0 airmasses. First data row 2980 0.0000 0.0000 0.0000 0.0727. The file does not spell the wavelength unit. Page column list: (1) wavelength; (2) on-the-sky sensitivity through 1.3 ...
- published / file_verified / original: Doi SDSS imager response tables, per camera column. Page, updated 2010-02-24: response of each CCD is CCD and filter and 1st corrector and mirrors, but no atmosphere. Format 'lambda(Angstrom) response'. Columns u1-u6, g1-g6, r1-r6, i1-i6, z1-z6. Opened g1.dat, 495 bytes, no header in the file, first row 3677.2 0.0039. Only g1 was opened. The Angstrom statement is the page, not the file.
- published / documented / original: SVO Filter Profile Service, SLOAN/SDSS.g and SDSS.gprime_filter. SDSS.g Description 'SDSS g full transmission', components Filter + Instrument + Atmosphere, DetectorType 1, ProfileReference http://www.sdss.org/dr7/instruments/imager/index.html, WavelengthRef 4702.50. gprime_filter Band is g and WavelengthRef is 4763.11, but Description is 'SDSS u' filter (filter only transmission) at USNO 40-in'. components Filter. ProfileReference http://filters.apo.nmsu.edu/curves/SDSS/. That US...

### APO/SDSS

SVO browse `APO/SDSS` returned 5 ids, HTTP 200. First ids: APO/SDSS.u, APO/SDSS.g, APO/SDSS.r, APO/SDSS.i, APO/SDSS.z.
Statuses: filter available; optics unknown; QE unknown; atmosphere unknown; published unknown; independent source original_files.
filters.apo.nmsu.edu g.dat was opened: no header, 3000-13000, peak about 0.92. Units and which components are included are not in the file. u,r,i,z on that server were not opened. Not byte-compared with SLOAN/SDSS.
Sources read for this key:
- filter / file_verified / original: APO filter curves g.dat. No header. n=10001 x=[3000,13000] y=[0,0.919102] peak 0.919102 at x=4991. Values span 3000 to 13000 with a peak near 0.92, which is consistent with Angstroms and a fraction, but the file does not say so. Not byte-compared with SLOAN/SDSS or with Doi et al.

### 2MASS/2MASS

SVO browse `2MASS/2MASS` returned 3 ids, HTTP 200. First ids: 2MASS/2MASS.J, 2MASS/2MASS.H, 2MASS/2MASS.Ks.
Statuses: filter available; optics available; QE available; atmosphere available; published available; independent source original_files.
Codex C10 accepted. VI.4a, re-read this pass, says the RSR is photon-counting after multiplying the QE component by wavelength, then peak-normalized, for direct F_lambda integration, with PLEXUS atmosphere. NICMOS3 DQE table 11 was fetched: xlam unlabeled, DQE about 0.43 at 0.50. Do not apply a second wavelength weight. Mirror table is one surface. The older QE row that says the table was not opened is the round-1 state; the tbl11 row supersedes it. The north atmosphere prefix supersedes the sentence that no atmosphere table was opened. The south table is still unopened, and the north full-file peak was not recomputed.
Sources read for this key:
- optics / file_verified / original: 2MASS All-Sky Explanatory Supplement section III.1.b1, main-mirror coating. Opened table title '2MASS MAIN MIRROR COATING REFLECTIVITY'. Columns n, xlam, yval. First row 1 1.000 0.9240. No unit word in the table. Parent page lists this as telescope mirror reflectivity and warns that the component products are not relative spectral response curves.
- filter / file_verified / original: 2MASS All-Sky Explanatory Supplement, J-band filter table. Opened table title '2MASS J-BAND FILTER TRANSMISSION'. Columns n, xlam, yval. First row 1 1.050 0.0000. No unit word. Parent page item 6 is J filter transmission. H and Ks filter tables are tbl8 and tbl10 and were not opened.
- qe / documented / original: 2MASS All-Sky Explanatory Supplement, NICMOS3 QE table. Parent page links 'NICMOS3 Detector Array Quantum Efficiency Data Table' to sec3_1b1.tbl11.html. The table file was not opened. Figure caption says it is a representative NICMOS3 array similar to those used in the 2MASS cameras.
- atmosphere / documented / original: 2MASS All-Sky Explanatory Supplement, site atmosphere tables. Page: atmospheric transmission for the north (Mt. Hopkins) and south (CTIO) sites, from the USAF PLEXUS code supplied by M. Cohen, including atomic and molecular absorption, Rayleigh scattering and site-specific mean aerosol scattering. Figure 16 is north, table tbl15. Figure 17 is south, table tbl16. Tables were not opened. A separate section shows the effect of 0.5 mm and 5.0 mm water vapour on J using ATRAN, table...
- published / documented / original: 2MASS total instrumental throughput without atmosphere. Page: figures 13-15 are the product of the relevant factors above, the total instrumental throughput of the camera plus telescope with no atmosphere. Tables tbl12 J, tbl13 H, tbl14 Ks. Not opened.
- published / documented / original: 2MASS total throughput including atmospheric absorption. Page: figures 21-24 are the product of the factors including atmospheric transmission. tbl19 J at 0.5 mm water, tbl20 J at 5.0 mm water, tbl21 H, tbl22 Ks. The page says in red that these are not relative spectral response curves suitable for synthetic photometry. Tables not opened.
- published / file_verified / original: 2MASS All-Sky Explanatory Supplement section VI.4.a relative spectral response. Opened J table title '2MASS J-band Relative Spectral Response Curve'. No column header. First rows 1.062 0. and 1.066 4.07068E-04. Page: Cohen et al. (2003 AJ 126 1090) determined photon-counting RSRs by combining the transmission and QE curves in section III.1.b1, peak-normalized, including atmospheric transmission modeled by PLEXUS. Quoted text: the curves are designed to be integrated directly over stellar spectra...
- qe / file_verified / original: 2MASS All-Sky explanatory supplement III.1.b table 11. Title: 2MASS NICMOS3 DETECTOR QUANTUM EFFICIENCY. Columns n, xlam, yval. First row n=1 xlam=0.500 yval=0.4335. xlam has no unit word. yval is labeled DQE in TITLEY. n=416 x=[0.5,2.575] y=[0,0.6586] peak 0.6586 at x=1.95
- published / file_verified / original: 2MASS VI.4a quoting Cohen et al. 2003 AJ 126 1090. Opened page quotes Cohen et al.: curves are designed to be integrated directly over spectra in F_lambda. The QE-based component was converted to photon-counting RSRs by multiplying by wavelength and renormalized. Peaks renormalized to unity. Atmosphere modeled by PLEXUS. Characteristic NICMOS III QE, not measured for each device.
- atmosphere / file_verified / original: 2MASS III.1.b table 15. Fetched prefix. Title text includes ATMOSPHERIC TRANSMISSION. First bytes: TITLE = ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE \CHAR TITLEY = TRANSMISSION |n  |xlam     |yval     | |i  |r        |r        |    1    1.0000    0.8948    2    1.0001    0.89

### PAN-STARRS/PS1

SVO browse `PAN-STARRS/PS1` returned 7 ids, HTTP 200. First ids: PAN-STARRS/PS1.g, PAN-STARRS/PS1.r, PAN-STARRS/PS1.w, PAN-STARRS/PS1.open, PAN-STARRS/PS1.i, PAN-STARRS/PS1.z, PAN-STARRS/PS1.y.
Statuses: filter documented; optics documented; QE documented; atmosphere available; published available; independent source original_files.
CDS J/ApJ/750/99 table3 and the ReadMe were fetched HTTP 200. Bands are capture cross-sections in m2, note says m^2^-e/photon, including 1.2 airmasses. Aero, Ray and Mol are separate. Not bare filters. Field-angle figures are not in the table. The empty IOP URL remains a failed path.
Sources read for this key:
- published / documented / original: STScI MAST Pan-STARRS documentation, PS1 filter properties. Page: grizy and w bandpasses for a standard airmass of 1.2. Mean wavelengths listed as g 4866, r 6215, i 7545, z 8679, y 9633 A. w spans gri and does not have full 3-pi coverage. Citations on the page: Chambers et al. 2016, Tonry et al. 2012, Stubbs et al. 2010, Scolnic et al. 2015. The page says a link to the overall throughputs in the Tonry figure is at Vizier and also a machine-readable ASCII file. Vizier was not ...
- published / blocked / original: IOP machine-readable table linked from the PS1 filter page. HTTP 200 with Content-Length 0 and an empty body. Final URL https://iopscience.iop.org:443/0004-637X/750/2/99/suppdata/apj425122t3_mrt.txt. Not a verified table.
- published / file_verified / original: Tonry et al. 2012 CDS J/ApJ/750/99 table3.dat. ReadMe: lambda nm bytes 1-4; Open and grizyw_P1 are m2 capture cross-sections. Note (1) says including nominal 1.2 airmasses and units m^2^-e/photon. Aero, Ray, Mol are separate transmissions of that atmosphere with unit '---'. Opened file starts '380 0.096'. n=721 x=[380,1100] y=[0.003,0.949] peak 0.949 at x=752
- atmosphere / file_verified / original: Tonry et al. 2012 table3 atmosphere columns and table4. Note (2): of the nominal 1.2 airmasses of the Pan-STARRS1 standard atmosphere. table4 lists extinction at 1.2 airmasses per filter, for example g_P1 Ext 0.22. No PWV, no model version name.

### CTIO/DECam

SVO browse `CTIO/DECam` returned 19 ids, HTTP 200. First ids: CTIO/DECam.u_filter, CTIO/DECam.u, CTIO/DECam.u_old, CTIO/DECam.g_filter, CTIO/DECam.g_old, CTIO/DECam.g, CTIO/DECam.VR_filter, CTIO/DECam.r_filter....
Statuses: filter documented; optics unknown; QE unknown; atmosphere documented; published available; independent source original_files.
Two products: round-1 Y3A2 FITS at airmass 1.2 with an atm column, and DR10 decam.g.am1p4.dat.txt at airmass 1.4. Do not merge them. A no-atmosphere curve is a division, not a file. The Asahi link was not a curve file. Only g was opened in the DR10 set.
Sources read for this key:
- published / file_verified / original: NOIRLab DECam DES standard bandpass FITS, Y3A2_20170715. The .txt URL is a FITS file, 83520 bytes. Primary HDU NAXIS 0. Extension is a 1441-row table, NAXIS1 52. TTYPE1 LAMBDA (TFORM D), TTYPE2 g, TTYPE3 r, TTYPE4 i, TTYPE5 z, TTYPE6 Y (those TFORM D), TTYPE7 atm (TFORM E). No TUNIT cards. Page text: version Y3A2_20170715; grizY are total system response (instrument and atmosphere) at airmass 1.2, including the average instrumental response across the science CCDs; respons...
- published / documented / original: NOIRLab instruction to divide the DES standard grizY columns by atm. Page: 'For system throughput without the atmosphere, divide the grizY column by the atm column.' The quotient is not a separate file. The page calls the result system throughput without atmosphere. It does not separate filter, optics and QE.
- filter / documented / original: NOIRLab link to Asahi area-weighted DECam filter transmission. Page link text: 'Filter transmission (area weighted response) from Asahi for u,g,r,i,z,Y and VR', with a plot at https://noirlab.edu/science/images/decamfilterstransmission. Fetch of the document URL returned a Drupal HTML shell, not a numeric table. The table itself was not opened.
- published / documented / original: NOIRLab DECam Science Verification u total-throughput table. Page: total throughput for u comes from the DECam Science Verification phase (credit William Wester). The document URL returned a Drupal HTML shell, not the table.
- published / documented / original: NOIRLab DECam standard-bandpass ASCII, README, legacy table, and medium/narrow workbook. Page also links README https://noirlab.edu/science/documents/scidoc0433 and legacy ugrizY measurements https://noirlab.edu/science/documents/scidoc0473. Those three URLs returned Drupal HTML shells. Medium and narrow-band workbook https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc3138.xls is linked and was not opened. Legacy note on the page: provided for legacy use; DR1 values are the de...
- published / documented / original: SVO Filter Profile Service, CTIO/DECam. DECam.g Description 'DECam g filter (total transmission). DR1'. Comments say the system response includes instrument and atmosphere. components Filter + Instrument + Atmosphere. DetectorType 1. DECam.g_filter Description 'DECam g filter (filter only)', components Filter. DECam.u Description 'DECam u filter (total transmission). DR1'. ProfileReference on these is the old CTIO DECam page. TABLEDATA not used as the prim...
- published / file_verified / original: Legacy Survey DR10 decam.g.am1p4.dat.txt. Header: DECam system throughput from the DECaLS wiki, source DECamFilters-14apr18.xslx, airmass 1.40000, g band. Wavelength unlabeled. n=736 x=[3000,10350] peak 0.363596 at 5440 and the grid 3000 to 10350 step 10 is complete in the saved file. This is not the NOIRLab Y3A2 scidoc1884 FITS product, which is airmass 1.2 and has an atm column.

### LSST/LSSTv19

SVO browse `LSST/LSSTv19` returned 4 ids, HTTP 200. First ids: LSST/LSSTv19.g_filter, LSST/LSSTv19.g, LSST/LSSTv19.i_filter, LSST/LSSTv19.i.
Statuses: filter available; optics available; QE available; atmosphere available; published available; independent source original_files.
Codex C05 accepted. Baseline files are version 1.9, not SVO LSST version 1.5. atmos_std is called both the 1.2-airmass fiducial and zenith in the README Codex opened. atmos_10 states no airmass, PWV or site. Dat headers are nm and throughput 0-1 and do not state photon versus energy.
Sources read for this key:
- filter / file_verified / original: Rubin Observatory lsst/throughputs baseline filter curves. filter_g.dat header: 'LSST Throughputs files created from syseng_throughputs repo', 'Version 1.9', sha1 fcc05772f99427e4a45cd1b9da1628dded9a06d5, 'Wavelen_cutoff_BLUE 385.60', 'Wavelen_cutoff_RED 566.30', 'Wavelength(nm) Throughput(0-1)'. Content-Length 326772. GitHub contents API lists filter_u/g/r/i/z/y.dat (sizes 326100, 326772, 329622, 326526, 324208, 326427). Only filter_g was opened. The same header and Content...
- optics / file_verified / original: Rubin Observatory lsst/throughputs baseline mirror and lens curves. README: m1.dat, m2.dat, m3.dat are mirror throughputs; lens1.dat, lens2.dat, lens3.dat are lens throughputs; all curves nm and throughput 0-1. Opened m1.dat and lens1.dat: same Version 1.9 / sha1 fcc05772 header and 'Wavelength(nm) Throughput(0-1)'. Contents API sizes: m1 306886, m2 306970, m3 306886, lens1 307336, lens2 307353, lens3 307262. m2, m3, lens2, lens3 not opened. syseng README Release 1.9: coatings move f...
- qe / file_verified / original: Rubin Observatory lsst/throughputs baseline detector.dat. README: 'detector.dat is the current detector sensitivity.' Opened file header is Version 1.9, sha1 fcc05772, columns 'Wavelength(nm) Throughput(0-1)'. Content-Length 304547. No photon/energy statement in the header. syseng Release 1.3 note, which this README also carries, says the generic detector was the minimum QE versus wavelength of ITL and e2v and that vendor curves are QE+AR. That sentence was not re-checked a...
- atmosphere / file_verified / original: Rubin Observatory lsst/throughputs baseline atmos_std.dat. File header adds 'Aerosols added to atmosphere' and 'Wavelength(nm) Throughput(0-1)'; it does not state airmass. README: 'atmos_std.dat is a fiducial atmosphere throughput at Rubin at 1.2 airmasses.' The same README later says total curves include 'the zenith atmos_std.dat atmosphere.' Content-Length 307669. main and v1.9.2 both returned that length.
- atmosphere / file_verified / original: Rubin Observatory lsst/throughputs baseline atmos_10.dat. Opened header: Version 1.9, sha1 fcc05772, 'Aerosols added to atmosphere', 'Wavelength(nm) Throughput(0-1)'. Content-Length 315285. The file does not state an airmass. The baseline README defines airmass only for atmos_std.dat.
- atmosphere / file_verified / original: Rubin lsst/throughputs atmos/ directory, 2010 David Burke MODTRAN set. README title 'v1.0 atmospheres from David Burke, MODTRAN, March 1, 2010.' Files without _aerosol have no aerosols; *_aerosol.dat include aerosols. 'conform to typical conditions expected at LSST's site at Cerro Pachon.' Airmass is the number in the filename: atmos_10.dat is 1.0, atmos_25.dat is 2.5. hiwater/lowater are 30 percent water scalings and the README says they should be redone. Individual atmos/*.dat curves ...
- published / file_verified / original: Rubin Observatory lsst/throughputs baseline hardware curves. README: 'hardware_*.dat curves are the hardware only, without atmospheric component.' hardware_g.dat header is Version 1.9, sha1 fcc05772, g cutoffs BLUE 385.60 RED 566.30, 'Wavelength(nm) Throughput(0-1)'. No component breakdown inside the file. Contents API lists hardware_u/g/r/i/z/y.dat. Only g was opened.
- published / file_verified / original: Rubin Observatory lsst/throughputs baseline total curves. README: total_*.dat is mirrors, lenses, filter, detector, and the zenith atmos_std.dat atmosphere, compiled with an X=1.2 atmosphere. total_g.dat header includes 'Aerosols added to atmosphere', the g cutoffs, and 'Wavelength(nm) Throughput(0-1)'. Contents API lists total_u/g/r/i/z/y.dat. Only g was opened.
- published / documented / svo_fallback: SVO Filter Profile Service, LSST/LSSTv19. Saved browse lists only LSST/LSSTv19.g, g_filter, i, i_filter. Opened g and i VOTable headers. g Description 'LSST full throughput. Version 1.9'. Comments 'Full throughput: filter+telescope+detector+atmosphere (@ airmass 1.2) Version 1.9'. components 'Filter + CCD + Instrument + Other optical components + Atmosphere'. i has the same full-throughput comment. WavelengthUnit Angstrom. ProfileReference https://github.com...
- published / documented / original: LSST systems engineering lsst-pst/syseng_throughputs. README: this repository is the ultimate source of the curves in lsst/throughputs. components/ holds each camera and telescope component; *_Losses holds time-averaged ten-year losses; some directories have *_Coatings. Release 1.9 text: same components as previous versions but mirror coatings move from Al-Ag-Al to Ag-Ag-Ag. Individual component files were not opened. LICENSE URLs on throughputs and syseng main returned...

### LSST/LSST

SVO browse `LSST/LSST` returned 12 ids, HTTP 200. First ids: LSST/LSST.u_filter, LSST/LSST.u, LSST/LSST.g_filter, LSST/LSST.g, LSST/LSST.r_filter, LSST/LSST.r, LSST/LSST.i, LSST/LSST.i_filter....
Statuses: filter documented; optics documented; QE documented; atmosphere documented; published documented; independent source svo_only.
SVO comments say version 1.5, full system plus zenith atmosphere. That is not the v1.9 baseline. lsst.py writes LSST/LSST ids from v1.9.2; do not follow that mapping. Use LSSTv19 for the opened v1.9 files.
Sources read for this key:
- published / documented / svo_fallback: SVO Filter Profile Service, LSST/LSST.g. VOTable PARAM header: Description 'LSST g filter (full)'. Comments: 'Full throughput: system - mirrors, lenses, filter, detector, and the zenith atmosphere. Aerosols added to atmosphere. Version 1.5'. components 'Filter + Instrument + Atmosphere'. WavelengthUnit Angstrom. ProfileReference https://github.com/lsst/throughputs/blob/master/baseline/README.md. DetectorType value 1. TABLEDATA truncated at 80 KB and not use...
- filter / documented / svo_fallback: SVO Filter Profile Service, LSST/LSST.g_filter. VOTable PARAM header: filterID LSST/LSST.g_filter, Band g, WavelengthRef 4746.34, wavelength range in the table starts near 3864 A. Description is the mismatched string 'LSST u filter (filter only)'. Comments 'Version 1.5'. The components field is 'Filter + Instrument + Atmosphere', which contradicts both the description and a filter-only reading.
- published / documented / original: Rubin lsst/throughputs baseline index cited by the SVO profile reference. version_info file content is the four bytes '1.9\n'. The SVO LSST/LSST.g comment says Version 1.5 while its ProfileReference points at the baseline README, which now describes the current modeled Rubin curves. These are not established as the same bytes.

### Subaru/HSC

SVO browse `Subaru/HSC` returned 23 ids, HTTP 200. First ids: Subaru/HSC.NB387_filter, Subaru/HSC.NB468_filter, Subaru/HSC.g_filter, Subaru/HSC.g, Subaru/HSC.NB515_filter, Subaru/HSC.NB527_filter, Subaru/HSC.r2_filter, Subaru/HSC.r_filter....
Statuses: filter available; optics available; QE available; atmosphere available; published documented; independent source original_files.
Codex C06 accepted. r and r2 differ in sampling and values; do not upgrade r to r2 or i to i2. A negative r sample stays. Area-weighted and center curves differ. QE header is Angstrom and fraction. The pinned transmission README returned 404 in the Codex audit.
Sources read for this key:
- filter / file_verified / original: Subaru Telescope HSC sensitivity page, area-weighted and center filter lists. Page table columns include Center and Area-weighted mean. Each broad and user filter has txt/HSC-<name>.txt and txt/wHSC-<name>.txt. Opened wHSC-g.txt and HSC-g.txt: header '# wavelength[angstrom]  transmittance', values are fractions. At 3580 A, wHSC-g is -0.0004. wHSC-g and HSC-g differ at the same wavelength. wHSC-r.txt starts at 7500.0 and decreases; wHSC-r2.txt starts at 5001.9. They are different filters. Other...
- qe / file_verified / original: Subaru Telescope HSC CCD QE table. File is 603 bytes. Header '# wavelength[angstrom]  QE'. First row 3600.0 0.285. Values are fractions, not percent.
- optics / file_verified / original: Subaru Telescope HSC PFU and dewar-window transmission. Both files header '# wavelength[angstrom]  transmittance'. throughput_popt2.txt starts 3700.0 0.755 (1382 bytes). throughput_win.txt starts 3300.0 0.900 (1366 bytes). obs_subaru README: popt2 is transmission of the Primary Focus Unit; win is the HSC dewar window; both assumed spatially constant. Window file: https://subarutelescope.org/Instruments/HSC/txt/throughput_win.txt
- optics / file_verified / original: Subaru primary-mirror reflectivity, 2010 recoating, copy in lsst/obs_subaru. Header '#Wavelength(nm)\tReflectivity\tStddev-of-measurements' then 2600 0.9683 0.0046. Three columns. README in that commit: mirror reflectivity as of the 2010 recoating, assumed spatially constant, downloaded from the NAOJ reflectivity path. The commit's wHSC copies are narrow-band only; broadband wHSC-g is on the Subaru site, not in this directory. filterTraces.py is described as private-communication traces and w...
- atmosphere / file_verified / original: HSC pipeline MODTRAN table in lsst/obs_subaru. README: 'MODTRAN model for the atmospheric absorption generated by Naoki Yasuda, assuming airmass 1.2 and PWV=1.5mm.' The file has no column header. First rows 3000.0 0.0203, 3010.0 0.0301, 3020.0 0.0415. The filename contains binned10ang. The README does not state the wavelength unit or a site altitude. The public HSC sensitivity page does not host this atmosphere file.
- published / documented / original: SVO Filter Profile Service, Subaru/HSC. Opened headers: HSC.g Description 'HSC g (full transmission)', components 'Filter + Instrument + Atmosphere', ProfileReference http://cosmos.astro.caltech.edu/page/filterset, DetectorType 1. HSC.g_filter Description 'HSC g (filter only)', components Filter, ProfileReference the NAOJ sensitivity page, DetectorType 1. HSC.r is full transmission and is distinct from r2. HSC.r2_filter Description 'HSC r2 (filter only)', ...

### Subaru/Suprime

SVO browse `Subaru/Suprime` returned 34 ids, HTTP 200. First ids: Subaru/Suprime.IB427, Subaru/Suprime.B_filter, Subaru/Suprime.B, Subaru/Suprime.IB464, Subaru/Suprime.g_filter, Subaru/Suprime.g, Subaru/Suprime.IB484, Subaru/Suprime.IB505....
Statuses: filter available; optics available; QE not_found; atmosphere not_found; published not_found; independent source original_files.
Archive g.txt has no header, peak about 0.94, Angstrom-like. One band. PFU_transmission.txt is the telescope prime-focus unit in Angstrom, not a Suprime filter and not the HSC window. Do not copy HSC curves here.
Sources read for this key:
- filter / file_verified / original: Suprime-Cam archive g.txt. No header. n=200 x=[3737.7,5677] peak 0.93785 at 4838.9. Values are consistent with Angstroms and a fraction. The file does not say so. Some older SCam URLs now land on the Japanese homepage; this Archive URL returned the numeric file.
- optics / file_verified / original: Subaru PFU_transmission.txt. Header: wavelength(A), transmittance. n=65 x=[3600,10000] peak 0.94 at 5800. This is the telescope PFU, shared by instruments that use that focus. It is not an HSC dewar window and not a Suprime filter.

### CFHT/MegaCam

SVO browse `CFHT/MegaCam` returned 12 ids, HTTP 200. First ids: CFHT/MegaCam.u, CFHT/MegaCam.u_1, CFHT/MegaCam.g, CFHT/MegaCam.g_1, CFHT/MegaCam.gri, CFHT/MegaCam.r_1, CFHT/MegaCam.r, CFHT/MegaCam.i_2....
Statuses: filter available; optics not_found; QE available; atmosphere not_found; published documented; independent source original_files.
Round-1 files: filter table with u*g'r'i'z' and two QE files in nm and percent, one a mean with a scatter column. Three filter generations are documented. Atmosphere was not opened.
Sources read for this key:
- qe / file_verified / original: CFHT MegaCam mean QE table. Full file, 504 bytes. Header 'lambda (nm)       mean (%)   stdev (%)'. First row 350 48.9 6.2. This is an average with a standard-deviation column.
- qe / file_verified / original: CFHT e2v CCD42-90 QE model. Header 'lambda (nm)  mean (%)'. First row 280.000 22.0000. Content-Length 2136. A model, distinct from MegaCam_QE_data.txt. The rest of the file past the first rows was not interpreted.
- filter / file_verified / original: CFHT MegaCam filter table. Header row is 'wl u* g' i' r' z'' with extra blank columns. 401 rows, first column 300..1100, stored from the red end. Values include negatives (u* at the red end is positive and small; the table range of the numeric columns reaches about -0.85 to 78.95). No unit word in the header. Column order is u*, g', i', r', z' with i' before r'.
- published / documented / original: CADC MegaPipe MegaCam filter page. Page: three generations of MegaCam broad-band filters (2002; 2007 i replacement; 2015 larger filters that use the four wing CCDs). 'The filter responses shown include the full telescope (mirror+optics+CCD) response.' Mirror, optics and CCD curves are taken from Betoule et al. (2013). 'The filter curves also include 1.25 airmasses of atmospheric attenuation.' Responses shown are the average of measurements at multiple...

### CFHT/Megaprime

SVO browse `CFHT/Megaprime` returned 44 ids, HTTP 200. First ids: CFHT/Megaprime.u_fil, CFHT/Megaprime.u, CFHT/Megaprime.uS_fil, CFHT/Megaprime.uS, CFHT/Megaprime.CaHK, CFHT/Megaprime.CaHK_fil, CFHT/Megaprime.g_fil, CFHT/Megaprime.g....
Statuses: filter documented; optics available; QE unknown; atmosphere not_found; published available; independent source original_files.
Opened Megaprime curves have no headers. One peaks near 87 and one near 96. The files do not say which column is filter, optics or a full response. SVO text is not a substitute for that label.
Sources read for this key:
- published / file_verified / original: CFHT MegaPrime transmission table. No header. 801 rows, first column 300..1100, second about 1.7e-5..87.24. First row 300 1.71793E-05. The file does not say what product the second column is.
- optics / file_verified / original: CFHT primary transmission table in the MegaPrime data directory. No header. 801 rows, first column 300..1100, second 86.37..95.7. First row 300 92.3. Unit not stated.
- filter / documented / original: SVO Filter Profile Service, CFHT/Megaprime.g and g_fil. Megaprime.g Description 'Megaprime g', cfh9402 (full)'. Comments 'New filter. Since 10/21/2014'. components 'Filter + CCD + Instrument + Other optical components' with no atmosphere term. WavelengthRef 4741.49. g_fil Description 'Megaprime g', cfh9402 (filter only)', components Filter, same date comment, ProfileReference http://www.cfht.hawaii.edu/Instruments/Filters/megaprime.html, WavelengthRef 4734.90. The filter-...

### Paranal/VISTA

SVO browse `Paranal/VISTA` returned 16 ids, HTTP 200. First ids: Paranal/VISTA.Z_filter, Paranal/VISTA.Z, Paranal/VISTA.NB980_filter, Paranal/VISTA.NB980, Paranal/VISTA.NB990_filter, Paranal/VISTA.NB990, Paranal/VISTA.Y_filter, Paranal/VISTA.Y....
Statuses: filter available; optics available; QE available; atmosphere available; published documented; independent source original_files.
Codex C07 accepted. Hardware is nm and percent-like; atmosphere files are already in micrometres with transmission near 1, at 1 mm/X=1 and 3 mm/X=2. f02.sm divides component wavelength by 1000. SkyCalc is not the supplied atmosphere. QE is not per device in the opened table.
Sources read for this key:
- filter / file_verified / original: ESO VISTA/VIRCAM ETC filter, QE and atmosphere bundle. Tar is 448951 bytes, 19 members, no README. Broad-band members are VISTA_Filters_at80K_forETC_{Z,Y,J,H,Ks}.dat.gz. NB members are VISTA_Filters_forETC_{NB118,NB980,NB990}.dat.gz and the NB filenames do not contain at80K. J file has no header: 2176 rows, first column 500..3200, second -0.123..88.648, first row 500.00 0.0001. NB118: 201 rows, first column 1100..1300, second -0.1..67.195. Inst page says the tar contains...
- optics / file_verified / original: ESO VISTA/VIRCAM ETC mirror reflectivity files. Members VISTA_M1_Reflectivity_forETC_2009-Sep12.txt.gz and VISTA_M2_Reflectivity_forETC_2007-Jun19.txt.gz. No headers. M1: 1601 rows, first column 300..3000, second 3.84..98.52, first row 300.0 15.75. M2 first row 300.0 15.54, second column 4.33..98.57. Dates are the filenames. f02.sm reads column 1 and 2, divides wavelength by 1000, and plots on the percent axis.
- qe / file_verified / original: ESO VISTA/VIRCAM ETC qe.tab. qe.tab.gz has no header. 2724 rows, first column 806..3529, second 0.480261..94.762, first row 806 68. Inst page calls this the detector quantum efficiency. f02.sm divides the wavelength column by 1000 and plots it with the percent-scaled curves. No counting convention in the file.
- atmosphere / file_verified / original: ESO VISTA/VIRCAM ETC atmosphere tables inside the same tar. Only two atmosphere members. trans_10_10.dat.gz first line is '1.0mm H2O, airmass=1.0'; 9400 rows, first column 0.9..5.6, second 0..1, first data row 0.900000 0.973927. trans_30_20.dat.gz first line is '#3.0mm H2O, airmass=2.0'; second column also 0..1. f02.sm does not divide the atmosphere wavelength by 1000; it multiplies the transmission column by 100 before plotting. Wavelength unit is not in the comment line.
- published / documented / original: SVO Filter Profile Service, Paranal/VISTA.J. VISTA.J Description 'VISTA J filter (full)'. Comments: includes mirrors (M1, M2), ccd QE and atmosphere model (1.0). components Filter + Instrument + Other optical components + Atmosphere. VISTA.J_filter Description 'VISTA J filter (filter only)', components Filter. Wavelength unit of the SVO profile is Angstrom. Not byte-compared with the tar.

### UKIRT/WFCAM

SVO browse `UKIRT/WFCAM` returned 12 ids, HTTP 200. First ids: UKIRT/WFCAM.Z_filter, UKIRT/WFCAM.Z, UKIRT/WFCAM.Y, UKIRT/WFCAM.Y_filter, UKIRT/WFCAM.J, UKIRT/WFCAM.J_filter, UKIRT/WFCAM.H, UKIRT/WFCAM.H_filter....
Statuses: filter documented; optics not_found; QE not_found; atmosphere documented; published available; independent source original_files.
Hewett system tables 2-6 are the corrected ZYJHK totals at 1.0 mm water and 1.3 airmass. They are not the room-temperature scans and not H2 or Brackett-gamma. The numeric room-temperature file was not retrieved. IfA says those scans are not corrected for 120 K or 10 degrees.
Sources read for this key:
- published / documented / svo_fallback: SVO Filter Profile Service, UKIRT/WFCAM.J, citing Hewett et al. 2006. Description 'WFCAM J filter (total transmission)'. Comments: used for UKIDSS; total transmission includes cooling to 120 K, the 10-degree angle of incidence, quantum efficiency, 1.0 mm water vapour and 1.3 airmass. components Filter + Instrument + Atmosphere. ProfileReference http://adsabs.harvard.edu/abs/2006MNRAS.367..454H. WavelengthUnit Angstrom. WavelengthRef 12483. The paper PDF was not opened. Browse lists Z/Y...
- filter / documented / svo_fallback: SVO Filter Profile Service, UKIRT/WFCAM.J_filter. Description 'WFCAM J filter (filter only)'. Comments: measured at room temperature and not corrected for the wavelength shifts from cooling to 120 K or the 10-degree angle of incidence. components Filter. ProfileReference https://www.ukirt.hawaii.edu/instruments/wfcam/user_guide/description.html. That URL timed out. WavelengthRef 12546.98.
- filter / blocked / original: University of Hawaii IfA WFCAM description page. First fetch failed TLS certificate verification. A retry returned HTTP 200 and a 20000-byte HTML slice titled 'Description of WFCAM - UKIRT'. That slice contains no transmission table, no Cut-On text, and no curve-file href. The older ukirt.hawaii.edu user-guide URL timed out.
- published / file_verified / original: Hewett et al. 2006 tables 2-6. IfA description page links per-filter HTML and says room-temperature filter data are not corrected for 120 K, 10-degree incidence or QE, and points to Hewett et al. for the corrected total. The j-filter HTML did not expose a numeric text URL. n=798 x=[1.051,1.4495] y=[0,0.2137] peak 0.2137 at x=1.2855 Re-fetched http://www.ukidss.org/technical/photom/hewett-ukidss-tables.tar on this date: HTTP 200, 189952 bytes, sha2...

### UKIRT/UKIDSS

SVO browse `UKIRT/UKIDSS` returned 5 ids, HTTP 200. First ids: UKIRT/UKIDSS.Z, UKIRT/UKIDSS.Y, UKIRT/UKIDSS.J, UKIRT/UKIDSS.H, UKIRT/UKIDSS.K.
Statuses: filter documented; optics documented; QE documented; atmosphere documented; published available; independent source original_files.
The five SVO bands match Hewett tables 2-6, downloaded as hewett-ukidss-tables.tar HTTP 200. Wavelength microns, throughput, 1.0 mm, 1.3 airmass. Components are not separate columns. CDS FTP 404.
Sources read for this key:
- published / file_verified / original: Hewett, Warren, Leggett and Hodgkin, MNRAS 2006, local CDS-style tables. Local tar README names the paper and says files follow the table numbers. Table 2 header: System throughput curve for the WFCAM Z filter, 1.0 mm water vapour, 1.3 airmass, wavelength microns, throughput. Tables 3-6 are Y, J, H and K with the same conditions. Measured n=461 x=[0.8,1.03] y=[0,0.1906] peak 0.1906 at x=0.915 for Z. Public FTP Table02 and the ReadMe returned HTTP 404 on 2026-10-20. The VizieR catalog URL ...

### Palomar/ZTF

SVO browse `Palomar/ZTF` returned 6 ids, HTTP 200. First ids: Palomar/ZTF.g, Palomar/ZTF.g_fil, Palomar/ZTF.r, Palomar/ZTF.r_fil, Palomar/ZTF.i, Palomar/ZTF.i_fil.
Statuses: filter available; optics not_found; QE available; atmosphere not_found; published documented; independent source original_files.
Codex C08 accepted. Keep single-layer and multi-layer QE separate. An average is derived. Vendor g reaches about 94 and the QE files stay below 1; CSVs have no unit header. SVO g names the multi-layer CCD. No optics or site atmosphere file was in the opened tree.
Sources read for this key:
- filter / file_verified / original: Zwicky Transient Facility ztf_information vendor filter tables. README: 'Vendor-measured on-axis filter transmission functions are in filter_transmission.' No column header in the CSVs. Full g file, 46370 bytes: 2593 rows, first column 367.6..564.8, second 0.04691..93.7918098. Full i file ztf_i_band.csv, 61943 bytes: 4001 rows, first column 300..1100, second about 0..99.9883, including a tiny negative. r file was fetched but only the start was scanned (first column near 549.76 in...
- qe / file_verified / original: Zwicky Transient Facility ztf_information e2v QE tables. README: theoretical and vendor CCD quantum-efficiency measurements are in QE. Northernmost and southernmost eight CCDs have a single-layer AR coating; the inner eight have a multi-layer coating. No headers. Full single-layer file: 230 rows, first column 301.65..1098.19, second 0.000311..0.91512. Full multi2 file: 180 rows, first column 301.68..1038.49, second 0.01351..0.92567. Multi URL: https://raw.githubusercontent...
- published / documented / original: SVO Filter Profile Service, Palomar/ZTF.g and ZTF.g_fil. ZTF.g Description 'ZTF g filter (+ccd)'. Comments 'Includes CCD QE with a a multi-layer coating.' components Filter + CCD. DetectorType 1. ZTF.g_fil Description 'ZTF g filter (filter only)'. Comments 'Vendor-measured on-axis filter transmission function'. components Filter. ProfileReference is the ztf_information filter_transmission tree. Not byte-compared with the CSVs.

### SkyMapper/SkyMapper

SVO browse `SkyMapper/SkyMapper` returned 6 ids, HTTP 200. First ids: SkyMapper/SkyMapper.u, SkyMapper/SkyMapper.v, SkyMapper/SkyMapper.g, SkyMapper/SkyMapper.r, SkyMapper/SkyMapper.i, SkyMapper/SkyMapper.z.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere documented; published documented; independent source original_documentation.
The DR page describes effective transmission curves with atmosphere at airmasses 1 and 2 in a figure. No machine-readable component file was opened. Filter-transformation coefficients are not the bandpass.
Sources read for this key:
- published / documented / original: SkyMapper data-release filter plots. Page: 'The SkyMapper filter curves (with atmosphere) are shown below.' It points to Bessell et al. (2011) for the filter set and to a filter-transformations page. No machine-readable curve URL was found on the data-release page. The filter-transformations page (https://skymapper.anu.edu.au/filter-transformations/, HTTP 200, 2026-10-20T06:04:17Z) also yielded no txt, dat or csv curve link in the extracted hrefs. The B...

### Paranal/OmegaCAM

SVO browse `Paranal/OmegaCAM` returned 57 ids, HTTP 200. First ids: Paranal/OmegaCAM.u_CAL_filter, Paranal/OmegaCAM.u_CAL, Paranal/OmegaCAM.u_SDSS_filter, Paranal/OmegaCAM.u_SDSS, Paranal/OmegaCAM.v_strom_filter, Paranal/OmegaCAM.v_strom, Paranal/OmegaCAM.B_aux_filter, Paranal/OmegaCAM.B_qB_filter....
Statuses: filter available; optics not_found; QE documented; atmosphere not_found; published documented; independent source original_files.
The ESO tar has scan tables and with_ccd siblings. Round 1 opened the filter scans. with_ccd numeric columns were not read, so that sibling is documented from its header and size, not file-verified as a curve. Do not assume the sibling is a calibrated product times QE.
Sources read for this key:
- filter / file_verified / original: ESO OmegaCAM transmission archive. Tar is 758218 bytes, 86 members, no README. 43 scan tables each have a with_ccd sibling, including vstr_modified_transmission_data_20080623 and its with_ccd sibling. Names cover u, g, r, i, z plus quadrant-style B, V, Halpha and CAL parts (aux and partA-D with add variants) and vstr/vstraux. No member name is an atmosphere, mirror, or standalone QE table. g_modified_transmission_data is 211716 bytes. Header comments:...
- published / documented / original: ESO OmegaCAM with_ccd sibling in the same archive. g_modified_transmission_data_with_ccd is also 211716 bytes. Its first comment lines match the filter-only g file (point counts, positions, angle). Numeric columns after the header were not compared, so a multiplication by QE was not verified in the bytes. The filename is the with_ccd label. Verification is documented for that reason.

### Gemini/GMOS-N

SVO browse `Gemini/GMOS-N` returned 19 ids, HTTP 200. First ids: Gemini/GMOS-N.u, Gemini/GMOS-N.HeII, Gemini/GMOS-N.g, Gemini/GMOS-N.HeIIC, Gemini/GMOS-N.OIII, Gemini/GMOS-N.OIIIC, Gemini/GMOS-N.r, Gemini/GMOS-N.Ha....
Statuses: filter available; optics not_found; QE available; atmosphere documented; published not_found; independent source original_files.
2001 g file is nm and transmission, peak 0.905, Barr set two, fax digitization. Hamamatsu QE files are fractions at -100 C; the HSC-type curve is an eyeball from a SPIE figure. 2001 EEV QE is a different detector. Hamamatsu was installed in 2017. ATRAN mktrans file on disk is truncated.
Sources read for this key:
- filter / file_verified / original: GMOS-N g transmission, Inger Jorgensen / Isobel Hook 2001. Header: wavelength in nm, transmission, not percent. Barr flatter filters, set two, digitized by Isobel Hook 29 Jan 2001. n=41 x=[375,575] y=[0,0.905] peak 0.905 at x=505. GG455 file says wavelength nm and transmission, created 27 June 2002. n=661 x=[440,1100] y=[0,0.973] peak 0.973 at x=792
- qe / file_verified / original: GMOS QE text files. HSC-type file: wavelength nm, QE, operating temperature -100 C, eyeball-estimated from Kamata et al. 2010 SPIE 7742. n=37 x=[340,1060] y=[0.05,0.955] peak 0.955 at x=580. Delivery sheet S10892-01 20 Aug 2009, QE at -100 C. n=44 x=[300,1200] y=[0,0.93] peak 0.93 at x=720. Red EEV average from NOAO Sept 2000, Inger Jorgensen 22 Feb 2001, three devices plus average. n=18 x=[300,1000] y=[0.047,0.895] peak 0.895 at x=650....
- atmosphere / documented / original: Gemini atmospheric transmission page. Page: ATRAN, Lord 1992 NASA TM 103957. Near-IR raw files are 0.9-5.6 um, sampling 0.00002 um, two columns, wavelength and transmission. Filename numbers are 10 times the water vapor in mm and 10 times the airmass. The table places mktrans_zm_16_15.dat at Maunakea, 1.6 mm, airmass 1.5. The local save is 80000 bytes and does not reach 5.6 um. n=3861 x=[0.9,0.9772] y=[0.0160228,1] peak 1 at x=0.9039. Cerro Pachon uses t...

### Gemini/GMOS-S

SVO browse `Gemini/GMOS-S` returned 17 ids, HTTP 200. First ids: Gemini/GMOS-S.u, Gemini/GMOS-S.HeII, Gemini/GMOS-S.g, Gemini/GMOS-S.HeIIC, Gemini/GMOS-S.OIII, Gemini/GMOS-S.OIIIC, Gemini/GMOS-S.r, Gemini/GMOS-S.Ha....
Statuses: filter available; optics not_found; QE available; atmosphere documented; published not_found; independent source original_files.
2005 g file peaks at 0.905 and does not say percent. June 2017 g file says percent and peaks at 90.2. Do not scale them the same way. The opened QE is the 2003 EEV mean, not the current Hamamatsu array.
Sources read for this key:
- filter / file_verified / original: GMOS-S g scans. 2005 header: wavelength nm, transmission, scanned by Tomislav Vucina 30 Sep 2005, Rodrigo Carrasco 3 Oct 2005, effective wavelength 475. n=951 x=[380,570] y=[0,0.905] peak 0.905 at x=501.2. June 2017 header: wavelength nm, transmission in percent, Cristian Moreno, Hwihyun Kim 29 Aug 2017. n=4750 x=[250.2,1200] y=[-0.007678,90.233] peak 90.233 at x=500.8
- qe / file_verified / original: GMOS-S blue EEV QE, Rodrigo Carrasco June 2003. Header: wavelength nm, QE for three EEV devices and a mean. CCD1 was replaced in April 2003. n=18 x=[300,1000] y=[0.054,0.868] peak 0.868 at x=550. Not the post-upgrade Hamamatsu QE.

### Gemini/NIRI

SVO browse `Gemini/NIRI` returned 48 ids, HTTP 200. First ids: Gemini/NIRI.Y-G0241w, Gemini/NIRI.Jcont1065-G0239, Gemini/NIRI.HeI-G0234w, Gemini/NIRI.PaGamma-G0240w, Gemini/NIRI.Jcont1207-G0232w, Gemini/NIRI.J-G0202w, Gemini/NIRI.PaBeta-G0221, Gemini/NIRI.PaBeta-G0221w....
Statuses: filter available; optics not_found; QE not_found; atmosphere documented; published not_found; independent source original_files.
Warm J G0202 and 298 K Y G0241 files say nm and percent. Cold .dat files are linked and were not opened. Do not cool a warm scan by hand. QE was not found.
Sources read for this key:
- filter / file_verified / original: NIRI filter text. J file header: scanned warm, Wavelength (nm), %T. n=2501 x=[800,3300] y=[0.0049538,91.485] peak 91.485 at x=1281. Y file header: room temp 298 K, 0 deg AOI, f/8, wl(nm), %T at center. n=1001 x=[900,1150] y=[-0.00464,80.724] peak 80.724 at x=1053. The NIRI component page links many additional .dat files, including cold traces. Those were not opened.

### Gemini/GSAOI

SVO browse `Gemini/GSAOI` returned 22 ids, HTTP 200. First ids: Gemini/GSAOI.Z, Gemini/GSAOI.HeI, Gemini/GSAOI.HIPaGamma, Gemini/GSAOI.JCont, Gemini/GSAOI.J, Gemini/GSAOI.HIPaBeta, Gemini/GSAOI.Hcont, Gemini/GSAOI.CH4short....
Statuses: filter available; optics not_found; QE not_found; atmosphere documented; published not_found; independent source original_files.
J_G1103 file says nm and percent and was scaled from room temperature to 70 K. Other ED*.dat links were not opened.
Sources read for this key:
- filter / file_verified / original: GSAOI J filter. Header: Filter ID ED191-1, Gemini name J_G1103, central wavelength 1.250 um, room temperature scaled to 70 K using a cold witness. Lambda(nm), Transmission (%). n=801 x=[1050,1450] y=[-0.01909,92.0532] peak 92.0532 at x=1299.5

### Gemini/Flamingos2

SVO browse `Gemini/Flamingos2` returned 13 ids, HTTP 200. First ids: Gemini/Flamingos2.Y, Gemini/Flamingos2.Jlow, Gemini/Flamingos2.J, Gemini/Flamingos2.JH, Gemini/Flamingos2.H, Gemini/Flamingos2.HK, Gemini/Flamingos2.K-blue_filter, Gemini/Flamingos2.K-blue....
Statuses: filter available; optics documented; QE not_found; atmosphere not_found; published available; independent source original_files.
J system file says microns and 1=100%, peak 0.64, and the page includes window, filter, nine lenses and two flats, excluding detector, grisms, telescope and atmosphere. The JH filter header says microns but the wavelengths run 817-2000 while the transmission peaks at 1.0. Do not honor that micron label.
Sources read for this key:
- published / file_verified / original: Flamingos-2 system transmission J, page dated in the filename 22 Apr 2019. Opened file header: Wavelength (micron), F2 Net System Transmission with J Filter (BARR MAN256-1; Gemini ID J_G0802), (1=100%). n=601 x=[1.1,1.4] y=[6.834e-05,0.63951] peak 0.63951 at x=1.292. The component page says the throughput data include the entrance window, filter, nine lenses and two flat mirrors, and exclude the detector, grisms, telescope and atmosphere.
- filter / file_verified / original: F2 JH filter transmission. Header says Wavelength (micron) and Transmission at 100 K (1=100%). The numeric wavelength column starts at 817, not 0.817. n=1184 x=[817,2000] y=[1.6e-06,0.999571] peak 0.999571 at x=1607. Treat the header unit as contradicted by the numbers.

### Gemini/GPI

SVO browse `Gemini/GPI` returned 5 ids, HTTP 200. First ids: Gemini/GPI.Y, Gemini/GPI.J, Gemini/GPI.H, Gemini/GPI.K1, Gemini/GPI.K2.
Statuses: filter available; optics not_found; QE not_found; atmosphere available; published not_found; independent source original_files.
Y FITS is 77 K, 2001 samples, no TUNIT, wavelength 0.90-1.30, transmission peak 0.871. J,H,K1,K2 were not decoded. 2MASS RSR files in the same repo are not GPI. GPI2 is not this set. trans_16_15.dat is 1.6 mm water and airmass 1.5; the IDL airmass keyword is ignored.
Sources read for this key:
- filter / file_verified / original: GPI pipeline GPI-filter-Y.fits. Primary header: FILTER Y, TEMP 77 K, vendor-supplied profile originally in Y.csv, converted by Marshall Perrin. Extension is one row, TTYPE WAVELENGTH and TRANSMISSION, TFORM 2001E, no TUNIT. n=2001 x=[0.9,1.3] y=[0,0.871139] peak 0.871139 at x=1.1288. filters.rst gives 50 percent cut wavelengths in microns: Y 0.95-1.14. The stored wavelength runs 0.90 to 1.30, consistent with microns, but the FITS header does not sa...
- atmosphere / file_verified / original: gpi trans_16_15.dat and drp_atmos_trans.pro. File begins '1.6mm H2O, airmass=1.5'. n=9401 x=[0.9,5.6] y=[0,1] peak 1 at x=1.0425. The IDL reader says the lookup is lambda in microns, sampling 5 Angstroms, reference airmass 1.5, and that the airmass keyword is currently ignored. It points at the Gemini transmission-spectra page.

### Gemini/GNIRS

SVO browse `Gemini/GNIRS` returned 12 ids, HTTP 200. First ids: Gemini/GNIRS.Y, Gemini/GNIRS.X, Gemini/GNIRS.J-MK, Gemini/GNIRS.J, Gemini/GNIRS.H, Gemini/GNIRS.H2, Gemini/GNIRS.K-MK, Gemini/GNIRS.K....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
j_mk.dat HTTP 200 says microns. The response column is unlabeled and peaks near 0.98. Other .dat links, including ytrans.dat, were not opened.
Sources read for this key:
- filter / file_verified / original: GNIRS j_mk.dat. HTTP 200. Header says Wavelength and Microns. Response column has no unit word. File identifies an NDF under /Users/tgeballe/GNIRS/FILTER_XMISSION/. n=271 x=[1.13,1.4] y=[-0.0151134,0.983219] peak 0.983219 at x=1.304

### Gemini/NIFS

SVO browse `Gemini/NIFS` returned 6 ids, HTTP 200. First ids: Gemini/NIFS.ZJ, Gemini/NIFS.ZJ_298K, Gemini/NIFS.JH, Gemini/NIFS.JH_298K, Gemini/NIFS.HK, Gemini/NIFS.HK_298K.
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
ZJ file says um and two percent columns at 298 K and 77 K. In-band peaks are 97.10 and 97.27, matching the header, so percent is real. JH and HK were not opened.
Sources read for this key:
- filter / file_verified / original: NIFS ZJ.dat. Header: ED055-W Z-J, peak T% 97.10 at 298 K and 97.27 at 77 K, cut-on and cut-off listed, then Wavelength (um) and two %T columns. Measured column maxima: 298 K column n=1801 x=[0.7,1.6] y=[0.000654418,97.0988] peak 97.0988 at x=1.264; 77 K column n=1801 x=[0.7,1.6] y=[-0.0123848,97.2725] peak 97.2725 at x=1.263. Compare those maxima with the header's 97 before choosing percent versus fraction. Measured in-band maxim...

### Keck/LRIS

SVO browse `Keck/LRIS` returned 17 ids, HTTP 200. First ids: Keck/LRIS.NB4000, Keck/LRIS.NB4325, Keck/LRIS.B, Keck/LRIS.g, Keck/LRIS.NB5390, Keck/LRIS.V, Keck/LRIS.OG570, Keck/LRIS.R....
Statuses: filter available; optics documented; QE not_found; atmosphere not_found; published not_found; independent source original_files.
g_blue and R have no headers and peak below 1. The page says manufacturer curves, equal-energy integrals, no telescope and no atmosphere, and the filter is not convolved with the CCD. Only two files were read. Match the dichroic.
Sources read for this key:
- filter / file_verified / original: LRIS filter data_files. Neither file has a header. g n=2048 x=[3119.45,5623.31] peak 0.918793 at 5021.5. R n=2048 x=[5237.82,10267.9] peak 0.781162 at 5947.98. The filters page says integrals assume equal energy and neglect telescope and atmosphere, FWHM is the filter curve not convolved with the CCD, and the curves are based on the manufacturer curve. Filters sit behind the dichroic. I is an xls file and was not opened.

### Keck/NIRC2

SVO browse `Keck/NIRC2` returned 11 ids, HTTP 200. First ids: Keck/NIRC2.J, Keck/NIRC2.Hcont, Keck/NIRC2.H, Keck/NIRC2.Fell, Keck/NIRC2.Kp, Keck/NIRC2.Ks, Keck/NIRC2.Brgamma, Keck/NIRC2.K....
Statuses: filter documented; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The xlsx has Wavelength and Transmission strings and named sheets, and no units. One sheet starts near 1.06 with a negative value. In-band curves were not reduced. QE was not found.
Sources read for this key:
- filter / documented / original: NIRC2 filter_passbands xlsx. Shared strings are only Wavelength and Transmission. Sheet names include J, H, K, Ks, Kp, Lp, Ms, Br gamma and FeII, plus plot sheets. One data sheet starts near wavelength 1.06 with a small negative transmission. Units are not in the workbook strings. A full in-band maximum was not extracted for every sheet. The .xls OLE file is also on disk and was not parsed.

### Keck/NIRSPEC

SVO browse `Keck/NIRSPEC` returned 16 ids, HTTP 200. First ids: Keck/NIRSPEC.N1, Keck/NIRSPEC.N2, Keck/NIRSPEC.N3, Keck/NIRSPEC.N4, Keck/NIRSPEC.N5, Keck/NIRSPEC.FeII, Keck/NIRSPEC.thin, Keck/NIRSPEC.N6....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
The filter workbook says Lambda(nm), Transmission(%) and optical density. One worksheet near 1000 nm was sampled. Other sheets were named and not reduced. The SPEC/SCAM workbook was not reduced.
Sources read for this key:
- filter / file_verified / original: NIRSPEC filters xlsx. Shared strings: Lambda(nm), Transmission(%), Optical Density. One worksheet has rows beginning 999.5 nm with transmission about 87.5 and a third column near 0.06. The workbook sheet list includes Y, J, H, K, K-prime, L-prime, M-prime and blockers. Not every sheet was reduced. A second workbook, nirspec_thru.xlsx, has sheets SPEC and SCAM and was not reduced to a curve.

### Keck/OSIRIS

SVO browse `Keck/OSIRIS` returned 38 ids, HTTP 200. First ids: Keck/OSIRIS.Zn2_spec, Keck/OSIRIS.Zn3_spec, Keck/OSIRIS.Zn3_imag, Keck/OSIRIS.Zbb_spec, Keck/OSIRIS.Zbb_imag, Keck/OSIRIS.Zn4_spec, Keck/OSIRIS.Zn5_spec, Keck/OSIRIS.Jn1_spec....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
J file says l(nm) and 298 K and peaks at 85, so it is not a 0-1 fraction. The response unit is not named. This is not GTC OSIRIS. Do not rescale 298 K to 77 K.
Sources read for this key:
- filter / file_verified / original: Keck OSIRIS J filter text. Header columns: l(nm) and 298K. No response-unit word. n=721 x=[1130,1490] peak 85.37 at 1396. The peak is 85, so it is not a 0-1 fraction. This is not GTC OSIRIS. An index note that the operating temperature is 77 K was reported by the collector and was not re-read as a separate file; the opened file says 298 K.

### Paranal/FORS2

SVO browse `Paranal/FORS2` returned 7 ids, HTTP 200. First ids: Paranal/FORS2.ESO1073, Paranal/FORS2.ESO1074, Paranal/FORS2.ESO1082, Paranal/FORS2.ESO1075, Paranal/FORS2.ESO1076, Paranal/FORS2.ESO1077, Paranal/FORS2.Gunn_z.
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
Three opened files disagree: Bessell B has no header and peaks at 0.69 near 420; Bessell I has no header and peaks at 0.95 near 7813; u_HIGH says nm and percent and peaks at 88.7 at 380 nm. SVO ids are ESO numbers and were not mapped. Interference peaks are uncalibrated on the page.
Sources read for this key:
- filter / file_verified / original: FORS filter curve directory. M_BESS_B.txt has no header. n=88 x=[330,1200] peak 0.687122 at 420. I_BESS+77COLL_SR.ascii has no header. n=1569 x=[5900.2,10290.6] peak 0.954332 at 7812.6. u_HIGH.txt header says Wavelength (nm) and %T. n=601 x=[300,1500] peak 88.6738 at 380. The curves page says u/b/v/g HIGH filters were bought for the FORS1 blue mosaic beginning 2007, lab curves, on-site measurements pending, and that interference-filter absolute ...

### Paranal/HAWKI

SVO browse `Paranal/HAWKI` returned 10 ids, HTTP 200. First ids: Paranal/HAWKI.Y, Paranal/HAWKI.NB1060, Paranal/HAWKI.NB1190, Paranal/HAWKI.J, Paranal/HAWKI.CH4, Paranal/HAWKI.H, Paranal/HAWKI.NB2090, Paranal/HAWKI.H2....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
Y is nm and percent at 87 K, peak about 94. One K file is nm and percent at 90 K. A second K download was byte-identical, so the 2007 and 2008 Ks epochs were not both in hand. No QE curve; the detector table is gain and read noise.
Sources read for this key:
- filter / file_verified / original: HAWK-I filter .dat files. Y header: Y filter curve at T=87 K, Y 3 (Iz881, SN003), Wavelength (nm), T (%). n=1251 x=[499.5,2997] peak 93.6957 at 1034.96. K header: T=90 K, Wavelength (nm), T (%). n=1251 x=[497,2982] peak 84.9992 at 2252.4. A second local K file advertised as the old K was byte-identical to this one, so two epochs were not compared. The HTML page describes Ks(1) before Jan 2008 and Ks(2) after. Detector pages give gain and read...

### Paranal/SPHERE

SVO browse `Paranal/SPHERE` returned 46 ids, HTTP 200. First ids: Paranal/SPHERE.ZIMPOL_V_S, Paranal/SPHERE.ZIMPOL_V, Paranal/SPHERE.ZIMPOL_V_L, Paranal/SPHERE.ZIMPOL_HeI, Paranal/SPHERE.ZIMPOL_R_PRIM, Paranal/SPHERE.ZIMPOL_OI_630, Paranal/SPHERE.ZIMPOL_N_R, Paranal/SPHERE.ZIMPOL_Cnt_Ha....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
IRDIS B_Y and ZIMPOL VBB headers name the filter and the 2015-04-16 update and do not name units. B_Y peaks at 0.96 near 1059. VBB reaches 1 near 710. A .dat URL for VBB returned HTML; the .txt was numeric. ND filters are a separate file.
Sources read for this key:
- filter / file_verified / original: SPHERE filter .dat files, Arthur Vigan / Julien Milli, 2015-04-16. B_Y header names the filter and the 2015-04-16 update and has columns wavelength and B_Y with no unit words. n=2301 x=[400,2700] peak 0.960018 at 1059. ZIMPOL VBB header likewise has no unit words. n=2301 x=[400,2700] peak 1 at 710. ND file header lists ND0.0, ND1.0, ND2.0 and ND3.5. The HTML page quotes filter wavelengths in nm. A downloaded VBB .dat URL returned HTML; the .txt sibling was the numeric file.

### Paranal/NACO

SVO browse `Paranal/NACO` returned 34 ids, HTTP 200. First ids: Paranal/NACO.NB104, Paranal/NACO.NB108, Paranal/NACO.NB109, Paranal/NACO.NB124, Paranal/NACO.NB126, Paranal/NACO.J, Paranal/NACO.NB128, Paranal/NACO.NB164....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
One 22-point J file, no header, peak 0.78 near 1.34. Decommissioned. Other filters and the five dichroics were not opened as curves.
Sources read for this key:
- filter / file_verified / original: NACO J curve. No header. n=22 x=[1.04,1.46] peak 0.78 at 1.34. Twenty-two points from 1.04 to 1.46, peak 0.78. Consistent with microns and a fraction, not stated. NACO is decommissioned. The characteristics page mentions transmission curves; most sibling files were not opened.

### Paranal/VIMOS

SVO browse `Paranal/VIMOS` returned 6 ids, HTTP 200. First ids: Paranal/VIMOS.U, Paranal/VIMOS.B, Paranal/VIMOS.V, Paranal/VIMOS.R, Paranal/VIMOS.I, Paranal/VIMOS.z.
Statuses: filter available; optics documented; QE documented; atmosphere not_found; published available; independent source original_files.
Imaging page: U-only is filter only and the bandpass file is filter+CCD+telescope optics, four quadrants. Both U files were read and have no units. B, V, R, I and z were not opened. Atmosphere is not in the page phrase. 2007 setup only.
Sources read for this key:
- filter / file_verified / original: VIMOS U_only_transmission_q1_to_q4.dat. Saved imaging HTML says: click the filter name to retrieve the transmission curve (filter only) and bandpass data (filter+CCD+telescope optics) for the four quadrants. The U-only href is filters_2007-04-04/U_only_transmission_q1_to_q4.dat. Local file header: wave, U1simple, U2simple, U3simple, U4simple. No unit words. Values start at 3300. B, V, R, I and z files linked on the same row pattern were not opened. HTTP st...
- published / file_verified / original: VIMOS U total transmission. Local file header: lambda, U1, U2, U3, U4. No unit words. The imaging page calls the bandpass link filter+CCD+telescope optics. Atmosphere is not in that phrase. Only the U total file was opened.

### Paranal/ISAAC

SVO browse `Paranal/ISAAC` returned 13 ids, HTTP 200. First ids: Paranal/ISAAC.SZ, Paranal/ISAAC.Js, Paranal/ISAAC.H, Paranal/ISAAC.NB207, Paranal/ISAAC.NB213, Paranal/ISAAC.Ks, Paranal/ISAAC.NB217, Paranal/ISAAC.NB229....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
FJ_BB.ASCII is the J link. The blue end is railed at ±512 and is not a transmission. Units are absent. Other filters were not opened.
Sources read for this key:
- filter / file_verified / original: ISAAC FJ_BB.ASCII. Imaging HTML links the J name to FJ_BB.ASCII and says to click the filter name to download the transmission curve. The file has no header. It starts near wavelength 0.83 with values that hit -512 and +512. That rail is not a transmission. A later portion of the file near 1.2-1.4 is on a 0-1 scale. Units are not stated. Other ISAAC filters were not opened.

### Paranal/ERIS

SVO browse `Paranal/ERIS` returned 17 ids, HTTP 200. First ids: Paranal/ERIS.J, Paranal/ERIS.Pa-b, Paranal/ERIS.Fe-II, Paranal/ERIS.H, Paranal/ERIS.H2_cont, Paranal/ERIS.H2-1-OS, Paranal/ERIS.Br-g, Paranal/ERIS.K....
Statuses: filter available; optics available; QE not_found; atmosphere not_found; published not_found; independent source original_files.
nix-filters.zip HTTP 200. filter_J wavelengths are about 1.2e-6 and peak near 0.86, with no unit words. Dichroic.txt says nm and percent and June 2021. ND.txt exceeds 1 and has no header. Do not share units across members.
Sources read for this key:
- filter / file_verified / original: ERIS NIX nix-filters.zip. Zip Content-Length 179815. filter_J.dat comment says data from ERIS/NIX/filter_J.txt, processed by cleantrim.pro. n=315 x=[1.122e-06,1.436e-06] peak 0.8607 at 1.223e-06 so wavelengths are about 1.2e-6 and the response peaks at 0.86. The file does not name metres. Dichroic.txt header: installed June 2021, wavelength nm, transmission percent. n=4501 x=[900,5400] peak 96.53 at 1436. ND.txt has no header. n=4000 x=[1000,...

### Paranal/VISIR

SVO browse `Paranal/VISIR` returned 20 ids, HTTP 200. First ids: Paranal/VISIR.J79, Paranal/VISIR.PAH1, Paranal/VISIR.J89, Paranal/VISIR.B87, Paranal/VISIR.ArIII, Paranal/VISIR.J98, Paranal/VISIR.SIV_1, Paranal/VISIR.B97....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
PAH1 and Q1 say microns and absolute percent, cold FTS, and include negative samples. Peaks are about 76% and 68%. Some HTML URLs require an ESO login. QE was not in these files.
Sources read for this key:
- filter / file_verified / original: VISIR PAH1 and Q1 FTS files. Both headers: Absolute filter transmission measurement at cold using a FTS; lambda (microns); absolute transmission (%). PAH1 n=1000 x=[5.58,11.58] peak 75.9894 at 8.61303. Q1 n=1000 x=[14.65,20.65] peak 68.2177 at 17.677. Negative transmission samples are present. Some VISIR HTML URLs redirect to an ESO login; these two text files were readable without a login.

### Paranal/FORS1

SVO browse `Paranal/FORS1` returned 14 ids, HTTP 200. First ids: Paranal/FORS1.ESO1038, Paranal/FORS1.ESO1033, Paranal/FORS1.ESO1112, Paranal/FORS1.ESO1039, Paranal/FORS1.ESO1034, Paranal/FORS1.ESO1113, Paranal/FORS1.ESO1115, Paranal/FORS1.ESO1040....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
Same curve directory as FORS2. u_HIGH is the file the page ties to the FORS1 blue mosaic from 2007, still pending on-site measurement when the page was written. The same three unit conventions apply. ESO-number ids were not mapped.
Sources read for this key:
- filter / file_verified / original: FORS filter curve directory, used by FORS1. u_HIGH is documented on the page as bought for the FORS1 blue mosaic. The same unit split as FORS2 applies. SVO FORS1 ids are ESO numbers and were not mapped to filenames.

### Paranal/NGTS

SVO browse `Paranal/NGTS` returned 1 ids, HTTP 200. First ids: Paranal/NGTS.NGTS.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The Warwick and ngtransits pages that were fetched did not contain a curve. A local file named as an SVO extract was not used as an original source. No airmass or PWV was found.

### Subaru/MOIRCS

SVO browse `Subaru/MOIRCS` returned 5 ids, HTTP 200. First ids: Subaru/MOIRCS.Y, Subaru/MOIRCS.J, Subaru/MOIRCS.H, Subaru/MOIRCS.Ks, Subaru/MOIRCS.K.
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
J1 file says nm and percent, 5 deg AOI, 120 K, and is already combined with the J filter. One file. Broadband replacements in 2008-2009 are documented. Do not copy HSC.
Sources read for this key:
- filter / file_verified / original: MOIRCS J1_convolved_1_aoi5_t120.txt. Header: MB_J1 filter, AOI 5 deg, 120 K correction applied; transmission curve combined with the J-band filter; Wavelength[nm] Transmission[%]. n=196 x=[1101,1296] peak 87.94 at 1228. The sensitivity page says broadband filters follow MKO and that transmission curves come from supplier datasheets. YJHKs were replaced with wedged substrates: Ks July 2008, J and H July 2009.

### Subaru/IRCS

SVO browse `Subaru/IRCS` returned 6 ids, HTTP 200. First ids: Subaru/IRCS.J, Subaru/IRCS.H, Subaru/IRCS.Kp, Subaru/IRCS.K, Subaru/IRCS.Lp, Subaru/IRCS.Mp.
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
j.txt says micrometres and percent, peak about 94 at 1.27 um, with negative out-of-band samples. Other bands were not opened.
Sources read for this key:
- filter / file_verified / original: IRCS j.txt. Header: WL (Micrometers), Transmission (%). Wavelength decreases from 1.5. n=2501 x=[1,1.5] peak 93.5175 at 1.2738. Negative out-of-band samples are in the file.

### Subaru/FOCAS

SVO browse `Subaru/FOCAS` returned 13 ids, HTTP 200. First ids: Subaru/FOCAS.U, Subaru/FOCAS.N373, Subaru/FOCAS.N386, Subaru/FOCAS.B, Subaru/FOCAS.N487, Subaru/FOCAS.N502, Subaru/FOCAS.N512, Subaru/FOCAS.V....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
B.txt has no header and peaks near 84 at 481. The HTML sensitivity table uses Angstrom and must not be copied onto this file. One band.
Sources read for this key:
- filter / file_verified / original: FOCAS camera B.txt. No header. n=224 x=[357,580] peak 84.26 at 481. The filters HTML table of sensitivity uses Wavelength (A). B.txt runs 357-580 and peaks near 84, so it is not an Angstrom fraction. Do not copy the sensitivity table unit onto this file.

### CFHT/Wircam

SVO browse `CFHT/Wircam` returned 13 ids, HTTP 200. First ids: CFHT/Wircam.Y, CFHT/Wircam.LowOH1, CFHT/Wircam.LowOH2, CFHT/Wircam.J, CFHT/Wircam.W, CFHT/Wircam.CH4_off, CFHT/Wircam.H, CFHT/Wircam.CH4_on....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
cfh8101.dat is the database J filter, no header in the file, peak near 86, matching the catalog peak of 86.1. Other WIRCam ids were not opened. Old WIRCam filters page was 404.
Sources read for this key:
- filter / file_verified / original: CFHT filter database cfh8101.dat. The database row says J, cfh8101, central 1253, width 158, peak transmission 86.1, WIRCam, 2004. The .dat file has no header. n=1077 x=[1147,1362.2] peak 86.259 at 1267. The catalog peak 86.1 matches a percent-like file peak near 86. The file itself does not say nm or percent. Central wavelength in the HTML is consistent with nm.

### CFHT/CFH12k

SVO browse `CFHT/CFH12k` returned 14 ids, HTTP 200. First ids: CFHT/CFH12k.Bc, CFHT/CFH12k.B, CFHT/CFH12k.HBeta_off, CFHT/CFH12k.HBeta_on, CFHT/CFH12k.OIII, CFHT/CFH12k.V, CFHT/CFH12k.Halpha_off, CFHT/CFH12k.R....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
cfh7403 is Mould B. The file peaks near 86 at wavelength about 466, while the catalog central wavelength is 4312. Those are not the same unit. One filter.
Sources read for this key:
- filter / file_verified / original: CFHT filter database cfh7403.dat. Database row: Mould B, cfh7403, central 4312, width 990, peak 86, rectangle, 1999. The .dat has no header. n=178 x=[300,654] peak 85.852 at 466. File wavelengths run about 300-654 while the catalog central wavelength is 4312, so the catalog central value is not in the same unit as the file column. File peak near 86 matches the catalog peak of 86.

### UKIRT/UFTI

SVO browse `UKIRT/UFTI` returned 14 ids, HTTP 200. First ids: UKIRT/UFTI.I, UKIRT/UFTI.Z, UKIRT/UFTI.Y, UKIRT/UFTI.J, UKIRT/UFTI.157cont, UKIRT/UFTI.H, UKIRT/UFTI.164Fe, UKIRT/UFTI.CH4_L....
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The UFTI URL that was fetched resolved to a sky-properties page, not a filter curve. No numeric UFTI file was opened. Do not copy WFCAM or UKIDSS totals onto UFTI.

### DENIS/DENIS

SVO browse `DENIS/DENIS` returned 3 ids, HTTP 200. First ids: DENIS/DENIS.I, DENIS/DENIS.J, DENIS/DENIS.Ks.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The CDS DENIS HTML page is not a curve. J/A+AS/124/379 ReadMe was 404 in the earlier pass. I, J and Ks curves were not opened.

### INT/WFC

SVO browse `INT/WFC` returned 19 ids, HTTP 200. First ids: INT/WFC.Strom_u, INT/WFC.RGO_u, INT/WFC.OII, INT/WFC.Strom_v, INT/WFC.B, INT/WFC.HeII, INT/WFC.Strom_b, INT/WFC.Gunn_g....
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The ING WFC page links a filter database. list.php returned an HTML page, not curves. An earlier request returned application source and no curve. No numeric WFC file was opened.

### INT/IPHAS

SVO browse `INT/IPHAS` returned 3 ids, HTTP 200. First ids: INT/IPHAS.gR, INT/IPHAS.Ha, INT/IPHAS.gI.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
IPHAS is a WFC survey subset, g, r and H-alpha. No IPHAS curve file was opened. Do not assume it equals every INT/WFC filter of the same name.

### INT/UVEX

SVO browse `INT/UVEX` returned 4 ids, HTTP 200. First ids: INT/UVEX.U, INT/UVEX.g, INT/UVEX.HeI, INT/UVEX.r.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
UVEX is a WFC survey subset. No UVEX curve file was opened. The ADS abstract was not used as a throughput table.

### OAJ/JPLUS

SVO browse `OAJ/JPLUS` returned 24 ids, HTTP 200. First ids: OAJ/JPLUS.uJAVA_filter, OAJ/JPLUS.uJAVA, OAJ/JPLUS.J0378, OAJ/JPLUS.F0378_filter, OAJ/JPLUS.J0395, OAJ/JPLUS.F0395_filter, OAJ/JPLUS.F0410_filter, OAJ/JPLUS.J0410....
Statuses: filter available; optics documented; QE documented; atmosphere documented; published documented; independent source original_files.
g filter file has no header and peaks near 0.93 at 515. The full-efficiency header says Angstrom, photons 0-1, and an Apache Point atmosphere, not OAJ. The saved full-efficiency body stops at 4283 Angstrom, so the in-band numbers were not read. Do not copy to J-PAS or S-PLUS.
Sources read for this key:
- filter / file_verified / original: J-PLUS JPLUS_gSDSS.dat. No header. n=1901 x=[250,1200] peak 0.930633 at 515. Peak near 0.93 at 515 is consistent with nm and a fraction. The file does not say so.
- published / documented / original: J-PLUS JPLUS_gSDSS_FullEfficiency.tab. Header read in full: authors J. Varela, A. Marin, C. Iniguez; date 2016-05-18; transmission = CCD QE * filter * optics * atmosphere; filter measured by CEFCA at the central point; optics = mirror reflectivity + field corrector + T80Cam entrance window; atmosphere model taken from Apache Point measurements, not an OAJ model; Lambda in Angstrom; T in photons, range 0-1. The local save is 20000 bytes and its numeric row...

### OAJ/JPAS

SVO browse `OAJ/JPAS` returned 60 ids, HTTP 200. First ids: OAJ/JPAS.uJava, OAJ/JPAS.u, OAJ/JPAS.J0378, OAJ/JPAS.J0390, OAJ/JPAS.J0400, OAJ/JPAS.J0410, OAJ/JPAS.J0420, OAJ/JPAS.J0430....
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The J-PAS instrumentation URL returned 404. The public site did not yield curve files. J-PLUS files are a different camera and were not copied. SVO lists about 60 filters.

### CTIO/S-PLUS

SVO browse `CTIO/S-PLUS` returned 12 ids, HTTP 200. First ids: CTIO/S-PLUS.u, CTIO/S-PLUS.F378, CTIO/S-PLUS.F395, CTIO/S-PLUS.F410, CTIO/S-PLUS.F430, CTIO/S-PLUS.g, CTIO/S-PLUS.F515, CTIO/S-PLUS.r....
Statuses: filter available; optics unknown; QE unknown; atmosphere unknown; published unknown; independent source original_files.
GitHub gSDSS.csv HTTP 200 has columns wavelength, transmittance, std, peak about 0.63, and does not say whether it is the lab curve or the atmosphere-convolved product. Other CSVs were listed and not opened. The GPLv3 badge was not read as a licence. Not J-PLUS.
Sources read for this key:
- filter / file_verified / original: splus-collab/splus_filters gSDSS.csv. Header: wavelength,transmittance,std. n=230 x=[360,589] peak 0.632898 at 518. README says the package can calculate lab curves and curves convolved with atmosphere and instrument. This CSV does not say which product it is. README shows a GPLv3 badge; the LICENSE file was not read.

### BOK/BASS

SVO browse `BOK/BASS` returned 4 ids, HTTP 200. First ids: BOK/BASS.g_filter, BOK/BASS.g, BOK/BASS.r, BOK/BASS.r_filter.
Statuses: filter available; optics not_found; QE available; atmosphere available; published available; independent source original_files.
bass-g.txt states Angstrom and three columns: filter, filter+QE, and filter+QE+atmosphere at airmass 1.0. r has the same header. DR9 paths 404. DR10 does not state an airmass in the prose that was read; the file does. Do not use DECam extinction coefficients as this bandpass.
Sources read for this key:
- filter / file_verified / original: Legacy Survey DR10 bass-g.txt. Header: wavelength in Angstrom; col2 pure filter; col3 filter plus CCD QE; col4 filter plus CCD QE plus atmosphere at airmass 1.0. Calls the filter 90Prime SDSS g. n=152 x=[3251.58,6495] peak 0.909 at 5114.48 for the filter column; QE column and airmass column were measured separately in facts bass_g_qe and bass_g_atm. DR9 copies of this path returned 404.

### BOK/90prime

SVO browse `BOK/90prime` returned 12 ids, HTTP 200. First ids: BOK/90prime.sdss_u, BOK/90prime.U, BOK/90prime.B, BOK/90prime.Wash_M, BOK/90prime.V, BOK/90prime.R, BOK/90prime.halpha_657, BOK/90prime.halpha_659....
Statuses: filter available; optics not_found; QE available; atmosphere available; published available; independent source original_files.
The BASS g and r files are 90Prime survey filters only. The rest of the SVO 90prime set, including U, B, Washington, V, R, H-alpha, I and z, was not opened.
Sources read for this key:
- filter / file_verified / original: the same Legacy Survey DR10 bass-g.txt and bass-r.txt recorded under BOK/BASS. The g header says 90Prime SDSS g, Angstrom, and three response columns (filter, filter plus CCD QE, filter plus CCD QE plus atmosphere at airmass 1.0). Those files do not cover the other 90prime filters.

### BOK/SCUSS

SVO browse `BOK/SCUSS` returned 2 ids, HTTP 200. First ids: BOK/SCUSS.u_noatm, BOK/SCUSS.u.
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
No SCUSS numeric file was opened. The u and u_noatm SVO ids were not copied from another u filter.

### KPNO/MzLS

SVO browse `KPNO/MzLS` returned 2 ids, HTTP 200. First ids: KPNO/MzLS.z, KPNO/MzLS.z_filter.
Statuses: filter available; optics documented; QE documented; atmosphere documented; published available; independent source original_files.
kpzd.txt is the area-weighted zd filter and reaches about 99 near 8960 with no unit word. kpzdccdcorr3.txt is a different, smaller scale and has no header. DR10 says the corrected file includes telescope, corrector, camera and atmosphere, and the opened sentence does not state airmass.
Sources read for this key:
- filter / file_verified / original: Legacy Survey DR10 kpzd.txt and kpzdccdcorr3.txt. kpzd.txt header: area-weighted transmission for the zd filter, DECam-style z for MOSAIC, k1038. No unit words. n=251 x=[8000,10500] peak 98.782 at 8960, so the response reaches about 99 near 8960 while wavelength starts at 8000. kpzdccdcorr3.txt has no header and a much smaller scale. DR10 text says the corrected file includes telescope, corrector, camera and atmosphere, and the opened DR10 HTML does not state that a...

### KPNO/Mosaic

SVO browse `KPNO/Mosaic` returned 53 ids, HTTP 200. First ids: KPNO/MOSAIC.BATC_337, KPNO/MOSAIC.U_Dey, KPNO/MOSAIC.Un_Steidel, KPNO/Mosaic.U, KPNO/MOSAIC.U_solid, KPNO/MOSAIC.BATC_390, KPNO/Mosaic.C, KPNO/MOSAIC.BATC_420....
Statuses: filter documented; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
SVO has about 53 Mosaic filters. The MzLS z files are one DECam-style z product, not that library. A NOIRLab Mosaic page returned a large HTML document without a curve file in the extract. SVO k1020 is described as filter only and was not byte-compared with kpzd.

### KPNO/DESI

SVO browse `KPNO/DESI` returned 3 ids, HTTP 200. First ids: KPNO/DESI.Blue, KPNO/DESI.Red, KPNO/DESI.NIR.
Statuses: filter documented; optics documented; QE documented; atmosphere not_found; published documented; independent source original_documentation.
DESI is a three-arm spectrograph. desimodel documents specter binary tables from DESI-0347. Those FITS files were not opened. A guessed GitHub contents URL 404'd. SVO says the Blue curve was digitized from a plot. Do not use DECam or Mosaic imaging curves.
Sources read for this key:
- published / documented / original: desimodel throughput documentation. The readthedocs page says the throughput model comes from systems-engineering spreadsheet DESI-0347 plus higher-resolution terms, with fiber geometric loss in DESIMODEL/data/throughput/fiberloss*.dat. Curves are specter binary tables. Columns described include a throughput term for mirrors, spectrograph and CCDs. A guessed GitHub contents URL for data/throughput returned 404. The instrument page states about 70 perce...

### LaSilla/WFI

SVO browse `LaSilla/WFI` returned 92 ids, HTTP 200. First ids: LaSilla/WFI.ESO877_filter, LaSilla/WFI.ESO877, LaSilla/WFI.ESO841_filter, LaSilla/WFI.ESO841, LaSilla/WFI.ESO865_filter, LaSilla/WFI.ESO865, LaSilla/WFI.ESO872_filter, LaSilla/WFI.ESO872....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
Filter 842 says Lambda [nm] and Transmission and peaks near 0.91. One ESO number. The filter tar URL 404'd. SVO has 92 ids and they were not mapped.
Sources read for this key:
- filter / file_verified / original: WFI 842.dat. Header: TITLE FILTER 842 Transmission Curve; LABX=Lambda [nm]; LABY=Transmission. n=291 x=[300,1200] peak 0.91179 at 475. An HTTPS copy of the full WFI filter tar ended on an ESO 404 page. This single .dat was readable.

### LaSilla/GROND

SVO browse `LaSilla/GROND` returned 7 ids, HTTP 200. First ids: LaSilla/GROND.g, LaSilla/GROND.r, LaSilla/GROND.i, LaSilla/GROND.z, LaSilla/GROND.J, LaSilla/GROND.H, LaSilla/GROND.K.
Statuses: filter available; optics unknown; QE unknown; atmosphere unknown; published unknown; independent source original_files.
One ASCII file contains grizJHK. The first column is named A, but g peaks at 0.71 near 436, which is not an Angstrom g filter. The file does not say nm, QE, optics or atmosphere. All seven bands are in the opened file. The PASP PDF was not opened.
Sources read for this key:
- filter / file_verified / original: GROND_filtercurves.txt. Header columns: A, gBand, rBand, iBand, zBand, JBand, HBand, KBand. g column n=2151 x=[350,2500] peak 0.70636 at 436. The column name A and a g peak at 436 cannot both be Angstroms and a g filter. The file does not say nm. It does not say whether QE, optics or atmosphere are included. The page calls the file GROND filter curves. The PASP 2008 PDF was not opened.

### LaSilla/EFOSC

SVO browse `LaSilla/EFOSC` returned 14 ids, HTTP 200. First ids: LaSilla/EFOSC.ESO640, LaSilla/EFOSC.ESO639, LaSilla/EFOSC.ESO724, LaSilla/EFOSC.ESO743, LaSilla/EFOSC.ESO742, LaSilla/EFOSC.ESO687, LaSilla/EFOSC.ESO782, LaSilla/EFOSC.ESO641....
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
EFOSC filter pages were fetched and did not yield a numeric curve in the extract. No GIF was opened. ESO-number SVO ids were not mapped.

### LaSilla/SOFI

SVO browse `LaSilla/SOFI` returned 20 ids, HTTP 200. First ids: LaSilla/SOFI.NB106, LaSilla/SOFI.NB108, LaSilla/SOFI.NB119, LaSilla/SOFI.NB121, LaSilla/SOFI.Js, LaSilla/SOFI.J, LaSilla/SOFI.NB126, LaSilla/SOFI.NB128....
Statuses: filter available; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_files.
One J file, no header, peak 0.79 near 1.35. Units not stated. Other bands and a QE curve were not opened.
Sources read for this key:
- filter / file_verified / original: SOFI J curve. No header. n=3334 x=[1.0001,1.50015] peak 0.79043 at 1.3541. Wavelength 1.00-1.50 and peak 0.79 are consistent with microns and a fraction. The file does not say so. The old decommissioned inst URL returned an ESO 404.

### LCO/IMACS

SVO browse `LCO/IMACS` returned 44 ids, HTTP 200. First ids: LCO/IMACS.sloan_u, LCO/IMACS.Rauch, LCO/IMACS.Bessell_B1, LCO/IMACS.Bessell_B2, LCO/IMACS.WB3600, LCO/IMACS.sloan_g, LCO/IMACS.WBP4507, LCO/IMACS.Roman_OIII....
Statuses: filter not_found; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
Pages that were fetched describe the instrument and do not include a numeric filter file. The SVO set was not copied from another Mosaic or FourStar curve.

### LCO/FourStar

SVO browse `LCO/FourStar` returned 10 ids, HTTP 200. First ids: LCO/FourStar.J1, LCO/FourStar.J2, LCO/FourStar.NB118, LCO/FourStar.J, LCO/FourStar.J3, LCO/FourStar.Hshort, LCO/FourStar.H, LCO/FourStar.Hlong....
Statuses: filter not_found; optics available; QE not_found; atmosphere not_found; published not_found; independent source original_files.
J_cam_optics.txt is optics by filename, no header, 10900-14000, peak about 0.86. The filter file was not opened. A manual URL 404'd.
Sources read for this key:
- optics / file_verified / original: FourStar J_cam_optics.txt. No header. n=311 x=[10900,14000] peak 0.8571 at 13440. A guessed user-manual URL returned 404. The filters page itself was fetched and links this curves directory.

### LCO/OGLE-IV

SVO browse `LCO/OGLE-IV` returned 4 ids, HTTP 200. First ids: LCO/OGLE-IV.V, LCO/OGLE-IV.V_filter, LCO/OGLE-IV.I, LCO/OGLE-IV.I_filter.
Statuses: filter available; optics not_found; QE available; atmosphere not_found; published not_found; independent source original_files.
I filter header says Angstrom and mean transmission of nine spots, peak about 0.99. The QE file has no header; the filename says min and max and the column order is not in the file. V was not opened.
Sources read for this key:
- filter / file_verified / original: OGLE-IV I filter transmission. Header: OGLE-IV I-band filter, Asahi Spectra, measurements in 9 spots, wavelength[A], mean_transmission, standard_dev, then spot1-9. n=682 x=[2495.43,11020.2] peak 0.99134 at 8264.91 on the mean column, which peaks below 1.
- qe / file_verified / original: OGLE-IV QE min/max curves. No header. Three columns. Wavelength-like values from about 3032 to 10787. Column 2 peak n=93 x=[3032.08,10787.2] peak 0.839162 at 4912.01. The filename says min and max. Which column is min is not written in the file.

### LCO/CSP

SVO browse `LCO/CSP` returned 16 ids, HTTP 200. First ids: LCO/CSP.u, LCO/CSP.B, LCO/CSP.g, LCO/CSP.V, LCO/CSP.V_LC3014, LCO/CSP.V_LC3009, LCO/CSP.r, LCO/CSP.i....
Statuses: filter documented; optics documented; QE documented; atmosphere documented; published available; independent source original_files.
g_tel_ccd_atm_ext_1.2.dat is the 2003-2013 Swope Site2 g link. The file has no header. The page describes telescope-plus-CCD scans. ext_1.2 is not defined in the opened text. The component tarball was not opened. This is not V-LC3014 on Site3 and not CSP-II.
Sources read for this key:
- published / file_verified / original: CSP g_tel_ccd_atm_ext_1.2.dat. No header in the file. n=101 x=[3740,5740] peak 0.717912 at 5380. Page: scans by Rheault, Marshall, Depoy and Villanueva of the relative throughput of the telescope and SiTE #3 CCD, also e2V later for CSP-II, monochromator 2.5 nm FWHM, filters B,V,u,g,r,i. The g link in the 2003-2013 table is this file and the instrument cell says Swope Site2 CCD. Effective-wavelength column is in Angstrom and the g value is 4765.1. ...

### LCO/CSP2

SVO browse `LCO/CSP2` returned 9 ids, HTTP 200. First ids: LCO/CSP2.u, LCO/CSP2.B, LCO/CSP2.g, LCO/CSP2.V, LCO/CSP2.r, LCO/CSP2.i, LCO/CSP2.Y, LCO/CSP2.J....
Statuses: filter documented; optics documented; QE documented; atmosphere documented; published documented; independent source original_documentation.
The same CSP page says CSP-II filters are included and links CSPCSPII_passbands.tgz. That archive was not opened. CSP-I g was not copied onto CSP-II.

### GTC/OSIRIS

SVO browse `GTC/OSIRIS` returned 60 ids, HTTP 200. First ids: GTC/OSIRIS.sdss_g_filter, GTC/OSIRIS.sdss_g, GTC/OSIRIS.F643_filter, GTC/OSIRIS.F643, GTC/OSIRIS.sdss_r_filter, GTC/OSIRIS.sdss_r, GTC/OSIRIS.F648_filter, GTC/OSIRIS.F648....
Statuses: filter available; optics not_found; QE documented; atmosphere not_found; published not_found; independent source original_files.
Sloang.txt has no header and peaks near 82, so it is not a 0-1 fraction. u is absent on the page. Old QE is four percentages, not a curve. OSIRIS+ is a later detector and was not assigned these numbers.
Sources read for this key:
- filter / file_verified / original: GTC OSIRIS Sloang.txt. No header. n=251 x=[350,600] peak 82.4751 at 488. The peak is about 82, so the column is not a 0-1 fraction, and the file never says percent. The page gives old-detector QE as four points, not a curve. OSIRIS+ is a later deep-depleted CCD and was not given these numbers.

### GTC/HIPERCAM

SVO browse `GTC/HIPERCAM` returned 30 ids, HTTP 200. First ids: GTC/HIPERCAM.u_filter, GTC/HIPERCAM.u_s_filter, GTC/HIPERCAM.u_noatm, GTC/HIPERCAM.u, GTC/HIPERCAM.u_s_noatm, GTC/HIPERCAM.u_s, GTC/HIPERCAM.g_s_filter, GTC/HIPERCAM.g_s_noatm....
Statuses: filter documented; optics documented; QE documented; atmosphere documented; published documented; independent source original_files.
ucam_thruput mode strings select telescope and instrument. Use hcam and gtc, not ULTRACAM or ULTRASPEC. noatmos is optional, so the default includes atmosphere. hcam_g.txt on disk is 80071 bytes against 100050 in the git tree, so it is truncated. CCD and collimator sizes match two tree entries but were not hashed. No file header states units, airmass or PWV.
Sources read for this key:
- published / documented / original: StuartLittlefair/ucam_thruput. README on the GitHub page: an observing-mode string selects telescope (gtc, tnt, wht, ntt or vlt), instrument (ucam, uspec, hcam) and filter. noatmos ignores the atmosphere. ULTRACAM and ULTRASPEC graphs are separate from HiPERCAM. Files have no headers. Git tree sha 15315eaea46eb69847cc6f9e4ac4c2767951b6ac lists hcam_g.txt at 100050 bytes. The local copy is 80071 bytes, so it is truncated. Within that prefix, n=1601...

### LasCumbres/LasCumbres

SVO browse `LasCumbres/LasCumbres` returned 18 ids, HTTP 200. First ids: LasCumbres/LasCumbres.SDSS_up, LasCumbres/LasCumbres.UV, LasCumbres/LasCumbres.BSkymapper_v, LasCumbres/LasCumbres.Bessel_B, LasCumbres/LasCumbres.SDSS_gp, LasCumbres/LasCumbres.Hbeta, LasCumbres/LasCumbres.OIII, LasCumbres/LasCumbres.D51....
Statuses: filter documented; optics not_found; QE not_found; atmosphere not_found; published not_found; independent source original_documentation.
The filter index separates Sinistro, QHY and Spectral imagers and Bessell versus SDSS/Pan-STARRS sets. The all-filter zip returned 403. A saved .txt path was HTML, not a curve. One network is not one atmosphere. Curves may change when CCDs are replaced.

## Limits

Seventy SVO keys have a dossier. That is the milestone count. It is not a claim that every band, detector epoch or component file was opened. Sibling files are named only where a page listed them. Representative inspections are marked above.

No photon-versus-energy convention was stated by the opened GMOS, HiPERCAM, GROND, SPHERE, FORS, LRIS, OSIRIS, HAWK-I or Hewett files. PS1 and the J-PLUS full-efficiency header do state a convention. 2MASS states it for the RSR only.

Licence text was not interpreted. A GPLv3 badge on the S-PLUS README and a LICENSE file in the ZTF tree were not read as clearance.

The Gemini Maunakea ATRAN file on disk ends at 0.977 um because the save is 80000 bytes. The GPI copy `trans_16_15.dat` does reach 5.6 um and is the file that was measured.

Subagent notes that were not re-read as files are not in `sources.jsonl`. VIMOS and ISAAC files were re-read. The NGTS file whose name marks it as an SVO extract was not entered as an original curve.

Wavelength and response fields on the round-1 records were filled from each record's evidence. Files that do not state a unit still say unlabeled. OmegaCAM `with_ccd` is documented, not file-verified, because its numeric columns were not read.

Keys whose first checkpoint had a review and no source row now have a search record: NGTS, UFTI, DENIS, INT/WFC, IPHAS, UVEX, J-PAS, SCUSS, Mosaic, EFOSC, IMACS, CSP-II and Las Cumbres. BOK/90prime points at the opened BASS g and r files only.
