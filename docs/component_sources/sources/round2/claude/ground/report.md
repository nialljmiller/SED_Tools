# Ground-based continuation (Claude, 2026-10-20) — extends round2/grok

Scope: deepen Grok's weakest ground keys by actually retrieving original data. Grok's files are untouched. Evidence is in `evidence/` (hashes in `SHA256SUMS`). No licence reviewed.
WebSearch was denied in this session, so only direct URL probes were used. Keys below not listed were not worked on; Grok's 70-key set remains representative-file depth for most.

## LCO/CSP and LCO/CSP2 — upgraded to files in hand (verified)
- Whole passband bundle (CSP-I and CSP-II, 21 FITS): https://csp.obs.carnegiescience.edu/data/CSPCSPII_passbands.tgz (56,027 B). Members `passbands/{u,g,r,i,B,V,V0,V1,Y,J,J2,H}_CSP.fits` and `{u,g,r,i,B,V,Y,J,H}_CSP2.fits`, plus `docsp.pro` and `standards/CSP.mags.*.dat`.
- FITS: primary HDU empty; first extension is an ASCII TABLE (`WAVELENGTH`, `RESPONSE`, D20.7). Header: `WAVEUNIT='Angstroms'`, `RESPUNIT='photons'` (photon-counting response), `ZEROPT`, `MODLSPEC`, `MODLWEFF`. g_CSP2: 204 rows, peak 0.7931 at 5020 A, zero-point -20.7808 (natural system; BD+17d4708 model).
- Original text files: `https://csp.obs.carnegiescience.edu/data/{filter}_tel_ccd_atm_ext_1.2.dat` (CSP-I) and `{filter}_CSP2_tel_ccd_atm_ext_1.2.dat`; NIR `{Y,J,H}_{SWO|DUP}_TAM_scan_atm.dat`, `Jrc1/Jrc2`. Two columns, Angstrom and fractional response. **Cross-check:** FITS g_CSP2 interpolated onto the .dat grid matches to ratio 0.9999998–1.0000004.
- **Definition (what the curve contains):** telescope + CCD/camera + filter + atmosphere at **1.2 airmass** (CTIO). Not filter-only. Do not apply an extra atmosphere. CSP-I vs CSP-II are separate files; three V filters (LC3014 until 2006-01-13, LC3009 2006-01-14..24, LC9844 afterwards) and two J (Jrc1 until 2009-01-15, Jrc2 after); the SVO name set must be mapped to these epochs.
- **Components exist:** `CSP_filter_package.tgz` (201,981 B) contains `Telescope.txt` (nm, 2515 rows), `Telescope+Camera.txt` (nm, ~598 rows), raw `*_texas_WLcorr.txt` scans (Telescope+Camera+Filter, Angstrom), `atm_ext_ctio.dat` (extinction), `atm_ext_lines.dat` (telluric lines), README with zero-points and the flux definition F = (1/ch) ∫ S f_λ λ dλ (photon convention). So optics/QE are `derived` (combined Telescope+Camera) — not separate mirror/QE files.
- Correction to Grok: "CSP2 archive not opened / Las Cumbres 403" are obsolete — both URLs return 200 with real data.

## LasCumbres — files in hand (verified structure)
- https://lco.global/documents/2/lco_filter_curves.zip (545,526 B, SHA in evidence): 18 text files: `SDSS.{up,gp,rp,ip}.txt` (nm, header `\lambda [nm]`, 4666 rows, g peak 0.9958, includes small negatives), `bssl-{ux,bx,vx,rx,ix}-*.txt` (Bessell/Bessell-like), `skmp-vs-008.txt` (SkyMapper-style v), `PSTR-{ZS,YX}-avg.txt` (Pan-STARRS z_s,Y), `solr-vr-008.txt`, `astrodon.uv.txt`, narrowband `NBND-HA/HB`, `nbnd-o3`, `DDOX-51`. 200–1300 nm at 0.2 nm; two columns, fractional (<=1).
- Not verified: whether these are filter-only transmission (peaks 0.73–0.99 are consistent with filter-only) — treat as filter transmission until LCO documents otherwise. No QE/telescope/atmosphere here.

## CTIO/DECam
- https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt is actually a FITS file (despite .txt). BINTABLE, 1441 rows, columns `LAMBDA` (3800–11000, step 5; no TUNIT; Angstrom by magnitude) and `g,r,i,z,Y` (double) plus `atm` (float). Column maxima: g 0.678 @5440 A, r 0.846 @6855, i 0.963 @8500, z 0.977 @8890, Y 0.770 (an earlier note of 0.957 was the `atm` column, not Y). Whether CCD QE, optics or atmosphere are included is **not established** from the file (no units or comments in the header); read scidoc0472/0434 before using it as a total. Not verified against Grok's DECam record.
- `https://des.ncsa.illinois.edu/files/y3a2/STD_BANDPASSES_Y3A2_20180302.fits` returns an HTML 998-byte page (moved/blocked); Grok's legacysurvey `decam.g.am1p4.dat.txt` returns 404 — both Grok URLs are stale; list the DR10 file index (https://www.legacysurvey.org/dr10/files/, HTTP 200) to locate the current names.

## Searched, still no numeric curve (honest negatives, not proof of absence)
- SkyMapper: `/filter-transmission/` 404; /filter-transformations/ page cites Bessell 2011 and Casagrande/Wolf 2018; only standards FITS linked. Next lead: Bessell+2011 PASP 123,683 tables
- INT/WFC & IPHAS: ING filter database (https://astro.ing.iac.es/filter/) is an INGRID exposure calculator page, no files; `Astronomy/instruments/wfc/filters.html` 404; ING detectors page (ultra_wfc.htm) 200 but no curves. Next: ING WFC filter pages under /Astronomy/instruments/wfc/, CASU, Barentsen+2014 (VizieR).
- UKIRT/UFTI: ukirt.hawaii.edu pages unreachable (connection fail). Next: Wayback/IfA mirrors, UKIRT filter page used by Hewett+2006 (Grok has WFCAM tables).
- DENIS: cds.unistra.fr/denis/ HTTP 401. J-PAS: j-pas.org /survey/filters 404, home 200. NGTS: ngts.warwick.ac.uk 403. EFOSC: ESO page lists filters at https://filters.sc.eso.org/efs/efs_fi/ which did not connect; Efosc2Filters.html (200) has no files. IMACS: LCO(Chile) filter page 404.

## Implementation notes
Prefer `.dat`/FITS over SVO for CSP; treat their curves as already containing atmosphere (airmass 1.2). Keep CSP-I/II separate.


## Continuation 2 (2026-10-20, web access enabled)
**DECam atm column (decoded from evidence/scidoc1884_decam.fits):** `atm` is a smooth atmosphere curve, 0.307–0.957 (0.61 @3800 A, 0.81 @5000, 0.94 @8600). Dividing grizY by atm (the NOIRLab instruction) gives peaks g 0.8125 @5440, r 0.9319 @6860, **i 1.022 @8500, z 1.037 @9060**, Y 0.861 @9570. Values >1 mean the quotient is not a clean physical throughput in the i/z/Y red (narrow telluric absorption is in atm, grid is 5 A); clip/validate before use. Components (filter/optics/QE) remain not separable here.
**Legacy Survey DECam files — URL fixed:** the working path is `https://www.legacysurvey.org/files/decam.{g,r,i,z}.am1p4.dat.txt` (HTTP 200, 21.7 kB each; the earlier `/dr10/files/` path 404s). Header: 'DECam system throughput, from DECaLS wiki, source DECamFilters-14apr18.xslx, airmass 1.40'; two columns, Angstrom, 3000–, 10 A step. r peak 0.512 @6830, i 0.560 @8470, z 0.552 @8890 (these are system throughput incl. atmosphere at X=1.4, clearly lower than the DES scidoc1884 X=1.2 curves, so the two are different products). g file's peak was not parsed (header line break), re-run if needed. Files + SHA256 in evidence/legacysurvey_decam.
**SkyMapper:** official curves exist only as a figure (airmass 1 and 2). A third-party coarse 100 A table is at `http://www.mso.anu.edu.au/~brad/Filters/SkyMapper{u,v,g,r,i,z}` (saved under evidence/skymapper_thirdparty; first line looks like a zero-point, 100 A sampling, provenance unverified — NOT an original source; use only as a fallback to SVO).
**NGTS:** Wheatley+2018 (MNRAS 475, 4476, doi 10.1093/mnras/stx2836) states a data file with the combined NGTS system throughput is in the *online supplementary material*; OUP returns 403 to scripted access (archived copy at web.archive.org does not expose the file). Lead for a browser/other-machine download; no numeric curve obtained. Design band 520–890 nm.
**J-PAS:** miniJPAS paper says curves are on SVO and include telescope, CCD and atmosphere ('overall system efficiency'); j-pas.org /datareleases and /ancillarydata show no curve files. Remains SVO-only.
**IPHAS/INT-WFC:** ING filter DB (astro.ing.iac.es/filter) is currently broken (returns PHP source, no data); only a PNG plot (c.ing.iac.es/filter/filtersWFC.png). IPHAS DR2 VizieR II/321 has catalogue only. No numeric curve.
**EFOSC:** Efosc2Filters.html links only GIF plots (eso0623.gif etc.); filters.sc.eso.org unreachable. No numeric curve.
**DENIS:** Fouqué+2000 (A&AS 141, 313) describes total profiles (atmosphere, mirrors, lenses, dichroics, filters, detector) but CDS ReadMe for J/A+AS/124/379 still 404; table not located.
**UFTI:** all ukirt.hawaii.edu hosts fail to connect. **SCUSS:** only paper figure (Zou+2015, arXiv 1509.02647; u 3538 A FWHM 520 A); no table. **IMACS, UVEX:** not re-probed.
Net: no new key moved to numeric curves in this pass except fixing DECam/Legacy URLs; the rest need a browser-capable fetch (NGTS supplement, ESO filter DB), figure digitisation, or SVO fallback.
