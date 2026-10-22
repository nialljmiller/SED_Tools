# Ground pass 3 (Claude, 2026-10-21): original sources at ESO, NASA/STScI/IRSA, NAOJ, Gemini, Rubin

Policy applied (per Nill): look for original facility files first; whatever cannot be found quickly falls back to SVO. Grok's files and my earlier ground/ files are untouched. Files in `evidence/` (SHA256SUMS, `inventory.csv` lists 633 numeric curves with source URL, rows, x range, y max and header). Scripts in `tools/`. No licences reviewed.

## Found (original, numeric, in hand)
| Facility/instrument | Source | What is there |
|---|---|---|
| VISTA/VIRCAM | ESO `.../vircam/inst/filters/Filters_QE_Atm_curves.tar.gz` | Y,Z,J,H,Ks (80 K) + NB118/980/990 filters, **M1 and M2 reflectivity, detector QE, atmosphere transmission grid** (percent, nm). The best-separated component set found. |
| HAWK-I | eso.org `hawki_*.dat` | 11 filters (nm, percent, ~87 K) |
| NACO | eso.org `naco/inst/filters/*.dat` | 34 filters (J,H,Ks,L',M', NB, IB) |
| SPHERE | eso.org `sphere/inst/filters/` | 42 files: IRDIS BB/DB/NB, ZIMPOL, ND, CPI_ND |
| ERIS/NIX | eso.org `nix-filters.zip` | filters, cameras, dichroic, ND, ISSM (24) |
| FORS1/2 | eso.org `fors/inst/Filters/` | ~76 curves: Bessell/Gunn/special, narrow-band, HR/SR collimator × 5 field positions (mixed units: nm/Å, %/fraction; some arbitrarily scaled — see headers) |
| HSC | naoj.org HSC sensitivity page | 25 filters (+ `wHSC-*` variants, meaning not documented on the page), `qe_ccd_HSC`, dewar window, POpt2, vignetting (Å, fraction) |
| GMOS-N/S | gemini.edu `/instrumentation/gmos/components` | filters, gratings, CCD QE (Hamamatsu, e2v, red/blue) |
| NIRI, GSAOI, Flamingos-2, GNIRS, NIFS | gemini.edu `.../components` | 49 / 22 / 12 / 9 / 3 files |
| LSST | github `lsst/throughputs` (commit 0bcec2a) | baseline filters, lens1-3, m1-m3, detector, atmos, darksky; also mirrored SDSS (Doi), MegaCam, 2MASS, Pan-STARRS, WISE, Johnson |
| JWST NIRCam | STScI `nircam_throughputs_4Nov2022_v5` | 29 mean *system* throughputs (not filter-only) |
| Spitzer IRAC, MIPS | IRSA `080924ch{1-4}trans_full.txt` (µm), `MIPSfiltsumm.txt` | system spectral response |
| TESS | HEASARC `tess-response-function-v2.0.csv` | full band response |

Also fetched: WISE page tables (8 files) and HST WFC3 throughput page files (2) — not yet reviewed.

## Not obtained
- **NOIRLab (KPNO Mosaic/NEWFIRM/WHIRC/FLAMINGOS/GOLDCAM/KOSMOS, CTIO MOSAIC/ISPI/Hydra/SOAR, DECam spec)**: structure is known — 257 filter ids at `noirlab.edu/science/filters/<id>`, each linking `/science/documents/<id>` → `/science/sites/default/files/media/archives/documents/<file>.txt`; DECam page also lists `scidoc3138.xls`. My first fast burst tripped AWS WAF (HTTP 202 challenge). I did **not** try to bypass it. A background job (`tools/noirlab_when_clear.sh`, log `noirlab_harvest.log`) probes every 5 min for 2 h and then harvests serially at 1.5 s/request. If it never clears, fetch from a browser/other IP or use SVO.
- **ESO WFI**: filter page links to a dead host (`www.ls.eso.org`) and the per-filter .dat / `WFI-Filters-2006-09-18.tar.gz` return an HTML stub. Lead: Wayback copy. Otherwise SVO.
- CFHT MegaCam/WIRCam, Keck (LRIS/NIRC2/OSIRIS), UKIRT, CTIO others: pages fetched but no data links parsed (JS-driven or off-page). Curves for MegaCam exist only through the lsst mirror. Not chased.
- Still SVO-only per earlier notes: SkyMapper, NGTS, DENIS, INT/WFC, IPHAS, UVEX, J-PAS, SCUSS, EFOSC, IMACS, FourStar, UFTI.

## Caveats
Per-file units differ (nm/Å/µm; percent/fraction); definitions (filter-only vs system) must be read per file — headers mostly say, HSC `w*` and ESO FORS HR/SR files do not. Nothing here is independently verified by Codex yet.
