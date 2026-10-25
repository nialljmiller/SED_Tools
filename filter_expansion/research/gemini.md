STATUS: DONE

Gemini (Antigravity CLI, Gemini 3.8 Flash), 2026-10-25. Research only (Scope: Block C Part 2 — large telescopes).
135 filter passband curves cataloged in [gemini.csv](gemini.csv) across 18 instruments at 7 facilities (Keck, Paranal/VLT, VST, Subaru, Magellan/LCO, NOT, WIYN, KPNO).
88 curves verified from original observatory ASCII/tables (Keck, ESO, NAOJ, Carnegie/LCO, Harvard/CfA, NOT, WIYN, NOIRLab) via curl HEAD (HTTP 200).
47 curves provided with SVO FPS identifiers and URLs as fallbacks (where observatory sites require logins, redirect to portals, or lack direct single-filter files).
Zero dead links on original endpoints; all detector technologies confirmed as photon-counting (optical CCDs, near-IR HgCdTe/InSb, and mid-IR Si:As arrays).

## Instrument Summary

| Facility | Instrument | Bands | Priority 1 / 2 / 3 | Source Kind | Detector Type | Notes |
|---|---|---|---|---|---|---|
| Keck | NIRC2 | z, Y, J, H, K, Ks, Kp, Lp, Ms, Br_gamma, FeII | 9 / 0 / 2 | Original (HTTP 200) | Photon (InSb) | Cryogenic filter curves from `filter_passbands.xls` |
| Keck | LRIS | u, g, B, V, R, Rs, I, RG850 | 7 / 1 / 0 | Original (HTTP 200) / SVO (u) | Photon (CCD) | ASCII curves in `data_files/`; u fallback to SVO |
| Keck | MOSFIRE | Y, J, H, K, Ks, J2, J3, H1, H2 | 5 / 4 / 0 | Original (HTTP 200) | Photon (HgCdTe H2RG) | Direct ASCII files in `data/filter/` |
| Keck | DEIMOS | B, V, R, I, Z | 5 / 0 / 0 | Original (HTTP 200) | Photon (CCD) | 3-column ASCII (wavelength, convolved, pure filter) |
| Paranal | FORS1 | u_BESS, b_BESS, v_BESS, r_BESS, i_BESS, u/g/r/z_GUNN | 5 / 4 / 0 | Original (HTTP 200) | Photon (CCD) | Optical bench ASCII curves at ESO FORS site |
| Paranal | FORS2 | u_HIGH, b_HIGH, g_HIGH, v_HIGH, R_SPECIAL, I_BESS, z_GUNN, H_Alpha | 7 / 0 / 1 | Original (HTTP 200) | Photon (CCD) | In-instrument SR collimator curves at ESO FORS site |
| Paranal | ISAAC | SZ, Js, J, H, Ks, L, M | 7 / 0 / 0 | SVO FPS | Photon (HgCdTe/InSb) | Decommissioned instrument pages redirect to login; SVO IDs provided |
| Paranal | VISIR | B8.7, B9.7, B10.7, B11.7, B12.4, Q1, Q2 | 7 / 0 / 0 | SVO FPS | Photon (Si:As IBC) | Documented in VISIR User Manual; SVO IDs provided |
| Paranal | OmegaCAM | u, g, r, i, z (SDSS), B, V (Johnson) | 7 / 0 / 0 | Original (HTTP 200) | Photon (CCD) | Verified in `OmegaCAM_transmission.tar.gz` |
| Subaru | Suprime-Cam | B, V, R, I, g, r, i, z | 8 / 0 / 0 | Original (HTTP 200) | Photon (FDCCD) | Direct 2-column ASCII files in `txt/` on Subaru archive |
| Subaru | MOIRCS | Y, J, H, Ks, K | 5 / 0 / 0 | SVO FPS | Photon (HgCdTe HAWAII-2) | NAOJ portal 302 redirect; SVO IDs provided |
| Subaru | FOCAS | U, B, V, R, I | 5 / 0 / 0 | Original (HTTP 200) | Photon (FDCCD) | Direct 2-column ASCII files in `camera/txt/` on NAOJ site |
| LCO | FourStar | J, H, Ks, J1, J2, J3, Hshort, Hlong | 3 / 5 / 0 | Original (HTTP 200) | Photon (HgCdTe H2RG) | Direct ASCII files `*_cam_optics.txt` at Carnegie Observatories |
| LCO | IMACS | B, V, R, I, g, r, i, z | 8 / 0 / 0 | SVO FPS | Photon (CCD) | LCO website restructured; Dressler et al. (2011) curves in SVO |
| Magellan | MegaCam | u, g, r, i, z | 5 / 0 / 0 | Original (HTTP 200) | Photon (CCD) | McLeod et al. (2015) multi-column tables with separate components |
| NOT | ALFOSC | U_Bes, B_Bes, V_Bes, R_Bes, i_int, u/g/r/i/z_SDSS | 10 / 0 / 0 | Original (HTTP 200) | Photon (CCD) | Direct ASCII files in `curves-ascii/` at NOT site |
| WIYN | ODI | u, g, r, i, z, NB695 | 5 / 1 / 0 | Original (HTTP 200) | Photon (OTA CCD) | Direct ASCII files in `ODI/Observe/` on WIYN site |
| KPNO | Mosaic | B_Harris, V_Harris, R_Harris, I_NearlyMould, g/r/i/z_SDSS, k1026 | 8 / 1 / 0 | Original (HTTP 200) / SVO | Photon (CCD) | NOAO spectrophotometer laboratory measurements in NOIRLab archives |
