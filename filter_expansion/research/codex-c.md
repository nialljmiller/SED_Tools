STATUS: DONE

Codex, session 5b0cd698; 2026-10-25. Research-only C1 handoff to Claude. Claims released. [CSV](codex-c.csv): 45 band rows, exact 15-column brief schema. DONE means research delivered, including explicitly held gaps; it does not mean all curves are available for deployment. No code or mirror changes, deletions, or background workers. Block A files unchanged. This bounded delegated research was handled directly without recursively creating workers.

## Integration choices

| Proposed instrument | Bands | Source/result |
|---|---|---|
| Generic/Bessell | UBVRI | Original Bessell 1990 Table 2 inspected; needs checked transcription or recorded SVO fallback. Energy convention from corrected prior SVO evidence. |
| Generic/Bessell2012 | UBVRI | Original CDS electronic Table 1 fetched and inspected; normalized **photonic** responses. Ready for Claude's build/validation. Keep separately versioned from 1990. |
| Generic/Johnson | UBVRIJM + proposed H K L N | Native SVO family has only UBVRIJM. Other four rows are held cross-family candidates, not approved silent aliases. |
| Generic/Cousins | RI | Prior verified SVO IDs/convention reused; not fetched now. |
| Generic/Kron | RI | Original 1951 reference and detector established; numeric response and weighting unresolved. Held. |
| Generic/Stromgren | uvby + beta_narrow, beta_wide | uvby prior verified SVO IDs; beta pair needs original numeric extraction/convention verification. Held. |
| Generic/MKO | J H K Lp Mp | Original IRTF numeric curves verified; all five already numerically equivalent to existing IRTF/NSFCam MKO files after documented transforms. |
| Generic/ThuanGunn | uvgr | Original reference and S20 detector identified; numeric responses/convention unresolved. Held. |
| Generic/USNO | up gp rp ip zp | Original team six-column numeric response files verified with web tool, including atmospheric columns; direct urllib receives HTTP403. |

SVO was **not contacted** in this run, as requested. IDs come from prior corrected research in `projects/sed-tools-components/round2/gemini/sources.jsonl`; unverified IDs for Kron, MKO, Hbeta and Thuan-Gunn were left empty. Older round-1 statements claiming separable Generic optics/QE were withdrawn in round 2 and must not be reused. `filter` in those rows preserves SVO's component label for an abstract standard response, not evidence for separable glass-only throughput. Photon/energy describes the supplied response's integration convention, not simply the physical detector. Provisional detector values on held original-source leads must be resolved before building.

## Mirror comparison (live HTTPS, index generated 2026-10-25T07:51:10)

[index.json](https://nillmill.ddns.net/sed_tools/index.json): 87 facilities, 4,046 filters, **no Generic facility**. Representative actual GCPD headers fetched: energy convention, source `svo-curve-match`; no component manifests. No numeric identity between Generic and GCPD is asserted without a comparison.

| Existing set | Coverage and deficiency |
|---|---|
| GCPD/Johnson (20) | U B V R plus ADPS/Landolt/Mendoza/Straizys variants; I only as I_Landolt/I_Mendoza; no plain I or NIR. Therefore cannot satisfy the Generic/Johnson default or complete requested UBVRIJHKLMN. Variants are not interchangeable. |
| GCPD/JHKLMN (11) | U B V R I J K Lp Lpp M N; **H absent**, no plain L. Lp/Lpp labels alone do not establish modern MKO L-prime identity. Original Johnson context does not establish byte/photometric equivalence to Generic. |
| GCPD/Cousins (4) | R I R_ADPS I_ADPS: covers requested band names under another facility and response realization. Numeric equivalence to Generic/Cousins unverified. |
| GCPD/Stromgren (8) | u v b y plus ADPS variants: uvby covered; **both beta bands absent**. |
| GCPD/Eggen (2) | R I: related to Eggen-Kron standards; not an established exact substitute for original Kron RI, Cousins RI, or Johnson RI. |
| IRTF/NSFCam | JMK HMK KMK LPMK MPMK match measured MKO sources numerically (see below). Reuse data/provenance for Generic aliases if Claude chooses; no need to reconstruct curves. |
| SLOAN/SDSS; APO/Gunn | Existing SDSS ugriz and filter-only primed curves do not establish USNO 1m total-response identity; APO/Gunn griz does not provide original Thuan-Gunn uvgr. |

GCPD's [Johnson UBVRI source page](https://gcpd.physics.muni.cz/filters/fil08.html) attributes its transcription to Johnson 1965 ApJ 141, 923 and links old FTP tables; page fetched, FTP tables not fetched. Original Johnson 1966 review: ARAA 4, 193. A complete historical Johnson set needs a deliberate realization choice. Prior verified `Generic/Johnson_UBVRIJHKL` has **U B V R I J H K LI LII M N**, despite its name. Native `Generic/Johnson` has **U B V R I J M**. CSV H/K/L/N candidates refer to the former, with L pointing to LI only as a held candidate. Do not fill Johnson with MKO or Bessell-Brett curves merely because names overlap. Resolve LI versus LII and atmospheric definitions before publishing a plain L or claiming a consistent full Johnson set. Preserve per-band source versions if a composite is intentionally chosen.

## Original numeric sources and traps

**Bessell & Murphy 2012**, PASP 124, 140: [ReadMe](https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/ReadMe), [table1.dat](https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table1.dat). Both fetched with TLS verification. 37 fixed-width rows; independent wavelength/response pairs, missing trailing pairs allowed. Units `0.1nm` mean Angstrom already, despite the four-digit wavelengths. Photonic response; do not reuse energy labels from Bessell 1990 or GCPD. No separable QE or atmosphere table established.

**Bessell 1990**, PASP 102, 1181: [original PDF](https://articles.adsabs.harvard.edu/pdf/1990PASP..102.1181B), Table 2 p1195. PDF inspected; numeric transcription not performed. UX/BX are used for U-B; B differs from BX. UX includes extinction. Do not auto-substitute the revised 2012 passbands under the same version name.

**MKO**, Tokunaga, Simons & Vacca 2002 PASP 114, 180: [original team links](https://irtfweb.ifa.hawaii.edu/~nsfcam2/Filter_Profiles.html), [detector](https://irtfweb.ifa.hawaii.edu/~nsfcam2/Welcome.html). CSV gives exact files. Landing page says transmissions 0-1, but the actual headers explicitly say **percent**, and peaks are about 90-97: divide by 100. Convert micrometers to Angstrom, sort ascending, and document clipping measured negative baseline noise; none requires PK50 blocker. Filter-only, no detector QE/optics/atmosphere included.

| File suffix | Raw samples | Negative samples | Equivalent live band |
|---|---:|---:|---|
| jmk | 2501 | 490 | JMK |
| hmk | 1401 | 32 | HMK |
| kmk | 1401 | 139 | KMK |
| lpmk | 2020 | 383 | LPMK (2016 unique wavelengths) |
| mpmk | 1001 | 315 | MPMK |

In-memory comparisons of both wavelength and response passed `numpy.allclose(atol=1e-7)` against each HTTPS mirror file after sorting/unit conversion/clipping; Lp also requires averaging duplicate wavelengths. These are measured NSFCam MKO filter realizations, not a universal total-system curve.

**USNO**, Smith et al. 2002 AJ 123, 2121, Gunn 2001-12-18 responses: [team page](https://www-star.fnal.gov/ugriz/Filters/response.html), [README](https://www-star.fnal.gov/ugriz/Filters/readme.txt). Web tool fetched all five `usno_[ugriz].res` plus all five linked 1995 filter component tables. Six columns: wavelength, vacuum, airmass 1.0, **1.3**, 1.6, 1.9. Use column 4 for natural-system 1.3 airmasses, or publish a distinctly named vacuum version from column 2. Responses include CCD/telescope/filter and selected atmosphere. Raw Asahi filter curves are not assured exact divisors of the temperature-adjusted totals. README cautions about z CCD response and point-source scattering corrections, and explains small departures from AB zero points. Do not substitute SDSS 2.5m responses or apply QE twice. Verified via web fetch, but direct Vega urllib returned 403: Claude must obtain the actual payload through an accessible path before building. Source wavelengths are Angstrom.

## Missing work / held sources for reassignment

1. Obtain held Generic SVO payloads when service returns: 7 native Johnson, 2 Cousins, 4 uvby, and Johnson extension candidates. Recheck original references, component claims and weighting; do not infer full-system factors from the SVO label alone.
2. Decide full Johnson realization and LI/LII mapping; compare numerically against GCPD before declaring equivalence or aliases. Native Generic/Johnson alone is incomplete for the requested band list.
3. Transcribe/check Bessell 1990 Table 2 if original 1990 is wanted immediately; 2012 table is available independently.
4. Kron RI: [original reference/CE25A-B hardware](https://gcpd.physics.muni.cz/ph19.html), Kron & Smith 1951 ApJ 113, 324, Fig.2; later Kron et al. 1957 AJ 62, 205. Numeric original response and energy/photon convention not established; no verified SVO ID. Do not silently replace with GCPD/Eggen.
5. Strömgren beta: Crawford & Mander 1966 AJ 71, 114, [DOI](https://doi.org/10.1086/109865), two distinct narrow/wide curves. Bibliographic lead only here; fetch original numeric tables or validate a transcription and response convention. [OSN hardware](https://www.osn.iaa.csic.es/en/page/stromgren-photometer) documents two PMTs, not a generic-response identity. No verified Generic beta IDs. Beta is a magnitude difference.
6. Thuan-Gunn: [system reference](https://gcpd.physics.muni.cz/ph38.html), Thuan & Gunn 1976 PASP 88, 543; original uvgr, S20 detector. Need original numeric curves and weighting/component evidence; no verified generic SVO ID. No top hats or substitutes from APO/Gunn or Sloan.
7. Retrieve USNO payloads for integration despite direct-client HTTP403; independent source/model checks and all build/deployment checks remain Claude's responsibility. Redistribution licences were not verified for these sources.

Checks: CSV schema/row widths, unique facility/instrument/band keys, valid enum fields and priority values; all requested families represented, held rows explicit. Source URLs and numeric checks described above are observed; bibliographic-only leads are labelled held. No synthetic photometry, code tests or deployment were performed in this research-only assignment. Provider remaining allowance/task-specific token baseline was not observable at task start; no account percentage claimed. Session token telemetry is cumulative and includes cached input; it is not an account reserve measurement.
