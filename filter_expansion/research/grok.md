STATUS: DONE

Grok, request d4e048346ba3ac2ef6c4b2a4, 2026-10-25. Research only. Claims were `grok.csv` and `grok.md`. No mirror writes and no code. File claims released.

Block B source packet: [grok.csv](grok.csv). 82 rows. REUSE points at prior research. HOLD means a numeric curve was not in hand. Licences were not cleared.

Mirror baseline: [index.json](https://nillmill.ddns.net/sed_tools/index.json), generated 2026-10-25T05:22:31.428526, 3,963 filters / 75 facilities. Spitzer, Swift, XMM, IRAS, Herschel, MSX, ISO, Hipparcos, Tycho, Planck, SPHEREx and PLATO are absent. GAIA/GAIA is present.

| Set | Disposition |
|---|---|
| Gaia on the mirror | Headers fetched. `G`, `Gbp`, `Grp`, `Grvs` are `GAIA/GAIA3` (EDR3/DR3). `Gbp_bright` and `Gbp_faint` are `GAIA/Gaia2m`. Skipped. Original EDR3 zip already in [claude.jsonl](../../../sed-tools-components/results/claude.jsonl). |
| DR2 revised and the DR2 catalogue passbands | New. [Revised zip](https://www.cosmos.esa.int/documents/29201/1645651/GaiaDR2_Revised_Passbands_ZeroPoints.zip/54db454f-69cb-ea0c-15be-f1b1f597f191) opened: nm, G/BP/RP, `99.99` means undefined. The other zip is the one that generated published DR2 magnitudes. |
| IRAC | [Prior verified ch1](../../../sed-tools-components/results/claude.jsonl). ch2–ch4 HEAD 200. Electrons/photon. |
| MIPS 24/70/160 | New file opened. Handbook: electrons per energy. 70 µm has negatives. |
| IRS peak-up blue/red | New files opened. RQExT is QE times gain times filter, electrons per photon. Negatives. |
| Swift UVOT | HOLD. Vega’s lookup of heasarc.gsfc.nasa.gov was 192.0.66.108 and returned 404, so ARFs were not opened. Photon-counting wording is in the indexed filter note. |
| XMM OM | [Prior six-filter effective areas](../../../sed-tools-components/round3/space/report.md). Areas in the ASCII file are m² times 1000. |
| IRAS | [Prior IPAC table](../../../sed-tools-components/round3/space/report.md). Keep the 25 µm printed row; do not rebuild it from components. |
| Herschel PACS 70/100/160 | New. Filter transmission only, µm. Bolometers. |
| Herschel SPIRE 250/350/500 | New. One RSRF FITS, frequency in GHz, tiny negatives. |
| MSX A/C/D/E | [Prior RSR tables](../../../sed-tools-components/round3/space/report.md). B1/B2 are not in that directory. |
| ISOCAM | SVO ids from an indexed browse excerpt. VOTables not downloaded. Handbook curves are filter convolved with QE. |
| ISOPHOT main bands | HOLD. Handbook defines the bandpass; the Cal-G file was not found as a public URL. |
| Hipparcos Hp, Tycho BT/VT | [Bessell & Murphy 2012 table 2](https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table2.dat). Photonic passbands. |
| Planck LFI/HFI | [Prior RIMO extraction](../../../sed-tools-components/round3/space/report.md). One row per band; files are per detector. HEMT radiometers and bolometers, energy. |
| SPHEREx bands 1–6 | [Prior tar record](../../../sed-tools-components/results/claude.jsonl). 437 MB tar not downloaded. H2RG, photon. Each band is many LVF channels. |
| PLATO | HOLD. Only the secondary pycheops column from round 3. No official throughput file found. CCD270 is photon-counting. |

Fresh checks: IRAC ch2–ch4, MIPS summary, both IRS peak-up files, PACS blue FITS, SPIRE RSRF FITS, Gaia revised zip, Bessell ReadMe, MSX directory. SVO timed out, so SVO ids are aliases, not re-verified VOTables.

Integration holds: Swift ARFs until HEASARC resolves; MSX B1/B2; ISOPHOT numeric Cal-G; PLATO official throughput; SPHEREx until a chosen sub-channel set is taken from the tar or from the pipeline products. Do not install a second copy of GAIA3 or Gaia2m BP bright/faint.

Validation: 82 data rows, 15 columns, no duplicate facility/instrument/band. No tests, research files only. No account usage percentage is available from this session.
