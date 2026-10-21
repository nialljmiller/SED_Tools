# Manager normalization of source roles

Claude's 16 later source records omitted the required `component` field. The manager added `qe` to the Roman WFI detector-QE record and `published` to 15 records that document a combined passband, effective area, or its definition. No source URL, numeric claim, unit, or verification label was changed. This is metadata classification for the research index; published does not assert an independently verified numeric total.

- WISE/WISE: published — IPAC WISE All-Sky Explanatory Supplement, RSR files
- WISE/WISE: published — Wright et al. 2010 AJ 140 1868
- Spitzer/MIPS: published — IRSA MIPS Spectral Response (MIPSfiltsumm)
- GALEX/GALEX: published — Morrissey et al. 2007 ApJS 173 682
- Herschel/Pacs: published — Poglitsch et al. 2010 A&A 518 L2
- Herschel/SPIRE: published — Griffin et al. 2010 A&A 518 L3
- GAIA/GAIA3: published — Jordi et al. 2010 A&A 523 A48
- GAIA/GAIA3: published — Riello et al. 2021 A&A 649 A3
- TESS/TESS: published — Ricker et al. 2015 JATIS 1 014003
- TESS/TESS: published — TESS Instrument Response Function v2.0 (HEASARC)
- Kepler/Kepler: published — Kepler Instrument Handbook KSCI-19033-002
- Roman/WFI: qe — Roman WFI QE (FPSPerformance)
- Spitzer/MIPS: published — IRSA MIPS Instrument Handbook sec 2.2.4 Spectral Response
- GALEX/GALEX: published — GALEX archive ground-calibration effective area tables EA-fuv_im.tbl / EA-nuv_im.tbl
- WISE/WISE: published — IRSA RSR-W1..W4.txt peak check
- Herschel/Pacs: published — Manager independent check (Codex Mac) of PACS blue vs SVO

One Pandeia reference-data lead had used its archive name as a component role. The manager classified the blocked multi-factor archive as `components`; its `includes` list and `access_blocked` verification remain unchanged.

A WISE historical record marked `superseded_in_part` still stated “peak 1” in present tense. The manager changed that sentence to say explicitly that the original W1 maximum is 0.8798; no numeric file or verification status changed.
