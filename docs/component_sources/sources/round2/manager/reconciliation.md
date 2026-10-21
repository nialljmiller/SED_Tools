# Manager reconciliation after the Codex audit

The Codex [final audit](../codex/final_audit.md) inspected one exact input snapshot. After it began, the manager corrected source-role metadata, re-fetched a complete HiPERCAM file, and made the old WISE normalization note explicitly superseded. Its scientific samples remain useful, but its structural counts and hashes describe the earlier files. The [current manifest](../../../research_manifest.json) and regenerated [matrix](../../../review_matrix.csv) describe the current local checkpoint.

The Codex markdown says “287 source rows”; its own JSON counts 79 Claude + 130 Grok + 92 Gemini = **301**. Current rows also total 301, with 163 assigned SVO review keys plus one extra SPHEREx row. The audit's warning about ten extended Claude roles referred to the initial snapshot. The current source roles are the five physical roles plus `components` for a multi-factor source; the blocked Pandeia lead uses that category and is not claimed as inspected. The original-file classification now has at least one linked, verified original record for every one of its 106 keys. This proves linkage, not every sibling band or factor.

After the Codex input boundary, these input files changed (exact current SHA-256 below):

- `round2/manager/schema_corrections.md` — `a6af78c6466df7c1355ef995af1426ffc328ba8b081d3a063b703df36df9d400`
- `round2/claude/sources.jsonl` — `542dddfa6c590a81a3a601194317486aee2294bc410b19586c39274b232d27e6`
- `round2/grok/sources.jsonl` — `eaa6be52d3b222d44b1f4cd6afa8a47563caef95b719b0e58de0c1d7032d9da4`
- `round2/grok/reviews.csv` — `a5435e2b65b5531fa36977faebf04fcadc0184ecc73f893f0a068a5d52aefc56`

The latest static validation parses all 301 JSONL source records and 164 CSV review rows, checks unique owner assignments against the 649-key inventory, validates source role and basis, requires retrieval instructions, and checks every `original_files` classification for a linked verified original. It reports zero structural errors. [HiPERCAM evidence](hcam_check.md) covers the complete original g curve only. [Manager WISE and PACS checks](report.md) retain their numeric limits.

Retrieval remains incomplete for parts of the 70-key ground milestone: **12 Grok source rows** name local files instead of providing a direct absolute data URL. A [manager follow-up](ground_source_followup.md) found official landing pages for ten of the eleven previously unlinked searches, leaving **one unlinked DENIS record**. The new pages are leads, not verified numeric files; they do not make those component sets complete. These rows require working original files or explicit SVO fallback before coding. The audit did not resolve the original LSST and NIRCam release mappings, all detector variants, units on several ground files, or redistribution rights.

The final scientific audit spot-checked WISE W1, Cassini's filter-pair definition, PS1 documentation and one GOES-16 EUMETSAT channel. It did not independently sample JPSS J1 or NOAA's separate ABI component columns. Those remain researcher claims with the stated original-source links and search notes.
