# Independent component-source research: quarter-coverage milestone

Reference: `sed-components-quarter-20261020`.
Coordinator: Mac Codex. Independent verifier: Codex on Vega.

This brief supersedes the earlier limit of 12 instrument families. Continue from
the saved results; do not restart discovery or discard useful previous evidence.

## User's scope

Keep the existing researcher partitions. Each researcher should aim to
characterize approximately 25% of their assigned SVO entries, working from the
most widely used filter sets towards the less widely used ones. Codex verifies
the other researchers' work. The deliverable is a detailed report with links,
information and retrieval instructions for later implementation.

The aim is an independently sourced service with every obtainable optical-path
component, not a mirror dependent on SVO. Search original instrument teams,
observatory and mission archives, official repositories, calibration databases,
instrument manuals, original papers and their supplementary tables. Follow
references and historical archive links. SVO may help discover a lead, but its
hosted data are the fallback when a substantive original-source search fails.

No application code or website changes are part of this milestone.

## Coverage targets and ownership

| Researcher | Existing partition | Baseline entries | Cumulative target |
|---|---|---:|---:|
| Claude | Space astronomy | 96 | approximately 24 |
| Grok | Ground observatories and surveys | 280 | approximately 70 |
| Gemini | Other missions, cameras and photometric systems | 273 | approximately 69 |
| Codex | Independent verification of all three partitions | all delivered findings | audit the milestone |

The targets include previous work, once its evidence has been reconciled with
this brief. Existing reported coverage was 13, 14 and 52 SVO entries respectively.
SPHEREx is additional and must not inflate the 96-entry space baseline. An extra
source for an existing Sony entry is not another instrument.

These are instrument-coverage targets, not token budgets, file counts or a new
12-family limit. Distinct SVO keys count separately only when their particular
release, detector, optical mode and applicability have been characterized.
Report distinct families separately as context. It is fine to exceed a target
slightly when a shared primary source naturally covers related instruments.

Use `round2_plan.json` and `priority_queue.csv` as the initial ordered work list.
Popularity is a qualitative judgement about scientific use and public-survey
reach, not a measured usage statistic. The plan prioritizes major survey systems
and widely used observatory instruments, rather than convenient alphabetical
groups. Improve that order if evidence warrants it, recording the reason. Do not
skip difficult high-priority entries merely to increase the count.

Each researcher owns only `round2/<agent>/`. Codex owns `round2/codex/` and must
not edit another researcher's evidence. Mac Codex owns the master inventory and
final consolidation. Existing `results/` files remain historical evidence.

## What constitutes a characterized instrument

Produce a dossier with the best obtainable information for that specific
instrument/release. Finding a homepage or copying an SVO link is not enough.
Missing components are acceptable conclusions when supported by a documented
search. A dossier must include:

1. Identity, aliases, versions, observing modes, bands and detector variants.
   Explain the relationship to the SVO browse key and to any newer product.
2. The original published passbands and an explicit status for **every** role:
   filter, telescope/instrument optics, detector QE, atmosphere and full response.
   Separate mirrors, lenses, windows and dichroics further when available.
3. Direct file/archive/API links and explanatory primary references. If only an
   index is available, give the actual selection procedure and filename pattern
   supported by the source, rather than inventing links.
4. Retrieval instructions: file format, archive members or columns, API query,
   authentication if needed, approximate size if known, and reproducible
   processing needed to extract a usable curve. Do not download multi-GB bundles
   merely to check that they exist; inspect manifests or range-accessible members.
   Explain how to enumerate the bands and modes from the original source itself.
   An independent downloader must not require an SVO request to obtain its band
   list, filenames or essential detector metadata.
5. Wavelength and response units, percentage versus fraction, photon versus
   energy convention, any wavelength weighting, normalization and effective-area
   factors. Record unknowns explicitly. QE alone does not establish the
   convention of a published passband.
6. What each curve includes and whether it is measured, combined or derived.
   Explain justified division/interpolation steps and their validity limits.
   Never silently substitute a detector, release, site or atmosphere.
7. For terrestrial atmosphere: site, airmass, PWV and other supplied conditions,
   model identity and version. For space instruments mark it not applicable.
8. Provenance, version/date, citation and redistribution information supported by
   evidence. Public download access is not proof of a particular licence.
9. A search log for unavailable components: references and locations inspected,
   access failures and plausible next leads. Label a curve SVO-only only after
   searching the original references and plausible successor archives.
10. A short implementation recipe and scientific pitfalls for later coding.

A completed search may honestly find that some factors are not separately
published. Distinguish that from unsearched, documented-but-unretrieved and
access-blocked data. Do not claim that absence from one page proves nonexistence.

## Research deliverables

Write cumulative results under your own `round2/<agent>/` directory:

- `sources.jsonl`: the existing source-record schema, with evidence preserved and
  corrections to previous claims. Add `source_basis` (`original`, `svo_fallback`
  or `secondary_lead`), `retrieval_instructions`, `wavelength_unit`,
  `response_unit`, and `search_notes`. Record a representative-file inspection
  as such; it does not verify every sibling file.
- `reviews.csv`: one row per characterized SVO instrument, with columns
  `facility_key,instrument_key,review_status,filter_status,optics_status,qe_status,atmosphere_status,published_status,independent_source_status,notes`.
  Use component states `available`, `documented`, `derived`, `combined_only`, `not_found`,
  `blocked`, `not_applicable` or `unknown`. Use `independent_source_status`
  `original_files`, `original_documentation`, `mixed` or `svo_only`.
- `report.md`: readable per-instrument dossiers containing the above information,
  links and later coding instructions. Group common retrieval procedures to avoid
  repetition, but state each instrument's applicability explicitly.
- `progress.json`: target, characterized SVO keys, remaining priority keys,
  limitations, last update UTC and `complete` boolean. Checkpoint atomically
  during substantive work so the verifier can inspect stable output.

Additional instruments should be clearly separated from baseline counts.
Do not mark the milestone complete merely because some files were written.
If a genuine source/provider/budget blocker prevents the target, checkpoint
everything and report exactly what remains; do not silently reduce the target.
Follow current provider and budget policies without inventing quota measurements.

## Codex verification

Start by auditing the saved first-round results and this plan. Independently open
primary evidence and inspect representative data, table definitions and archive
contents for substantive claims. Validate HTTP successes against actual content;
HTML landing pages are not curve files. Identify unstable URLs, wrong releases,
units, double-counted throughput factors and unsupported extrapolation across
bands or detectors. Verify the per-instrument counts against the assigned keys.

Check the earlier weak points explicitly: 2MASS's incorrect total-only
classification; already wavelength-weighted RSRs; LSST release mapping and
atmosphere conditions; HSC r versus r2/i versus i2; VISTA component scales and
supplied atmosphere; ZTF detector mixtures; JWST archive/file version differences;
Euclid component definitions; Cassini filter recovery assumptions; Sony IMX455
provenance; generic-system detector conventions; and unsupported licence claims.

Researchers run concurrently. Begin with stable historical records and audit
available checkpoints. Record which source-file hashes and instrument keys were
actually audited. Do not claim to have verified work that arrived after the
audit. If the research is still running when the first audit pass finishes,
return a clearly scoped preliminary audit; Mac Codex will request the final pass
on completed outputs. Avoid repeated model calls just to poll files.

Write `round2/codex/report.md`, `issues.jsonl`, and `coverage.json`. Each issue
must identify instrument/source, severity, exact evidence, proposed correction
and whether it blocks using the result. State which findings are verified,
which are only plausible leads, and what could not be independently checked.
Do not fix application code. Communicate concrete findings to the coordinator
and researchers as needed, without starting duplicate research assignments.

## Final synthesis

Mac Codex will reconcile the research and audit into one report in
`docs/component_sources/`, with instrument coverage, component availability,
primary links, retrieval recipes, SVO fallback cases, corrections, limitations
and implementation priorities. This report will support a later coding task.
Progress messages should be brief and limited to substantive developments.
