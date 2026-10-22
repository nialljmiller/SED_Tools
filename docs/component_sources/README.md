# Facility and instrument component sources

This is the research inventory for expanding SED_Tools component coverage.
SVO's complete browse catalogue is the starting list, including planetary and
Earth-observation missions, generic photometric systems and commercial cameras.
These entries are preserved as SVO names them; a browse group is not necessarily
a physical observatory or a distinct instrument. Additional facilities can be
added with `origin=additional` and an authoritative reference.

The 2026-10-20 snapshot contains **206 facility/group entries and 649 instrument
entries**, with no failed or empty facility listings. All rows have a research
owner; component-source review is a separate, ongoing task.

## Files

- [Research report](report.md): current coverage, instrument index, evidence
  links, limitations and implementation priorities.
- [Component matrix](review_matrix.csv): component and source availability from
  the saved research checkpoints; these are not blanket verification approvals.
- [Source links](source_links.csv): original and fallback URLs, units, versions,
  retrieval instructions and component roles in a sortable table.
- [Vega follow-up handoff](supplemental_handoff.md): later ESO, Gemini, Subaru,
  Rubin and other source harvests, their corrected audit, provenance for
  archive members and implementation priorities. Supplemental worker reviews
  are evidence checkpoints, not automatic upgrades to the component matrix.
- `research_manifest.json`: checkpoint hashes and structural validation results.
- `build_report.py`: rebuilds the matrix, source-link table, report and manifest
  from the saved local checkpoints without making network requests.
- `facilities.csv`: every SVO facility/group, its browse link and discovery status.
- `instruments.csv`: every discovered facility/instrument pair, its SVO browse
  link, review state and research owner. Instrument variants remain distinct.
- `snapshot.json`: discovery time, counts, missing instrument lists and errors.
- `research_brief.md`: assignments, evidence requirements and the source format.
- `sources/`: source records returned by the research agents, one JSON object per
  line, plus readable summaries. Candidate research is not runtime configuration.

Discovery completeness and component research completeness are separate. An
instrument marked `unreviewed` is on the work list, not evidence that its team
publishes no components. Likewise, an HTTP success alone does not establish that
a link provides usable throughput data.

The component roles are `filter`, `optics`, `qe`, `atmosphere` and `published`
(the published response, which may contain several factors). Each record must
explain what is included and distinguish separate measurements from combined or
derived curves. Atmosphere sources must identify the applicable site and
conditions. Detector quantum efficiency and photon/energy response convention
are recorded separately.

A source record may use `components` when one original table or archive holds
several distinct factors. Its `includes` field identifies the factors; this
does not imply that an inaccessible archive has been inspected.

The first result of the supplemental audit is the shared
[`tabular`](../../sed_tools/component_sources/tabular.py) builder: 11
HAWK-I, 33 NACO, 24 SPHERE/IRDIS, 22 Gemini GSAOI and six NIFS filter curves. Their original tables can
be enumerated without SVO; dual IRDIS channels are kept distinct. These are
filter-only scans, and the builder does not infer missing optics or QE. The
remaining evidence is retained here for source-by-source interpretation; a
numeric file in the harvest does not automatically become an enabled curve.

Coordination lives on Vega at
`/home/nill/agent-share/projects/sed-tools-components/`. Codex maintains the master
inventory; each research agent writes its own source and review files. Their
findings require review before enabling a component downloader or mirroring data
on NJM. This inventory does not deploy the website or download bulk data.

## Research queue

The [current brief](round2_brief.md) supersedes the historical first batch's
12-family cap. Each researcher retains their partition and aims to review about
one quarter of its entries cumulatively, starting with widely used filter sets.
The [priority queue](priority_queue.csv) retains all 649 entries.

| Agent | Assigned instrument entries | Current milestone | Responsibility |
|---|---:|---:|---|
| Claude | 96 | 24 | Space astronomy |
| Grok | 280 | 70 | Ground facilities |
| Gemini | 273 | 69 | Other systems, planetary and Earth observation |
| Codex | — | Independent audit | Check evidence, interpretation and coverage |

Inspect current status on Vega with `agent-chat tasks`. Current research lives
in the shared project's `round2/` directory, with local checkpoints under
`sources/round2/`. The report distinguishes checkpoint coverage from completed
verification. The master inventory's `unreviewed` field is the original discovery
state; consult `review_matrix.csv` for current research status.
