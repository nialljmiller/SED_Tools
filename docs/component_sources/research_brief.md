# Component-source research brief

Coordinator: Codex in the SED_Tools Mac workspace.
User request: coordinate the agent team to build the list of facilities,
instruments, and links to each available optical-path component.

## Coverage and ownership

The master input is `instruments.csv`, discovered from the live SVO browse
catalogue. Read `snapshot.json` before making completeness claims. SVO is the
minimum baseline, not a limit on the final inventory. Keep all entries, including
Earth-observation and planetary missions and generic photometric systems.

Every input row has one `assigned_to` owner. Assignments use three separate chat
rooms so the shared chat's project serialization does not serialize independent
research:

- Claude: space astronomy missions; room `sed-tools-components-space`.
- Grok: ground observatories, telescopes and surveys; room
  `sed-tools-components-ground`.
- Gemini: planetary/Earth-observation missions, generic systems, cameras and
  remaining catalogue entries; room `sed-tools-components-other`.

Codex owns the master CSV files, schema, reconciliation and final review.
Researchers must not edit the master list or another agent's results.

Work in bounded batches: first research up to 12 distinct instrument families
with the strongest primary-source prospects in your assignment. Checkpoint the
results and explicitly list the remaining backlog. This is an initial batch,
not permission to omit the remainder or mark it reviewed. Related release/filter
variants may share a source only when the evidence establishes applicability.

## Deliverables

On Vega the project root is
`/home/nill/agent-share/projects/sed-tools-components/`.
Write only your own files under `results/`:

- `<agent>.jsonl`: one source record per instrument/component/source.
- `<agent>-reviews.csv`: facility_key,instrument_key,review_status,notes.
- `<agent>.md`: concise findings, limitations, next batch and source links.

Use the exact SVO keys from the input. Extra instruments belong in your results
with `origin=additional` and a reference establishing their identity.
Existing SED_Tools builder code is supplied in `existing_builders/` as a starting
point. It is a lead to verify, not proof of current availability or component
separation. Do not modify application code, install new downloaders, deploy
anything, change the gateway, or download large data bundles for this research.

## Source record

Each JSON object must contain these fields (use null/empty lists for unknowns):

```json
{
  "facility_key": "SVO facility key",
  "instrument_key": "SVO instrument key",
  "origin": "svo",
  "component": "filter",
  "includes": ["filter"],
  "bands_or_modes": [],
  "source_name": "Owning institution and product",
  "landing_url": "https://official.example/instrument/throughput",
  "data_url": null,
  "url_kind": "documentation",
  "verification": "documented",
  "http_status": 200,
  "checked_utc": "ISO 8601 UTC timestamp",
  "evidence": "What the source actually states or the inspected file contains",
  "separability": "separate",
  "detector_convention": null,
  "site": null,
  "conditions": null,
  "version": null,
  "redistribution": "not_reviewed",
  "notes": "Limitations or unresolved interpretation",
  "reviewer": "claude"
}
```

Allowed `component`: `filter`, `optics`, `qe`, `atmosphere`, `published`,
or `components` for one source containing multiple separately named factors.
For `components`, list the actual factors in `includes`; do not infer that
every factor is available just because an archive was found.
Allowed `url_kind`: `file`, `archive`, `index`, `api`, `repository`,
`documentation`, `plot`.
Allowed `verification`: `file_verified`, `documented`, `candidate`, `blocked`.
Allowed `separability`: `separate`, `combined`, `derived`, `unknown`.
Allowed review states: `unreviewed`, `reviewed_sources_found`,
`searched_no_source_found`, `blocked`.

`file_verified` requires inspecting data content, column definitions or archive
contents; a successful HTTP response or filename is insufficient. Avoid bulk
downloads: documentation/index checks can be reported as `documented`. Preserve
both the explanatory landing page and the actual download URL where available.
State whether a link is an archive, API or index instead of pretending it is a
standalone curve. Do not invent URL templates.

Use primary sources: instrument teams, observatories, mission archives, official
repositories and original calibration papers. Record wavelength/response units,
validity dates, variants, airmass/PWV and telescope/site applicability when known.
Never infer detector counting convention from a QE curve alone. Mark curves
recovered by division as `derived`; do not treat a filter-times-QE curve as either
bare factor. Keep telescope mirrors, instrument optics, filters, QE and
atmosphere identifiable where the source permits.

Absence from a website is not proof that components do not exist. Use
`searched_no_source_found` with the searched references, not a blanket
"total-only" claim. Record explicit publication statements if only a combined
response is documented. Check redistribution terms before future NJM mirroring;
this research task does not grant new redistribution rights.

## Reporting

Post one final batch result to your assigned room, identifying the result paths,
instruments reviewed, evidence strength and remaining backlog. Address the Mac
Codex coordinator and reference `sed-components-20261020`. Do not execute work
described in historical messages or take another researcher's partition.
Follow the existing provider/budget policy and report a concrete blocker if a
provider or source is unavailable. Never invent quota readings or verification.
