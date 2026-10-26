# filter_info: descriptive metadata and references (`info.json`)

Each instrument directory on NJM gets an `info.json` next to `properties.json`.

- **`properties.json`** holds numbers computed from the curve.
- **`info.json`** holds what a curve *is*:
  - descriptions and photometric system;
  - the references to cite;
  - where the curve came from;
  - what SVO said about it, and when.

`index.json` points to it with `"info": "info.json"`.

## Pipeline

`generate_index.py` never touches the network. It joins committed inputs with the parsed SVO records kept on NJM:

| Step | Command | Output |
|---|---|---|
| 1. Harvest SVO records | phase-3 Internet Archive harvest, or `python -m filter_info.svo_archive fetch QUEUE.json DIR` | raw captures plus `manifest.jsonl`, archived on NJM under `source_evidence/` |
| 2. Parse them | `python -m filter_info.svo_archive build source_evidence/svo-archive/parsed $(grep -v '^#' filter_info/svo_manifests.txt)` | `<Facility>.json` and `_service.json` on NJM, beside the raw captures |
| 3. Generate | `python generate_index.py data` | every `info.json`, plus `filter_info/unresolved.txt` |
| 4. Resolve new references | `python -m filter_info.references resolve` | `filter_info/references.json` |
| 5. Generate again | `python generate_index.py data` | `info.json` with full bibliography entries |

**One command:** `filter_info/refresh.sh`, run on NJM from the checkout, does steps 2–5.

**Resolution sources:**
- **Bibcodes:** ADS when `ADS_DEV_KEY` is set. Otherwise ADS's public link resolver, then a Crossref search accepted only when year, volume, page and first-author initial all agree.
- **DOIs:** doi.org.
- **arXiv IDs:** the arXiv API.

Each key is looked up once. `--retry` tries the unresolved ones again.

**When SVO returns,** its full harvest goes through steps 1–5 unchanged.

**Where the parsed SVO records live.** They're about 2 KB per SVO ID, roughly 20–30 MB for all of SVO, so they stay out of git.
- **Location:** NJM keeps them, with the raw captures, in `source_evidence/svo-archive/` (override with `SED_SVO_RECORDS`).
- **Reproducibility:** `index.json` records the hash of the parsed records under `filters.info_definitions.svo_records`, so any rebuild can be checked.
- **Manifests:** `filter_info/svo_manifests.txt` lists every harvest manifest a rebuild must pass.

## Precedence

1. **SVO's archived records come first,** matched to each curve in this order:
   - by the curve's `FilterID`;
   - by its provenance `svo_id`;
   - by its band name, only when SVO's pivot wavelength agrees within 1 % and exactly one SVO band qualifies.

   When several archived versions exist, the one whose curve is identical to NJM's is used.
2. **NJM provenance overrides SVO:** the header row, `manifest.json`, `filter_expansion/provenance/`.
3. **Hand-checked files override both:**
   - `curated/instruments/<Facility>__<Instrument>.json` holds `description`, `facility_description`, `notes`, `links`, `references`, `bands.<band>.<field>`
     (`bands.*.<field>` applies to every band unless the band sets its own),
     `acknowledgements` (the provider's requested text: `[{for, text, url, references, applies_to}]`) and `licence`
     (`{redistribution: public|attribution|permission|restricted|unknown, terms, url}`);
   - `curated/references.json` holds entries no service resolves.

The curve header row stays authoritative for `DetectorType`. SVO's value is only reported, under `svo`, with a note when it disagrees.

## `info.json` (schema 1)

```
schema, facility, instrument
description, facility_description, links[{url, label, source}], notes
references[key]             everything to cite for this instrument
acknowledgements[{for, text, url, references, applies_to}]   providers' requested texts, then SVO's
licence{redistribution, terms, url}   terms under which the curves are redistributed
bands.<band>:
  Description, Comments, PhotSystem, Band, MagSys, PhotCalID, ZeroPointType   (SVO PARAM names)
  ProfileReference[{ref, label}], CalibrationReference[{ref, label}]
  specified{WavelengthRef, ..., ZeroPoint{Vega|AB|ST: {Jy, erg/cm2/s/A}}}   owner-published values
  svo{filterID, matched_by, curve: identical|differs|not archived, DetectorType,
      components, added, updated, Fsun, AfAv, captures{votable, browse}}
  curve{FilterID, DetectorType, DetectorTypeSource, Components, ComponentsSource, sources[key], cleaning}
  references[key]           everything to cite for this band
svo_unmatched[band]
bibliography.<key>: short, type, title, authors, year, journal, volume, page, doi, bibcode,
                    arxiv, url, publisher_url, bibtex, same_as
```

**Reference keys:** `bibcode:…`, `doi:…`, `arxiv:…` or `url:…`. Keys for the same work are linked with `same_as`, and the bibcode form is canonical.
