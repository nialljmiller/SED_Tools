"""Build the research index from the saved agent checkpoints.

This script only reads local research files. It never downloads or publishes data.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "sources" / "round2"
OWNERS = {"claude": (96, 24), "grok": (280, 70), "gemini": (273, 69)}
ROLES = ("filter", "optics", "qe", "atmosphere", "published")
STATES = {
    "available", "documented", "derived", "combined_only", "not_found",
    "blocked", "not_applicable", "unknown",
}


def records(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build():
    inventory = {
        (row["facility_key"], row["instrument_key"]): row
        for row in records(ROOT / "instruments.csv")
    }
    reviewed = []
    source_rows = []
    counts = {}
    errors = []
    keys = set()
    hashes = {}

    for owner, (assigned, target) in OWNERS.items():
        folder = SOURCE_ROOT / owner
        rows = records(folder / "reviews.csv")
        sources = [json.loads(line) for line in (folder / "sources.jsonl").read_text().splitlines() if line]
        source_rows.extend({"agent": owner, **row,
                            "svo_browse_url": inventory.get(
                                (row["facility_key"], row["instrument_key"]), {}
                            ).get("svo_browse_url", "")}
                           for row in sources)
        source_keys = {(r["facility_key"], r["instrument_key"]) for r in sources}
        for source in sources:
            if source.get("component") not in {*ROLES, "components"}:
                errors.append(f"Invalid source role: {owner}/{source['facility_key']}/{source['instrument_key']}")
            if source.get("source_basis") not in {"original", "svo_fallback", "secondary_lead"}:
                errors.append(f"Invalid source basis: {owner}/{source['facility_key']}/{source['instrument_key']}")
            if not source.get("retrieval_instructions"):
                errors.append(f"No retrieval instructions: {owner}/{source['facility_key']}/{source['instrument_key']}")
        baseline = []
        for row in rows:
            key = (row["facility_key"], row["instrument_key"])
            if key in keys:
                errors.append(f"Duplicate review: {key}")
            keys.add(key)
            if key not in source_keys:
                errors.append(f"No source record: {key}")
            if row["independent_source_status"] == "original_files" and not any(
                source["facility_key"] == key[0] and source["instrument_key"] == key[1]
                and source.get("source_basis") == "original"
                and source.get("verification") in {"file_verified", "manager_verified"}
                for source in sources
            ):
                errors.append(f"No verified original file for {owner}/{key[0]}/{key[1]}")
            if key in inventory:
                baseline.append(row)
                if inventory[key]["assigned_to"] != owner:
                    errors.append(f"Wrong owner: {key}")
            for role in ROLES:
                if row[f"{role}_status"] not in STATES:
                    errors.append(f"Invalid {role} status: {key}")
            reviewed.append({"agent": owner, "origin": "svo" if key in inventory else "additional", **row})
        if len(baseline) != target:
            errors.append(f"{owner} reviewed {len(baseline)}, target {target}")
        counts[owner] = {"assigned": assigned, "target": target, "reviewed": len(baseline),
                         "basis": dict(Counter(r["independent_source_status"] for r in baseline)),
                         "sources": len(sources)}
        for path in folder.iterdir():
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    for folder in (SOURCE_ROOT / "codex", SOURCE_ROOT / "manager"):
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file():
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    fields = ["agent", "origin", "facility_key", "instrument_key", "review_status",
              *(f"{role}_status" for role in ROLES), "independent_source_status",
              "release_mapping", "component_search_scope", "superseded_claims", "notes"]
    with (ROOT / "review_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(reviewed)

    source_fields = ["agent", "facility_key", "instrument_key", "component", "source_basis",
                     "verification", "source_name", "landing_url", "data_url", "version",
                     "bands_or_modes", "wavelength_unit", "response_unit", "retrieval_instructions",
                     "search_notes", "redistribution", "svo_browse_url"]
    with (ROOT / "source_links.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, source_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(source_rows)

    baseline_count = sum(key in inventory for key in keys)
    basis = Counter(row["independent_source_status"] for row in reviewed if row["origin"] == "svo")
    unlinked_ground = sum(
        row["agent"] == "grok" and not row.get("landing_url") and not row.get("data_url")
        for row in source_rows
    )
    report = [
        "# Independent component-source research\n",
        f"The inventory contains **{len(inventory)} SVO facility/instrument browse keys** across 206 groups. "
        f"The current research milestone has dossiers for **{baseline_count} keys** "
        f"({baseline_count / len(inventory):.1%} of the inventory), plus SPHEREx outside the SVO baseline. "
        "A key is a browse entry, not necessarily a distinct physical instrument. "
        "The full [priority queue](priority_queue.csv) preserves the remaining entries, ordered "
        "qualitatively from widely used filter sets to narrower systems.\n",
        "## Coverage and limits\n",
        "| Owner | Assigned | Target | Reviewed keys | Original files | Original documentation | Mixed | SVO only |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for owner, item in counts.items():
        b = item["basis"]
        report.append(f"| {owner.title()} | {item['assigned']} | {item['target']} | {item['reviewed']} | "
                      f"{b.get('original_files', 0)} | {b.get('original_documentation', 0)} | "
                      f"{b.get('mixed', 0)} | {b.get('svo_only', 0)} |")
    report.extend([
        "",
        f"Across the {baseline_count} reviewed keys, researchers found some original numeric files for "
        f"{basis['original_files']}, original documentation for {basis['original_documentation']}, "
        f"mixed sources for {basis['mixed']}, and SVO-only data for {basis['svo_only']}. "
        "These are researcher classifications. One verified representative file does not establish "
        "every band, component, observing mode or detector variant. A combined passband cannot "
        "generally be split into physical parts. Many dossiers remain incomplete in that depth.\n",
        f"{unlinked_ground} ground source {'record' if unlinked_ground == 1 else 'records'} "
        "still lack both a data URL and a saved "
        "landing URL. Other documentation leads may point to a paper or instrument page "
        "without an opened numeric curve. See the "
        "[manager follow-up](sources/round2/manager/ground_source_followup.md) "
        "and [reconciliation](sources/round2/manager/reconciliation.md).\n",
        "The status `combined_only` means the factor is present in a combined curve but no "
        "separate usable curve was established. `not_found` records the documented search, "
        "not proof of absence. `unknown` includes work not yet done. "
        "`file_verified` on an SVO fallback means its numeric response was opened; it does not "
        "make the source independent. Redistribution terms have not been cleared for NJM hosting.\n",
        "## Source dossiers and independent checks\n",
        "- [Claude's space dossiers](sources/round2/claude/report.md) and "
        "[source records](sources/round2/claude/sources.jsonl): JWST, HST, Gaia, TESS, Kepler, "
        "WISE, Spitzer, Euclid, Herschel, Roman and others. [Peer review](sources/round2/claude/peer-review.md).",
        "- [Grok's ground dossiers](sources/round2/grok/report.md) and "
        "[source records](sources/round2/grok/sources.jsonl): major surveys, VLT, Gemini, "
        "Subaru, Keck and others. Its attempted later peer review hit a provider balance limit.",
        "- [Gemini's other-system dossiers](sources/round2/gemini/report.md) and "
        "[source records](sources/round2/gemini/sources.jsonl): planetary, Earth-observation, "
        "generic systems and cameras. Its report includes a VISTA/ZTF peer check.",
        "- [Codex audit](sources/round2/codex/report.md), "
        "[issues](sources/round2/codex/issues.jsonl) and "
        "[focused review](sources/round2/codex/focused-review.md). Each states its actual "
        "input hashes and verification boundary; structural checks do not certify all numeric claims.",
        "- [Mac manager checks](sources/round2/manager/report.md) and "
        "[exact retrieval log](sources/round2/manager/checks.json): independent WISE W1 and "
        "PACS blue numerical checks plus MIPS documentation. "
        "[Post-audit reconciliation](sources/round2/manager/reconciliation.md) identifies "
        "later corrections and remaining retrieval-link gaps. "
        "[Ground follow-up](sources/round2/manager/ground_source_followup.md) adds "
        "official landing pages without upgrading unverified numeric curves.\n",
        "Source records carry landing/data URLs, response contents, retrieval instructions, "
        "units, version, detector convention and search notes where established. Read the dossier "
        "and linked original definition before coding a parser; bare URLs are insufficient. "
        "The [matrix](review_matrix.csv) provides all five component-role statuses per entry. "
        "The [source-link table](source_links.csv) is a sortable list of original and fallback "
        "URLs, component roles, units, versions and retrieval instructions.\n",
        "## Scientific and sourcing decisions for later coding\n",
        "- Preserve the source release, detector, filter generation and observing mode. Rubin "
        "v1.9 cannot silently stand in for SVO's older LSST v1.5; JWST v5 and v7 curves differ.",
        "- Preserve each curve's physical quantity. A WISE per-photon response, a MIPS "
        "electrons-per-energy response, a GALEX effective area and a dimensionless filter "
        "transmission are not interchangeable. The [manager check](sources/round2/manager/report.md) "
        "documents a corrected WISE normalization claim.",
        "- Cassini's published system curve contains two filter wheels with QE and optics. "
        "Dividing by QE and optics yields a **filter pair**; dividing paired system curves can "
        "yield a ratio relative to a clear filter. Neither operation automatically isolates a "
        "single absolute filter curve. Keep calibration and camera applicability explicit.",
        "- VISTA's hardware tables use nanometres and percent-like response, while its supplied "
        "atmosphere files use micrometres and fractions. Keep site and observing conditions "
        "attached to each atmosphere product. ZTF's detector coatings are distinct variants.",
        "- Original-source discovery must enumerate bands and components without SVO. SVO "
        "aliases and fallback remain useful. The code paths and acceptance checks for this later "
        "work are in [implementation notes](implementation_notes.md).\n",
        "## Retrieval and implementation handoff\n",
        "1. Choose an instrument below; follow the dossier's original archive index or manifest "
        "to enumerate every band, mode and component. Record exact source version and checksum.",
        "2. Download original files, parse native units and definitions, and retain measured, "
        "combined and derived quantities separately. Retrieve official totals for comparisons "
        "only where the matching factors are supplied.",
        "3. Record gaps at the component level, including the search trail and access restrictions. "
        "Use SVO as an explicit fallback with its provenance. Do not treat SVO data as independent.",
        "4. Resolve the relevant [audit issues](sources/round2/codex/issues.jsonl), release "
        "mappings and redistribution rights before enabling an NJM-hosted downloader. Then "
        "test discovery with SVO disabled and fallback with NJM offline.\n",
        "Research has not changed the application, deployed the website or downloaded a bulk "
        "mirror. The report is a handoff for the later coding task, not an executable source registry.\n",
        "## Instrument index\n",
        "Each link opens the owner's dossier; the full machine-readable statuses are in the matrix.\n",
        "| Instrument | Example source | Filter | Optics | QE | Atmosphere | Published | Source basis |\n"
        "|---|---|---|---|---|---|---|---|",
    ])
    for row in sorted(reviewed, key=lambda item: (item["facility_key"], item["instrument_key"])):
        key = f"{row['facility_key']}/{row['instrument_key']}"
        name = f"[{key}](sources/round2/{row['agent']}/report.md)"
        linked = [source for source in source_rows if source["facility_key"] == row["facility_key"]
                  and source["instrument_key"] == row["instrument_key"]]
        linked.sort(key=lambda source: (source.get("source_basis") != "original",
                                        source.get("verification") != "file_verified"))
        urls = (url for source in linked
                for url in (source.get("data_url"), source.get("landing_url"))
                if isinstance(url, str) and url.startswith("https://"))
        url = next(urls, "")
        fallback = inventory.get((row["facility_key"], row["instrument_key"]), {}).get(
            "svo_browse_url", "")
        source_link = f"[source]({url})" if url else f"[SVO browse]({fallback})" if fallback else "see dossier"
        status = " | ".join(row[f"{role}_status"] for role in ROLES)
        report.append(f"| {name} | {source_link} | {status} | {row['independent_source_status']} |")
    (ROOT / "report.md").write_text("\n".join(report) + "\n")
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_keys": len(inventory), "reviewed_baseline_keys": baseline_count,
                "additional_keys": len(keys) - baseline_count, "by_owner": counts,
                "source_sha256": hashes, "validation_errors": errors}
    (ROOT / "research_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if errors:
        raise SystemExit("; ".join(errors))
    print(f"Validated {baseline_count}/{len(inventory)} baseline keys and built report")


if __name__ == "__main__":
    build()
