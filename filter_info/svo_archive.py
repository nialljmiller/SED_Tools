#!/usr/bin/env python3
"""Archived SVO Filter Profile Service records: fetch, parse and build.

SVO's service (svo2.cab.inta-csic.es) has been down since October 2026, so its
descriptive metadata comes from Internet Archive captures of two page kinds:

- ``fps_votable``: ``fps.php?ID=<id>``, the VOTable with every PARAM and the curve;
- ``fps_browse``: ``index.php?id=<id>&mode=browse``, the web page.  It adds the
  reference labels, owner-specified versus calculated values, zero points per
  magnitude system, and the facility and instrument descriptions.

Raw captures are listed in ``manifest.jsonl`` files, one JSON object per line
(id, kind, capture_timestamp, original_url, wayback_url, sha256, status, and
file relative to the manifest): the format of the phase-3 harvest.  ``fetch``
writes the same format for small gap fills.  ``build`` parses every manifest
into ``filter_info/svo/<Facility>.json``, which generate_index.py reads offline.

    python -m filter_info.svo_archive fetch QUEUE.json OUT_DIR [--ids FILE]
    python -m filter_info.svo_archive build filter_info/svo MANIFEST.jsonl [...]

``build`` regenerates its output from exactly the manifests it is given, so
pass every harvest each time.
"""

import argparse
import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"
EMPTY = {"", "-", "--", "---", "----", "-----", "------", "-------", "--------"}

# Browse-page labels -> SVO VOTable PARAM names.
BROWSE_FIELDS = {
    "Filter ID": "filterID", "Description": "Description", "Phot.System": "PhotSystem",
    "Detector Type": "DetectorType", "Band Name": "Band", "Obs. Facility": "Facility",
    "Instrument": "Instrument", "Comments": "Comments",
}
BROWSE_PROPERTIES = {
    "λref": "WavelengthRef", "λmean": "WavelengthMean", "λcen": "WavelengthCen",
    "λeff": "WavelengthEff", "λpeak": "WavelengthPeak", "λpivot": "WavelengthPivot",
    "λphot": "WavelengthPhot", "λmin": "WavelengthMin", "λmax": "WavelengthMax",
    "Weff": "WidthEff", "FWHM": "FWHM", "Af/AV": "AfAv", "Fsun": "Fsun",
}
BROWSE_REFERENCES = {
    "Reference for filter response": "ProfileReference",
    "Reference for calibration": "CalibrationReference",
}


def _text(value):
    value = re.sub(r"\s+", " ", html.unescape(value or "")).strip()
    return "" if value in EMPTY else value


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decode(raw):
    """Text of a capture: UTF-8, else its declared charset (SVO's pages say ISO-8859-1)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        declared = re.search(rb'charset=["\']?([A-Za-z0-9_-]+)', raw[:4096])
        name = declared.group(1).decode().lower() if declared else "cp1252"
        # Browsers read ISO-8859-1 as Windows-1252, which also maps 0x80-0x9F.
        return raw.decode("cp1252" if name in ("iso-8859-1", "latin1", "latin-1") else name, errors="replace")


def curve_digest(samples):
    """SHA-256 of a curve's samples at 8 significant figures (format-independent)."""
    text = "\n".join(f"{float(w):.8g},{float(t):.8g}" for w, t in samples)
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# VOTable (fps.php?ID=...)
# --------------------------------------------------------------------------
def parse_votable(text):
    """Return {"params", "samples"} from an SVO filter VOTable, or None."""
    params = {}
    for match in re.finditer(r"<PARAM\b([^>]*)>", text):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        name = attrs.get("name")
        if name and name not in params:  # SVO repeats Description; the first wins
            params[name] = html.unescape(attrs.get("value", ""))
    if not params.get("filterID"):
        return None
    body = text[text.find("<TABLEDATA>"):]
    samples = [(float(w), float(t)) for w, t in re.findall(
        r"<TR>\s*<TD>\s*([^<\s]+)\s*</TD>\s*<TD>\s*([^<\s]+)\s*</TD>", body)]
    return {"params": params, "samples": samples}


# --------------------------------------------------------------------------
# Browse page (index.php?id=...&mode=browse)
# --------------------------------------------------------------------------
class _Page(HTMLParser):
    """Every table as rows of cells ({"text", "links"}), plus the Tip() tooltips."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.tips, self._open, self._link = [], [], [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for value in attrs.values():
            if value and "Tip('" in value:
                tip = re.search(r"Tip\('((?:[^'\\]|\\.)*)'", value)
                if tip:
                    self.tips.append(tip.group(1).replace("\\'", "'"))
        if tag == "table":
            self._open.append({"rows": [], "cell": None})
        elif tag == "tr" and self._open:
            self._open[-1]["rows"].append([])
        elif tag in ("td", "th") and self._open:
            table = self._open[-1]
            if not table["rows"]:
                table["rows"].append([])
            table["cell"] = {"text": "", "links": []}
            table["rows"][-1].append(table["cell"])
        elif tag in ("p", "br") and self._cell() is not None:
            self._cell()["text"] += "\n"
        elif tag == "a" and (attrs.get("href") or "").startswith("http"):
            cell = self._cell()
            self._link = {"url": attrs["href"], "label": "",
                          "after": cell["text"] if cell else ""}

    def handle_endtag(self, tag):
        if tag == "table" and self._open:
            self.tables.append(self._open.pop()["rows"])
        elif tag in ("td", "th") and self._open:
            self._open[-1]["cell"] = None
        elif tag == "a" and self._link is not None:
            cell = self._cell()
            if cell is not None:
                cell["links"].append(self._link)
            self._link = None

    def handle_data(self, data):
        cell = self._cell()
        if cell is not None:
            cell["text"] += data
        if self._link is not None:
            self._link["label"] += data

    def _cell(self):
        return self._open[-1]["cell"] if self._open else None


def _label(text):
    return re.sub(r"\s+|\(\?\)|:$", "", _text(text).replace("(?)", "")).rstrip(":")


def _tip(tip_html):
    name = re.search(r"<b>(.*?)</b>", tip_html, re.S)
    desc = re.search(r"class=helptd>(.*?)</td>", tip_html, re.S)
    links = [{"url": u, "label": _text(re.sub(r"<[^>]+>", "", l))}
             for u, l in re.findall(r'<a href="([^"]+)"[^>]*>(.*?)</a>', tip_html, re.S)]
    return (_text(re.sub(r"<[^>]+>", "", name.group(1))) if name else "",
            _text(re.sub(r"<[^>]+>", " ", desc.group(1))) if desc else "", links)


def parse_browse(text):
    """Return the fields of an SVO filter page, or None when it shows no filter."""
    page = _Page()
    page.feed(text)
    fields, specified, calculated, zero_points, references = {}, {}, {}, {}, {}
    labels = {_label(k): v for k, v in BROWSE_FIELDS.items()}
    for rows in page.tables:
        columns = {}
        system = {"specified": {}, "calculated": {}}
        for row in rows:
            cells = [_text(c["text"]) for c in row]
            if len(cells) >= 2 and "Specified" in cells and "Calculated" in cells:
                columns = {"specified": cells.index("Specified"),
                           "calculated": cells.index("Calculated")}
                continue
            if not cells:
                continue
            label = _label(row[0]["text"])
            if len(cells) == 2 and label in labels:
                fields[labels[label]] = cells[1]
            elif label in ("ZPType", "PhotCalID") and len(cells) >= 2:
                system["ZeroPointType" if label == "ZPType" else "PhotCalID"] = cells[1]
            elif columns and len(cells) > max(columns.values()):
                unit = cells[-1].strip("()")
                for kind, col in columns.items():
                    value = _number(cells[col])
                    if value is None:
                        continue
                    if label in BROWSE_PROPERTIES:
                        (specified if kind == "specified" else calculated)[BROWSE_PROPERTIES[label]] = value
                    elif label in ("ZeroPoint", ""):
                        system[kind][unit] = value
            for cell in row:
                for link in cell["links"]:
                    for prefix, name in BROWSE_REFERENCES.items():
                        if link["after"].strip().rstrip(":").endswith(prefix):
                            references[name] = {"url": link["url"], "label": _text(link["label"])}
                if "Transmission components:" in cell["text"]:
                    comp = cell["text"].split("Transmission components:", 1)[1].lstrip()
                    fields["components"] = _text(comp.split("\n")[0].split("id:")[0])
        if system.get("PhotCalID"):
            zero_points[system["PhotCalID"].rsplit("/", 1)[-1]] = system
    filter_id = fields.get("filterID", "")
    if not filter_id:
        return None
    flat = _text(re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<script.*?</script>", " ", text)))
    dates = re.search(r"Filter added:\s*([\d-]+ [\d:]+)\s*Last update:\s*([\d-]+ [\d:]+)", flat)
    if dates:
        fields["added"], fields["updated"] = dates.groups()
    ack = re.search(r"acknowledgement in your publication:\s*(.*?)\s*and we would appreciate", flat)
    service = {"acknowledgement": ack.group(1), "references": sorted({
        unquote(b) for b in re.findall(r"adsabs\.harvard\.edu/abs/([^/\s]+)", flat[ack.end():])})} if ack else {}
    facility, group = filter_id.split("/", 1)[0], filter_id.rsplit(".", 1)[0]
    info = {"service": service} if service else {}
    for tip in page.tips:
        name, description, links = _tip(tip)
        for key, wanted in (("facility", facility), ("instrument", group)):
            # A page repeats these tooltips; keep the fullest copy.
            if name == wanted and (len(links), len(description)) > (
                    len(info.get(key, {}).get("links", [])), len(info.get(key, {}).get("description", ""))):
                info[key] = {"name": name, "description": description, "links": links}
    return {"fields": fields, "references": references, "specified": specified,
            "calculated": calculated, "zero_points": zero_points, **info}


# --------------------------------------------------------------------------
# fetch: small gap fills from the Internet Archive
# --------------------------------------------------------------------------
def _candidates(item):
    """(timestamp, original URL) captures of a queue item, newest first."""
    if "captures" in item:
        caps = [(c["ts"], c["original"]) for c in item["captures"]]
    else:
        caps = [(item["ts"], item["original"])]
        caps += [(ts, item["original"]) for ts in item.get("all_ts", []) if ts != item["ts"]]
    return sorted(set(caps), reverse=True)


def _valid(kind, filter_id, raw):
    text = decode(raw)
    record = parse_votable(text) if kind == "fps_votable" else parse_browse(text)
    if record is None:
        return "no filter record"
    found = record["params"]["filterID"] if kind == "fps_votable" else record["fields"]["filterID"]
    return "" if found == filter_id else f"page is for {found}"


def fetch(queue_path, out_dir, ids=None, delay=3.0, attempts=3):
    """Download the newest valid capture of each queued ID into out_dir (resumable)."""
    import requests

    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok":
                done.add((row["id"], row["kind"]))
    items = json.loads(Path(queue_path).read_text())
    session = requests.Session()
    with manifest.open("a") as log:
        for item in items:
            key = (item["id"], item["kind"])
            if (ids is not None and item["id"] not in ids) or key in done:
                continue
            for ts, original in _candidates(item):
                url = WAYBACK.format(ts=ts, url=original)
                raw, error = None, ""
                for attempt in range(attempts):
                    try:
                        resp = session.get(url, timeout=60)
                        if resp.status_code == 200:
                            raw = resp.content
                            break
                        error = f"HTTP {resp.status_code}"
                    except requests.RequestException as exc:
                        error = type(exc).__name__
                    # 429: the Archive is throttling this IP (shared with other harvests); wait longer.
                    time.sleep(60 * (attempt + 1) if error == "HTTP 429" else delay * (attempt + 2))
                if error == "HTTP 429" and raw is None:
                    sys.exit("The Internet Archive is still throttling (HTTP 429); rerun later to resume.")
                time.sleep(delay)
                row = {"id": item["id"], "kind": item["kind"], "original_url": original,
                       "wayback_url": url, "capture_timestamp": ts,
                       "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                if raw is not None:
                    error = _valid(item["kind"], item["id"], raw)
                if raw is None or error:
                    log.write(json.dumps({**row, "status": "rejected", "reason": error}) + "\n")
                    continue
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", item["id"])
                ext = "xml" if item["kind"] == "fps_votable" else "html"
                path = out_dir / "raw" / f"{safe}__{item['kind']}__{ts}.{ext}"
                path.write_bytes(raw)
                log.write(json.dumps({**row, "status": "ok", "size": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest(),
                                      "file": str(path.relative_to(out_dir))}) + "\n")
                log.flush()
                break


# --------------------------------------------------------------------------
# build: every manifest -> filter_info/svo/<Facility>.json
# --------------------------------------------------------------------------
def build(out_dir, manifests):
    """Parse each manifest's captures; keep the distinct versions of every record."""
    records, service, counts = {}, {}, Counter()
    for manifest in map(Path, manifests):
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "ok" or row.get("kind") not in ("fps_votable", "fps_browse"):
                continue
            url = urlsplit(row["original_url"])
            family = url.hostname + url.path.rsplit("/", 1)[0]
            raw = (manifest.parent / row["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                counts[(row["kind"], family, "SHA-256 differs from manifest")] += 1
                continue
            text = decode(raw)
            capture = {"timestamp": row["capture_timestamp"], "url": row["original_url"],
                       "sha256": row["sha256"]}
            if row["kind"] == "fps_votable":
                parsed = parse_votable(text)
                if parsed is None:
                    counts[(row["kind"], family, "unparseable")] += 1
                    continue
                version = {"params": parsed["params"], "n_samples": len(parsed["samples"]),
                           "curve_sha256": curve_digest(parsed["samples"])}
                kind = "votable"
            else:
                version = parse_browse(text)
                if version is None:
                    counts[(row["kind"], family, "unparseable")] += 1
                    continue
                # SVO's acknowledgement request is service-wide: keep the newest.
                credit = version.pop("service", None)
                if credit and capture["timestamp"] >= service.get("capture", {}).get("timestamp", ""):
                    service = {**credit, "capture": capture}
                kind = "browse"
            counts[(row["kind"], family, "parsed")] += 1
            versions = records.setdefault(row["id"], {}).setdefault(kind, [])
            same = next((v for v in versions if v["record"] == version), None)
            if same is None:
                versions.append({"record": version, "captures": [capture]})
            elif capture not in same["captures"]:
                same["captures"].append(capture)
    by_facility = {}
    for filter_id, kinds in records.items():
        for versions in kinds.values():
            for v in versions:
                v["captures"].sort(key=lambda c: c["timestamp"])
            versions.sort(key=lambda v: v["captures"][-1]["timestamp"])
        by_facility.setdefault(filter_id.split("/", 1)[0], {})[filter_id] = kinds
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Regenerated from exactly these manifests, so pass every harvest each time.
    for facility, entries in by_facility.items():
        path = out_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', facility)}.json"
        path.write_text(json.dumps(dict(sorted(entries.items())), indent=1, sort_keys=True) + "\n")
    if service:
        (out_dir / "_service.json").write_text(json.dumps(service, indent=1, sort_keys=True) + "\n")
    for (kind, family, outcome), n in sorted(counts.items()):
        print(f"  {kind:12s} {family:45s} {outcome:30s} {n}")
    return sum(len(e) for e in by_facility.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the newest valid capture of queued IDs")
    f.add_argument("queue")
    f.add_argument("out_dir")
    f.add_argument("--ids", help="file with one SVO filter ID per line")
    f.add_argument("--delay", type=float, default=3.0)
    b = sub.add_parser("build", help="parse manifests into filter_info/svo/<Facility>.json")
    b.add_argument("out_dir")
    b.add_argument("manifests", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        ids = set(Path(args.ids).read_text().split()) if args.ids else None
        fetch(args.queue, args.out_dir, ids, args.delay)
    else:
        print(f"{build(args.out_dir, args.manifests)} SVO filter IDs written to {args.out_dir}")


if __name__ == "__main__":
    main()
