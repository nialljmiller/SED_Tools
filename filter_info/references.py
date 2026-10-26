#!/usr/bin/env python3
"""Literature references for the filter mirror: normalise, resolve, cache.

Every reference NJM records (SVO's ProfileReference and CalibrationReference,
curve Sources, provenance and curated files) is reduced to keys such as

    bibcode:2003AJ....126.1090C   doi:10.1086/376474   arxiv:1802.00879   url:https://...

``resolve`` looks each new key up once and caches it in filter_info/references.json,
so generate_index.py never touches the network.  It uses the ADS API when
ADS_DEV_KEY is set; otherwise doi.org, Crossref and the arXiv API, which need no
token.  A bibcode matched through Crossref is accepted only when its year, volume,
page and first-author initial all agree.  Keys resolving to the same DOI are
linked with ``same_as``; the one with a bibcode is canonical.

    python -m filter_info.references resolve [KEY ...]   # default: filter_info/unresolved.txt
"""

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
CACHE = HERE / "references.json"
CURATED = HERE / "curated" / "references.json"
UNRESOLVED = HERE / "unresolved.txt"
USER_AGENT = "SED_Tools-filter_info/1 (https://github.com/nialljmiller/SED_Tools)"

# ADS bibstems -> journal names Crossref can search for.
JOURNALS = {
    "AJ": "Astronomical Journal", "ApJ": "Astrophysical Journal",
    "ApJS": "Astrophysical Journal Supplement Series", "ApJL": "Astrophysical Journal Letters",
    "MNRAS": "Monthly Notices of the Royal Astronomical Society", "A&A": "Astronomy and Astrophysics",
    "A&AS": "Astronomy and Astrophysics Supplement Series",
    "PASP": "Publications of the Astronomical Society of the Pacific",
    "PASJ": "Publications of the Astronomical Society of Japan",
    "PASA": "Publications of the Astronomical Society of Australia",
    "AN": "Astronomische Nachrichten", "ARA&A": "Annual Review of Astronomy and Astrophysics",
    "SPIE": "Proceedings of SPIE", "JATIS": "Journal of Astronomical Telescopes Instruments and Systems",
    "ExA": "Experimental Astronomy", "SSRv": "Space Science Reviews", "Natur": "Nature",
    "Sci": "Science", "ApOpt": "Applied Optics", "RNAAS": "Research Notes of the AAS",
    "PSJ": "Planetary Science Journal", "Icar": "Icarus",
}

_ADS = re.compile(r"adsabs\.harvard\.edu/(?:#?abs/|full/|cgi-bin/nph-bib_query\?bibcode=)"
                  r"([12][0-9]{3}[A-Za-z0-9&.]{14}[A-Z.])")
_DOI = re.compile(r"(?:doi\.org/|\bdoi:\s*)(10\.\d{4,9}/[^\s\"<>|,;]+)", re.I)
_ARXIV = re.compile(r"(?:arxiv\.org/(?:abs|pdf|e-print)/|\barXiv:\s*)"
                    r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)
_URL = re.compile(r"https?://[^\s\"<>|]+")


def reference_keys(text):
    """Every reference key in a string of URLs and prose, in order of appearance."""
    text = unquote(text or "")
    found = []
    for pattern, kind in ((_ADS, "bibcode"), (_DOI, "doi"), (_ARXIV, "arxiv")):
        for match in pattern.finditer(text):
            value = match.group(1).rstrip(".)]")
            value = value.lower() if kind == "doi" else value
            found.append((match.start(), f"{kind}:{value}", match.span()))
    taken = [span for _, _, span in found]
    for match in _URL.finditer(text):
        if not any(a < match.end() and match.start() < b for a, b in taken):
            found.append((match.start(), "url:" + match.group(0).rstrip(".,;)]"), match.span()))
    keys = []
    for _, key, _ in sorted(found):
        if key not in keys:
            keys.append(key)
    return keys


def link(key):
    """The page a reader should open for a key."""
    kind, _, value = key.partition(":")
    return {"bibcode": f"https://ui.adsabs.harvard.edu/abs/{value}",
            "doi": f"https://doi.org/{value}",
            "arxiv": f"https://arxiv.org/abs/{value}"}.get(kind, value)


def short(entry):
    """``Cohen et al. 2003`` style label."""
    authors = entry.get("authors") or []
    if not authors:
        return entry.get("title") or entry.get("bibcode") or entry.get("url") or entry["key"]
    names = [a.split(",")[0] for a in authors]
    lead = names[0] if len(names) == 1 else (
        f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} et al.")
    return f"{lead} {entry.get('year', '')}".strip()


def load_cache(path=CACHE):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def save_cache(cache, path=CACHE):
    Path(path).write_text(json.dumps(dict(sorted(cache.items())), indent=1, sort_keys=True) + "\n")


def load_references(cache_path=CACHE, curated_path=CURATED):
    """The resolved cache with hand-checked entries (curated/references.json) on top."""
    cache = load_cache(cache_path)
    for key, fields in load_cache(curated_path).items():
        entry = {**cache.get(key, {"key": key, "url": link(key)}), **fields}
        entry.setdefault("resolved_by", "curated")
        entry["short"] = fields.get("short") or short(entry)
        cache[key] = entry
    return cache


def canonical(key, cache):
    """Follow ``same_as`` links to the entry a key stands for."""
    seen = set()
    while key in cache and cache[key].get("same_as") and key not in seen:
        seen.add(key)
        key = cache[key]["same_as"]
    return key


# --------------------------------------------------------------------------
# Network lookups (resolve only)
# --------------------------------------------------------------------------
class _Lookup:
    def __init__(self, delay=1.0):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.ads_token = os.environ.get("ADS_DEV_KEY") or os.environ.get("ADS_API_TOKEN")

    def get(self, url, **kwargs):
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, timeout=30, **kwargs)
        except Exception as exc:  # noqa: BLE001 - any network failure leaves the key unresolved
            print(f"  [net] {url}: {type(exc).__name__}", file=sys.stderr)
            return None
        return resp if resp.status_code == 200 else None

    # -- ADS (token) -------------------------------------------------------
    def ads(self, query):
        if not self.ads_token:
            return None
        resp = self.get("https://api.adsabs.harvard.edu/v1/search/query",
                        params={"q": query, "rows": 1,
                                "fl": "bibcode,title,author,year,pub,volume,page,doi,identifier"},
                        headers={"Authorization": f"Bearer {self.ads_token}"})
        docs = resp.json().get("response", {}).get("docs", []) if resp else []
        if not docs:
            return None
        doc = docs[0]
        entry = {"type": "article", "bibcode": doc["bibcode"], "title": (doc.get("title") or [""])[0],
                 "authors": doc.get("author", []), "year": doc.get("year"), "journal": doc.get("pub"),
                 "volume": doc.get("volume"), "page": (doc.get("page") or [None])[0],
                 "doi": (doc.get("doi") or [None])[0], "resolved_by": "ads"}
        time.sleep(self.delay)
        bib = self.session.post("https://api.adsabs.harvard.edu/v1/export/bibtex",
                                json={"bibcode": [doc["bibcode"]]}, timeout=30,
                                headers={"Authorization": f"Bearer {self.ads_token}"})
        if bib.status_code == 200:
            entry["bibtex"] = bib.json().get("export", "").strip()
        return entry

    # -- doi.org / Crossref / arXiv ------------------------------------------
    def doi(self, doi):
        resp = self.get(f"https://doi.org/{doi}",
                        headers={"Accept": "application/vnd.citationstyles.csl+json"})
        if resp is None:
            return None
        csl = resp.json()
        parts = (csl.get("issued") or csl.get("published") or {}).get("date-parts", [[None]])
        entry = {
            "type": {"journal-article": "article", "paper-conference": "inproceedings"}.get(
                csl.get("type"), "misc"),
            "title": html.unescape(re.sub(r"<[^>]+>", "", csl.get("title") or "")),
            "authors": [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                        for a in csl.get("author", []) if a.get("family") or a.get("given")],
            "year": parts[0][0] if parts and parts[0] else None,
            "journal": csl.get("container-title-short") or csl.get("container-title"),
            "volume": csl.get("volume"), "page": csl.get("page") or csl.get("article-number"),
            "doi": (csl.get("DOI") or doi).lower(), "resolved_by": "doi.org",
        }
        bib = self.get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"})
        if bib is not None:
            entry["bibtex"] = bib.text.strip()
        return entry

    def arxiv(self, arxiv_id):
        resp = self.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id})
        if resp is None or "<entry>" not in resp.text:
            return None
        body = resp.text.split("<entry>", 1)[1]
        def tag(name):
            m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", body, re.S)
            return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else None
        doi = tag("arxiv:doi")
        entry = self.doi(doi) if doi else None
        if entry is None:
            names = [re.sub(r"\s+", " ", n).strip() for n in re.findall(r"<name>(.*?)</name>", body, re.S)]
            entry = {"type": "misc", "title": tag("title"),
                     "authors": [f"{n.split()[-1]}, {' '.join(n.split()[:-1])}".strip(", ") for n in names],
                     "year": int((tag("published") or "0")[:4]) or None,
                     "journal": tag("arxiv:journal_ref") or "arXiv", "resolved_by": "arxiv"}
        entry["arxiv"] = arxiv_id
        return entry

    def gateway(self, bibcode, target):
        """Where ADS's public link resolver sends a bibcode (PUB_HTML or EPRINT_HTML)."""
        time.sleep(self.delay)
        try:
            resp = self.session.get(f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode.replace('&', '%26')}/{target}",
                                    timeout=30, allow_redirects=False)
        except Exception:  # noqa: BLE001
            return None
        location = resp.headers.get("Location", "")
        return location if resp.status_code in (301, 302) and location.startswith("http") else None

    def crossref_bibcode(self, bibcode):
        """DOI for a journal bibcode, accepted only when every field agrees."""
        year, stem = int(bibcode[:4]), bibcode[4:9].strip(".")
        volume, qualifier, page, initial = bibcode[9:13].strip("."), bibcode[13], bibcode[14:18].strip("."), bibcode[18]
        if not volume or not page:
            return None
        # Article numbers: A&A 537A.146 = A146; PASP 130f4505 = 064505 (issue letter f = 06).
        printed = (f"{qualifier}{page}" if qualifier.isupper() else
                   f"{ord(qualifier) - 96:02d}{page}" if qualifier.islower() else page)
        pages = {page.lstrip("0"), printed.lstrip("0").lower()}
        resp = self.get("https://api.crossref.org/works", params={
            "query.bibliographic": f"{JOURNALS.get(stem, stem)} {volume} {printed} {year}",
            "filter": f"from-pub-date:{year - 1},until-pub-date:{year + 1}", "rows": 10,
            "select": "DOI,volume,page,article-number,author,issued"})
        for item in (resp.json().get("message", {}).get("items", []) if resp else []):
            first = (item.get("page") or "").split("-")[0]
            got = {first.lstrip("0").lower(), (item.get("article-number") or "").lstrip("0").lower()}
            family = ((item.get("author") or [{}])[0].get("family") or " ")[0].upper()
            issued = (item.get("issued", {}).get("date-parts") or [[0]])[0][0] or 0
            if (str(item.get("volume", "")).lstrip("0") == volume.lstrip("0") and pages & got
                    and family == initial and abs(issued - year) <= 1):
                return item["DOI"].lower()
        return None

    def resolve(self, key):
        kind, _, value = key.partition(":")
        entry = None
        if kind == "url":
            entry = {"type": "web"}
        elif kind == "bibcode":
            entry, publisher = self.ads(f'bibcode:"{value}"'), None
            if entry is None:
                publisher = self.gateway(value, "PUB_HTML")
                doi = (_DOI.search(publisher or "") or [None, None])[1]
                if doi is None:
                    eprint = _ARXIV.search(self.gateway(value, "EPRINT_HTML") or "")
                    entry = self.arxiv(eprint.group(1)) if eprint else None
                    doi = None if entry else self.crossref_bibcode(value)
                if doi:
                    entry = self.doi(doi.lower())
            entry = entry or {"type": "article", "year": int(value[:4]), "journal": value[4:9].strip("."),
                              "publisher_url": publisher, "resolved_by": None}
            entry["bibcode"] = value
        elif kind == "doi":
            entry = self.ads(f'doi:"{value}"') or self.doi(value)
        elif kind == "arxiv":
            entry = self.ads(f'identifier:"arXiv:{value}"') or self.arxiv(value)
        entry = entry or {"type": "misc", "resolved_by": None}
        if entry.get("bibtex") and entry.get("bibcode"):
            entry["bibtex"] = re.sub(r"^(@\w+\{)[^,]*,", rf"\g<1>{entry['bibcode']},", entry["bibtex"], count=1)
        entry.update(key=key, url=link(key) if kind != "url" else value, resolved=date.today().isoformat())
        entry["short"] = short(entry)
        return {k: v for k, v in entry.items() if v not in (None, "", [])}


def link_duplicates(cache):
    """Point keys that share a DOI at the entry with a bibcode (else the first key)."""
    by_doi = {}
    for key, entry in cache.items():
        if entry.get("doi") and not entry.get("same_as"):
            by_doi.setdefault(entry["doi"], []).append(key)
    for keys in by_doi.values():
        if len(keys) > 1:
            keys.sort(key=lambda k: (not k.startswith("bibcode:"), k))
            for key in keys[1:]:
                cache[key]["same_as"] = keys[0]
                if cache[key].get("bibcode") and not cache[keys[0]].get("bibcode"):
                    cache[keys[0]]["bibcode"] = cache[key]["bibcode"]


def resolve(keys, cache_path=CACHE, delay=1.0, retry=False):
    """Resolve keys missing from the cache (or unresolved ones, with retry) and save it."""
    cache = load_cache(cache_path)
    todo = [k for k in dict.fromkeys(keys)
            if k not in cache or (retry and cache[k].get("resolved_by") is None and not k.startswith("url:"))]
    lookup = _Lookup(delay) if todo else None
    for n, key in enumerate(todo, 1):
        cache[key] = lookup.resolve(key)
        print(f"  [{n}/{len(todo)}] {key}: {cache[key].get('short')} ({cache[key].get('resolved_by')})")
        if n % 25 == 0:
            save_cache(cache, cache_path)
    link_duplicates(cache)
    save_cache(cache, cache_path)
    return len(todo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("resolve", help="look up keys missing from references.json")
    r.add_argument("keys", nargs="*", help=f"default: the keys listed in {UNRESOLVED.name}")
    r.add_argument("--retry", action="store_true", help="retry keys that did not resolve before")
    r.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)
    keys = args.keys or (UNRESOLVED.read_text().split() if UNRESOLVED.is_file() else [])
    if args.retry:
        keys += [k for k, e in load_cache().items() if e.get("resolved_by") is None]
    print(f"{resolve(keys, delay=args.delay, retry=args.retry)} keys looked up; cache: {CACHE}")


if __name__ == "__main__":
    main()
