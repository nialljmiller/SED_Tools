#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import urllib.parse

import requests
from bs4 import BeautifulSoup
from astroquery.svo_fps import SvoFps

from SED_tools.cli import _prompt_choice

SvoFps.TIMEOUT = 300  # give SVO a generous timeout

BASE_URL = "https://svo2.cab.inta-csic.es/theory/fps/"
INDEX_URL = urllib.parse.urljoin(BASE_URL, "index.php")

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "filters")


@dataclass(frozen=True)
class Facility:
    key: str
    label: str


@dataclass(frozen=True)
class Instrument:
    facility_key: str
    key: str
    label: str


@dataclass(frozen=True)
class FilterRow:
    filter_id: str
    band: str
    facility: str
    instrument: str
    description: str


class SVOFilterBrowser:
    """Navigate the SVO Filter Profile Service and download filters."""

    def __init__(self, base_dir: str = DEFAULT_BASE_DIR, session: Optional[requests.Session] = None):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "SED_Tools/filters (https://github.com)")

        self._facility_cache: Optional[List[Facility]] = None
        self._instrument_cache: Dict[str, List[Instrument]] = {}
        self._filters_cache: Dict[tuple[str, str], List[FilterRow]] = {}

    # -------------------- discovery helpers --------------------
    def list_facilities(self) -> List[Facility]:
        if self._facility_cache is not None:
            return self._facility_cache

        soup = self._fetch_index({"mode": "browse"})
        facilities: Dict[str, Facility] = {}
        for link in soup.find_all("a", href=True):
            params = self._parse_params(link["href"])
            if params.get("mode") != "browse":
                continue
            gname = params.get("gname")
            if not gname or "gname2" in params:
                continue
            label = link.get_text(" ", strip=True) or gname
            label = label.split("(")[0].strip() or gname
            facilities.setdefault(gname, Facility(key=gname, label=label))

        ordered = sorted(facilities.values(), key=lambda f: f.label.lower())
        self._facility_cache = ordered
        return ordered

    def list_instruments(self, facility_key: str) -> List[Instrument]:
        if facility_key in self._instrument_cache:
            return self._instrument_cache[facility_key]

        soup = self._fetch_index({"mode": "browse", "gname": facility_key, "asttype": ""})
        instruments: Dict[str, Instrument] = {}
        for link in soup.find_all("a", href=True):
            params = self._parse_params(link["href"])
            if params.get("mode") != "browse":
                continue
            if params.get("gname") != facility_key:
                continue
            gname2 = params.get("gname2")
            if not gname2:
                continue
            label = link.get_text(" ", strip=True) or gname2
            label = label.split("(")[0].strip() or gname2
            instruments.setdefault(gname2, Instrument(facility_key=facility_key, key=gname2, label=label))

        ordered = sorted(instruments.values(), key=lambda inst: inst.label.lower())
        self._instrument_cache[facility_key] = ordered
        return ordered

    def list_filters(self, facility_key: str, instrument_key: str) -> List[FilterRow]:
        cache_key = (facility_key, instrument_key)
        if cache_key in self._filters_cache:
            return self._filters_cache[cache_key]

        soup = self._fetch_index(
            {"mode": "browse", "gname": facility_key, "gname2": instrument_key, "asttype": ""}
        )
        table = self._find_filter_table(soup)
        rows: List[FilterRow] = []
        if table:
            for tr in table.find_all("tr"):
                cells = tr.find_all("td")
                if not cells:
                    continue
                link = cells[0].find("a", href=True)
                if not link:
                    continue
                filter_id = link.get_text(strip=True)
                if not filter_id:
                    continue
                # Guard against header rows duplicated as td
                if filter_id.lower() == "filter id":
                    continue
                band = _band_from_filter_id(filter_id)
                facility = cells[-3].get_text(strip=True) if len(cells) >= 3 else facility_key
                instrument = cells[-2].get_text(strip=True) if len(cells) >= 2 else instrument_key
                description = cells[-1].get_text(strip=True) if cells else ""
                rows.append(
                    FilterRow(
                        filter_id=filter_id,
                        band=band or filter_id,
                        facility=facility or facility_key,
                        instrument=instrument or instrument_key,
                        description=description,
                    )
                )

        # de-duplicate by filter_id while keeping order
        seen = set()
        unique_rows: List[FilterRow] = []
        for row in rows:
            if row.filter_id in seen:
                continue
            seen.add(row.filter_id)
            unique_rows.append(row)

        self._filters_cache[cache_key] = unique_rows
        return unique_rows

    # -------------------- downloading --------------------
    def download_filters(self, filters: Sequence[FilterRow]) -> None:
        if not filters:
            print("No filters to download.")
            return
        touched_dirs = {}  # out_dir -> instrument filename
        
        for idx, row in enumerate(filters, 1):
            print(f"  [{idx:3d}/{len(filters):3d}] {row.filter_id}")
            facility_dir = _clean_path(row.facility or "UnknownFacility")
            instrument_dir = _clean_path(row.instrument or "UnknownInstrument")
            band_name = _clean_filename(row.band) or _clean_filename(_band_from_filter_id(row.filter_id))
            if not band_name:
                band_name = _clean_filename(row.filter_id.split("/")[-1])
            out_dir = os.path.join(self.base_dir, facility_dir, instrument_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{band_name}.dat")
            touched_dirs[out_dir] = instrument_dir

            try:
                table = SvoFps.get_transmission_data(row.filter_id)
                if table is None or len(table) == 0:
                    print(f"    [skip] No transmission data returned for {row.filter_id}")
                    continue
                table.write(out_path, format="ascii.csv", overwrite=True)
                print(f"    [saved] {out_path}")
            except Exception as exc:  # pragma: no cover - network issues
                print(f"    [error] Failed to download {row.filter_id}: {exc}")

        # -------------------- final index files --------------------
        for out_dir, instrument_dir in touched_dirs.items():
            index_path = os.path.join(out_dir, instrument_dir)  # e.g., .../Generic/Johnson/Johnson
            # list all .dat files in the folder (names only), sorted; exclude the index file itself if it ends with .dat (it won't)
            dat_files = sorted(
                f for f in os.listdir(out_dir)
                if os.path.isfile(os.path.join(out_dir, f)) and f.endswith(".dat")
            )
            with open(index_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(dat_files) + ("\n" if dat_files else ""))




    # -------------------- HTTP helpers --------------------
    def _fetch_index(self, params: Dict[str, str]) -> BeautifulSoup:
        try:
            resp = self.session.get(INDEX_URL, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch SVO index: {exc}") from exc
        return BeautifulSoup(resp.text, "html.parser")

    def _parse_params(self, href: str) -> Dict[str, str]:
        url = urllib.parse.urljoin(BASE_URL, href)
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        return {k: v[0] for k, v in qs.items() if v}

    def _find_filter_table(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        for table in soup.find_all("table"):
            header = table.find("tr")
            if not header:
                continue
            titles = [cell.get_text(strip=True).lower() for cell in header.find_all(["th", "td"])]
            if titles and titles[0] == "filter id":
                return table
        return None


# -------------------- CLI helpers --------------------
def _band_from_filter_id(filter_id: str) -> str:
    if "." in filter_id:
        return filter_id.rsplit(".", 1)[-1]
    if "/" in filter_id:
        return filter_id.rsplit("/", 1)[-1]
    return filter_id


def _clean_path(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", ".", "_") else "_" for ch in value)
    cleaned = cleaned.strip()
    return cleaned or "Unknown"


def _clean_filename(value: str) -> str:
    return _clean_path(value).replace(" ", "_")




def run_interactive(base_dir: str = DEFAULT_BASE_DIR) -> None:
    browser = SVOFilterBrowser(base_dir=base_dir)

    facilities = browser.list_facilities()
    if not facilities:
        print("No facilities discovered.")
        return

    while True:
        
        fac_idx = _prompt_choice(facilities, "Facilities")

        if fac_idx is None:
            print("Exiting.")
            return
        facility = facilities[fac_idx]

        instruments = browser.list_instruments(facility.key)
        if not instruments:
            print(f"No instruments found for {facility.label}.")
            continue

        while True:
            inst_idx = _prompt_choice(instruments, f"Instruments for {facility.label}", allow_back=True)
            if inst_idx is None:
                print("Exiting.")
                return
            if inst_idx == -1:
                # user chose to go back to facility list
                break
            instrument = instruments[inst_idx]

            filters = browser.list_filters(facility.key, instrument.key)
            if not filters:
                print(f"No filters found for {instrument.label}.")
                continue

            print(
                textwrap.dedent(
                    f"""
                    \nSelected:
                      Facility : {facility.label} ({facility.key})
                      Instrument: {instrument.label} ({instrument.key})
                      Filters  : {len(filters)} available
                    """
                ).strip()
            )

            confirm = input("Download all filters for this instrument? [Y/n] ").strip().lower()
            if confirm and not confirm.startswith("y"):
                continue

            browser.download_filters(filters)

            again = input("Download another instrument from this facility? [y/N] ").strip().lower()
            if not again.startswith("y"):
                break


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print(
            textwrap.dedent(
                f"""
                Usage: {os.path.basename(sys.argv[0])} [base_dir]

                Interactively browse the SVO Filter Profile Service, choose a facility and
                instrument, then download all filters for that instrument into
                <base_dir>/<Facility>/<Instrument>/<Band>.dat.

                If base_dir is not provided, the default is {DEFAULT_BASE_DIR}.
                """
            ).strip()
        )
        return 0

    base_dir = argv[0] if argv else DEFAULT_BASE_DIR
    try:
        run_interactive(base_dir)
        return 0
    except RuntimeError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
