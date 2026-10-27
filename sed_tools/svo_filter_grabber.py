#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import textwrap
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup

from .filter_components import InstrumentManifest
from .filter_metadata import (ENERGY, SOURCE_ASSUMED, SOURCE_SVO, SVOMetadataClient, SVOUnreachableError,
                              svo_components_header, svo_fps_url, svo_get, write_filter_file)
from .ui_utils import prompt_choice
from .parsing import parse_multi_selection

SVO_READ_TIMEOUT = 300  # give a connected SVO a generous time to respond

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
        self.metadata_client = SVOMetadataClient(session=self.session, timeout=SVO_READ_TIMEOUT)

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

        sibling_keys = [i.key for i in self.list_instruments(facility_key) if i.key != instrument_key]
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
                if _would_collide(instrument, instrument_key, sibling_keys):
                    # SVO lists several instruments under one name (every Gaia
                    # release is "GAIA"); file them under their unique key
                    # (GAIA/GAIA0, GAIA/GAIA3) so one never overwrites another.
                    instrument = instrument_key
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
        """Save ``filters`` as ``<base_dir>/<Facility>/<Instrument>/<band>.dat`` with their SVO metadata.

        Every filter that can be saved is saved, and each instrument's index and
        manifest are written.  Then RuntimeError lists every filter that could not
        be saved, and why.  An unreachable SVO raises :class:`SVOUnreachableError`
        at once, since every remaining filter would fail the same way.
        """
        if not filters:
            print("No filters to download.")
            return
        expected: Dict[str, set] = {}  # instrument directory -> bands it should hold
        failed = []
        for idx, row in enumerate(filters, 1):
            print(f"  [{idx:3d}/{len(filters):3d}] {row.filter_id}")
            out_dir, band_name = self._target(row)
            expected.setdefault(out_dir, set()).add(band_name)
            try:
                self._save_filter(row, os.path.join(out_dir, f"{band_name}.dat"))
            except SVOUnreachableError:
                raise
            except (requests.RequestException, ValueError) as exc:  # malformed or unavailable source response
                print(f"    [error] {row.filter_id}: {exc}")
                failed.append(f"{row.filter_id}: {exc}")
        for out_dir, bands in expected.items():
            _write_index_and_manifest(Path(out_dir), bands)
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(filters)} SVO filters could not be saved; "
                               "run again to retry them:\n  " + "\n  ".join(failed))

    def _target(self, row: FilterRow) -> Tuple[str, str]:
        """The instrument directory (created) and band file name for ``row``."""
        facility_dir = _clean_path(row.facility or "UnknownFacility")
        instrument_dir = _clean_path(row.instrument or "UnknownInstrument")
        out_dir = os.path.join(self.base_dir, facility_dir, instrument_dir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, _clean_filename(row.band)

    def _save_filter(self, row: FilterRow, out_path: str) -> None:
        """Write one SVO curve with its metadata header."""
        record = self.metadata_client.get_filter(row.filter_id)
        if record is None or record.wavelength.size == 0:
            raise ValueError("SVO returned no transmission data")
        if record.detector_type:
            detector_type, source = record.detector_type, SOURCE_SVO
        else:
            detector_type, source = ENERGY, SOURCE_ASSUMED
            print(f"    [warn] SVO gives no DetectorType for {row.filter_id}; assuming energy")
        write_filter_file(
            out_path,
            record.wavelength,
            record.transmission,
            detector_type=detector_type,
            source=source,
            filter_id=row.filter_id,
            wavelength_unit=record.wavelength_unit,
            extra=svo_components_header(record.components),
        )
        print(f"    [saved] {out_path} ({detector_type}-counting)")

    # -------------------- HTTP helpers --------------------
    def _fetch_index(self, params: Dict[str, str]) -> BeautifulSoup:
        try:
            resp = svo_get(self.session, urllib.parse.urljoin(svo_fps_url(), "index.php"),
                           params=params, read_timeout=60)
            resp.raise_for_status()
        except SVOUnreachableError:
            raise
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch SVO index: {exc}") from exc
        return BeautifulSoup(resp.text, "html.parser")

    def _parse_params(self, href: str) -> Dict[str, str]:
        url = urllib.parse.urljoin(svo_fps_url(), href)
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


def _write_index_and_manifest(directory: Path, expected: Iterable[str]) -> None:
    """Write the instrument's MESA index (every ``.dat`` present) and merge its curves into the manifest.

    The manifest keeps every expected band even after a partial download, so a
    retry cannot mistake one saved curve for a complete instrument.
    """
    out_dir = str(directory)
    dat_files = sorted(
        f for f in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, f)) and f.endswith(".dat")
    )
    with open(os.path.join(out_dir, directory.name), "w", encoding="utf-8") as fh:  # e.g. .../Generic/Johnson/Johnson
        fh.write("\n".join(dat_files) + ("\n" if dat_files else ""))
    manifest = InstrumentManifest.load(directory)
    headers = InstrumentManifest.from_published(directory)
    manifest.bands = sorted(set(manifest.bands) | set(expected))
    manifest.published = headers.published
    manifest.published_bands.update(headers.published_bands)
    manifest.detector_types.update(headers.detector_types)
    manifest.detector_type_sources.update(headers.detector_type_sources)
    for role, entries in headers.components.items():
        for band, rel in entries.items():
            manifest.components.setdefault(role, {}).setdefault(band, rel)
    manifest.save(directory)


def _would_collide(instrument_name: str, instrument_key: str, other_keys: Sequence[str]) -> bool:
    """True if filing ``instrument_key`` under ``instrument_name`` could clash
    with another instrument of the same facility."""
    def norm(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    name = norm(instrument_name)
    if not name or name == norm(instrument_key):
        return False
    return any(norm(k).startswith(name) for k in other_keys)


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
        bulk_spec = input(
            "Bulk telescope mode: enter facility IDs (e.g. 1,3-5) to download ALL instruments, "
            "or press Enter for single-facility mode: "
        ).strip()
        if bulk_spec:
            try:
                facility_indexes = parse_multi_selection(bulk_spec, len(facilities))
            except ValueError as exc:
                print(f"Invalid selection: {exc}")
                continue
            if not facility_indexes:
                print("No facilities selected.")
                continue

            selected = [facilities[i] for i in facility_indexes]
            print(f"Selected {len(selected)} facilities for bulk download.")
            confirm = input("Download ALL instruments for selected facilities? [Y/n] ").strip().lower()
            if confirm and not confirm.startswith("y"):
                continue
            _download_facilities(browser, selected)
            again = input("Bulk download another set of facilities? [y/N] ").strip().lower()
            if again.startswith("y"):
                continue
            return

        fac_idx = prompt_choice(facilities, "Facilities")

        if fac_idx is None:
            print("Exiting.")
            return
        facility = facilities[fac_idx]

        instruments = browser.list_instruments(facility.key)
        if not instruments:
            print(f"No instruments found for {facility.label}.")
            continue
        if not _browse_instruments(browser, facility, instruments):
            return


def _download_facilities(browser: SVOFilterBrowser, facilities: Sequence[Facility]) -> None:
    """Download every instrument of ``facilities``, then raise RuntimeError listing any that failed."""
    failed = []
    for facility in facilities:
        instruments = browser.list_instruments(facility.key)
        if not instruments:
            print(f"No instruments found for {facility.label}.")
            continue
        print(f"\nFacility: {facility.label} ({len(instruments)} instruments)")
        for instrument in instruments:
            filters = browser.list_filters(facility.key, instrument.key)
            if not filters:
                print(f"  [skip] {instrument.label}: no filters found")
                continue
            print(f"  Downloading {len(filters)} filters for {instrument.label}")
            try:
                browser.download_filters(filters)
            except SVOUnreachableError:
                raise
            except RuntimeError as exc:  # some filters of this instrument could not be saved
                failed.append(f"{facility.label}/{instrument.label}: {exc}")
    if failed:
        raise RuntimeError("Bulk download incomplete:\n" + "\n".join(failed))


def _browse_instruments(browser: SVOFilterBrowser, facility: Facility, instruments: Sequence[Instrument]) -> bool:
    """Offer one facility's instruments for download; False when the user chooses to exit."""
    while True:
        inst_idx = prompt_choice(instruments, f"Instruments for {facility.label}", allow_back=True)
        if inst_idx is None:
            print("Exiting.")
            return False
        if inst_idx == -1:
            # user chose to go back to facility list
            return True
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
            return True


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
