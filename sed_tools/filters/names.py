"""Instrument names for the pickers: NJM's catalogue as bundled with SED_Tools, and one entry per product.

The bundled ``instrument_catalogue.json`` names and describes NJM's instruments, and their SVO
equivalents, even while NJM is down.  Nothing here downloads; :mod:`.library` lists and fetches.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..component_sources.catalog import match_name, name_key

#: Whose spelling of a name the picker shows and downloads with, when several sources list it.
FETCH_ORDER = ("njm", "original", "local", "svo")


# ---- The bundled instrument catalogue
#: NJM's facilities and instruments, their descriptions and SVO equivalents, as
#: of the release; regenerate with scripts/make_instrument_catalogue.py.
INSTRUMENT_CATALOGUE = Path(__file__).parent.parent / "instrument_catalogue.json"


def load_instrument_catalogue(path=INSTRUMENT_CATALOGUE):
    """The bundled catalogue: ``{facility: {"description", "instruments": {name: {...}}}}``.

    Keyed by NJM's spelling; each instrument has ``description`` and
    ``svo_instruments`` (SVO ``Facility/Instrument`` names of the same
    product), empty where NJM gives none.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return instrument_catalogue(data["facilities"])


def instrument_catalogue(facilities):
    """Normalise ``filters.facilities_detailed`` of NJM's ``index.json`` (or a saved catalogue).

    Only ``description``, ``svo_instruments``, and the bundled catalogue's
    ``bands`` (NJM's served bands) and ``svo_bands`` (SVO filter ID -> one of
    those bands) are kept; a field in another form than the contract's is left
    empty rather than guessed at.
    """
    if not isinstance(facilities, dict):
        raise ValueError("expected a mapping of facilities")
    catalogue = {}
    for facility, record in sorted(facilities.items()):
        record = record if isinstance(record, dict) else {}
        instruments = record.get("instruments") if isinstance(record.get("instruments"), dict) else {}
        catalogue[facility] = {
            "description": _plain_text(record.get("description")),
            "instruments": {name: _instrument_metadata(meta) for name, meta in sorted(instruments.items())},
        }
    return catalogue


def _instrument_metadata(record):
    record = record if isinstance(record, dict) else {}
    claims = record.get("svo_instruments")
    claims = [c for c in claims if isinstance(c, str) and "/" in c] if isinstance(claims, list) else []
    bands = record.get("bands")
    bands = [b for b in bands if _band_name(b)] if isinstance(bands, list) else []
    return {"description": _plain_text(record.get("description")), "svo_instruments": claims,
            "bands": bands, "svo_bands": _svo_bands(record.get("svo_bands"), claims, bands)}


def _svo_bands(stored, claims, bands):
    """``{SVO filter ID: NJM band}`` from the stored ``{SVO instrument: ["G", "Gbp=BP", ...]}`` (``claims`` only)."""
    held = {}
    for claim, entries in (stored.items() if isinstance(stored, dict) else ()):
        if claim not in claims or not isinstance(entries, list):
            continue
        for entry in entries:
            svo_band, _, band = entry.partition("=") if isinstance(entry, str) else ("", "", "")
            band = band or svo_band
            if svo_band and band in bands:
                held[f"{claim}.{svo_band}"] = band
    return held


def _band_name(value):
    """Whether ``value`` can name a curve file in an instrument directory."""
    return isinstance(value, str) and value not in ("", ".", "..") and Path(value).name == value


def _plain_text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


# ---- Picker entries
def _merge(listings):
    """Group the names each source lists (``[(source, [(name, description), ...]), ...]``) into entries.

    One source's own names never join each other: NJM serves both ``Cameras/Sony_NEX5N`` and ``Cameras/Sony_NEX_5N``.
    """
    entries = []
    for source, listing in listings:
        named = {}
        for name, description in listing:
            named.setdefault(name, description)
        unmatched = []
        for name, description in named.items():
            entry = next((e for e in entries if source not in e["names"] and name in e["names"].values()), None)
            if entry is None:
                unmatched.append((name, description))
            else:
                _join(entry, source, name, description)
        for name, description in unmatched:
            entry = next((e for e in entries if source not in e["names"] and name_key(name) in e["keys"]), None)
            if entry is None:
                entry = {"names": {}, "descriptions": {}, "keys": set()}
                entries.append(entry)
            _join(entry, source, name, description)
    return entries


def _join(entry, source, name, description):
    entry["names"][source] = name
    entry["keys"].add(name_key(name))
    if description:
        entry["descriptions"][source] = description


def _drop_svo_duplicates(entries, claims):
    """Remove SVO's copy of a product NJM lists now as its own (``claims``: SVO name key -> NJM name).

    While NJM is unreachable, SVO's copies are what can be downloaded, so they stay.
    """
    listed = {e["names"]["njm"]: e for e in entries if "njm" in e["names"]}
    for entry in [e for e in entries if "svo" in e["names"]]:
        claimer = listed.get(claims.get(name_key(entry["names"]["svo"])))
        if claimer is None:
            continue
        del entry["names"]["svo"]
        text = entry["descriptions"].pop("svo", "")
        if text:
            claimer["descriptions"].setdefault("svo", text)
    entries[:] = [e for e in entries if e["names"]]


def _adopt_claimed_svo(entries, claims, njm_known):
    """Name SVO's copy of a product of NJM's after NJM when NJM does not list it now, so each product is listed once.

    Of two SVO instruments one NJM product claims (Euclid's NISP and NISP0) one entry remains, holding the first.
    """
    listed = {e["names"]["njm"] for e in entries if "njm" in e["names"]}
    for entry in [e for e in entries if "svo" in e["names"]]:
        owner = claims.get(name_key(entry["names"]["svo"]))
        if owner is None or owner in listed:
            continue  # not NJM's product, or NJM lists it now (_drop_svo_duplicates kept NJM's)
        target = next((e for e in entries if e is not entry and _stands_for(e, owner, njm_known)), None)
        if target is None:
            if set(entry["names"]) == {"svo"} or _stands_for(entry, owner, njm_known):
                entry["njm_name"] = owner
                continue
            target = {"names": {}, "descriptions": {}, "keys": set(), "njm_name": owner}
            entries.append(target)
        name = entry["names"].pop("svo")
        text = entry["descriptions"].pop("svo", "")
        if "svo" not in target["names"]:
            _join(target, "svo", name, text)
    entries[:] = [e for e in entries if e["names"]]


def _stands_for(entry, owner, njm_known):
    """Whether an entry is NJM's product ``owner``: named so already, or by a source other than SVO."""
    return entry.get("njm_name") == owner or any(
        match_name(name, njm_known) == owner for source, name in entry["names"].items() if source != "svo")


def _picker_entries(entries, njm_known, claims=None):
    """The entries as the pickers show them, sorted by label.

    Each has ``label`` and ``key`` (both the ``fetch_key``: the spelling in :data:`FETCH_ORDER`), ``sources``
    (those listing it now), ``description`` (NJM's, else SVO's, else ``""``) and ``names`` (each source's own
    spelling).  ``njm_known`` names entries NJM does not list now, so names do not change with NJM's
    availability; an entry only SVO lists takes NJM's name only if ``claims`` says it is NJM's product.
    """
    rows = []
    for entry in entries:
        names = dict(entry["names"])
        if "njm" not in names:
            names["njm"] = entry.get("njm_name") or _catalogue_name(names, njm_known, claims)
        names = {source: name for source, name in names.items() if name}
        fetch_key = next(names[source] for source in FETCH_ORDER if source in names)
        description = (njm_known.get(names.get("njm")) or {}).get("description") or entry["descriptions"].get("svo", "")
        rows.append({"key": fetch_key, "label": fetch_key, "sources": list(entry["names"]),
                     "description": description, "fetch_key": fetch_key, "names": names})
    return sorted(rows, key=lambda row: (row["label"].casefold(), row["label"]))


def _catalogue_name(names, njm_known, claims):
    """NJM's name for an entry it does not list now, from the bundled catalogue, or None."""
    for name in names.values():
        found = match_name(name, njm_known)
        if found is None:
            continue
        if claims is not None and set(names) == {"svo"} and claims.get(name_key(names["svo"])) != found:
            return None  # the same name, but NJM does not say it is the same product
        return found
    return None
