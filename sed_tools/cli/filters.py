"""The filter-set flows of ``sed-tools``: the browser, the availability and catalogue reports,
interactive composition and combining sets. The cli package re-exports them."""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from requests import RequestException

from ..api import Filters, _listing_failures
from ..cube.model import FILTER_DIR_DEFAULT
from ..filters.compose import ComponentError
from ..filters.njm_catalogue import CatalogueError
from ..terminal_plots import terminal_width
from ..ui_utils import NO_DEFAULT as _NO_DEFAULT, ask_number as _ask_number, ask_option, ask_yes, cut, prompt_choice, wrap


def _print_field(name: str, text) -> None:
    """``  name : text``, wrapped to the terminal width under the text column."""
    print(wrap(text, indent=f"  {name:<25}: ", hanging=" " * 29))


def print_filter_availability(info: dict) -> None:
    """Print what a filter set contains and what can be composed from it."""
    print(f"\n{info['facility']}/{info['instrument']}"
          + (f"  ({len(info['bands'])} bands)" if info["bands"] else ""))
    if _missing_bands(info):
        _print_field("Incomplete", _incomplete_summary(info) + "; fetch it again when NJM is reachable to complete it")
    src = f" [{info['published_source']}]" if info["published_source"] else ""
    _print_field("Published curves contain", f"{info['published'] or 'unknown'}{src}")
    if info["published"] == "mixed":
        for band, recipe in info["published_bands"].items():
            print(f"    {band}: {recipe}")
    _print_field("Separate components", _component_summary(info))
    if info["buildable"]:
        _print_field("Can be built", ", ".join(info["buildable"]))
    if info.get("variants"):
        _print_field("Variants (API only)", _variants_summary(info["variants"]))
    if info["component_status"]:
        _print_field("Note", info["component_status"])
    for note in info.get("notes") or []:
        _print_field("Note", note)
    for role, bands in info["partial_components"].items():
        print(f"    {role} only for: {', '.join(bands)}")
    review = info["component_availability"]
    _print_field("Source review", review["status"] + (f" ({review['builder']})" if review["builder"] else ""))
    if info["missing_files"]:
        _print_field("Missing files", ", ".join(info["missing_files"]))
    for error in info["errors"]:  # errors stay on one line, as elsewhere, so they can be searched
        print(f"  {'Source unavailable':<25}: {error}")
    _print_field("Atmosphere", _atmosphere_summary(info))
    if info["detector_type"]:
        _print_field("Detector", f"{info['detector_type']}-counting")
    elif info["bands"] and info.get("availability_source") in ("original", "unknown"):
        # bands from the catalogue bundled with SED_Tools: no curve has been seen yet
        _print_field("Detector", "not known until the curves are installed")
        return
    assumed = [band for band, source in info["detector_type_sources"].items()
               if source in ("assumed", "unknown")]
    if assumed:
        _print_field("Detector unverified for", ", ".join(assumed))


def _missing_bands(info: dict) -> List[str]:
    """The bands an installed set declares but does not hold, e.g. those SVO lacks when NJM was down."""
    return [band for band in info.get("expected_bands") or [] if band not in info["bands"]]


def _incomplete_summary(info: dict, room: Optional[int] = None) -> str:
    """``4 of 6 bands installed; missing Gbp_bright, Gbp_faint``, the names cut to fit ``room`` if given."""
    missing = _missing_bands(info)
    text = f"{len(info['bands'])} of {len(info['expected_bands'])} bands installed; missing "
    return text + (", ".join(missing) if room is None else _names_to_fit(missing, room - len(text)))


def _component_summary(info: dict) -> str:
    """The instrument's own separate components, e.g. ``filter, optics, qe``."""
    if info["components"]:
        return ", ".join(info["components"])
    return "available for selected bands" if info["partial_components"] else "none identified"


def _atmosphere_sites(info: dict) -> List[str]:
    """The instrument's own atmosphere sites, default first; empty when no atmosphere is offered."""
    if not info.get("atmosphere"):
        return []
    sites = list(info.get("atmosphere_sites") or [])
    default = info.get("atmosphere_site")
    if default and default not in sites:
        sites.insert(0, default)
    return sites


def _atmosphere_summary(info: dict) -> str:
    sites = _atmosphere_sites(info)
    if sites:
        default = _default_site(info, sites)
        text = f"can be added at its own site{'s' * (len(sites) > 1)}: {_site_names(sites, default)}"
        if default is None:
            text += "; no default, so choose one with --atmosphere-site"
        return text + " (in Python, Filters.atmospheres() lists the conditions)"
    if info.get("platform") in ("space", "airborne"):
        return f"not applicable ({info['platform']}-based)"
    return "not offered (no site of its own assigned)"


def _default_site(info: dict, sites: List[str]) -> Optional[str]:
    """The instrument's default atmosphere site: the one it names, else its only site."""
    return info.get("atmosphere_site") or (sites[0] if len(sites) == 1 else None)


def _site_names(sites: List[str], default: Optional[str]) -> str:
    return ", ".join(f"{s} (default)" if s == default else s for s in sites)


def _default_variant(named: dict) -> Optional[str]:
    return next((name for name, entry in named.items() if entry.get("default")), None)


def _variants_summary(variants: dict) -> str:
    """``role: a (default), b; role2: c`` for the named alternative component curves."""
    parts = []
    for role, named in variants.items():
        default = _default_variant(named)
        names = ", ".join([f"{default} (default)"] * bool(default) + [n for n in named if n != default])
        parts.append(f"{role}: {names}" + ("" if default else f" (default: the standard {role} curve)"))
    return "; ".join(parts)


def _catalogue_entry(facility: str, instrument: str, base: str, *, offline: bool) -> dict:
    """NJM's info.json and properties.json for ``filters-info``, and where info.json came from
    (FilterCatalogue.origin); errors are reported, not raised."""
    entry = {"info": None, "properties": None, "errors": [], "origin": None}
    for kind in ("info", "properties"):
        try:
            entry[kind] = getattr(Filters, kind)(facility, instrument, filter_root=base, offline=offline)
        except (CatalogueError, OSError, ValueError) as exc:
            if str(exc) not in entry["errors"]:
                entry["errors"].append(str(exc))
    entry["origin"] = Filters._catalogue(base, offline=offline).origin(facility, instrument, "info")
    return entry


def _print_filter_set(target: str, info: dict, catalogue: dict) -> None:
    """The filters-info report: availability (Filters.available) and NJM's catalogue entry (_catalogue_entry)."""
    print_filter_availability(info)
    print_catalogue_entry(target, catalogue)


def print_catalogue_entry(target: str, entry: dict) -> None:
    """Print the NJM description, licence summary and per-band properties of an instrument."""
    info, props = entry["info"] or {}, (entry["properties"] or {}).get("bands", {})

    if info.get("description"):
        _print_field("Description", info["description"])
    origin = entry.get("origin") or {}
    if origin.get("njm_unreachable"):
        checked = origin.get("checked")
        when = time.strftime(", checked %Y-%m-%d %H:%M UTC", time.gmtime(checked)) if checked else ""
        _print_field("Catalogue", f"from the cached copy{when}; NJM could not be asked")
    licence = info.get("licence") or {}
    if licence:
        _print_field("Licence", _licence_summary(licence))
    if info.get("references"):
        _print_field("Cite", _cite_hint(target, info["references"]))
    bands = sorted(set(props) | set(info.get("bands", {})),
                   key=lambda b: (props.get(b, {}).get("WavelengthPivot") or float("inf"), b))
    if bands:
        _print_band_table(bands, props, info.get("bands", {}))
    for error in entry["errors"]:
        print(f"  {'NJM catalogue':<25}: {error}")
    if entry["errors"] and not info and not props:
        print(wrap("so no description, licence or band properties are shown; "
                    "the availability above does not depend on them", indent=" " * 29, hanging=" " * 29))


def _cite_hint(target: str, references: list) -> str:
    """How many references NJM records for an instrument, and the API call that lists them with the licences."""
    return f"{len(references)} references; in Python: Filters.cite('{target}')"


def _first_sentence(text: str, limit: int = 200) -> str:
    """The first sentence of ``text``, cut at a word with '…' if longer than ``limit`` characters."""
    text = " ".join(str(text).split())
    match = re.match(r'(.+?[.!?]["\')\]]?)(?=\s+["\'(\[]?[A-Z0-9]|\s*$)', text)
    sentence = match.group(1) if match else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + " …"
    return sentence


def _licence_summary(licence: dict) -> str:
    """Redistribution status, the terms' first sentence and the licence URL."""
    text = str(licence.get("redistribution", "unknown")) + "."
    if licence.get("terms"):
        text += " " + _first_sentence(licence["terms"])
    if licence.get("url"):
        text += " " + licence["url"]
    return text


def _print_band_table(bands: List[str], props: dict, band_info: dict) -> None:
    """Band, pivot, FWHM, detector and magnitude system, with descriptions wrapped to the terminal."""
    head = f"  {'Band':<16}{'Pivot (Å)':>11}{'FWHM (Å)':>11}  {'Detector':<9}{'MagSys':<8}"
    descriptions = [" ".join(str(band_info.get(b, {}).get("Description") or "").split()) for b in bands]
    room = terminal_width() - len(head)
    beside = all(len(d) <= room for d in descriptions)  # else each goes on its own lines below its row
    print((head + ("Description" if beside else "")).rstrip())
    for band, description in zip(bands, descriptions):
        p, i = props.get(band, {}), band_info.get(band, {})
        num = lambda v: f"{v:>11.1f}" if isinstance(v, (int, float)) else f"{'-':>11}"
        row = (f"  {band:<16}{num(p.get('WavelengthPivot'))}{num(p.get('FWHM'))}  "
               f"{p.get('DetectorType', '-'):<9}{i.get('MagSys') or '-':<8}")
        if beside:
            print((row + description).rstrip())
        else:
            print(row.rstrip())
            if description:
                print(wrap(description, indent=" " * 6, hanging=" " * 6))


_UNSTATED = "unstated"


def _offer_composition(instrument_dir: Path) -> None:
    """After a download, offer to build a composed product; the build's errors propagate."""
    info = _installed_availability(instrument_dir)
    print_filter_availability(info)
    if not _has_parts(info):
        return
    question = ("Build a composed version (e.g. filter only"
                + (", or with atmosphere" if _atmosphere_sites(info) else "") + ")? [y/N] ")
    if ask_yes(question):
        _compose_interactively(instrument_dir, info)


def _installed_availability(instrument_dir: Path) -> dict:
    """Filters.available for an installed instrument directory, without asking any source."""
    return Filters.available(instrument_dir.parent.name, instrument_dir.name,
                             filter_root=instrument_dir.parent.parent, include_remote=False)


def _has_parts(info: dict) -> bool:
    """Whether a curve can be built from the instrument's own parts: components, an atmosphere site,
    or an original-source builder that provides them."""
    return bool(info["buildable"] or _atmosphere_sites(info)
                or info["component_availability"]["status"] == "supported")


def _compose_interactively(instrument_dir: Path, info: dict) -> None:
    """Ask for the recipe, detector convention, large downloads and atmosphere, then build the product."""
    filter_root = instrument_dir.parent.parent
    review = info["component_availability"]
    roles = _ask_recipe(info)
    request = {
        "components": roles,
        "detector_type": _ask_detector_type(),
        "allow_heavy": bool(review.get("large_download")
                            and ask_yes("Allow a large original-source download if needed? [y/N] ")),
        "atmosphere": _ask_atmosphere(info, filter_root),
    }
    try:
        Filters.compose(instrument_dir, filter_root=filter_root, **request)
    except ComponentError as exc:
        _compose_the_bands_that_work(instrument_dir, request, exc)


def _ask_recipe(info: dict) -> List[str]:
    """Ask for a recipe (a listed one, or one typed as roles) until it names known roles."""
    from ..filters.compose import ATMOSPHERE, INSTRUMENT, parse_components

    options = [recipe.replace("+", ",") for recipe in info["buildable"]] + ["published"]
    print("Recipes (or type your own, e.g. published,optics):")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        pick = input("Choose a recipe [1]: ").strip() or "1"
        if pick.isdigit():
            if 1 <= int(pick) <= len(options):
                return list(parse_components(options[int(pick) - 1]))
            print(f"  Choose 1 to {len(options)}, or type a recipe.")
            continue
        try:
            roles = parse_components(pick)
        except ComponentError as exc:
            print(f"  {exc}; use filter, optics, qe or published.")
            continue
        if ATMOSPHERE in roles or INSTRUMENT in roles:
            print("  Use filter, optics, qe or published; the atmosphere is asked about separately.")
        elif roles:
            return list(roles)


def _ask_detector_type() -> Optional[str]:
    pick = ask_option("Detector convention [inherit], energy or photon: ",
                       ["inherit", "energy", "photon"], default="inherit")
    return None if pick == "inherit" else pick


def _ask_atmosphere(info: dict, filter_root: Path) -> Optional[Dict]:
    """The atmosphere to fold in, chosen by number from the instrument's own sites; None for none.

    Asked only when the instrument offers an atmosphere (ground-based, with a
    site of its own).  Its sites are listed with their source, the generic
    (model-derived) ones last; Enter keeps the curve without atmosphere.
    """
    sites = _atmosphere_sites(info)
    if not sites:
        return None
    grids = _site_grids(sites, filter_root)
    sites.sort(key=lambda site: (grids.get(site) or {}).get("kind") == "derived")  # stable: default first
    _print_site_choices(sites, _default_site(info, sites), grids)
    numbers = [str(n) for n in range(len(sites) + 1)]
    pick = ask_option("Atmosphere [0, none]: ", numbers + sites, default="0")
    if pick == "0":
        return None
    site = sites[int(pick) - 1] if pick in numbers else pick
    airmass, pwv = _ask_conditions(site, grids.get(site))
    return {"site": site, "airmass": airmass, "pwv": pwv}


def _site_grids(sites: List[str], filter_root: Path) -> Dict[str, dict]:
    """The atmosphere grids by site, cached ones refreshed from NJM's manifests (checked once a day)."""
    try:
        grids = Filters.atmospheres(filter_root=filter_root)
    except (RequestException, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not list the atmosphere grids for {', '.join(sites)}: {exc}") from exc
    return grids


def _print_site_choices(sites: List[str], default: Optional[str], grids: Dict[str, dict]) -> None:
    """``0. none``, then each site numbered, with its kind where its manifest states one and its source."""
    def name(site):
        notes = [note for note in ("default site" if site == default else "", (grids.get(site) or {}).get("kind"))
                 if note]
        return site + (f" ({', '.join(notes)})" if notes else "")

    names = [name(site) for site in sites]
    width = max(len("none"), *map(len, names))
    print("Atmosphere to fold in:")
    print(f"  0. {'none':<{width}}  the curve without atmosphere")
    for number, (site, label) in enumerate(zip(sites, names), 1):
        source = (grids.get(site) or {}).get("source") or "not downloaded yet"
        print(cut(f"  {number}. {label:<{width}}  {' '.join(source.split())}", terminal_width()))


def _ask_conditions(site: str, grid: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Airmass (None for 'unstated') and PWV, defaulting to the first conditions the site's grid has."""
    if grid:
        print(wrap(_conditions_text(grid), indent=f"  {site}: ", hanging=" " * (len(site) + 4)))
    else:
        print(f"  {site}: no grid listed yet; the conditions you give are generated if the site allows it")
    if grid and grid["airmass"]:
        default = grid["airmass"][0]
    elif grid and grid.get("unstated_airmass"):
        default = _UNSTATED
    else:
        default = _NO_DEFAULT
    words = (_UNSTATED,) if not grid or grid.get("unstated_airmass") else ()
    shown = "" if default is _NO_DEFAULT else f" [{default:g}]" if isinstance(default, float) else f" [{default}]"
    airmass = _ask_number(f"Airmass{shown}: ", default=default, words=words)
    pwv = None
    if not grid:
        pwv = _ask_number("PWV mm (blank if not applicable): ", default=None)
    elif grid["pwv"]:
        pwv = _ask_number(f"PWV mm [{grid['pwv'][0]:g}]: ", default=grid["pwv"][0])
    return (None if airmass == _UNSTATED else airmass), pwv


def _conditions_text(grid: dict) -> str:
    """The airmasses and PWVs an atmosphere grid has, e.g. ``airmass 1, 1.5; pwv 2.5 mm``."""
    parts = []
    if grid.get("airmass"):
        parts.append("airmass " + ", ".join(f"{a:g}" for a in grid["airmass"]))
    if grid.get("unstated_airmass"):
        parts.append("airmass 'unstated' (as published)")
    if grid.get("pwv"):
        parts.append("pwv " + ", ".join(f"{p:g}" for p in grid["pwv"]) + " mm")
    return "; ".join(parts) or "no conditions listed"


def _compose_the_bands_that_work(instrument_dir: Path, request: Dict, error: ComponentError) -> None:
    """When composing every band failed, offer the bands that can be composed; re-raise ``error`` otherwise."""
    working, failed = _composable_bands(instrument_dir, request)
    if not working or not failed:
        raise error
    print(f"Not every band can be composed as asked ({len(failed)} of {len(working) + len(failed)}):")
    for band, why in failed.items():
        print(wrap(why, indent=f"  {band}: ", hanging="    "))
    if not ask_yes(f"Build only {', '.join(working)}? [y/N] "):
        raise error
    Filters.compose(instrument_dir, filter_root=instrument_dir.parent.parent, bands=working, **request)


def _composable_bands(instrument_dir: Path, request: Dict) -> Tuple[List[str], Dict[str, str]]:
    """Each primary band composed in memory (nothing is written): the bands that work, and why the others fail."""
    from ..filters.compose import InstrumentManifest, compose_band
    from ..filters.library import FilterLibrary

    manifest = InstrumentManifest.load(instrument_dir)
    atmosphere_root = FilterLibrary(instrument_dir.parent.parent).atmosphere_root
    working, failed = [], {}
    for band in manifest.primary_bands():
        try:
            compose_band(instrument_dir, band, request["components"], atmosphere=request["atmosphere"],
                         atmosphere_root=atmosphere_root, manifest=manifest)
        except (ComponentError, OSError) as exc:
            failed[band] = str(exc)
        else:
            working.append(band)
    return working, failed


def run_filters_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Browse available instruments; use the API for every download and product.

    A download or build that fails is reported at once, and every one that
    failed is listed again when browsing ends.  Returns the exit status:
    1 if anything failed, else 0.
    """
    from ..filters.library import FilterLibrary

    library = FilterLibrary(base_dir)
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    failures: List[str] = []
    download = lambda facility, instrument: _offer_download(library, facility, instrument, failures)
    while facilities:
        selected = prompt_choice(_picker_labels(facilities), "Filter Facilities", multi=True)
        if selected is None:
            break
        if len(selected) > 1:
            _download_every_instrument(library, [facilities[i] for i in selected], failures, shown)
            break
        if not _browse_instruments(library, facilities[selected[0]], download, shown=shown):
            break
    if not failures:
        return 0
    print(f"\n[filters] Failed ({len(failures)}):")
    for failure in failures:
        print(wrap(failure, indent="  - ", hanging="    "))
    return 1


def run_filter_info_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Pick instruments, as when downloading, and show what ``filters-info`` shows for each.

    Returns 1 if any report was incomplete because a source could not be asked, else 0.
    """
    from ..filters.library import FilterLibrary

    library = FilterLibrary(base_dir)
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    incomplete: List[str] = []
    show = lambda facility, instrument: _show_filter_set(library, facility, instrument, incomplete)
    while facilities:
        index = prompt_choice(_picker_labels(facilities), "Filter Facilities")
        if index is None or not _browse_instruments(library, facilities[index], show, shown=shown):
            break
    return 1 if incomplete else 0


def run_filter_compose_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Pick an instrument and build a curve from its own parts, asking what is asked after a download.

    The published curves are downloaded first unless installed; an instrument
    without parts of its own is named in one line.  Errors propagate.
    """
    from ..filters.library import FilterLibrary

    library = FilterLibrary(base_dir)
    picked = _pick_one_instrument(library)
    if picked is None:
        return 0
    facility, instrument, _ = _picked_names(library, *picked)
    info = library.available(facility, instrument)
    if info.get("availability_source") == "unknown":
        raise RuntimeError(f"cannot tell what {facility}/{instrument} is made of: " + "; ".join(info["errors"]))
    if not _has_parts(info):
        print(f"{facility}/{instrument} has no separate parts to build a filter curve from.")
        return 0
    directory = library.published(facility, instrument)
    info = _installed_availability(directory)
    print_filter_availability(info)
    _compose_interactively(directory, info)
    return 0


def _pick_one_instrument(library) -> Optional[Tuple[dict, dict]]:
    """The facility and instrument entries the user picks, or None if they quit."""
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    picked: List[Tuple[dict, dict]] = []

    def pick(facility: dict, instrument: dict) -> bool:
        picked.append((facility, instrument))
        return False  # stop browsing

    while facilities and not picked:
        index = prompt_choice(_picker_labels(facilities), "Filter Facilities")
        if index is None or not _browse_instruments(library, facilities[index], pick, shown=shown):
            break
    return picked[0] if picked else None


def _listed_facilities(library, shown: set) -> List[dict]:
    """The facilities the library lists; a listing failure is printed once."""
    catalogue = library.facilities()
    _print_listing_errors(catalogue["errors"], shown)
    if not catalogue["entries"]:
        print("No facilities found.")
    return catalogue["entries"]


#: Sources in the order Filters.fetch tries them, and how the browser names them.
_SOURCE_ORDER = ("local", "njm", "original", "svo")


_SOURCE_NAMES = {"njm": "NJM", "original": "the original source", "svo": "SVO"}


def _fetch_key(entry: dict) -> str:
    """The spelling to download an entry with: its ``fetch_key`` when the catalogue gives one."""
    return entry.get("fetch_key") or entry["key"]


def _picker_labels(entries: List[dict]) -> List[str]:
    """The name the pickers show for each catalogue entry: the library's label, nothing else."""
    return [entry.get("label") or entry["key"] for entry in entries]


def _source_rank(source: str) -> int:
    return _SOURCE_ORDER.index(source) if source in _SOURCE_ORDER else len(_SOURCE_ORDER)


def _print_listing_errors(errors: List[str], shown: set) -> None:
    """Print each listing failure once per browsing session."""
    for error in _listing_failures(errors):
        if error not in shown:
            shown.add(error)
            print(f"[filters] {error}")


def _download_every_instrument(library, facilities: List[dict], failures: List[str], shown: set) -> None:
    """After confirming the count, download every instrument of each facility.

    An incomplete instrument list is a failure too.
    """
    listings = []
    for facility in facilities:
        result = library.instruments(facility["key"])
        _print_listing_errors(result["errors"], shown)
        for error in _listing_failures(result["errors"]):
            failures.append(f"{facility['key']}: instrument list incomplete: {error}")
        listings.append((facility, result["entries"]))
    count = sum(len(entries) for _, entries in listings)
    print(f"\n{count} instrument{'s' * (count != 1)} in {len(facilities)} facilities: "
          + ", ".join(f"{f['label']} ({len(entries)})" for f, entries in listings))
    if not ask_yes(f"Download all {count}? [y/N] "):
        print("Nothing downloaded.")
        return
    for facility, entries in listings:
        for instrument in entries:
            _fetch_instrument(library, _fetch_key(facility), _fetch_key(instrument), failures)


def _browse_instruments(library, facility_entry: dict, act, *, shown: Optional[set] = None) -> bool:
    """Pick instruments of one facility, calling ``act(facility_entry, instrument_entry)`` for each.

    ``act`` returns whether to show the instrument list again.  Returns False if the user quit.
    """
    facility = _fetch_key(facility_entry)
    result = library.instruments(facility_entry["key"])
    _print_listing_errors(result["errors"], set() if shown is None else shown)
    instruments = result["entries"]
    if not instruments:
        print(f"No instruments found for {facility}.")
        return True
    _print_instrument_descriptions(facility, instruments)
    while True:
        index = prompt_choice(_picker_labels(instruments), f"Instruments for {facility}", allow_back=True)
        if index is None:
            return False
        if index == -1 or not act(facility_entry, instruments[index]):
            return True


def _print_instrument_descriptions(facility: str, instruments: List[dict]) -> None:
    """Each instrument of a facility, named in full, with its one-line description cut to the terminal."""
    labels = _picker_labels(instruments)
    width, name_width = terminal_width(), max(map(len, labels))
    print(f"\n{facility}: {len(labels)} instrument{'s' * (len(labels) != 1)}")
    for label, entry in zip(labels, instruments):
        description = " ".join(str(entry.get("description") or "").split()) or "no description"
        room = width - name_width - 4
        print(f"  {label:<{name_width}}  {cut(description, room)}" if room > 1 else f"  {label}")


def _offer_download(library, facility_entry: dict, instrument_entry: dict, failures: List[str]) -> bool:
    """Describe a picked instrument, then download it unless the user declines.

    Returns whether to show the instrument list again: at once when the
    download is declined, else if the user asks for another instrument.
    """
    facility, instrument, installed = _picked_names(library, facility_entry, instrument_entry)
    _print_pick_summary(library, facility, instrument, instrument_entry, installed)
    if installed is None and input(f"Download {facility}/{instrument}? [Y/n] ").strip().lower().startswith("n"):
        return True
    _fetch_instrument(library, facility, instrument, failures, offer=True)
    return ask_yes("Choose another instrument? [y/N] ")


def _show_filter_set(library, facility_entry: dict, instrument_entry: dict, incomplete: List[str]) -> bool:
    """Print what ``filters-info`` prints for a picked instrument; returns whether to pick another."""
    facility, instrument, _ = _picked_names(library, facility_entry, instrument_entry)
    info = library.available(facility, instrument)
    _print_filter_set(f"{facility}/{instrument}", info,
                      _catalogue_entry(facility, instrument, str(library.root), offline=False))
    incomplete.extend(_availability_errors(info))
    return ask_yes("Choose another instrument? [y/N] ")


def _picked_names(library, facility_entry: dict, instrument_entry: dict) -> Tuple[str, str, Optional[Path]]:
    """The names to fetch a picked instrument with (the library's ``fetch_key``), and its installed
    directory when it holds the complete set, else None."""
    facility, instrument = _fetch_key(facility_entry), _fetch_key(instrument_entry)
    try:
        directory = library.installed_directory(facility, instrument)
    except ComponentError:  # installed under two spellings: the download step names both
        directory = None
    return facility, instrument, directory if directory is not None and library.published_complete(directory) else None


def _print_pick_summary(library, facility: str, instrument: str, entry: dict, installed: Optional[Path]) -> None:
    """Before the download question, a short version of what ``filters-info`` says about the instrument."""
    info = library.available(facility, instrument)
    catalogue = _catalogue_entry(facility, instrument, str(library.root), offline=False)
    for line in _pick_summary(f"{facility}/{instrument}", entry, info, catalogue,
                              _download_origin(library, entry, installed)):
        print(line)


#: The width of a summary line's ``  Name       : `` part.
_FIELD_WIDTH = 15


def _download_origin(library, entry: dict, installed: Optional[Path]) -> str:
    """Where a download comes from: the installed copy, or the sources the library lists in download order.

    The installed copy's full path is given when it fits the terminal beside
    the field name, else its place in the filter directory.
    """
    if installed is not None:
        text = f"installed in {installed}"
        if len(text) > terminal_width() - _FIELD_WIDTH:
            text = f"installed in the filter directory, as {installed.relative_to(library.root)}"
        return text
    sources = sorted(set(entry.get("sources") or []) - {"local"}, key=_source_rank)
    njm_down = "njm" in sources and not library.mirror.is_available()
    names = [_SOURCE_NAMES.get(source, source) for source in sources if not (source == "njm" and njm_down)]
    if not names:
        return "only the installed copy, which is incomplete" if "local" in (entry.get("sources") or []) \
            else "no source lists it"
    return ", then ".join(names) + ("; NJM is unreachable" if njm_down else "")


def _pick_summary(target: str, entry: dict, info: dict, catalogue: dict, origin: str) -> List[str]:
    """The lines of the summary: description, sources, bands, components, atmosphere and licence.

    ``info`` is Filters.available's report and ``catalogue`` _catalogue_entry's,
    as filters-info shows them; each line is cut with '…' to fit the terminal.
    """
    width = terminal_width()
    label = lambda name: f"  {name:<{_FIELD_WIDTH - 4}}: "
    field = lambda name, text: cut(label(name) + " ".join(str(text).split()), width)
    njm = catalogue["info"] or {}
    description = entry.get("description") or _first_sentence(njm.get("description") or "")
    count = len(info["bands"])
    lines = ["", target + (f"  ({count} band{'s' * (count != 1)})" if count else "")]
    if description:
        lines.append(cut("  " + " ".join(description.split()), width))
    lines.append(field("From", origin))
    if _missing_bands(info):
        lines.append(field("Incomplete", _incomplete_summary(info, width - _FIELD_WIDTH)))
    lines += [field("Note", note) for note in [info.get("component_status"), *(info.get("notes") or [])] if note]
    lines += [field("Unavailable", error) for error in info["errors"]]
    bands = _names_to_fit(info["bands"], width - _FIELD_WIDTH) if info["bands"] else "not known"
    lines.append(field("Bands", bands))
    lines.append(field("Components", _curve_contents(info)))
    lines.append(field("Separate", _component_summary(info)))
    if info.get("variants"):
        lines.append(field("Variants", "API only; " + _variants_summary(info["variants"])))
    sites = _atmosphere_sites(info)
    lines.append(field("Atmosphere", f"own site{'s' * (len(sites) > 1)}: "
                       + _site_names(sites, _default_site(info, sites)) if sites else _atmosphere_summary(info)))
    if njm.get("licence"):
        lines.append(field("Licence", _licence_summary(njm["licence"])))
    elif catalogue["errors"]:
        lines.append(field("Licence", f"not known ({catalogue['errors'][0]})"))
    if njm.get("references"):
        lines.append(field("Cite", _cite_hint(target, njm["references"])))
    return lines


def _names_to_fit(names: List[str], room: int) -> str:
    """``names`` joined with commas, cut after a name to fit ``room`` characters, ending '… (N more)'."""
    for shown in range(len(names), -1, -1):
        rest = len(names) - shown
        text = ", ".join(names[:shown] + ([f"… ({rest} more)"] if rest else []))
        if len(text) <= room or shown == 0:
            return text


def _curve_contents(info: dict) -> str:
    """What each published curve contains, e.g. ``filter+optics+qe in each curve``."""
    published = info["published"] or "unknown"
    if published == "mixed":
        return "differs by band: " + ", ".join(sorted(set(info["published_bands"].values())))
    return published if published == "unknown" else f"{published} in each curve"


def _fetch_instrument(library, facility: str, instrument: str, failures: List[str], *,
                      offer: bool = False) -> None:
    """Download one instrument (then offer a composed product); a failure is printed and recorded.

    This is Filters.fetch with the browsing session's library, so NJM is
    checked, and an unreachable NJM reported, once per session, not per download.
    """
    try:
        directory = library.published(facility, instrument)
    except (ValueError, RequestException, OSError, RuntimeError) as exc:
        print(f"Cannot fetch: {exc}")
        failures.append(f"{facility}/{instrument}: {exc}")
        return
    if not offer:
        return
    try:
        _offer_composition(directory)
    except (ValueError, RequestException, OSError, RuntimeError) as exc:  # ComponentError is a ValueError
        print(f"Composed product of {facility}/{instrument} not built: {exc}")
        failures.append(f"{facility}/{instrument} composed product: {exc}")


class _FilterSetOption:
    """Display wrapper for a local filter-set directory, used by run_filter_combine_flow."""
    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        self.label = f"{rel} ({_set_curve_count(path)} filters)"


def _set_curve_count(directory: Path) -> int:
    """How many curves combine_filter_sets takes from a filter set.

    Those its MESA index file lists (every ``*.dat`` without one), less its
    manifest's ``archived_bands``; so NJM's ``_svo`` and ``_old`` copies beside
    a set are not counted.
    """
    from ..filters.compose import archived_bands, index_listing

    archived = archived_bands(directory)
    listed = index_listing(directory)
    return sum(1 for path in directory.glob("*.dat")
               if path.stem not in archived and (listed is None or path.name in listed))


def run_filter_combine_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """Interactive wizard for combining local filter sets."""
    from ..filters.combine import combine_filter_sets, find_filter_sets

    print("\nCombine photometric filter sets")
    print("This creates one MESA-compatible filter folder and index file")
    print("from existing .dat files.")

    chosen_base = input(f"Filter base directory [{base_dir}]: ").strip() or base_dir
    filter_sets = find_filter_sets(chosen_base)
    if not filter_sets:
        print(f"No local filter sets containing .dat files were found under {chosen_base}.")
        print("Download filters first with option 1, or run `sed-tools filters`.")
        return

    root = Path(chosen_base)
    options = [_FilterSetOption(path, root) for path in filter_sets]
    selected = prompt_choice(
        options,
        "Local filter sets to combine",
        multi=True,
        page_size=50,
        max_cols=2,
    )
    if selected is None:
        print("No filter sets selected.")
        return

    selected_paths = [options[i].path for i in selected]
    print("\nSelected filter sets:")
    for path in selected_paths:
        print(f"  - {path}")

    default_name = "_".join(path.name for path in selected_paths[:3]) or "CombinedFilters"
    output = input(f"Combined instrument name [{default_name}]: ").strip() or default_name
    facility = input("Output facility label [Combined]: ").strip() or None
    instrument = input(f"Output index/instrument name [{output}]: ").strip() or None

    conflict = input("Duplicate band names: rename, overwrite, or error? [rename]: ").strip().lower() or "rename"
    while conflict not in {"rename", "overwrite", "error"}:
        conflict = input("Please enter rename, overwrite, or error [rename]: ").strip().lower() or "rename"

    print("\nAbout to create a combined filter set:")
    print(f"  Base      : {chosen_base}")
    print(f"  Output    : {output}")
    print(f"  Facility  : {facility or 'Combined'}")
    print(f"  Instrument: {instrument or output}")
    print(f"  Conflicts : {conflict}")
    confirm = input("Proceed? [Y/n] ").strip().lower()
    if confirm and not confirm.startswith("y"):
        print("Cancelled.")
        return

    out = combine_filter_sets(
        output,
        selected_paths,
        filter_root=chosen_base,
        facility=facility,
        instrument=instrument,
        on_conflict=conflict,
    )
    dat_count = len(list(out.glob("*.dat")))
    index_name = instrument or out.name
    print(f"\n[filters] Combined {dat_count} filters into {out}")
    print(f"[filters] Wrote MESA index file: {out / index_name}")
    print("Use this path as the MESA instrument/filter-set directory.")


def _availability_errors(info: dict) -> List[str]:
    """The failures that leave an availability report incomplete.

    A source in ``fallback_from`` (NJM, say, when SVO answered instead) leaves nothing out.
    """
    fallback = set(info.get("fallback_from") or ())
    return [message for source, messages in (info.get("sources_failed") or {}).items()
            if source not in fallback for message in messages]
