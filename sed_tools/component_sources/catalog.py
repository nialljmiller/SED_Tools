"""Which instruments have separable components, from where, and why not.

``BUILDERS_FOR`` maps a local ``<Facility>/<Instrument>`` directory to the
builder that fetches that instrument's own published factors.  The njm
mirror runs them ahead of time; a client that downloaded the published
curves elsewhere (e.g. SVO) runs them itself (see
:func:`ensure_components`).  ``heavy`` builders download large archives and
only run on request.  ``publishes`` builders can also write an instrument's
published curves from the original sources, so :meth:`FilterLibrary.published`
uses them when the njm mirror is unavailable; builders with
``components=False`` only reproduce the published curves (the team publishes
no separate factors), so they are not run to obtain components.

``STATUS`` records instruments whose teams publish only a total response,
or whose component data cannot currently be retrieved, so that users are
told why only the published curve is offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

@dataclass(frozen=True)
class BuilderRef:
    name: str
    kwargs: Tuple[Tuple[str, object], ...] = ()
    heavy: bool = False  # large download; not run automatically on clients
    publishes: bool = False  # can rebuild the published curves from the original sources
    components: bool = True  # supplies separately published components


def _tabular(instrument: str) -> BuilderRef:
    return BuilderRef("tabular", (("instrument", instrument),), publishes=True)


#: Instruments rebuilt from their teams' original files (``component_sources``
#: modules ``surveys``, ``space``, ``observatories``, ``standards``, ``swift``,
#: ``csp`` and ``cassini``), as
#: (facility, instrument, separate components published?).
ORIGINAL_SOURCES: Dict[str, Tuple[Tuple[str, str, bool], ...]] = {
    "surveys": (
        ("ATLAS", "ATLAS", False), ("PAN-STARRS", "PS1", False), ("OGLE", "OGLE-IV", True),
        ("UKIRT", "WFCAM", True), ("LasCumbres", "Sinistro", False), ("LasCumbres", "Spectral", False),
    ),
    "space": (
        ("GAIA", "DR2", False), ("GAIA", "DR2r", False), ("Herschel", "PACS", False),
        ("Herschel", "SPIRE", False), ("Hipparcos", "Hipparcos", False), ("Hipparcos", "Tycho", False),
        ("IRAS", "IRAS", True), ("MSX", "MSX", False), ("Planck", "HFI", False), ("Planck", "LFI", False),
        ("Spitzer", "IRAC", False), ("Spitzer", "MIPS", False), ("Spitzer", "IRS", False),
        ("XMM", "OM", False),
    ),
    "observatories": (
        ("Paranal", "FORS1", False), ("Paranal", "FORS2", False), ("Paranal", "OmegaCAM", True),
        ("Keck", "NIRC2", False), ("Keck", "LRIS", False), ("Keck", "MOSFIRE", False),
        ("Keck", "DEIMOS", False), ("KPNO", "Mosaic", False), ("Magellan", "FourStar", False),
        ("Magellan", "MegaCam", True), ("NOT", "ALFOSC", False), ("Subaru", "SuprimeCam", True),
        ("Subaru", "FOCAS", False), ("WIYN", "ODI", False),
    ),
    "standards": (
        ("Generic", "Bessell", False), ("Generic", "MKO", False), ("Generic", "USNO", False),
    ),
    "swift": (("Swift", "UVOT", True),),
    "csp": (("LCO", "CSP", True), ("LCO", "CSP2", False)),
    "cassini": (("Cassini", "ISS_NAC", True), ("Cassini", "ISS_WAC", True)),
}


def _original() -> Dict[Tuple[str, str], BuilderRef]:
    return {(fac, inst): BuilderRef(module, (("instrument", inst),), publishes=True, components=comps)
            for module, entries in ORIGINAL_SOURCES.items() for fac, inst, comps in entries}


def _hst() -> Dict[Tuple[str, str], BuilderRef]:
    from .hst import MODES

    return {("HST", inst): BuilderRef("hst", (("instruments", (inst,)),)) for inst in MODES}


BUILDERS_FOR: Dict[Tuple[str, str], BuilderRef] = {
    ("LSST", "LSST"): BuilderRef("lsst"),
    ("LSST", "LSSTv19"): BuilderRef("lsst", (("instrument", "LSSTv19"),), publishes=True),
    ("CFHT", "Megaprime"): BuilderRef("cfht", (("instrument", "Megaprime"),)),
    ("CFHT", "MegaCam"): BuilderRef("cfht", (("instrument", "MegaCam"),)),
    ("ERIS", "NIX"): BuilderRef("eris"),
    ("2MASS", "2MASS"): BuilderRef("twomass"),
    ("Subaru", "HSC"): BuilderRef("hsc"),
    ("CTIO", "DECam"): BuilderRef("decam"),
    ("Paranal", "VISTA"): BuilderRef("vista"),
    ("Euclid", "NISP"): BuilderRef("euclid"),
    ("Roman", "WFI"): BuilderRef("roman"),
    ("Palomar", "ZTF"): BuilderRef("ztf"),
    ("BOK", "BASS"): BuilderRef("legacy_surveys", (("instrument", "BASS"),)),
    ("KPNO", "MzLS"): BuilderRef("legacy_surveys", (("instrument", "MzLS"),), publishes=True),
    ("Gemini", "GMOS_N"): BuilderRef("gmos", (("instrument", "GMOS_N"),)),
    ("Gemini", "GMOS_S"): BuilderRef("gmos", (("instrument", "GMOS_S"),)),
    ("SLOAN", "SDSS"): BuilderRef("sdss"),
    # Argus inputs are partly non-public: without them the builder raises and points to NJM.
    ("Argus", "DR280"): BuilderRef("argus", (("instrument", "DR280"),), publishes=True),
    ("Argus", "System"): BuilderRef("argus", (("instrument", "System"),)),
    ("Paranal", "HAWKI"): _tabular("HAWKI"),
    ("Paranal", "NACO"): _tabular("NACO"),
    ("Paranal", "SPHERE"): _tabular("SPHERE"),
    ("Gemini", "GSAOI"): _tabular("GSAOI"),
    ("Gemini", "NIFS"): _tabular("NIFS"),
    ("CAHA", "PANIC"): _tabular("PANIC"),
    ("WHT", "INGRID"): _tabular("INGRID"),
    ("IRTF", "NSFCam"): _tabular("NSFCam"),
    ("Liverpool", "IOO"): _tabular("IOO"),
    ("JWST", "NIRCam"): BuilderRef("jwst", (("instruments", ("NIRCam",)),)),
    ("JWST", "NIRISS"): BuilderRef("jwst", (("instruments", ("NIRISS",)),)),
    ("JWST", "MIRI"): BuilderRef("jwst", (("instruments", ("MIRI",)),)),
    **_hst(),
    **_original(),
}

TOTAL_ONLY = "total-only"
BLOCKED = "blocked"
#: The team publishes separate factors, but no builder uses them yet.
NOT_BUILT = "not-built"

#: (facility, instrument or "*") -> (status, explanation)
STATUS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("GAIA", "*"): (TOTAL_ONLY, "ESA publishes only total passbands (Riello et al. 2021)."),
    ("WISE", "*"): (TOTAL_ONLY, "WISE publishes only total relative spectral responses (Wright et al. 2010)."),
    ("Spitzer", "IRAC"): (TOTAL_ONLY, "IRAC publishes only total spectral response curves."),
    ("Spitzer", "MIPS"): (TOTAL_ONLY, "MIPS publishes only total spectral response curves."),
    ("GALEX", "*"): (TOTAL_ONLY, "GALEX publishes only total effective areas."),
    ("LCO", "CSP2"): (TOTAL_ONLY, "CSP-II publishes only total passbands (telescope+camera scans including atmosphere); no separate components."),
    ("TESS", "*"): (TOTAL_ONLY, "TESS publishes only its total response function (filter x detector)."),
    ("Kepler", "*"): (TOTAL_ONLY, "Kepler publishes only its total response function."),
    ("Hipparcos", "*"): (TOTAL_ONLY, "Only total passbands are published."),
    ("TYCHO", "*"): (TOTAL_ONLY, "Only total passbands are published."),
    ("PAN-STARRS", "PS1"): (NOT_BUILT, "Tonry et al. (2012) Table 3 (CDS J/ApJ/750/99, table3.dat) gives each "
                                       "band's capture cross-section and the open (no filter) one, both through "
                                       "1.2 airmasses, and that atmosphere as Aero, Ray and Mol. It has no separate "
                                       "filter, optics or QE curves, and no builder registers its atmosphere yet."),
}


def name_key(name: str) -> str:
    """A name compared without case or punctuation: ``S-PLUS``, ``S_PLUS`` and ``s plus`` match."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def match_name(name: str, names: Iterable[str]) -> Optional[str]:
    """The one of ``names`` that is ``name``, else the only one equal to it under :func:`name_key`.

    None when nothing matches, or when several names match only under
    :func:`name_key` (NJM serves both ``Sony_NEX5N`` and ``Sony_NEX_5N``).
    """
    names = list(names)
    if name in names:
        return name
    found = [candidate for candidate in names if name_key(candidate) == name_key(name)]
    return found[0] if len(found) == 1 else None


def registered_name(facility: str, instrument: str) -> Optional[Tuple[str, str]]:
    """The ``BUILDERS_FOR`` spelling of an instrument (``Gemini/GMOS-N`` -> ``Gemini/GMOS_N``), or None."""
    pairs = list(BUILDERS_FOR)
    fac = match_name(facility, {f for f, _ in pairs})
    if fac is None:
        return None
    inst = match_name(instrument, [i for f, i in pairs if f == fac])
    return None if inst is None else (fac, inst)


def builder_for(facility: str, instrument: str) -> Optional[BuilderRef]:
    """The builder of an instrument, matched as :func:`registered_name` does."""
    names = registered_name(facility, instrument)
    return None if names is None else BUILDERS_FOR[names]


def status_for(facility: str, instrument: str) -> Optional[Tuple[str, str]]:
    """Status of an instrument; ``(facility, "*")`` entries cover every instrument of a facility.

    Names are matched as :func:`match_name` does.
    """
    pairs = list(STATUS)
    fac = match_name(facility, {f for f, _ in pairs})
    if fac is None:
        return None
    inst = match_name(instrument, [i for f, i in pairs if f == fac and i != "*"])
    return STATUS.get((fac, inst)) or STATUS.get((fac, "*"))


def component_availability(facility: str, instrument: str) -> Dict[str, object]:
    """Source review status, distinct from which files are installed.

    An absent registry entry means unreviewed, not that a team's components
    do not exist. This vocabulary also applies to newly discovered SVO sets.
    """
    ref = builder_for(facility, instrument)
    status = status_for(facility, instrument)
    review, note = status or ("unreviewed", "Separate upstream components have not yet been reviewed.")
    if ref and ref.components:
        review, note = "supported", ""
    return {
        "status": review,
        "builder": ref.name if ref and ref.components else None,
        "original_published": bool(ref and ref.publishes),
        "large_download": bool(ref and ref.heavy),
        "note": note,
    }


def ensure_components(
    instrument_dir: Union[str, Path], *, allow_heavy: bool = False,
    components=None, bands=None,
) -> bool:
    """Compatibility entry point for the shared filter acquisition workflow."""
    from ..filter_library import FilterLibrary

    directory = Path(instrument_dir)
    return FilterLibrary(directory.parent.parent).prepare_components(
        directory, components=components, bands=bands, allow_heavy=allow_heavy)
