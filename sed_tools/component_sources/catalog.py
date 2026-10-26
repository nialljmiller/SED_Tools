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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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
#: modules ``surveys``, ``space``, ``observatories`` and ``standards``), as
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
}


def _original() -> Dict[Tuple[str, str], BuilderRef]:
    return {(fac, inst): BuilderRef(module, (("instrument", inst),), publishes=True, components=comps)
            for module, entries in ORIGINAL_SOURCES.items() for fac, inst, comps in entries}


def _hst() -> Dict[Tuple[str, str], BuilderRef]:
    from .hst import MODES

    return {("HST", inst): BuilderRef("hst", (("instruments", (inst,)),)) for inst in MODES}


BUILDERS_FOR: Dict[Tuple[str, str], BuilderRef] = {
    ("LSST", "LSST"): BuilderRef("lsst"),
    ("Subaru", "HSC"): BuilderRef("hsc"),
    ("CTIO", "DECam"): BuilderRef("decam"),
    ("Paranal", "VISTA"): BuilderRef("vista"),
    ("Palomar", "ZTF"): BuilderRef("ztf"),
    ("BOK", "BASS"): BuilderRef("legacy_surveys", (("instrument", "BASS"),)),
    ("KPNO", "MzLS"): BuilderRef("legacy_surveys", (("instrument", "MzLS"),), publishes=True),
    ("Gemini", "GMOS_N"): BuilderRef("gmos", (("instrument", "GMOS_N"),)),
    ("Gemini", "GMOS_S"): BuilderRef("gmos", (("instrument", "GMOS_S"),)),
    ("SLOAN", "SDSS"): BuilderRef("sdss"),
    ("Paranal", "HAWKI"): _tabular("HAWKI"),
    ("Paranal", "NACO"): _tabular("NACO"),
    ("Paranal", "SPHERE"): _tabular("SPHERE"),
    ("Gemini", "GSAOI"): _tabular("GSAOI"),
    ("Gemini", "NIFS"): _tabular("NIFS"),
    ("CAHA", "PANIC"): _tabular("PANIC"),
    ("WHT", "INGRID"): _tabular("INGRID"),
    ("IRTF", "NSFCam"): _tabular("NSFCam"),
    ("Liverpool", "IOO"): _tabular("IOO"),
    ("JWST", "NIRCam"): BuilderRef("jwst", (("instruments", ("NIRCam",)),), heavy=True),
    ("JWST", "NIRISS"): BuilderRef("jwst", (("instruments", ("NIRISS",)),), heavy=True),
    ("JWST", "MIRI"): BuilderRef("jwst", (("instruments", ("MIRI",)),), heavy=True),
    **_hst(),
    **_original(),
}

TOTAL_ONLY = "total-only"
BLOCKED = "blocked"

#: (facility, instrument or "*") -> (status, explanation)
STATUS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("GAIA", "*"): (TOTAL_ONLY, "ESA publishes only total passbands (Riello et al. 2021)."),
    ("2MASS", "*"): (TOTAL_ONLY, "2MASS publishes only total relative spectral responses incl. atmosphere (Cohen et al. 2003)."),
    ("WISE", "*"): (TOTAL_ONLY, "WISE publishes only total relative spectral responses (Wright et al. 2010)."),
    ("Spitzer", "IRAC"): (TOTAL_ONLY, "IRAC publishes only total spectral response curves."),
    ("Spitzer", "MIPS"): (TOTAL_ONLY, "MIPS publishes only total spectral response curves."),
    ("GALEX", "*"): (TOTAL_ONLY, "GALEX publishes only total effective areas."),
    ("TESS", "*"): (TOTAL_ONLY, "TESS publishes only its total response function (filter x detector)."),
    ("Kepler", "*"): (TOTAL_ONLY, "Kepler publishes only its total response function."),
    ("Hipparcos", "*"): (TOTAL_ONLY, "Only total passbands are published."),
    ("TYCHO", "*"): (TOTAL_ONLY, "Only total passbands are published."),
    ("PAN-STARRS", "PS1"): (BLOCKED, "Components are in Tonry et al. (2012) Table 3, which is not retrievable "
                                     "programmatically; the IPP wiki copy is the superseded 2010 version."),
}


def builder_for(facility: str, instrument: str) -> Optional[BuilderRef]:
    return BUILDERS_FOR.get((facility, instrument))


def status_for(facility: str, instrument: str) -> Optional[Tuple[str, str]]:
    """Status of an instrument; ``(facility, "*")`` entries cover every instrument of a facility."""
    return STATUS.get((facility, instrument)) or STATUS.get((facility, "*"))


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
