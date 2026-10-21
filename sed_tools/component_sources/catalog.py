"""Which instruments have separable components, from where, and why not.

``BUILDERS_FOR`` maps a local ``<Facility>/<Instrument>`` directory to the
builder that fetches that instrument's own published factors.  The njm
mirror runs them ahead of time; a client that downloaded the published
curves elsewhere (e.g. SVO) runs them itself (see
:func:`ensure_components`).  ``heavy`` builders download large archives and
only run on request.

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


def _hst() -> Dict[Tuple[str, str], BuilderRef]:
    from .hst import MODES

    return {("HST", inst): BuilderRef("hst", (("instruments", (inst,)),)) for inst in MODES}


BUILDERS_FOR: Dict[Tuple[str, str], BuilderRef] = {
    ("LSST", "LSST"): BuilderRef("lsst"),
    ("Subaru", "HSC"): BuilderRef("hsc"),
    ("CTIO", "DECam"): BuilderRef("decam"),
    ("Paranal", "VISTA"): BuilderRef("vista"),
    ("Palomar", "ZTF"): BuilderRef("ztf"),
    ("JWST", "NIRCam"): BuilderRef("jwst", (("instruments", ("NIRCam",)),), heavy=True),
    ("JWST", "NIRISS"): BuilderRef("jwst", (("instruments", ("NIRISS",)),), heavy=True),
    ("JWST", "MIRI"): BuilderRef("jwst", (("instruments", ("MIRI",)),), heavy=True),
    **_hst(),
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
    if ref:
        review, note = "supported", ""
    return {
        "status": review,
        "builder": ref.name if ref else None,
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
