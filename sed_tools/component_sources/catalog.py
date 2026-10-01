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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


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

#: (facility, instrument) -> (status, explanation)
STATUS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("GAIA", "GAIA"): (TOTAL_ONLY, "ESA publishes only total passbands (Riello et al. 2021)."),
    ("2MASS", "2MASS"): (TOTAL_ONLY, "2MASS publishes only total relative spectral responses incl. atmosphere (Cohen et al. 2003)."),
    ("WISE", "WISE"): (TOTAL_ONLY, "WISE publishes only total relative spectral responses (Wright et al. 2010)."),
    ("Spitzer", "IRAC"): (TOTAL_ONLY, "IRAC publishes only total spectral response curves."),
    ("Spitzer", "MIPS"): (TOTAL_ONLY, "MIPS publishes only total spectral response curves."),
    ("GALEX", "GALEX"): (TOTAL_ONLY, "GALEX publishes only total effective areas."),
    ("TESS", "TESS"): (TOTAL_ONLY, "TESS publishes only its total response function (filter x detector)."),
    ("Kepler", "Kepler"): (TOTAL_ONLY, "Kepler publishes only its total response function."),
    ("Hipparcos", "Hipparcos"): (TOTAL_ONLY, "Only total passbands are published."),
    ("TYCHO", "TYCHO"): (TOTAL_ONLY, "Only total passbands are published."),
    ("PAN-STARRS", "PS1"): (BLOCKED, "Components are in Tonry et al. (2012) Table 3, which is not retrievable "
                                     "programmatically; the IPP wiki copy is the superseded 2010 version."),
}


def builder_for(facility: str, instrument: str) -> Optional[BuilderRef]:
    return BUILDERS_FOR.get((facility, instrument))


def status_for(facility: str, instrument: str) -> Optional[Tuple[str, str]]:
    return STATUS.get((facility, instrument))


def ensure_components(
    instrument_dir: Union[str, Path], *, allow_heavy: bool = False
) -> bool:
    """Run the instrument's own builder if its components are not present.

    Returns True if components were (or already are) available.
    """
    from . import get_builder
    from ..filter_components import InstrumentManifest

    instrument_dir = Path(instrument_dir)
    facility, instrument = instrument_dir.parent.name, instrument_dir.name
    manifest = InstrumentManifest.load(instrument_dir)
    if any("/" in rel for entries in manifest.components.values() for rel in entries.values()):
        return True  # already has downloaded/built components
    ref = builder_for(facility, instrument)
    if ref is None:
        return False
    if ref.heavy and not allow_heavy:
        logger.info("%s/%s: components need a large download; run "
                    "`sed-tools filters-build-components %s` to fetch them", facility, instrument, ref.name)
        return False
    try:
        get_builder(ref.name)(instrument_dir.parent.parent, **dict(ref.kwargs))
        return True
    except Exception as exc:  # network or source changes; the published curves still work
        logger.warning("%s/%s: could not build components (%s)", facility, instrument, exc)
        return False
