"""Where each SVO facility observes from: the ground, space, or the stratosphere.

Only ground-based facilities can have a telluric (atmosphere) component
composed in; SkyCalc models ground sites only.  Keys are SVO facility names;
lookups ignore case and punctuation so directory names written by the
grabbers (``Bepi_Colombo`` for ``Bepi-Colombo``) match.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

GROUND, SPACE, AIRBORNE = "ground", "space", "airborne"

_GROUND = """
2MASS AAO AndromedA APEX APO Argus Atlas BICEP BOK CAHA CFHT CMP-SAI CRTS CSTAR
CTIO DDO DECam DENIS DOT ERIS EROS ESO Flock FLWO Gemini GCPD Generic Geneva GOTO GTC
HAWK-I HCT SPHERE
IAC80 ING INT IRAM IRSF IRTF JCMT Keck KPNO La-Silla LasCumbres LaSilla LBT LCO
LICK Liverpool LMT LSST LT McDonalds Mercator MKO MMT MOA NIRT NOAO NOT OAF OAJ
OAN-SPM OAO OHP OSN OVRO P200 Palomar PAN-STARRS Paranal QUIET QUIJOTE ROTSE
SALT SAO Scorpio SkyMapper SLOAN SPECULOOS_S SPT STELLA Subaru TCS Teide TJO TNG
TNO TNT UKIRT VATT WASP WHT WIYN Xinglong
"""

_SPACE = """
ADEOS AKARI Akatsuki AlSat1 Aqua Astrosat Bepi-Colombo Cassini CASTOR CHEOPS
Clementine COBE COMS Contour Corot CSST Dawn DeepImpact DSCOVR Envisat ERBS ERS
Euclid ExoMars FY-1D FY-2C FY-2D FY-3A FY-3B FY-3C GAIA GALEX Galileo GCOM-C
GeoEye Giotto GOES Hayabusa2 Herschel Himawari-1 Himawari-2 Himawari-3 Himawari-4
Himawari-5 Himawari-6 Himawari-7 Himawari-8 Himawari-9 Hipparcos HST IKONOS
INSAT-1B INSAT-3D INSAT-3DR InSight Integral IRAS IRS ISO ISS IUE JPSS Juno JWST
Kepler KOMPSTAT Landsat LRO LROC Mariner10 Mars2020 Megha-Tropiques MER Messenger
Meteosat METOP MEX MOM MOST MRO MSX NEAR NewHorizons NIMBUS NOAA Odyssey
OSIRIS-REX Parasol Pathfinder PLANCK Pleiades PRIRODA QuickBird RapidEye Roman
Rosetta SeaStar Selene Sentinel Sentinel3 SEOSAT SkySat SMART1 SOHO SolarOrbiter
Spitzer SPOT SSOT Stardust Swift TAUVEX TD1 Terra TESS TIROS-N TRMM TYCHO
VenusExpress Viking Voyager WFIRST WISE WMAP WorldView XMM ZiYuan
"""

#: Stratospheric balloons and aircraft: above most, but not all, of the atmosphere.
_AIRBORNE = "ARCHEOPS ARGO BLAST BOOMERANG MAXIMA OLIMPO SOFIA SPIDER TopHat"


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_PLATFORMS = {}
for _platform, _names in ((GROUND, _GROUND), (SPACE, _SPACE), (AIRBORNE, _AIRBORNE)):
    for _name in _names.split():
        _PLATFORMS[_key(_name)] = _platform


def platform(facility: str) -> Optional[str]:
    """``"ground"``, ``"space"``, ``"airborne"`` or ``None`` if unknown."""
    return _PLATFORMS.get(_key(facility))


def is_ground_based(facility: str) -> Optional[bool]:
    p = platform(facility)
    return None if p is None else p == GROUND


#: Atmosphere site of each facility, where one is available *for that site*.
#: SkyCalc models Cerro Paranal (and La Silla's altitude in the same
#: climate), so only ESO facilities there get a SkyCalc atmosphere; other
#: observatories get one only when their team publishes a standard
#: atmosphere (set by that instrument's component builder).
_OWN_SITES = {
    "paranal": "Paranal ERIS HAWK-I SPHERE",
    "lasilla": "LaSilla La-Silla",
}
_SITE_OF = {_key(name): site for site, names in _OWN_SITES.items() for name in names.split()}


def own_site(facility: str) -> Optional[str]:
    """The atmosphere site a facility actually observes from, if we model it."""
    return _SITE_OF.get(_key(facility))


#: Standard atmospheres published by an instrument's own team, registered as
#: named sites (built by ``component_sources.team_atmospheres``).  They are
#: offered next to the instrument's ``default_site``; the first one is the
#: default when the manifest has none.  An instrument is listed only when the
#: atmosphere describes the location it observed from; instruments that moved
#: between sites (GPI, GNIRS) are deliberately absent.
_INSTRUMENT_SITES = {
    ("PAN-STARRS", "PS1"): ("haleakala_ps1",),
    ("Paranal", "VISTA"): ("paranal_vircam",),
    ("2MASS", "2MASS"): ("hopkins_2mass", "ctio_2mass"),
    ("SLOAN", "SDSS"): ("apo_sdss",),
    ("BOK", "BASS"): ("kpno_bass",),
    ("LCO", "CSP"): ("lco_csp",),
    ("Gemini", "NIFS"): ("maunakea_gemini",),
    ("Gemini", "GSAOI"): ("cpachon_gemini",),
    ("Gemini", "Flamingos2"): ("cpachon_gemini",),
}
_SITES_OF = {(_key(f), _key(i)): sites for (f, i), sites in _INSTRUMENT_SITES.items()}


def instrument_sites(facility: str, instrument: str) -> Tuple[str, ...]:
    """Team atmosphere sites registered for one instrument (may be empty)."""
    return _SITES_OF.get((_key(facility), _key(instrument)), ())
