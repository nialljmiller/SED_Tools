"""VISTA/VIRCAM components (ESO).

ESO's VIRCAM ETC bundle (``Filters_QE_Atm_curves.tar.gz``) has the filter
transmissions at operating temperature, the detector QE and both mirror
reflectivities, all in nm and percent:

* filter: ``VISTA_Filters_at80K_forETC_<b>`` (broad bands), ``VISTA_Filters_forETC_<NB>``
* optics: M1 x M2 reflectivity
* qe:     ``qe.tab``

VISTA is at Paranal, so its own-site atmosphere is the SkyCalc ``paranal`` grid.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

from ._common import SimpleSource, Table, build_simple
from ._original import CONNECT_TIMEOUT

BUNDLE = "https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst/Filters_QE_Atm_curves.tar.gz"
_CACHE = {}


@dataclass
class TarMember(Table):
    member: str = ""

    def read(self, session):
        if self.url not in _CACHE:
            resp = session.get(self.url, timeout=(CONNECT_TIMEOUT, 300))
            resp.raise_for_status()
            _CACHE[self.url] = resp.content
        with tarfile.open(fileobj=io.BytesIO(_CACHE[self.url]), mode="r:gz") as tf:
            return self.parse(tf.extractfile(f"Filters_QE_Atm_curves/{self.member}").read())


def _m(member, note=""):
    return TarMember(BUNDLE, wl_scale=10.0, tr_scale=0.01, member=member, note=note)


BROAD = ("Z", "Y", "J", "H", "Ks")
NARROW = ("NB118", "NB980", "NB990")

SPEC = SimpleSource(
    facility="Paranal",
    instrument="VISTA",
    filters={**{b: _m(f"VISTA_Filters_at80K_forETC_{b}.dat.gz", "at 80 K") for b in BROAD},
             **{b: _m(f"VISTA_Filters_forETC_{b}.dat.gz") for b in NARROW}},
    optics=(_m("VISTA_M1_Reflectivity_forETC_2009-Sep12.txt.gz", "M1"),
            _m("VISTA_M2_Reflectivity_forETC_2007-Jun19.txt.gz", "M2")),
    qe=_m("qe.tab.gz", "VIRCAM detector QE"),
    svo_ids={b: f"Paranal/VISTA.{b}" for b in BROAD + NARROW},
    default_site="paranal",
    # Mirror tables stop at 3.0 um while the filters' red leaks (and the QE)
    # extend to ~3.2-3.5 um; metal reflectivity is flat there, so hold it.
    outside_range="edge",
    references={"eso": BUNDLE},
    notes=("ESO VIRCAM ETC curves; atmosphere is the SkyCalc Paranal grid (VISTA's own site). "
           "Mirror reflectivities are held at their 3.0 um values beyond their tables (SED_Tools "
           "choice: metal reflectivity is flat there; ESO does not document its ETC's treatment)."),
)


def build(filter_root, **kwargs):
    return build_simple(SPEC, filter_root, **kwargs)
