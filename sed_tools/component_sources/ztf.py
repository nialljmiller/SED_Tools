"""ZTF components (ZwickyTransientFacility/ztf_information).

* filter: vendor-measured on-axis filter transmissions (g, r, i)
* qe:     the e2v CCD QE.  ZTF's 16 CCDs have two anti-reflection coatings
          (8 single-layer, 8 multi-layer), so the camera-averaged QE is the
          equal-weight mean of the two published curves (SED_Tools-derived;
          both inputs are recorded in the header).

No optics (mirror/corrector) throughput is published, so ``filter+qe`` is
the most complete composition.  Palomar has no modelled own-site atmosphere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import SimpleSource, Table, build_simple

REPO = "https://raw.githubusercontent.com/ZwickyTransientFacility/ztf_information/7c1d1ed1adeea47ee024272e797e77d7e65160d5"
SINGLE = f"{REPO}/QE/e2v_ztf_singlelayer.csv"
MULTI = f"{REPO}/QE/e2v_ztf_multi2.csv"


@dataclass
class _MeanQE(Table):
    def read(self, session):
        a = Table(SINGLE, wl_scale=10.0).read(session)
        b = Table(MULTI, wl_scale=10.0).read(session)
        lo, hi = max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1])
        grid = np.unique(np.concatenate([a[0], b[0]]))
        grid = grid[(grid >= lo) & (grid <= hi)]
        return grid, 0.5 * (np.interp(grid, *a) + np.interp(grid, *b))


SPEC = SimpleSource(
    facility="Palomar",
    instrument="ZTF",
    filters={
        "g": Table(f"{REPO}/filter_transmission/ZTF_g_band_test_data_pos1_0AOI.csv", wl_scale=10.0, tr_scale=0.01, note="vendor, on-axis"),
        "r": Table(f"{REPO}/filter_transmission/ZTF_r_band_test_data_pos1_0AOI.csv", wl_scale=10.0, tr_scale=0.01, note="vendor, on-axis"),
        "i": Table(f"{REPO}/filter_transmission/ztf_i_band.csv", wl_scale=10.0, tr_scale=0.01, note="vendor"),
    },
    qe=_MeanQE(f"{SINGLE} + {MULTI}", note="equal-weight mean of single- and multi-layer AR coatings (8+8 CCDs)"),
    svo_ids={"g": "Palomar/ZTF.g", "r": "Palomar/ZTF.r", "i": "Palomar/ZTF.i"},
    outside_range="zero",  # the QE tables end at ~1% on the way to the silicon cutoff
    references={"repository": "https://github.com/ZwickyTransientFacility/ztf_information"},
    notes="QE is the equal-weight mean of the two CCD coatings; no optics throughput is published.",
)


def build(filter_root, **kwargs):
    return build_simple(SPEC, filter_root, **kwargs)
