"""ZTF components (ZwickyTransientFacility/ztf_information).

* filter: vendor-measured on-axis filter transmissions (g, r, i), in nm and
  percent
* qe:     the e2v CCD QE.  ZTF's 16 CCDs have two anti-reflection coatings:
          the northernmost and southernmost eight are single-layer, the
          inner eight multi-layer (repository README).  Both published
          curves are separate components:

          ``qe`` (default) = ``e2v_ztf_multi2.csv``, variant ``multi_layer``;
          variant ``single_layer`` = ``e2v_ztf_singlelayer.csv``;
          variant ``mean`` = their equal-weight mean, a SED_Tools derivation
          kept because the mirror served it as ``qe.dat`` before
          2026-10-26.  No team document gives a camera-average QE.

The default is the multi-layer QE because SVO's published ``Palomar/ZTF``
curves use it: SVO's ZTF.g comment says "Includes CCD QE with a multi-layer
coating", and filter x multi-layer QE reproduces SVO's r and i to 5e-7 and
2e-6 of the peak, and g to 1.6% of the peak (0.8% when repeated wavelengths
in the vendor g table are averaged); the single-layer and mean QE differ by
9-16% and 4.5-8%.

No optics (mirror/corrector) throughput is published, so ``filter+qe`` is
the most complete composition.  Palomar has no modelled own-site atmosphere.
"""

from __future__ import annotations

import numpy as np

from ..filters.compose import QE, InstrumentManifest, add_component
from ._common import SimpleSource, Table, build_simple, register_variant

REPO = "https://raw.githubusercontent.com/ZwickyTransientFacility/ztf_information/7c1d1ed1adeea47ee024272e797e77d7e65160d5"
README = f"{REPO}/README.md"
SINGLE = f"{REPO}/QE/e2v_ztf_singlelayer.csv"
MULTI = f"{REPO}/QE/e2v_ztf_multi2.csv"
COATINGS = ("ZTF's 16 CCDs have two anti-reflection coatings: the northernmost and southernmost "
            "eight are single-layer, the inner eight multi-layer (ztf_information README).")


def _mean_qe(single, multi):
    """Equal-weight mean of the two QE curves, on their merged samples over their common range."""
    lo, hi = max(single[0][0], multi[0][0]), min(single[0][-1], multi[0][-1])
    grid = np.unique(np.concatenate([single[0], multi[0]]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    return grid, 0.5 * (np.interp(grid, *single) + np.interp(grid, *multi))


SPEC = SimpleSource(
    facility="Palomar",
    instrument="ZTF",
    filters={
        "g": Table(f"{REPO}/filter_transmission/ZTF_g_band_test_data_pos1_0AOI.csv", wl_scale=10.0, tr_scale=0.01, note="vendor, on-axis"),
        "r": Table(f"{REPO}/filter_transmission/ZTF_r_band_test_data_pos1_0AOI.csv", wl_scale=10.0, tr_scale=0.01, note="vendor, on-axis"),
        "i": Table(f"{REPO}/filter_transmission/ztf_i_band.csv", wl_scale=10.0, tr_scale=0.01, note="vendor"),
    },
    qe=None,  # written by build(): one curve per coating
    svo_ids={"g": "Palomar/ZTF.g", "r": "Palomar/ZTF.r", "i": "Palomar/ZTF.i"},
    outside_range="zero",  # the QE tables end at ~1% on the way to the silicon cutoff
    references={"repository": "https://github.com/ZwickyTransientFacility/ztf_information"},
    notes=("QE: the default is the multi-layer AR-coated e2v CCD QE (the inner 8 of 16 CCDs), which SVO's "
           "published g, r and i curves use; the single-layer QE (outer 8 CCDs) is variant 'single_layer'. "
           "Variant 'mean' (qe.dat) is a SED_Tools-derived equal-weight mean of the two, not a team product. "
           "No optics throughput is published."),
)

QE_VARIANTS = {
    # name: (file stem, source, description)
    "multi_layer": ("qe_multi", MULTI,
                    "e2v CCD QE, multi-layer AR coating (the inner 8 of ZTF's 16 CCDs); used by SVO's "
                    "published Palomar/ZTF g, r and i"),
    "single_layer": ("qe_single", SINGLE,
                     "e2v CCD QE, single-layer AR coating (the northernmost and southernmost 8 CCDs)"),
}


def build(filter_root, *, session=None, **kwargs):
    if session is None:
        import requests

        session = requests.Session()
    curves = {name: Table(url, wl_scale=10.0).read(session) for name, (_, url, _) in QE_VARIANTS.items()}
    mean = _mean_qe(curves["single_layer"], curves["multi_layer"])
    inst_dir = build_simple(SPEC, filter_root, session=session, **kwargs)
    manifest = InstrumentManifest.load(inst_dir)
    manifest.variants.pop(QE, None)
    for name, (stem, url, description) in QE_VARIANTS.items():
        rel = add_component(inst_dir, QE, *curves[name], name=stem, source=f"{url} ({description})",
                            manifest=manifest if name == "multi_layer" else None)
        register_variant(manifest, QE, name, {"*": rel}, description=description, source=url,
                         **({"default": True} if name == "multi_layer" else {}))
    rel = add_component(inst_dir, QE, *mean, name="qe",
                        source=f"{SINGLE} + {MULTI} (SED_Tools-derived equal-weight mean; not a team product)")
    register_variant(manifest, QE, "mean", {"*": rel}, derived=True, source=f"{SINGLE} + {MULTI}",
                     description=("equal-weight mean of the single- and multi-layer QE, derived by SED_Tools "
                                  "(8+8 CCDs); no team document gives a camera-average QE"))
    manifest.sources["coatings"] = f"{README}: {COATINGS}"
    manifest.save(inst_dir)
    return inst_dir
