"""2MASS components from the All-Sky Explanatory Supplement, section III.1.b1.

"Below are laboratory measurements of the various transmission, reflection,
and response data needed to calculate the absolute response of the 2MASS
cameras" (https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec3_1b1.html).
Each table is an HTML ``<pre>`` block with columns ``n, xlam, yval``; xlam
is not labelled and is read as micrometres (the J filter table spans
1.05-1.45).  The light path the page describes:

* two reflections off the telescope mirrors (tbl1 is "a single surface");
* the ZnSe dewar window (tbl2) and seven anti-reflection coated lenses
  (tbl3 is "a single ... lens"; one common, six per band);
* J reflects off the J dichroic (tbl4); H passes it (tbl6) and reflects off
  the H dichroic (tbl7); Ks passes both (tbl6, tbl9);
* the band filters: J tbl5, H tbl8, Ks tbl10;
* the NICMOS3 detector QE (tbl11; "characteristic" Rockwell curves, not
  measured per device).

``filter`` is the band filter, ``optics`` the band's own mirror, window,
lens and dichroic path, ``qe`` the NICMOS3 curve.  Their product reproduces
the supplement's total instrumental throughput (tbl12-14) with each factor
zero outside its table (to <=0.0003 mag in H and Ks; to <=0.007 mag in J,
whose tbl12 starts at 1.095 um while the J-dichroic table starts at 1.105).

The published curves are Cohen et al. (2003) relative spectral responses:
QE-based throughput times the PLEXUS atmosphere, multiplied by wavelength
and peak-normalised "to be integrated directly over stellar spectra in
F_lambda form" (energy convention in SED_Tools).  ``filter+optics+qe`` here
is a photon-counting throughput (no atmosphere, no wavelength factor).  The
PLEXUS atmospheres (tbl15/16) are not added here.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..filters.compose import FILTER, OPTICS, QE, SHARED
from ._common import fresh_manifest, published_bands, register, with_earlier_note
from ._original import Fetcher, clean_curve, common_product

logger = logging.getLogger(__name__)

FACILITY, INSTRUMENT = "2MASS", "2MASS"
DOC = "https://www.ipac.caltech.edu/2mass/releases/allsky/doc/"
PAGE = DOC + "sec3_1b1.html"
RSR_PAGE = DOC + "sec6_4a.html"
TABLE = DOC + "sec3_1b1.tbl{n}.html"
TITLES = {
    1: "2MASS MAIN MIRROR COATING REFLECTIVITY",
    2: "2MASS DEWAR WINDOW TRANSMISSION",
    3: "2MASS LENS ELEMENT COATING",
    4: "2MASS J-BAND DICHROIC REFLECTION",
    5: "2MASS J-BAND FILTER TRANSMISSION",
    6: "2MASS J-BAND DICHROIC TRANSMISSION",
    7: "2MASS H-BAND DICHROIC REFLECTION",
    8: "2MASS H-BAND FILTER TRANSMISSION",
    9: "2MASS H-BAND DICHROIC TRANSMISSION",
    10: "-BAND FILTER TRANSMISSION",  # "2MASS K<sub>s</sub>-BAND ..."
    11: "2MASS NICMOS3 DETECTOR QUANTUM EFFICIENCY",
}
FILTER_TABLE = {"J": 5, "H": 8, "Ks": 10}
DICHROIC_PATH = {"J": (4,), "H": (6, 7), "Ks": (6, 9)}
MIRROR_REFLECTIONS = 2
LENSES = 7
#: Instrument totals (no atmosphere), used only to validate the factors.
TOTAL_TABLE = {"J": 12, "H": 13, "Ks": 14}

NOTES = (
    "Components from the 2MASS All-Sky Explanatory Supplement III.1.b1 ({page}); wavelengths "
    "(unlabelled xlam) read as micrometres. filter: J tbl5, H tbl8, Ks tbl10. optics (per band): "
    "telescope mirror tbl1 squared (two reflections; the table is one surface) x ZnSe dewar window "
    "tbl2 x lens coating tbl3 to the 7th power (seven lenses per band) x the band's dichroic path "
    "(J: J-dichroic reflection tbl4; H: J-dichroic transmission tbl6 x H-dichroic reflection tbl7; "
    "Ks: tbl6 x H-dichroic transmission tbl9). qe: NICMOS3 tbl11 (a characteristic Rockwell curve, "
    "not measured per device). Every factor is zero outside its table. The supplement's own total "
    "instrumental throughput (tbl12-14) is reproduced to <=0.0003 mag in H and Ks (shape, black "
    "bodies 3000-30000 K); in J to <=0.007 mag, because tbl12 is 0 at 1.080-1.090 um but not at "
    "1.095-1.100 um, before the J-dichroic table starts at 1.105 um (holding the dichroic's edge "
    "value instead gives -0.006 mag). The published curves are Cohen et al. (2003) RSRs "
    "(Explanatory Supplement VI.4a, "
    "{rsr}): that throughput times the PLEXUS atmosphere, multiplied by wavelength and "
    "peak-normalised for direct F_lambda integration, hence DetectorType=energy. filter+optics+qe "
    "here carries no wavelength factor and no atmosphere; SED_Tools integrates it as photon "
    "counting, which applies the wavelength factor once. A filter or filter+optics product inherits "
    "the RSR's 'energy' label in SED_Tools although the 2MASS cameras count photons: build it with "
    "detector_type='photon'. Against the published RSRs (shape, black bodies 3000-30000 K and flat "
    "F_nu), filter x optics x qe x the supplement's PLEXUS atmosphere agrees to <=0.006 mag in J, "
    "<=0.0005 mag in H (0.0000 with the south table) and <=0.0043 mag in Ks; without an atmosphere "
    "J differs by up to 0.065 mag. The PLEXUS north (Mt. Hopkins) and south (CTIO) atmospheres "
    "(tbl15/16) are not part of this instrument's components."
)


def pre_table(html: str, url: str, title: str = ""):
    """``(xlam, yval)`` of a supplement table (columns ``n xlam yval`` or ``xlam yval``)."""
    if title and title not in html:
        raise ValueError(f"{url}: expected title {title!r}")
    m = re.search(r"<pre>(.*?)(?:</pre>|$)", html, re.S | re.I)
    if not m:
        raise ValueError(f"{url}: no <pre> table")
    rows = []
    for line in m.group(1).splitlines():
        parts = line.split()
        if not parts or parts[0][0] in "\\|":
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) == 3:
            rows.append(vals[1:])
        elif len(vals) == 2:
            rows.append(vals)
    if len(rows) < 2:
        raise ValueError(f"{url}: no numeric rows")
    return np.array(rows)


def micron_curve(html: str, url: str, title: str = ""):
    data = pre_table(html, url, title)
    wl, tr, _ = clean_curve(data[:, 0] * 1e4, data[:, 1])
    if tr.max() > 1.0:
        raise ValueError(f"{url}: exceeds 1")
    return wl, tr


def fetch_tables(fetch: Fetcher, numbers) -> Dict[int, tuple]:
    return {n: micron_curve(fetch.text(TABLE.format(n=n), "latin-1", allow_html=True),
                            TABLE.format(n=n), TITLES.get(n, "")) for n in numbers}


def band_optics(t: Dict[int, tuple], band: str):
    mirror = (t[1][0], t[1][1] ** MIRROR_REFLECTIONS)
    lenses = (t[3][0], t[3][1] ** LENSES)
    return common_product([mirror, t[2], lenses] + [t[n] for n in DICHROIC_PATH[band]])


def build(filter_root: Union[str, Path], *, session=None, **_ignored) -> Path:
    inst_dir = Path(filter_root) / FACILITY / INSTRUMENT
    present = published_bands(inst_dir)
    missing = sorted(set(FILTER_TABLE) - present)
    if missing:
        logger.warning("2MASS: no published curve for %s; their filter and optics are skipped", missing)
    t = fetch_tables(Fetcher(session), range(1, 12))
    notes = with_earlier_note(inst_dir, NOTES.format(page=PAGE, rsr=RSR_PAGE))
    manifest = fresh_manifest(inst_dir, FACILITY, INSTRUMENT)
    for band in sorted(set(FILTER_TABLE) & present):
        n = FILTER_TABLE[band]
        register(manifest, inst_dir, FILTER, *t[n], bands=[band], name=band,
                 source=f"{TABLE.format(n=n)} ({band} filter; xlam read as um)")
        path = "; ".join(f"tbl{n}" for n in DICHROIC_PATH[band])
        register(manifest, inst_dir, OPTICS, *band_optics(t, band), bands=[band], name=band,
                 source=(f"{PAGE}: tbl1^2 (mirror, two reflections) x tbl2 (dewar window) x tbl3^7 "
                         f"(lens coating, seven lenses) x {path} (dichroic path)"))
    register(manifest, inst_dir, QE, *t[11], bands=[SHARED], name="nicmos3",
             source=f"{TABLE.format(n=11)} (NICMOS3 DQE, characteristic Rockwell curve)")
    manifest.notes = notes
    manifest.ground_based = True
    manifest.default_site = None
    manifest.outside_range = "zero"
    manifest.sources.update({FILTER: f"{PAGE}: J tbl5, H tbl8, Ks tbl10",
                             OPTICS: f"{PAGE}: tbl1^2 x tbl2 x tbl3^7 x the band's dichroic path",
                             "page": PAGE, "rsr": RSR_PAGE})
    manifest.save(inst_dir)
    return inst_dir


def validate(instrument_dir: Union[str, Path], session=None) -> Dict[str, Dict[str, float]]:
    """filter x optics x qe against the supplement's instrument totals (tbl12-14).

    Returns per band the photon-weighted magnitude difference for a flat
    F_nu source and the largest absolute difference on the table grid.
    """
    from ..filters.compose import compose_band

    fetch = Fetcher(session)
    out = {}
    for band, n in TOTAL_TABLE.items():
        url = TABLE.format(n=n)
        ref = pre_table(fetch.text(url, "latin-1", allow_html=True), url)
        wl, tot = ref[:, 0] * 1e4, ref[:, 1]
        mine = compose_band(instrument_dir, band, (FILTER, OPTICS, QE))
        ours = np.interp(wl, mine.wavelength, mine.transmission, left=0.0, right=0.0)
        ratio = np.trapezoid(ours / wl, wl) / np.trapezoid(tot / wl, wl)
        out[band] = {"dmag": float(-2.5 * np.log10(ratio)),
                     "maxdiff": float(np.max(np.abs(ours - tot)))}
    return out
