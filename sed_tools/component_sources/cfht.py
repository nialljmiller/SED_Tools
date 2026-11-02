"""CFHT MegaPrime/MegaCam components from CFHT's own MegaPrime pages.

CFHT's MegaPrime technical page
(https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/specsinformation.html)
links every factor of the imager's throughput:

* filter scans, one file per physical filter, in CFHT's filter database
  (``Instruments/Filters/curves/cfhNNNN.dat``; nm, percent).  The 2014-2015
  filters (u, g, r, i, z, gri and the full-mosaic narrow bands) have 17
  scans at different positions, one per column; CFHT reports their mean.
  The 2002 SAGEM set (uS gS rS iS zS), the 2007 replacement i (iS2) and the
  CFH12K narrow bands have one averaged scan each;
* the primary-mirror reflectivity (a freshly coated bare aluminium model;
  ``CFHT_Primary_Transmission.txt``, nm, percent).  MegaPrime sits at the
  prime focus, so light meets this mirror once;
* the wide-field corrector plus camera window
  (``CFHT_MegaPrime_Transmission.txt``, nm, percent);
* the CCD quantum efficiency: a 14-point mean of measurements of the 40
  E2V CCDs (``MegaCam_QE_data.txt``, 350-1000 nm) and a 10 nm simulation
  "for a given set of CCD characteristics that match best the table of real
  measurements" (``E2V_CCD42-90_QEmodel.txt``, 280-1050 nm).

The two filter generations are kept apart: each published band gets the
scan of its own physical filter (``BANDS``), never another generation's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from ..filters.compose import FILTER, OPTICS, QE, SHARED
from ._common import fresh_manifest, published_bands, register, with_earlier_note, write_unregistered
from ._original import Fetcher, clean_curve, common_product

logger = logging.getLogger(__name__)

FACILITY = "CFHT"
PAGE = "https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/specsinformation.html"
FILTER_PAGE = "https://www.cfht.hawaii.edu/Instruments/Filters/megaprime.html"
DATA = "https://www.cfht.hawaii.edu/Instruments/Imaging/MegaPrime/data.MegaPrime/"
CURVE = "https://www.cfht.hawaii.edu/Instruments/Filters/curves/cfh{code}.dat"
PRIMARY = DATA + "CFHT_Primary_Transmission.txt"
CORRECTOR = DATA + "CFHT_MegaPrime_Transmission.txt"
QE_MODEL = DATA + "E2V_CCD42-90_QEmodel.txt"
QE_MEASURED = DATA + "MegaCam_QE_data.txt"
CADC_PAGE = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html"

#: CFHT filter number -> (generation, CFHT filter-database label and receipt date)
FILTERS: Dict[str, Tuple[str, str]] = {
    "9301": ("2002 SAGEM set", "U' 9301, 8/1/2002"),
    "9401": ("2002 SAGEM set", "G' 9401, 8/1/2002"),
    "9601": ("2002 SAGEM set", "R' 9601, 8/1/2002"),
    "9701": ("2002 SAGEM set", "I' 9701, 8/1/2002 (broken in 2007)"),
    "9702": ("2007 replacement i", "I' #2 9702, 10/10/2007"),
    "9801": ("2002 SAGEM set", "Z' 9801, 8/1/2002"),
    "9302": ("2014-2015 set", "u' 9302 (Asahi), 8/18/2014"),
    "9402": ("2014-2015 set", "g' 9402, 10/21/2014"),
    "9602": ("2014-2015 set", "r' 9602, 9/8/2014"),
    "9703": ("2014-2015 set", "i' 9703, 9/8/2014"),
    "9901": ("2014-2015 set", "z' 9901, 10/23/2014"),
    "9605": ("2014-2015 set", "gri 9605, 6/9/2015"),
    "9603": ("2014-2015 set", "Halpha 9603, 6/20/2014"),
    "9604": ("2014-2015 set", "Halpha Off 9604, 7/14/2014"),
    "9501": ("2014-2015 set", "OIII 9501, 9/29/2014"),
    "9502": ("2014-2015 set", "OIII Off 9502, 10/2/2014"),
    "9303": ("2014-2015 set", "CaHK 9303, 11/05/2014"),
    "7604": ("CFH12K narrow band", "Halpha off 7604, 9/1/1998"),
    "7605": ("CFH12K narrow band", "H-alpha on 7605, 4/28/1999"),
    "7701": ("CFH12K narrow band", "TiO 7701, 9/1/1998"),
    "7803": ("CFH12K narrow band", "CN 7803, 4/1/2000"),
    "7902": ("CFH12K narrow band", "NB920 7902, 7/28/2003"),
}

#: Published band -> CFHT filter number.  Each mapping was checked against the
#: published curve: the SVO MegaPrime totals equal this scan x primary x
#: corrector x model QE to <= 0.4 % of the peak (u: 3.5 % on its steep blue
#: edge); the MegaCam (CADC) curves follow the same scans most closely.
BANDS: Dict[str, Dict[str, str]] = {
    "Megaprime": {
        "u": "9302", "g": "9402", "r": "9602", "i": "9703", "z": "9901", "gri": "9605",
        "Halpha": "9603", "Halpha_off": "9604", "OIII": "9501", "OIII_off": "9502", "CaHK": "9303",
        "uS": "9301", "gS": "9401", "rS": "9601", "iS": "9701", "iS2": "9702", "zS": "9801",
        "Halpha_on": "7605", "Halpha_off_old": "7604", "TiO": "7701", "CN": "7803", "NB920": "7902",
    },
    "MegaCam": {
        "u": "9302", "g": "9402", "r": "9602", "i": "9703", "z": "9901", "gri": "9605",
        "u_1": "9301", "g_1": "9401", "r_1": "9601", "i_1": "9701", "z_1": "9801",
        "i_2": "9702",
    },
}

NOTES = {
    "Megaprime": (
        "Components from CFHT's MegaPrime technical page ({page}). Filter: the CFHT filter-database "
        "scan of each band's own physical filter ({db}); 2014-2015 filters are the mean of their 17 "
        "position scans, older filters a single averaged scan; generations are never mixed (band -> "
        "filter: {mapping}). Optics: primary-mirror reflectivity (bare Al model, one reflection at the "
        "prime focus) x wide-field corrector and camera window. QE: CFHT's simulated E2V CCD42-90 QE "
        "(10 nm, 280-1050 nm), the curve that reproduces SVO's totals; CFHT's 14-point measured mean "
        "(350-1000 nm, MegaCam_QE_data.txt) is kept unregistered as components/qe/measured_mean.dat. "
        "Measured/model is 0.84 at 350 nm, 1.01-1.02 at 450-650 nm and 0.86-0.92 at 750-1000 nm; the "
        "measured table does not cover u, uS, z or zS, and where it covers it would change synthetic "
        "magnitudes by <=0.008 mag (gri 0.018 mag). Outside a "
        "factor's tabulated range it is zero: SVO's zS total ends at 10520 A, where the model QE ends, "
        "which a held edge value would not reproduce. The published SVO totals (filter+optics+qe) "
        "equal filter x optics x qe here to <=0.4% of the peak (u: 3.5% on the steep corrector edge "
        "below 3300 A), up to a constant: SVO's 2014-2015 _fil curves and totals are 17/18 of the "
        "17-scan mean, CaHK_fil and the cfhNNNN bands are the first scan column only, and z_fil is "
        "the old Z' 9801 scan (identical to zS_fil), not the z' 9901 filter of the z band; the z "
        "band's filter here is 9901. No CFHT site atmosphere is published, so none is offered."
    ),
    "MegaCam": (
        "Components from CFHT's MegaPrime technical page ({page}): the CFHT filter-database scan of "
        "each band's own physical filter (band -> filter: {mapping}; 2014-2015 filters are the mean of "
        "17 position scans), primary mirror (one reflection) x corrector and window, and CFHT's "
        "simulated E2V CCD42-90 QE (the measured 14-point mean is kept unregistered as "
        "components/qe/measured_mean.dat). The published curves are CADC MegaPipe's ({cadc}), which "
        "take the mirror, optics and CCD curves (and the first- and second-generation filters) from "
        "Betoule et al. (2013) and include 1.25 airmasses of atmosphere, so filter x optics x qe from "
        "CFHT's own files is not expected to reproduce them. Outside a factor's tabulated range it is "
        "zero. No CFHT site atmosphere is published, so none is offered."
    ),
}


def _numeric_rows(text: str):
    rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError("no numeric rows")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("rows have different numbers of columns")
    return np.array(rows)


def scan(text: str, url: str):
    """A filter-database scan as (Angstrom, fraction) and a cleaning note.

    Two-column files are one averaged scan; wider files hold one scan per
    column and are averaged (CFHT reports the mean).  Values are percent.
    """
    data = _numeric_rows(text)  # rows of at least two numbers
    mean = data[:, 1:].mean(axis=1)
    if mean.max() < 2.0 or mean.max() > 101.0:
        raise ValueError(f"{url}: peak {mean.max():.3g} is not a percentage")
    wl, tr, note = clean_curve(data[:, 0] * 10.0, mean / 100.0)
    ncol = data.shape[1] - 1
    return wl, tr, (f"mean of {ncol} scans; " if ncol > 1 else "") + (note or "")


def percent_table(text: str, url: str, cols=(0, 1)):
    data = _numeric_rows(text)
    wl, tr, _ = clean_curve(data[:, cols[0]] * 10.0, data[:, cols[1]] / 100.0)
    if tr.max() > 1.0:
        raise ValueError(f"{url}: exceeds 100%")
    return wl, tr


def build(filter_root: Union[str, Path], *, instrument: str = "Megaprime", session=None,
          **_ignored) -> Path:
    if instrument not in BANDS:
        raise ValueError(f"No CFHT component recipe for {instrument!r}; available: {sorted(BANDS)}")
    inst_dir = Path(filter_root) / FACILITY / instrument
    present = published_bands(inst_dir)
    mapping = {b: c for b, c in BANDS[instrument].items() if b in present}
    missing = sorted(set(BANDS[instrument]) - present)
    if missing:
        logger.warning("CFHT/%s: no published curve for %s; their filters are skipped", instrument, missing)

    # Fetch and check everything before writing anything.
    fetch = Fetcher(session)
    filters = {code: scan(fetch.text(CURVE.format(code=code), "latin-1"), CURVE.format(code=code))
               for code in sorted(set(mapping.values()))}
    primary, corrector, qe_model, qe_measured = (percent_table(fetch.text(url), url)
                                                 for url in (PRIMARY, CORRECTOR, QE_MODEL, QE_MEASURED))

    text = ", ".join(f"{b}={c}" for b, c in sorted(mapping.items()))
    notes = with_earlier_note(
        inst_dir, NOTES[instrument].format(page=PAGE, db=FILTER_PAGE, mapping=text, cadc=CADC_PAGE))
    manifest = fresh_manifest(inst_dir, FACILITY, instrument)
    for band, code in sorted(mapping.items()):
        wl, tr, note = filters[code]
        generation, label = FILTERS[code]
        register(manifest, inst_dir, FILTER, wl, tr, bands=[band], name=band,
                 source=f"{CURVE.format(code=code)} (CFHT {label}; {generation}"
                        + (f"; {note.strip('; ')})" if note.strip("; ") else ")"))
    wl, tr = common_product([primary, corrector])
    register(manifest, inst_dir, OPTICS, wl, tr, bands=[SHARED], name="primary_corrector",
             source=f"product of {PRIMARY} (primary mirror, one reflection) and {CORRECTOR} "
                    "(wide-field corrector and camera window), nm and percent")
    register(manifest, inst_dir, QE, *qe_model, bands=[SHARED], name="e2v_ccd42-90_model",
             source=f"{QE_MODEL} (CFHT's simulated E2V CCD42-90 QE fitted to the measured mean; percent)")
    write_unregistered(inst_dir, QE, *qe_measured, name="measured_mean",
                       source=f"{QE_MEASURED} (mean measured QE of the 40 CCDs at CEA; percent; "
                              "not registered: covers only 350-1000 nm)")

    manifest.notes = notes
    manifest.ground_based = True
    manifest.default_site = None
    manifest.outside_range = "zero"
    manifest.sources.update({
        FILTER: f"CFHT filter-database scans ({FILTER_PAGE}), one per band: {text}",
        "page": PAGE,
        "filter_database": FILTER_PAGE,
        "qe_measured_unregistered": "components/qe/measured_mean.dat <- " + QE_MEASURED,
    })
    manifest.save(inst_dir)
    return inst_dir
