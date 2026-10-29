"""Gemini GMOS-N and GMOS-S detector QE by detector epoch.

The published GMOS curves (SVO ``Gemini/GMOS-N.*`` and ``GMOS-S.*``) are
filter-only lab scans, so none of them includes a detector QE; each is its
own filter component.  The QE depends on which detector array was in use
(Gemini GMOS "Components" page, Detector Array section):

GMOS-N
    ``eev``           original EEV array, replaced in October 2011
                      (``gmos_n_ccd_red.txt``, average of the three CCDs)
    ``e2v_dd``        e2v 42-90 deep-depletion CCDs, installed mid-October
                      2011 (first science 2011-11-22) until February 2017.
                      The page gives no data ("not yet available"); the
                      Gemini ITC file was read off a plot in the vendor's
                      proposal (Aug 2011).
    ``hamamatsu_bb``  Hamamatsu array from February/March 2017: the ITC uses
    ``hamamatsu_hsc`` "BB" for CCDr and CCDb and "HSC" for CCDg; the page
                      links no GMOS-N data, so these are the ITC files.
GMOS-S
    ``eev``           original EEV array, removed May 2014
                      (``gmos_s_ccd_blue.txt``, mean of the three CCDs)
    ``hamamatsu_red`` Hamamatsu array from June 2014: the page's "Hamamatsu
    ``hamamatsu_blue`` Red data" (CCDr, CCDg) and "Hamamatsu Blue data"
                      (CCDb) files.

A Hamamatsu array mixes CCD types, so no QE is the default: select one
explicitly (``variants={"qe": "eev"}``).  The Gemini ITC differs from the
page for GMOS-S (it uses the red EEV curve for the old array and BB/HSC/SC
for the three Hamamatsu CCDs); the page is followed where it gives data.
"""

from __future__ import annotations

from pathlib import Path

from ..filter_components import QE, InstrumentManifest, add_component
from ._attach import published_bands
from ._common import Table, register_variant

PAGE = "https://www.gemini.edu/instrumentation/gmos/components"
SCIOPS = "https://www.gemini.edu/sciops/instruments/gmos"
ITC_SHA = "0bc1607406428405eae962ccdf0bc681ff7b6c5a"
ITC = (f"https://raw.githubusercontent.com/gemini-hlsw/ocs/{ITC_SHA}"
       "/bundle/edu.gemini.itc/src/main/resources/gmos")
ITC_CODE = (f"https://github.com/gemini-hlsw/ocs/blob/{ITC_SHA}"
            "/bundle/edu.gemini.itc/src/main/java/edu/gemini/itc/gmos/")

#: instrument -> variant -> (table, epoch, description)
QE_VARIANTS = {
    "GMOS_N": {
        "eev": (Table(f"{SCIOPS}/gmos_n_ccd_red.txt", wl_scale=10.0, cols=(0, 4)),
                "original GMOS-N array; replaced by e2v deep-depletion CCDs in October 2011",
                "EEV CCDs delivered with GMOS-N: average QE of the three CCDs (NOAO data, Sept 2000; "
                "I. Jorgensen, 2001)"),
        "e2v_dd": (Table(f"{ITC}/gmos_n_E2V4290DDmulti3.dat", wl_scale=10.0),
                   "installed mid-October 2011 (first science data 2011-11-22) until February 2017",
                   "e2v 42-90 deep-depletion CCDs (multi-layer coating; 'multi-layer 3' on the Gemini page, "
                   "'multi-layer 2' in the file header): Gemini ITC file, values read off a plot in the vendor "
                   "proposal (S. Kleinman, Aug 2011); the Gemini page gives no data"),
        "hamamatsu_bb": (Table(f"{ITC}/gmos_n_ccd_hamamatsu_bb.dat", wl_scale=10.0),
                         "from February/March 2017",
                         "Hamamatsu 'Better Blue' (BB) CCD QE, Gemini ITC file dated 2013 Oct 1; the ITC uses it "
                         "for GMOS-N CCDr and CCDb (GmosNorth.java)"),
        "hamamatsu_hsc": (Table(f"{ITC}/gmos_n_ccd_hamamatsu_hsc.dat", wl_scale=10.0),
                          "from February/March 2017",
                          "Hamamatsu Hyper Suprime-Cam (HSC) type CCD QE, Gemini ITC file dated 2013 Oct 1; the "
                          "ITC uses it for GMOS-N CCDg, the central CCD (GmosNorth.java)"),
    },
    "GMOS_S": {
        "eev": (Table(f"{SCIOPS}/gmos_s_ccd_blue.txt", wl_scale=10.0, cols=(0, 4)),
                "original GMOS-S array; removed May 2014",
                "blue EEV CCDs delivered with GMOS-S: mean QE of the three CCDs (NOAO data, 2000; "
                "R. Carrasco, June 2003)"),
        "hamamatsu_red": (Table(f"{SCIOPS}/gmos_n_ccd_hamamatsu_sc.txt", wl_scale=10.0),
                          "from June 2014",
                          "Gemini page 'Hamamatsu Red data' for CCDr and CCDg: Hamamatsu S10892-01 delivery "
                          "specification (August 2009), QE at -100 C"),
        "hamamatsu_blue": (Table(f"{SCIOPS}/gmos_n_ccd_hamamatsu_hsc.txt", wl_scale=10.0),
                           "from June 2014",
                           "Gemini page 'Hamamatsu Blue data' for CCDb: HSC-type (visible enhanced) Hamamatsu "
                           "CCD QE at -100 C, estimated by eye from Kamata et al. 2010, SPIE 7742, 774229"),
    },
}

NOTES = {
    "GMOS_N": ("GMOS-N published curves are filter-only scans, so none includes a detector QE. Detector QE by "
               "epoch is in the qe variants: eev (original array, to October 2011), e2v_dd (October 2011 to "
               "February 2017), hamamatsu_bb and hamamatsu_hsc (from February/March 2017; BB for CCDr/CCDb and "
               "HSC for CCDg in the Gemini ITC). No QE is the default; select one. The EEV table ends at "
               "1000 nm and the others at 1100 nm, so no QE variant covers Y (and EEV not z); compose those "
               "bands only with allow_partial."),
    "GMOS_S": ("GMOS-S published curves are filter-only scans, so none includes a detector QE. Detector QE by "
               "epoch is in the qe variants: eev (original array, removed May 2014), hamamatsu_red (CCDr, CCDg) "
               "and hamamatsu_blue (CCDb) from June 2014, as the Gemini page labels them. The Gemini ITC instead "
               "uses the red EEV curve for the old array and BB/HSC/SC for the three Hamamatsu CCDs. No QE is "
               "the default; select one. The EEV table ends at 1000 nm and the Hamamatsu Blue table covers "
               "340-1060 nm, so those variants do not cover Y (Blue also not u); compose those bands only with "
               "allow_partial."),
}


def build(filter_root, *, instrument, session=None, **_ignored) -> Path:
    if instrument not in QE_VARIANTS:
        raise ValueError(f"No GMOS builder for {instrument!r}; available: {sorted(QE_VARIANTS)}")
    if session is None:
        import requests

        session = requests.Session()
    inst_dir = Path(filter_root) / "Gemini" / instrument
    published_bands(inst_dir)  # refuses an instrument with no published curves
    curves = {name: table.read(session) for name, (table, _, _) in QE_VARIANTS[instrument].items()}

    manifest = InstrumentManifest.from_published(inst_dir)  # filter-only curves are their own filter
    manifest.facility, manifest.instrument = "Gemini", instrument
    for name, (table, epoch, description) in QE_VARIANTS[instrument].items():
        rel = add_component(inst_dir, QE, *curves[name], name=f"qe_{name}",
                            source=f"{table.url} ({description})")
        register_variant(manifest, QE, name, {"*": rel}, epoch=epoch, description=description,
                         source=table.url)
    manifest.outside_range = "error"
    manifest.sources.update({"gemini": PAGE, "itc": ITC_CODE})
    manifest.notes = NOTES[instrument]
    manifest.save(inst_dir)
    return inst_dir
