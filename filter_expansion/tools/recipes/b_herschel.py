"""Herschel PACS 70/100/160 filter transmissions and SPIRE 250/350/500 RSRFs (HSA legacy ADP FITS)."""
import numpy as np
from astropy.io import fits
from fxlib import ORIGINAL, build_instrument, fetch, make_band

PACS = "http://archives.esac.esa.int/hsa/legacy/ADP/PACS/PACS-P_filter_curves/PACS_phot_filterTrans_{}.fits.gz"
SPIRE = "http://archives.esac.esa.int/hsa/legacy/ADP/SPIRE/SPIRE-P_filter_curves/SPIRE_phot_RSRF_14_3.fits"
INFO = "https://www.herschel.caltech.edu/page/instruments"


def run():
    paths, bands = [], []
    for name, col, key in (("70", "blue", "BS"), ("100", "green", "BL"), ("160", "red", "R")):
        url = PACS.format(col)
        p = fetch(url, "Herschel")
        paths.append(p)
        with fits.open(p) as h:
            t = h[1]
            assert h[1].header.get("BAND") == key, (name, h[1].header.get("BAND"))
            assert t.columns["wavelength"].unit == "um"
            wl = np.asarray(t.data["wavelength"], float)
            tr = np.asarray(t.data["transmission"], float)
        bands.append(make_band(name, wl, tr, unit="micron", detector_type="energy", detector_source=ORIGINAL,
                               components="filter", components_source=ORIGINAL, sources=f"{url} | {INFO}",
                               svo_id=f"Herschel/Pacs.{col}"))
    build_instrument("Herschel", "PACS", bands, raw_files=paths, notes=(
        "PACS photometer filter transmission curves (HSA legacy ADP, from PacsCal_FM_FilterTransmission; "
        "FITS BAND keyword BS/BL/R = 70/100/160 um). Filter only: telescope, optics and bolometer "
        "absorption not included. DetectorType=energy because PACS photometry uses bolometer arrays "
        "(power detectors). Native transmission scale kept."))

    p = fetch(SPIRE, "Herschel")
    with fits.open(p) as h:
        t = h["rsrf"]
        assert t.columns["frequency"].unit == "GHz"
        nu = np.asarray(t.data["frequency"], float)
        cols = {k: np.asarray(t.data[k], float) for k in ("psw", "pmw", "plw")}
    bands = [make_band(name, nu, cols[k], unit="GHz", detector_type="energy", detector_source=ORIGINAL,
                       components="filter+optics+qe", components_source=ORIGINAL, sources=f"{SPIRE} | {INFO}",
                       normalize=True, svo_id=f"Herschel/SPIRE.{k.upper()}")
             for name, k in (("250", "psw"), ("350", "pmw"), ("500", "plw"))]
    build_instrument("Herschel", "SPIRE", bands, raw_files=[p], notes=(
        "SPIRE photometer Relative Spectral Response Functions (SCalPhotRsrf v3, FM; HSA legacy ADP "
        "SPIRE_phot_RSRF_14_3.fits, ext 'rsrf'), PSW/PMW/PLW = 250/350/500 um. Frequency grid in GHz "
        "converted to Angstrom and sorted. Instrument RSRF (filters, feedhorn/optics, bolometer "
        "absorption); telescope and aperture efficiency not included. DetectorType=energy (bolometers). "
        "Relative response: tiny negative samples set to 0, then peak-normalised (normalize=True)."))
