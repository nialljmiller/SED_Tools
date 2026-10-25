"""Planck HFI 100-857 and LFI 30/44/70 band-averaged bandpasses (IRSA PR2 RIMO FITS)."""
import numpy as np
from astropy.io import fits
from fxlib import ORIGINAL, build_instrument, fetch, make_band

BASE = "https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/"
HFI = BASE + "HFI_RIMO_R2.00.fits"
LFI = BASE + "LFI_RIMO_R2.50.fits"
HFI_EV = "https://arxiv.org/abs/1101.2039"
LFI_EV = "https://arxiv.org/abs/1001.3321"
# LFI 30 GHz spans 21-40 GHz = 7.5e7-1.4e8 A; fx_validate rejects wavelengths > 1e8 A, so the band is
# held until that limit is relaxed (then set True and rebuild). Trimming would cut the band core.
INCLUDE_LFI30 = True


def read(path, ext, unit):
    with fits.open(path) as h:
        t = h[ext]
        assert t.columns[0].name == "WAVENUMBER" and t.columns[0].unit == unit, (ext, t.columns[0].unit)
        return np.asarray(t.data.field(0), float), np.asarray(t.data.field(1), float)


def run():
    p = fetch(HFI, "Planck")  # local copy of round3/space/raw/PLANCK (IRSA PR2)
    bands = []
    for f in (100, 143, 217, 353, 545, 857):
        x, t = read(p, f"BANDPASS_F{f}", "1/cm")
        keep = x >= 1.0  # drop the far tail beyond 1 cm (< 1e-7 of the integral; validator limit 1e8 A)
        x, t = x[keep], t[keep]
        bands.append(make_band(str(f), x, t, unit="1/cm", detector_type="energy", detector_source=ORIGINAL,
                               components="filter+optics+qe", components_source=ORIGINAL,
                               sources=f"{HFI} | {HFI_EV}", normalize=True, svo_id=f"Planck/HFI.{f}"))
    build_instrument("Planck", "HFI", bands, raw_files=[p], notes=(
        "Planck HFI RIMO R2.00 (PR2) band-averaged spectral transmission, extensions BANDPASS_F100..F857 "
        "('weighted combination of the individual detector bandpasses'). Wavenumber in cm^-1 converted "
        "to Angstrom and sorted; the 60 samples below 1 cm^-1 (lambda > 1 cm, incl. 0 cm^-1; max transmission <= 4e-7, <= 5e-8 of the band integral) dropped; FLAG column (marks the "
        "in-band region) ignored, all samples kept. Ground-measured instrument response (feedhorns, "
        "filters, bolometers), telescope not included. DetectorType=energy (bolometers; band-averaged "
        "fluxes integrate tau(nu) I_nu dnu). Peak-normalised (already 1)."))

    p = fetch(LFI, "Planck")
    bands = []
    for f in (30, 44, 70) if INCLUDE_LFI30 else (44, 70):
        x, t = read(p, f"BANDPASS_{f:03d}", "GHz")
        bands.append(make_band(str(f), x, t, unit="GHz", detector_type="energy", detector_source=ORIGINAL,
                               components="filter+optics+qe", components_source=ORIGINAL,
                               sources=f"{LFI} | {LFI_EV}", normalize=True, svo_id=f"Planck/LFI.{f}"))
    build_instrument("Planck", "LFI", bands, raw_files=[p], notes=(
        "Planck LFI RIMO R2.50 (PR2) band-averaged bandpasses, extensions BANDPASS_044/070 (030 held: "
        "its 7.5e7-1.4e8 A range exceeds the validator's 1e8 A plausibility limit) (the "
        "per-radiometer BANDPASS_0xx-NNS/M curves are not used). Column named WAVENUMBER is in GHz "
        "(TUNIT), converted to Angstrom and sorted. Modelled radiometer-chain response (feedhorn, OMT, "
        "HEMT amplifiers, filters, detector diode), telescope not included. DetectorType=energy (coherent "
        "radiometers measure power; band integrals of tau(nu) I_nu dnu). Peak-normalised (normalize=True)."))
