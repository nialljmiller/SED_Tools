"""OGLE-IV V and I (Las Campanas 1.3 m Warsaw telescope, 32-chip e2v CCD44-82 mosaic).

Sources (J. Skowron's OGLE-IV directory, codex.csv):
  OGLE-IV-{V,I}-filter-transmission.dat  Asahi filter scans, Å, mean of 9 spots (col 2)
  OGLE-IV-QE-curve-from-specification.dat  e2v CCD44-82 Astro Broadband QE from the manual
  OGLE-IV-QE-measured-min-max-curves.dat   measured QE envelope (Å, min, max; no header)
Published curve = mean filter transmission x specification QE (photon counting); the filter
and QE are also published as components.  No telescope optics or atmosphere.
"""
import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, fetch, make_band, pivot_wavelength

BASE = "https://www.astrouw.edu.pl/~jskowron/pub/ogle/OGLE-IV/"
FILT = {b: BASE + f"OGLE-IV-{b}-filter-transmission.dat" for b in ("V", "I")}
QE_SPEC = BASE + "OGLE-IV-QE-curve-from-specification.dat"
QE_MEAS = BASE + "OGLE-IV-QE-measured-min-max-curves.dat"


def _table(path, ncol):
    rows = [ln.split()[:ncol] for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    return np.array(rows, float)


def run():
    fpaths = {b: fetch(u, "OGLE") for b, u in FILT.items()}
    qs_path, qm_path = fetch(QE_SPEC, "OGLE"), fetch(QE_MEAS, "OGLE")
    for b, p in fpaths.items():
        if f"OGLE-IV {b}-band filter" not in p.read_text()[:200] or "wavelength[A]" not in p.read_text()[:2000]:
            raise ValueError(f"{p}: unexpected header")
    qs = _table(qs_path, 2)
    qm = _table(qm_path, 3)
    if not np.all(qm[:, 1] <= qm[:, 2]):
        raise ValueError("measured QE columns are not (min, max)")
    qmid = 0.5 * (qm[:, 1] + qm[:, 2])
    bands, comps, check = [], [Component("qe", qs[:, 0], qs[:, 1], band="*", source=QE_SPEC)], {}
    for b, p in fpaths.items():
        f = _table(p, 2)
        filt = make_band(b, f[:, 0], f[:, 1], unit="Angstrom", detector_type="photon",
                         detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                         sources=FILT[b])
        wl = filt.wavelength
        prod = filt.transmission * np.interp(wl, qs[:, 0], qs[:, 1], left=0.0, right=0.0)
        band = make_band(b, wl, prod, unit="Angstrom", detector_type="photon",
                         detector_source=ORIGINAL, components="filter+qe", components_source=ORIGINAL,
                         sources=f"{FILT[b]} | {QE_SPEC}", svo_id=f"OGLE/OGLE-IV.{b}")
        band.clean_note = "; ".join(x for x in (f"filter: {filt.clean_note}" if filt.clean_note else "",
                                                 band.clean_note) if x)
        alt = filt.transmission * np.interp(wl, qm[:, 0], qmid, left=0.0, right=0.0)
        check[b] = {"pivot_spec_qe_A": round(pivot_wavelength(wl, prod, "photon"), 1),
                    "pivot_measured_mid_qe_A": round(pivot_wavelength(wl, alt, "photon"), 1),
                    "pivot_filter_only_A": round(pivot_wavelength(wl, filt.transmission, "photon"), 1),
                    "uv_noise_mean_2495_3500A": round(float(filt.transmission[wl < 3500].mean()), 4)}
        comps.append(Component("filter", wl, filt.transmission, band=b, source=FILT[b]))
        bands.append(band)
    note = ("OGLE-IV (Warsaw 1.3 m, Las Campanas) V and I. Published curve = Asahi filter "
            "transmission (mean of 9 spots, OGLE-IV-{V,I}-filter-transmission.dat) x e2v CCD44-82 "
            "specification QE (OGLE-IV-QE-curve-from-specification.dat, set to 0 outside "
            "3032-10787 A); photon-counting CCDs. No telescope optics or atmosphere included. "
            "Components: filter (per band) and qe (shared). The measured QE min/max envelope "
            "(OGLE-IV-QE-measured-min-max-curves.dat, columns verified as min<=max) is not used; "
            "the pivot with its midpoint is in provenance. Caveat: the filter scans have a "
            "low-signal noise floor below ~3500 A (quantised spot values, up to 0.1 in V and 0.29 "
            "in I) kept as published apart from negatives set to 0; after the QE (0 below 3032 A) it carries ~0.2 % of the photon-weighted response and shifts the pivot by ~0.2-0.5 % vs zeroing below 3500 A. No published pivot.")
    build_instrument("OGLE", "OGLE-IV", bands, notes=note, components=comps, ground_based=True,
                     raw_files=[*fpaths.values(), qs_path, qm_path],
                     extra_provenance={"qe_choice_check": check})
