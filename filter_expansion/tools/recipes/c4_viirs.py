"""JPSS VIIRS on NOAA-20 (J1) and NOAA-21 (J2): DAWG at-launch band-averaged RSRs from NOAA NCC.

Source pages:
  https://ncc.nesdis.noaa.gov/NOAA-20/J1VIIRS_NOAA20_SpectralResponseFunctions.php
    -> NOAA-20/docs/J1_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2.1_Nov2016.zip ("Final prelaunch version")
  https://ncc.nesdis.noaa.gov/NOAA-21/J2VIIRS_SpectralResponseFunctions.php
    -> NOAA-21/docs/J2_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2_Jul2019.zip ("Final prelaunch version")
Each zip has a README (J1 .docx, J2 .pdf) from the VIIRS Data Analysis Working Group (DAWG). Band average
files ("BA") hold band name, wavelength [nm] and response; "The RSR are stitched IB+OOB RSR with all RSR
normalized at the peak of 1.0"; low-quality (noise-floor) wavelengths are set to 1E-10.  Some V1-derived
headers label the wavelength column "wvl_um" but the values are nm (checked against the band's
specified spectral range below).  Each curve is the whole-band system response (filter+optics+qe and, for
the "Fused" VisNIR bands, crosstalk), averaged over all detectors (and I-band subsamples).

J1: V2.1 M9 (J1_VIIRS_BA_RSR_V2.1F, spectral-position fix) replaces earlier M9; M1-M7, I1, I2, DNBLGS, M13
from V2F; M8, M10-M12, M14-M16(A,B), I3-I5, DNBMGS are the V1 files the README says are "repeated in V2".
J2: all J2_VIIRS_BA_RSR_V2F files.

Suomi NPP (F1) is NOT built: the official band-averaged NPP set on
https://ncc.nesdis.noaa.gov/VIIRS/VIIRSSpectralResponseFunctions.php (NG_VIIRS_NPP_RSR_filtered_Oct2011_BA.zip)
carries "Copyright 2011, Northrop Grumman Systems Corporation. All Rights Reserved." in its README, so it is
HELD; snpp_source_builder() below is the source-builder recipe (users fetch the zip from NOAA themselves).

Naming: SVO-style IDs JPSS/VIIRS-J1.<band>, JPSS/VIIRS-J2.<band> (= NOAA's own J1/J2 flight designations).
Pivot check: NOAA publishes no central wavelengths, only the "Spectral Range (um)" specification column of
NCC's NOAA-20 VIIRS On-orbit Performance table; each band's pivot must lie inside that range (2 % pad).
No ref_pivot is passed (a range midpoint is not a published centre).  DNB is not in the table.
"""
import zipfile
from pathlib import Path

import numpy as np
from fxlib import DOWNLOADS, build_instrument, make_band, pivot_wavelength

NCC = "https://ncc.nesdis.noaa.gov/"
PERF_URL = NCC + "NOAA-20/docs/NOAA-20-VIIRS-on-OrbitPerformanceTable.pdf"
# Spectral Range (um), specification, from PERF_URL.
SPEC_UM = {
    "M1": (0.402, 0.422), "M2": (0.436, 0.454), "M3": (0.478, 0.498), "M4": (0.545, 0.565),
    "I1": (0.600, 0.680), "M5": (0.662, 0.682), "M6": (0.739, 0.754), "I2": (0.846, 0.885),
    "M7": (0.846, 0.885), "M8": (1.230, 1.250), "M9": (1.371, 1.386), "I3": (1.580, 1.640),
    "M10": (1.580, 1.640), "M11": (2.225, 2.275), "I4": (3.550, 3.930), "M12": (3.660, 3.840),
    "M13": (3.973, 4.128), "M14": (8.400, 8.700), "M15": (10.263, 11.263), "I5": (10.500, 12.400),
    "M16": (11.538, 12.488), "M16A": (11.538, 12.488), "M16B": (11.538, 12.488),
}
PIVOTS = {}  # (instrument, band) -> pivot um, for the report
# DNB rough window [nm] only for the sanity check of the peak position.
DNB_NM = (450.0, 950.0)

SETS = {
    "VIIRS-J1": dict(
        page=NCC + "NOAA-20/J1VIIRS_NOAA20_SpectralResponseFunctions.php",
        zip="J1_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2.1_Nov2016.zip", url_dir=NCC + "NOAA-20/docs/",
        readme="Readme_J1_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2.1_Nov2016.docx",
        sat="NOAA-20 (JPSS-1, J1)",
        members={
            **{f"M{n}": f"J1_VIIRS_BA_RSR_V2F/J1_VIIRS_RSR_M{n}_BA_Fused_V2F.txt" for n in range(1, 8)},
            "M8": "J1_VIIRS_V1_RSR_used_in_V2/J1_VIIRS_RSR_M8_BA_V1F.txt",
            "M9": "J1_VIIRS_BA_RSR_V2.1F/J1_VIIRS_RSR_M9_BA_HB_V2.1F.txt",
            **{f"M{n}": f"J1_VIIRS_V1_RSR_used_in_V2/J1_VIIRS_RSR_M{n}_BA_V1F.txt" for n in (10, 11, 12, 14, 15)},
            "M13": "J1_VIIRS_BA_RSR_V2F/J1_VIIRS_RSR_M13_BA_V2F.txt",
            **{b: f"J1_VIIRS_V1_RSR_used_in_V2/J1_VIIRS_RSR_{b}_BA_V1F.txt" for b in ("M16", "M16A", "M16B")},
            "I1": "J1_VIIRS_BA_RSR_V2F/J1_VIIRS_RSR_I1_BA_Fused_V2F.txt",
            "I2": "J1_VIIRS_BA_RSR_V2F/J1_VIIRS_RSR_I2_BA_Fused_V2F.txt",
            **{f"I{n}": f"J1_VIIRS_V1_RSR_used_in_V2/J1_VIIRS_RSR_I{n}_BA_V1F.txt" for n in (3, 4, 5)},
            "DNBLGS": "J1_VIIRS_BA_RSR_V2F/J1_VIIRS_RSR_DNBLGS_BA_Fused_V2FS.txt",
            "DNBMGS": "J1_VIIRS_V1_RSR_used_in_V2/J1_VIIRS_RSR_DNBMGS_BA_V1F.txt",
        }),
    "VIIRS-J2": dict(
        page=NCC + "NOAA-21/J2VIIRS_SpectralResponseFunctions.php",
        zip="J2_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2_Jul2019.zip", url_dir=NCC + "NOAA-21/docs/",
        readme="Readme_J2_VIIRS_RSR_DAWG_At-Launch_Public_Release_V2_July2019.pdf",
        sat="NOAA-21 (JPSS-2, J2)",
        members={},  # filled from the archive listing (one BA file per band)
    ),
}

DET_NOTE = (
    "DetectorType=energy is assumed: the DAWG READMEs call the curves 'Relative Spectral Response' "
    "normalised to a peak of 1.0 and do not state photon or energy weighting; the band averages populate "
    "the SDR RSR look-up tables, where they weight spectral radiance (energy convention).")


def band_of_member(name):
    """J2_VIIRS_RSR_<band>_BA_... -> <band>."""
    return Path(name).name.split("_")[3]


def read_ba(zpath, member):
    """Return (band label, wavelength nm, response) of a DAWG band-average file, sorted by wavelength."""
    with zipfile.ZipFile(zpath) as z:
        lines = z.read(member).decode("ascii", errors="replace").splitlines()
    rows = [ln.split() for ln in lines if ln.strip() and ln.lstrip()[0] not in "#%"]
    assert all(len(r) == 3 for r in rows), member
    labels = {r[0] for r in rows}
    assert len(labels) == 1, (member, labels)
    d = np.array([[float(r[1]), float(r[2])] for r in rows])
    d = d[np.argsort(d[:, 0])]
    assert np.all(np.diff(d[:, 0]) > 0), (member, "duplicate wavelengths")
    return labels.pop(), d[:, 0], d[:, 1]


def check(band, member, label, wl, rsr):
    head = band if not band.startswith("DNB") else band[:4]
    assert label == head, (member, label, band)
    assert 0.999 < rsr.max() <= 1.0001, (member, rsr.max())
    # J2 Fused VisNIR/SWIR files (M1-M4, M10, I1) carry small negative out-of-band responses (min -1.3E-3,
    # I1); clean_curve sets them to 0 and records it.
    assert rsr.min() > -0.002, (member, rsr.min())
    peak = wl[np.argmax(rsr)]
    lo, hi = (DNB_NM if band.startswith("DNB") else (SPEC_UM[band][0] * 1e3, SPEC_UM[band][1] * 1e3))
    pad = 0.05 * (lo + hi) / 2
    assert lo - pad <= peak <= hi + pad, (member, "peak", peak, lo, hi)  # proves the column is nm


def run():
    for inst, s in SETS.items():
        zpath = DOWNLOADS / "JPSS" / s["zip"]
        members = s["members"]
        if not members:
            with zipfile.ZipFile(zpath) as z:
                members = {band_of_member(n): n for n in z.namelist() if "_BA_RSR_" in n and n.endswith(".txt")}
        assert len(members) == 25, (inst, sorted(members))
        bands = []
        for band in sorted(members, key=lambda b: (b[0] != "M", b[0] != "I", b)):
            member = members[band]
            label, wl, rsr = read_ba(zpath, member)
            check(band, member, label, wl, rsr)
            src = f"{s['url_dir']}{s['zip']} ({member}) | {s['page']}"
            bands.append(make_band(
                band, wl, rsr, unit="nm", detector_type="energy", detector_source="assumed",
                components="filter+optics+qe", components_source="original", sources=src,
                svo_id=f"JPSS/{inst}.{band}"))
            b = bands[-1]
            if band in SPEC_UM:
                # NOAA publishes a spectral range, not a centre: require the pivot inside it (2 % pad).
                piv = pivot_wavelength(b.wavelength, b.transmission, b.detector_type) / 1e4
                lo, hi = SPEC_UM[band]
                assert lo * 0.98 <= piv <= hi * 1.02, (inst, band, piv, lo, hi)
                PIVOTS[(inst, band)] = round(piv, 4)
        build_instrument("JPSS", inst, bands, raw_files=[zpath, DOWNLOADS / "JPSS" / PERF_URL.rsplit("/", 1)[1]],
                         ground_based=False, notes=(
            f"VIIRS on {s['sat']}: VIIRS Data Analysis Working Group (DAWG) at-launch band-averaged relative "
            f"spectral responses (RSR), 'Final prelaunch version', from NOAA NCC ({s['page']}); README "
            f"{s['readme']} inside the zip. Band average = average of the high-quality detector RSR (all "
            "detectors and, for I bands, both subsamples) at native sampling; stitched in-band + out-of-band, "
            "peak-normalised to 1.0 by the DAWG (not rescaled here); low-quality wavelengths carry 1E-10 as "
            "published; any small negative out-of-band responses (J2 M1-M4, M10, I1; min -1.3E-3) are set to 0. Wavelength column in nm (some headers say 'wvl_um'; values are nm, verified by each "
            "band's peak lying in its specified spectral range). M16 is the DAWG average of M16A and M16B "
            "(on-orbit M16 is a TDI of both); DNBLGS/DNBMGS are the Day/Night Band low- and mid-gain stages "
            "(the README recommends DNBMGS for the unmeasured high-gain stage). Components: whole-band system "
            "response (filter+optics+qe as one curve; 'Fused' VisNIR bands include crosstalk). " + DET_NOTE +
            f" Pivot check: every M/I band pivot lies inside its specification spectral range in NOAA NCC's "
            f"{PERF_URL} (no published central wavelengths)."))


def snpp_source_builder(zip_path, root):
    """Source-builder for Suomi NPP VIIRS (HELD: Northrop Grumman copyright, all rights reserved).

    The user downloads NG_VIIRS_NPP_RSR_filtered_Oct2011_BA.zip from
    https://ncc.nesdis.noaa.gov/documents/documentation/NG_VIIRS_NPP_RSR_filtered_Oct2011_BA.zip
    and builds JPSS/VIIRS locally (``root``).  Files: two columns, wavelength [nm] and band-averaged RSR
    (peak 1.0, Government Team NOTE 3); I3 uses the "_BA_Mod" file (NOTE 2).  No DNB or M16 in this set.
    """
    root = Path(root)
    bands = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("_BA.dat") or n.endswith("_BA_Mod.dat")]
        names = [n for n in names if "/._" not in n and not n.endswith("I3_filtered_Oct2011f_BA.dat")]
        for n in names:
            band = Path(n).name.split("_")[4]
            d = np.loadtxt(z.read(n).decode("ascii").splitlines())
            bands.append(make_band(
                band, d[:, 0], d[:, 1], unit="nm", detector_type="energy", detector_source="assumed",
                components="filter+optics+qe", components_source="original",
                sources=f"{NCC}documents/documentation/NG_VIIRS_NPP_RSR_filtered_Oct2011_BA.zip ({n})",
                svo_id=f"JPSS/VIIRS.{band}"))
    return build_instrument("JPSS", "VIIRS", bands, ground_based=False, root=root, notes=(
        "Suomi NPP VIIRS (F1) Northrop Grumman October 2011 band-averaged RSR, built locally from the "
        "user's own NOAA NCC download (not redistributed: NG copyright). " + DET_NOTE))
