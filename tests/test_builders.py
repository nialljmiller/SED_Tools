"""Builders (``component_sources``): every instrument built end to end from its team's or archive's files, and
every check that stops a build, or a source, before it writes a wrong curve or skips a band in silence.

Offline: the files are saved in tests/fixtures (builders/README.md says where each comes from and how it was
cut; argus/ and original_sources/ hold excerpts of the real inputs) or assembled in memory with the layout of
the original (FITS extensions and keywords, tarball members, PDS and IPAC tables).
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import logging
import shutil
import sys
import tarfile
import types
import zipfile

import numpy as np
import pytest
import requests

from sed_tools import filter_library
from sed_tools.api import Filters
from sed_tools.component_sources import (argus, cassini, catalog, cfht, csp, decam, eris, gmos, hsc, hst,
                                         legacy_surveys, lsst, observatories as obs, sdss, skycalc, space,
                                         standards, surveys, swift, team_atmospheres as team, twomass, vista, ztf)
from sed_tools.component_sources._common import (DerivedCurve, SimpleSource, Table, build_simple, ensure_published,
                                                 write_site)
from sed_tools.component_sources._original import Fetcher, clean_curve, make_band, to_angstrom
from sed_tools.filter_components import (DERIVED, FILTER, OPTICS, QE, SHARED, TEAM, AtmosphereGrid,
                                         ComponentCoverageError, ComponentError, InstrumentManifest, add_component,
                                         build_all_products, build_product, compose_band, read_curve)
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import (ENERGY, PHOTON, SVOFilterRecord, read_filter_header, released_mesa_reads,
                                       write_filter_file)
from sed_tools.filters import svo
from sed_tools.filters.curve_metadata import SVOMetadataClient, _param_text, svo_fps_url
from sed_tools.grids import mast, msg, svo as svo_grid

from support import TESTS, WL, Resp, Session, answer, derived, gauss, published_set, scenarios, team_site

SCENARIOS, row = scenarios()

# ======================================================================
# Files as the sources serve them
# ======================================================================
FIX = TESTS / "fixtures" / "builders"


def saved():
    """URL -> the bytes it served, for every saved response."""
    out = {}
    for url, spec in json.loads((FIX / "sources.json").read_text()).items():
        out[url] = (FIX / spec).read_bytes() if isinstance(spec, str) else gzip.decompress(
            (FIX / spec["gunzip"]).read_bytes())
    return out


ORIGINAL = TESTS / "fixtures" / "original_sources"  # a few lines of each team's file, same layout and header


def fx(name):
    return (ORIGINAL / name).read_text(encoding="latin-1")


def text_table(wl, tr, fmt="{} {}"):
    return "".join(fmt.format(w, t) + "\n" for w, t in zip(wl, tr))


def tar_gz(members, prefix=""):
    """A .tar.gz of ``members`` {name: text or bytes}, under ``prefix``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            data = data if isinstance(data, bytes) else data.encode("ascii")
            info = tarfile.TarInfo(f"{prefix}/{name}" if prefix else name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buf.getvalue()


def fits_table(columns, name=None, header=None, fmt="D"):
    """A FITS binary table of ``columns`` [(name, unit or None, values)], in ``fmt`` (TIME always "D")."""
    from astropy.io import fits

    hdu = fits.BinTableHDU.from_columns([fits.Column(name=n, format="D" if n == "TIME" else fmt, unit=u,
                                                     array=np.asarray(v))
                                         for n, u, v in columns], name=name)
    hdu.header.update(header or {})
    return hdu


def fits_file(*tables, comments=()):
    """The bytes of a FITS file: a primary HDU with these header comments, then ``tables``."""
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    for comment in comments:
        primary.header.add_comment(comment)
    buf = io.BytesIO()
    fits.HDUList([primary, *tables]).writeto(buf)
    return buf.getvalue()


def throughput(wl, tr):
    """A synphot throughput table."""
    return fits_file(fits_table([("WAVELENGTH", None, wl), ("THROUGHPUT", None, tr)]))


def published(directory, band, wl=(4000.0, 5000.0), tr=(0.5, 0.6), components="filter", detector=PHOTON,
              source="svo"):
    """A curve as SVO publishes it, with its components (and ComponentsSource ``source``, if any)."""
    directory.mkdir(parents=True, exist_ok=True)
    extra = {"Components": components, **({"ComponentsSource": source} if source else {})}
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=detector, source="svo",
                      wavelength_unit="Angstrom", extra=extra)


def curve(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2).T


class SVOClient:
    """SVO's metadata client: ``get_filter(fid)`` is ``answer(fid)``; ``asked`` lists the FilterIDs asked for."""

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def get_filter(self, fid):
        self.asked.append(fid)
        return self.answer(fid)


def record(fid, wl=(4000.0, 5000.0), tr=(0.5, 0.6), components="filter"):
    return SVOFilterRecord(fid, PHOTON, np.asarray(wl), np.asarray(tr), wavelength_unit="Angstrom",
                           components=components)


def check_published(inst_dir, facility, instrument, bands, components, detector):
    """Curves, headers, MESA index and manifest of an instrument built from its team's files."""
    assert sorted(p.stem for p in inst_dir.glob("*.dat")) == sorted(bands)
    assert (inst_dir / instrument).read_text().split() == sorted(f"{b}.dat" for b in bands)
    manifest = InstrumentManifest.load(inst_dir)
    assert (manifest.facility, manifest.instrument) == (facility, instrument)
    for band in bands:
        path = inst_dir / f"{band}.dat"
        header = read_filter_header(path)
        assert header["FilterID"] == f"{facility}/{instrument}.{band}"
        assert (header["DetectorType"], header["DetectorTypeSource"]) == (detector, "original")
        assert (header["Components"], header["ComponentsSource"]) == (components, "original")
        assert header["Sources"].startswith("http")
        wl, tr = curve(path)
        assert released_mesa_reads(path) and np.all(np.diff(wl) > 0) and tr.min() >= 0 and tr.max() > 0
        assert manifest.published_bands[band] == components
    assert manifest.detector_type_sources == dict.fromkeys(bands, "original")
    return manifest


class Book:
    """An xlrd workbook of ``sheets`` {name: rows}."""

    def __init__(self, sheets):
        self.sheets = [types.SimpleNamespace(name=name, nrows=len(rows), row_values=rows.__getitem__)
                       for name, rows in sheets.items()]

    def sheet_by_name(self, name):
        return next(s for s in self.sheets if s.name == name)

    def sheet_by_index(self, i):
        return self.sheets[i]


def xlrd_stand_in(t):
    """xlrd itself (an optional extra) is not needed: the saved workbooks are their cells as JSON."""
    t.mp.setitem(sys.modules, "xlrd", types.SimpleNamespace(
        open_workbook=lambda file_contents: Book(dict(json.loads(file_contents)["sheets"]))))


def lco_verified_copy(t):
    """The saved LCO zip is cut down, so it is checked against its own SHA-256."""
    sha, lco = hashlib.sha256(saved()[surveys.LCO_URL]).hexdigest(), surveys._lco
    t.mp.setattr(surveys, "_lco", lambda instrument, fetch: lco(instrument, fetch, sha256=sha))


#: What the observatories' and the surveys' rows set up first.
SETUP = {obs: xlrd_stand_in, surveys: lco_verified_copy}


# ======================================================================
# Every instrument built end to end from its saved files
# ======================================================================
@dataclasses.dataclass
class Build:
    """An instrument a module builds, and what it publishes. ``first`` is its first band's samples, first
    and last wavelength (A), peak wavelength (A) and peak."""

    module: object
    instrument: str
    facility: str
    bands: str
    components: str
    roles: str
    first: tuple
    detector: str = PHOTON
    site: str | None = None
    phrase: str = ""
    normalised: bool = False
    once: bool = True


def named(builds):
    """Each build under its row name: builds_from_the_<team or archive>_files_<instrument>."""
    return {f"builds_from_the_{'archive' if b.module is space else 'team'}_files_{b.instrument.replace('-', '_')}": b
            for b in builds}


BUILDS = named([
    # The observatory imagers, with a phrase of their notes.
    Build(obs, "ALFOSC", "NOT", "B R U V gp i_int ip rp up", "filter", "filter", (34, 3195.0, 7500.0, 4345.0, 0.684),
          phrase="up=#109 u'_SDSS 353_55"),
    Build(obs, "DEIMOS", "Keck", "B I R Z", "filter+optics+qe", "", (33, 3500.0, 10900.0, 4499.97, 0.397678),
          phrase="column 2 (THROUGHPUT) of the synphot tables"),
    Build(obs, "FOCAS", "Subaru", "B I R U V", "filter", "filter", (35, 3570.0, 5800.0, 4810.0, 0.8426),
          phrase="percent converted to fractions"),
    Build(obs, "FORS1", "Paranal", "B_BESS I_BESS R_BESS U_BESS V_BESS g_GUNN r_GUNN u_GUNN v_GUNN", "filter",
          "filter", (30, 3300.0, 10800.0, 4200.0, 0.68712199), site="paranal", phrase="z_GUNN and H_Alpha held"),
    Build(obs, "FORS2", "Paranal", "I_BESS R_SPECIAL b_HIGH g_HIGH u_HIGH v_HIGH", "filter", "filter",
          (34, 5900.2, 10290.6, 7812.6, 0.954332), site="paranal", phrase="current FORS2 standard configuration"),
    Build(obs, "FourStar", "Magellan", "H Hlong Hshort J J1 J2 J3 Ks NB118 NB209", "filter+optics", "",
          (33, 14300.0, 18200.0, 16520.0, 0.9082),
          phrase="re-derived from the *_cam_optics.txt / *_convolve.txt files"),
    Build(obs, "LRIS", "Keck", "B I R Rs V g", "filter", "filter", (34, 3267.93, 8249.45, 4299.77, 0.7164691),
          phrase="I is the vendor test report I.xls"),
    Build(obs, "MegaCam", "Magellan", "g i r u z", "filter+optics+qe+atmosphere", "filter optics qe",
          (102, 1000.0, 29990.0, 5000.0, 0.540648), phrase="atmosphere at 1.2 airmasses"),
    Build(obs, "Mosaic", "KPNO", "B I R U V i_SDSS r_SDSS", "filter", "filter", (33, 3600.0, 6000.0, 4418.0, 0.6937),
          phrase="U=k1044 Us solid U (k1044bp.txt); B=k1002"),
    Build(obs, "MOSFIRE", "Keck", "H H1 H2 J J2 J3 K Ks Y", "filter", "filter",
          (33, 13575.0, 19075.0, 17050.0, 0.993073), phrase="J2, J3, H1 and H2 are the intermediate bands"),
    Build(obs, "NIRC2", "Keck", "H J K Kp Ks Lp Ms", "filter", "filter", (34, 13968.46, 19140.01, 15198.75, 0.8981323),
          phrase="filter_passbands.xls (micron, percent -> fraction)"),
    Build(obs, "ODI", "WIYN", "NB422 NB695 r u", "filter", "filter", (33, 3000.0, 11000.0, 4245.0, 0.974354),
          phrase="g, i, z, NB659 and NB746 are HELD"),
    Build(obs, "OmegaCAM", "Paranal", "B_JOHN V_JOHN g_SDSS i_SDSS r_SDSS u_SDSS z_SDSS", "filter+qe", "filter",
          (21, 2996.45, 12008.38, 4799.16, 0.70318575), site="paranal",
          phrase="mean of their 4 segment-centre scans"),
    Build(obs, "SuprimeCam", "Subaru", "B Ic Rc V Y g i r z", "filter", "filter qe",
          (32, 3520.2, 5490.0, 4777.3, 0.90685), phrase="Hamamatsu FDCCD QE"),
    # The space missions, some peak-normalised.
    Build(space, "DR2", "GAIA", "BP G RP", "filter+optics+qe", "", (16, 3265.0, 7240.0, 4855.0, 0.659394605)),
    Build(space, "DR2r", "GAIA", "BP G RP", "filter+optics+qe", "", (16, 3265.0, 7240.0, 4855.0, 0.673483243)),
    Build(space, "HFI", "Planck", "100 143 217 353 545 857", "filter+optics+qe", "",
          (48, 166666.667, 34676769.2, 33313077.4, 1.0), ENERGY, normalised=True),
    Build(space, "Hipparcos", "Hipparcos", "Hp", "filter+optics+qe", "", (5, 3900.0, 8900.0, 4100.0, 0.423)),
    Build(space, "IRAC", "Spitzer", "I1 I2 I3 I4", "filter+optics+qe", "", (38, 30810.6, 40103.8, 37439.9, 0.472956)),
    Build(space, "IRAS", "IRAS", "100 12 25 60", "filter+optics+qe", "filter+optics qe",
          (6, 650000.0, 1400000.0, 1100000.0, 1.0), ENERGY, normalised=True),
    Build(space, "IRS", "Spitzer", "PU_blue PU_red", "filter+qe", "", (38, 100104.0, 350363.0, 147312.0, 2.96)),
    Build(space, "LFI", "Planck", "30 44 70", "filter+optics+qe", "", (53, 74948114.5, 140747632.9, 109413305.8, 1.0),
          ENERGY, normalised=True),
    Build(space, "MIPS", "Spitzer", "160 24 70", "filter+qe", "", (6, 1000851.0, 1164018.0, 1164018.0, 1.0), ENERGY,
          normalised=True),
    Build(space, "MSX", "MSX", "A C D E", "filter+optics+qe", "", (38, 58000.0, 113600.0, 99100.0, 1.0), ENERGY,
          normalised=True),
    Build(space, "OM", "XMM", "B U UVM2 UVW1 UVW2 V", "filter+optics+qe", "", (79, 1000.0, 7000.0, 3980.0, 1.0),
          normalised=True),
    Build(space, "PACS", "Herschel", "100 160 70", "filter", "filter", (49, 25003.0994, 5.0e6, 977994.995, 0.58252603),
          ENERGY),
    Build(space, "SPIRE", "Herschel", "250 350 500", "filter+optics+qe", "",
          (52, 1666666.667, 1.0e7, 2666666.667, 1.0), ENERGY, normalised=True),
    Build(space, "Tycho", "Hipparcos", "BT VT", "filter+optics+qe", "", (3, 3750.0, 3850.0, 3850.0, 0.556)),
    # The survey and standard photometric systems, some with a default atmosphere site.
    Build(surveys, "ATLAS", "ATLAS", "c o", "filter+optics+qe+atmosphere", "", (57, 4000.0, 6800.0, 6400.0, 0.717)),
    Build(standards, "Bessell", "Generic", "B I R U V", "filter+optics+qe", "", (4, 4100.0, 4400.0, 4100.0, 1.0)),
    Build(standards, "MKO", "Generic", "H J K Lp Mp", "filter", "filter",
          (38, 12977.0, 19964.0, 16111.0, 0.90177917)),
    Build(surveys, "OGLE-IV", "OGLE", "I V", "filter+qe", "filter qe", (8, 3101.2, 9475.09, 7619.99, 0.6787273)),
    Build(surveys, "PS1", "PAN-STARRS", "g i r w y z", "filter+optics+qe+atmosphere", "",
          (73, 3800.0, 11000.0, 5400.0, 0.501), site="haleakala_ps1", once=False),  # its site reads table3 again
    Build(surveys, "Sinistro", "LasCumbres", "B I R U V Y gp ip rp up zs", "filter", "filter",
          (39, 2000.0, 13000.0, 4320.0, 0.7274351)),
    Build(surveys, "Spectral", "LasCumbres", "B I R V Y gp ip rp up zs", "filter", "filter",
          (39, 2000.0, 13000.0, 4320.0, 0.7274351)),
    Build(standards, "USNO", "Generic", "gp ip rp up zp", "filter+optics+qe+atmosphere", "",
          (20, 3630.0, 5830.0, 5130.0, 0.38349)),
    Build(surveys, "WFCAM", "UKIRT", "H J K Y Z", "filter+optics+qe+atmosphere", "qe",
          (39, 13510.0, 19500.0, 16300.0, 0.266)),
])


def builds(t, b):
    SETUP.get(b.module, lambda t: None)(t)
    session = Session(saved())
    bands = b.bands.split()
    inst_dir = b.module.build(t.tmp / "filters", instrument=b.instrument, session=session)
    assert inst_dir == t.tmp / "filters" / b.facility / b.instrument
    manifest = check_published(inst_dir, b.facility, b.instrument, bands, b.components, b.detector)
    assert manifest.ground_based is (b.module is not space) and manifest.outside_range == "zero"
    assert manifest.default_site == b.site and b.phrase in manifest.notes
    assert sorted(manifest.components) == b.roles.split()
    for files in manifest.components.values():
        assert all(released_mesa_reads(inst_dir / rel) for rel in files.values())
    n, first, last, at, peak = b.first
    wl, tr = curve(inst_dir / f"{bands[0]}.dat")
    assert (len(wl), wl[0], wl[-1], wl[np.argmax(tr)]) == (n, pytest.approx(first), pytest.approx(last),
                                                          pytest.approx(at))
    assert tr.max() == pytest.approx(peak)
    if b.normalised:
        assert all(curve(inst_dir / f"{band}.dat")[1].max() == 1.0 for band in bands)
    if b.once:
        assert len(session.calls) == len(set(session.calls))  # each file is downloaded once


def every_recipe_is_covered(t, *modules):
    assert (sorted(b.instrument for b in BUILDS.values() if b.module in modules)
            == sorted(name for module in modules for name in module.RECIPES))


EVERY_RECIPE = {"every_recipe_is_covered_observatories": (obs,), "every_recipe_is_covered_space": (space,),
                "every_recipe_is_covered_surveys": (surveys, standards)}


# ======================================================================
# A wrong file stops the build before anything is written
# ======================================================================
def copy_of(url):
    return lambda files: files[url]


def without_atlas_row_450(files):
    with tarfile.open(fileobj=io.BytesIO(files[surveys.ATLAS_URL])) as tf:
        tex = tf.extractfile("atlas_defn.tex").read().decode()
    row = next(ln for ln in tex.splitlines() if ln.startswith("450 & "))
    return tar_gz({"atlas_defn.tex": tex.replace(row + "\n", "")})


KECK, KPNO, FS, MC = obs.KECK, obs.KPNO_DOCS, obs.FS, obs.MC
#: name: (module, instrument, the URL whose file is wrong, the wrong file (from the saved ones), the error)
WRONG_FILE = {
    "a_wrong_file_stops_the_build_Mosaic": (obs, "Mosaic", KPNO + "k1002bp_jul04.txt",
                                            copy_of(KPNO + "k1003bp_jul04.txt"), r"k1002bp_jul04.txt: filter line"),
    "a_wrong_file_stops_the_build_LRIS": (obs, "LRIS", KECK + "lris/data_files/B.dat",
                                          copy_of(KECK + "mosfire/data/filter/mosfire_J.txt"),
                                          "LRIS B: not Angstrom / fraction"),
    "a_wrong_file_stops_the_build_MOSFIRE": (obs, "MOSFIRE", KECK + "mosfire/data/filter/mosfire_J.txt",
                                             copy_of(KECK + "lris/data_files/B.dat"),
                                             "MOSFIRE J: not micron / fraction"),
    "a_wrong_file_stops_the_build_DEIMOS": (obs, "DEIMOS", KECK + "deimos/Bfaber13norm.txt",
                                            copy_of(KECK + "deimos/RlrisDredonorm.txt"), "DEIMOS B: header PIVWV"),
    "a_wrong_file_stops_the_build_FourStar": (obs, "FourStar", FS + "HST_STD/J.txt", copy_of(FS + "HST_STD/J1.txt"),
                                              "FourStar J: HST_STD differs"),
    "a_wrong_file_stops_the_build_MegaCam": (obs, "MegaCam", MC + "filterthroughput.magr",
                                             copy_of(MC + "filterthroughput.magu"), "MegaCam r: T column differs"),
    "a_wrong_file_stops_the_build_FOCAS": (obs, "FOCAS", obs.FO + "txt/U.txt", copy_of(obs.SC + "txt/B.txt"),
                                           "outside 290-1001"),
    "a_wrong_file_stops_the_build_IRAS": (space, "IRAS", space.IRAS_URL,
                                          lambda f: f[space.IRAS_URL].replace(b"100 &nbsp;&micro;m Band", b""),
                                          "IRAS: unexpected bands"),
    "atlas_needs_every_table_row": (surveys, "ATLAS", surveys.ATLAS_URL, without_atlas_row_450,
                                    "ATLAS c: parsed 54 rows, expected 57"),
    "bessell_checks_the_readme": (standards, "Bessell", standards.B12_URL.replace("table1.dat", "ReadMe"),
                                  lambda f: b"Byte-by-byte Description of file: table1.dat\n",
                                  "unexpected Bessell & Murphy ReadMe"),
}


def stops_before_writing(t, module, instrument, url, wrong, message):
    SETUP.get(module, lambda t: None)(t)
    files = saved()
    files[url] = wrong(files)
    with pytest.raises(ValueError, match=message):
        module.build(t.tmp, instrument=instrument, session=Session(files))
    assert not any(t.tmp.iterdir())


# ======================================================================
# Argus (DR280, and the legacy System)
# ======================================================================
ARGUS = TESTS / "fixtures" / "argus"
PUBLIC = ("argus_beta_ct446.csv", "argus_rho_ct699.csv", "imx455_qe.csv")
PRIVATE = ("addem_optics.csv", "synthref_response_rho.csv")  # DR280's non-public inputs
SYSTEM_RAW = ("argus_g.csv", "bandpass_560_825nm.csv", "optics_throughput.csv", "imx455_qe.csv")


def argus_url(name):
    return argus.ARGUS_SIM_RAW.format(commit=argus.ARGUS_SIM_COMMIT, file=name)


def argus_inputs(t):
    """tests/fixtures/argus holds excerpts of the inputs pinned in argus (every 5th data row of each 1 nm table,
    the whole IMX455 QE, every 100th row of the System curves): every pin is pointed at its excerpt's SHA-256.
    Returns the raw inputs' folder and a session serving the public ones at their pinned URLs."""
    def repinned(inputs):
        return {key: dataclasses.replace(spec, sha256=hashlib.sha256((ARGUS / spec.file).read_bytes()).hexdigest())
                for key, spec in inputs.items()}

    t.mp.setattr(argus, "DR280_INPUTS", repinned(argus.DR280_INPUTS))
    t.mp.setattr(argus, "SYSTEM_INPUTS", repinned(argus.SYSTEM_INPUTS))
    raw = t.tmp / "raw"
    raw.mkdir()
    for name in PRIVATE + SYSTEM_RAW:
        shutil.copyfile(ARGUS / name, raw / name)
    return raw, Session({argus_url(name): (ARGUS / name).read_bytes() for name in PUBLIC})


def build_dr280(t):
    raw, session = argus_inputs(t)
    return argus.build(t.tmp / "filters", raw, instrument="DR280", session=session)


def argus_system(t):
    """Argus/System as Nill wrote it."""
    inst = t.tmp / "filters" / "Argus" / "System"
    inst.mkdir(parents=True)
    for band in ("beta", "rho"):
        shutil.copyfile(ARGUS / f"system_{band}.dat", inst / f"{band}.dat")
    (inst / "System").write_text("beta.dat\nrho.dat\n")
    return inst


def nm_table(name, column=1, skip=1):
    table = np.loadtxt(ARGUS / name, delimiter=",", skiprows=skip, usecols=(0, column))
    return table[:, 0] * 10.0, table[:, 1]


@row
def dr280_writes_mesa_readable_totals_and_index(t):
    dr280 = build_dr280(t)
    assert (dr280 / "DR280").read_text() == "beta.dat\nrho.dat\n"
    for band in ("beta", "rho"):
        path = dr280 / f"{band}.dat"
        assert released_mesa_reads(path)
        lines = path.read_text().splitlines()
        assert lines[0].startswith("Wavelength,Transmission,")
        assert not any(line.startswith("#") or not line.strip() for line in lines)
        header = read_filter_header(path)
        assert header["FilterID"] == f"Argus/DR280.{band}"
        assert header["DetectorType"] == "energy" and header["DetectorTypeSource"] == "user"
        assert header["Components"] == "filter+optics+qe"
        assert header["ComponentsSource"] == "composed"
        assert header["WavelengthUnit"] == "Angstrom"
        assert argus.ARGUS_SIM_COMMIT in header["Sources"] and "addem_optics.csv" in header["Sources"]
        curve = read_curve(path)
        assert curve.wavelength[0] == 3000.0 and curve.wavelength[-1] == 11000.0  # nm -> Angstrom
        assert np.all(np.diff(curve.wavelength) > 0) and curve.transmission.min() >= 0


@row
def dr280_components_are_the_inputs_in_angstrom(t):
    dr280 = build_dr280(t)
    expected = {
        "filter/beta_ct446.dat": nm_table("argus_beta_ct446.csv"),
        "filter/rho_ct699.dat": nm_table("argus_rho_ct699.csv"),
        "filter/rho0_ct692.dat": nm_table("synthref_response_rho.csv", column=2),
        "optics/dr280.dat": nm_table("addem_optics.csv"),
        "qe/imx455.dat": nm_table("imx455_qe.csv", skip=0),
    }
    for rel, (wavelength, transmission) in expected.items():
        curve = read_curve(dr280 / "components" / rel)
        np.testing.assert_array_equal(curve.wavelength, wavelength)
        np.testing.assert_array_equal(curve.transmission, transmission)
        assert released_mesa_reads(dr280 / "components" / rel)


@row
def filters_compose_builds_an_energy_labelled_mesa_product_for_each_filter_variant(t):
    dr280 = build_dr280(t)
    for filt in ("production", "addem"):
        out = Filters.compose(dr280, "filter+optics+qe", variants={"filter": filt})
        assert out.name == f"filter+optics+qe+filter_{filt}"
        assert (out / out.name).read_text() == "beta.dat\nrho.dat\n"
        for band in ("beta", "rho"):
            assert released_mesa_reads(out / f"{band}.dat")
            header = read_filter_header(out / f"{band}.dat")
            assert header["DetectorType"] == "energy" and header["DetectorTypeSource"] == "composed"
    # Every recipe keeps the team's convention, with or without the QE.
    for recipe in ("filter", "filter+optics", "filter+qe"):
        out = Filters.compose(dr280, recipe)
        assert {read_filter_header(out / f"{b}.dat")["DetectorType"] for b in ("beta", "rho")} == {"energy"}
    production = Filters.compose(dr280, "filter+optics+qe", variants={"filter": "production"})
    for band in ("beta", "rho"):  # the product's samples are the published curve's
        assert ((production / f"{band}.dat").read_text().splitlines()[1:]
                == (dr280 / f"{band}.dat").read_text().splitlines()[1:])


@row
def dr280_refuses_an_input_that_is_not_the_pinned_file(t):
    raw, session = argus_inputs(t)
    url = argus_url("argus_beta_ct446.csv")
    session.files[url] = session.files[url] + b"1101.000000,0.000000\n"
    with pytest.raises(ValueError, match="SHA-256 .* is not the pinned"):
        argus.build(t.tmp / "filters", raw, instrument="DR280", session=session)
    assert not (t.tmp / "filters").exists()


@row
def dr280_never_replaces_a_different_published_file(t):
    raw, session = argus_inputs(t)
    inst = t.tmp / "filters" / "Argus" / "DR280"
    inst.mkdir(parents=True)
    text = "Wavelength,Transmission\n4000.0,0.5\n5000.0,0.5\n"
    (inst / "beta.dat").write_text(text)
    with pytest.raises(RuntimeError, match="never replaced; nothing was written"):
        argus.build(t.tmp / "filters", raw, instrument="DR280", session=session)
    assert (inst / "beta.dat").read_text() == text
    assert [p.name for p in inst.iterdir()] == ["beta.dat"]


@row
def system_builder_labels_line_1_and_keeps_every_sample(t):
    raw, session = argus_inputs(t)
    system_dir = argus_system(t)
    before = {p.name: p.read_bytes() for p in system_dir.iterdir()}
    argus.build(t.tmp / "filters", raw, instrument="DR280", session=session)
    assert {p.name: p.read_bytes() for p in system_dir.iterdir()} == before
    assert read_filter_header(system_dir / "beta.dat") == {}  # as Nill wrote them
    argus.build(t.tmp / "filters", raw)  # System is the default instrument
    assert (system_dir / "System").read_bytes() == before["System"]
    for band in ("beta", "rho"):
        path = system_dir / f"{band}.dat"
        old, new = before[f"{band}.dat"].splitlines(keepends=True), path.read_bytes().splitlines(keepends=True)
        assert new[1:] == old[1:]  # MESA discards line 1, so it reads the same numbers
        assert old[0] == b"Wavelength,Transmission\n" and new[0].startswith(b"Wavelength,Transmission,")
        assert released_mesa_reads(path)
        header = read_filter_header(path)
        assert header == argus.system_header(band)
        assert header["FilterID"] == f"Argus/System.{band}"
        assert (header["DetectorType"], header["DetectorTypeSource"]) == ("energy", "user")
        assert "optics_throughput.csv" in header["Sources"] and "785c92e8" in header["Sources"]
    labelled = {p.name: p.read_bytes() for p in system_dir.rglob("*") if p.is_file()}
    argus.build(t.tmp / "filters", raw)
    assert {p.name: p.read_bytes() for p in system_dir.rglob("*") if p.is_file()} == labelled


@row
def system_curve_without_a_column_row_is_refused(t):
    raw, _ = argus_inputs(t)
    system_dir = argus_system(t)
    samples = (system_dir / "beta.dat").read_text().splitlines(keepends=True)[1:]
    (system_dir / "beta.dat").write_text("".join(samples))
    with pytest.raises(ValueError, match="beta.dat has no column row"):
        argus.build(t.tmp / "filters", raw)
    assert (system_dir / "beta.dat").read_text() == "".join(samples)


@row
def system_components_and_composition(t):
    raw, _ = argus_inputs(t)
    system_dir = argus_system(t)
    argus.build(t.tmp / "filters", raw, instrument="System")
    manifest = InstrumentManifest.load(system_dir)
    assert manifest.components == {
        "filter": {"beta": "components/filter/beta.dat", "rho": "components/filter/rho.dat"},
        "optics": {"*": "components/optics/optics.dat"},
        "qe": {"*": "components/qe/imx455.dat"},
    }
    assert manifest.published["components"] == "filter+optics+qe"
    assert manifest.detector_type == "energy"
    assert manifest.detector_types == {"beta": "energy", "rho": "energy"}
    assert manifest.detector_type_sources == {"beta": "user", "rho": "user"}
    assert "Argus/DR280" in manifest.notes and "DetectorType energy (user)" in manifest.notes
    beta = read_curve(system_dir / "components/filter/beta.dat")
    wavelength, percent = nm_table("argus_g.csv")
    np.testing.assert_array_equal(beta.wavelength, wavelength)
    np.testing.assert_array_equal(beta.transmission, percent / 100.0)
    for band in ("beta", "rho"):
        curve = compose_band(system_dir, band, "filter+optics+qe")
        assert curve.transmission.max() > 0.5
    out = Filters.compose(system_dir, "filter+optics+qe")
    assert {read_filter_header(out / f"{b}.dat")["DetectorType"] for b in ("beta", "rho")} == {"energy"}


@row
def system_without_raw_inputs_points_to_njm(t):
    argus_inputs(t)
    system_dir = argus_system(t)
    with pytest.raises(FileNotFoundError, match="optics_throughput.csv .*ArgusSim.*NJM"):
        argus.build(t.tmp / "filters")
    assert not (system_dir / "components").exists()


@row
def only_dr280_is_registered_and_njm_serves_its_curves(t):
    dr280 = catalog.builder_for("Argus", "DR280")
    assert (dr280.name, dict(dr280.kwargs), dr280.publishes) == ("argus", {"instrument": "DR280"}, False)
    assert catalog.builder_for("Argus", "System") is None
    assert catalog.component_availability("Argus", "DR280")["status"] == "supported"


# ======================================================================
# CFHT, ERIS/NIX, 2MASS and LSST
# ======================================================================
def cfht_files(codes):
    nm = np.arange(300, 1101, 50)
    files = {
        cfht.PRIMARY: text_table(nm, np.full(nm.size, 90.0)),
        cfht.CORRECTOR: text_table(nm, np.full(nm.size, 80.0)),
        cfht.QE_MODEL: "lambda (nm)  mean (%)\n" + text_table(np.arange(280, 1051, 10), np.full(78, 50.0)),
        cfht.QE_MEASURED: "lambda (nm) mean (%) stdev (%)\n350 48.9 6.2\n1000 4.3 0.2\n",
    }
    for code in codes:
        files[cfht.CURVE.format(code=code)] = "".join(
            f"{w},{t},{t}\n" for w, t in ((800, 0), (850, 90), (900, 90), (950, 0)))
    return files


def ipac_page(title, um, values):
    """An IPAC table page as 2MASS serves its relative spectral responses."""
    rows = "".join(f"  {i + 1:3d}  {w:.3f}  {v:.4f}\n" for i, (w, v) in enumerate(zip(um, values)))
    return (f"<html><head><title>{title}</title></head><body><b><pre>\n\\CHAR TITLE = {title}\n"
            f"|n  |xlam     |yval     |\n|i  |r        |r        |\n{rows}</pre></body></html>")


def twomass_files():
    um = np.linspace(1.0, 2.5, 31)
    files = {}
    values = {1: 0.9, 2: 0.95, 3: 0.97, 4: 0.99, 6: 0.98, 7: 0.97, 9: 0.96, 11: 0.6}
    for n, title in twomass.TITLES.items():
        if n in values:
            files[twomass.TABLE.format(n=n)] = ipac_page(title, um, np.full(um.size, values[n]))
    for n, (lo, hi) in ((5, (1.1, 1.4)), (8, (1.5, 1.8)), (10, (2.0, 2.3))):
        x = np.linspace(lo - 0.05, hi + 0.05, 13)
        files[twomass.TABLE.format(n=n)] = ipac_page(twomass.TITLES[n].replace("-BAND", "K<sub>s</sub>-BAND")
                                                     if n == 10 else twomass.TITLES[n], x,
                                                     np.where((x > lo) & (x < hi), 0.8, 0.0))
    return files


def lsst_files(tag, scale=1.0):
    nm = np.arange(300.0, 1151.0, 50.0)
    one = text_table(nm, np.full(nm.size, 0.9))
    files = {}
    for name in lsst.OPTICS_FILES + ("detector", "atmos_10", "atmos_std"):
        files[lsst._url(name, tag)] = "# header\n" + one
    for b in lsst.BANDS:
        filt = np.where((nm >= 500) & (nm <= 700), 0.8 * scale, 0.0)
        files[lsst._url(f"filter_{b}", tag)] = text_table(nm, filt)
        files[lsst._url(f"total_{b}", tag)] = text_table(nm, filt * 0.9 ** 8)
        files[lsst._url(f"hardware_{b}", tag)] = text_table(nm, filt * 0.9 ** 7)
    return files


def nix_zip():
    def metres(lo_um, hi_um, value):
        x = np.linspace(lo_um, hi_um, 50) * 1e-6
        return "# data from ERIS/NIX/x.txt\n" + "".join(f"  {w:.16f}  {value}\n" for w in x)

    return zip_bytes({"nix-filters/cameras1-13masJHK.dat": metres(1.0, 2.59, 0.9),
                      "nix-filters/cameras2-27mas-JHK.dat": metres(1.0, 2.69, 0.85),
                      "nix-filters/cameras3-13mas-LM.dat": metres(2.93, 5.5, 0.8),
                      "nix-filters/Dichroic.txt": "# \n# [1] : Wavelength [nm]\n# [2] : Transmission [%]\n# \n"
                                                  + text_table(np.arange(900, 5401, 100), np.full(46, 50.0)),
                      "nix-filters/ND.txt": text_table([1000, 5000], [0.19, 1.61])})


@row
def cfht_bands_get_their_own_generation_and_ignore_mislabelled_sibling(t):
    inst = t.tmp / "CFHT" / "Megaprime"
    wl = np.linspace(8000, 9500, 16)
    published(inst, "z", wl, np.ones_like(wl) * 0.3, "filter+optics+qe")
    published(inst, "z_fil", wl, np.ones_like(wl), "filter")    # SVO's sibling: really the old Z'
    published(inst, "zS", wl, np.ones_like(wl) * 0.3, "filter+optics+qe")
    before = (inst / "z.dat").read_bytes()
    files = cfht_files(["9901", "9801"])
    files[cfht.CURVE.format(code="9801")] = "1200.00 94.0\n800.00 50.0\n700.00 0.0\n"

    cfht.build(t.tmp, instrument="Megaprime", session=Session(files))

    m = InstrumentManifest.load(inst)
    assert (inst / "z.dat").read_bytes() == before
    assert m.component_file(FILTER, "z") == "components/filter/z.dat"
    assert "cfh9901" in read_filter_header(inst / "components/filter/z.dat")["Sources"]
    assert "cfh9801" in read_filter_header(inst / "components/filter/zS.dat")["Sources"]
    assert m.component_file(FILTER, "z_fil") == "z_fil.dat"           # the sibling stays itself
    assert m.component_file(OPTICS, "z") == "components/optics/primary_corrector.dat"
    optics = read_curve(inst / "components/optics/primary_corrector.dat")
    np.testing.assert_allclose(optics.transmission, 0.72)               # one reflection x corrector
    assert m.component_file(QE, "z") == "components/qe/e2v_ccd42-90_model.dat"
    assert (inst / "components/qe/measured_mean.dat").exists()
    assert "measured_mean" not in str(m.components)
    assert m.outside_range == "zero" and m.default_site is None
    # zS (old long-pass to 1200 nm) composes: the model QE ends at 1050 nm and is zero beyond.
    zs = compose_band(inst, "zS", (FILTER, OPTICS, QE))
    assert zs.transmission[zs.wavelength > 10500].max() == 0.0
    for p in inst.rglob("*.dat"):
        assert released_mesa_reads(p)


@row
def cfht_needs_published_curves(t):
    with pytest.raises(FileNotFoundError):
        cfht.build(t.tmp, instrument="MegaCam", session=Session({}))


@row
def eris_unknown_band_is_refused(t):
    published(t.tmp / "ERIS" / "NIX", "Q", np.linspace(11500, 14000, 20), np.full(20, 0.8))
    with pytest.raises(ValueError, match="camera"):
        eris.build(t.tmp, session=Session({eris.ZIP_URL: nix_zip()}))


@row
def twomass_table_title_is_checked(t):
    with pytest.raises(ValueError, match="title"):
        twomass.micron_curve(ipac_page("SOMETHING ELSE", [1.0, 2.0], [0.5, 0.5]), "u", twomass.TITLES[1])


@row
def lsst_releases_are_separate_instruments(t):
    filters = t.tmp / "filters"
    old = filters / "LSST" / "LSST"
    wl = np.linspace(5000, 7000, 21)
    for b in lsst.BANDS:
        published(old, b, wl, np.full(wl.size, 0.4), "filter+instrument+atmosphere")
    site = t.tmp / "atmospheres" / "lsst_std"
    site.mkdir(parents=True)
    AtmosphereGrid(site="lsst_std", entries=[{"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}]).save(site)
    write_filter_file(site / "am1.20.dat", [3000.0, 12000.0], [0.9, 0.9], detector_type=ENERGY,
                      source="component", wavelength_unit="Angstrom")
    site_before = (site / "manifest.json").read_bytes()
    session = Session({**lsst_files("1.5"), **lsst_files("v1.9.2", scale=0.5)})
    svo_client = SVOClient(lambda fid: None)

    lsst.build(filters, instrument="LSST", session=session, svo_client=svo_client)
    lsst.build(filters, instrument="LSSTv19", session=session)

    assert svo_client.asked == []  # published curves exist; SVO must not be asked
    m15 = InstrumentManifest.load(old)
    assert m15.component_file(FILTER, "g") == "components/filter/g_v1p5.dat"
    assert m15.component_file(QE, "g") == "components/qe/detector_v1p5.dat"
    # Component sources are recorded per band ("<role>:<band>") since PR #51.
    assert "lsst/throughputs 1.5" in m15.sources[f"{FILTER}:g"]
    assert "X=1.0" in m15.notes and "1.2 airmasses" in m15.notes
    new = filters / "LSST" / "LSSTv19"
    m19 = InstrumentManifest.load(new)
    assert m19.bands == list(lsst.BANDS) and m19.default_site == "lsst_std"
    assert m19.published_bands["r"] == "filter+optics+qe+atmosphere"
    header = read_filter_header(new / "r.dat")
    assert header["FilterID"] == "LSST/LSSTv19.r" and header["DetectorTypeSource"] == "assumed"
    assert (new / "LSSTv19").read_text().split() == [f"{b}.dat" for b in sorted(lsst.BANDS)]
    composed = compose_band(new, "r", (FILTER, OPTICS, QE), atmosphere={"airmass": 1.2},
                            atmosphere_root=t.tmp / "atmospheres")
    published_r = read_curve(new / "r.dat")
    np.testing.assert_allclose(np.interp(published_r.wavelength, composed.wavelength, composed.transmission),
                               published_r.transmission, atol=1e-12)
    assert (site / "manifest.json").read_bytes() == site_before        # existing site is not rewritten
    for p in new.rglob("*.dat"):
        assert released_mesa_reads(p)


@row
def earlier_manifest_note_is_carried_once(t):
    inst = t.tmp / "ERIS" / "NIX"
    published(inst, "J", np.linspace(11500, 14000, 20), np.full(20, 0.8))
    m = InstrumentManifest.load(inst)
    m.notes = "One negative sample clipped to 0."
    m.save(inst)
    session = Session({eris.ZIP_URL: nix_zip()})
    eris.build(t.tmp, session=session)
    first = InstrumentManifest.load(inst).notes
    eris.build(t.tmp, session=session)
    assert InstrumentManifest.load(inst).notes == first
    assert first.endswith("Earlier note: One negative sample clipped to 0.")
    assert first.count("Earlier note") == 1


# ======================================================================
# SVO's filter browser and VOTables (saved pages), and the spectra grabbers' discovery
# ======================================================================
GAIA3 = svo_fps_url().replace("fps.php", "index.php?mode=browse&gname=GAIA&gname2=GAIA3&asttype=")


@row
def browser_lists_an_instruments_filters_from_svos_page(t):
    session = Session(saved())
    browser = svo.SVOFilterBrowser(base_dir=str(t.tmp), session=session)
    rows = browser.list_filters("GAIA", "GAIA3")
    # SVO's instrument column says "GAIA" for every Gaia release; the rows are filed under GAIA3.
    assert [(r.filter_id, r.band, r.facility, r.instrument, r.description) for r in rows] == [
        ("GAIA/GAIA3.Gbp", "Gbp", "GAIA", "GAIA3", "GAIA Gbp filter, DR3"),
        ("GAIA/GAIA3.G", "G", "GAIA", "GAIA3", "GAIA G filter, DR3"),
        ("GAIA/GAIA3.Grp", "Grp", "GAIA", "GAIA3", "GAIA Grp filter, DR3"),
        ("GAIA/GAIA3.Grvs", "Grvs", "GAIA", "GAIA3", "GAIA Grvs filter, DR3"),
    ]
    assert [i.key for i in browser.list_instruments("GAIA")] == ["GAIA0", "GAIA2", "Gaia2m", "GAIA2r", "GAIA3"]
    assert browser.list_filters("GAIA", "GAIA3") is rows and len(session.calls) == 2  # both pages cached


@row
def browser_finds_no_filters_on_a_page_without_the_table(t):
    files = saved()
    files[GAIA3] = b"<html><body><table><tr><td>No filters</td></tr></table></body></html>"
    assert svo.SVOFilterBrowser(base_dir=str(t.tmp), session=Session(files)).list_filters("GAIA", "GAIA3") == []


@row
def band_from_filter_id(t):
    for filter_id, band in [("Generic/Bessell", "Bessell"), ("J", "J")]:
        assert svo._band_from_filter_id(filter_id) == band


@row
def metadata_client_reads_svos_votable(t):
    record = SVOMetadataClient(session=Session(saved())).get_filter("GAIA/GAIA3.Grvs")
    assert (record.filter_id, record.detector_type, record.wavelength_unit, record.components) == (
        "GAIA/GAIA3.Grvs", "photon", "Angstrom", "Filter + CCD + Instrument")
    assert len(record.wavelength) == 32 and record.wavelength[0] == 8459.75 and np.all(np.diff(record.wavelength) > 0)
    assert 0 < record.transmission.max() <= 1


@row
def param_text(t):
    assert _param_text(b"Angstrom") == "Angstrom"  # astropy gives char PARAMs as str, or bytes in older versions


def probe_finds_a_contiguous_grid(t, present):
    grabber = svo_grid.SVOSpectraGrabber(base_dir=str(t.tmp))
    t.mp.setattr(grabber, "_head_exists", lambda model, fid: fid in present)
    grabber.expand_window, grabber.expand_max_gap = 12, 20
    assert grabber._bounded_probe("Kurucz") == present


PROBES = {"probe_finds_a_contiguous_grid": set(range(1, 61)),
          # a short block beside its one probed FID (100): it used to give {100}
          "probe_finds_a_contiguous_grid_short_block_beside_one_probed_fid": set(range(100, 106))}


@row
def mast_lists_every_bosz_resolution_and_accepts_each(t):
    grabber = mast.MASTSpectraGrabber(base_dir=str(t.tmp))
    names = grabber.discover_models()
    assert names == ["BOSZ-2024-r500", "BOSZ-2024-r1000", "BOSZ-2024-r2000", "BOSZ-2024-r5000",
                     "BOSZ-2024-r10000", "BOSZ-2024-r20000", "BOSZ-2024-r50000", "BOSZ-2024-rorig"]
    assert "Discovering spectral grids from MAST" in t.capsys.readouterr().out
    answer(t.mp, *[""] * len(names))
    assert [grabber.get_model_metadata(n)["reskey"] for n in names] == [n.rsplit("-", 1)[1] for n in names]


@row
def msg_takes_the_most_complete_order_when_neither_is_plausible(t):
    grabber = msg.MSGSpectraGrabber(base_dir=str(t.tmp), max_workers=1)
    grabber.model_axes["sg-Cool"] = {
        "axes": [np.array([100.0, 200.0]), np.array([4.0, 4.5]), np.array([-1.0, np.nan])], "shape": (2, 2, 2),
        "vlin": np.array([1]), "roles": {"teff": 0, "logg": 1, "meta": 2}}
    teff, logg, metal = grabber._params_from_index("sg-Cool", 0)
    # ... and loses to the order with more finite values
    np.testing.assert_array_equal([teff, logg, metal], (200.0, 4.0, -1.0))


# ======================================================================
# ZTF, HSC, BASS/MzLS, GMOS and SDSS
# ======================================================================
@row
def ztf_default_qe_is_multi_layer_and_coatings_are_variants(t):
    inst = t.tmp / "Palomar" / "ZTF"
    wl_nm = np.linspace(300, 1100, 81)
    files = {ztf.SINGLE: text_table(wl_nm, np.full(wl_nm.size, 0.6), "{:g},{:g}"),
             ztf.MULTI: text_table(wl_nm, np.full(wl_nm.size, 0.8), "{:g},{:g}")}
    for band, table in ztf.SPEC.filters.items():
        published_set(inst, band, WL, gauss(5000, 300) * 0.8, "filter+qe")
        files[table.url] = text_table(WL / 10.0, gauss(5000, 300) / 0.01, "{:g},{:g}")
    ztf.build(t.tmp, session=Session(files))
    m = InstrumentManifest.load(inst)
    assert m.component_file(QE, "g") == "components/qe/qe_multi.dat"
    assert set(m.variants["qe"]) == {"multi_layer", "single_layer", "mean"}
    assert m.variants["qe"]["mean"]["derived"] is True
    assert read_curve(inst / "components/qe/qe.dat").transmission.max() == pytest.approx(0.7)
    assert "filter" not in m.sources and m.sources["filter:i"].startswith(ztf.SPEC.filters["i"].url)
    single = compose_band(inst, "g", "filter+qe", variants={"qe": "single_layer"})
    assert single.transmission.max() == pytest.approx(0.6, rel=1e-3)


@row
def hsc_maps_r_i_to_r2_i2_and_publishes_user_filters(t):
    inst = t.tmp / "filters" / "Subaru" / "HSC"
    for band in hsc.BANDS:
        published_set(inst, band, WL, gauss(5000, 300) * 0.5, "filter+instrument+atmosphere")
    files = {}
    for name in set(hsc.CURRENT.values()) | set(hsc.FIRST.values()):
        scale = 0.9 if name.endswith("2") else 0.7
        files[f"{hsc.SUBARU}/wHSC-{name}.txt"] = "# wavelength transmittance\n" + text_table(
            WL, gauss(5000, 300) * scale, "{:g} {:g}")
    for band in hsc.USER_FILTERS:
        files[f"{hsc.SUBARU}/wHSC-{band}.txt"] = text_table(WL, gauss(5000, 20), "{:g} {:g}")
    files[f"{hsc.SUBARU}/wHSC-MBQ1.zip"] = zip_bytes({f"wHSC-{band}.txt": text_table(WL, gauss(5000, 50), "{:g} {:g}")
                                                      for band in hsc.MBQ1})
    flat = "".join(f"{w:g} 0.9\n" for w in np.linspace(3000, 11000, 81))
    for table in hsc.SPEC.optics + (hsc.SPEC.qe,):
        files[table.url] = flat if table.wl_scale == 1.0 else "".join(
            f"{w / 10:g} 0.9\n" for w in np.linspace(3000, 11000, 81))
    for _, _, table in hsc.SPEC.site.entries:
        files[table.url] = flat
    hsc.build(t.tmp / "filters", session=Session(files))

    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "r") == "components/filter/r2.dat"
    assert m.component_file(FILTER, "i") == "components/filter/i2.dat"
    assert m.variants["filter"]["first_r_i"]["files"]["r"] == "components/filter/r.dat"
    assert read_curve(inst / "components/filter/r.dat").transmission.max() == pytest.approx(0.7, rel=1e-3)
    assert {"NB387", "EB-gri", "MBQ1-413"} <= set(m.bands)
    nb = inst / "NB387.dat"
    assert released_mesa_reads(nb)
    assert read_filter_header(nb)["FilterID"] == "Subaru/HSC.NB387"
    assert m.published_bands["NB387"] == "filter"
    assert "NB387.dat" in (inst / "HSC").read_text().split()
    early = compose_band(inst, "r", "filter", variants={"filter": "first_r_i"})
    assert early.transmission.max() == pytest.approx(0.7, rel=1e-3)


BASS_TEXT = """# 90Prime SDSS g filter curve: ...
# wavelength (A)  res_noatm  respons_qe  res_qe_atm
4000.00 0.010 0.009 0.006
4500.00 0.900 0.810 0.648
5000.00 0.900 0.792 0.666
5500.00 0.020 0.017 0.015
"""


@row
def bass_builder_adds_filter_qe_and_qe(t):
    inst = t.tmp / "BOK" / "BASS"
    for band in legacy_surveys.BASS_BANDS:
        published_set(inst, band, WL, gauss(5000, 300) * 0.4, "filter+optics+qe+atmosphere")
        published_set(inst, f"{band}_filter", WL, gauss(5000, 300), "filter")
    files = {f"{legacy_surveys.FILES}/bass-{b}.txt": BASS_TEXT for b in legacy_surveys.BASS_BANDS}
    legacy_surveys.build(t.tmp, instrument="BASS", session=Session(files))
    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "g") == "g_filter.dat"
    assert m.component_file("filter+qe", "g") == "components/filter+qe/g.dat"
    assert m.component_file(QE, "r") == "components/qe/qe_r.dat"
    assert m.primary_bands() == ["g", "r"]


@row
def mzls_publishes_the_corrected_curve_with_the_filter(t):
    wl = np.linspace(8000, 10500, 26)
    files = {f"{legacy_surveys.FILES}/kpzd.txt": "# Area weighted\n" + text_table(wl, 98 * gauss(9200, 500, wl),
                                                                                  "{:.2f} {:.5f}"),
             f"{legacy_surveys.FILES}/kpzdccdcorr3.txt": text_table(wl, 0.5 * gauss(9200, 500, wl), "{:.1f} {:.5e}")}
    inst = legacy_surveys.build(t.tmp, instrument="MzLS", session=Session(files))
    m = InstrumentManifest.load(inst)
    assert m.bands == ["z"] and m.published_bands["z"] == "filter+optics+qe+atmosphere"
    assert read_curve(inst / m.component_file(FILTER, "z")).transmission.max() == pytest.approx(0.98, rel=1e-3)
    assert read_filter_header(inst / "z.dat")["FilterID"] == "KPNO/MzLS.z"


@row
def gmos_qe_epochs_are_variants_without_default(t):
    inst = t.tmp / "Gemini" / "GMOS_N"
    published_set(inst, "g", WL, gauss(5000, 300), "filter", filter_id="Gemini/GMOS-N.g")
    nm = np.linspace(300, 1100, 17)
    files = {}
    for table, _, _ in gmos.QE_VARIANTS["GMOS_N"].values():
        cols = table.cols[1] + 1
        files[table.url] = "".join(" ".join([f"{w:g}"] + ["0.8"] * (cols - 1)) + "\n" for w in nm)
    gmos.build(t.tmp, instrument="GMOS_N", session=Session(files))
    m = InstrumentManifest.load(inst)
    assert QE not in m.components
    assert set(m.variants["qe"]) == {"eev", "e2v_dd", "hamamatsu_bb", "hamamatsu_hsc"}
    assert "October 2011" in m.variants["qe"]["e2v_dd"]["epoch"]
    with pytest.raises(ComponentError):
        compose_band(inst, "g", "filter+qe")
    assert compose_band(inst, "g", "filter+qe", variants={"qe": "eev"}).transmission.max() == pytest.approx(0.8)


@row
def sdss_camera_response_left_out_of_build_all_products_is_named(t):
    # The primed filters are bands of their own (6c8739a), so filter+optics+qe
    # covers u ... z only and build_all_products builds no product for it (0.4.1
    # built one). It must say so and give the call that builds it.
    inst = t.tmp / "SLOAN" / "SDSS"
    for band in sdss.BANDS:
        published_set(inst, band, WL, gauss(5000, 300) * 0.4, "filter+instrument+atmosphere")
    published_set(inst, "gprime_filter", WL, gauss(5000, 300), "filter")
    files = {f"{sdss.DOI}/{b}{c}.dat": text_table(WL[::10], np.full(WL[::10].size, 0.1 * c), "{:g} {:g}")
             for b in sdss.BANDS for c in sdss.CAMCOLS}
    files[sdss.GUNN] = fits_file(*[fits_table(
        [(n, None, v) for n, v in (("wavelength", WL), ("respt", gauss(5000, 300) * 0.4),
                                   ("resbig", gauss(5000, 300) * 0.4), ("resnoa", gauss(5000, 300) * 0.5),
                                   ("xatm", np.full(WL.size, 0.85)))], name=b.upper(), fmt="E") for b in sdss.BANDS])
    sdss.build(t.tmp, session=Session(files))
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.filter_components"):
        built = build_all_products(inst)
    assert built == []
    assert not (inst / "products").exists()
    bands = sorted(sdss.BANDS)
    assert ("SLOAN/SDSS: filter+optics+qe not built, because only 5 of 6 bands have the components "
            "(none for gprime_filter)") in t.caplog.text
    assert f"build_product({str(inst)!r}, 'filter+optics+qe', bands={bands!r})" in t.caplog.text

    product = build_product(inst, "filter+optics+qe", bands=bands)
    assert product.name == "filter+optics+qe"
    assert (product / product.name).read_text().split() == [f"{b}.dat" for b in bands]
    assert all(released_mesa_reads(product / f"{b}.dat") for b in bands)


# ======================================================================
# No silent failures: CALDB mirrors, SkyCalc, bands without a published curve
# ======================================================================
@row
def caldb_names_every_mirror_when_all_fail(t):
    primary, mirror = (root + swift.FTRANS for root in swift.CALDB_ROOTS)
    fetch = Fetcher(Session({primary: b"not fits"}))  # the mirror answers 404
    with pytest.raises(RuntimeError) as err:
        swift.caldb_get(fetch, swift.FTRANS)
    assert f"{primary}: not a FITS file" in str(err.value)
    assert f"{mirror}: 404" in str(err.value)


def skytable(fields, lam, trans, version="2.0.9"):
    """SkyCalc's skytable.fits, echoing ``fields`` in its header."""
    return fits_file(fits_table([("lam", None, lam), ("trans", None, trans)]),
                     comments=[f"SkyCalc version: {version}", *(f"{k}={v}" for k, v in fields.items())])


class SkyCalc:
    """ESO SkyCalc. Its API host is down unless ``api_up``; its web form answers ``form_html``, linking a
    skytable.fits that echoes the form's inputs (``echo_override`` replaces some): 200 samples between the
    asked wavelengths, transmission 0.8 ** airmass."""

    LINK = "<h4><a href=/observing/etc/tmp/abc/skytable.fits> skytable.fits</a></h4>"

    def __init__(self, echo_override=None, *, api_up=False, form_html=LINK, rmtmp_error=None, api_output=None):
        self.echo_override, self.api_up, self.form_html = echo_override or {}, api_up, form_html
        self.rmtmp_error, self.api_output, self.posts, self.api_gets = rmtmp_error, api_output, [], 0

    def post(self, url, data=None, json=None, timeout=None, **kw):
        if json is None:
            self.posts.append(dict(data))
            return Resp(self.form_html, encoding="latin-1")
        if not self.api_up:
            raise requests.exceptions.ConnectTimeout("api host down")
        return Resp('{"status": "success", "tmpdir": "t1"}')

    def get(self, url, timeout=None, **kw):
        if url.endswith("/api/rmtmp"):
            if self.rmtmp_error is not None:
                raise self.rmtmp_error
            return Resp("{}")
        if "/tmp/t1/" in url:
            self.api_gets += 1
            lam = np.geomspace(1000.0, 1100.0, 50)
            return Resp(self.api_output or skytable({}, lam, np.full_like(lam, 0.8)))
        fields = {**self.posts[-1], **self.echo_override}
        lo, hi = float(fields["SKYMODEL.WAVELENGTH.MIN"]), float(fields["SKYMODEL.WAVELENGTH.MAX"])
        lam = np.geomspace(lo, hi, 200)
        return Resp(skytable(fields, lam, np.full_like(lam, 0.8 ** float(fields["SKYMODEL.TARGET.AIRMASS"]))),
                    encoding="latin-1")


def no_wait(t):
    t.mp.setattr(skycalc.time, "sleep", lambda seconds: None)


@row
def skycalc_form_echo_is_checked_after_a_fallback(t):
    no_wait(t)
    client = skycalc.SkyCalcClient(session=SkyCalc({"SKYMODEL.TARGET.AIRMASS": "1.50"}), interface="auto", pause=0)
    with pytest.raises(RuntimeError, match="echoed SKYMODEL.TARGET.AIRMASS"):
        client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)


@row
def skycalc_failure_after_a_fallback_names_both_errors(t):
    no_wait(t)
    session = SkyCalc(form_html="<p>Service unavailable</p>")
    client = skycalc.SkyCalcClient(session=session, interface="auto", pause=0, retries=3)
    with pytest.raises(RuntimeError) as err:
        client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    message = str(err.value)
    assert "returned no skytable.fits" in message and "Service unavailable" in message  # the form's error
    assert "api host down" in message and skycalc.BASE_URL in message   # and why the form was used
    assert len(session.posts) == 2  # one attempt went to the switch
    # An output that is not FITS reruns SkyCalc: one download per attempt (retried at one layer only)
    session = SkyCalc(api_up=True, api_output=b"<html>busy</html>")
    with pytest.raises(RuntimeError, match=r"failed after 3 attempts: SkyCalc output \S+ was not a FITS file"):
        skycalc.SkyCalcClient(session=session, interface="api", pause=0, retries=3).fetch(
            "paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    assert session.api_gets == 3


@row
def skycalc_api_temporary_directory_left_behind_is_warned(t):
    no_wait(t)
    session = SkyCalc(api_up=True, rmtmp_error=requests.exceptions.ReadTimeout("rmtmp timed out"))
    client = skycalc.SkyCalcClient(session=session, interface="api", pause=0)
    with t.caplog.at_level(logging.WARNING, logger=skycalc.__name__):
        result = client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    assert result.interface == "api" and result.version == "2.0.9"
    assert "Could not delete SkyCalc tmpdir t1" in t.caplog.text and "rmtmp timed out" in t.caplog.text


@row
def skycalc_version_change_leaves_no_unregistered_point(t):
    no_wait(t)
    site = t.tmp / "atmospheres" / "paranal"
    site.mkdir(parents=True)
    AtmosphereGrid(site="paranal", entries=[{"airmass": 1.0, "pwv": 2.5, "file": "am1.00_pwv2.5.dat"}],
                   model_version="2.0.8").save(site)
    write_filter_file(site / "am1.00_pwv2.5.dat", [1000.0, 30000.0], [0.9, 0.9], detector_type=ENERGY,
                      source="component", wavelength_unit="Angstrom")
    client = skycalc.SkyCalcClient(session=SkyCalc(), interface="form", pause=0)
    with pytest.raises(RuntimeError, match=r"version changed .*'2.0.8', '2.0.9'"):
        skycalc.build(t.tmp / "filters", sites=["paranal"], airmasses=[1.0, 2.0], pwvs=[2.5],
                      fetch_resolution=2000, store_resolution=500, chunks=2, client=client)
    # Nothing made with 2.0.9 is left for a later run to register under the grid's 2.0.8.
    assert not (site / "am2.00_pwv2.5.dat").exists()
    grid = AtmosphereGrid.load(site)
    assert [e["file"] for e in grid.entries] == ["am1.00_pwv2.5.dat"] and grid.model_version == "2.0.8"


@row
def twomass_warns_about_bands_without_a_published_curve(t):
    inst = t.tmp / "2MASS" / "2MASS"
    published(inst, "J", np.linspace(10600, 14600, 30), np.ones(30), "filter+instrument+atmosphere", ENERGY,
              source=None)
    curve = (np.linspace(10000.0, 25000.0, 31), np.full(31, 0.9))
    t.mp.setattr(twomass, "fetch_tables", lambda fetch, numbers: {n: curve for n in numbers})
    with t.caplog.at_level(logging.WARNING, logger=twomass.__name__):
        twomass.build(t.tmp, session=Session({}))
    assert "no published curve for ['H', 'Ks']" in t.caplog.text
    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "J") == "components/filter/J.dat" and "H" not in m.components[FILTER]


# ======================================================================
# Builders that used to skip a band or file and still report success now stop
# ======================================================================
@row
def ensure_published_raises_for_ids_svo_lacks_and_writes_no_index(t):
    inst = t.tmp / "Fac" / "Cam"
    client = SVOClient({"Fac/Cam.g": record("Fac/Cam.g")}.get)
    with pytest.raises(RuntimeError, match="SVO has no Fac/Cam.r"):
        ensure_published(inst, {"g": "Fac/Cam.g", "r": "Fac/Cam.r"}, client)
    assert (inst / "g.dat").is_file() and not (inst / "r.dat").exists()
    assert not (inst / "Cam").exists()  # no MESA index for an incomplete set


@row
def ensure_published_writes_index_when_complete(t):
    inst = t.tmp / "Fac" / "Cam"
    ensure_published(inst, {"g": "Fac/Cam.g"}, SVOClient({"Fac/Cam.g": record("Fac/Cam.g")}.get))
    assert (inst / "Cam").read_text() == "g.dat\n"


@row
def build_simple_refuses_a_filter_band_without_a_published_curve(t):
    inst = t.tmp / "Fac" / "Cam"
    published(inst, "g", source=None)
    spec = SimpleSource("Fac", "Cam", filters={"g": Table("f"), "r": Table("f")})
    with pytest.raises(FileNotFoundError, match="no published curve for r"):
        build_simple(spec, t.tmp, session=Session({"f": "4000 0.5\n5000 0.8\n"}), svo_client=SVOClient({}.get))
    assert not (inst / "manifest.json").exists() and not (inst / "components").exists()


@row
def build_simple_refuses_a_composite_band_without_a_published_curve(t):
    inst = t.tmp / "Fac" / "Cam"
    published(inst, "g", source=None)
    spec = SimpleSource("Fac", "Cam", filters={},
                        composites={"z": ("filter+qe", lambda session: (np.array([4000.0, 5000.0]),
                                                                        np.array([0.5, 0.5])))})
    with pytest.raises(FileNotFoundError, match="no published curve for z"):
        build_simple(spec, t.tmp, session=Session({}), svo_client=SVOClient({}.get))
    assert not (inst / "manifest.json").exists()


def hst_tables():
    """ACS/SBC graph: F122M splits into ota x filter x mama; FR388N has no chain."""
    t = hst.SynphotTables.__new__(hst.SynphotTables)
    t.tmg_name, t.tmc_name = "x_tmg.fits", "x_tmc.fits"
    t.innode, t.outnode = np.array([1, 2, 3, 3]), np.array([2, 3, 4, 4])
    t.keyword = np.array(["acs", "sbc", "f122m", "default"])
    t.compname = np.array(["hst_ota", "acs_sbc_mama", "acs_f122m", "clear"])
    names = ("hst_ota", "acs_sbc_mama", "acs_f122m")
    t.files = {n: f"acscomp${n}_001_syn.fits" for n in names}
    wl = np.linspace(1000.0, 3000.0, 21)
    t._get = {f"{hst.TRDS}/comp/acs/{n}_001_syn.fits": throughput(wl, np.full(21, 0.5)) for n in names}.__getitem__
    t._cache = {}
    return t


def hst_published(root, bands=("F122M", "FR388N")):
    inst = root / "HST" / "ACS_SBC"
    for band in bands:
        published(inst, band, source=None)
    return inst


@row
def hst_build_records_bands_it_cannot_decompose_in_the_notes(t):
    inst = hst_published(t.tmp)
    with t.caplog.at_level("INFO", logger="sed_tools.component_sources.hst"):
        assert hst.build(t.tmp, instruments=["ACS_SBC"], tables=hst_tables()) == [inst]
    assert "1 bands not decomposed" in t.caplog.text and "FR388N" in t.caplog.text
    assert not [r for r in t.caplog.records if r.levelname == "WARNING"]  # expected for ramp filters
    manifest = InstrumentManifest.load(inst)
    assert "Not decomposed: FR388N." in manifest.notes
    assert set(manifest.components["filter"]) == {"F122M"}


@row
def hst_build_raises_after_building_the_others_for_instruments_without_curves(t):
    inst = hst_published(t.tmp)
    with pytest.raises(FileNotFoundError, match="no published curves for ACS_WFC.*Built: ACS_SBC"):
        hst.build(t.tmp, instruments=["ACS_SBC", "ACS_WFC"], tables=hst_tables())
    assert (inst / "manifest.json").is_file()
    assert not (t.tmp / "HST" / "ACS_WFC").exists()  # no empty directory left behind


@row
def hst_build_refuses_an_unreadable_published_curve(t):
    inst = hst_published(t.tmp, bands=())
    inst.mkdir(parents=True)
    (inst / "F122M.dat").write_text("Wavelength,Transmission\n1500,0.1\nbad,row\n")
    with pytest.raises(ValueError, match="F122M.dat: cannot read the published curve"):
        hst.build(t.tmp, instruments=["ACS_SBC"], tables=hst_tables())


def pandeia_raw(root):
    """Unpacked Pandeia reference data with the NIRCam F200W imaging factors."""
    def table(path, wl, tr):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fits_file(fits_table([("wavelength", "microns", wl), ("throughput", None, tr)])))

    wl = np.linspace(1.0, 3.0, 11)
    (root / "jwst" / "telescope").mkdir(parents=True)
    (root / "jwst" / "telescope" / "config.json").write_text(json.dumps({"paths": {"otefile": "ote.fits"}}))
    table(root / "jwst" / "telescope" / "ote.fits", wl, np.full(11, 0.9))
    paths = {"f200w": "filters/f200w.fits", "internal": "optical/internal.fits",
             "dbs_sw": "optical/dbs_sw.fits", "qe_sw": "qe/qe_sw.fits"}
    (root / "jwst" / "nircam").mkdir(parents=True)
    (root / "jwst" / "nircam" / "config.json").write_text(json.dumps({"paths": paths}))
    for i, rel in enumerate(paths.values()):
        table(root / "jwst" / "nircam" / rel, wl[i:], np.linspace(0.5, 0.8, 11 - i))
    return root


@row
def jwst_build_raises_after_building_the_others_for_instruments_without_curves(t):
    from sed_tools.component_sources import jwst

    raw = pandeia_raw(t.tmp / "pandeia")
    inst = t.tmp / "filters" / "JWST" / "NIRCam"
    published(inst, "F200W", source=None)
    with pytest.raises(FileNotFoundError, match="no published curves for MIRI.*Built: NIRCam"):
        jwst.build(t.tmp / "filters", raw=raw, instruments=["NIRCam", "MIRI"])
    assert set(InstrumentManifest.load(inst).components) == {"filter", "optics", "qe"}


@row
def tabular_raw_inventory_warns_when_a_file_is_downloaded_instead(t):
    from sed_tools.component_sources import tabular

    folder = t.tmp / "raw" / tabular.RAW_GROUPS["PANIC"]
    folder.mkdir(parents=True)
    scan = b"400 50\n500 80\n"
    *saved_files, (band, missing) = tabular.PANIC_FILES.items()
    for _, filename in saved_files:
        (folder / filename).write_bytes(scan)
    url = tabular.PANIC.filters[band].url
    session = Session({url: scan})
    with t.caplog.at_level("WARNING", logger="sed_tools.component_sources.tabular"):
        tabular.build(t.tmp / "filters", instrument="PANIC", raw=t.tmp / "raw", session=session)
    assert session.calls == [url]
    assert f"{missing} is not in the saved inventory" in t.caplog.text
    assert (t.tmp / "filters" / "CAHA" / "PANIC" / f"{band}.dat").is_file()


# ======================================================================
# The observatories' Keck workbooks
# ======================================================================
@row
def keck_workbooks_need_xlrd(t):
    xlrd_stand_in(t)
    t.mp.setitem(sys.modules, "xlrd", None)  # not installed
    with pytest.raises(RuntimeError, match="needs xlrd"):
        obs.build(t.tmp, instrument="NIRC2", session=Session(saved()))
    assert not (t.tmp / "Keck").exists()


# ======================================================================
# Space: Euclid NISP, Roman WFI, HST's pinned releases, JWST's Pandeia data, the catalog
# ======================================================================
def nisp_table(neg=False):
    wave = np.arange(1000.0, 1011.0)
    filt = np.full(wave.size, 0.9)
    filt[0] = -1e-5 if neg else 1e-5
    tel, nioa, qe = np.full(wave.size, 0.88), np.full(wave.size, 0.95), np.full(wave.size, 0.93)
    total = filt * tel * nioa * qe
    rows = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in zip(wave, total, filt, tel, nioa, qe))
    return "# WAVE T_TOTAL T_FILTER T_TELESCOPE T_NI-OA T_QE\n" + rows + "\n", wave, total


@row
def euclid_parse_checks_columns_and_product(t):
    from sed_tools.component_sources import euclid

    text, wave, total = nisp_table(neg=True)
    cols = euclid.parse_table(text)
    wl, curves, negative = euclid.factors(cols)
    np.testing.assert_allclose(wl, wave * 10.0)
    assert negative == 1 and curves[FILTER].min() == 0.0
    np.testing.assert_allclose(curves[OPTICS], 0.88 * 0.95)
    with pytest.raises(ValueError, match="columns"):
        euclid.parse_table(text.replace("T_NI-OA", "T_NIOA"))
    bad = text.splitlines()
    bad[3] = bad[3].replace(bad[3].split()[1], "0.5", 1)
    with pytest.raises(ValueError, match="multiply"):
        euclid.parse_table("\n".join(bad))


@row
def euclid_build_writes_components_and_compares(t):
    from sed_tools.component_sources import euclid

    text, wave, total = nisp_table()
    inst = t.tmp / "Euclid" / "NISP"
    published(inst, "Y", wave * 10.0, total * 0.6, "filter+optics+qe", source=None)
    published(inst, "Blue", wave * 10.0, total, components="filter+instrument", source=None)
    (inst / "NISP").write_text("Blue.dat\nY.dat\n")  # as NJM serves the set
    euclid.build(t.tmp, session=Session({euclid.URL.format(band=b): text for b in euclid.BANDS}))
    m = InstrumentManifest.load(inst)
    assert {r: m.components[r] for r in (FILTER, OPTICS, QE)} == {
        r: {b: f"components/{r}/{b}.dat" for b in euclid.BANDS} for r in (FILTER, OPTICS, QE)}
    assert "Release mismatch" in m.notes and "Blue" in m.notes
    assert all(released_mesa_reads(inst / rel) for rel in (m.components[r]["Y"] for r in (FILTER, OPTICS, QE)))


@row
def euclid_build_writes_missing_published_curves(t):
    from sed_tools.component_sources import euclid

    text, wave, total = nisp_table()
    lines = text.splitlines()
    lines[1] = lines[1].replace(lines[1].split()[1], "-1e-07", 1).replace(lines[1].split()[2], "-1e-07", 1)
    text = "\n".join(lines) + "\n"
    inst = t.tmp / "Euclid" / "NISP"
    published(inst, "Y_svo", wave * 10.0, total * 0.6, "filter+optics+qe", source=None)
    # NJM keeps SVO's copy beside the set, outside its bands and index file.
    archived = {"Y_svo": {"detector_type": "photon", "detector_type_source": "svo", "published": "filter+optics+qe"}}
    InstrumentManifest("Euclid", "NISP", bands=[], archived_bands=archived).save(inst)
    euclid.build(t.tmp, session=Session({euclid.URL.format(band=b): text for b in euclid.BANDS}))
    header = read_filter_header(inst / "Y.dat")
    assert header["FilterID"] == "Euclid/NISP.Y" and header["DetectorType"] == "photon"
    assert header["DetectorTypeSource"] == "original" and header["Components"] == "filter+optics+qe"
    assert released_mesa_reads(inst / "Y.dat")
    assert (inst / "NISP").read_text() == "Y.dat\nJ.dat\nH.dat\n"
    m = InstrumentManifest.load(inst)
    assert m.bands == ["H", "J", "Y"] and "reproduces" in m.notes and "Y_svo are SVO's copies" in m.notes
    assert m.archived_bands == archived  # the archive copy never goes back into bands
    assert "negative samples set to 0: Y 1" in m.notes


def roman_qe(values):
    head = ["# %ECSV 1.0", "# ---", "# datatype:",
            "# - {name: wavelength, unit: nm, datatype: float64, description: wavelength}"]
    head += [f"# - {{name: SCU {i}, unit: '', datatype: float64, description: Sensor Control Unit Number {i}}}"
             for i in range(1, 19)]
    head += ["# delimiter: ','", "# schema: astropy-2.0", "wavelength," + ",".join(f"SCU {i}" for i in range(1, 19))]
    rows = [f"{w}," + ",".join(f"{v:.4f}" for v in values[k]) for k, w in enumerate([800.0, 1000.0, 2000.0])]
    return "\n".join(head + rows) + "\n"


def roman_summary(below, above):
    return ("# %ECSV 1.0\n# ---\nwavelength range, mean quantum efficiency, median quantum efficiency\n"
            f"<900, {below}, {below}\n900 <= wave <= 2200, {above}, {above}\n")


@row
def roman_qe_is_the_detector_mean(t):
    from sed_tools.component_sources import roman

    rng = np.random.default_rng(1)
    values = np.round(np.clip(np.array([0.7, 0.88, 0.9])[:, None] + rng.normal(0, 0.02, (3, 18)), 0, 1), 4)
    qe_text = roman_qe(values)
    wl, qe = roman.parse_qe(qe_text)
    assert qe.shape == (18, 3)
    roman.check_summary(roman_summary(values[0].mean(), values[1:].mean()), wl, qe)
    with pytest.raises(ValueError, match="summary"):
        roman.check_summary(roman_summary(0.5, 0.5), wl, qe)
    with pytest.raises(ValueError, match="unit"):
        roman.parse_qe(qe_text.replace("unit: nm", "unit: um"))

    inst = t.tmp / "Roman" / "WFI"
    published(inst, "F062", np.array([4720.0, 6000.0, 7700.0]), np.array([0.1, 0.5, 0.1]), "filter+optics+qe",
              source=None)
    roman.build(t.tmp, session=Session({roman.QE_URL: qe_text,
                                        roman.SUMMARY_URL: roman_summary(values[0].mean(), values[1:].mean())}))
    m = InstrumentManifest.load(inst)
    assert m.components == {QE: {SHARED: "components/qe/qe.dat"}}
    got = np.loadtxt(inst / "components/qe/qe.dat", delimiter=",", skiprows=1)
    np.testing.assert_allclose(got[:, 0], [8000.0, 10000.0, 20000.0])
    np.testing.assert_allclose(got[:, 1], values.mean(1))
    assert "F062" in m.notes and "mean of the 18 detectors" in m.notes and m.buildable() == []


@row
def hst_pins_replace_only_the_named_component(t):
    tables = hst.SynphotTables.__new__(hst.SynphotTables)
    tables.tmg_name = "x_tmg.fits"
    tables.files = {"acs_f115lp_tds_def": "acscomp$acs_f115lp_tds_def_010_syn.fits",
                    "acs_sbc_mama": "acscomp$acs_sbc_mama_013_syn.fits"}
    tables._cache = {}
    wl = np.array([1150.0, 1500.0, 2000.0])
    tables._get = {
        f"{hst.TRDS}/comp/acs/acs_f115lp_tds_def_010_syn.fits": throughput(wl, [0.875, 0.875, 0.875]),
        f"{hst.TRDS}/comp/acs/acs_f115lp_tds_def_004_syn.fits": throughput(wl, [0.909, 0.909, 0.909]),
        f"{hst.TRDS}/comp/acs/acs_sbc_mama_013_syn.fits": throughput(wl, [0.1, -0.01, 0.2]),
    }.__getitem__
    pins = {"acs_f115lp_tds_def": "acs_f115lp_tds_def_004_syn.fits"}
    assert tables.throughput("acs_f115lp_tds_def", pins)[1][0] == pytest.approx(0.909)
    assert tables.throughput("acs_f115lp_tds_def")[1][0] == pytest.approx(0.875)
    assert tables.throughput("acs_sbc_mama", pins)[1][1] == 0.0  # negative noise set to 0
    assert "pinned" in tables.source("acs_f115lp_tds_def", pins) and "pinned" not in tables.source("acs_sbc_mama", pins)
    note = hst._pin_note(tables, pins, ["acs_f115lp_tds_def", "acs_sbc_mama"])
    assert note == "acs_f115lp_tds_def_010_syn.fits = 0.9626 x acs_f115lp_tds_def_004_syn.fits"
    _, prod = hst._product(tables, ["acs_f115lp_tds_def", "acs_sbc_mama"], pins)
    assert prod[-1] == pytest.approx(0.909 * 0.2)


@row
def jwst_refdata_is_checked_and_filtered(t):
    from sed_tools.component_sources import jwst

    assert jwst.REFDATA_VERSION == "2026.7" and "v2026p7" in jwst.REFDATA_URL
    blob = tar_gz({"jwst/telescope/config.json": b"{}", "jwst/nircam/psfs/big.fits": b"x",
                   "jwst/nircam/filters/f200w.fits": b"y", "devtools/tool.py": b"z"}, "pandeia_data-2026.7-jwst")
    session = Session({"u": blob})
    root = jwst.fetch_refdata(t.tmp, url="u", session=session, sha1=hashlib.sha1(blob).hexdigest())
    assert root == t.tmp / "pandeia_data-2026.7-jwst"
    assert (root / "jwst/nircam/filters/f200w.fits").is_file()
    assert not (root / "jwst/nircam/psfs/big.fits").exists() and not (root / "devtools").exists()
    with pytest.raises(RuntimeError, match="SHA-1"):
        jwst.fetch_refdata(t.tmp / "b", url="u", session=session)


@row
def jwst_read_rejects_non_micron_wavelengths(t):
    from sed_tools.component_sources.jwst import _read

    path = t.tmp / "t.fits"
    path.write_bytes(fits_file(fits_table([("WAVELENGTH", "ANGSTROM", [1.0, 2.0]), ("THROUGHPUT", None, [0.5, 0.6])])))
    with pytest.raises(ValueError, match="microns"):
        _read(path)


@row
def status_of_2mass_and_ps1_and_new_builders(t):
    info = catalog.component_availability("PAN-STARRS", "PS1")
    assert info["status"] == catalog.NOT_BUILT, info
    assert "Table 3" in catalog.status_for("PAN-STARRS", "PS1")[1]
    # 2MASS's separate tables are built since PR #48 (component_sources.twomass).
    info = catalog.component_availability("2MASS", "2MASS")
    assert info["status"] == "supported" and info["builder"] == "twomass", info
    info = catalog.component_availability("Roman", "WFI")
    assert info["status"] == "supported" and info["builder"] == "roman"
    info = catalog.component_availability("Euclid", "NISP")
    assert info["status"] == "supported" and info["builder"] == "euclid"
    # Pandeia 2026.7 is 21.6 MB: JWST components no longer need allow_heavy.
    assert not any(catalog.builder_for("JWST", i).heavy for i in ("NIRCam", "NIRISS", "MIRI"))


# ======================================================================
# Swift/UVOT, LCO CSP/CSP2 and Cassini ISS
# ======================================================================
SWIFT_GRID = np.arange(1600.0, 8001.0, 10.0)


def uvot_files(root):
    """Swift/UVOT's CALDB files under ``root``: filter transmissions, mirror and instrument areas,
    sensitivity corrections and the effective-area files, in single precision as CALDB writes them."""
    grid = SWIFT_GRID
    instea = 60.0 * gauss(3000.0, 1500.0, grid)
    filt = {f: 0.9 * gauss(c, 300.0, grid) for f, c in
            (("U", 3500), ("B", 4400), ("V", 5400), ("UVW1", 2600), ("UVW2", 1950), ("UVM2", 2250), ("WHITE", 3800))}
    files = {
        root + swift.FTRANS: fits_file(*[
            fits_table([("ENERGY", "eV", 12398.4 / grid), ("WAVELENGTH", "angstrom", grid), ("TRANSMIS", "", t)],
                       "TRANSMISSION" + f, {"FILTER": f}, "E") for f, t in filt.items()]),
        root + swift.MIRROR: fits_file(fits_table([
            ("ENERGY", "eV", 12398.4 / grid[1:441]), ("WAVELENGTH", "angstrom", grid[1:441]),
            ("EA", "cm**2", np.full(440, 590.0))], "MIRROREFFAREA", fmt="E")),
        root + swift.INSTEA: fits_file(fits_table([
            ("ENERGY", "eV", 12398.4 / grid), ("WAVELENGTH", "angstrom", grid), ("EA", "cm**2", instea)],
            "INSTEFFAREA", fmt="E")),
        root + swift.SENSCORR: fits_file(*[
            fits_table([("TIME", "s", [1.2e8, 7.9e8]), ("OFFSET", "", [0.0, 0.2]), ("SLOPE", "", [0.01, 0.0])],
                       "SENSCORR" + f, {"MJDREFI": 51910, "MJDREFF": 7.4287037e-4}, "E") for f in filt]),
    }
    for band, (f, arf) in swift.BANDS.items():
        scale = 0.9 if band == "white" else 1.0
        files[root + "cpf/arf/" + arf] = fits_file(fits_table([
            ("ENERG_LO", "keV", 12.3984 / (grid[::-1] + 5)), ("ENERG_HI", "keV", 12.3984 / (grid[::-1] - 5)),
            ("WAVE_MIN", "angstrom", grid[::-1] + 5), ("WAVE_MAX", "angstrom", grid[::-1] - 5),
            ("SPECRESP", "cm**2", (scale * filt[f] * instea)[::-1])], "SPECRESP" + f,
            {"FILTER": f, "CONTENT": "SPECRESP"}, "E"))
    return files


def pds_table(rows):
    return "PDS_VERSION_ID = PDS3\nEND\n\\begindata\n" + "".join(" ".join(f"{x:.6f}" for x in r) + "\n" for r in rows)


def cassini_files(camera):
    """Cassini ISS's QE, optics and system-transmission tables for ``camera``; also the grid and each band's
    filter pair."""
    nm = np.arange(200.0, 1057.05, 0.1)
    qe = np.clip(0.5 - np.abs(nm - 650.0) / 1000.0, 0.01, None)
    optics = np.full(nm.size, 0.6)
    grid = np.arange(200.0, 1101.0)
    files = {cassini.QE_URL: pds_table(np.c_[grid, np.interp(grid, nm, qe), np.interp(grid, nm, qe)]),
             cassini.EFFICIENCY + cassini.CAMERAS[camera][1]: pds_table(np.c_[np.arange(200.0, 1100.1, 2.5),
                                                                              np.full(361, 0.6)])}
    pairs = {}
    for i, band in enumerate(cassini.BANDS[camera]):
        centre = 1000.0 if band == "IR4" else 300.0 + 15.0 * i
        pairs[band] = 0.9 * np.exp(-0.5 * ((nm - centre) / 20.0) ** 2)
        system = qe * optics * pairs[band]
        files[cassini.systrans_url(camera, band)] = pds_table(np.c_[nm, system, system])
    return files, nm, pairs


@row
def uvot_falls_back_to_the_mirror_and_keeps_one_absolute_scale(t):
    session = Session(uvot_files(swift.CALDB_ROOTS[1]))  # the primary host serves nothing
    out = swift.build(t.tmp, instrument="UVOT", session=session)
    assert session.calls[0].startswith(swift.CALDB_ROOTS[0])
    m = InstrumentManifest.load(out)
    assert sorted(m.bands) == sorted(swift.BANDS)
    assert m.outside_range == "error" and m.ground_based is False
    assert m.components["optics"] == {"*": "components/optics/mirror.dat"}
    assert m.components["optics+qe"] == {"*": "components/optics+qe/instrument.dat"}
    h = read_filter_header(out / "U.dat")
    assert h["DetectorType"] == "photon" and h["Components"] == "filter+optics+qe"
    assert swift.CALDB_ROOTS[1] in h["Sources"]
    u = np.loadtxt(out / "U.dat", delimiter=",", skiprows=1)
    assert np.all(np.diff(u[:, 0]) > 0) and u[0, 0] == 1600.0
    assert u[:, 1].max() < 60.0 / swift.GEOMETRIC_AREA_CM2 + 1e-6  # cm^2 / 659 cm^2
    # Composition reproduces the published shape; white differs only by a constant.
    for band in ("U", "white"):
        comp = compose_band(out, band, "filter+optics+qe")
        pub = np.loadtxt(out / f"{band}.dat", delimiter=",", skiprows=1)
        ratio = np.interp(pub[:, 0], comp.wavelength, comp.transmission)[pub[:, 1] > 0.01] / pub[:, 1][pub[:, 1] > 0.01]
        assert np.allclose(ratio, 1.0 if band == "U" else 1 / 0.9, rtol=1e-5)
    with pytest.raises(ComponentCoverageError):  # mirror stops at 6000 A
        compose_band(out, "V", "filter+optics")
    assert "1+OFFSET=1.2000" in m.notes and "NOT applied" in m.notes
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


@row
def uvot_rejects_a_wrong_filter_extension(t):
    files = uvot_files(swift.CALDB_ROOTS[0])
    v = swift.BANDS["V"][1]
    files[swift.CALDB_ROOTS[0] + "cpf/arf/" + v] = files[swift.CALDB_ROOTS[0] + "cpf/arf/" + swift.BANDS["B"][1]]
    with pytest.raises(ValueError, match="FILTER"):
        swift.uvot(Fetcher(Session(files)))


def csp_package():
    two = " {:.1f} {:.8f}"
    members = {"README": "These final fitlers have atmosphere corresponding to 1.2 airmass.\n"}
    opt = {"u": 3550, "g": 4750, "r": 6200, "i": 7600, "B": 4350, "V_LC3014": 5400, "V_LC3009": 5400,
           "V_LC9844": 5400}
    atm_wl = np.array([2900.0, 4000.0, 6000.0, 70000.0])
    atm_tr = np.array([0.3, 0.7, 0.9, 1.0])
    members["atm_ext_ctio.dat"] = text_table(atm_wl, atm_tr, two)
    members["atm_ext_lines.dat"] = text_table([2950.0, 70000.0], [1.0, 1.0], two)
    for band, c in opt.items():
        wl = np.arange(c - 900.0, c + 901.0, 20.0)
        raw = 0.8 * np.exp(-0.5 * ((wl - c) / 120.0) ** 2)
        members[f"{band}_texas_WLcorr.txt"] = text_table(wl, raw, two)
        members[f"{band}_tel_ccd_atm_ext_1.2.dat"] = text_table(wl, raw * np.interp(wl, atm_wl, atm_tr) ** 1.2, two)
    for name, c in (("Y_SWO", 10350), ("Jrc1_SWO", 12400), ("Jrc2_SWO", 12350), ("H_SWO", 16300),
                    ("Y_DUP", 10400), ("J_DUP", 12380), ("H_DUP", 16280)):
        wl = np.arange(c - 1500.0, c + 1501.0, 4.0)
        members[f"{name}_TAM_scan_atm.dat"] = text_table(wl, np.exp(-0.5 * ((wl - c) / 300.0) ** 2) - 0.001, two)
    nm = np.arange(300.0, 800.01, 0.5)
    members["Telescope.txt"] = "#Wavelength (nm)\n" + text_table(nm, np.full(nm.size, 0.012), two)
    nm = np.arange(302.0, 900.01, 1.0)
    members["Telescope+Camera.txt"] = text_table(nm, np.full(nm.size, 0.011), two)
    return tar_gz({**members, "._README": "resource fork"}, "CSP_filter_package")


@row
def csp1_epochs_atmosphere_and_per_band_components(t):
    out = csp.build(t.tmp, instrument="CSP", session=Session({csp.PACKAGE: csp_package()}))
    m = InstrumentManifest.load(out)
    assert out == t.tmp / "LCO" / "CSP"
    assert set(m.bands) == set(csp.CSP1_OPTICAL) | set(csp.CSP1_NIR)
    assert {"V_LC3014", "V_LC3009", "V_LC9844", "Jrc1", "Jrc2"} <= set(m.bands)
    h = read_filter_header(out / "Jrc2.dat")
    assert h["Components"] == "filter+optics+qe+atmosphere" and h["Atmosphere"] == "airmass=1.2"
    # Components only on the Swope optical bands, never shared with the NIR bands.
    for entries in m.components.values():
        assert "*" not in entries and not set(entries) & set(csp.CSP1_NIR)
    assert "i" not in m.components["optics"]  # Telescope.txt ends at 800 nm
    assert set(m.components["optics+qe"]) == set(csp.CSP1_OPTICAL)
    assert m.components["optics"]["u"] == "components/optics/telescope.dat"
    assert m.buildable() == []  # no filter-only curve is published
    assert m.default_site == csp.CSP_SITE
    comp = compose_band(out, "g", "filter+optics+qe")
    assert read_filter_header(out / m.components["filter+optics+qe"]["g"])["Components"] == "filter+optics+qe"
    assert comp.transmission.max() == pytest.approx(0.8, rel=1e-6)
    assert "g 1.000 (+-0.000)" in m.notes
    assert np.loadtxt(out / "Y.dat", delimiter=",", skiprows=1)[:, 1].min() == 0.0  # noise clipped
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


@row
def csp2_reads_the_linked_files(t):
    wl = np.arange(3000.0, 20000.0, 50.0)
    files = {csp.DATA + name: text_table(wl, np.exp(-0.5 * ((wl - 9000.0) / 3000.0) ** 2), " {:.1f} {:.8f}")
             for name in csp.CSP2_FILES.values()}
    out = csp.build(t.tmp, instrument="CSP2", session=Session(files))
    m = InstrumentManifest.load(out)
    assert sorted(m.bands) == sorted(csp.CSP2_FILES) and m.components == {}
    assert read_filter_header(out / "V.dat")["Atmosphere"] == "airmass=1.2"
    assert "Atmosphere" not in read_filter_header(out / "J.dat")  # NIR airmass not stated
    assert catalog.builder_for("LCO", "CSP2").components is False


@row
def cassini_publishes_pds_curves_and_keeps_svo_copies_apart(t):
    files, nm, pairs = cassini_files("ISS_NAC")
    inst = t.tmp / "Cassini" / "ISS_NAC"
    inst.mkdir(parents=True)
    # An SVO copy (a window of the PDS curve) kept as <band>_svo, as on the njm mirror.
    sys_bl1 = np.loadtxt(io.StringIO(files[cassini.systrans_url("ISS_NAC", "BL1")].split("\\begindata\n")[1]))
    window = (sys_bl1[:, 0] > 400) & (sys_bl1[:, 0] < 450)
    write_filter_file(inst / "BL1_svo.dat", sys_bl1[window, 0] * 10, sys_bl1[window, 1], detector_type="photon",
                      source="svo", filter_id="Cassini/ISS_NAC.BL1", wavelength_unit="Angstrom",
                      extra={"Components": "filter+optics+qe+instrument", "ComponentsSource": "svo"})
    svo_before = (inst / "BL1_svo.dat").read_bytes()
    out = cassini.build(t.tmp, instrument="ISS_NAC", session=Session(files))
    assert (out / "BL1_svo.dat").read_bytes() == svo_before
    assert (out / "ISS_NAC").read_text().split() == [f"{b}.dat" for b in sorted(cassini.BANDS["ISS_NAC"])]
    h = read_filter_header(out / "BL1.dat")
    assert h["FilterID"] == "Cassini/ISS_NAC.BL1" and h["DetectorType"] == "photon"
    assert h["Components"] == "filter+optics+qe" and h["ComponentsSource"] == "original"
    assert cassini.systrans_url("ISS_NAC", "BL1") in h["Sources"]
    pub = np.loadtxt(out / "BL1.dat", delimiter=",", skiprows=1)
    assert np.allclose(pub[:, 0], nm * 10) and np.allclose(pub[:, 1], sys_bl1[:, 1])
    m = json.loads((out / "manifest.json").read_text())
    assert m["bands"] == sorted(cassini.BANDS["ISS_NAC"]) and "BL1_svo" not in m["detector_types"]
    assert m["published"] == {"components": "filter+optics+qe", "source": "original"}
    assert "IR4" not in m["components"]["filter"]  # IR4 extends past 1000 nm
    assert m["components"]["qe"] == {"*": "components/qe/qe.dat"}
    assert read_filter_header(out / "components/filter/BL1.dat")["Sources"].startswith(
        "DERIVED filter pair T(BL1)*T(CL2)")
    f = np.loadtxt(out / "components/filter/BL1.dat", delimiter=",", skiprows=1)
    assert f[-1, 0] == pytest.approx(10000.0)
    assert np.allclose(f[:, 1], np.interp(f[:, 0] / 10, nm, pairs["BL1"]), atol=2e-3)
    comp = compose_band(out, "BL1", "filter+optics+qe")
    assert np.allclose(np.interp(comp.wavelength, pub[:, 0], pub[:, 1]), comp.transmission, rtol=1e-9, atol=1e-15)
    assert "IR4 (IR4xCL2" in m["notes"] and "1 SVO copies equal the PDS curve" in m["notes"]
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


@row
def cassini_band_names_map_to_filter_pairs(t):
    assert cassini.filter_pair("ISS_NAC", "Clear") == ("cl1", "cl2")
    assert cassini.filter_pair("ISS_NAC", "BL2") == ("cl1", "bl2")
    assert cassini.filter_pair("ISS_NAC", "RED_GRN") == ("red", "grn")
    assert cassini.filter_pair("ISS_WAC", "VIO") == ("cl1", "vio")
    assert cassini.filter_pair("ISS_WAC", "MT2_IR1") == ("mt2", "ir1")
    with pytest.raises(ValueError):
        cassini.filter_pair("ISS_WAC", "UV3")
    ref = catalog.builder_for("Cassini", "ISS_WAC")
    assert ref.name == "cassini" and ref.publishes and ref.components


# ======================================================================
# Teams' tabulated factors, team atmospheres and the checks against teams' totals
# ======================================================================
def cassini_recipes_describe_each_camera(t, camera):
    files, _nm, _pairs = cassini_files(camera)
    spec = cassini.RECIPES[camera][1](Fetcher(Session(files)))
    assert (spec.facility, spec.instrument, spec.ground_based) == ("Cassini", camera, False)
    assert spec.outside_range == "error"
    assert [b.name for b in spec.bands] == cassini.BANDS[camera]
    assert all(b.components == "filter+optics+qe" and b.detector_type == "photon" for b in spec.bands)
    roles = [c.role for c in spec.components]
    assert roles[-2:] == ["qe", "optics"] and set(roles[:-2]) == {"filter"}
    assert f"Cassini ISS {camera[-3:]} from the PDS" in spec.notes


CASSINI_CAMERAS = {f"cassini_recipes_describe_each_camera_{camera}": camera for camera in sorted(cassini.CAMERAS)}


@row
def decam_instrument_response_is_the_des_band_over_its_atmosphere(t):
    from astropy.io import fits

    data = fits.open(io.BytesIO(saved()[decam.DES_STD]))[1].data  # SVO's DECam curves are the DES band columns

    def des(fid):
        band = fid.rsplit(".", 1)[1]
        wl = np.asarray(data["LAMBDA"], float)
        if band.endswith("_filter"):  # and flat filter-only siblings
            return SVOFilterRecord(fid, "photon", wl, np.where(np.asarray(data[band[0]]) > 0, 0.9, 0.0), "Angstrom",
                                   "filter")
        return SVOFilterRecord(fid, "photon", wl, np.asarray(data[band], float), "Angstrom",
                               "filter+instrument+atmosphere")

    svo_client = SVOClient(des)
    session = Session(saved())
    inst = decam.build(t.tmp / "filters", session=session, svo_client=svo_client,
                       atmosphere_root=t.tmp / "atmospheres")
    assert session.calls == [decam.DES_STD]  # one download for the five bands and the atmosphere
    assert sorted(svo_client.asked) == sorted(decam.SPEC.svo_ids.values())
    manifest = InstrumentManifest.load(inst)
    assert manifest.default_site == "ctio_des" and manifest.outside_range == "error"
    assert sorted(manifest.components[decam.RECIPE]) == sorted(decam.BANDS)
    grid = AtmosphereGrid.load(t.tmp / "atmospheres" / "ctio_des")
    assert [(e["airmass"], e["file"]) for e in grid.entries] == [(1.2, "am1.20.dat")]
    for band in decam.BANDS:
        composed = compose_band(inst, band, decam.RECIPE, atmosphere={"airmass": 1.2},
                                atmosphere_root=t.tmp / "atmospheres")
        wl, published_curve = curve(inst / f"{band}.dat")
        np.testing.assert_allclose(np.interp(wl, composed.wavelength, composed.transmission), published_curve,
                                   rtol=1e-6, atol=1e-7)
    # The quotient exceeds 1 in i and z (see the module docstring), as in the full table; the notes say so.
    assert manifest.notes.startswith(
        decam.SPEC.notes + " Flag: the instrument-only quotient exceeds 1 in i peaks at 1.021746 at 8500 A "
        "(118 samples above 1 between 7275 and 8510 A); z peaks at 1.036993 at 9060 A (126 samples above 1 "
        "between 8635 and 9260 A). A physical throughput cannot")


@row
def vista_components_come_from_the_eso_bundle(t):
    session = Session(saved())
    flat = SVOClient(lambda fid: record(fid, np.linspace(8000.0, 25000.0, 50), np.full(50, 0.5), "filter+instrument"))
    inst = vista.build(t.tmp / "filters", session=session, svo_client=flat, atmosphere_root=t.tmp / "atmospheres")
    assert session.calls == [vista.BUNDLE]  # one download for every member
    again = Session(saved())
    vista.build(t.tmp / "filters", session=again, svo_client=flat, atmosphere_root=t.tmp / "atmospheres")
    assert again.calls == [vista.BUNDLE]  # a second build downloads afresh, not the first build's copy
    manifest = InstrumentManifest.load(inst)
    assert manifest.default_site == "paranal" and manifest.outside_range == "edge"
    assert not (t.tmp / "atmospheres").exists()  # the SkyCalc site is not a team file
    for band in vista.BROAD + vista.NARROW:
        path = inst / manifest.component_file(FILTER, band)
        wl, tr = curve(path)
        assert released_mesa_reads(path) and 8000 < wl[np.argmax(tr)] < 25000 and 0.5 < tr.max() <= 1.0
    assert "at 80 K" in read_filter_header(inst / manifest.component_file(FILTER, "J"))["Sources"]
    for role in (OPTICS, QE):
        assert manifest.component_file(role, SHARED) == f"components/{role}/{role}.dat"
    wl, optics = curve(inst / manifest.component_file(OPTICS, SHARED))
    assert wl[0] == 3000.0 and wl[-1] == 30000.0 and 0.8 < optics.max() < 1.0  # M1 x M2, nm and % converted
    wl, qe = curve(inst / manifest.component_file(QE, SHARED))
    assert wl[0] == 8060.0 and 0.6 < qe.max() <= 1.0


SITES = ["haleakala_ps1", "paranal_vircam"]


@row
def team_atmospheres_build_writes_archives_and_resumes(t):
    root, archive = t.tmp / "atmospheres", t.tmp / "evidence"
    session = Session(saved())
    assert team.build(t.tmp / "filters", sites=SITES, session=session, archive_dir=archive) == root
    assert sorted(session.calls) == sorted([team.PS1_TABLE3, team.VIRCAM_BUNDLE])
    vircam = AtmosphereGrid.load(root / "paranal_vircam")
    assert [(e["airmass"], e["pwv"], e["file"]) for e in vircam.entries] == [
        (1.0, 1.0, "am1.00_pwv1.dat"), (2.0, 3.0, "am2.00_pwv3.dat")]
    wl, tr = curve(root / "paranal_vircam" / "am1.00_pwv1.dat")
    assert wl[0] == 9000.0 and tr[0] == pytest.approx(0.973927)  # 0.900000 um, as published
    assert AtmosphereGrid.load(root / "haleakala_ps1").kind == "team"
    record = json.loads((archive / "manifest.json").read_text())["files"]
    served = saved()
    assert {f["url"]: f["sha256"] for f in record} == {
        u: hashlib.sha256(served[u]).hexdigest() for u in (team.PS1_TABLE3, team.VIRCAM_BUNDLE)}

    before = (root / "haleakala_ps1" / "manifest.json").read_bytes()
    again = Session(saved())
    team.build(t.tmp / "filters", sites=SITES, session=again)
    assert again.calls == [] and (root / "haleakala_ps1" / "manifest.json").read_bytes() == before
    team.build(t.tmp / "filters", sites=["haleakala_ps1"], session=again, overwrite=True)
    assert again.calls == [team.PS1_TABLE3]


@row
def hst_tables_take_the_newest_graph_and_walk_it(t):
    session = Session(saved())
    tables = hst.SynphotTables(session=session)
    assert (tables.tmg_name, tables.tmc_name) == ("99c1930om_tmg.fits", "aa621124m_tmc.fits")
    assert session.calls == [f"{hst.TRDS}/mtab/"] * 2 + [f"{hst.TRDS}/mtab/{n}" for n in
                                                         ("99c1930om_tmg.fits", "aa621124m_tmc.fits")]
    comps, roles = hst._split_chain(tables, "acs,sbc", "F150LP")
    assert comps == ["hst_ota", "acs_hrc_m12", "acs_f150lp", "acs_f150lp_tds_def", "acs_sbc_mama"]
    assert roles == {FILTER: ["acs_f150lp"], OPTICS: ["hst_ota", "acs_hrc_m12"],
                     QE: ["acs_f150lp_tds_def", "acs_sbc_mama"]}
    assert tables.files["acs_sbc_mama"] == "cracscomp$acs_sbc_mama_013_syn.fits"
    assert tables.source("acs_f150lp") == "STScI synphot acs_f150lp_007_syn.fits (99c1930om_tmg.fits)"
    ir = tables.chain("wfc3,ir,f160w")
    assert ir[0] == "wfc3_ir_primary" and ir[-3:] == ["wfc3_ir_win", "wfc3_ir_qe", "wfc3_ir_cor"]
    with pytest.raises(ValueError, match="no branch at graph node"):
        tables.chain("acs,sbc,f999w")


@row
def hst_tables_need_a_listed_table(t):
    files = saved()
    files[f"{hst.TRDS}/mtab/"] = files[f"{hst.TRDS}/mtab/"].replace(b"_tmc.fits", b"_tmx.fits")
    with pytest.raises(RuntimeError, match=r"No \*_tmc.fits found"):
        hst.SynphotTables(session=Session(files))


@row
def lsst_validate_compares_with_rubins_curves(t):
    session = Session(lsst_files("v1.9.2", scale=0.5))
    lsst.build(t.tmp / "filters", instrument="LSSTv19", session=session)
    result = lsst.validate(t.tmp / "filters" / "LSST" / "LSSTv19", session=session)
    assert sorted(result) == sorted(lsst.BANDS)
    for values in result.values():  # Rubin's hardware and total curves are these products
        assert values == {k: pytest.approx(0.0, abs=1e-6) for k in
                          ("hardware_dmag", "hardware_maxdiff", "total_dmag", "total_maxdiff")}
    assert lsst._url("hardware_r", "v1.9.2") in session.calls


@row
def twomass_validate_compares_with_the_supplement_totals(t):
    inst = t.tmp / "2MASS" / "2MASS"
    for band, lo in (("J", 10600), ("H", 15000), ("Ks", 19500)):
        wl = np.linspace(lo, lo + 4000, 30)
        published(inst, band, wl, np.ones_like(wl), "filter+instrument+atmosphere", detector=ENERGY)
    files = twomass_files()
    twomass.build(t.tmp, session=Session(files))
    um = np.round(np.linspace(1.0, 2.5, 151), 3)
    for band, n in twomass.TOTAL_TABLE.items():  # J and H as composed, Ks at half
        mine = compose_band(inst, band, (FILTER, OPTICS, QE))
        total = np.interp(um * 1e4, mine.wavelength, mine.transmission, left=0, right=0) * (0.5 if band == "Ks" else 1)
        files[twomass.TABLE.format(n=n)] = ipac_page(f"2MASS {band} TOTAL", um, total)
    result = twomass.validate(inst, session=Session(files))
    assert result["J"]["dmag"] == pytest.approx(0.0, abs=1e-3) and result["H"]["maxdiff"] < 1e-4
    assert result["Ks"]["dmag"] == pytest.approx(-2.5 * np.log10(2.0), abs=1e-3)


@row
def skycalc_transmission_is_the_fetched_curve(t):
    client = skycalc.SkyCalcClient(session=SkyCalc(), interface="form", pause=0)
    wl, tr = client.transmission("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)
    np.testing.assert_allclose(wl, np.geomspace(1000.0, 1100.0, 200))
    np.testing.assert_allclose(tr, 0.8 ** 1.5)
    assert client.session.posts[-1]["SKYMODEL.TARGET.AIRMASS"] == "1.50"


@row
def write_site_refuses_a_bad_curve_before_writing(t):
    bad = team_site(entries=[(1.2, None, DerivedCurve("u", derived([0.5, 1.2, 0.9]), "x"))])
    with pytest.raises(ValueError, match="not a transmission"):
        write_site(t.tmp, bad, None)
    assert not (t.tmp / "here_team").exists()


PS1_ROWS = (" 500 0.500 0.400 0.000 0.000 0.000 0.000 0.000 0.970 0.880 0.990\n"
            " 501 0.500 0.410 0.000 0.000 0.000 0.000 0.000 0.970 0.881 0.980\n")


@row
def ps1_build_registers_its_team_site(t):
    session = Session({surveys.PS1_URL: PS1_ROWS, surveys.PS1_BASE + "table4.dat": "g_P1|0.4|481|414|551|24.6\n"})
    t.mp.setattr(surveys, "ps1", lambda fetch: surveys.Instrument(
        "PAN-STARRS", "PS1", [surveys.make_band("g", [500.0, 501.0], [0.4, 0.41], unit="nm",
                                                detector_type="photon", components="filter+optics+qe+atmosphere",
                                                sources="x")], "n", ground_based=True,
        sites=[team.HALEAKALA_PS1], default_site="haleakala_ps1"))
    t.mp.setitem(surveys.RECIPES, "PS1", ("PAN-STARRS", surveys.ps1))
    build = surveys.make_builder(surveys.RECIPES)
    inst = build(t.tmp / "filters", instrument="PS1", session=session)
    assert InstrumentManifest.load(inst).default_site == "haleakala_ps1"
    grid = AtmosphereGrid.load(t.tmp / "atmospheres" / "haleakala_ps1")
    assert grid.kind == TEAM and grid.entries[0]["airmass"] == 1.2 and "Aero" in grid.entries[0]["derivation"]


@row
def csp_atmosphere_raises_the_one_airmass_transmission_to_1p2(t):
    raw = tar_gz({"atm_ext_ctio.dat": b"3000. 0.5\n5000. 0.8\n11000. 1.0\n",
                  "atm_ext_lines.dat": b"2950. 1.\n4000. 0.9\n6000. 1.0\n12000. 1.\n"}, "CSP_filter_package")
    wl, tr = team.csp_atmosphere(raw)
    assert wl[0] == 3000.0 and wl[-1] == 11000.0
    np.testing.assert_allclose(tr[wl == 5000.0], 0.8 ** 1.2 * 0.95)


@row
def vircam_atmosphere_skips_its_title_lines(t):
    member = gzip.compress(b"1.0mm H2O, airmass=1.0\n\n0.900000   0.97\n0.900500   0.96\n")
    raw = tar_gz({"trans_10_10.dat.gz": member}, "Filters_QE_Atm_curves")
    wl, tr = team.vircam_atmosphere(raw, "trans_10_10.dat.gz", "1.0mm H2O, airmass=1.0")
    np.testing.assert_allclose(wl, [9000.0, 9005.0])
    with pytest.raises(ValueError, match="unexpected title"):
        team.vircam_atmosphere(raw, "trans_10_10.dat.gz", "3.0mm H2O, airmass=2.0")


@row
def twomass_table_refuses_a_truncated_copy(t):
    page = ("<html><head><title>ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE</title></head><pre>\n"
            "|n  |xlam     |yval     |\n   1    1.0000    0.8948\n   2    1.0001    0.8922\n</pre>")
    wl, tr = team.twomass_table(page.encode(), "ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE")
    np.testing.assert_allclose(wl, [10000.0, 10001.0])
    with pytest.raises(ValueError, match="consecutive"):
        team.twomass_table(page.replace("   2 ", "   3 ").encode(), "ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE")


@row
def gemini_atran_refuses_a_truncated_file(t):
    wl, _ = team.gemini_atran(b".900000 0.99\n3.0 0.5\n5.59998 0.1\n", "u")
    assert wl[-1] == pytest.approx(55999.8)
    with pytest.raises(ValueError, match="truncated"):
        team.gemini_atran(b".900000 0.99\n1.2 0.5\n", "u")


@row
def sdss_transparency_merges_bands_and_refuses_conflicts(t):
    head = "# assumed\n# atmospheric\n# extinction\n# through\n# 1.0 airmasses\n"
    files = {url: head + "3000 0 0 0 0.10\n3025 0 0 0 0.12\n" for url in team.SDSS_FILES.values()}
    wl, tr = team.sdss_transparency(Session(files))
    np.testing.assert_allclose(tr, [0.10, 0.12])
    files[team.SDSS_FILES["z"]] = head + "3000 0 0 0 0.11\n"
    with pytest.raises(ValueError, match="disagrees"):
        team.sdss_transparency(Session(files))


@row
def bass_ratio_keeps_only_well_measured_rows(t):
    head = "# ... atmospheric extinction at airmass of 1.0.\n"
    files = {team.BASS_FILES["g"]: head + "4000 0.5 0.400 0.300\n4010 0.1 0.050 0.040\n",
             team.BASS_FILES["r"]: head + "6000 0.9 0.800 0.720\n"}
    wl, tr = team.bass_atmosphere(Session(files))
    np.testing.assert_allclose(wl, [4000.0, 6000.0])
    np.testing.assert_allclose(tr, [0.75, 0.9])


@row
def form_fields_follow_skycalc_spelling(t):
    params = dict(skycalc._BASE_PARAMS, airmass=2.0, pwv=0.5, observatory="2400")
    fields = skycalc._form_fields(params)
    assert fields["SKYMODEL.PWV"] == "0.50" and fields["TEL.SITE.HEIGHT"] == "2400"
    assert fields["SKYMODEL.TARGET.AIRMASS"] == "2.00" and fields["SKYMODEL.TARGET.ALT"] == "30.000"
    assert "SKYMODEL.INCL.MOON" not in fields  # unticked checkbox
    assert skycalc.airmass_altitude(3.0) == 19.5


@row
def form_client_checks_the_echo(t):
    client = skycalc.SkyCalcClient(session=SkyCalc(), interface="form", pause=0)
    result = client.fetch("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)
    assert result.interface == "form" and result.version == "2.0.9"
    assert result.url == skycalc.FORM_HOST + "/observing/etc/tmp/abc/skytable.fits"
    bad = skycalc.SkyCalcClient(session=SkyCalc({"SKYMODEL.PWV": "3.5"}), interface="form", pause=0)
    with pytest.raises(RuntimeError, match="echoed SKYMODEL.PWV"):
        bad.fetch("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)


@row
def skycalc_grid_is_labelled_derived_and_archived(t):
    client = skycalc.SkyCalcClient(session=SkyCalc(), interface="form", pause=0)
    root = skycalc.build(t.tmp / "filters", sites=["lasilla"], airmasses=[1.0], pwvs=[2.5],
                         fetch_resolution=2000, store_resolution=500, chunks=2, client=client,
                         archive_dir=t.tmp / "evidence")
    grid = AtmosphereGrid.load(root / "lasilla")
    assert grid.kind == DERIVED and grid.model_version == "2.0.9" and "Noll" in grid.citation
    assert grid.parameters["observatory"] == "2400" and grid.parameters["fetch_resolution"] == 2000
    assert grid.parameters["interface"].startswith("form: https://www.eso.org")
    header = read_filter_header(root / "lasilla" / "am1.00_pwv2.5.dat")
    assert header["ComponentsSource"] == "derived" and header["Sources"] == "ESO SkyCalc 2.0.9"
    evidence = json.loads((t.tmp / "evidence" / "lasilla" / "manifest.json").read_text())
    assert len(evidence["files"]) == 2
    record = evidence["files"][0]
    raw = gzip.decompress((t.tmp / "evidence" / "lasilla" / record["file"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == record["sha256"] and record["skycalc_version"] == "2.0.9"


@row
def skycalc_resume_leaves_a_complete_grid_untouched(t):
    client = skycalc.SkyCalcClient(session=SkyCalc(), interface="form", pause=0)
    kw = dict(sites=["paranal"], airmasses=[1.0], pwvs=[2.5], chunks=2, client=client)
    root = skycalc.build(t.tmp / "filters", fetch_resolution=2000, store_resolution=500, **kw)
    before = (root / "paranal" / "manifest.json").read_bytes()
    # A curator re-running the builder (other settings, nothing to fetch) must not rewrite served files.
    skycalc.build(t.tmp / "filters", client=skycalc.SkyCalcClient(session=SkyCalc(), pause=0),
                  sites=["paranal"], airmasses=[1.0], pwvs=[2.5])
    assert (root / "paranal" / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="different settings"):
        skycalc.build(t.tmp / "filters", fetch_resolution=2000, store_resolution=400, **dict(kw, airmasses=[1.5]))


@row
def hsc_builder_and_team_registry_write_the_same_site(t):
    assert team.SITES["maunakea_hsc"] is hsc.SPEC.site and hsc.SPEC.site.kind == TEAM


# ======================================================================
# The builders' shared machinery
# ======================================================================
@row
def table_floors_negative_measured_transmission(t):
    wl, tr = Table("u").parse(b"4000,-0.0001\n5000,0.8\n")
    np.testing.assert_allclose(wl, [4000, 5000])
    np.testing.assert_allclose(tr, [0.0, 0.8])


@row
def percent_component_is_rejected(t):
    with pytest.raises(ComponentError, match="percent"):
        add_component(t.tmp, FILTER, [4000.0, 5000.0], [10.0, 95.0], band="g")


@row
def panic_br_gamma_uses_existing_mirror_band_name(t):
    from sed_tools.component_sources.tabular import SOURCES

    assert "Brgamma" in SOURCES["PANIC"].filters
    assert "BrGamma" not in SOURCES["PANIC"].filters


@row
def jwst_reference_negative_tail_is_zero(t):
    from sed_tools.component_sources.jwst import _read

    path = t.tmp / "throughput.fits"
    path.write_bytes(fits_file(fits_table([("wavelength", None, [2.0, 1.0]), ("throughput", None, [0.7, -0.02])])))
    wl, tr = _read(path)
    np.testing.assert_allclose(wl, [10000.0, 20000.0])
    np.testing.assert_allclose(tr, [0.0, 0.7])


@row
def jwst_band_factors(t):
    from sed_tools.component_sources.jwst import _band_factors

    paths = {"f200w": "filters/a.fits", "f444w": "filters/b.fits", "grismr": "blaze/c.fits"}
    assert _band_factors("nircam", "F200W", paths) == {FILTER: ["f200w"], OPTICS: ["internal", "dbs_sw"], QE: ["qe_sw"]}
    assert _band_factors("nircam", "F444W", paths)[QE] == ["qe_lw"]
    assert _band_factors("nircam", "GRISMR", paths) is None
    assert _band_factors("miri", "F444W", paths) == {FILTER: ["f444w"], OPTICS: [], QE: ["miri_imager_qe"]}


@row
def ensure_components_runs_own_builder_and_skips_heavy(t):
    calls = []
    t.mp.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("fake"))
    t.mp.setitem(catalog.BUILDERS_FOR, ("Fac", "Big"), catalog.BuilderRef("fake", heavy=True))
    t.mp.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)

    def build(root, **kw):
        calls.append(("fake", root))
        inst = root / "Fac" / "Cam"
        manifest = InstrumentManifest.load(inst)
        add_component(inst, FILTER, [4000.0, 5000.0], [1.0, 1.0], band="g", manifest=manifest)
        manifest.save(inst)

    t.mp.setattr(filter_library, "get_builder", lambda name: build)
    for name in ("Cam", "Big", "None"):
        d = t.tmp / "Fac" / name
        d.mkdir(parents=True)
        (d / "g.dat").write_text("Wavelength,Transmission\n4000,1\n5000,1\n")

    assert catalog.ensure_components(t.tmp / "Fac" / "Cam") is True
    assert catalog.ensure_components(t.tmp / "Fac" / "Big") is False
    assert catalog.ensure_components(t.tmp / "Fac" / "None") is False
    assert calls == [("fake", t.tmp)]


def svo_instruments_that_share_a_name_get_their_own_directory(t, name, key, others, expected):
    from sed_tools.svo_filter_grabber import _would_collide

    assert _would_collide(name, key, others) is expected


#: Instrument selection (regressions from interactive testing).
SHARED_NAMES = {f"svo_instruments_that_share_a_name_get_their_own_directory_{case[1]}": case for case in [
    ("GAIA", "GAIA0", ["GAIA2", "GAIA3", "Gaia2m"], True),   # every Gaia release is "GAIA" at SVO
    ("GAIA", "GAIA3", ["GAIA0"], True),
    ("Johnson", "Johnson", ["Bessell"], False),             # name == key: path unchanged
    ("ACS_WFC", "ACS_WFC", ["ACS_HRC"], False),
    ("Cousins", "Cousins_old", ["Bessell"], False),          # differs but nothing to clash with
]}


@row
def njm_instrument_match_is_exact(t):
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    njm = NJMFilterGrabber(base_dir="/tmp/unused")
    t.mp.setattr(njm, "discover_instruments", lambda fac: ["GAIA", "ACS_WFC"])
    assert njm.find_instrument("GAIA", "GAIA0") is None        # was: prefix-matched to GAIA
    assert njm.find_instrument("GAIA", "gaia") == "GAIA"
    assert njm.find_instrument("HST", "ACS-WFC") == "ACS_WFC"


@row
def facility_wide_status_applies_to_every_instrument(t):
    assert catalog.status_for("GAIA", "GAIA0")[0] == catalog.TOTAL_ONLY
    assert catalog.status_for("GAIA", "GAIA3")[0] == catalog.TOTAL_ONLY
    assert catalog.status_for("LSST", "LSST") is None


# ======================================================================
# Instruments rebuilt from their teams' original files (excerpts in tests/fixtures/original_sources)
# ======================================================================
MODULES = {"surveys": surveys, "space": space, "observatories": obs, "standards": standards,
           "swift": swift, "csp": csp, "cassini": cassini}


@row
def original_sources_do_not_shadow_other_builders(t):
    original = {(f, i) for entries in catalog.ORIGINAL_SOURCES.values() for f, i, _ in entries}
    others = {k for k, ref in catalog.BUILDERS_FOR.items() if ref.name not in MODULES}
    assert not original & others
    assert len(others) >= 28  # the builders registered before these were added
    assert all(ref.publishes for ref in catalog.BUILDERS_FOR.values() if ref.name == "tabular")


@row
def units_and_cleaning(t):
    np.testing.assert_allclose(to_angstrom([1.0], "micron"), [1e4])
    np.testing.assert_allclose(to_angstrom([1.0], "1/cm"), [1e8])
    np.testing.assert_allclose(to_angstrom([299.792458], "GHz"), [1e7])
    wl, tr, note = clean_curve([3, 1, 2, 2, 4], [0.5, -0.1, 0.2, 0.4, np.nan], normalize=True)
    np.testing.assert_allclose(wl, [1, 2, 3])
    np.testing.assert_allclose(tr, [0.0, 0.6, 1.0])
    assert "1 negative" in note and "1 undefined" in note and "1 duplicate" in note and "scaled" in note
    with pytest.raises(ValueError):
        clean_curve([1, 2], [0.0, -1.0])


@row
def fetcher_rejects_html_unless_allowed(t):
    fetch = Fetcher(Session({"u": b"<!DOCTYPE html><p>moved</p>"}))
    with pytest.raises(ValueError, match="HTML"):
        fetch.get("u")
    assert fetch.get("u", allow_html=True).startswith(b"<!DOCTYPE")


def ogle_files():
    return {surveys.OGLE_FILT["V"]: fx("OGLE-IV-V-filter-transmission.dat"),
            surveys.OGLE_FILT["I"]: fx("OGLE-IV-I-filter-transmission.dat"),
            surveys.OGLE_QE: fx("OGLE-IV-QE-curve-from-specification.dat")}


@row
def published_rebuilds_from_original_sources_when_njm_is_down(t):
    t.mp.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)
    session = Session(ogle_files())
    t.mp.setattr(requests, "Session", lambda: session)
    library = FilterLibrary(t.tmp)
    library.svo = None  # SVO must not be needed
    directory = library.published("OGLE", "OGLE-IV")
    assert directory == t.tmp / "OGLE" / "OGLE-IV"
    assert FilterLibrary.published_complete(directory)
    assert (directory / "OGLE-IV").read_text() == "I.dat\nV.dat\n"
    header = read_filter_header(directory / "V.dat")
    assert header["FilterID"] == "OGLE/OGLE-IV.V"
    assert header["Components"] == "filter+qe" and header["ComponentsSource"] == "original"
    assert header["DetectorType"] == "photon" and header["DetectorTypeSource"] == "original"
    manifest = InstrumentManifest.load(directory)
    assert manifest.ground_based is True and manifest.outside_range == "zero"
    assert set(manifest.components) == {"filter", "qe"}
    assert manifest.sources["published"] == surveys.OGLE_FILT["V"]
    # published = filter x QE interpolated (0 outside the QE range)
    raw = np.loadtxt(ORIGINAL / "OGLE-IV-V-filter-transmission.dat", usecols=(0, 1))
    qe = np.loadtxt(ORIGINAL / "OGLE-IV-QE-curve-from-specification.dat")
    data = np.loadtxt(directory / "V.dat", delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 0], raw[:, 0])
    np.testing.assert_allclose(data[:, 1], np.maximum(raw[:, 1], 0) *
                               np.interp(raw[:, 0], qe[:, 0], qe[:, 1], left=0, right=0), rtol=1e-5)
    # A second call is served locally.
    session.calls.clear()
    assert library.published("OGLE", "OGLE-IV") == directory and not session.calls


@row
def builder_keeps_existing_published_curves(t):
    directory = t.tmp / "OGLE" / "OGLE-IV"
    directory.mkdir(parents=True)
    (directory / "V.dat").write_text("Wavelength,Transmission\n4000,0.5\n5000,0.5\n")
    surveys.build(t.tmp, instrument="OGLE-IV", session=Session(ogle_files()))
    assert (directory / "V.dat").read_text() == "Wavelength,Transmission\n4000,0.5\n5000,0.5\n"
    assert (directory / "I.dat").is_file() and (directory / "components" / "qe" / "qe.dat").is_file()


@row
def ogle_header_is_checked(t):
    files = ogle_files()
    files[surveys.OGLE_FILT["V"]] = files[surveys.OGLE_FILT["I"]]
    with pytest.raises(ValueError, match="unexpected header"):
        surveys.build(t.tmp, instrument="OGLE-IV", session=Session(files))


@row
def ukirt_tar_members_refuse_unsafe_names(t):
    member = fx("ukirt_Table02_online.dat")
    text = surveys.tar_members(tar_gz({"Table02_online.dat": member}), {"Table02_online.dat"})
    assert "WFCAM Z filter" in text["Table02_online.dat"]
    with pytest.raises(ValueError, match="refusing"):
        surveys.tar_members(tar_gz({"../evil": "x", "Table02_online.dat": member}), {"Table02_online.dat"})
    with pytest.raises(ValueError, match="lacks"):
        surveys.tar_members(tar_gz({"other.dat": member}), {"Table02_online.dat"})


@row
def lco_zip_member_and_checksum(t):
    raw = zip_bytes({"SDSS.up.txt": fx("lco_SDSS.up.txt")})
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        a = surveys.lco_member(zf, "SDSS.up.txt")
    assert a.shape == (5, 2) and a[0, 1] < 0  # "\lambda [nm]" header skipped
    with pytest.raises(ValueError, match="differs from the verified copy"):
        surveys.sinistro(Fetcher(Session({surveys.LCO_URL: raw})))


@row
def herschel_and_planck_fits(t):
    from astropy.io import fits

    def table(cols, name=None, **header):
        return fits.HDUList([fits.PrimaryHDU(), fits_table(cols, name, header)])

    pacs = fits_file(fits_table([("wavelength", "um", [60.0, 70.0]), ("transmission", "", [0.2, 0.5])],
                                header={"BAND": "BS"}))
    wl, tr = space.pacs_curve(gzip.compress(pacs), "BS")
    assert wl.tolist() == [60.0, 70.0] and tr.tolist() == [0.2, 0.5]
    with pytest.raises(ValueError, match="BAND"):
        space.pacs_curve(pacs, "R")
    spire = fits_file(fits_table([("frequency", "GHz", [1000.0, 1200.0]), ("psw", "", [0.5, 1.0]),
                                  ("pmw", "", [0.0, 0.1]), ("plw", "", [0.0, 0.0])], "RSRF"))
    nu, cols = space.spire_rsrf(spire)
    assert nu.tolist() == [1000.0, 1200.0] and cols["psw"].tolist() == [0.5, 1.0]
    rimo = table([("WAVENUMBER", "1/cm", [0.5, 2.0, 3.0]), ("TRANSMISSION", "", [1e-8, 0.5, 1.0])],
                 name="BANDPASS_F100")
    x, _ = space.planck_bandpass(rimo, "BANDPASS_F100", "1/cm")
    assert x.tolist() == [0.5, 2.0, 3.0]
    with pytest.raises(ValueError):
        space.planck_bandpass(rimo, "BANDPASS_F100", "GHz")


@row
def iras_html_table(t):
    page = fx("iras_tabC5.html")
    with pytest.raises(ValueError, match="rows of 4"):  # the excerpt ends inside a row
        space.iras_tables(page)
    tables = space.iras_tables(page.replace("<TD>33.0</TD>", ""))
    assert sorted(tables) == ["12", "60"]
    assert tables["12"].tolist() == [[7.0, 0.0, 0.3, 0.0], [7.5, 0.01, 0.35, 0.008]]
    assert tables["60"].shape == (2, 4)


@row
def msx_ipac_table(t):
    a = space.msx_table(fx("msx_rsr_a.tbl"))
    assert a.shape == (4, 2) and a[0].tolist() == [5.8, 0.0]
    with pytest.raises(ValueError):
        space.msx_table(fx("msx_rsr_a.tbl").replace("microns", "nm"))


@row
def omegacam_centre_scan(t):
    wl, tr = obs.ocam_centre(fx("ocam_u_with_ccd.txt"))
    assert wl.tolist() == [479.916, 482.905, 485.894] and tr[0] == 0.000042
    with pytest.raises(ValueError, match="centre"):
        obs.ocam_centre(fx("ocam_u_with_ccd.txt").replace("# x_pos 0.000000", "# x_pos 2.0"))


@row
def keck_xls_sheets(t):
    a = obs.nirc2_sheet(Book({"J": [["Wavelength", "Transmission"], [1.1, 1.0], [1.2, 90.0], [1.3, 2.0]]}), "J")
    assert a[:, 1].max() == 90.0
    with pytest.raises(ValueError, match="percent"):
        obs.nirc2_sheet(Book({"J": [["Wavelength", "Transmission"], [1.1, 0.9]]}), "J")
    rows = [["W.M. Keck Observatory test report"]] + [[""] * 2] * 5 + [["", "center"]] + [[""] * 2] * 3
    rows += [["wl(nm)", "T%"], [700.0, 10.0], [760.0, 90.0]]
    np.testing.assert_allclose(obs.lris_i_sheet(Book({"s": rows})), [[7000, 0.1], [7600, 0.9]])


@row
def keck_width_check(t):
    wl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs.check_width("x", wl, np.array([0, 1, 1, 1, 0.0]), 3.0, 2.0, tol=0.01)
    with pytest.raises(ValueError, match="50% points"):
        obs.check_width("x", wl, np.array([0, 1, 1, 1, 0.0]), 3.5, 2.0)


@row
def not_scan_header_and_scale(t):
    a = obs.not_curve(109, fx("not_109.txt"), False, "u' SDSS")
    assert a.shape == (5, 2) and a[:, 1].max() < 1
    with pytest.raises(ValueError, match="does not identify"):
        obs.not_curve(110, fx("not_109.txt"), False, "r' SDSS")
    with pytest.raises(ValueError, match="scale"):
        obs.not_curve(109, fx("not_109.txt"), True, "u' SDSS")


@row
def subaru_and_odi_tables(t):
    qe = obs.subaru_table(fx("subaru_QE_Hamamatsu.txt"), 300, 1100, 0.2, 1.0, "qe")
    assert qe.shape == (5, 2)
    with pytest.raises(ValueError, match="peak"):
        obs.subaru_table(fx("subaru_QE_Hamamatsu.txt"), 300, 1100, 0.8, 1.0, "qe")
    session = Session({obs.ODI + fn: fx("odi_NB_422.txt") for fn, _, _ in obs.ODI_FILES.values()})
    with pytest.raises(ValueError, match="scale"):   # u/r/NB695 are fractions; this scan is percent
        obs.odi(Fetcher(session))
    wl = np.array([4000.0, 9000.0, 11000.0, 12000.0])
    assert obs.odi_leak(wl, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert 0.5 < obs.odi_leak(wl, np.array([0.0, 0.0, 1.0, 1.0])) < 1.0


@row
def bessell_and_mko_parsers(t):
    curves = space.fixed_width_pairs(fx("bessell_table1.dat"), standards.B12_COLS)
    assert list(curves) == ["U", "B", "V", "R", "I"] and all(len(w) == 4 for w, _ in curves.values())
    data = standards.mko_scan(fx("nsfcam_lpmk_trans.dat"))
    b = make_band("Lp", data[:, 0], data[:, 1] / 100, unit="micron", detector_type="photon",
                  components="filter", sources="x")
    assert np.all(np.diff(b.wavelength) > 0) and b.transmission.min() == 0.0
    with pytest.raises(ValueError, match="unexpected header"):
        standards.mko_scan("WL\tT\n1 2\n")


@row
def usno_reads_only_the_wayback_snapshots(t):
    text = fx("usno_u.res")
    assert standards.usno_table(text, "u").shape == (4, 6)
    with pytest.raises(ValueError, match="unexpected header"):
        standards.usno_table(text, "g")
    # The curves cite the snapshots, so an unreachable archive stops the build
    # instead of silently reading the live files.
    session = Session({standards.USNO_README: "readme", standards.USNO_ORIG.format("u"): text})
    with pytest.raises(OSError):
        standards.build(t.tmp, instrument="USNO", session=session)
    assert session.calls == [standards.USNO_README_WB]
    assert not (t.tmp / "Generic").exists()


@row
def naco_band_names_have_no_inner_dot(t):
    from sed_tools.component_sources.tabular import NACO

    assert "IB_2p00" in NACO.filters and "NB_3p74" in NACO.filters
    assert all("." not in name for name in NACO.filters)
    assert NACO.filters["IB_2p00"].url.endswith("/IB_2.00.dat")


def test_builders(rows):
    rows(BUILDS, builds)
    rows(EVERY_RECIPE, every_recipe_is_covered)
    rows(WRONG_FILE, stops_before_writing)
    rows(PROBES, probe_finds_a_contiguous_grid)
    rows(CASSINI_CAMERAS, cassini_recipes_describe_each_camera)
    rows(SHARED_NAMES, svo_instruments_that_share_a_name_get_their_own_directory)
    rows(SCENARIOS)
