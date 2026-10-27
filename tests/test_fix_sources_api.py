"""Four bugs that gave wrong results without an error.

5. MAST (BOSZ): the first spectrum of a download was written with 51 rows,
   and lookup_table.csv held only file names.
6. MSG: grid names were cut short by ``rstrip(".h5")``.
7. ``SED.query`` / ``SED.fetch`` ignored a bound of 0.
8. ``photometry(("V", path))`` unpacked one (name, path) spec into two.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import csv
import gzip

import numpy as np
import pytest
import requests

from sed_tools import mast_spectra_grabber as mast
from sed_tools import msg_spectra_grabber as msg
from sed_tools._lookup_io import find_logg_column, find_metallicity_column, find_teff_column
from sed_tools.api import SED
from sed_tools.models import SEDModel


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test tried to use the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)


class FakeResponse:
    def __init__(self, status=200, body=b""):
        self.status_code = status
        self.content = body if isinstance(body, bytes) else body.encode()
        self.text = self.content.decode("utf-8", "ignore")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)


class FakeSession:
    """Answers from a {url: response} table; any other URL gets a 404."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(404))

    def mount(self, *args):
        pass


# ---------------------------------------------------------------- 5. MAST (BOSZ)

# Real BOSZ 2024 names, from the MAST bulk download scripts
#   .../download_scripts/hlsp_bosz_bosz2024_sim_rorig_m-1.00_v1_bulkdl.sh
#   .../download_scripts/hlsp_bosz_bosz2024_sim_r500_m%2B0.00_v1_bulkdl.sh
# retrieved 2026-10-27.  Between them they carry both signs of logg, [M/H],
# [alpha/M] and [C/M], and a five-digit Teff.
RORIG_NAMES = [
    "bosz2024_ap_t10500_g+2.0_m-1.00_a+0.25_c+0.50_v4_rorig_noresam.txt.gz",
    "bosz2024_ms_t2800_g-0.5_m-1.00_a-0.25_c-0.75_v2_rorig_noresam.txt.gz",
    "bosz2024_ms_t4250_g+3.0_m-1.00_a+0.50_c+0.25_v0_rorig_noresam.txt.gz",
]
R500_NAMES = [
    "bosz2024_ap_t10500_g+4.5_m+0.00_a+0.00_c-0.50_v0_r500_resam.txt.gz",
    "bosz2024_ap_t9250_g+2.5_m+0.00_a-0.25_c+0.50_v0_r500_resam.txt.gz",
    "bosz2024_ms_t3600_g-0.5_m+0.00_a+0.25_c-0.75_v1_r500_resam.txt.gz",
]
N_ROWS = 600
WAVE = 1000.0 + 10.0 * np.arange(N_ROWS)  # rises 50 % over the first 51 rows


def flux_of(k):
    """A spectrum that is not monotonic, so it is never taken for wavelengths."""
    i = np.arange(N_ROWS)
    return (k + 1) * 1e-3 * (1.0 + 0.5 * np.sin(i))


def bosz_session(reskey, metal, names, bodies, extra=None):
    urls = [f"{mast.DATA_BASE}/{reskey}/{metal}/{name}" for name in names]
    script = (f"{mast.SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_{reskey}_"
              f"{metal.replace('+', '%2B')}_v1_bulkdl.sh")
    lines = ["#!/bin/sh"] + [f'curl -f --create-dirs --output "{reskey}/{metal}/{n}" "{u}"'
                             for n, u in zip(names, urls)]
    routes = {script: FakeResponse(200, "\n".join(lines) + "\n")}
    for url, body in zip(urls, bodies):
        routes[url] = FakeResponse(200, gzip.compress(body.encode()))
    routes.update(extra or {})
    return FakeSession(routes)


def download(tmp_path, reskey, metal, session):
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path), max_workers=1, session=session)
    model = f"BOSZ-2024-{reskey}"
    n = grabber.download_model_spectra(model, {"reskey": reskey, "metals": [metal]})
    return n, tmp_path / model


def written(model_dir, name):
    return np.loadtxt(model_dir / mast._sanitize_filename(name.replace(".txt.gz", ".txt")), comments="#")


def test_mast_rorig_writes_every_row_of_every_spectrum_including_the_first(tmp_path):
    bodies = ["".join(f"{w:.3f} {f:.6e} {2 * f:.6e}\n" for w, f in zip(WAVE, flux_of(k)))
              for k in range(len(RORIG_NAMES))]
    n, model_dir = download(tmp_path, "rorig", "m-1.00", bosz_session("rorig", "m-1.00", RORIG_NAMES, bodies))

    assert n == len(RORIG_NAMES)
    for k, name in enumerate(RORIG_NAMES):  # the first URL in sorted order is RORIG_NAMES[0]
        data = written(model_dir, name)
        assert data.shape == (N_ROWS, 3), name
        np.testing.assert_allclose(data[:, 0], WAVE)
        np.testing.assert_allclose(data[:, 1], flux_of(k), rtol=1e-6)
        np.testing.assert_allclose(data[:, 2], 2 * flux_of(k), rtol=1e-6)


def test_mast_two_column_wave_flux_files_are_written_in_full(tmp_path):
    # Detected as carrying their own wavelengths, so no wavelength grid is fetched
    # (FakeSession would answer 404 and the download would raise).
    bodies = ["".join(f"{w:.3f} {f:.6e}\n" for w, f in zip(WAVE, flux_of(k)))
              for k in range(len(R500_NAMES))]
    session = bosz_session("r500", "m+0.00", R500_NAMES, bodies)
    n, model_dir = download(tmp_path, "r500", "m+0.00", session)

    assert n == len(R500_NAMES)
    assert not any("wave" in url for url in session.calls)
    for k, name in enumerate(R500_NAMES):
        data = written(model_dir, name)
        assert data.shape == (N_ROWS, 3), name
        np.testing.assert_allclose(data[:, 0], WAVE)
        np.testing.assert_allclose(data[:, 1], flux_of(k), rtol=1e-6)


def test_mast_resampled_files_on_the_wavelength_grid_are_written_in_full(tmp_path):
    bodies = ["".join(f"{f:.6e} {2 * f:.6e}\n" for f in flux_of(k)) for k in range(len(R500_NAMES))]
    wave_url = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r500.txt"
    grid = FakeResponse(200, "".join(f"{w:.3f}\n" for w in WAVE))
    n, model_dir = download(tmp_path, "r500", "m+0.00",
                            bosz_session("r500", "m+0.00", R500_NAMES, bodies, {wave_url: grid}))

    assert n == len(R500_NAMES)
    for k, name in enumerate(R500_NAMES):
        data = written(model_dir, name)
        assert data.shape == (N_ROWS, 3), name
        np.testing.assert_allclose(data[:, 0], WAVE)
        np.testing.assert_allclose(data[:, 1], flux_of(k), rtol=1e-6)


def read_lookup(model_dir):
    with open(model_dir / "lookup_table.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_mast_lookup_table_has_every_parameter_of_real_bosz_names(tmp_path):
    bodies = ["".join(f"{w:.3f} {f:.6e} {f:.6e}\n" for w, f in zip(WAVE, flux_of(k)))
              for k in range(len(RORIG_NAMES))]
    _, model_dir = download(tmp_path, "rorig", "m-1.00", bosz_session("rorig", "m-1.00", RORIG_NAMES, bodies))

    rows = read_lookup(model_dir)
    assert list(rows[0]) == ["file_name", "am", "atmos", "cm", "logg", "mh", "res", "teff", "vturb"]
    assert rows == [
        {"file_name": "bosz2024_ap_t10500_g_2.0_m-1.00_a_0.25_c_0.50_v4_rorig_noresam.txt",
         "am": "0.25", "atmos": "ap", "cm": "0.50", "logg": "2.00", "mh": "-1.00",
         "res": "rorig", "teff": "10500", "vturb": "4"},
        {"file_name": "bosz2024_ms_t2800_g-0.5_m-1.00_a-0.25_c-0.75_v2_rorig_noresam.txt",
         "am": "-0.25", "atmos": "ms", "cm": "-0.75", "logg": "-0.50", "mh": "-1.00",
         "res": "rorig", "teff": "2800", "vturb": "2"},
        {"file_name": "bosz2024_ms_t4250_g_3.0_m-1.00_a_0.50_c_0.25_v0_rorig_noresam.txt",
         "am": "0.50", "atmos": "ms", "cm": "0.25", "logg": "3.00", "mh": "-1.00",
         "res": "rorig", "teff": "4250", "vturb": "0"},
    ]
    for row in rows:
        assert (model_dir / row["file_name"]).is_file()
    # The package's lookup readers find Teff, logg and [M/H] in it.
    assert (find_teff_column(rows[0]), find_logg_column(rows[0]), find_metallicity_column(rows[0])) == (
        "teff", "logg", "mh")


def test_mast_lookup_table_keeps_plus_signs_dropped_by_the_file_names(tmp_path):
    bodies = ["".join(f"{w:.3f} {f:.6e}\n" for w, f in zip(WAVE, flux_of(k))) for k in range(len(R500_NAMES))]
    _, model_dir = download(tmp_path, "r500", "m+0.00", bosz_session("r500", "m+0.00", R500_NAMES, bodies))

    params = {r["file_name"]: (r["teff"], r["logg"], r["mh"], r["am"], r["cm"], r["vturb"], r["atmos"], r["res"])
              for r in read_lookup(model_dir)}
    assert params == {
        "bosz2024_ap_t10500_g_4.5_m_0.00_a_0.00_c-0.50_v0_r500_resam.txt":
            ("10500", "4.50", "0.00", "0.00", "-0.50", "0", "ap", "r500"),
        "bosz2024_ap_t9250_g_2.5_m_0.00_a-0.25_c_0.50_v0_r500_resam.txt":
            ("9250", "2.50", "0.00", "-0.25", "0.50", "0", "ap", "r500"),
        "bosz2024_ms_t3600_g-0.5_m_0.00_a_0.25_c-0.75_v1_r500_resam.txt":
            ("3600", "-0.50", "0.00", "0.25", "-0.75", "1", "ms", "r500"),
    }


# ---------------------------------------------------------------- 6. MSG grid names

def test_msg_grid_names_keep_their_last_characters(tmp_path, monkeypatch):
    index_url = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"
    names = ["sg-OSTAR2002-high", "sg-CAP18-high", "sg-Test-h5", "sg-Grid-1.5", "sg-demo"]
    links = "".join(f'<a href="http://user.astro.wisc.edu/msg/grids/{n}.h5">{n}</a>' for n in names)
    session = FakeSession({index_url: FakeResponse(200, links)})
    monkeypatch.setattr(msg.requests, "Session", lambda: session)

    grabber = msg.MSGSpectraGrabber(base_dir=str(tmp_path), max_workers=1)

    assert grabber.discover_models() == sorted(names)
    assert grabber.model_urls["sg-OSTAR2002-high"] == "http://user.astro.wisc.edu/msg/grids/sg-OSTAR2002-high.h5"


# ---------------------------------------------------------------- 7. zero bounds

# logg and [M/H] ranges chosen so that a bound of 0 keeps some grids and drops others.
LOCAL_GRIDS = {
    "Negative": {"logg": [-1.0, -0.5], "meta": [-2.0, -1.0]},
    "Positive": {"logg": [0.5, 1.0], "meta": [0.25, 0.5]},
    "Straddle": {"logg": [-0.5, 0.5], "meta": [-0.5, 0.5]},
}


@pytest.fixture
def local_grids(tmp_path, monkeypatch, cube_writer):
    for name, axes in LOCAL_GRIDS.items():
        teff = [4000.0, 6000.0]
        flux = np.ones((len(teff), len(axes["logg"]), len(axes["meta"]), 2))
        cube_writer(tmp_path / name / "flux_cube.bin", teff, axes["logg"], axes["meta"], [4000.0, 5000.0], flux)
    monkeypatch.setattr(SED, "_model_root", tmp_path)
    return tmp_path


@pytest.mark.parametrize("bound, expected", [
    ({"teff_min": 0}, {"Negative", "Positive", "Straddle"}),
    ({"teff_max": 0}, set()),
    ({"logg_min": 0.0}, {"Positive", "Straddle"}),
    ({"logg_max": 0.0}, {"Negative", "Straddle"}),
    ({"metallicity_min": 0.0}, {"Positive", "Straddle"}),
    ({"metallicity_max": 0.0}, {"Negative", "Straddle"}),
    ({"metallicity_max": 0}, {"Negative", "Straddle"}),
    ({"logg_min": 0, "metallicity_max": 0.0}, {"Straddle"}),
])
def test_query_applies_a_bound_of_zero(local_grids, bound, expected):
    found = SED.query(source="local", include_remote=False, **bound)
    assert {info.name for info in found} == expected


@pytest.fixture
def downloaded_grid(tmp_path, monkeypatch):
    """A grid already on disk; SED.fetch's download step is replaced by a no-op."""
    grid = tmp_path / "Tiny"
    grid.mkdir()
    rows = ["file_name,teff,logg,metallicity"]
    for logg in (-0.5, 0.0, 0.5):
        for meta in (-0.5, 0.0, 0.5):
            name = f"g{logg:+.1f}_m{meta:+.1f}.txt"
            (grid / name).write_text("4000 1.0\n5000 2.0\n")
            rows.append(f"{name},5000,{logg},{meta}")
    (grid / "lookup_table.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(SED, "_download_from", classmethod(lambda cls, src, catalog, model_dir, workers: None))
    return tmp_path


@pytest.mark.parametrize("bound, keep", [
    ({"teff_min": 0}, lambda logg, meta: True),
    ({"teff_max": 0}, lambda logg, meta: False),
    ({"logg_min": 0.0}, lambda logg, meta: logg >= 0),
    ({"logg_max": 0.0}, lambda logg, meta: logg <= 0),
    ({"metallicity_min": 0.0}, lambda logg, meta: meta >= 0),
    ({"metallicity_max": 0.0}, lambda logg, meta: meta <= 0),
    ({"metallicity_max": 0}, lambda logg, meta: meta <= 0),
])
def test_fetch_applies_a_bound_of_zero(downloaded_grid, bound, keep):
    sed = SED.fetch("Tiny", source="njm", clean=False, model_root=downloaded_grid, **bound)

    expected = {(logg, meta) for logg in (-0.5, 0.0, 0.5) for meta in (-0.5, 0.0, 0.5) if keep(logg, meta)}
    assert {(s.logg, s.metallicity) for s in sed.grid} == expected


# ---------------------------------------------------------------- 8. photometry(("V", path))

@pytest.fixture
def evaluated(tmp_path, cube_writer):
    teff, logg, meta = [4000.0, 6000.0], [4.0, 5.0], [-0.5, 0.5]
    wave = np.arange(3000.0, 8001.0, 100.0)
    shape = np.exp(-((wave - 5500.0) / 2000.0) ** 2) + 0.1
    flux = np.array([[[t / 5000.0 * shape for _ in meta] for _ in logg] for t in teff])
    cube_writer(tmp_path / "flux_cube.bin", teff, logg, meta, wave, flux)
    return SEDModel(name="tiny", flux_cube_path=tmp_path / "flux_cube.bin")(5000.0, 4.5, 0.0)


def filter_file(path, lo, hi):
    rows = [(lo - 100, 0.0), (lo, 1.0), (hi, 1.0), (hi + 100, 0.0)]
    path.write_text("# WavelengthUnit = Angstrom\n" + "".join(f"{w:.1f} {t}\n" for w, t in rows))
    return path


@pytest.fixture
def bands(tmp_path):
    return filter_file(tmp_path / "johnson_v.dat", 5000, 6000), filter_file(tmp_path / "johnson_b.dat", 4000, 5000)


def magnitudes(results):
    return {name: result.magnitude for name, result in results.items()}


def test_photometry_takes_a_single_name_path_tuple_as_one_filter(evaluated, bands):
    v, _ = bands
    plain = magnitudes(evaluated.photometry(v))

    assert magnitudes(evaluated.photometry(("V", v))) == {"V": plain["johnson_v"]}
    assert magnitudes(evaluated.photometry(("V", str(v)))) == {"V": plain["johnson_v"]}


def test_photometry_takes_several_name_path_tuples(evaluated, bands):
    v, b = bands
    plain = magnitudes(evaluated.photometry(v, b))

    expected = {"V": plain["johnson_v"], "B": plain["johnson_b"]}
    assert magnitudes(evaluated.photometry(("V", v), ("B", b))) == expected
    assert magnitudes(evaluated.photometry((("V", v), ("B", b)))) == expected  # one tuple of specs


def test_a_lone_pair_of_names_stays_two_filters(bands):
    # ("V", "B") must not become filter "B" labelled "V": only a path makes a (name, path) spec.
    from sed_tools.models import _normalize_filter_sequence

    v, _ = bands
    assert _normalize_filter_sequence([("V", "B")]) == ["V", "B"]
    assert _normalize_filter_sequence([("V", str(v))]) == [("V", str(v))]
    assert _normalize_filter_sequence([("V", v)]) == [("V", v)]


def test_photometry_plain_paths_and_sequences_of_them_are_unchanged(evaluated, bands):
    v, b = bands
    plain = magnitudes(evaluated.photometry(str(v), str(b)))

    assert set(plain) == {"johnson_v", "johnson_b"}
    assert plain["johnson_v"] != plain["johnson_b"]
    assert magnitudes(evaluated.photometry(v, b)) == plain
    assert magnitudes(evaluated.photometry([str(v), str(b)])) == plain
    assert magnitudes(evaluated.photometry((v, b))) == plain  # Paths, so not a (name, path) spec
