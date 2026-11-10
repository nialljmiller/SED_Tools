"""What the SED_Tools tests share: the row runner every area test uses, and the fakes for NJM, SVO,
MAST, SkyCalc and the other sources.

Every row runs as a test of its own (``Rows``): its own folder, monkeypatch, output and log capture,
config file, NJM and SVO state, and random state. No row uses the network: a real request raises
``NetworkUsed``. Each part below is extended only by the package that owns it (tests-min/PLAN.md).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import re
import shutil
import struct
import sys
import tarfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools.combine_filters import write_filter_index
from sed_tools.component_sources import cassini, lsst, swift, twomass
from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere
from sed_tools.filter_components import TEAM, InstrumentManifest
from sed_tools.filter_library import instrument_catalogue
from sed_tools.filter_metadata import PHOTON, read_filter_header, released_mesa_reads, write_filter_file
from sed_tools.njm_filter_grabber import NJMFilterGrabber
from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument, SVOFilterBrowser

TESTS = Path(__file__).resolve().parent


# ======================================================================
# Core: rows, isolation, the network guard, input(), the CLI (frozen)
# ======================================================================
class NetworkUsed(BaseException):
    """Raised past every ``except Exception``, so a request a row forgot to fake cannot pass silently."""


def _refuse_network(*args, **kwargs):
    raise NetworkUsed("tests must not use the network")


@dataclass
class Row:
    """One row's own world. ``stack`` holds what to undo when the row ends."""

    name: str
    tmp: Path
    mp: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture
    caplog: pytest.LogCaptureFixture
    stack: ExitStack


class Rows:
    """``rows(table, check)`` runs ``check(row, *data)`` for each ``name: data`` (``check(row, data)`` when
    data is not a tuple), each as a named subtest; ``rows(table)`` calls each value with the row."""

    def __init__(self, subtests, tmp_path_factory, capsys, caplog):
        self._subtests, self._factory, self._capsys, self._caplog = subtests, tmp_path_factory, capsys, caplog
        self._names = set()

    def __call__(self, table, check=None):
        for name, data in table.items():
            assert name not in self._names, f"two rows are named {name!r}"
            self._names.add(name)
            with self._subtests.test(name), self._isolated(name) as row:
                if check is None:
                    data(row)
                else:
                    check(row, *data) if isinstance(data, tuple) else check(row, data)

    @contextmanager
    def _isolated(self, name):
        """What the autouse fixtures and pytest give each test, given to each row."""
        from sed_tools import config

        rng = _random_state()
        config_dir = self._factory.mktemp("sed_tools_config")
        try:
            with pytest.MonkeyPatch.context() as mp, ExitStack() as stack:
                mp.setattr(config, "_CONFIG_DIR", config_dir)
                mp.setattr(config, "_CONFIG_FILE", config_dir / "config.toml")
                mp.setattr(requests.Session, "send", _refuse_network)
                _reset_sources()
                stack.callback(_reset_sources)
                self._capsys.readouterr()
                self._caplog.clear()
                yield Row(name, self._factory.mktemp(re.sub(r"\W", "_", name)[:30]), mp, self._capsys,
                          self._caplog, stack)
        finally:
            _set_random_state(rng)


def _reset_sources():
    """What a process knows about NJM and SVO is shared by its clients; each row starts as a new process."""
    from sed_tools import njm_filter_grabber, svo_filter_grabber

    njm_filter_grabber._STATE.reset()
    svo_filter_grabber._STATE.reset()


def _random_state():
    """random's and numpy's state, and torch's (CPU and MPS) once something has imported torch."""
    torch = sys.modules.get("torch")
    mps = torch is not None and torch.backends.mps.is_available()
    return (random.getstate(), np.random.get_state(), torch and torch.get_rng_state(),
            torch.mps.get_rng_state() if mps else None)


def _set_random_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    if state[2] is not None:
        sys.modules["torch"].set_rng_state(state[2])
    if state[3] is not None:
        sys.modules["torch"].mps.set_rng_state(state[3])


def answer(mp, *replies, echo=False):
    """Feed ``input()`` these replies in order (printing prompt and reply when ``echo``); return the prompts
    asked. A further prompt fails the row."""
    queue, prompts = list(replies), []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        reply = queue.pop(0)
        if echo:
            print(prompt + reply)
        return reply

    mp.setattr("builtins.input", fake_input)
    return prompts


def narrow_terminal(t):
    """An 80-column terminal, without SED_TOOLS_DEBUG."""
    t.mp.setenv("COLUMNS", "80")
    t.mp.delenv("SED_TOOLS_DEBUG", raising=False)


def wide_lines(text, *allowed):
    """Lines over 80 columns that contain none of the ``allowed`` substrings (paths)."""
    return [line for line in text.splitlines() if len(line) > 80 and not any(a in line for a in allowed)]


def run_cli(mp, *argv):
    """``sed-tools *argv``: what ``cli.main()`` returns."""
    from sed_tools import cli

    mp.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


def run_module(mp, module, *argv):
    """``python -m sed_tools.<module> *argv`` in this process, so it is measured: the exit status."""
    import runpy

    mp.setattr(sys, "argv", [module, *argv])
    try:
        runpy.run_module(f"sed_tools.{module}", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code
    return 0


class Resp:
    """A response: ``content`` (a str is encoded with ``encoding``) and ``status``; from 400 up,
    ``raise_for_status`` raises ``error`` saying "<status> for <url>"."""

    def __init__(self, content=b"", status=200, *, url="", encoding="utf-8", error=requests.HTTPError):
        self.content = content.encode(encoding) if isinstance(content, str) else content
        self.text = self.content.decode(encoding, "replace")
        self.status_code, self.url, self.headers, self._error = status, url, {}, error

    def raise_for_status(self):
        if self.status_code >= 400:
            message = f"{self.status_code} for {self.url}" if self.url else str(self.status_code)
            if issubclass(self._error, requests.RequestException):
                raise self._error(message, response=self)
            raise self._error(message)

    def json(self):
        return json.loads(self.text)

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Session:
    """A requests session serving ``files``: URL (or, with ``base``, the path under it) -> bytes or text, an
    HTTP status, or an exception to raise. Any other URL is a 404. ``calls`` lists what was asked."""

    def __init__(self, files, base="", **resp):
        self.files, self.base, self.resp = dict(files), base, resp
        self.headers, self.verify, self.calls = {}, True, []

    def get(self, url, timeout=None, params=None, **kwargs):
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        key = url[len(self.base) + 1:] if self.base else url
        self.calls.append(key)
        body = self._body(key)
        if isinstance(body, BaseException):
            raise body
        return Resp(b"", body, url=key, **self.resp) if isinstance(body, int) else Resp(body, url=key, **self.resp)

    def _body(self, key):
        return self.files.get(key, 404)


# ======================================================================
# Flux cubes and grids (package C: grids + cubes)
# ======================================================================
GRID = "Kurucz2003all"
#: The golden test grids; Kurucz2003all has Teff 5750, 6000, 6250; logg 4.0, 4.5; [M/H] -0.5, 0.0.
GOLDEN_MODELS = TESTS / "golden" / "inputs" / "stellar_models"
KURUCZ = GOLDEN_MODELS / GRID
NJM = "https://data.nialljmiller.com/sed_tools"


def write_flux_cube(path, teff, logg, meta, wavelengths, flux):
    """Write a ``flux_cube.bin``. ``flux`` has the in-memory shape ``(nt, nl, nm, nw)``; on disk it is stored
    axis-reversed, ``(nw, nm, nl, nt)``. Getting that backwards was the "Flux cube axis mismatch" bug."""
    teff, logg, meta, wavelengths, flux = (np.asarray(a, dtype=np.float64)
                                           for a in (teff, logg, meta, wavelengths, flux))
    expected_shape = (teff.size, logg.size, meta.size, wavelengths.size)
    if flux.shape != expected_shape:
        raise ValueError(f"flux shape {flux.shape} != expected {expected_shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", *expected_shape))
        for grid in (teff, logg, meta, wavelengths):
            grid.tofile(fh)
        flux.transpose(3, 2, 1, 0).ravel().tofile(fh)


@dataclass
class AffineCube:
    """F(T, g, Z, wl) = a + b*T + c*g + d*Z + e*T*g, which the cubic-Hermite interpolation reproduces exactly:
    any mismatch against :meth:`expected` is an interpolation or axis-order bug, not curvature."""

    path: Path
    teff: np.ndarray
    logg: np.ndarray
    meta: np.ndarray
    wavelengths: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray

    def expected(self, teff, logg, meta):
        return self.a + self.b * teff + self.c * logg + self.d * meta + self.e * teff * logg


def affine_cube(root, teff=(4000.0, 4500.0, 6000.0, 6200.0), logg=(1.0, 2.5, 5.0), meta=(-2.0, -0.3, 0.0),
                wavelengths=(4000.0, 4500.0, 6000.0), name="affine_demo", include_cross_term=True):
    """An :class:`AffineCube` at ``root/name/flux_cube.bin``. The coefficients differ per wavelength and per
    axis, and the grids are unevenly spaced, so a swapped axis gives a wrong answer, not a lucky match."""
    teff, logg, meta, wavelengths = (np.asarray(a, dtype=np.float64) for a in (teff, logg, meta, wavelengths))
    n = np.arange(wavelengths.size, dtype=np.float64)
    a, b, c, d = 1.0 + n, 0.0010 + 0.0003 * n, 0.30 - 0.05 * n, 0.90 + 0.20 * n
    e = 1e-6 * (1.0 + n) if include_cross_term else np.zeros(n.size)
    flux = np.zeros((teff.size, logg.size, meta.size, n.size))
    for ti, T in enumerate(teff):
        for li, L in enumerate(logg):
            for mi, M in enumerate(meta):
                flux[ti, li, mi, :] = a + b * T + c * L + d * M + e * T * L
    path = Path(root) / name / "flux_cube.bin"
    write_flux_cube(path, teff, logg, meta, wavelengths, flux)
    return AffineCube(path, teff, logg, meta, wavelengths, a, b, c, d, e)


class GridServer(Session):
    """NJM's web server for spectral grids: ``root`` holds index.json and stellar_models/<grid>/. A folder
    URL lists its files. It stands in for ``requests.Session`` too (calling it returns itself)."""

    def __init__(self, root):
        super().__init__({}, base=NJM)
        self.root = Path(root)

    def __call__(self):
        return self

    def get(self, url, **kwargs):
        return super().get(url.split("?")[0])

    def head(self, url, **kwargs):
        return Resp(b"", 200 if (self.root / url[len(NJM) + 1:]).exists() else 404)

    def _body(self, key):
        path = self.root / key
        if key.endswith("/") and path.is_dir():
            return "<html><body>%s</body></html>" % "".join(f'<a href="{p.name}">{p.name}</a>\n'
                                                            for p in sorted(path.iterdir()))
        return path.read_bytes() if path.is_file() else 404

    def spectra_requested(self):
        return [Path(key).name for key in self.calls if key.endswith(".txt")]


_NJM_GRID = {}


def njm_grid(t):
    """NJM's copy of the Kurucz test grid, with the whole grid's flux cube and HDF5 bundle beside it,
    built once and shared, as the old module-scoped fixture was."""
    if "root" not in _NJM_GRID:
        from sed_tools.api import _build_flux_cube
        from sed_tools.spectrum_io import build_h5_bundle

        root = t.tmp.parent / "njm_grid"
        served = root / "stellar_models" / GRID
        shutil.copytree(KURUCZ, served)
        build_h5_bundle(served, served / f"{GRID}.h5")
        _build_flux_cube(served)
        (root / "index.json").write_text('{"models": ["%s"]}' % GRID)
        _NJM_GRID["root"] = root
    return _NJM_GRID["root"]


def njm_grids(t):
    """NJM serving :func:`njm_grid` to the spectra grabbers; returns the server."""
    server = GridServer(njm_grid(t))
    t.mp.setattr("sed_tools.njm_spectra_grabber.requests.Session", server)
    return server


def copy_without_times(src, dst):
    """Copy a folder as ``cp -R`` or ``scp`` without ``-p`` do, then make every spectrum newer than the cube,
    bundle and lookup table, as a copy in the wrong order leaves them."""
    shutil.copytree(src, dst, copy_function=shutil.copyfile, symlinks=True)
    now = time.time()
    for path in Path(dst).rglob("*"):
        if path.is_file() and not path.is_symlink():
            stamp = now + 100 if path.suffix == ".txt" else now
            os.utime(path, (stamp, stamp))
    return Path(dst)


def write_lookup(model_dir, header, rows):
    model_dir.mkdir(parents=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    (model_dir / "lookup_table.csv").write_text("\n".join(lines) + "\n")


# ======================================================================
# Filters: NJM and SVO stand-ins (package A: filters)
# ======================================================================
CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=original\n4000,0\n5000,1\n6000,0\n"


def njm_index(detailed):
    return json.dumps({"generated": "2026-10-27T00:00:00",
                       "filters": {"facilities": sorted(detailed), "facilities_detailed": {
                           fac: {"instruments": insts} for fac, insts in detailed.items()}}})


def atmosphere_site(root, site, manifest, grids=()):
    directory = root / "atmospheres" / site
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name in grids:
        (directory / name).write_text("Wavelength,Transmission\n5000,0.9\n6000,0.95\n")
    return directory


STALE = {"site": "lsst_std", "schema": 1, "source": "Rubin", "altitude_m": 2647.0,
         "entries": [{"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}]}


# ======================================================================
# Builders: saved pages, sources' files, published curves (package B: builders + ML)
# ======================================================================
WL = np.linspace(4000.0, 6000.0, 201)


def gauss(centre, width, wl=WL):
    return np.exp(-0.5 * ((wl - centre) / width) ** 2)


def published_set(directory, band, wl, tr, components, filter_id=None):
    """A curve as SVO publishes it, with its components and FilterID, and the set's MESA index rewritten."""
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=PHOTON, source="svo",
                      filter_id=filter_id or f"{directory.parent.name}/{directory.name}.{band}",
                      wavelength_unit="Angstrom",
                      extra={"Components": components, "ComponentsSource": "svo"})
    write_filter_index(directory, directory.name)


def derived(values):
    """A DerivedCurve fetch: 3000, 5000 and 7000 Å with these values."""
    return lambda session: (np.array([3000.0, 5000.0, 7000.0]), np.array(values))


def team_site(name="here_team", entries=None, **kw):
    entries = entries if entries is not None else [(1.2, None, DerivedCurve("u", derived([0.5, 0.8, 0.9]), "x"))]
    return SiteAtmosphere(site=name, altitude_m=1000.0, entries=entries, source="test team",
                          kind=TEAM, model="MODTRAN", model_version="not stated", citation="Someone 2001", **kw)


# ----------------------------------------------------------------------
# ML (package B)
# ----------------------------------------------------------------------
def cpu_torch(t):
    """torch on the CPU, single-threaded and seeded, and matplotlib on Agg; the row skips without torch or
    scikit-learn. The ``.device`` properties and the networks pick CUDA or MPS when they can, and the two
    can disagree, so everything is pinned to the CPU."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import matplotlib

    matplotlib.use("Agg")
    import sed_tools.ml_utils as ml_utils

    t.mp.setattr(ml_utils, "get_device", lambda: torch.device("cpu"))
    t.mp.setattr(torch.cuda, "is_available", lambda: False)
    t.stack.callback(torch.set_num_threads, torch.get_num_threads())
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    return torch


def tiny_library(t):
    """A 3 x 2 x 2 flux cube whose spectra differ in level and slope, in ``t.tmp / "lib"``."""
    teff, logg, meta = np.array([4000.0, 5000.0, 6000.0]), np.array([4.0, 4.5]), np.array([-0.5, 0.0])
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.zeros((3, 2, 2, 200))
    for ti in range(3):
        for li in range(2):
            for mi in range(2):
                flux[ti, li, mi] = 1e-10 * (1 + 0.1 * ti + 0.01 * li + 0.02 * mi) * (wl / 5000.0) ** (-2 - 0.1 * ti)
    write_flux_cube(t.tmp / "lib" / "flux_cube.bin", teff, logg, meta, wl, flux)
    return t.tmp / "lib"


def fixed_completer(torch):
    from sed_tools import ml_sed_completer as mc

    torch.manual_seed(0)
    completer = mc.SEDCompleter()
    completer.model = mc.SEDCompleterNetwork(input_dim=400, output_dim=1000, hidden_layers=[16, 8], dropout=0.3)
    completer.wavelength_grid = completer._create_wavelength_grid()
    completer._known_start, completer._known_end = 300, 700
    completer.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "logg_mean": 4.0, "logg_std": 0.5,
        "meta_mean": 0.0, "meta_std": 0.5, "flux_log_mean": 6.5, "flux_log_std": 1.0,
        "ext_log_mean": 6.0, "ext_log_std": 1.5,
    }
    return completer


def fixed_generator(torch):
    from sed_tools import ml_sed_generator as mg

    torch.manual_seed(0)
    generator = mg.SEDGenerator()
    generator.model = mg.SEDGeneratorNetwork(output_dim=1000, hidden_layers=[16, 8], dropout=0.2)
    generator.wavelength_grid = generator._create_wavelength_grid()
    generator.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "teff_min": 4000.0, "teff_max": 7000.0,
        "logg_mean": 4.0, "logg_std": 0.5, "logg_min": 3.5, "logg_max": 5.0,
        "meta_mean": 0.0, "meta_std": 0.5, "meta_min": -1.0, "meta_max": 0.5,
        "flux_log_mean": -10.0, "flux_log_std": 1.0,
    }
    generator.config = {"parameter_ranges": {"teff": [4000.0, 7000.0], "logg": [3.5, 5.0], "meta": [-1.0, 0.5]}}
    return generator


# ======================================================================
# CLI and API: grabbers, the filters API, pickers and catalogues (package D: CLI + API + golden)
# ======================================================================
GRABBERS = {
    "njm": "sed_tools.njm_spectra_grabber.NJMSpectraGrabber",
    "svo": "sed_tools.svo_spectra_grabber.SVOSpectraGrabber",
    "msg": "sed_tools.msg_spectra_grabber.MSGSpectraGrabber",
    "mast": "sed_tools.mast_spectra_grabber.MASTSpectraGrabber",
}


def fake_grabber(models=(), available=True, metadata=None, downloaded=0, download_error=None):
    """A grabber class; ``metadata`` may be an exception to raise, and ``download_error`` is raised after
    ``downloaded`` spectra have been written."""
    class Grabber:
        base_url = "https://mirror.invalid"

        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = base_dir

        def is_available(self):
            return available

        def discover_models(self):
            return list(models)

        def get_model_metadata(self, name):
            if metadata is None:
                raise RuntimeError(f"no metadata for {name}")  # as the real grabbers do
            if isinstance(metadata, Exception):
                raise metadata
            return metadata

        def download_model_spectra(self, name, meta, **cuts):
            for i in range(downloaded):
                path = Path(self.base_dir) / name / f"spectrum{i}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("1000 1\n2000 1\n")
            if download_error is not None:
                raise download_error
            return downloaded

    return Grabber


def filters_api(t):
    """``sed_tools.api.Filters`` with its filter root in the row's folder and no catalogue cached."""
    from sed_tools.api import Filters

    t.mp.setattr(Filters, "_filter_root", t.tmp / "filters")
    t.mp.setattr(Filters, "_catalogues", {})
    return Filters


CLEAN = {"total": 1, "converted": [], "skipped_already": [], "skipped_invalid": [], "skipped_index": [],
         "error": [], "recovered": [], "detection_stats": {}, "catalog_units": None}


def scenarios():
    """An empty table of scenarios, and the decorator that adds a function to it as a row named after it."""
    table = {}

    def row(function):
        assert function.__name__ not in table, f"two scenarios are named {function.__name__!r}"
        table[function.__name__] = function
        return function

    return table, row
