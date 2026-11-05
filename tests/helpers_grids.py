"""Stand-ins shared by the spectral-grid tests: NJM's copy of the golden Kurucz grid (a
module-scoped fixture), its fake web server, the CLI runner, file digests, and the guard that fails any real
HTTP request.
"""

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import requests

from sed_tools import cli


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


GRID = "Kurucz2003all"
NJM_URL = "https://data.nialljmiller.com/sed_tools"
#: The golden Kurucz test grid: Teff 5750, 6000, 6250; logg 4.0, 4.5; [M/H] -0.5, 0.0.
KURUCZ = Path(__file__).parent / "golden" / "inputs" / "stellar_models" / GRID


class Reply:
    def __init__(self, status, body=b""):
        self.status_code, self.content, self.headers = status, body, {}
        self.text = body.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        import json
        return json.loads(self.text)

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


class FakeNJM:
    """A stand-in for NJM's web server: ``root`` holds index.json and stellar_models/<grid>/ with its files."""

    def __init__(self, root):
        self.root, self.requested, self.headers, self.verify = Path(root), [], {}, True

    def __call__(self):  # requests.Session()
        return self

    def get(self, url, **kwargs):
        rel = url.split("?")[0][len(NJM_URL) + 1:]
        self.requested.append(rel)
        path = self.root / rel
        if rel.endswith("/") and path.is_dir():
            links = "".join(f'<a href="{p.name}">{p.name}</a>\n' for p in sorted(path.iterdir()))
            return Reply(200, f"<html><body>{links}</body></html>".encode())
        return Reply(200, path.read_bytes()) if path.is_file() else Reply(404)

    def head(self, url, **kwargs):
        return Reply(200 if (self.root / url[len(NJM_URL) + 1:]).exists() else 404)

    def spectra_requested(self):
        return [Path(rel).name for rel in self.requested if rel.endswith(".txt")]


@pytest.fixture(scope="module")
def njm_grid(tmp_path_factory):
    """NJM's copy of the Kurucz test grid, with the whole grid's flux cube and HDF5 bundle beside it."""
    from sed_tools.api import _build_flux_cube
    from sed_tools.spectrum_io import build_h5_bundle

    root = tmp_path_factory.mktemp("njm")
    served = root / "stellar_models" / GRID
    shutil.copytree(KURUCZ, served)
    build_h5_bundle(served, served / f"{GRID}.h5")
    _build_flux_cube(served)
    (root / "index.json").write_text('{"models": ["%s"]}' % GRID)
    return root


@pytest.fixture
def njm(njm_grid, monkeypatch):
    server = FakeNJM(njm_grid)
    monkeypatch.setattr("sed_tools.njm_spectra_grabber.requests.Session", server)
    return server


def digests(directory):
    """Each file's bytes and modification time."""
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def copy_without_times(src, dst):
    """Copy a folder as ``cp -R`` or ``scp`` without ``-p`` do, then make every spectrum newer than the
    cube, bundle and lookup table, as a copy in the wrong order leaves them."""
    shutil.copytree(src, dst, copy_function=shutil.copyfile, symlinks=True)
    now = time.time()
    for path in Path(dst).rglob("*"):
        if path.is_file() and not path.is_symlink():
            stamp = now + 100 if path.suffix == ".txt" else now
            os.utime(path, (stamp, stamp))
    return Path(dst)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test tried to use the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)
