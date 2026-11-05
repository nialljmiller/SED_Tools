"""MAST (BOSZ) and MSG downloads are written in full: every row of every spectrum,
a lookup table with every parameter, and MSG grid names kept whole.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import gzip

import numpy as np
import requests

from sed_tools import mast_spectra_grabber as mast

from helpers_grids import no_network  # noqa: F401 (an autouse fixture)


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
