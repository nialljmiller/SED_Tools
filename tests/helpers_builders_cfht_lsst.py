"""Fakes shared by the CFHT, ERIS/NIX, 2MASS and LSST builder tests: a fake session,
published curves, and the files those builders download.
"""

from __future__ import annotations

import numpy as np

from sed_tools.component_sources import lsst, twomass
from sed_tools.filter_metadata import PHOTON, write_filter_file


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.text = self.content.decode("latin-1")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, files):
        self.files = files
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _published(directory, band, wl, tr, components, detector=PHOTON):
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=detector, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": components, "ComponentsSource": "svo"})


def _table(wl, tr, sep=" "):
    return "".join(f"{w}{sep}{t}\n" for w, t in zip(wl, tr))


def _pre(title, um, values):
    rows = "".join(f"  {i + 1:3d}  {w:.3f}  {v:.4f}\n" for i, (w, v) in enumerate(zip(um, values)))
    return (f"<html><head><title>{title}</title></head><body><b><pre>\n\\CHAR TITLE = {title}\n"
            f"|n  |xlam     |yval     |\n|i  |r        |r        |\n{rows}</pre></body></html>")


def _twomass_files():
    um = np.linspace(1.0, 2.5, 31)
    files = {}
    values = {1: 0.9, 2: 0.95, 3: 0.97, 4: 0.99, 6: 0.98, 7: 0.97, 9: 0.96, 11: 0.6}
    for n, title in twomass.TITLES.items():
        if n in values:
            files[twomass.TABLE.format(n=n)] = _pre(title, um, np.full(um.size, values[n]))
    for n, (lo, hi) in ((5, (1.1, 1.4)), (8, (1.5, 1.8)), (10, (2.0, 2.3))):
        x = np.linspace(lo - 0.05, hi + 0.05, 13)
        files[twomass.TABLE.format(n=n)] = _pre(twomass.TITLES[n].replace("-BAND", "K<sub>s</sub>-BAND")
                                                if n == 10 else twomass.TITLES[n], x,
                                                np.where((x > lo) & (x < hi), 0.8, 0.0))
    return files


def _lsst_files(tag, scale=1.0):
    nm = np.arange(300.0, 1151.0, 50.0)
    one = _table(nm, np.full(nm.size, 0.9))
    files = {}
    for name in lsst.OPTICS_FILES + ("detector", "atmos_10", "atmos_std"):
        files[lsst._url(name, tag)] = "# header\n" + one
    for b in lsst.BANDS:
        filt = np.where((nm >= 500) & (nm <= 700), 0.8 * scale, 0.0)
        files[lsst._url(f"filter_{b}", tag)] = _table(nm, filt)
        files[lsst._url(f"total_{b}", tag)] = _table(nm, filt * 0.9 ** 8)
        files[lsst._url(f"hardware_{b}", tag)] = _table(nm, filt * 0.9 ** 7)
    return files
