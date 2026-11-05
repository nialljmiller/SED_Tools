"""Fakes shared by the Swift/UVOT, CSP and Cassini builder tests: FITS and PDS
tables, a fake session, and the files those builders download.
"""

from __future__ import annotations

import io

import numpy as np

from sed_tools.component_sources import cassini, swift


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode("latin-1")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, files):
        self.files, self.headers, self.calls = files, {}, []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _hdul(*tables) -> bytes:
    from astropy.io import fits

    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), *tables]).writeto(buf)
    return buf.getvalue()


def _table(name, columns, header):
    from astropy.io import fits

    cols = [fits.Column(name=n, format="E" if n != "TIME" else "D", unit=u, array=np.asarray(a))
            for n, u, a in columns]
    hdu = fits.BinTableHDU.from_columns(cols, name=name)
    for k, v in header.items():
        hdu.header[k] = v
    return hdu


GRID = np.arange(1600.0, 8001.0, 10.0)


def _gauss(centre, width):
    return np.exp(-0.5 * ((GRID - centre) / width) ** 2)


def _uvot_files(root):
    instea = 60.0 * np.exp(-0.5 * ((GRID - 3000.0) / 1500.0) ** 2)
    filt = {f: 0.9 * _gauss(c, 300.0) for f, c in
            (("U", 3500), ("B", 4400), ("V", 5400), ("UVW1", 2600), ("UVW2", 1950), ("UVM2", 2250), ("WHITE", 3800))}
    files = {
        root + swift.FTRANS: _hdul(*[
            _table("TRANSMISSION" + f, [("ENERGY", "eV", 12398.4 / GRID), ("WAVELENGTH", "angstrom", GRID),
                                        ("TRANSMIS", "", t)], {"FILTER": f}) for f, t in filt.items()]),
        root + swift.MIRROR: _hdul(_table("MIRROREFFAREA", [
            ("ENERGY", "eV", 12398.4 / GRID[1:441]), ("WAVELENGTH", "angstrom", GRID[1:441]),
            ("EA", "cm**2", np.full(440, 590.0))], {})),
        root + swift.INSTEA: _hdul(_table("INSTEFFAREA", [
            ("ENERGY", "eV", 12398.4 / GRID), ("WAVELENGTH", "angstrom", GRID), ("EA", "cm**2", instea)], {})),
        root + swift.SENSCORR: _hdul(*[
            _table("SENSCORR" + f, [("TIME", "s", [1.2e8, 7.9e8]), ("OFFSET", "", [0.0, 0.2]),
                                    ("SLOPE", "", [0.01, 0.0])], {"MJDREFI": 51910, "MJDREFF": 7.4287037e-4})
            for f in filt]),
    }
    for band, (f, arf) in swift.BANDS.items():
        scale = 0.9 if band == "white" else 1.0
        files[root + "cpf/arf/" + arf] = _hdul(_table("SPECRESP" + f, [
            ("ENERG_LO", "keV", 12.3984 / (GRID[::-1] + 5)), ("ENERG_HI", "keV", 12.3984 / (GRID[::-1] - 5)),
            ("WAVE_MIN", "angstrom", GRID[::-1] + 5), ("WAVE_MAX", "angstrom", GRID[::-1] - 5),
            ("SPECRESP", "cm**2", (scale * filt[f] * instea)[::-1])], {"FILTER": f, "CONTENT": "SPECRESP"}))
    return files


def _pds(rows):
    return "PDS_VERSION_ID = PDS3\nEND\n\\begindata\n" + "".join(" ".join(f"{x:.6f}" for x in r) + "\n" for r in rows)


def _cassini_files(camera):
    nm = np.arange(200.0, 1057.05, 0.1)
    qe = np.clip(0.5 - np.abs(nm - 650.0) / 1000.0, 0.01, None)
    optics = np.full(nm.size, 0.6)
    grid = np.arange(200.0, 1101.0)
    files = {cassini.QE_URL: _pds(np.c_[grid, np.interp(grid, nm, qe), np.interp(grid, nm, qe)]),
             cassini.EFFICIENCY + cassini.CAMERAS[camera][1]: _pds(np.c_[np.arange(200.0, 1100.1, 2.5),
                                                                          np.full(361, 0.6)])}
    pairs = {}
    for i, band in enumerate(cassini.BANDS[camera]):
        centre = 1000.0 if band == "IR4" else 300.0 + 15.0 * i
        pairs[band] = 0.9 * np.exp(-0.5 * ((nm - centre) / 20.0) ** 2)
        system = qe * optics * pairs[band]
        files[cassini.systrans_url(camera, band)] = _pds(np.c_[nm, system, system])
    return files, nm, pairs
