"""Fakes shared by the team-atmosphere tests: a fake response, curves and sites, and a
SkyCalc form session.
"""

from __future__ import annotations

import io

import numpy as np

from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere
from sed_tools.filter_components import TEAM


class _Resp:
    def __init__(self, content, status=200, text=None):
        self.content = content if isinstance(content, bytes) else content.encode("latin-1")
        self.status_code = status
        self.headers = {}
        self.text = text if text is not None else self.content.decode("latin-1")

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


def _curve(values):
    return lambda session: (np.array([3000.0, 5000.0, 7000.0]), np.array(values))


def _site(name="here_team", entries=None, **kw):
    entries = entries if entries is not None else [(1.2, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), "x"))]
    return SiteAtmosphere(site=name, altitude_m=1000.0, entries=entries, source="test team",
                          kind=TEAM, model="MODTRAN", model_version="not stated", citation="Someone 2001",
                          **kw)


def _skytable(fields, lam, trans, version="2.0.9"):
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    primary.header.add_comment(f"SkyCalc version: {version}")
    for key, value in fields.items():
        primary.header.add_comment(f"{key}={value}")
    table = fits.BinTableHDU.from_columns([fits.Column("lam", "D", array=lam), fits.Column("trans", "D", array=trans)])
    buf = io.BytesIO()
    fits.HDUList([primary, table]).writeto(buf)
    return buf.getvalue()


class _FormSession:
    """The SkyCalc form: a POST returns HTML linking skytable.fits, which echoes its inputs."""

    def __init__(self, echo_override=None):
        self.posts, self.echo_override = [], echo_override or {}

    def post(self, url, data=None, timeout=None, **kw):
        self.posts.append(dict(data))
        return _Resp(b"<h4><a href=/observing/etc/tmp/abc/skytable.fits> skytable.fits</a></h4>")

    def get(self, url, timeout=None, **kw):
        fields = {**self.posts[-1], **self.echo_override}
        lo, hi = float(fields["SKYMODEL.WAVELENGTH.MIN"]), float(fields["SKYMODEL.WAVELENGTH.MAX"])
        lam = np.geomspace(lo, hi, 200)
        return _Resp(_skytable(fields, lam, np.full_like(lam, 0.8 ** float(fields["SKYMODEL.TARGET.AIRMASS"]))))
