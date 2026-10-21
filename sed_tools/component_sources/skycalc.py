"""Telluric transmission grid from the ESO SkyCalc sky model.

SkyCalc (Noll et al. 2012; Jones et al. 2013) models the Cerro Paranal
atmosphere.  Its ``observatory`` option only changes the site altitude, so the
"sites" here are SkyCalc's four altitude settings; observatories elsewhere
should pick the closest altitude and a PWV typical of their site.

Telluric lines are unresolved at the resolutions used for photometry, so
point-sampling a low-resolution model biases band transmissions (~1e-3 in
2MASS J at R=20000).  Spectra are therefore fetched at ``fetch_resolution``
(default R=200000, converged to ~1e-4 against R=1e6) and bin-averaged onto a
log-uniform grid at ``store_resolution``; bin averages preserve the integral.

Run where the data is curated; it is resumable (existing files are kept)::

    sed-tools filters-build-components skycalc --sites paranal lasilla
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from requests import RequestException

from ..filter_components import ATMOSPHERE, AtmosphereGrid, AtmosphereSpec
from ..filter_metadata import ENERGY, write_filter_file

logger = logging.getLogger(__name__)

BASE_URL = "https://etimecalret-002.eso.org/observing/etc"

#: SkyCalc observatory codes (altitude in m) by site key.
SITES: Dict[str, Tuple[str, float]] = {
    "lasilla": ("2400", 2400.0),
    "paranal": ("2640", 2640.0),
    "armazones": ("3060", 3060.0),
    "highanddry": ("5000", 5000.0),
}

DEFAULT_AIRMASSES = (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
#: SkyCalc's own PWV grid (mm)
DEFAULT_PWVS = (0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0)

WMIN_NM, WMAX_NM = 300.0, 30000.0  # SkyCalc's range

#: Defaults sent by ESO's own client (skycalc_ipy); the transmission does not
#: depend on the emission switches.
_BASE_PARAMS = {
    "airmass": 1.0, "pwv_mode": "pwv", "season": 0, "time": 0, "pwv": 3.5, "msolflux": 130.0,
    "incl_moon": "N", "moon_sun_sep": 90.0, "moon_target_sep": 45.0, "moon_alt": 45.0,
    "moon_earth_dist": 1.0, "incl_starlight": "N", "incl_zodiacal": "N", "ecl_lon": 135.0,
    "ecl_lat": 90.0, "incl_loweratm": "N", "incl_upperatm": "N", "incl_airglow": "N",
    "incl_therm": "N", "therm_t1": 0.0, "therm_e1": 0.0, "therm_t2": 0.0, "therm_e2": 0.0,
    "therm_t3": 0.0, "therm_e3": 0.0, "vacair": "vac", "wmin": WMIN_NM, "wmax": WMAX_NM,
    "wgrid_mode": "fixed_spectral_resolution", "wdelta": 0.1, "wres": 20000,
    "lsf_type": "none", "lsf_gauss_fwhm": 5.0, "lsf_boxcar_fwhm": 5.0, "observatory": "2640",
}


class SkyCalcClient:
    def __init__(self, session=None, timeout: float = 600.0, retries: int = 4):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.timeout = timeout
        self.retries = retries

    def transmission(
        self, site: str, airmass: float, pwv: float, wmin_nm: float, wmax_nm: float, resolution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (wavelength_nm, transmission) for one SkyCalc run."""
        from astropy.io import fits

        params = dict(_BASE_PARAMS)
        params.update(
            airmass=float(airmass), pwv=float(pwv), observatory=SITES[site][0],
            wmin=float(wmin_nm), wmax=float(wmax_nm), wres=int(resolution),
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            tmpdir = None
            try:
                resp = self.session.post(f"{BASE_URL}/api/skycalc", json=params, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json()
                if result.get("status") != "success":
                    raise RuntimeError(f"SkyCalc rejected {params}: {result.get('error')}")
                tmpdir = result["tmpdir"]
                content = b""
                for _ in range(5):
                    content = self.session.get(f"{BASE_URL}/tmp/{tmpdir}/skytable.fits", timeout=self.timeout).content
                    if content[:6] == b"SIMPLE":
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError("SkyCalc output was not a FITS file")
                data = fits.open(io.BytesIO(content))[1].data
                return np.asarray(data["lam"], dtype=float), np.asarray(data["trans"], dtype=float)
            except RuntimeError as exc:
                if "rejected" in str(exc):
                    raise
                last_error = exc
            except (RequestException, OSError) as exc:
                last_error = exc
            finally:
                if tmpdir:
                    try:
                        self.session.get(f"{BASE_URL}/api/rmtmp", params={"d": tmpdir}, timeout=60)
                    except RequestException:
                        logger.debug("Could not delete SkyCalc tmpdir %s", tmpdir)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"SkyCalc request failed after {self.retries} attempts: {last_error}")


def bin_average(
    wavelength: np.ndarray, values: np.ndarray, resolution: float, lo: float, hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Average ``values`` in log-uniform bins of width 1/``resolution`` in ln(lambda).

    Input samples are assumed (near-)uniform in ln(lambda), as SkyCalc's
    fixed-resolution grid is, so the sample mean is the bin integral average.
    """
    n_bins = int(np.ceil(np.log(hi / lo) * resolution))
    edges = lo * np.exp(np.arange(n_bins + 1) / resolution)
    idx = np.searchsorted(edges, wavelength, side="right") - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], weights=values[ok], minlength=n_bins)
    counts = np.bincount(idx[ok], minlength=n_bins)
    filled = counts > 0
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres[filled], sums[filled] / counts[filled]


def _file_name(airmass: float, pwv: float) -> str:
    return f"am{airmass:.2f}_pwv{pwv:g}.dat"


def build(
    filter_root: Union[str, Path],
    *,
    atmosphere_root: Union[str, Path, None] = None,
    sites: Sequence[str] = ("paranal",),
    airmasses: Iterable[float] = DEFAULT_AIRMASSES,
    pwvs: Iterable[float] = DEFAULT_PWVS,
    fetch_resolution: float = 200000,
    store_resolution: float = 20000,
    chunks: int = 8,
    client: Optional[SkyCalcClient] = None,
) -> Path:
    """Fetch and store a (site, airmass, PWV) telluric grid.

    ``atmosphere_root`` defaults to ``<filter_root>/../atmospheres``.
    """
    root = Path(atmosphere_root) if atmosphere_root else Path(filter_root).parent / "atmospheres"
    client = client or SkyCalcClient()
    airmasses = [float(a) for a in airmasses]
    pwvs = [float(p) for p in pwvs]
    # Split the range so each request stays a manageable size at high R.
    edges = WMIN_NM * (WMAX_NM / WMIN_NM) ** (np.arange(chunks + 1) / chunks)

    for site in sites:
        if site not in SITES:
            raise ValueError(f"Unknown SkyCalc site '{site}'. Choose from {sorted(SITES)}")
        site_dir = root / site
        site_dir.mkdir(parents=True, exist_ok=True)
        grid_path = site_dir / "manifest.json"
        grid = AtmosphereGrid.load(site_dir) if grid_path.exists() else AtmosphereGrid(site=site, entries=[])
        grid.source = (
            "ESO SkyCalc sky model (Noll et al. 2012, A&A 543, A92; Jones et al. 2013, A&A 560, A91), "
            f"fetched at R={fetch_resolution:g}, bin-averaged to R={store_resolution:g}"
        )
        grid.altitude_m = SITES[site][1]
        grid.notes = "Cerro Paranal climatology at this altitude; transmission only (no emission)."
        known = {e["file"] for e in grid.entries}

        for airmass in airmasses:
            for pwv in pwvs:
                name = _file_name(airmass, pwv)
                path = site_dir / name
                if not path.exists():
                    logger.info("SkyCalc %s airmass=%g pwv=%g", site, airmass, pwv)
                    wl_parts: List[np.ndarray] = []
                    tr_parts: List[np.ndarray] = []
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        lam, trans = client.transmission(site, airmass, pwv, lo, hi, fetch_resolution)
                        wl, tr = bin_average(lam, trans, store_resolution, lo, hi)
                        wl_parts.append(wl)
                        tr_parts.append(tr)
                    wl_nm = np.concatenate(wl_parts)
                    tr = np.concatenate(tr_parts)
                    order = np.argsort(wl_nm)
                    wl_nm, tr = wl_nm[order], np.clip(tr[order], 0.0, 1.0)
                    # The atmosphere is opaque below SkyCalc's 300 nm limit (ozone).
                    wl_ang = np.concatenate([[1000.0, WMIN_NM * 10 - 1.0], wl_nm * 10])
                    tr = np.concatenate([[0.0, 0.0], tr])
                    spec = AtmosphereSpec(site, airmass, pwv)
                    write_filter_file(
                        path, wl_ang, tr,
                        detector_type=ENERGY, source="component", wavelength_unit="Angstrom",
                        extra={
                            "Role": ATMOSPHERE,
                            "Components": ATMOSPHERE,
                            "Atmosphere": f"site={site} airmass={airmass:g} pwv={pwv:g}",
                            "Sources": "ESO SkyCalc",
                            "Recipe": spec.label,
                        },
                    )
                if name not in known:
                    grid.entries.append({"airmass": airmass, "pwv": pwv, "file": name})
                    known.add(name)
                    grid.entries.sort(key=lambda e: (float(e["airmass"]), float(e["pwv"])))
                    grid.save(site_dir)  # save as we go so the build is resumable
        grid.save(site_dir)
    return root
