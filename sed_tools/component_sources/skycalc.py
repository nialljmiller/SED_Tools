"""Telluric transmission grid from the ESO SkyCalc sky model.

SkyCalc (Noll et al. 2012; Jones et al. 2013) models the Cerro Paranal
atmosphere.  Its ``observatory`` option only changes the site altitude, so the
"sites" here are SkyCalc's four altitude settings; observatories elsewhere
should pick the closest altitude and a PWV typical of their site.

These grids are *derived*: SED_Tools computes them from a model, unlike the
standard atmospheres that instrument teams publish (see
:mod:`sed_tools.component_sources.team_atmospheres`).  Each grid manifest
records ``kind="derived"``, the model, the SkyCalc version reported in the
output FITS header, the fixed model parameters and the citation.

Telluric lines are unresolved at the resolutions used for photometry, so
point-sampling a low-resolution model biases band transmissions (~1e-3 in
2MASS J at R=20000).  Spectra are therefore fetched at ``fetch_resolution``
(default R=200000, converged to ~1e-4 against R=1e6) and bin-averaged onto a
log-uniform grid at ``store_resolution``; bin averages preserve the integral.

SkyCalc is reached through ESO's JSON API (the host ``skycalc_cli`` and
``skycalc_ipy`` use) or, when that host does not answer, through the SkyCalc
web form at ``www.eso.org``, which runs the same calculator.

Run where the data is curated; it is resumable (existing files are kept)::

    sed-tools filters-build-components skycalc --sites paranal lasilla
"""

from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from requests import RequestException

from ..filter_components import ATMOSPHERE, AtmosphereGrid, AtmosphereSpec, DERIVED
from ..filter_metadata import ENERGY, write_filter_file

logger = logging.getLogger(__name__)

#: SkyCalc JSON API (``skycalc_cli``/``skycalc_ipy``; SKYCALC_CLI_BASE_URL there).
BASE_URL = "https://etimecalret-002.eso.org/observing/etc"
#: The SkyCalc web form's action, on ESO's main web server.
FORM_URL = "https://www.eso.org/observing/etc/bin/simu/skycalc"
FORM_HOST = "https://www.eso.org"

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
#: PWV values SkyCalc accepts, spelt as its web form's options.
PWV_OPTIONS = ("0.05", "0.10", "0.25", "0.50", "1.0", "1.5", "2.5", "3.5", "5.0", "7.5", "10.0", "20.0", "30.0")

WMIN_NM, WMAX_NM = 300.0, 30000.0  # SkyCalc's range

MODEL = "ESO SkyCalc Sky Model Calculator (Cerro Paranal Advanced Sky Model)"
CITATION = ("Noll et al. 2012, A&A 543, A92 (2012A&A...543A..92N); "
            "Jones et al. 2013, A&A 560, A91 (2013A&A...560A..91J)")

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
#: Parameters that vary per request; everything else in _BASE_PARAMS is fixed.
_VARIABLE = ("airmass", "pwv", "observatory", "wmin", "wmax", "wres")


def pwv_option(pwv: float) -> str:
    """The SkyCalc spelling of a PWV value; SkyCalc offers a fixed list."""
    for option in PWV_OPTIONS:
        if math.isclose(float(option), pwv, rel_tol=0, abs_tol=1e-9):
            return option
    raise ValueError(f"SkyCalc accepts pwv in {PWV_OPTIONS} mm, not {pwv:g}")


def airmass_altitude(airmass: float) -> float:
    """Target altitude (deg) for an airmass, as SkyCalc's form computes it (sec z)."""
    if not 1.0 <= airmass <= 3.0:
        raise ValueError(f"SkyCalc accepts airmass 1-3, not {airmass:g}")
    if airmass > 2.9964:  # the form's own special case near airmass 3
        return 19.5
    return round(90.0 - math.degrees(math.acos(1.0 / airmass)), 3)


def _form_fields(params: Dict[str, object]) -> Dict[str, str]:
    """Translate API parameters into the SkyCalc web form's fields."""
    yes = lambda key: params[key] == "Y"
    fields = {
        "INS.NAME": "SKYCALC", "INS.MODE": "swspectr", "POSTFILE.FLAG": "0",
        "TEL.SITE.HEIGHT": str(params["observatory"]),
        "SKYMODEL.TARGET.AIRMASS": f"{float(params['airmass']):.2f}",
        "SKYMODEL.TARGET.ALT": f"{airmass_altitude(float(params['airmass'])):.3f}",
        "SKYMODEL.SEASON": str(params["season"]), "SKYMODEL.TIME": str(params["time"]),
        "SKYMODEL.PWV": pwv_option(float(params["pwv"])),
        "SKYMODEL.MSOLFLUX": f"{float(params['msolflux']):.2f}",
        "SKYMODEL.MOON.SUN.SEP": f"{float(params['moon_sun_sep']):g}",
        "SKYMODEL.MOON.TARGET.SEP": f"{float(params['moon_target_sep']):g}",
        "SKYMODEL.MOON.ALT": f"{float(params['moon_alt']):g}",
        "SKYMODEL.MOON.EARTH.DIST": f"{float(params['moon_earth_dist']):.3f}",
        "SKYMODEL.ECL.LON": f"{float(params['ecl_lon']):g}",
        "SKYMODEL.ECL.LAT": f"{float(params['ecl_lat']):g}",
        "SKYMODEL.VACAIR": str(params["vacair"]),
        "SKYMODEL.WAVELENGTH.MIN": f"{float(params['wmin']):.4f}",
        "SKYMODEL.WAVELENGTH.MAX": f"{float(params['wmax']):.4f}",
        "SKYMODEL.WAVELENGTH.GRID.MODE": str(params["wgrid_mode"]),
        "SKYMODEL.WAVELENGTH.DELTA": f"{float(params['wdelta']):.2f}",
        "SKYMODEL.WAVELENGTH.RESOLUTION": str(int(params["wres"])),
        "SKYMODEL.LSF.KERNEL.TYPE": str(params["lsf_type"]),
        "SKYMODEL.LSF.GAUSS.FWHM": f"{float(params['lsf_gauss_fwhm']):.2f}",
        "SKYMODEL.LSF.BOXCAR.WIDTH": f"{float(params['lsf_boxcar_fwhm']):.2f}",
        "SKYCALC.TRANS.PLOT.FLAG": "1",
    }
    for i in (1, 2, 3):
        fields[f"SKYMODEL.THERMAL.T{i}"] = f"{float(params[f'therm_t{i}']):.1f}"
        fields[f"SKYMODEL.THERMAL.E{i}"] = f"{float(params[f'therm_e{i}']):.2f}"
    # Unticked checkboxes are simply absent from a form post.
    for key, name in (("incl_moon", "MOON"), ("incl_starlight", "STARLIGHT"), ("incl_zodiacal", "ZODIACAL"),
                      ("incl_loweratm", "MOLEC.EMIS.LOWER.ATM"), ("incl_upperatm", "MOLEC.EMIS.UPPER.ATM"),
                      ("incl_airglow", "AIRGLOW"), ("incl_therm", "THERMAL")):
        if yes(key):
            fields[f"SKYMODEL.INCL.{name}"] = "Y"
    return fields


def fits_comments(content: bytes) -> List[str]:
    """COMMENT cards of the primary header of a SkyCalc ``skytable.fits``."""
    from astropy.io import fits

    with fits.open(io.BytesIO(content)) as hdul:
        return [str(c) for c in hdul[0].header.get("COMMENT", [])]


def fits_version(comments: Sequence[str]) -> Optional[str]:
    for line in comments:
        m = re.search(r"SkyCalc version:\s*(\S+)", line)
        if m:
            return m.group(1)
    return None


def check_form_echo(comments: Sequence[str], fields: Dict[str, str]) -> None:
    """The form echoes its inputs into the FITS header; refuse a mismatch."""
    echoed = {}
    for line in comments:
        if "=" in line and line.split("=", 1)[0].strip().isupper():
            key, value = line.split("=", 1)
            echoed[key.strip()] = value.strip()
    for key in ("TEL.SITE.HEIGHT", "SKYMODEL.TARGET.AIRMASS", "SKYMODEL.WAVELENGTH.RESOLUTION"):
        if echoed.get(key) != fields[key]:
            raise RuntimeError(f"SkyCalc echoed {key}={echoed.get(key)!r}, requested {fields[key]!r}")
    for key in ("SKYMODEL.PWV", "SKYMODEL.WAVELENGTH.MIN", "SKYMODEL.WAVELENGTH.MAX"):
        if key not in echoed or not math.isclose(float(echoed[key]), float(fields[key]), rel_tol=1e-6):
            raise RuntimeError(f"SkyCalc echoed {key}={echoed.get(key)!r}, requested {fields[key]!r}")


@dataclass
class SkyCalcResult:
    """One SkyCalc run: the transmission and what is needed to archive it."""

    wavelength_nm: np.ndarray
    transmission: np.ndarray
    content: bytes = b""            # the raw skytable.fits
    url: str = ""                   # where the FITS was downloaded
    interface: str = ""             # "api" or "form"
    request: Dict[str, object] = field(default_factory=dict)
    version: Optional[str] = None   # "SkyCalc version" from the FITS header
    retrieved_utc: str = ""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SkyCalcClient:
    """Run SkyCalc through its JSON API, falling back to the web form.

    ``interface`` is ``"api"``, ``"form"`` or ``"auto"`` (API first; switch to
    the form for the rest of the session if the API host cannot be reached).
    """

    def __init__(self, session=None, timeout: float = 600.0, retries: int = 4,
                 base_url: str = BASE_URL, form_url: str = FORM_URL, interface: str = "auto",
                 pause: float = 1.0):
        if session is None:
            import requests

            session = requests.Session()
        if interface not in ("auto", "api", "form"):
            raise ValueError("interface must be 'auto', 'api' or 'form'")
        self.session = session
        self.timeout = timeout
        self.retries = retries
        self.base_url = base_url.rstrip("/")
        self.form_url = form_url
        self.interface = interface
        self.pause = pause

    def transmission(
        self, site: str, airmass: float, pwv: float, wmin_nm: float, wmax_nm: float, resolution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (wavelength_nm, transmission) for one SkyCalc run."""
        result = self.fetch(site, airmass, pwv, wmin_nm, wmax_nm, resolution)
        return result.wavelength_nm, result.transmission

    def fetch(self, site: str, airmass: float, pwv: float, wmin_nm: float, wmax_nm: float,
              resolution: float) -> SkyCalcResult:
        params = dict(_BASE_PARAMS)
        params.update(
            airmass=float(airmass), pwv=float(pwv), observatory=SITES[site][0],
            wmin=float(wmin_nm), wmax=float(wmax_nm), wres=int(resolution),
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            use_form = self.interface == "form"
            try:
                result = self._form(params) if use_form else self._api(params)
                time.sleep(self.pause)
                return result
            except RuntimeError as exc:
                if "rejected" in str(exc) or "echoed" in str(exc):
                    raise
                last_error = exc
            except (RequestException, OSError) as exc:
                last_error = exc
                if self.interface == "auto" and not use_form and _unreachable(exc):
                    logger.warning("SkyCalc API host %s unreachable (%s); using the web form", self.base_url, exc)
                    self.interface = "form"
                    continue
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"SkyCalc request failed after {self.retries} attempts: {last_error}")

    def _api(self, params: Dict[str, object]) -> SkyCalcResult:
        tmpdir = None
        try:
            resp = self.session.post(f"{self.base_url}/api/skycalc", json=params, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") != "success":
                raise RuntimeError(f"SkyCalc rejected {params}: {result.get('error')}")
            tmpdir = result["tmpdir"]
            url = f"{self.base_url}/tmp/{tmpdir}/skytable.fits"
            content = self._download(url)
            return self._result(content, url, "api", dict(params))
        finally:
            if tmpdir:
                try:
                    self.session.get(f"{self.base_url}/api/rmtmp", params={"d": tmpdir}, timeout=60)
                except RequestException:
                    logger.debug("Could not delete SkyCalc tmpdir %s", tmpdir)

    def _form(self, params: Dict[str, object]) -> SkyCalcResult:
        fields = _form_fields(params)
        resp = self.session.post(self.form_url, data=fields, timeout=self.timeout)
        resp.raise_for_status()
        m = re.search(r'href=["\']?(/observing/etc/tmp/[^"\'\s>]+/skytable\.fits)', resp.text)
        if not m:
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", resp.text))[:300]
            raise RuntimeError(f"SkyCalc form returned no skytable.fits: {text}")
        url = FORM_HOST + m.group(1)
        content = self._download(url)
        check_form_echo(fits_comments(content), fields)
        return self._result(content, url, "form", {"form_url": self.form_url, **fields})

    def _download(self, url: str) -> bytes:
        content = b""
        for _ in range(5):
            content = self.session.get(url, timeout=self.timeout).content
            if content[:6] == b"SIMPLE":
                return content
            time.sleep(2)
        raise RuntimeError("SkyCalc output was not a FITS file")

    @staticmethod
    def _result(content: bytes, url: str, interface: str, request: Dict[str, object]) -> SkyCalcResult:
        from astropy.io import fits

        with fits.open(io.BytesIO(content)) as hdul:
            data = hdul[1].data
            lam = np.asarray(data["lam"], dtype=float)
            trans = np.asarray(data["trans"], dtype=float)
        return SkyCalcResult(lam, trans, content, url, interface, request,
                             fits_version(fits_comments(content)), _utc_now())


def _unreachable(exc: Exception) -> bool:
    from requests.exceptions import ConnectionError as ReqConnectionError, ConnectTimeout

    return isinstance(exc, (ConnectTimeout, ReqConnectionError))


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


class _Archive:
    """Raw SkyCalc FITS, gzipped, with a provenance manifest (URL, UTC time, SHA-256)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"files": []}

    def add(self, name: str, result: SkyCalcResult, **extra) -> None:
        stored = gzip.compress(result.content, 6, mtime=0)
        (self.root / name).write_bytes(stored)
        self.data["files"] = [f for f in self.data["files"] if f["file"] != name]
        self.data["files"].append({
            "file": name, "url": result.url, "retrieved_utc": result.retrieved_utc,
            "sha256": hashlib.sha256(result.content).hexdigest(), "bytes": len(result.content),
            "stored_as": "gzip", "stored_sha256": hashlib.sha256(stored).hexdigest(),
            "interface": result.interface, "skycalc_version": result.version,
            "request": result.request, **extra,
        })
        self.path.write_text(json.dumps(self.data, indent=1) + "\n")


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
    client=None,
    archive_dir: Union[str, Path, None] = None,
) -> Path:
    """Fetch and store a (site, airmass, PWV) telluric grid.

    ``atmosphere_root`` defaults to ``<filter_root>/../atmospheres``.  With
    ``archive_dir`` every raw SkyCalc FITS is kept there (gzipped) and listed
    in ``archive_dir/manifest.json`` with its URL, retrieval time and SHA-256.
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
        archive = _Archive(Path(archive_dir) / site) if archive_dir else None
        grid_path = site_dir / "manifest.json"
        grid = AtmosphereGrid.load(site_dir) if grid_path.exists() else AtmosphereGrid(site=site, entries=[])
        grid.source = (
            "ESO SkyCalc sky model (Noll et al. 2012, A&A 543, A92; Jones et al. 2013, A&A 560, A91), "
            f"fetched at R={fetch_resolution:g}, bin-averaged to R={store_resolution:g}"
        )
        grid.altitude_m = SITES[site][1]
        grid.notes = "Cerro Paranal climatology at this altitude; transmission only (no emission)."
        grid.kind = DERIVED
        grid.model = MODEL
        grid.citation = CITATION
        fixed = {k: v for k, v in _BASE_PARAMS.items() if k not in _VARIABLE}
        grid.parameters = {
            **fixed, "observatory": SITES[site][0], "fetch_resolution": fetch_resolution,
            "store_resolution": store_resolution, "chunks": chunks, "wmin_nm": WMIN_NM, "wmax_nm": WMAX_NM,
            "below_wmin": "transmission set to 0 below 300 nm (ozone; SkyCalc's limit)",
            "clip": "bin averages clipped to [0, 1]",
        }
        known = {e["file"] for e in grid.entries}
        versions = set(filter(None, [grid.model_version]))
        interfaces = set()

        for airmass in airmasses:
            for pwv in pwvs:
                name = _file_name(airmass, pwv)
                path = site_dir / name
                if not path.exists():
                    logger.info("SkyCalc %s airmass=%g pwv=%g", site, airmass, pwv)
                    wl_parts: List[np.ndarray] = []
                    tr_parts: List[np.ndarray] = []
                    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                        if hasattr(client, "fetch"):
                            result = client.fetch(site, airmass, pwv, lo, hi, fetch_resolution)
                        else:
                            result = SkyCalcResult(*client.transmission(site, airmass, pwv, lo, hi, fetch_resolution))
                        if result.version:
                            versions.add(result.version)
                        if result.interface:
                            interfaces.add(f"{result.interface}: "
                                           + (client.form_url if result.interface == "form" else client.base_url))
                        if archive is not None and result.content:
                            archive.add(f"{Path(name).stem}_chunk{i}.fits.gz", result, site=site, airmass=airmass,
                                        pwv=pwv, chunk=i, wmin_nm=float(lo), wmax_nm=float(hi))
                        wl, tr = bin_average(result.wavelength_nm, result.transmission, store_resolution, lo, hi)
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
                            "ComponentsSource": DERIVED,
                            "Atmosphere": f"site={site} airmass={airmass:g} pwv={pwv:g}",
                            "Sources": "ESO SkyCalc" + (f" {result.version}" if result.version else ""),
                            "Recipe": spec.label,
                        },
                    )
                if len(versions) > 1:
                    raise RuntimeError(f"SkyCalc version changed during the {site} grid: {sorted(versions)}")
                grid.model_version = next(iter(versions), grid.model_version)
                if interfaces:
                    grid.parameters["interface"] = "; ".join(sorted(interfaces))
                if name not in known:
                    grid.entries.append({"airmass": airmass, "pwv": pwv, "file": name})
                    known.add(name)
                    grid.entries.sort(key=lambda e: (float(e["airmass"]), float(e["pwv"])))
                grid.save(site_dir)  # save as we go so the build is resumable
        grid.save(site_dir)
    return root
