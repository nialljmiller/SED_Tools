"""Argus Array: ``Argus/DR280`` (current hardware) and ``Argus/System`` (legacy).

Argus/DR280
    The survey bands ``beta`` (Chroma CT446-162bp) and ``rho`` (Chroma
    CT699-259bp) on PlaneWave DeltaRho 280 (DR280) telescopes with Sony
    IMX461 sensors.  Each factor is a separate component, and filter
    variants select alternative filter curves:

    ``filter``  ``production`` (default): argus_sim's ``argus_beta_ct446.csv``
                and ``argus_rho_ct699.csv``, the vendor design curves
                averaged over the DR280 beam.  ``addem``: the ADDem
                demonstrator, which flew the same CT446 beta filter and the
                earlier CT692-276bp red filter (``rho0``); its rho is
                synthref's beam-averaged rho0 curve.
    ``optics``  the DR280 vendor-coating model ``dr280_vendor_s2s2`` (hdps
                ``addem_optics.csv``), which ADDem and production share.
    ``qe``      ``imx455_qe.csv``.  The IMX455 stands in for the deployed
                IMX461, for which no QE curve is available.

    The published ``beta.dat`` and ``rho.dat`` are the default filter x
    optics x qe, composed from the component files by :func:`compose_band`,
    so composing the defaults reproduces them exactly.  No atmosphere.

    Only the production filters and the QE are public (argus-hdps/argus_sim,
    MIT); they are downloaded at a pinned commit.  The optics curve and the
    rho0 curve are not public: they are read from ``raw``, and without them
    the build stops and says to install Argus/DR280 from the NJM mirror,
    which serves them.

Argus/System
    Nill's beta and rho curves of 30 September 2026, which MESA_ARGUS,
    argus_star and custom_colors read: ArgusSim's ``argus_g.csv`` (beta) or
    ``bandpass_560_825nm.csv`` (an idealised rho box) x
    ``optics_throughput.csv`` (a PlaneWave CDK model) x ``imx455_qe.csv``.
    This builder adds their components from those four files in ``raw`` and
    writes SED_Tools metadata on the curves' column row (line 1, which
    released MESA discards); their samples are never touched.  Three inputs
    are public at argus_sim commit 785c92e8; ``optics_throughput.csv`` is
    only in the private ArgusSim repository.  All four are archived on NJM
    (``source_evidence/filters/2026-10-27-argus``).

Both instruments are labelled energy-counting (``DetectorType=energy``,
``DetectorTypeSource=user``): the Argus team states that the Argus sensors
are energy counting (N. Miller, 2026-10-27), although argus_sim's
throughput.py describes the product as a photon-rate response.  A ``user``
type is never replaced by an SVO resolution, and products composed from the
components keep it.

Every input is pinned by SHA-256, and a different file stops the build.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

from ..filters.combine import write_filter_index
from ..filters.compose import (FILTER, OPTICS, QE, SHARED, InstrumentManifest, add_component,
                                 compose_band)
from ..filters.curve_metadata import ENERGY, read_filter_header, write_filter_file, write_filter_header
from ._common import register_variant
from ._original import CachingSession

FACILITY = "Argus"
BANDS = ("beta", "rho")

ARGUS_SIM_REPO = "https://github.com/argus-hdps/argus_sim"
ARGUS_SIM_COMMIT = "37a25ebbdc4f5151d5845e1cb1ccfc4c1abf7b85"
ARGUS_SIM_SYSTEM_COMMIT = "785c92e8e423055e933d19f934054ca76ebd20e3"
ARGUS_SIM_RAW = ("https://raw.githubusercontent.com/argus-hdps/argus_sim/{commit}"
                 "/src/argus_sim/data/throughputs/{file}")
NJM_HINT = ("Install the curves and components from the NJM mirror instead "
            "(sed_tools.Filters.fetch('Argus', '{instrument}')), or pass raw= (--raw) a directory "
            "holding these files.")

Curve = Tuple[np.ndarray, np.ndarray]  # (Angstrom, fraction)


@dataclass(frozen=True)
class Input:
    """A tabulated input in nm, pinned by SHA-256.

    It is downloaded from ``url`` when one is given, and otherwise read from
    the ``raw`` directory as ``file``.  ``origin`` says where it comes from.
    """

    file: str
    sha256: str
    origin: str
    url: Optional[str] = None
    skip: int = 1        # header lines
    column: int = 1      # transmission column (column 0 is the wavelength)
    percent: bool = False

    @property
    def source(self) -> str:
        return self.url or self.origin


def _argus_sim(file: str, sha256: str, commit: str = ARGUS_SIM_COMMIT, **kw) -> Input:
    url = ARGUS_SIM_RAW.format(commit=commit, file=file)
    return Input(file, sha256, f"argus-hdps/argus_sim @{commit[:8]} (public, MIT)", url=url, **kw)


HDPS = "argus-hdps/hdps (private, BSD-3-Clause)"
ARGUSSIM = "argus-hdps/ArgusSim (private)"
SYNTHREF = "argus-hdps/synthref (removed from GitHub; no licence file)"

CDK_ORIGIN = (f"{ARGUSSIM} src/argus_sim/data/throughputs/optics_throughput.csv @d3255b74; byte-identical "
              f"to {HDPS} Argus_optics_throughput.csv on unmerged branches")

#: Argus/DR280 inputs by component file stem.
DR280_INPUTS: Dict[str, Input] = {
    "beta_ct446": _argus_sim("argus_beta_ct446.csv",
                             "415924a517adbbf4e392f485d0c9b2ae4d89004ff6f1b37f0358b0bb1464d879"),
    "rho_ct699": _argus_sim("argus_rho_ct699.csv",
                            "e4d2e035f8f87a824a203a07756e058699877962807c7be2412c7815228a328f"),
    "imx455": _argus_sim("imx455_qe.csv", "69970a4e9dbab1b42a9f9a27f01e30cc49cc724ce6c0a9613c5d1f55015b1838",
                         skip=0),
    "rho0_ct692": Input(
        "synthref_response_rho.csv", "0e3fe5043012b74a6071063d77bf6fa6023e42eee7be489b7a290478e67fbbef",
        f"{SYNTHREF} passbands/synthref_response_rho.csv @9b29bc7f, column filter_beam", column=2),
    "dr280": Input(
        "addem_optics.csv", "2a0b646ee16a9d2a856af3dfedcdecde3e0c613a0bcf94f35775d3b0964de739",
        f"{HDPS} packages/hdps_orchestrate/src/hdps/orchestrate/data/throughput/addem_optics.csv @8745a72c, "
        "on unmerged branches feat/photocal-gaia-xp and feature/gaia-xp-pipeline; numerically identical to "
        "the optics column of synthref's passbands"),
}

#: filter variant -> (band -> DR280_INPUTS key, description)
FILTER_VARIANTS: Dict[str, Tuple[Dict[str, str], str]] = {
    "production": ({"beta": "beta_ct446", "rho": "rho_ct699"},
                   "Production filters: Chroma CT446-162bp (beta) and CT699-259bp (rho), vendor design curves "
                   "averaged over the DR280 beam (argus_sim argus_beta_ct446.csv, argus_rho_ct699.csv)"),
    "addem": ({"beta": "beta_ct446", "rho": "rho0_ct692"},
              "ADDem demonstrator: the same CT446-162bp beta filter (production curve) and the CT692-276bp red "
              "filter rho0 in the rho band (synthref's beam-averaged rho0 curve)"),
}
DEFAULT_FILTER = "production"
OPTICS_KEY = "dr280"  # the only DR280 optics: ADDem and production share it

#: Nill's decision for both instruments (2026-10-27).  ``user`` is outside
#: the SVO precedence order, so an SVO resolution never replaces it, and
#: composed products keep it (filter_components._band_detector_type).
DETECTOR_TYPE, DETECTOR_SOURCE = ENERGY, "user"
DETECTOR_NOTE = (
    "user: the Argus team states that the Argus sensors are energy counting (N. Miller, 2026-10-27). "
    "This overrides argus_sim's throughput.py, which gives the IMX455 QE in electrons per photon "
    "(qe_freq = ... * (u.electron / u.photon)) and calls the product a 'photon-rate response'; synthref's "
    "passbands README also says 'Per-star rate ∝ ∫ F_λ · λ · R dλ (photon counting)'."
)

DR280_NOTES = (
    "Argus Array survey bands beta (Chroma CT446-162bp) and rho (Chroma CT699-259bp) on PlaneWave DeltaRho 280 "
    "(DR280) telescopes with Sony IMX461 sensors. The published curves are filter x optics x qe with the default "
    "filter variant ('production'), composed by SED_Tools from the component files (linear "
    "interpolation on the merged grid; zero outside a component's tabulated range), so composing the defaults "
    "reproduces them exactly. "
    "Filter variants: 'production' (default), argus_sim's argus_beta_ct446.csv and argus_rho_ct699.csv, the "
    "vendor design curves averaged over the DR280 beam; 'addem', the ADDem (Argus Deep-field Demonstrator) "
    "filters: the same CT446-162bp beta filter, with the production curve, and the earlier CT692-276bp red "
    "filter rho0 in the rho band, from the filter_beam column of synthref's synthref_response_rho.csv. "
    "synthref's README calls that column 'rho0 CT692-276 theory, AOI n_eff 1.6', while its code defaults say "
    "the measured lot 452749; which it is is unresolved. ADDem's own calibration curves (synthref, hdps) used "
    "a beta curve with a provisional +5 nm red edge; that curve is archive-only, so 'addem' beta equals "
    "'production' beta. Caveat: argus_rho_ct699.csv (2026-09-25) may use synthref's earlier single-field-angle "
    "beam model rather than the field average used for argus_beta_ct446.csv (2026-10-21). "
    "Optics: the DR280 vendor-coating model dr280_vendor_s2s2, which ADDem and production share (two "
    "'supplier 2' enhanced-aluminium mirrors digitised by eye from the vendor coating sheet, three CDGM corrector "
    "lenses, six AR surfaces at 0.5%; no window, since the filter substrate is in the filter curve and the "
    "sensor cover glass in the QE). It is a model; no measured optics exist. "
    "QE: Sony IMX455, digitised from the datasheet (version and figure not recorded). The deployed sensor is "
    "the IMX461; no IMX461 QE curve is available, so the IMX455 stands in, as in every Argus pipeline. "
    "DetectorType energy (user): the Argus team states that the Argus sensors are energy counting (N. Miller, "
    "2026-10-27); argus_sim's throughput.py describes the product as a photon-rate response. "
    "No atmosphere is published or offered: the only one in the Argus repositories "
    "(pachon_modtran_atm_aerosols.txt) is LSST's Cerro Pachon MODTRAN model at airmass 1.2 "
    "(syseng_throughputs siteProperties/pachonModtranAtm_12_aerosol.dat), labelled airmass 1.0, and not the "
    "Argus site. "
    "Not public: the optics curve (hdps unmerged branches) and the rho0 curve (synthref, removed from GitHub); "
    "NJM is their source. "
    "Archive-only, available on request: the 22 measured Chroma production-lot curves (order SO448371; 8 beta, "
    "14 rho), the legacy design-study bands (r, i, A, r+i, i+z, r+i+z and the 560-825 nm rho box) and the "
    "provisional red-edge beta curve. Nill's earlier curves, which his MESA projects use, are Argus/System; "
    "their PlaneWave CDK optics curve is Argus/System's optics component."
)

#: Argus/System inputs, read from ``raw``: role or band -> input.
SYSTEM_INPUTS: Dict[str, Input] = {
    "beta": Input("argus_g.csv", "17b780b86c231c1c0084ce22d2d4b77b383d55f44733a5d361af1078e94b0d6c",
                  f"{ARGUSSIM} @d3255b74; public at {ARGUS_SIM_REPO} @{ARGUS_SIM_SYSTEM_COMMIT[:8]} "
                  "(Chroma CT446-162bp collimated design curve, percent)", percent=True),
    "rho": Input("bandpass_560_825nm.csv", "f0b82e3c445c62d90738987af3743915621ff92dfe6aa9d496b206497d781ce6",
                 f"{ARGUSSIM} @0e4716a2; public at {ARGUS_SIM_REPO} @{ARGUS_SIM_SYSTEM_COMMIT[:8]} "
                 "(idealised 560-825 nm box, not the real rho filter)"),
    OPTICS: Input("optics_throughput.csv", "8cf704337544e227c7f4c5fdb8add77f44c064af233bca2d4659ce6b6af727e9",
                  CDK_ORIGIN + " (PlaneWave CDK model, not the DR280)"),
    QE: Input("imx455_qe.csv", "69970a4e9dbab1b42a9f9a27f01e30cc49cc724ce6c0a9613c5d1f55015b1838",
              f"{ARGUSSIM} @d3255b74; public at {ARGUS_SIM_REPO} @{ARGUS_SIM_SYSTEM_COMMIT[:8]} "
              "(Sony IMX455 QE, digitised)", skip=0),
}

SYSTEM_NOTES = (
    "Legacy: Nill's Argus beta and rho curves of 30 September 2026, which MESA_ARGUS, argus_star and "
    "custom_colors read. Their samples are served unchanged; only the column row (line 1, which released MESA "
    "discards) carries SED_Tools metadata, so MESA reads the same numbers. beta = argus_g.csv (Chroma "
    "CT446-162bp collimated design curve) and rho = bandpass_560_825nm.csv (an idealised 560-825 nm box, not the "
    "real rho filter), each x optics_throughput.csv (a PlaneWave CDK model, not the DR280) x imx455_qe.csv. "
    "DetectorType energy (user): the Argus team states that the Argus sensors are energy counting (N. Miller, "
    "2026-10-27); argus_sim's throughput.py describes the product as a photon-rate response. "
    "Published curves are ArgusSim's system response (PCHIP filters/optics, "
    "CubicSpline QE extrapolated to 300 nm). Composed products use linear "
    "interpolation and do not extrapolate the QE: filter x optics x qe differs from the published curves "
    "near the steep filter edges (by up to 0.072 in beta at 527.8 nm and 0.026 in rho at 561.7 nm; in-band rms "
    "0.0034 and 0.0015), and the pivot wavelengths agree within 0.4 A. The current hardware is Argus/DR280."
)


# --------------------------------------------------------------------- Inputs
def _parse(data: bytes, spec: Input) -> Curve:
    """(Angstrom, fraction) from a comma-separated table in nm (its bytes already match the pin)."""
    table = np.loadtxt(io.StringIO(data.decode("utf-8")), delimiter=",", skiprows=spec.skip,
                       usecols=(0, spec.column), ndmin=2)
    wavelength = table[:, 0] * 10.0
    transmission = table[:, 1] / 100.0 if spec.percent else table[:, 1]
    return wavelength, transmission


def _missing_local(inputs: Mapping[str, Input], raw: Optional[Path]) -> list:
    return [spec for spec in inputs.values()
            if spec.url is None and (raw is None or not (raw / spec.file).is_file())]


def _read_inputs(inputs: Mapping[str, Input], raw: Optional[Path], instrument: str,
                 fetch: Optional[CachingSession] = None) -> Dict[str, Curve]:
    """Every input as (Angstrom, fraction), SHA-256 checked; the local (non-public) ones before any download."""
    missing = _missing_local(inputs, raw)
    if missing:
        listing = "; ".join(f"{spec.file} ({spec.origin})" for spec in missing)
        where = f"in {raw}" if raw is not None else "(no raw directory given)"
        raise FileNotFoundError(
            f"{FACILITY}/{instrument} needs files that are not public, and they are missing {where}: "
            f"{listing}. " + NJM_HINT.format(instrument=instrument))
    curves = {}
    for key, spec in inputs.items():
        data = fetch.get(spec.url).content if spec.url else (raw / spec.file).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.sha256:
            raise ValueError(f"{spec.source}: SHA-256 {digest} is not the pinned {spec.sha256}; "
                             f"this is not the file {FACILITY}/{instrument} is built from")
        curves[key] = _parse(data, spec)
    return curves


# ---------------------------------------------------------------- Argus/DR280
def _label_detector(manifest: InstrumentManifest) -> None:
    """Record the Argus team's detector convention (both instruments) in ``manifest``."""
    manifest.detector_type = DETECTOR_TYPE
    manifest.detector_types = {band: DETECTOR_TYPE for band in BANDS}
    manifest.detector_type_sources = {band: DETECTOR_SOURCE for band in BANDS}
    manifest.sources["detector_type"] = DETECTOR_NOTE


def _add_dr280_components(inst_dir: Path, manifest: InstrumentManifest, curves: Mapping[str, Curve]) -> None:
    """Write every component file and register the defaults and the filter variants."""
    filter_files = {}
    for band, key in FILTER_VARIANTS[DEFAULT_FILTER][0].items():
        filter_files[key] = add_component(inst_dir, FILTER, *curves[key], band=band, name=key,
                                          source=DR280_INPUTS[key].source, manifest=manifest)
    for bands, _ in FILTER_VARIANTS.values():
        for band, key in bands.items():
            if key not in filter_files:  # a variant's own curve, not a default component
                filter_files[key] = add_component(inst_dir, FILTER, *curves[key], band=band, name=key,
                                                  source=DR280_INPUTS[key].source)
    for name, (bands, description) in FILTER_VARIANTS.items():
        register_variant(manifest, FILTER, name, {b: filter_files[k] for b, k in bands.items()},
                         description=description, sources={b: DR280_INPUTS[k].source for b, k in bands.items()},
                         **({"default": True} if name == DEFAULT_FILTER else {}))
    add_component(inst_dir, OPTICS, *curves[OPTICS_KEY], band=SHARED, name=OPTICS_KEY,
                  source=DR280_INPUTS[OPTICS_KEY].source, manifest=manifest)
    qe = DR280_INPUTS["imx455"]
    add_component(inst_dir, QE, *curves["imx455"], band=SHARED, name="imx455", manifest=manifest,
                  source=f"{qe.source} (Sony IMX455, digitised; stands in for the deployed IMX461)")


def _write_dr280(inst_dir: Path, curves: Mapping[str, Curve]) -> None:
    """Write the components, the published totals of the defaults, the MESA index and the manifest."""
    manifest = InstrumentManifest(facility=FACILITY, instrument="DR280", bands=list(BANDS), ground_based=True)
    _label_detector(manifest)
    _add_dr280_components(inst_dir, manifest, curves)
    for band, filter_key in FILTER_VARIANTS[DEFAULT_FILTER][0].items():
        total = compose_band(inst_dir, band, (FILTER, OPTICS, QE), manifest=manifest)
        sources = " | ".join(DR280_INPUTS[k].source for k in (filter_key, OPTICS_KEY, "imx455"))
        write_filter_file(inst_dir / f"{band}.dat", total.wavelength, total.transmission,
                          detector_type=DETECTOR_TYPE, source=DETECTOR_SOURCE, filter_id=f"{FACILITY}/DR280.{band}",
                          wavelength_unit="Angstrom",
                          extra={"Components": "filter+optics+qe", "ComponentsSource": "composed",
                                 "Sources": sources})
    write_filter_index(inst_dir, "DR280")

    manifest.published = {"components": "filter+optics+qe", "source": "composed"}
    manifest.published_bands = {band: "filter+optics+qe" for band in BANDS}
    manifest.sources["argus_sim"] = (f"{ARGUS_SIM_REPO}/tree/{ARGUS_SIM_COMMIT} (MIT, Copyright (c) 2026 "
                                     "The University of North Carolina at Chapel Hill)")
    manifest.notes = DR280_NOTES
    manifest.save(inst_dir)


def build_dr280(filter_root: Union[str, Path], raw: Union[str, Path, None] = None, *,
                session=None, archive_dir: Union[str, Path, None] = None) -> Path:
    """Write Argus/DR280: components, variants, published totals, MESA index and manifest.

    Everything is built in a temporary directory first.  A published curve
    or MESA index already installed is never replaced: if it differs from
    the new build, nothing is written and this raises.  ``archive_dir``
    keeps every download with its URL, retrieval time (UTC) and SHA-256.
    """
    raw = Path(raw) if raw else None
    curves = _read_inputs(DR280_INPUTS, raw, "DR280", CachingSession(session, archive_dir))
    inst_dir = Path(filter_root) / FACILITY / "DR280"
    with tempfile.TemporaryDirectory(prefix="argus-dr280-") as tmp:
        staged = Path(tmp) / "DR280"
        staged.mkdir()
        _write_dr280(staged, curves)
        for name in [f"{band}.dat" for band in BANDS] + ["DR280"]:
            installed = inst_dir / name
            if installed.exists() and installed.read_bytes() != (staged / name).read_bytes():
                raise RuntimeError(f"{installed} differs from a fresh build of its components and is never "
                                   "replaced; nothing was written. Build into an empty directory to compare.")
        shutil.copytree(staged, inst_dir, dirs_exist_ok=True)
    return inst_dir


# ------------------------------------------------------ Argus/System (legacy)
def system_header(band: str) -> Dict[str, str]:
    """The column-row metadata of the legacy ``Argus/System`` curve of ``band``.

    ``ComponentsSource=original``: the curves are the Argus team's own,
    stored as published.
    """
    return {"FilterID": f"{FACILITY}/System.{band}", "DetectorType": DETECTOR_TYPE,
            "DetectorTypeSource": DETECTOR_SOURCE, "WavelengthUnit": "Angstrom",
            "Components": "filter+optics+qe", "ComponentsSource": "original",
            "Sources": " | ".join(SYSTEM_INPUTS[key].source for key in (band, OPTICS, QE))}


def build_system(filter_root: Union[str, Path], raw: Union[str, Path, None] = None) -> Path:
    """Add the components of the legacy Argus/System curves and label the curves' column row.

    Only line 1 of each published curve changes (:func:`system_header`);
    its samples are kept byte for byte.
    """
    filter_root = Path(filter_root)
    instrument_dir = filter_root / FACILITY / "System"
    raw = Path(raw) if raw else filter_root / FACILITY / "raw"
    curves = _read_inputs(SYSTEM_INPUTS, raw, "System")
    instrument_dir.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        published = instrument_dir / f"{band}.dat"
        if published.is_file() and not write_filter_header(
                published, {**read_filter_header(published), **system_header(band)}):
            raise ValueError(f"{published} has no column row (line 1) to hold its metadata")

    manifest = InstrumentManifest.load(instrument_dir)
    manifest.facility, manifest.instrument = FACILITY, "System"
    manifest.bands = list(BANDS)
    manifest.published = {"components": "filter+optics+qe", "source": "Argus ArgusSim system response"}
    manifest.published_bands = {band: "filter+optics+qe" for band in BANDS}
    _label_detector(manifest)
    manifest.ground_based = True
    manifest.components = {}

    for band in BANDS:
        add_component(instrument_dir, FILTER, *curves[band], band=band,
                      source=SYSTEM_INPUTS[band].source, manifest=manifest)
    add_component(instrument_dir, OPTICS, *curves[OPTICS], band=SHARED, name="optics",
                  source=SYSTEM_INPUTS[OPTICS].source, manifest=manifest)
    add_component(instrument_dir, QE, *curves[QE], band=SHARED, name="imx455",
                  source=SYSTEM_INPUTS[QE].source, manifest=manifest)

    manifest.notes = SYSTEM_NOTES
    manifest.save(instrument_dir)
    return instrument_dir


def build(filter_root: Union[str, Path], raw: Union[str, Path, None] = None, *, instrument: str = "System",
          session=None, archive_dir: Union[str, Path, None] = None) -> Path:
    """Build ``Argus/<instrument>`` (``DR280`` or ``System``); see the module docstring."""
    if instrument == "DR280":
        return build_dr280(filter_root, raw, session=session, archive_dir=archive_dir)
    if instrument == "System":
        if session is not None or archive_dir is not None:
            raise ValueError("Argus/System is built from local files only; session and archive_dir apply to DR280")
        return build_system(filter_root, raw)
    raise ValueError(f"No Argus builder for {instrument!r}; available: DR280, System")
