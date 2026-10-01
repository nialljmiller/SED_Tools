"""Argus Array components (filters, optics, IMX455 QE).

Inputs are the raw vendor/design files kept alongside the Argus filter set
(``argus_g.csv``, ``bandpass_560_825nm.csv``, ``optics_throughput.csv``,
``imx455_qe.csv``), all tabulated in nm.  Components are attached to the
existing ``Argus/System`` instrument, whose published curves are the
end-to-end system response (filter x optics x QE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ..filter_components import FILTER, OPTICS, QE, SHARED, InstrumentManifest, add_component
from ..filter_metadata import PHOTON

FACILITY, INSTRUMENT = "Argus", "System"


def _read(path: Path, skip_header: int) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_header, usecols=(0, 1), dtype=float)
    data = np.atleast_2d(data)
    return data[np.isfinite(data).all(axis=1)]


def build(filter_root: Union[str, Path], raw: Union[str, Path, None] = None) -> Path:
    filter_root = Path(filter_root)
    instrument_dir = filter_root / FACILITY / INSTRUMENT
    raw = Path(raw) if raw else filter_root / FACILITY / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"Argus raw directory not found: {raw}")
    instrument_dir.mkdir(parents=True, exist_ok=True)

    manifest = InstrumentManifest.load(instrument_dir)
    manifest.facility, manifest.instrument = FACILITY, INSTRUMENT
    manifest.bands = ["beta", "rho"]
    manifest.published = {"components": "filter+optics+qe", "source": "Argus ArgusSim system response"}
    manifest.detector_type = PHOTON
    manifest.ground_based = True
    manifest.components = {}

    beta = _read(raw / "argus_g.csv", 1)              # nm, percent
    rho = _read(raw / "bandpass_560_825nm.csv", 1)    # nm, fraction
    optics = _read(raw / "optics_throughput.csv", 1)  # nm, fraction
    qe = _read(raw / "imx455_qe.csv", 0)              # nm, fraction

    add_component(instrument_dir, FILTER, beta[:, 0] * 10, beta[:, 1] / 100, band="beta",
                  source="Argus beta filter (argus_g.csv)", manifest=manifest)
    add_component(instrument_dir, FILTER, rho[:, 0] * 10, rho[:, 1], band="rho",
                  source="Argus rho filter (bandpass_560_825nm.csv)", manifest=manifest)
    add_component(instrument_dir, OPTICS, optics[:, 0] * 10, optics[:, 1], band=SHARED, name="optics",
                  source="Argus optics throughput (optics_throughput.csv)", manifest=manifest)
    add_component(instrument_dir, QE, qe[:, 0] * 10, qe[:, 1], band=SHARED, name="imx455",
                  source="Sony IMX455 QE, digitised (imx455_qe.csv)", manifest=manifest)

    manifest.notes = (
        "Published curves are ArgusSim's system response (PCHIP filters/optics, "
        "CubicSpline QE extrapolated to 300 nm). Composed products use linear "
        "interpolation and do not extrapolate the QE."
    )
    manifest.save(instrument_dir)
    return instrument_dir
