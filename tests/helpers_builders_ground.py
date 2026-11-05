"""Fakes shared by the ground-builder and component-variant tests: a wavelength grid, a
Gaussian curve, a fake response, and published curves.
"""

from __future__ import annotations

import numpy as np

from sed_tools.combine_filters import write_filter_index
from sed_tools.filter_metadata import PHOTON, write_filter_file


WL = np.linspace(4000.0, 6000.0, 201)


def _gauss(centre, width, wl=WL):
    return np.exp(-0.5 * ((wl - centre) / width) ** 2)


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


def _published(directory, band, wl, tr, components, filter_id=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=PHOTON, source="svo",
                      filter_id=filter_id or f"{directory.parent.name}/{directory.name}.{band}",
                      wavelength_unit="Angstrom",
                      extra={"Components": components, "ComponentsSource": "svo"})
    write_filter_index(directory, directory.name)
