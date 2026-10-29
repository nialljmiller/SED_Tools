"""Second cleanup of the builders: refusals whose message changed (offline; nothing is downloaded)."""

from __future__ import annotations

import pytest

from sed_tools.component_sources import gmos, roman


class _NoNetwork:
    def get(self, *args, **kwargs):
        raise AssertionError("nothing may be downloaded before the refusal")


@pytest.mark.parametrize("build, kwargs, directory", [
    (gmos.build, {"instrument": "GMOS_N"}, ("Gemini", "GMOS_N")),
    (roman.build, {}, ("Roman", "WFI")),
])
def test_builder_refuses_an_instrument_without_published_curves(tmp_path, build, kwargs, directory):
    (tmp_path.joinpath(*directory)).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="has no published curves; fetch the instrument"):
        build(tmp_path, session=_NoNetwork(), **kwargs)
