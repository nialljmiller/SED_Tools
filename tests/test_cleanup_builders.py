"""Second cleanup of the builders (offline): refusals whose message changed, and one warning per CALDB outage."""

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


def test_uvot_with_the_primary_caldb_host_down_warns_once_and_asks_it_once(tmp_path, caplog):
    from sed_tools.component_sources import swift
    from test_swift_csp_cassini import _Session, _uvot_files

    primary, mirror = swift.CALDB_ROOTS
    session = _Session(_uvot_files(mirror))  # the primary host serves nothing
    with caplog.at_level("WARNING", logger="sed_tools.component_sources.swift"):
        swift.build(tmp_path, instrument="UVOT", session=session)
    assert len([r for r in caplog.records if r.name == swift.__name__ and r.levelname == "WARNING"]) == 1
    assert sum(url.startswith(primary) for url in session.calls) == 1
    assert len(session.calls) == 12  # 11 CALDB files from the mirror, after one refusal
