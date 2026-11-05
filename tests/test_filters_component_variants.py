"""Component variants in filter manifests: selected only on request, kept apart,
round-tripped and listed, kept inside the instrument, cited per band and downloaded from NJM; and a
component above one kept unclipped.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sed_tools.filter_components import (QE, ComponentError, InstrumentManifest, add_component, compose_band)

from helpers_builders_ground import WL, _Resp, _gauss, _published


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def _instrument_with_qe_variants(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g", WL, _gauss(5000, 300), "filter")
    m = InstrumentManifest.load(inst)
    add_component(inst, QE, WL, np.full(WL.size, 0.9), name="qe_a", source="a", manifest=m)
    add_component(inst, QE, WL, np.full(WL.size, 0.5), name="qe_b", source="b", manifest=m,
                  variant="b", variant_info={"epoch": "from 2017", "description": "B"})
    m.variants["qe"]["a"] = {"files": {"*": "components/qe/qe_a.dat"}, "default": True}
    m.save(inst)
    return inst


def test_variant_is_selected_only_on_request(tmp_path):
    inst = _instrument_with_qe_variants(tmp_path)
    default = compose_band(inst, "g", "filter+qe")
    chosen = compose_band(inst, "g", "filter+qe", variants={"qe": "b"})
    assert default.transmission.max() == pytest.approx(0.9)
    assert chosen.transmission.max() == pytest.approx(0.5)
    assert "[qe=b]" in chosen.header["Recipe"]
    with pytest.raises(ComponentError, match="available: a, b"):
        compose_band(inst, "g", "filter+qe", variants={"qe": "nope"})


def test_variants_round_trip_and_are_listed(tmp_path):
    inst = _instrument_with_qe_variants(tmp_path)
    m = InstrumentManifest.load(inst)
    assert m.variants["qe"]["b"]["epoch"] == "from 2017"
    assert m.variants["qe"]["b"]["sources"] == {"*": "b"}
    assert "components/qe/qe_b.dat" in m.variant_files()
    info = m.describe(inst)
    assert info["variants"]["qe"]["b"]["bands"] == ["*"]
    (inst / "components/qe/qe_b.dat").unlink()
    assert m.missing_files(inst) == ["components/qe/qe_b.dat"]
    assert m.local(inst).variants["qe"]["b"]["files"] == {}


def test_manifest_without_variants_is_written_unchanged(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g", WL, _gauss(5000, 300), "filter")
    InstrumentManifest.load(inst).save(inst)
    assert "variants" not in json.loads((inst / "manifest.json").read_text())


def test_variant_paths_must_stay_inside_the_instrument():
    with pytest.raises(ComponentError):
        InstrumentManifest("F", "I", ["g"], variants={"qe": {"x": {"files": {"*": "../other/qe.dat"}}}})
    with pytest.raises(ComponentError):
        InstrumentManifest("F", "I", ["g"], variants={"qe": {"bad/name": {"files": {}}}})


def test_njm_downloads_variant_files(tmp_path, monkeypatch):
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    manifest = {"facility": "F", "instrument": "I", "bands": ["g"], "components": {},
                "variants": {"qe": {"a": {"files": {"*": "components/qe/qe_a.dat"}}}}}
    files = {"/F/I/manifest.json": json.dumps(manifest),
             "/F/I/components/qe/qe_a.dat": "Wavelength,Transmission\n4000,0.5\n5000,0.5\n"}
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    njm = NJMFilterGrabber(base_dir=str(tmp_path))

    class R(_Resp):
        @property
        def text(self):
            return self.content.decode()

        def json(self):
            return json.loads(self.text)

    def fake_get(url, timeout=None, **kw):
        for suffix, text in files.items():
            if url.endswith(suffix):
                return R(text)
        return R("", 404)

    monkeypatch.setattr(njm.session, "get", fake_get)
    assert njm.download_components("F", "I") == 1
    assert (tmp_path / "F/I/components/qe/qe_a.dat").exists()
