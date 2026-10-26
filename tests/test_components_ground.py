"""Offline tests for component variants and the ZTF, HSC, BASS/MzLS, GMOS and SDSS builders."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from sed_tools.combine_filters import write_filter_index
from sed_tools.component_sources import gmos, hsc, legacy_surveys, sdss, ztf
from sed_tools.filter_components import (FILTER, QE, ComponentError, InstrumentManifest, add_component,
                                         build_product, compose_band, product_name, read_curve)
from sed_tools.filter_metadata import PHOTON, read_filter_header, released_mesa_reads, write_filter_file

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


class _Session:
    def __init__(self, files):
        self.files = files
        self.headers = {}

    def get(self, url, timeout=None, **kw):
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _published(directory, band, wl, tr, components, filter_id=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=PHOTON, source="svo",
                      filter_id=filter_id or f"{directory.parent.name}/{directory.name}.{band}",
                      wavelength_unit="Angstrom",
                      extra={"Components": components, "ComponentsSource": "svo"})
    write_filter_index(directory, directory.name)


def _table(wl, tr, scale_wl=1.0, scale_tr=1.0):
    return "".join(f"{w / scale_wl:g},{t / scale_tr:g}\n" for w, t in zip(wl, tr))


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


def test_variant_products_do_not_replace_each_other(tmp_path):
    inst = _instrument_with_qe_variants(tmp_path)
    a = build_product(inst, "filter+qe")
    b = build_product(inst, "filter+qe", variants={"qe": "b"})
    assert a.name == "filter+qe" and b.name == "filter+qe+qe_b"
    assert read_curve(b / "g.dat").transmission.max() == pytest.approx(0.5)
    assert read_curve(a / "g.dat").transmission.max() == pytest.approx(0.9)
    assert product_name(["filter", "qe"], variants={"qe": "b"}) == "filter+qe+qe_b"


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


def test_band_components_cite_their_own_source(tmp_path):
    m = InstrumentManifest("F", "I", ["g", "r"])
    add_component(tmp_path, FILTER, WL, _gauss(5000, 300), band="g", source="g-url", manifest=m)
    add_component(tmp_path, FILTER, WL, _gauss(5000, 300), band="r", source="r-url", manifest=m)
    add_component(tmp_path, QE, WL, np.full(WL.size, 0.9), source="qe-url", manifest=m)
    assert m.sources == {"filter:g": "g-url", "filter:r": "r-url", "qe": "qe-url"}


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


# ---------------------------------------------------------------------------
# ZTF
# ---------------------------------------------------------------------------
def test_ztf_default_qe_is_multi_layer_and_coatings_are_variants(tmp_path):
    inst = tmp_path / "Palomar" / "ZTF"
    wl_nm = np.linspace(300, 1100, 81)
    files = {
        ztf.SINGLE: _table(wl_nm, np.full(wl_nm.size, 0.6)),
        ztf.MULTI: _table(wl_nm, np.full(wl_nm.size, 0.8)),
    }
    for band, table in ztf.SPEC.filters.items():
        _published(inst, band, WL, _gauss(5000, 300) * 0.8, "filter+qe", filter_id=f"Palomar/ZTF.{band}")
        files[table.url] = _table(WL, _gauss(5000, 300), 10.0, 0.01)
    ztf.build(tmp_path, session=_Session(files))
    m = InstrumentManifest.load(inst)
    assert m.component_file(QE, "g") == "components/qe/qe_multi.dat"
    assert set(m.variants["qe"]) == {"multi_layer", "single_layer", "mean"}
    assert m.variants["qe"]["mean"]["derived"] is True
    assert read_curve(inst / "components/qe/qe.dat").transmission.max() == pytest.approx(0.7)
    assert "filter" not in m.sources and m.sources["filter:i"].startswith(ztf.SPEC.filters["i"].url)
    single = compose_band(inst, "g", "filter+qe", variants={"qe": "single_layer"})
    assert single.transmission.max() == pytest.approx(0.6, rel=1e-3)


# ---------------------------------------------------------------------------
# HSC
# ---------------------------------------------------------------------------
def test_hsc_maps_r_i_to_r2_i2_and_publishes_user_filters(tmp_path, monkeypatch):
    inst = tmp_path / "filters" / "Subaru" / "HSC"
    for band in hsc.BANDS:
        _published(inst, band, WL, _gauss(5000, 300) * 0.5, "filter+instrument+atmosphere",
                   filter_id=f"Subaru/HSC.{band}")
    files = {}
    for name in set(hsc.CURRENT.values()) | set(hsc.FIRST.values()):
        scale = 0.9 if name.endswith("2") else 0.7
        files[f"{hsc.SUBARU}/wHSC-{name}.txt"] = "# wavelength transmittance\n" + "".join(
            f"{w:g} {t:g}\n" for w, t in zip(WL, _gauss(5000, 300) * scale))
    for band in hsc.USER_FILTERS:
        files[f"{hsc.SUBARU}/wHSC-{band}.txt"] = "".join(f"{w:g} {t:g}\n" for w, t in zip(WL, _gauss(5000, 20)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for band in hsc.MBQ1:
            zf.writestr(f"wHSC-{band}.txt", "".join(f"{w:g} {t:g}\n" for w, t in zip(WL, _gauss(5000, 50))))
    files[f"{hsc.SUBARU}/wHSC-MBQ1.zip"] = buf.getvalue()
    flat = "".join(f"{w:g} 0.9\n" for w in np.linspace(3000, 11000, 81))
    for table in hsc.SPEC.optics + (hsc.SPEC.qe,):
        files[table.url] = flat if table.wl_scale == 1.0 else "".join(
            f"{w / 10:g} 0.9\n" for w in np.linspace(3000, 11000, 81))
    for _, _, table in hsc.SPEC.site.entries:
        files[table.url] = flat
    hsc.build(tmp_path / "filters", session=_Session(files))

    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "r") == "components/filter/r2.dat"
    assert m.component_file(FILTER, "i") == "components/filter/i2.dat"
    assert m.variants["filter"]["first_r_i"]["files"]["r"] == "components/filter/r.dat"
    assert read_curve(inst / "components/filter/r.dat").transmission.max() == pytest.approx(0.7, rel=1e-3)
    assert {"NB387", "EB-gri", "MBQ1-413"} <= set(m.bands)
    nb = inst / "NB387.dat"
    assert released_mesa_reads(nb)
    assert read_filter_header(nb)["FilterID"] == "Subaru/HSC.NB387"
    assert m.published_bands["NB387"] == "filter"
    assert "NB387.dat" in (inst / "HSC").read_text().split()
    early = compose_band(inst, "r", "filter", variants={"filter": "first_r_i"})
    assert early.transmission.max() == pytest.approx(0.7, rel=1e-3)


# ---------------------------------------------------------------------------
# BASS and MzLS
# ---------------------------------------------------------------------------
BASS_TEXT = """# 90Prime SDSS g filter curve: ...
# wavelength (A)  res_noatm  respons_qe  res_qe_atm
4000.00 0.010 0.009 0.006
4500.00 0.900 0.810 0.648
5000.00 0.900 0.792 0.666
5500.00 0.020 0.017 0.015
"""


def test_bass_qe_is_the_column_ratio_above_the_floor():
    wl, filt, filt_qe, _ = legacy_surveys.read_bass(BASS_TEXT.encode())
    qwl, qe = legacy_surveys.bass_qe(wl, filt, filt_qe)
    np.testing.assert_allclose(qwl, [4500.0, 5000.0])
    np.testing.assert_allclose(qe, [0.9, 0.88])


def test_bass_builder_adds_filter_qe_and_qe(tmp_path):
    inst = tmp_path / "BOK" / "BASS"
    for band in legacy_surveys.BASS_BANDS:
        _published(inst, band, WL, _gauss(5000, 300) * 0.4, "filter+optics+qe+atmosphere",
                   filter_id=f"BOK/BASS.{band}")
        _published(inst, f"{band}_filter", WL, _gauss(5000, 300), "filter", filter_id=f"BOK/BASS.{band}_filter")
    files = {f"{legacy_surveys.FILES}/bass-{b}.txt": BASS_TEXT for b in legacy_surveys.BASS_BANDS}
    legacy_surveys.build(tmp_path, instrument="BASS", session=_Session(files))
    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "g") == "g_filter.dat"
    assert m.component_file("filter+qe", "g") == "components/filter+qe/g.dat"
    assert m.component_file(QE, "r") == "components/qe/qe_r.dat"
    assert m.primary_bands() == ["g", "r"]


def test_mzls_publishes_the_corrected_curve_with_the_filter(tmp_path):
    wl = np.linspace(8000, 10500, 26)
    files = {f"{legacy_surveys.FILES}/kpzd.txt": "# Area weighted\n" + "".join(
                 f"{w:.2f} {t:.5f}\n" for w, t in zip(wl, 98 * _gauss(9200, 500, wl))),
             f"{legacy_surveys.FILES}/kpzdccdcorr3.txt": "".join(
                 f"{w:.1f} {t:.5e}\n" for w, t in zip(wl, 0.5 * _gauss(9200, 500, wl)))}
    inst = legacy_surveys.build(tmp_path, instrument="MzLS", session=_Session(files))
    m = InstrumentManifest.load(inst)
    assert m.bands == ["z"] and m.published_bands["z"] == "filter+optics+qe+atmosphere"
    assert read_curve(inst / m.component_file(FILTER, "z")).transmission.max() == pytest.approx(0.98, rel=1e-3)
    assert read_filter_header(inst / "z.dat")["FilterID"] == "KPNO/MzLS.z"


# ---------------------------------------------------------------------------
# GMOS and SDSS
# ---------------------------------------------------------------------------
def test_gmos_qe_epochs_are_variants_without_default(tmp_path):
    inst = tmp_path / "Gemini" / "GMOS_N"
    _published(inst, "g", WL, _gauss(5000, 300), "filter", filter_id="Gemini/GMOS-N.g")
    nm = np.linspace(300, 1100, 17)
    files = {}
    for table, _, _ in gmos.QE_VARIANTS["GMOS_N"].values():
        cols = table.cols[1] + 1
        files[table.url] = "".join(" ".join([f"{w:g}"] + ["0.8"] * (cols - 1)) + "\n" for w in nm)
    gmos.build(tmp_path, instrument="GMOS_N", session=_Session(files))
    m = InstrumentManifest.load(inst)
    assert QE not in m.components
    assert set(m.variants["qe"]) == {"eev", "e2v_dd", "hamamatsu_bb", "hamamatsu_hsc"}
    assert "October 2011" in m.variants["qe"]["e2v_dd"]["epoch"]
    with pytest.raises(ComponentError):
        compose_band(inst, "g", "filter+qe")
    assert compose_band(inst, "g", "filter+qe", variants={"qe": "eev"}).transmission.max() == pytest.approx(0.8)


def test_sdss_camera_responses_are_variants(tmp_path):
    from astropy.io import fits

    inst = tmp_path / "SLOAN" / "SDSS"
    for band in sdss.BANDS:
        _published(inst, band, WL, _gauss(5000, 300) * 0.4, "filter+instrument+atmosphere",
                   filter_id=f"SLOAN/SDSS.{band}")
    _published(inst, "gprime_filter", WL, _gauss(5000, 300), "filter", filter_id="SLOAN/SDSS.gprime_filter")
    files = {f"{sdss.DOI}/{b}{c}.dat": "".join(f"{w:g} {0.1 * c:g}\n" for w in WL[::10])
             for b in sdss.BANDS for c in sdss.CAMCOLS}
    hdus = [fits.PrimaryHDU()]
    for b in sdss.BANDS:
        cols = [fits.Column(name=n, format="E", array=v) for n, v in
                (("wavelength", WL), ("respt", _gauss(5000, 300) * 0.4), ("resbig", _gauss(5000, 300) * 0.4),
                 ("resnoa", _gauss(5000, 300) * 0.5), ("xatm", np.full(WL.size, 0.85)))]
        hdus.append(fits.BinTableHDU.from_columns(cols, name=b.upper()))
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    files[sdss.GUNN] = buf.getvalue()
    sdss.build(tmp_path, session=_Session(files))
    m = InstrumentManifest.load(inst)
    assert set(m.variants["filter+optics+qe"]) == {f"camcol{c}" for c in sdss.CAMCOLS} | {"gunn2001"}
    assert m.component_file("filter+optics+qe", "g") == "components/filter+optics+qe/g_gunn2001.dat"
    assert m.component_file(FILTER, "g") is None  # the primed USNO filter is not the camera's filter
    col3 = compose_band(inst, "g", "filter+optics+qe", variants={"filter+optics+qe": "camcol3"})
    assert col3.transmission.max() == pytest.approx(0.3)
