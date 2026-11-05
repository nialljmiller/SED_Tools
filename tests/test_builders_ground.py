"""Offline tests for the ZTF, HSC, BASS/MzLS, GMOS and SDSS component builders."""

from __future__ import annotations

import io
import logging
import zipfile

import numpy as np
import pytest

from sed_tools.component_sources import gmos, hsc, legacy_surveys, sdss, ztf
from sed_tools.filter_components import (FILTER, QE, ComponentError, InstrumentManifest, build_all_products,
                                         build_product, compose_band, read_curve)
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

from helpers_builders_ground import WL, _Resp, _gauss, _published


class _Session:
    def __init__(self, files):
        self.files = files
        self.headers = {}

    def get(self, url, timeout=None, **kw):
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _table(wl, tr, scale_wl=1.0, scale_tr=1.0):
    return "".join(f"{w / scale_wl:g},{t / scale_tr:g}\n" for w, t in zip(wl, tr))


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


def _sdss(tmp_path):
    """SLOAN/SDSS built offline: published u ... z and gprime_filter, Gunn (2001) and Doi et al. (2010) curves."""
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
    return inst


def test_sdss_camera_responses_are_variants(tmp_path):
    inst = _sdss(tmp_path)
    m = InstrumentManifest.load(inst)
    assert set(m.variants["filter+optics+qe"]) == {f"camcol{c}" for c in sdss.CAMCOLS} | {"gunn2001"}
    assert m.component_file("filter+optics+qe", "g") == "components/filter+optics+qe/g_gunn2001.dat"
    assert m.component_file(FILTER, "g") is None  # the primed USNO filter is not the camera's filter
    col3 = compose_band(inst, "g", "filter+optics+qe", variants={"filter+optics+qe": "camcol3"})
    assert col3.transmission.max() == pytest.approx(0.3)


def test_sdss_camera_response_left_out_of_build_all_products_is_named(tmp_path, caplog):
    # The primed filters are bands of their own (6c8739a), so filter+optics+qe
    # covers u ... z only and build_all_products builds no product for it (0.4.1
    # built one). It must say so and give the call that builds it.
    inst = _sdss(tmp_path)
    with caplog.at_level(logging.WARNING, logger="sed_tools.filter_components"):
        built = build_all_products(inst)
    assert built == []
    assert not (inst / "products").exists()
    bands = sorted(sdss.BANDS)
    assert ("SLOAN/SDSS: filter+optics+qe not built, because only 5 of 6 bands have the components "
            "(none for gprime_filter)") in caplog.text
    assert f"build_product({str(inst)!r}, 'filter+optics+qe', bands={bands!r})" in caplog.text

    product = build_product(inst, "filter+optics+qe", bands=bands)
    assert product.name == "filter+optics+qe"
    assert (product / product.name).read_text().split() == [f"{b}.dat" for b in bands]
    assert all(released_mesa_reads(product / f"{b}.dat") for b in bands)
