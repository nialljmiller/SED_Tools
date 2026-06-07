import gzip
from pathlib import Path

from sed_tools.mast_spectra_grabber import (
    MASTSpectraGrabber,
    _filter_metals_by_range,
    _filter_urls_by_params,
)


def test_mast_download_accepts_generic_axis_kwargs(tmp_path, monkeypatch):
    url = (
        "https://archive.stsci.edu/hlsps/bosz/bosz2024/m+0.00/r1000/"
        "bosz2024_mp_t05750_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz"
    )
    wave = "1000\n2000\n3000\n4000\n"

    def fake_gather_urls(reskey, metals, session):
        return [url]

    def fake_try_wave_urls(reskey, session):
        return None, None

    def fake_download_text_gz(download_url, session=None):
        assert download_url == url
        return "1000 1.0\n2000 1.0\n3000 1.0\n4000 1.0\n"

    monkeypatch.setattr("sed_tools.mast_spectra_grabber._gather_urls_from_scripts", fake_gather_urls)
    monkeypatch.setattr("sed_tools.mast_spectra_grabber._try_wave_urls", fake_try_wave_urls)
    monkeypatch.setattr("sed_tools.mast_spectra_grabber._download_text_gz", fake_download_text_gz)

    grabber = MASTSpectraGrabber(base_dir=str(tmp_path))
    n = grabber.download_model_spectra(
        "BOSZ-2024-r1000",
        {"reskey": "r1000", "metals": ["m+0.00"]},
        teff_range=(5700, 5800),
        logg_range=(4.0, 5.0),
        meta_range=(-0.1, 0.1),
        wl_range=(1500, 3500),
    )

    assert n == 1
    model_dir = tmp_path / "BOSZ-2024-r1000"
    spectra = sorted(model_dir.glob("*.txt"))
    assert len(spectra) == 1
    body = [line for line in spectra[0].read_text().splitlines() if not line.startswith("#")]
    assert body[0].startswith("2000.000000 ")
    assert body[-1].startswith("3000.000000 ")
    assert (model_dir / "lookup_table.csv").exists()


def test_mast_param_filters_can_exclude_urls():
    urls = [
        "https://x/m+0.00/r1000/bosz2024_mp_t05750_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz",
        "https://x/m+0.00/r1000/bosz2024_mp_t10000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz",
    ]

    kept = _filter_urls_by_params(urls, teff_range=(5500, 6000), logg_range=(4.0, 5.0))

    assert kept == [urls[0]]


def test_mast_metallicity_filter_handles_bosz_tags():
    assert _filter_metals_by_range(["m-1.00", "m+0.00", "m+0.75"], (-0.2, 0.5)) == ["m+0.00"]
