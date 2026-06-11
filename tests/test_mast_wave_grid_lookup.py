import requests

import sed_tools.mast_spectra_grabber as mast


def test_try_wave_urls_uses_official_wavelength_grids_directory(monkeypatch):
    expected = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r1000.txt"
    calls = []

    def fake_download_text(url, session=None, timeout=180):
        calls.append(url)
        if url == expected:
            return "\n".join(str(x) for x in range(500, 512))
        return ""

    monkeypatch.setattr(mast, "_download_text", fake_download_text)

    wave, url = mast._try_wave_urls("r1000", requests.Session())

    assert url == expected
    assert wave.size == 12
    assert calls[0] == expected
