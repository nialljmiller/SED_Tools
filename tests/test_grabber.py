import pytest
from stellar_colors.atmosphere.grabber import AtmosphereGrabber
from astropy.io import fits
from io import BytesIO

@pytest.mark.parametrize("model_name", [
    "Kurucz2003",
    "NextGen",
    "bt-settl",
    "tmap",
    "coelho_sed"
])
def test_at_least_10_real_downloadable_spectra(model_name):
    grabber = AtmosphereGrabber()
    spectra = grabber._discover_spectra_votable(model_name)
    assert isinstance(spectra, list)
    assert len(spectra) >= 10

    count_valid = 0
    for entry in spectra:
        if count_valid >= 10:
            break
        access_url = entry.get('access_url')
        if not access_url:
            continue
        try:
            r = grabber.session.get(access_url, timeout=15)
            if r.status_code != 200 or len(r.content) < 1000:
                continue
            try:
                with fits.open(BytesIO(r.content), ignore_missing_end=True) as hdul:
                    if len(hdul) > 0:
                        count_valid += 1
            except Exception:
                continue
        except Exception:
            continue

    assert count_valid >= 10, f"Only {count_valid} valid spectra found for {model_name}"
