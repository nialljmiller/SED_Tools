import pytest
from stellar_colors.atmosphere.grabber import AtmosphereGrabber

@pytest.mark.parametrize("model_name", [
    "Kurucz2003",
    "NextGen", 
    "bt-settl",
    "tmap",
    "coelho_sed"
])
def test_at_least_10_real_downloadable_spectra(model_name):
    grabber = AtmosphereGrabber()
    
    # Use the main discovery method that tries multiple approaches
    spectra = grabber._discover_spectra(model_name)
    assert isinstance(spectra, list)
    assert len(spectra) >= 10

    count_valid = 0
    for entry in spectra:
        if count_valid >= 10:
            break
            
        # Get FID for ASCII-based download
        fid = entry.get('fid')
        if not fid:
            continue
            
        # Test if spectrum exists using ASCII format (like the working code)
        if grabber._test_spectrum_exists(model_name, fid):
            count_valid += 1

    assert count_valid >= 10, f"Only {count_valid} valid spectra found for {model_name}"