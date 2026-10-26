# Vega reference spectrum

The mirror's Vega quantities (zero points, effective and photon wavelengths, AB−Vega and ST−Vega offsets in each instrument's `properties.json`) and the website's Vega overlay use this spectrum.

| File | What it is |
|---|---|
| `alpha_lyr_mod_004.fits` | CALSPEC's model spectrum of Vega, unmodified, from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/alpha_lyr_mod_004.fits (SHA-256 `c80d6d92a9a23445d84193bb258829c5b2fe916d4ad89b43a06a86eaaddf65b2`) |
| `vega_flam.csv` | The same spectrum as MESA colors ships it (`data/colors_data/stellar_models/vega_flam.csv`, SHA-256 `f5d77623188ab923d63cbb67a76b6a575e04ffff5881c7eea7891a9502148478`): `WAVELENGTH` in Å (vacuum), `FLUX` and `CONTINUUM` in erg s⁻¹ cm⁻² Å⁻¹, 900 Å to 299 µm |

**Matching the FITS file:** the two files agree on all 8,094 wavelengths, with fluxes equal to within 6×10⁻⁸. The CSV is MESA's byte for byte, so NJM's zero points match MESA colors.

**The model:** Kurucz Vega model, T_eff = 9550 K, log g = 3.95, [M/H] = −0.5, at R = 500. It is normalised to 3.47×10⁻⁹ erg s⁻¹ cm⁻² Å⁻¹ at 5545–5570 Å, with wavelengths in vacuum and adjusted for a radial velocity of −21 km/s (from the FITS header).

**References:**
- Bohlin, Hubeny & Rauch 2020, AJ 160, 21 (doi:10.3847/1538-3881/ab94b4): the source the FITS header cites;
- Bohlin, Gordon & Tremblay 2014, PASP 126, 711 (2014PASP..126..711B): CALSPEC.

**Newer versions:** CALSPEC has since published `alpha_lyr_mod_005`, which differs by up to 1 % (median 0.01 %), and the observed `alpha_lyr_stis_012`. Switching would change NJM's Vega zero points slightly, and they would no longer match MESA.
