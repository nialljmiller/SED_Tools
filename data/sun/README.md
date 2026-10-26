# Solar reference spectrum

The mirror's `Fsun` values (the Sun's band-averaged flux density at 1 AU, in each instrument's `properties.json`) use this spectrum. `sed_tools.filter_properties` defines `Fsun` as ∫ W S dλ / ∫ W dλ: S is this spectrum, W the curve's detector-aware weight (T for energy-counting curves, T·λ for photon-counting ones), and the integrals use the trapezoidal rule on the curve's own samples. `Fsun` is null when the spectrum does not cover the wavelengths where the curve transmits at least 1 % of its peak.

| File | What it is |
|---|---|
| `sun_reference_stis_002.fits` | CALSPEC's solar reference spectrum, unmodified, from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_reference_stis_002.fits (SHA-256 `20133cdf2a402655e3726de6b334bf09d05c9429ebd27ebf5492dc37fc89cfd8`, 37,440 bytes) |
| `sun_mod_001.fits` | CALSPEC's Kurucz model of the Sun, unmodified, from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_mod_001.fits (SHA-256 `2622479d6582669c5218a726b7d8ef1555a963a8ecc59bb86b454c737d90510f`, 1,198,080 bytes) |
| `sun_flam.csv` | The two spliced at 9600 Å, as NJM uses them (SHA-256 `08b8dbcf908992460f02a0f2733a542e43dba3711cbf952263153b0e2903cdd5`): `WAVELENGTH` in Å (vacuum), `FLUX` in erg s⁻¹ cm⁻² Å⁻¹ at 1 AU; 56,996 rows, 1195 Å to 299.99 µm |
| `make_sun_flam.py` | The script that writes `sun_flam.csv` from the two FITS files |

Both FITS files were downloaded on 2026-10-26 (17:00 UTC). They are the only solar spectra in the live CALSPEC directory listing ([current_calspec](https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/)), where it dates both 2020-07-16.

## The two spectra (from their FITS headers)

**`sun_reference_stis_002`:** `DESCRIP = 'Sun with 2.5 to 2.7 micron extension'`, `SOURCE = 'Bohlin, Dickinson, & Calzetti 2001, AJ, 122, 2118'`, `TARGETID = 'SUN_REFERENCE'`. 1,467 rows, 1195.0 to 26957.4 Å, in `ANGSTROMS` and `FLAM`; the HISTORY says "Units: Angstroms(A) and erg/s/cm**2/A" and "All wavelengths are in vacuum." Its HISTORY lists the sources of each range:

```
WAVELENGTH RANGE         SOURCE                FILE
 1195    4100       Woods et al. 1996       SUN_UV
 4100    8700       Neckel & Labs 1984      SUN_NL84
 8700    9600       Arvesen et al. 1969     SUN_ARVESEN
 9600   26950       Model                   SUN_CASTELLI
```

and "ver _001 comment: Colina, Bohlin, & Castelli 1996,AJ,112,307". Version _002 (2013 Nov 22) only removed null keywords and all-null columns.

**`sun_mod_001`:** `SOURCE = 'http://kurucz.harvard.edu/stars/sun/fsunallp.5000resam51'`, `COMMENT = 'Created by R. Bohlin from Kurucz Special Model'`, `TEFFGRAV = '5777/4.4'`, `RESOLUTI = 5000`, `PEDIGREE = 'MODEL'`. 74,519 rows, 1500.08 Å to 2,999,932.9 Å. HISTORY: "Sun T/g=5777/4.4 [M/H]=0.0 at R=5000", "Model Normalized to obs. by 2.71677e-05", "Model normalized to observation at 6800-7700A", "UNITS: Wavelength(Angstroms), Flux(erg s-1 cm-2 Ang-1)", "All wavelengths are in vacuum."

The [CALSPEC page](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec) (read 2026-10-26) describes them: "CALSPEC also contains the ultraviolet to near-infrared absolute flux distribution of the Sun to 2.7 microns (filename: sun_reference_stis_002.fits). The solar reference spectrum combines absolute flux measurements from space and the ground with a model spectrum for the near-infrared (Colina, Bohlin, & Castelli 1996)." and "The Kurucz model for the Sun (sun_mod_001.fits) at a spectral resolution of R=5000 is in column 6 of Table 1b, where the model has been normalized to the SED of Thuillier (2003) at 6800-7700A."

**1 AU:** neither header states the distance. Integrating `sun_flam.csv` over its whole range gives 1.3637×10⁶ erg s⁻¹ cm⁻² (1363.7 W m⁻²), the solar constant, so the fluxes are at 1 AU.

## The splice at 9600 Å

`sun_flam.csv` takes every row of `sun_reference_stis_002` below 9600 Å (676 rows, to 9582.6 Å) and every row of `sun_mod_001` from 9600 Å (56,320 rows, from 9600.48 Å). Rows are copied as they are, with no interpolation, rescaling or smoothing. Values are written with the fewest digits that read back to the FITS values; they agree with the FITS files to within 6×10⁻⁸ (relative) in wavelength and flux.

Why splice, and why at 9600 Å:
- **What is measured:** stis_002 is measured only below 9600 Å. Its own HISTORY marks 9600–26950 Å as a model (`SUN_CASTELLI`). From 9600 Å the composite uses CALSPEC's newer solar model instead, which runs to 300 µm.
- **SVO uses the same composite:** the SVO Filter Profile Service computes its `Fsun` from the same two files. Its help text reads "Sun spectrum from calspec database: sun_reference_stis_002 composed with sun_mod_001 for larger wavelengths" (Internet Archive capture of 2026-01-18 of `http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=2MASS/2MASS.Ks&&mode=browse&gname=2MASS&gname2=2MASS`, archived on NJM's host under `source_evidence/svo-archive/ia-2026-10-26-b05-njm-browse/`).
- **Where 9600 Å comes from:** SVO does not state the splice wavelength. It is inferred from SVO's own values. The check ran `sed_tools.filter_properties` on SVO's own curves, with SVO's detector types, from the 4,598 archived SVO VOTables that carry an `Fsun` (Internet Archive captures under `source_evidence/svo-archive/*/raw/`, 2026-10-26):
  - **Pivots between 7000 and 12000 Å (906 curves):** switching at 9600 Å reproduces SVO's `Fsun` to within 0.02 % for all 906. Switching at 9500 Å matches 780, at 9700 Å 855, and stis_002 alone 621.
  - **All curves:** with `sun_flam.csv`, 4,333 of the 4,355 curves the composite covers match to within 0.02 %.
    - The other 22 differ by at most 0.31 %. The largest is CAHA 408_4 (a 2021 capture); the other 21 are within 0.17 %.
    - The remaining 243 curves extend past 300 µm, where the composite ends, so they get no `Fsun`.

**The step at the junction:** averaged over 9600–9800 Å, stis_002 is 0.29 % below sun_mod_001. Over 9400–9600 Å they agree to 0.02 %.

**Beyond the composite:** the composite covers 1195 Å to 300 µm. Outside that range NJM gives no `Fsun`, rather than extrapolating.

## SVO's archived `Fsun` values

The `svo.Fsun` values in each instrument's `info.json` are SVO's own numbers, copied from its pages. Their unit is erg s⁻¹ cm⁻² Å⁻¹, the same as NJM's `Fsun`. SVO's VOTable gives it as `<PARAM name="Fsun" value="22.596337946016" unit="erg/cm2/s/A" ucd="phot.flux.density" datatype="double" >`: that is 2MASS/2MASS.H, in the Internet Archive capture of 2025-03-23 of `fps.php?ID=2MASS/2MASS.H`, archived under `source_evidence/svo-archive/ia-2026-10-26-b01-njm-votable/`. SVO's browse page says "Unit :erg/cm2/s/A UCD :phot.flux.density". `index.json` records this under `filters.info_definitions.svo_fields`.

## References
- Bohlin, Gordon & Tremblay 2014, PASP 126, 711 ([2014PASP..126..711B](https://ui.adsabs.harvard.edu/abs/2014PASP..126..711B), doi:[10.1086/677655](https://doi.org/10.1086/677655)): CALSPEC.
- Colina, Bohlin & Castelli 1996, AJ 112, 307 (doi:[10.1086/118016](https://doi.org/10.1086/118016)): the solar reference spectrum, as cited by the CALSPEC page and stis_002's HISTORY.
- Bohlin, Dickinson & Calzetti 2001, AJ 122, 2118 (doi:[10.1086/323137](https://doi.org/10.1086/323137)): the `SOURCE` in stis_002's header.
- R. L. Kurucz's solar model, http://kurucz.harvard.edu/stars/sun/fsunallp.5000resam51: the `SOURCE` in sun_mod_001's header.
