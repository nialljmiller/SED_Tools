# Photometry and zero points

## Magnitudes from a spectrum

An interpolated spectrum (`EvaluatedSED`, what `sed(teff, logg, metallicity)`
returns) computes synthetic magnitudes with `photometry`:

```python
from sed_tools.api import SED, Filters

sed = SED.local("Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s")
sun = sed(teff=5777, logg=4.44, metallicity=0.0)

ab = sun.photometry("GAIA", system="AB")
for name, result in ab.items():
    print(name, result.magnitude, result.flux_density, result.system)

vega = sun.photometry("Generic/Johnson", system="Vega", vega_spectrum=Filters.fetch_vega())
print(vega["V"].magnitude)
```

- `system` is `"AB"` (the default) or `"Vega"`. AB magnitudes use the AB zero
  point, 3631 Jy; `flux_density` is then the band-averaged f_ν in
  erg/s/cm²/Hz. Vega magnitudes use the Vega zero point; `flux_density` is
  then the band-averaged f_λ in erg/s/cm²/Å.
- A Vega magnitude needs a Vega zero point: from the filter file's header, if
  it carries one, or from a Vega spectrum given as `vega_spectrum=` (a path or
  a loaded spectrum). `Filters.fetch_vega()` downloads NJM's Vega spectrum,
  the one MESA uses, and returns its path. The zero points in a set's
  `properties.json` are not read by `photometry`.
- The magnitudes are those of the surface flux, as the grids store it. For a
  star of radius R at distance d, add `-5 * log10(R / d)` (R and d in the same
  unit). The Sun's surface flux gives G ≈ −38.5 (AB).
- A band flux of zero gives an infinite magnitude and a negative one NaN, as
  computed, with no error. A band that reaches beyond the spectrum's
  wavelengths gives one warning naming the filters; the flux there counts as
  zero.

### Naming the filters

`photometry` accepts, in any mix:

- a filter set's folder name under the filter root: `"GAIA"` finds
  `filters/GAIA/GAIA/*.dat`, `"Generic/Johnson"` finds that instrument's curves;
- a band file's stem, when only one installed file has it: `"Gbp"`;
- a path to a `.dat` file or to an instrument folder;
- a `FilterCurve` (from `sed_tools.photometry.load_filter_curve`), or a
  `(name, path)` tuple to rename a band.

A name that matches more than one file raises an error that lists them; pass a
full path or an instrument folder instead. Components and composed products
are not found by name; give their folder.

```python
from pathlib import Path
from sed_tools.api import SED

sed = SED.local("Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s")
spec = sed(teff=6000, logg=4.0, metallicity=-1.0)
gaia = Path.home() / ".sed_tools/data/filters/GAIA/GAIA"
mags = {name: r.magnitude for name, r in spec.photometry(gaia / "Gbp.dat", gaia / "Grp.dat").items()}
print(mags["Gbp"] - mags["Grp"])
```

## How a band is integrated

Every band quantity (band averages, zero points, pivot and other
wavelengths) uses one rule, in `sed_tools.photometry.band_integral`:

- The integral is taken on the curve's own samples, the spectrum's
  wavelengths inside the band, and extra points that split any step longer
  than λ/R, with R = `BAND_RESOLVING_POWER`.
- The transmission T and the flux F are linear between those wavelengths,
  and the integral of their product (times λ for a photon-counting curve) is
  exact.
- An energy-counting curve is weighted by T(λ), a photon-counting one by
  T(λ)·λ ([Filters](filters.md) explains `DetectorType`).

`sed_tools.photometry.band_sampling(wavelengths, curve)` reports how well a
spectrum's or a cube's wavelengths sample a band: the share of the band's
weight outside them and in coarse stretches. `sed-tools coverage --filters`
uses it ([Spectral grids](grids.md)).

## Filter properties and zero points

`sed_tools.filters.properties` computes the characteristics the SVO Filter
Profile Service lists for a curve:

- mean, pivot, peak, central, minimum and maximum wavelengths;
- FWHM and effective width;
- the Vega-weighted effective and photon wavelengths;
- Vega, AB and ST zero points, in f_λ and in Jy;
- the AB−Vega and ST−Vega magnitude offsets (2MASS Ks: m_AB − m_Vega ≈ 1.85);
- `Fsun`, the Sun's band-averaged f_λ at 1 AU (erg s⁻¹ cm⁻² Å⁻¹), when a solar
  spectrum is given;
- `AfAv`, the extinction ratio A(λ)/A(V) at the pivot wavelength, from the
  [Fitzpatrick (1999)](https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F)
  law with R_V = 3.1.

The pivot, zero points and `Fsun` use the same detector-aware weights as the
photometry above, so band-averaged f_λ and f_ν convert exactly. The
definitions are in the module docstring.

```python
from pathlib import Path
from sed_tools.api import Filters
from sed_tools.filters.properties import curve_properties, load_vega

vega = load_vega(Filters.fetch_vega())                          # wavelength in Å, f_λ in erg/s/cm²/Å
ks = curve_properties(Path.home() / ".sed_tools/data/filters/2MASS/2MASS/Ks.dat", vega)
print(ks["WavelengthPivot"], ks["ZeroPointVega_Jy"], ks["ABminusVega"], ks["AfAv"])
```

Without a Vega spectrum the Vega-based values are `None`; without a solar
spectrum `Fsun` is `None`. `load_sun(path)` reads a solar spectrum at 1 AU in
the same two-column format. NJM serves the one it uses (CALSPEC's
`sun_reference_stis_002` below 9600 Å and `sun_mod_001` above, the composite
the SVO Filter Profile Service uses) at
`https://data.nialljmiller.com/sed_tools/sun/sun_flam.csv`:

```bash
curl -sSfo ~/sun_flam.csv https://data.nialljmiller.com/sed_tools/sun/sun_flam.csv
```

```python
from pathlib import Path
from sed_tools.api import Filters
from sed_tools.filters.properties import curve_properties, load_sun, load_vega

vega = load_vega(Filters.fetch_vega())
sun = load_sun(Path.home() / "sun_flam.csv")
ks = curve_properties(Path.home() / ".sed_tools/data/filters/2MASS/2MASS/Ks.dat", vega, sun)
print(ks["Fsun"])
```

NJM serves these values precomputed for every instrument, in its
`properties.json`, computed with MESA's `vega_flam.csv` so that they match
MESA colors. `Filters.properties("2MASS", "2MASS")` reads them.

`AfAv` is computed by the `F99` class of
[dust_extinction](https://github.com/karllark/dust_extinction) (Gordon 2024,
[JOSS 9(100), 7023](https://doi.org/10.21105/joss.07023)), an optional
dependency: `pip install "sed-tools[extinction]"`. Without it, `AfAv` is
`None` and a warning says why. It is also `None` where the pivot lies outside
the range dust_extinction defines for the law (1000 Å to 3.33 µm); the law is
never extrapolated. A(V) is the law's own normalisation (V-band averaged), so
`AfAv` is 1 at 5414 Å, not at 5500 Å. SVO's `Af/AV` uses a different law, so
the two differ.

## Display colours

`sed_tools.filters.display` gives each band a colour for plotting, computed
from its curve rather than guessed from its name:

- **Mostly visible bands** (at least half the throughput between 380 and
  750 nm) get their true colour: daylight (CIE D65) seen through the curve,
  using the CIE 1931 colour-matching functions shipped with the package.
- **Ultraviolet and infrared bands** get false colours in wavelength order.
  Next to visible bands they continue bluer or redder; in an all-infrared or
  all-ultraviolet instrument they span a rainbow shifted toward that regime.

```python
from pathlib import Path
from sed_tools.filters.display import instrument_colors

folder = Path.home() / ".sed_tools/data/filters/2MASS/2MASS"
colors = instrument_colors(sorted(folder.glob("*.dat")))
print(colors["J"]["DisplayColor"])                # "#6cee2b"
```

NJM serves the colours in `properties.json` (`DisplayColor`), and its website
and `Filters.plot` use them. One instrument's bands keep their own colours; a
plot across instruments colours the selection as one set.
