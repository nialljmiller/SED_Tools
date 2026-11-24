# Quick start

This page takes one filter set and part of one spectral grid from download to
magnitudes and to the lines MESA needs. It downloads about 110 MB. Every
command prints where it wrote its files.

## 1. Download a filter set and the Vega spectrum

```bash
sed-tools filters Generic/Johnson
sed-tools vega
```

The first command installs the Johnson UBVRIJM curves under
`~/.sed_tools/data/filters/Generic/Johnson/`; the second writes NJM's Vega
spectrum to `~/.sed_tools/data/vega/vega_flam.csv`. Both print the line to
give MESA (`instrument = '...'` and `vega_sed = '...'`).

## 2. Download part of a spectral grid

```bash
sed-tools grids --grids Kurucz2003all --teff 4000 8000 --logg 3 5
```

Only the spectra with 4000 K ≤ Teff ≤ 8000 K and 3 ≤ log g ≤ 5 are downloaded.
Kurucz2003all has two [alpha/M] values, 0 and 0.4, so SED_Tools splits it into
one folder per value (a *variant*), each with its own `flux_cube.bin` and
`lookup_table.csv`, and prints one `stellar_atm` line per variant.

## 3. Interpolate a spectrum and compute magnitudes

```python
from sed_tools.api import SED, Filters

sed = SED.local("Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s")
sun = sed(teff=5777, logg=4.44, metallicity=0.0)
print(sun.wavelength[:3], sun.flux[:3])            # Å, erg/s/cm²/Å at the stellar surface

ab = sun.photometry("Johnson", system="AB")
vega = sun.photometry("Johnson", system="Vega", vega_spectrum=Filters.fetch_vega())
print({band: round(r.magnitude, 3) for band, r in ab.items()})
print({band: round(r.magnitude, 3) for band, r in vega.items()})
```

The magnitudes are those of the surface flux. For a star of radius R at
distance d, add `-5 * log10(R / d)` (R and d in the same unit).
[Photometry and zero points](photometry.md) explains the systems.

The grid's top folder can also be loaded. It interpolates in every axis,
[alpha/M] included:

```python
from sed_tools.api import SED

spectrum = SED.local("Kurucz2003all")(5777, 4.44, 0.0, alpha=0.2)
```

## 4. Hand the files to MESA

```bash
sed-tools mesa-path ~/.sed_tools/data/filters/Generic/Johnson
sed-tools mesa-path ~/.sed_tools/data/stellar_models/Kurucz2003all
```

Each prints the absolute, quoted line to paste into the `&colors` namelist;
for the split grid it prints one `stellar_atm` line per variant. Use one of
them. [MESA](mesa.md) has the details.

## Next

- [Spectral grids](grids.md): other grids and sources, building, combining.
- [Filters](filters.md): other instruments, components and atmospheres.
- [Command line](cli.md) and [Python API](python-api.md): every command and class.
