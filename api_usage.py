"""
SED_Tools Python API - Complete Test Suite
Tests all API operations end-to-end.
"""
from sed_tools.api import SED, Filters


import numpy as np
import numpy as np

def ascii_sed_continuum(
    wavelength, flux,
    width=100, height=18,
    logy=True, logx=True,
    reducer="median",   # "median" | "mean"
    smooth=7,           # odd integer; set 0/1 to disable
    title=None
):
    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)

    # sort
    idx = np.argsort(w)
    w, f = w[idx], f[idx]

    # x transform (log wavelength bins usually look better for SEDs)
    x = np.log10(w) if logx else w

    # y transform
    y = f.copy()
    if logy:
        y = np.log10(np.clip(y, 1e-300, None))

    # bin into columns
    edges = np.linspace(x.min(), x.max(), width + 1)
    col_y = np.full(width, np.nan)

    # vectorized-ish binning using digitize
    bin_id = np.digitize(x, edges) - 1
    bin_id = np.clip(bin_id, 0, width - 1)

    for i in range(width):
        m = (bin_id == i)
        if np.any(m):
            if reducer == "mean":
                col_y[i] = np.mean(y[m])
            else:
                col_y[i] = np.median(y[m])

    # fill gaps by interpolation
    xi = np.arange(width)
    good = np.isfinite(col_y)
    if good.sum() < 2:
        raise ValueError("Not enough finite points to plot.")
    col_y = np.interp(xi, xi[good], col_y[good])

    # smooth (moving average) for a continuum-like curve
    if smooth and smooth > 1:
        if smooth % 2 == 0:
            smooth += 1
        kernel = np.ones(smooth, dtype=float) / smooth
        col_y = np.convolve(col_y, kernel, mode="same")

    # scale to rows
    y_min, y_max = float(col_y.min()), float(col_y.max())
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    rows = np.round((col_y - y_min) / (y_max - y_min) * (height - 1)).astype(int)
    rows = np.clip(rows, 0, height - 1)

    # canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # draw a connected line (simple interpolation between successive columns)
    for i in range(width - 1):
        r0 = height - 1 - rows[i]
        r1 = height - 1 - rows[i + 1]
        # vertical step fill
        if r0 == r1:
            canvas[r0][i] = "─"
        else:
            step = 1 if r1 > r0 else -1
            for rr in range(r0, r1 + step, step):
                canvas[rr][i] = "│"
            canvas[r1][i] = "╲" if (r1 > r0) else "╱"

    # place final point
    canvas[height - 1 - rows[-1]][-1] = "●"

    # print
    if title:
        print(title)

    y_label_top = f"{y_max: .3f}" if logy else f"{y_max: .3e}"
    y_label_bot = f"{y_min: .3f}" if logy else f"{y_min: .3e}"

    for r in range(height):
        if r == 0:
            prefix = y_label_top.rjust(10) + " | "
        elif r == height - 1:
            prefix = y_label_bot.rjust(10) + " | "
        else:
            prefix = " " * 10 + " | "
        print(prefix + "".join(canvas[r]))

    print(" " * 10 + " +-" + "-" * width)
    # label original wavelength range (not transformed)
    w_left = f"{w.min():.1f}"
    w_right = f"{w.max():.1f}"
    mid = " " * max(0, width - len(w_left) - len(w_right))
    xlab = "(log10 wavelength)" if logx else "(wavelength)"
    print(" " * 13 + w_left + mid + w_right)
    print(" " * 13 + xlab)




# =============================================================================
# 1. Query available catalogs
# =============================================================================
print("=" * 60)
print("1. SED.query() - Discover catalogs")
print("=" * 60)

# Query all (local + remote)
all_cats = SED.query()
print(f"Total catalogs found: {len(all_cats)}")

# Show first 5 local
local_cats = SED.query(include_remote=False)
print(f"\nLocal catalogs ({len(local_cats)}):")
for cat in local_cats[:5]:
    print(f"  {cat.name}: Teff={cat.teff_range}, logg={cat.logg_range}")

# Query by parameter coverage
solar_cats = SED.query(teff_min=5500, teff_max=6000, logg_min=4.0, logg_max=4.5)
print(f"\nCatalogs covering solar params: {len(solar_cats)}")

# Query specific source
svo_cats = SED.query(source='svo', include_local=False)
print(f"SVO remote catalogs: {len(svo_cats)}")


# =============================================================================
# 2. Fetch from remote
# =============================================================================
print("\n" + "=" * 60)
print("2. SED.fetch() - Download catalog")
print("=" * 60)

# Fetch with parameter filtering
sed = SED.fetch(
    'NextGen',
    teff_min=3000,
    teff_max=5000,
    logg_min=4.0,
    logg_max=5.0,
    metallicity_min=-0.5,
    metallicity_max=0.5,
)
print(f"\nFetched catalog: {sed.cat.name}")
print(f"Spectra count: {len(sed.cat)}")
print(f"Teff grid: {sed.cat.teff_grid}")
print(f"logg grid: {sed.cat.logg_grid}")


# =============================================================================
# 3. Catalog operations
# =============================================================================
print("\n" + "=" * 60)
print("3. Catalog operations")
print("=" * 60)

# Access parameters as DataFrame
df = sed.cat.parameters
print(f"Parameters DataFrame shape: {df.shape}")
print(df.head())

# Filter catalog
cool = sed.cat.filter(teff_max=4000)
print(f"\nFiltered (Teff < 4000): {len(cool)} spectra")

# Iterate spectra
print("\nFirst 3 spectra:")
for spec in sed.cat[:3]:
    print(f"  Teff={spec.teff}, logg={spec.logg}, [M/H]={spec.metallicity}")

# Write to disk
output_path = sed.cat.write()
print(f"\nWrote catalog to: {output_path}")


# =============================================================================
# 4. Load local and interpolate
# =============================================================================
print("\n" + "=" * 60)
print("4. SED.local() - Load and interpolate")
print("=" * 60)

sed = SED.local('Kurucz2003all')
print(f"Loaded: {sed.cat.name}")
print(f"Parameter ranges: {sed.parameter_ranges()}")

# Interpolate spectra
spectrum = sed(5777, 4.44, 0.0)  # Sun
ascii_sed_continuum(spectrum.wavelength, spectrum.flux,
                 width=110, height=18, logy=True, logx=True,
                 reducer="median", smooth=9,
                 title="SED continuum (median-binned, smoothed)")

print(f"\nSolar spectrum (5777 K, 4.44, 0.0):")
print(f"  Wavelength range: {spectrum.wavelength[0]:.1f} - {spectrum.wavelength[-1]:.1f} Å")
print(f"  Points: {len(spectrum.wavelength)}")


spectrum = sed(4000, 4.5, -0.5)  # Cool dwarf
ascii_sed_continuum(spectrum.wavelength, spectrum.flux,
                 width=110, height=18, logy=True, logx=True,
                 reducer="median", smooth=9,
                 title="SED continuum (median-binned, smoothed)")

print(f"\nCool dwarf (4000 K, 4.5, -0.5):")
print(f"  Wavelength range: {spectrum.wavelength[0]:.1f} - {spectrum.wavelength[-1]:.1f} Å")


# =============================================================================
# 5. Combine catalogs
# =============================================================================
print("\n" + "=" * 60)
print("5. SED.combine() - Create ensemble grid")
print("=" * 60)

ensemble = SED.combine(
    catalogs=['Kurucz2003all', 'NextGen'],
    output='api_test_combined',
)
print(f"\nCombined catalog: {ensemble.cat.name}")
print(f"Parameter ranges: {ensemble.parameter_ranges()}")

# Interpolate from combined grid
spectrum = ensemble(4567, 4.2, 0.0)
print(f"\nInterpolated from combined (4500 K):")
print(f"  Points: {len(spectrum.wavelength)}")


# usage:
ascii_sed_continuum(spectrum.wavelength, spectrum.flux,
                 width=110, height=18, logy=True, logx=True,
                 reducer="median", smooth=9,
                 title="SED continuum (median-binned, smoothed)")



output_path = ensemble.cat.write()
print(f"\nWrote catalog to: {output_path}")


# =============================================================================
# 6. Filter profiles
# =============================================================================
print("\n" + "=" * 60)
print("6. Filters.query() and Filters.fetch()")
print("=" * 60)

# Query local filters
local_filters = Filters.query(include_remote=False)
print(f"Local filter systems: {len(local_filters)}")
for f in local_filters:
    print(f"  {f['facility']}/{f['instrument']}: {f.get('n_filters', '?')} filters")

# Fetch filters
output = Filters.fetch('GAIA', 'GAIA')  # or 'LSST', 'LSST' or 'Generic', 'Johnson'
print(f"\nDownloaded Johnson filters to: {output}")


# =============================================================================
# 7. ML Completer (if models exist)
# =============================================================================
print("\n" + "=" * 60)
print("7. SED.ml_completer() - ML SED completion")
print("=" * 60)

completer = SED.ml_completer()
print("ML Completer initialized")
print("  - Use completer.train(grid='...') to train")
print("  - Use completer.extend(catalog='...') to extend incomplete SEDs")
print("  - Use completer.load(model_name='...') to load saved model")


# =============================================================================
# Done
# =============================================================================
print("\n" + "=" * 60)
print("All API operations completed successfully!")
print("=" * 60)