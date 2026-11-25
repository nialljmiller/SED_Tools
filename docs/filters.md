# Filters

A filter set is one instrument's photometric transmission curves: one `.dat`
file per band, plus an index file that tells MESA which curves to load.

## Downloading

```bash
sed-tools filters Generic/Johnson
sed-tools filters GAIA/GAIA
sed-tools filters 2MASS/2MASS
```

```python
from sed_tools.api import Filters

path = Filters.fetch("Generic", "Johnson")
path = Filters.fetch(facility="SLOAN", instrument="SDSS")
print(Filters.query(facility="HST")[:5])          # installed and remote filter sets, as dicts
```

`sed-tools filters` with no argument, in a terminal, opens the pickers. They
merge the instruments installed locally, on the NJM mirror, from the
original-source builders and on the SVO Filter Profile Service, and show each
one once, by name, in numbered columns. Pick one facility number to browse its
instruments, with their one-line descriptions; enter several numbers or
ranges, such as `1,3,5-7`, to download every instrument of those facilities
without further questions. Picking an instrument shows a summary before the
download question: its description, where the download will come from, its
bands, what each curve contains, its separate components, its atmosphere
sites and its licence. Names follow NJM's spelling where NJM has the
instrument; old spellings (`PLANCK/HFI`, `Gemini/GMOS-N`, `CTIO/S-PLUS`, ...)
still work as input.

A set is taken from the first of these sources that has it complete: an
installed copy, the NJM mirror, the instrument team's original files (the
original-source builders, for the instruments registered in
[`sed_tools/component_sources/catalog.py`](https://github.com/nialljmiller/SED_Tools/blob/main/sed_tools/component_sources/catalog.py)),
and SVO. [NJM mirror](njm.md) describes what happens without NJM.

Every download prints the MESA `instrument` line; see [MESA](mesa.md).

## How a filter set is stored

The filter root is `~/.sed_tools/data/filters/`. An installed instrument looks
like this:

```text
filters/<Facility>/<Instrument>/
  <band>.dat            published curves: what MESA reads
  <Instrument>          MESA index file: the curve files MESA loads, one per line
  manifest.json         what the instrument provides, with provenance
  properties.json       band properties computed by NJM (NJM downloads only)
  info.json             descriptions, references, acknowledgements, licence (NJM downloads only)
  components/<role>/    separate factors: filter/, optics/, qe/, or a composite such as filter+optics+qe/
  products/<recipe>/    composed sets; each is a MESA instrument folder with its own <recipe> index file
atmospheres/<site>/     beside filters/: the site's manifest.json and its .dat curves
cache/njm/              beside filters/: NJM's index.json, properties.json and info.json, cached
```

- **Curves.** Each `<band>.dat` starts with a column row,
  `Wavelength,Transmission,Key=value;...`, whose third field holds the
  metadata; then one `wavelength,transmission` sample per line.
- **MESA index file.** Named after the instrument folder. MESA loads only the
  curves it lists, so files beside it that it does not list (such as NJM's
  superseded `<band>_svo` copies) are not loaded.
- **`manifest.json`.** `bands`; `published` and `published_bands` (what each
  published curve contains, e.g. `filter+qe`); `components` (role, then band
  or `*`, then a file under `components/`); `detector_type`, `detector_types`
  and `detector_type_sources`; `variants` (alternative component curves, used
  only when selected with `variants=` in Python); `default_site`,
  `atmosphere_sites` and `ground_based`; `sources`, `notes` and
  `outside_range`; and `archived_bands` (superseded copies kept beside the set,
  never part of `bands` or the MESA index). `variants`, `atmosphere_sites` and
  `archived_bands` are written only when set.
- **Products.** A product folder's name records what was selected:
  `+<role>_<variant>` for a variant, `+atm_<label>` for an atmosphere,
  `+detector_<type>` for a detector override (e.g.
  `products/filter+qe+qe_single_layer/`).
- **`properties.json` and `info.json`** beside the curves describe exactly
  those curves, so SED_Tools prefers them to NJM's current files.

### The header row

```text
Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo-curve-match;WavelengthUnit=Angstrom;Components=filter+qe+instrument;ComponentsSource=svo
3200.0,2.37366962e-08
...
```

The metadata is a third CSV field with no whitespace (`%`, spaces, tabs, line
breaks, `,`, `;`, `"` and `#` in values are percent-escaped, e.g.
`Atmosphere=site=ctio_des%20airmass=1.2`). Released MESA, pandas, `csv`,
`np.loadtxt(skiprows=1)` and astropy's csv reader therefore see the same
`Wavelength` and `Transmission` columns and samples as without it; MESA's
photon-counting branch reads its `DetectorType`. A curve without metadata keeps
the bare `Wavelength,Transmission` row. Readers that require exactly two header
names, such as `np.genfromtxt(names=True)`, see the extra field.

SED_Tools 0.3.0 wrote these fields as `#` lines above the column row instead,
which released MESA (25.12.1, 26.4.1) misreads as extra samples. SED_Tools
still reads such files, and moves their fields onto the column row whenever it
fetches a cached set, downloads, tags or combines them. A curve with no
column-name row (its first line is a sample) is left as it is, with no
metadata.

`WavelengthUnit` is read from the same row. A file without it is taken to be
in Å unless its reddest point is below 1000 (then nm, µm or m), with a logged
warning; pass `load_filter_curve(path, wavelength_unit="nm")` from
`sed_tools.photometry` when a file is ambiguous.

## Describing a filter set

```bash
sed-tools filters-info 2MASS/2MASS
sed-tools filters-info LSST/LSST --offline --json
```

`filters-info` prints what the published curves contain, which components and
atmosphere sites exist, the detector type, the description, licence and
references, and a table of the bands. `--offline` uses only installed files
and the cached catalogue. Part of the output for LSST:

```text
LSST/LSST  (12 bands)
  Published curves contain : filter+instrument+atmosphere [svo]
  Separate components      : filter, optics, qe
  Can be built             : filter, filter+optics, filter+qe, filter+optics+qe
  Source review            : supported (lsst)
  Atmosphere               : can be added at its own site: lsst_std (default) (in Python,
                             Filters.atmospheres() lists the conditions)
  Detector                 : photon-counting
```

In Python, `Filters.available`, `Filters.info` and `Filters.properties` give
the same information (see [Python API](python-api.md)).

## Search and citation

NJM serves a catalogue beside the curves, so finding a band, reading its
properties and citing it do not need SVO:

```python
from sed_tools.api import Filters

bands = Filters.search(wavelength=21500, fwhm=(1000, 5000))     # pandas DataFrame, one row per band
print(bands[["facility", "instrument", "band", "WavelengthPivot", "FWHM", "DetectorType"]])
print(Filters.search(facility="HST", pivot=(5000, 6000), where={"ABminusVega": (None, 0.2)}))

print(Filters.properties("2MASS", "2MASS")["bands"]["Ks"]["ZeroPointVega_Jy"])
print(Filters.info("2MASS", "2MASS")["acknowledgements"])
print(Filters.cite("2MASS/2MASS", "GAIA/GAIA/G"))             # or format="bibtex"

ax = Filters.plot("2MASS/2MASS", normalize="peak", vega=True)   # matplotlib Axes
ax = Filters.plot(Filters.search(facility="2MASS"))             # a search result
ax.figure.savefig("2mass.png")
```

- **Search.** Name criteria are matched in NJM's `index.json` first; each
  instrument's `properties.json` is then fetched only if it can still match,
  at most six requests at a time, retrying when NJM briefly blocks a burst.
  A single `wavelength` keeps bands whose span (`WavelengthMin` to
  `WavelengthMax`, where T is at least 1 % of its peak) covers it; a range
  keeps bands overlapping it. Astropy quantities are accepted. A band with a
  missing value never matches a criterion on that value.
- **Citations.** References are merged across instruments, one per work. For
  chosen bands, the instrument-wide references plus those bands' own are
  given. Acknowledgement texts are quoted as the providers ask. References for
  which NJM has no BibTeX appear in the `.bib` output as `%` comments; nothing
  is made up.
- **Plots.** Curves are coloured as described in
  [Photometry](photometry.md); instruments not yet installed are downloaded
  first. `normalize` is `"peak"` or `"area"`; `vega=True` overlays NJM's Vega
  spectrum, scaled to the curves.

[Photometry and zero points](photometry.md) defines the properties.

## Combining filter sets

A filter set is only a folder of `.dat` curves and the index file that lists
them, so sets can be combined without rebuilding any spectra or cubes.
`filters-combine` copies the selected curves into a new instrument folder and
writes its index file:

```bash
# Creates filters/Combined/Gaia2MASS/ with a Gaia2MASS index file
sed-tools filters-combine Gaia2MASS GAIA/GAIA 2MASS/2MASS

# Choose the facility and instrument names, and stop on a duplicate band name
sed-tools filters-combine optical_ir GAIA/GAIA 2MASS/2MASS \
  --facility Custom --instrument optical_ir --on-conflict error
```

```python
from sed_tools.api import Filters

combined = Filters.combine("Gaia2MASS", "GAIA/GAIA", "2MASS/2MASS")
print(combined)
```

`--on-conflict` decides what happens when two inputs have a band file of the
same name: `rename` (the default) gives the later copy a new name built from
its facility and instrument (such as `SLOAN_SDSS_i.dat`), `overwrite` lets the
later copy replace the earlier, and `error` stops. Run again, the command adds
nothing that is already there. Generic/Johnson and
2MASS/2MASS both have `J.dat`, so combining them with `--on-conflict error`
stops on that file. Composed products can be inputs too, by their folder.

## Photon- and energy-counting curves

A transmission curve is defined either for an energy-counting detector (e.g.
Bessell's Johnson–Cousins curves) or for a photon-counting one (CCDs, e.g.
Gaia DR3, JWST). Synthetic photometry weights the integrand by `T(λ)` in the
first case and by `T(λ)·λ` in the second
([Bessell & Murphy 2012](https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B/abstract),
eq. A9); using the wrong one shifts Gaia G by about 0.035 mag. The detector
and the curve are different things: the 2MASS detectors count photons, but the
published 2MASS curves include a λ factor and are energy-counting curves.

Each file records the convention in `DetectorType` (`energy` or `photon`), and
`DetectorTypeSource` says how it was established, from the strongest evidence
to the weakest:

1. `original`: stated by the curve's publisher (paper, instrument page or data file).
2. `inferred`: not stated by the publisher, but established from documented
   facts, such as a filter-only curve used with a documented photon-counting
   detector.
3. `svo`: taken from the SVO record for `FilterID`.
4. `svo-curve-match`: the file was identified by matching its curve point by
   point against SVO.
5. `assumed`: unverified. `filters-detector` writes `energy` with this source
   when a file has no `DetectorType` and cannot be identified; the
   original-source builders write the convention set for the instrument
   (`photon` by default).

Types set by SED_Tools' own tools (`user` overrides, `composed` products and
`component` files) are outside this order and are never replaced.
`load_filter_curve` reads the type (`FilterCurve.detector_type`,
`FilterCurve.weight`), and every band average uses it; an untagged file is
treated as energy-counting.

### Tagging existing files

Downloads write the header. Files installed by older versions, or your own
files, can be tagged in place. `filters-detector` takes an installed set's
name (`Facility/Instrument`, under the filter root) or a path: the filter
root, an instrument folder or a single `.dat` file. A target that does not
exist is an error.

```bash
sed-tools filters-detector --dry-run                                         # whole filter root
sed-tools filters-detector GAIA/GAIA --dry-run                                # one installed set, by name
sed-tools filters-detector ~/.sed_tools/data/filters/Generic/Johnson --force --dry-run   # or by path
```

```python
from sed_tools.api import Filters

rows = Filters.tag_detector_types(dry_run=True)    # one row per file
```

By default `filters-detector` resolves untagged files, retries `assumed` ones
(except curves built from original sources) and completes tags that lack
`WavelengthUnit` or `Components`; `--force` re-resolves every file. An existing
type is replaced only by a resolution of equal or higher precedence, so SVO's
label never overrides an `original` or `inferred` type. A weaker resolution is
not written: the run lists the file as kept, with the SVO answer and the
reason (`kept photon [original]: original outranks svo; svo gives energy
(Generic/Bessell.B)`). The returned rows carry `kept`, `resolved_type` and
`resolved_source`.

To update an older collection: run `sed-tools filters-detector --dry-run`,
then `sed-tools filters-detector`, and review the entries marked `assumed`.
Existing flux cubes do not need rebuilding after tagging or composing; only
photometry needs recomputing.

### Correcting or converting a set's type

```bash
sed-tools filters-detector ~/.sed_tools/data/filters/SLOAN/SDSS --correct energy --bands g
sed-tools filters-detector ~/.sed_tools/data/filters/SLOAN/SDSS --convert energy --bands r
```

```python
from sed_tools.api import Filters

Filters.correct_detector_type("SLOAN/SDSS", "photon", bands=["g"])   # relabel only; the numbers are unchanged
Filters.convert_detector_type("SLOAN/SDSS", "photon", bands=["r"])   # relabel and convert the numbers
```

A **correction** says the curves were always of that convention, for example
a set used with a different detector. It writes `DetectorType`,
`DetectorTypeSource: user` and `DetectorTypeOriginal` (the type before the
first edit), and the manifest records it, so products composed later use it.

A **conversion** also converts the numbers: T_E(λ) = T_P(λ)·λ to energy
counting, or T_P(λ) = T_E(λ)/λ to photon counting, rescaled to the same peak.
MESA divides an energy-counting curve by λ and restores its peak, so the
converted curve gives the same MESA photometry. The header records
`DetectorTypeConversion` (`lambda` or `1/lambda`; none after converting back).
The detector itself is unchanged, so the manifest is too, and composition
starts from the unconverted numbers.

Both update the `info.json` and `properties.json` beside the curves for the
edited bands: a conversion changes only their detector fields; a correction
recomputes every property as NJM does, with NJM's Vega and Sun spectra (cached;
if NJM cannot be reached and they are not cached, the Vega- and Sun-based
values are left unknown and the reason is printed). Both take `filter_root=`
like the other `Filters` methods. The interactive menu reports each downloaded
set's and each new product's detector type and asks whether to correct or
convert it (default no).

A loaded `SED` reads the curves from disk at each `photometry` call, so it
uses the edited curves at once.

## Components and atmospheres

A published passband is usually the product of several throughputs, and which
ones varies: Gaia DR3 is filter + CCD + instrument, SDSS, 2MASS and PS1
include the atmosphere, Johnson is the filter alone. Every file records what
its curve contains (`Components=filter+qe+instrument`). Where an instrument's
team publishes the factors separately, SED_Tools stores them in `components/`
and composes the instrument's throughput with and without each one.
Components are never mixed between instruments or sites: an instrument is only
composed from its own filter, optics and QE, and only its own site's
atmosphere.

| Metadata | What it tells you |
|---|---|
| `Components` | Which throughput factors are multiplied into the curve: `filter`, `optics`, `qe`, `atmosphere`, or a source's unresolved `instrument` contribution |
| `DetectorType` | Whether integration uses the energy-counting weight `T(λ)` or the photon-counting weight `T(λ)·λ` |

QE is the detector's wavelength-dependent efficiency; the detector type is
how the response is weighted in integration. Tracking both prevents counting
a factor twice or applying the wrong weighting. What can be separated depends
on what the source publishes: a total throughput does not imply that all its
factors can be recovered.

### Composing

```bash
sed-tools filters-compose LSST/LSST --components filter                         # bare filters
sed-tools filters-compose LSST/LSST --components filter,optics,qe --airmass 1.2  # full system at its own site
sed-tools filters LSST/LSST --components filter,optics,qe --bands g r           # fetch and compose in one step
sed-tools filters-compose LSST/LSST --components filter,optics,qe \
  --bands g r --detector-type energy                                            # label a product energy-counting
```

```python
from sed_tools.api import Filters

info = Filters.available("LSST", "LSST", include_remote=False)
print(info["recipes_by_band"], info["missing_files"])
Filters.fetch("LSST", "LSST", components=["filter"])
Filters.fetch("LSST", "LSST", components=["filter", "optics", "qe"], atmosphere={"airmass": 1.2})
product = Filters.fetch("LSST", "LSST", components="filter+optics+qe", bands=["g", "r"], detector_type="energy")
```

Each product is a MESA instrument folder under `products/<recipe>/`, with its
own index file; give that folder to MESA to use the recipe. Published curves
remain the default, and photometry lookups by name skip `components/` and
`products/`.

- Requests that cannot be met are refused with an explanation, not
  approximated: components the team never published (e.g. Gaia's bare
  filters), an atmosphere from another site, adding an atmosphere to a curve
  that already includes one, or to a space or airborne instrument.
- A published curve that is the bare filter counts as the `filter` component.
- Products containing QE are photon-counting; otherwise they inherit the
  instrument's detector type. A band whose manifest records a `user` detector
  type (Argus, whose team states its sensors are energy counting) keeps it in
  every product.
- `--detector-type` (`detector_type=`) only labels the product, in a separate
  `+detector_energy` or `+detector_photon` folder with `DetectorTypeSource:
  user` and `DetectorTypeOriginal`; it does not convert the curves. Replacing a
  physical detector's QE needs its actual QE component.
- `Filters.available` separates installed component files from registered
  source support; `recipes_by_band` shows recipes available for only part of
  the bands. Instruments whose components have not been reviewed are marked
  `unreviewed` and remain available as published curves.
- A product is replaced only after every selected band is built, so a failed
  composition keeps the previous result.
- Source failures remain visible in inventory and availability results.
  Published downloads are checked against the bands expected from the
  manifests and index files; older installations without a catalogue
  declaration can only be checked against their existing contents.
- Unknown contents and assumed detector types remain marked as such in the
  metadata.

### Atmospheres

An instrument's atmosphere is one at its own location; you choose only the
airmass and PWV. Each site folder `atmospheres/<site>/` has a `manifest.json`
recording its kind, observing conditions, model, model version, parameters,
citation and any usage terms; `Filters.atmospheres()` lists them.

- **Team** sites (`kind: team`) hold the standard atmosphere an instrument
  team published, as published (wavelengths converted to Å) unless the entry
  records a `derivation`; such curves carry `ComponentsSource=derived`.
- **Derived** sites (`kind: derived`) are grids SED_Tools computes with the
  [ESO SkyCalc](https://www.eso.org/observing/etc/skycalc/) sky model
  (`lasilla`, `paranal`, `armazones`, `highanddry`), fetched at R = 200000 and
  bin-averaged to R = 20000 so that unresolved lines are integrated correctly.
  SkyCalc models Cerro Paranal, so it is used only for ESO facilities there and
  at La Silla. Other observatories are offered only their team's own
  atmospheres.

```bash
sed-tools filters-compose Paranal/VISTA --components filter,optics,qe --airmass 1.2 --pwv 2.5   # SkyCalc paranal
sed-tools filters-compose Paranal/VISTA --components filter,optics,qe --bands J H Ks \
  --atmosphere-site paranal_vircam --airmass 1.0                                            # ESO's own model, 0.9-5.6 µm
```

The default site is the manifest's `default_site`; other sites at the same
location are listed in `atmosphere_sites` or registered in
[`sed_tools/filters/sites.py`](https://github.com/nialljmiller/SED_Tools/blob/main/sed_tools/filters/sites.py),
and are chosen by name (`--atmosphere-site`). Some sources do not state their
airmass (2MASS's PLEXUS models); those atmospheres are kept as published and
selected with `airmass=None`, or `--airmass unstated`, and are never relabelled
with an assumed airmass. A site may set its own `outside_range` and
`coverage_tolerance`. Most team sites use `error` with a tolerance of 0.1 %:
composition is refused when more than 0.1 % of a band's throughput lies beyond
the atmosphere's table, and the edge values are held only over fainter leak
wings.

Installed atmosphere grids are used first, then NJM. During an NJM outage,
team sites are rebuilt from their original files and SkyCalc sites generate
the requested point (through ESO's API, or its web form when the API host does
not answer); SkyCalc generation requires PWV, and an existing grid with
several PWV values requires an explicit choice. SED_Tools never substitutes
another site's atmosphere or interpolates between grid points.

| Site | Kind | Conditions | Model and source | Instruments |
|---|---|---|---|---|
| `lsst_std` | team | X = 1.0, 1.2 | Rubin MODTRAN, `lsst/throughputs` v1.9.2 | LSST |
| `ctio_des` | team | X = 1.2 | DES standard bandpass table | DECam |
| `maunakea_hsc` | team | X = 1.2, PWV 1.5 mm | MODTRAN (N. Yasuda), `lsst/obs_subaru` | HSC |
| `haleakala_ps1` | team | X = 1.2 | MODTRAN; Aero × Ray × Mol of Tonry et al. 2012, table 3 | PS1 |
| `paranal_vircam` | team | X = 1.0 / PWV 1 mm; X = 2.0 / PWV 3 mm | unnamed model in ESO's VIRCAM ETC curves (0.9–5.6 µm) | VISTA (beside SkyCalc `paranal`) |
| `hopkins_2mass`, `ctio_2mass` | team | airmass not stated | PLEXUS (Cohen et al. 2003; Explanatory Supplement III.1.b1, tables 15/16) | 2MASS |
| `apo_sdss` | team | X = 1.0; X = 1.3 derived (power 1.3) | DR7 imager tables, column 5 | SDSS |
| `lco_csp` | team | X = 1.2, derived | CSP filter package: one-airmass CTIO extinction ^1.2 × telluric lines | LCO/CSP |
| `kpno_bass` | team | X = 1.0, derived, in-band only | BASS `bass-g/r.txt` column 4 / column 3 | BOK/BASS |
| `maunakea_gemini` | team | X = 1.0, 1.5, 2.0 × PWV 1.0, 1.6, 3.0, 5.0 mm | Gemini ATRAN (Lord 1992), 0.9–5.6 µm | NIFS |
| `cpachon_gemini` | team | X = 1.0, 1.5, 2.0 × PWV 2.3, 4.3, 7.6, 10 mm | Gemini ATRAN (Lord 1992), 0.9–5.6 µm | GSAOI, Flamingos-2 |
| `paranal`, `lasilla` | derived | X = 1.0–3.0 × PWV 0.5–20 mm (9 × 9) | ESO SkyCalc 2.0.9 (Noll et al. 2012; Jones et al. 2013) | ESO instruments |

### Where the components come from

| Instruments | Source of the separate factors |
|---|---|
| HST ACS (WFC/HRC/SBC), WFC3 (UVIS1/2, IR), WFPC2, NICMOS 1–3 | STScI synphot component tables (graph-table walk per obsmode); ACS/SBC's time-dependent sensitivity is pinned to the release (2021-04-25) its published curves were computed with |
| JWST NIRCam, NIRISS, MIRI imaging | STScI Pandeia reference data 2026.7 (in-flight; a 21.6 MB download without the PSF library) |
| Euclid NISP Y, J, H | ESA NISP-PHOTO-PASSBANDS V1.0: filter, telescope × NISP optics, and QE, which multiply to ESA's T_TOTAL, the published Y/J/H (SVO's earlier copies are kept on NJM as `Y_svo`, `J_svo`, `H_svo`) |
| Roman WFI | Detector QE only: the mean of the 18 flight detectors (Roman technical information repository) |
| Rubin/LSST | `lsst/throughputs` release 1.5 for `LSST/LSST` (SVO's curves) and release 1.9 for `LSST/LSSTv19` (+ Rubin's standard atmosphere `lsst_std`) |
| CFHT MegaPrime/MegaCam | CFHT filter-database scans of each band's own filter (2002, 2007 and 2014–2015 sets kept apart), primary mirror × corrector, simulated E2V QE |
| ERIS/NIX | ESO `nix-filters.zip`: AO dichroic × the band's default NIX camera (no QE or telescope curve published) |
| 2MASS | Explanatory Supplement III.1.b1 filter, mirror, window, lens and dichroic tables and NICMOS3 QE (the published RSRs also include PLEXUS atmosphere and a λ factor) |
| Subaru HSC | NAOJ/Subaru curves (+ HSC pipeline's Maunakea atmosphere `maunakea_hsc`) |
| CTIO DECam | DES standard bandpasses ÷ DES atmosphere (+ `ctio_des`); the i and z quotients exceed 1 and are kept unclipped, flagged in the manifest notes |
| VISTA/VIRCAM | ESO VIRCAM ETC curves (+ SkyCalc `paranal`, ESO's own `paranal_vircam`) |
| HAWK-I, NACO, SPHERE/IRDIS | ESO filter scans (11, 33 and 24 independently named curves; filter-only, with optics and QE still unavailable) |
| Gemini GSAOI, NIFS | Gemini filter scans (22 GSAOI bands and six NIFS temperature variants; filter-only) |
| CAHA PANIC, WHT INGRID, IRTF NSFCam | Independently sourced in-band filter scans (7 PANIC, 20 INGRID warm/cold variants, 26 historical NSFCam); filter-only |
| Liverpool IO:O | Seven filter scans plus the separately published CCD QE and cryostat-window transmission from the IO:O efficiency model; other telescope optics remain unavailable |
| ZTF | `ztf_information` filters and CCD QE |
| SLOAN/SDSS | Response without atmosphere from Gunn's 2001 `filter_curves.fits` (default `filter+optics+qe`), Doi et al. (2010) camera-column responses as variants (+ `apo_sdss`) |
| Swift/UVOT | HEASARC CALDB: current ARFs (published), `ftrans`, mirror and instrumental effective areas (÷ 659 cm²) |
| LCO CSP, CSP2 | CSP filter package: final passbands, optical scans, telescope and camera curves (+ `lco_csp`); CSP2 totals only |
| Cassini ISS NAC/WAC | PDS: system transmission (published), CCD QE, optics, derived filter pairs |
| Gemini GMOS-N/S | Detector QE by epoch (variants, no default) |
| BOK/BASS, KPNO/MzLS | Legacy Surveys curves: BASS filter+qe and qe (+ `kpno_bass`); MzLS corrected z with the filter |
| Argus/DR280 | Chroma filters (variants `production`, `addem`), DR280 optics, IMX455 QE |
| SVO instrument groups with filter-only sibling curves (UKIRT/WFCAM, OmegaCAM, Suprime-Cam, ...) | SVO's own filter-only sibling curves (`g_filter`, `g_fil`) |
| Pan-STARRS1 | not yet built: PS1's Table 3 has the open (no filter) cross-section and the atmosphere |
| Gaia, WISE, Spitzer, GALEX, TESS, Kepler, Hipparcos/Tycho | none: only totals are published (`filters-info` says so) |

The NJM mirror runs these builders ahead of time. A component request reuses
installed files, then tries NJM, then runs the instrument's original-source
builder. [NJM mirror](njm.md) describes running the builders yourself
(`sed-tools filters-build-components`).

## Checking a filter tree: `label_check`

`python -m sed_tools.filters.label_check` checks the rules a filter tree's
labels must keep. Among them: `DetectorType` is `energy` or `photon`, and each
label names a source SED_Tools can rank; `Components` names known roles only;
a band's filter-only sibling (`<band>_filter`) is labelled `filter`; curves
with the same samples carry the same `DetectorType` and say the same about
their components, across the whole tree; and every curve declares its
`WavelengthUnit`. With `--against OLD_TREE` it also requires that a changed
`DetectorType` comes from a source that outranks the old one. NJM runs it on
every refresh. The module docstring
([`label_check.py`](https://github.com/nialljmiller/SED_Tools/blob/main/sed_tools/filters/label_check.py))
lists all nine rules.

```bash
python -m sed_tools.filters.label_check ~/.sed_tools/data/filters
```

It exits 0 when the tree is clean, 1 when a rule is broken (each violation is
named), and 2 on a usage error.
