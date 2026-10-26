# NJM SVO-compatible Filter Profile Service

Service URL: **https://nillmill.ddns.net/sed_tools/svo_fps/fps.php**
(`https://nialljmiller.com/SED_Tools/...` redirects to `https://nillmill.ddns.net/sed_tools/...`).

This service answers the query interface of the SVO Filter Profile Service
(Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024) from the NJM SED_Tools
filter collection (`https://nillmill.ddns.net/sed_tools/filters/`). It is not
the SVO service. SVO-derived curves are credited to SVO in each record's
`Comments` and in the `ACKNOWLEDGEMENT` INFO elements of every response.

## Queries (as SVO FPS)

| Query | Returns |
|---|---|
| `fps.php?ID=Facility/Instrument.Band` | one filter: PARAMs + Wavelength/Transmission table |
| `fps.php?PhotCalID=Facility/Instrument.Band/Vega` (or `/AB`, `/ST`) | the same, with that zero point |
| `fps.php?Facility=..&Instrument=..&PhotSystem=..` | filter list (no curves) |
| `fps.php?WavelengthEff_min=..&WavelengthEff_max=..` (also `WavelengthRef`, `WavelengthMean`, `WavelengthMin`, `WavelengthMax`, `WidthEff`, `FWHM`; Angstrom) | filter list |
| `fps.php` or `fps.php?FORMAT=metadata` | service description |
| `getdata.php?format=ascii&id=Facility/Instrument.Band` | two columns: wavelength (Angstrom), transmission |

`ID` accepts:
1. the curve's `FilterID` header (for example `2MASS/2MASS.H`, `Roman/WFI.F062`);
2. the NJM path ID `Facility/Instrument.band` of `filters/<Facility>/<Instrument>/<band>.dat`;
3. SVO-form IDs that NJM holds under another name: the SVO match recorded in each
   instrument's `info.json`, and the alias map `filter_expansion/svo_aliases.json`
   (for example `Paranal/NACO.H` -> `Paranal/NACO/H_CONICA`, `WFIRST/WFI.R062` ->
   `Roman/WFI/F062`, `Herschel/SPIRE.PSW` -> `Herschel/SPIRE/250`,
   `Spitzer/MIPS.24mu` -> `Spitzer/MIPS/24`, `GAIA/GAIA2r.Grp` -> `GAIA/DR2r/RP`).
   The response then carries `<INFO name="ALIAS" value="<requested ID>">` naming the
   NJM filter, and `filterID` is NJM's own ID for that curve.

Matching is exact first, then a unique case-insensitive match.

## Copies: `_svo` and `_old`

Every published curve is in the service under its own `filterID`, copies included;
none is left out. `<band>_svo` is SVO's copy of a curve NJM holds from the original
source, and `<band>_old` is a superseded curve kept after a newer official curve took
the plain band name. The rule:

1. A copy's `filterID` is its NJM path ID, `Facility/Instrument.<band>_svo` or
   `Facility/Instrument.<band>_old` (for example `Euclid/NISP.Y_svo`,
   `JWST/NIRCam.F444W_old`). It never takes the `FilterID` header it shares with the
   plain-named curve.
2. That shared ID, the SVO-form IDs that point at it (`Euclid/NISP.Y`, `GAIA/GAIA2.G`,
   `Cassini/ISS_WAC.IR3`) and their PhotCalIDs return NJM's current, plain-named
   curve. To get SVO's curve, ask for the `_svo` ID, or ask SVO.
3. A copy's `Comments` starts by saying what it is:
   `SVO FPS copy of NJM curve <ID>; SVO serves it as <SVO ID>`, or
   `Superseded curve; the current one is <ID>`.
4. The rule also covers a curve whose `FilterID` header is another curve's path ID.
   `BOK/90prime/g` carries the header `BOK/BASS.g`, the path ID of `BOK/BASS/g`, so it
   is served as `BOK/90prime.g` with `Comments` `Same curve as BOK/BASS.g`. Both stay.

Filter lists (`Facility=`, wavelength ranges) include the copies. If you want one curve
per passband, drop the `filterID`s that end in `_svo` or `_old`.

## astroquery

astroquery 0.4.11 does not accept a non-SVO `conf.base_url` (its option validator
rejects it), so set the class attribute instead:

```python
from astroquery.svo_fps import SvoFps
import astropy.units as u

SvoFps.SVO_MAIN_URL = "https://nillmill.ddns.net/sed_tools/svo_fps/fps.php"

SvoFps.get_filter_list(facility="2MASS")
SvoFps.get_filter_index(16000 * u.AA, 17000 * u.AA)
SvoFps.get_transmission_data("Paranal/NACO.H")      # SVO-form ID, served by an alias
```

Set it back to `"https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php"` (the
astroquery default) to query SVO itself.

## Values

Every property comes from the instrument's `properties.json`
(`sed_tools.filter_properties`, computed from the NJM curve with the NJM Vega
spectrum), except values the filter owner published (`info.json`
`bands.<band>.specified`, listed in the NJM-extra PARAM `SpecifiedParams`).
`WavelengthRef` is the pivot wavelength unless specified. Missing values are empty.
NJM extras after the SVO fields: `Components`, `ComponentsSource`,
`DetectorTypeSource`, `NJMID`, `SpecifiedParams`.

## Maintenance

The service reads one file, `catalog.json`, built from the published curves.
`filter_info/refresh.sh` rebuilds it as its last step, so every refresh leaves it
current. To rebuild it on its own (after new curves are deployed and
`generate_index.py` has rewritten `properties.json`), run on Cirrus:

```sh
cd /media/data3/MESA/SED_Tools && flock /tmp/sed-filter-info.lock python3 filter_expansion/svc/build_catalog.py
```

(defaults: `--data data --out data/svo_fps --aliases filter_expansion/svo_aliases.json`;
the swap of `catalog.json` is atomic, so the live service never sees a partial file).
The alias map is rebuilt with `filter_expansion/phase4/tools/build_svo_aliases.py`
(see its `--help`). Source of the PHP and build script: `filter_expansion/svc/` on
the `server` branch.
