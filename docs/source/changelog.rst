=========
Changelog
=========

Version 0.5.0 (October 2026)
----------------------------

This release holds everything since 0.4.1, including the changes prepared for
0.4.2, which was not released. Several results change (see "Wrong results
fixed" and "Behaviour changes"), and some grids and flux cubes made with
earlier versions must be downloaded or built again (next section).

Download or build again
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Cleaning skips spectra already marked ``units_standardized``, so cleaning
   an old download again repairs nothing. Download the raw files again, clean
   them with 0.5.0, then rebuild every flux cube, combined grid and densified
   cube made from them.

- NextGen grids from SVO cleaned with SED_Tools 0.2.0 to 0.4.1 (July to
  October 2026) hold fluxes about 100 times too large at Teff 5000 K and
  above (99.5 to 100.7 times a blackbody at their Teff over 5000-10000 K; the
  cooler spectra are right). The renormalisation that corrects them had not
  run since 0.2.0 (see "Wrong results fixed").
- Grids cleaned with the releases of April to June 2026, before 0.2.0, went
  through the first version of that renormalisation, which could also
  rescale correct spectra: hot stars over a wide wavelength range divided by
  3 to 13, and 3000-3500 K giants observed only in the optical multiplied by
  10 to 50. Download and clean those grids again too.
- MSG grids whose name ends in ``.``, ``h`` or ``5`` (every ``...-high``
  grid) were saved by every earlier release under a cut name, e.g.
  ``sg-OSTAR2002-hig``. 0.5.0 downloads such a grid again, into a folder with
  its full name. The old folder is not moved or deleted; if it holds a flux
  cube it is still listed as a local grid under the cut name, so remove it
  yourself. MSG no longer answers to the cut name.
- BOSZ grids (MAST) whose files carry their own wavelengths
  (``BOSZ-2024-rorig``), downloaded with earlier releases: the first spectrum
  of each download has only 51 rows. Cleaning does not repair it; download
  the grid again. Fixed-resolution BOSZ grids, read with MAST's separate
  wavelength files, were written in full. Cleaning a BOSZ grid now stops and
  asks for its flux unit (see "Flux units").
- Flux cubes made by the densifier (menu option 8 or
  ``grid_densifier.densify_grid``) with any earlier release are scrambled
  (see "Wrong results fixed"). Densify the source cube again.

Wrong results fixed
~~~~~~~~~~~~~~~~~~~

- **Densified flux cubes were scrambled.** ``grid_densifier`` read and wrote
  ``flux_cube.bin`` as ``(nt, nl, nm, nw)``, but the file, MESA and every
  other reader and writer (``precompute_flux_cube``, ``combine_stellar_atm``,
  ``FluxCube.from_file``) store it as ``(nw, nm, nl, nt)``, with Teff varying
  fastest. On a Kurucz test grid every node of the densified cube integrated
  to 64-89 times σT⁴, and the rows copied from the source differed from it.
  The densifier now uses the file's order: copied rows equal the source
  exactly and every node integrates to σT⁴ within 0.1 %.
- **The bolometric renormalisation had not run since 0.2.0.** It is the
  ``spectra_cleaner`` step that corrects SVO NextGen spectra at Teff ≥ 5000 K,
  whose fluxes are about 100 times too large. It referred to a constant that
  had been removed, and the resulting ``NameError`` was logged at debug level
  only. It runs again, with two changes from the original version: the
  blackbody is integrated on the spectrum's own wavelengths (the old
  2000-point linear grid missed a hot star's peak, so correct hot spectra
  were rescaled), and only spectra that are too bright are rescaled (the old
  version also rescaled spectra below 0.1 times the blackbody, which
  multiplied correct cool giants by 10 to 50). A spectrum with
  a Teff whose wavelengths hold at least 5 % of σT⁴ is divided by the ratio of
  its integrated flux to π·B_λ(Teff) integrated over the same wavelengths
  when that ratio is above 3. Its status is ``converted_renormed`` or
  ``recovered_renormed``; ``clean_model_dir`` counts it as converted or
  recovered, lists it with its factor in ``summary['renormed']`` and logs one
  warning. (The ``_renormed`` suffix handling described under 0.1.0 was never
  committed: the releases before 0.2.0 counted such files as errors, although
  they were written renormalised.) The step is not
  specific to NextGen; of the grids checked (NextGen, NextGen2, BT-Settl,
  BT-NextGen, BT-Cond, Cond00, Dusty00, TLUSTY, Kurucz, HRES) only NextGen at
  5000-10000 K is rescaled.
- **The ML completer's** ``extend_catalog`` (also ``api.MLCompleter.extend``
  and the completer menu) read ``lookup_table.csv`` with ``comment='#'``,
  which drops the ``#file_name,teff,...`` header SED_Tools writes. The first
  spectrum became the header and was lost, and every other spectrum was
  extended as Teff 5000 K, log g 4.5, [M/H] 0. The header is now read, and
  each spectrum is extended with its own Teff, log g and [M/H]. The ML menus
  counted one spectrum too few for the same reason; they count every row.
- **MAST (BOSZ) downloads were cut short.** When a grid's files carry their
  own wavelengths (``rorig``), the first spectrum of the download was
  written from the 51 rows read to detect them; with two-column wavelength
  and flux files every later file was cut to 51 rows too, with its wavelength
  column written as flux. Every file is now read in full. ``lookup_table.csv`` was parsed from the written file names,
  in which ``+`` becomes ``_``, so a row whose name had a ``+`` held only the
  file name; the parameters now come from the original BOSZ names. (Through
  ``sed-tools grids`` the table was complete, because cleaning rebuilds it
  from the spectrum headers.)
- **MSG grid names lost their last characters.** ``rstrip(".h5")`` removes
  any trailing ``.``, ``h`` and ``5``, so ``sg-OSTAR2002-high.h5`` was listed
  as ``sg-OSTAR2002-hig`` and saved in a folder of that name. Only the
  ``.h5`` suffix is removed now.
- **A bound of 0 was ignored** by ``SED.query`` and ``SED.fetch``: when every
  bound given was 0 or 0.0 no parameter filter was applied, so
  ``SED.query(metallicity_max=0.0)`` listed metal-rich grids and
  ``SED.fetch(..., logg_min=0)`` kept spectra with negative log g. Every bound
  given is now applied.
- **photometry(("V", path))** unpacked the tuple into two filters, ``"V"``
  and ``path``. A lone two-item tuple is now one ``(name, path)`` filter when
  its second item is a path object or a string naming an existing file;
  ``("V", "B")`` is still two filter names.
- **ML on Apple silicon** crashed: every training, ``complete_sed`` and
  ``generate`` raised "Tensor for argument weight is on cpu but expected on
  mps". The networks now use the device their inputs use
  (``ml_utils.get_device()``: CUDA, else MPS, else the CPU).
- **sed-tools ml_generator --auto** trained on scrambled data (its dataset
  read ``flux_cube.bin`` in the wrong axis order, mixing stars and
  wavelengths), needed the undeclared ``optuna``, and saved models that
  ``SEDGenerator`` and ``api.MLGenerator`` could not load. It now tunes
  ``SEDGenerator`` itself (each trial trains for 10 epochs on its own training
  data) and trains the best parameters with ``SEDGenerator.train``, which
  writes ``model.pt`` and ``config.json`` in the usual format. ``optuna`` is
  the optional extra ``ml-tune``.

Other results that change:

- Headers declaring W/m²/nm or W/m²/µm were converted with the W/m²/Å factor,
  so those fluxes came out 10 and 10⁴ times too bright. The new unit names
  ``flam_si_nm`` (factor 1e2) and ``flam_si_um`` (0.1) carry the right
  factors.
- Some flux-unit headers were read as the wrong unit, with no error: text
  such as "low mass" matched the SI unit (×1e3), ``erg/s/cm2/arcsec2``
  matched erg/s/cm²/Å, ``F_nu (Jy)`` was read as cgs F_ν (wrong by 1e23) and
  ``Flambda (W/m2/um)`` as erg/s/cm²/Å. Units are now matched before the bare
  quantity names, and a unit must not run into other letters.
- The ML completer's extend step (``complete_sed``) resampled the SED onto
  1,000 log-spaced points inside the requested range, replacing its own
  samples and dropping everything outside (a 147 Å - 1.6e6 Å Kurucz spectrum
  came back as 100 Å - 1e5 Å). It now returns the SED's samples unchanged and
  adds only the points beyond its ends; their values are as before.
- ``sed-tools combine --non-interactive`` (and "all" in the combine menu)
  folded the previous ``combined_models`` back in, so a second run combined
  each spectrum twice.
- ``regenerate_lookup_table`` gave a parameter no spectrum states (log g and
  [M/H] of a blackbody grid) a column of ``nan``, from which
  ``precompute_flux_cube`` built an all-zero cube with NaN axes and reported
  success.
- ``sed-tools mesa_prepare`` could not list or export the variants that
  ``sed-tools rebuild`` writes: it looked only in ``fluxcube_library/``,
  which nothing writes any more, and stopped with "No flux_cube.bin found".
- ``sed-tools densify`` always failed with ``TypeError``.
- ``sed-tools rebuild`` split a combined or ML-extended grid into one
  "variant" per spectrum, with no flux cube for the grid; it builds one cube.
- ``MLGenerator.generate_with_outputs`` wrote to ``./output/SED`` instead of
  the documented ``<model>/SED``.

Flux units
~~~~~~~~~~

- Cleaning stops and asks when spectra do not declare their flux unit.
  0.4.1 guessed the unit from the size of the values, assuming flux received
  at Earth, and converted silently: a surface-flux grid was read as Jansky,
  and a 3500 K π·B_λ spectrum came out scaled by 3e-13 to 3e-12 and tilted 11
  times over 3000-10000 Å. A grid's flux unit counts as declared when every
  sampled spectrum not yet standardized declares it in its header. Grids
  whose headers declare their units (SVO, MSG) and standardized grids (NJM)
  clean exactly as before.
- Python: ``clean_model_dir``, ``SED.fetch`` and ``SED.import_grid`` take
  ``flux_unit=``, the unit of spectra that do not declare one: ``flam``
  (erg/s/cm²/Å), ``fnu`` (erg/s/cm²/Hz), ``fnu_jy`` (Jansky), ``flam_si``
  (W/m²/Å), ``flam_si_nm`` (W/m²/nm), ``flam_si_um`` (W/m²/µm) or
  ``normalized`` (left as it is). With none given and none declared, they
  raise ``ValueError`` before any file is changed, naming the grid, the best
  guess and how to pass a unit; ``import_grid`` checks before copying or
  moving files. An unknown name raises ``ValueError`` too.
  ``spectra_cleaner`` adds ``FLUX_UNITS``, ``FLUX_GUESS_WHY``,
  ``undeclared_flux_unit`` and ``check_flux_unit``.
- Command line: ``--flux-unit`` on ``grids`` (``spectra``), ``rebuild`` and
  ``import`` gives the unit of spectra that do not declare one; spectra that
  declare theirs keep it. Without it, in a terminal, sed-tools asks once per
  grid, listing the accepted units and the best guess. A unit name must be
  typed: Enter does not take the guess, and ``n`` leaves the grid uncleaned,
  reported as failed. Without a terminal the grid fails with a message naming
  ``--flux-unit``. ``import --dry-run`` reports whether the unit is declared.
- BOSZ 2024 files hold the Eddington flux H_λ in erg/s/cm²/Å, and the surface
  flux is F_λ = 4πH_λ (MAST BOSZ README; Mészáros et al. 2024, eq. 2).
  SED_Tools has no unit for H_λ, so the MAST writer declares none and
  cleaning a BOSZ grid stops and asks. None of the accepted names is right
  for it: ``flam`` and ``normalized`` keep the shape but leave the flux 4π
  too low.

Filters from their original sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 37 more instruments can be rebuilt from their teams' original files when
  the NJM mirror is unavailable (``component_sources`` modules ``surveys``,
  ``space``, ``observatories`` and ``standards``, registered in
  ``catalog.BUILDERS_FOR``): ATLAS, PAN-STARRS, OGLE-IV, UKIRT/WFCAM, LCO,
  Gaia DR2, Herschel, Hipparcos/Tycho, IRAS, MSX, Planck, Spitzer, XMM-OM,
  FORS1/2, OmegaCAM, Keck NIRC2/LRIS/MOSFIRE/DEIMOS, KPNO Mosaic, Magellan
  FourStar/MegaCam, NOT/ALFOSC, Subaru Suprime-Cam/FOCAS, WIYN/ODI and the
  Bessell, MKO and USNO standard systems. ``FilterLibrary.published`` tries
  an installed copy, NJM, then these builders, then SVO. Keck spreadsheets
  need ``xlrd`` (``full`` extra).
- Paranal/NACO bands are published as ``IB_2p00`` etc. instead of
  ``IB_2.00``, matching the NJM mirror: MESA names history columns by the
  text before the first dot.
- New component builders (``sed-tools filters-build-components <source>``),
  each writing the factors the team publishes:

  - ``euclid``: Euclid NISP Y, J, H from ESA's NISP-PHOTO-PASSBANDS V1.0
    (filter, telescope × NISP optics, QE). ESA's T_TOTAL is written as the
    published Y, J and H where none is installed; NJM keeps SVO's earlier
    copies as ``Y_svo``, ``J_svo``, ``H_svo``.
  - ``roman``: Roman WFI, the mean measured QE of the 18 detectors as a
    shared ``qe`` component.
  - ``cfht``: CFHT MegaPrime and MegaCam, each band's own filter scan (the
    2002, 2007 and 2014-2015 filters kept apart), primary mirror × corrector
    and window, and CFHT's simulated E2V CCD42-90 QE. They reproduce SVO's
    MegaPrime totals to 0.005 mag.
  - ``eris``: ERIS/NIX, the AO dichroic × the band's default NIX camera.
  - ``twomass``: 2MASS filter, per-band optics and NICMOS3 QE from the
    Explanatory Supplement III.1.b1. With its PLEXUS atmosphere they
    reproduce the Cohen et al. relative spectral responses to 0.006 mag.
  - ``swift``: Swift/UVOT from the HEASARC CALDB: the current ARFs as the
    published curves, and ``ftrans`` and the mirror and instrumental
    effective areas (divided by the primary mirror's 659 cm² geometric area)
    as components. The sensitivity-loss correction is recorded in the notes,
    not applied. Files are tried on HEASARC, then its AWS Open Data mirror.
  - ``csp``: LCO/CSP from the CSP filter package (final passbands at airmass
    1.2, with the three V and two J filters as separate bands, plus the
    optical scans and telescope and camera curves), and LCO/CSP2 (totals
    only).
  - ``cassini``: Cassini ISS NAC and WAC, the PDS system transmission as the
    published curve where none is installed, with the CCD QE, the optics and
    derived filter pairs as components; SVO's copies (``<band>_svo``) are left
    out of the manifest.
  - ``legacy_surveys``: BOK/BASS (filter+qe and qe) and KPNO/MzLS (the
    corrected z curve with the filter).
  - ``gmos``: Gemini GMOS-N and GMOS-S, the detector QE by epoch, with no
    default.
  - ``sdss``: SLOAN/SDSS, the response without atmosphere from Gunn's 2001
    ``filter_curves.fits`` as the default ``filter+optics+qe`` composite, and
    the Doi et al. (2010) responses of the six camera columns as variants.
    SVO's primed filters (USNO 40-inch, in air) are no longer taken as the
    2.5 m camera's filters.
  - ``argus``: Argus/DR280, bands beta and rho with separate filter, optics
    and QE (the IMX455 QE standing in for the deployed IMX461); filter
    variants ``production`` (default) and ``addem``. Inputs are pinned by
    SHA-256; the non-public ones are read from the raw directory, and
    without them the builder says to install the instrument from NJM.

- Rubin/LSST is split by release: ``LSST/LSST`` takes its components from
  release 1.5, which its SVO curves equal, and release 1.9 is
  ``LSST/LSSTv19``, published from the v1.9.2 totals.
- ZTF's QE has variants: the multi-layer QE is the default (it reproduces
  SVO's g, r and i); the single-layer QE and their mean are variants. Subaru
  HSC's r and i components are the r2 and i2 filters its published curves
  use, with the first r and i as a variant; HSC's user filters are published
  as filter-only bands.
- HST: ACS/SBC's time-dependent sensitivity is pinned to release 004
  (2021-04-25), which reproduces the published SBC curves; negative noise in
  throughput tables is set to 0; manifests say how far composed products
  extend beyond the published curves. JWST uses Pandeia reference data
  2026.7 (21.6 MB without the PSFs), checked against STScI's SHA-1, and is no
  longer marked as a large download.
- DECam's instrument-only quotients above 1 are kept unclipped and recorded
  in the manifest notes; ``add_component`` warns for any component peaking
  above 1.
- Band components cite their own source file, and repeated wavelengths in
  source tables are averaged instead of picked by sort order.

Component variants, team atmospheres and SkyCalc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Component variants: a manifest can list alternative curves for a
  component (a QE per detector coating or epoch, a filter generation,
  per-CCD responses). The default stays in ``components``; a variant is used
  only when selected (``Filters.fetch(..., variants={"qe": "single_layer"})``,
  ``Filters.compose``, ``--variant ROLE=NAME``), and its name becomes part of
  the product's directory name (``filter+qe+qe_single_layer``). NJM downloads
  fetch the variant files too. ``filters-info`` lists the variants, default
  first.
- Team atmospheres are registered as named sites: ``haleakala_ps1`` (PS1),
  ``paranal_vircam`` (VISTA, ESO's model), ``hopkins_2mass`` and
  ``ctio_2mass`` (2MASS PLEXUS), ``apo_sdss`` (SDSS DR7), ``lco_csp`` (CSP),
  ``kpno_bass`` (BASS), ``maunakea_gemini`` and ``cpachon_gemini`` (Gemini
  ATRAN) and ``maunakea_hsc`` (HSC). Each site's manifest records its kind
  (``team`` or ``derived``), model, model version, parameters, citation and
  usage terms; the new fields are written only when set.
  ``filters-build-components --archive DIR`` keeps every raw download of the
  ``skycalc`` and ``team_atmospheres`` builders with its URL, time and
  SHA-256.
- An instrument may use further sites at its own location (the manifest's
  ``atmosphere_sites``, or the registry in ``facility_sites``), chosen by
  name: ``--atmosphere-site`` on the command line.
- Atmospheres whose source states no airmass (2MASS's PLEXUS models) are kept
  as published and selected with ``airmass=None`` or ``--airmass unstated``.
- A site may set ``coverage_tolerance`` next to ``outside_range="error"``:
  composition is refused only when more than that fraction of a band's
  throughput lies beyond the atmosphere's table. The team sites use 0.1 %, so
  VISTA's ESO atmosphere composes Y, J, H and Ks and refuses Z.
- SkyCalc grids are labelled ``derived``. The SkyCalc client falls back to
  ESO's web form when the API host does not answer (the API connect timeout
  is now 10 s instead of 600 s), checks the inputs SkyCalc echoes, and
  records the SkyCalc version. A resumed grid build keeps the grid's recorded
  settings when it has nothing to fetch, refuses to add points made with
  other settings, and checks the SkyCalc version before writing a new point.
- New command ``sed-tools filters-atmospheres`` lists the atmosphere sites
  with the airmasses and PWVs each one has.

SVO fails fast
~~~~~~~~~~~~~~

- SVO Filter Profile Service requests use a 10 s connect timeout (read
  timeouts unchanged), so an unreachable SVO host is reported within seconds
  instead of after 120-300 s per request. The first connection failure raises
  ``filter_metadata.SVOUnreachableError`` with the URL tried and the
  alternatives (the NJM mirror, the original-source builders); later requests
  from the same client and the rest of an SVO download stop at once.
- The FPS endpoint can be overridden with ``SED_SVO_FPS_URL`` (default
  ``https://svo2.cab.inta-csic.es/theory/fps/fps.php``); the browse index is
  taken from the same location.
- An SVO that cannot be connected to, or does not answer its browse index in
  time, is not asked again for five minutes by anything in the process, as
  for NJM: the facility list and each instrument list after it fail at once
  with the same error, which the pickers print once.
- Spectral grid discovery (``grids``, ``SED.query``) asks the SVO and MSG
  grid indexes once, with a 5 s connect and 10 s read timeout, instead of
  retrying for up to 200 s; a failure is one line ("did not answer within
  5+10 s", "answered HTTP 403"). NJM's spectra client does not ask a mirror
  NJM's filter client found down in the same process.

Filter properties and colours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``sed_tools.filter_properties`` to compute the characteristics the SVO
  Filter Profile Service lists for a curve:

  - mean, pivot, peak, central, minimum and maximum wavelengths;
  - FWHM and effective width;
  - Vega-weighted effective and photon wavelengths;
  - Vega, AB and ST zero points, in f_lambda and Jy;
  - AB-Vega and ST-Vega magnitude offsets;
  - ``Fsun``, the Sun's band-averaged f_lambda at 1 AU, from a supplied solar
    spectrum (``load_sun()``);
  - ``AfAv``, A(lambda)/A(V) at the pivot wavelength from the Fitzpatrick
    (1999, PASP 111, 63) law with R_V = 3.1.

  The pivot, zero points and ``Fsun`` share the detector-aware weights of
  SED_Tools photometry. ``curve_properties()`` and ``instrument_properties()``
  read filter files with their metadata. The NJM mirror serves the values
  precomputed in each instrument's ``properties.json``.
- ``AfAv`` is evaluated by ``dust_extinction`` (Gordon 2024, JOSS 9, 7023), a
  new optional dependency: ``pip install "sed-tools[extinction]"`` (also in
  ``full``). Without it ``AfAv`` is ``None``, with a warning. It is also
  ``None`` outside the range the package defines for the law (1000 Angstrom to
  3.33 micron); the law is not extrapolated.
- Add ``sed_tools.filter_colors`` to give each band a plotting colour from its
  curve. Mostly-visible bands get their true colour (CIE 1931 2-degree observer
  and illuminant D65, from the CIE's published tables, now shipped in
  ``sed_tools/cie``). Ultraviolet and infrared bands get false colours in
  wavelength order.

NJM catalogue: search, descriptions and citations without SVO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``Filters.search()`` and ``sed-tools filters-search`` to find bands in
  NJM's catalogue by facility, instrument or band name, by wavelength (bands
  covering a wavelength or overlapping a range), by pivot or FWHM range, by
  detector type, or by any column of ``properties.json``. The result is a
  pandas DataFrame with one row per band. Names are matched in
  ``index.json`` first, so ``properties.json`` is fetched only for
  instruments that can still match, at most six requests at a time.
  Instruments NJM serves no ``properties.json`` for are listed in
  ``attrs["gaps"]`` (``"gaps"`` in the JSON output), apart from fetch
  failures in ``attrs["errors"]``; ``filters-search`` exits 1 only for fetch
  failures.
- Add ``Filters.properties()`` and ``Filters.info()`` for an instrument's
  ``properties.json`` and ``info.json``. Files are cached in
  ``cache/njm/`` beside the filter root and revalidated by ETag when
  ``index.json`` changes; cached copies are used when the mirror is
  unreachable or with ``offline=True``.
- ``sed-tools filters-info`` also prints the instrument's description,
  licence and a per-band table of pivot, FWHM, detector type and photometric
  system; ``--json`` adds them under ``catalogue``.
- Add ``Filters.citations()``, ``Filters.cite()`` and ``sed-tools
  filters-cite`` to list the references (with BibTeX), the providers'
  acknowledgement texts and the licences for chosen instruments or bands, or
  for everything installed. References are merged across instruments by work.
  Entries with no BibTeX on NJM are given as ``%`` comments rather than
  invented.
- Add ``Filters.plot()`` (``sed_tools.filter_plot``) to draw an
  instrument's or a selection's curves in their ``filter_colors`` colours,
  optionally normalised and with NJM's Vega spectrum overlaid.
- NJM downloads keep ``properties.json`` and ``info.json`` beside the curves
  they describe. The MESA index file still lists only the curves.
- The NJM availability check no longer reports the mirror as down during
  NJM's temporary 403 block on a burst of requests from one address (or a
  429/5xx). It confirms by fetching ``index.json`` with the usual retries,
  which wait out the block (up to about a minute). No answer, or a 404, still
  means the mirror is down. ``Filters.fetch()`` therefore keeps using NJM
  instead of falling back to SVO, and the catalogue keeps working.

Detector-type precedence
~~~~~~~~~~~~~~~~~~~~~~~~

- ``DetectorTypeSource`` values now have an order of precedence, strongest
  evidence first: ``original`` > ``inferred`` > ``svo`` > ``svo-curve-match``
  > ``assumed`` (``filter_metadata.DETECTOR_SOURCE_PRECEDENCE``).
  ``sed-tools filters-detector`` and ``Filters.tag_detector_types()`` replace
  an existing type only with a resolution of equal or higher precedence, with
  or without ``--force``. Previously ``--force`` replaced a type stated by the
  curve's publisher (``original``) or established from documented facts
  (``inferred``) with SVO's label. For example, Generic/Bessell B (Bessell &
  Murphy 2012 photon-counting passband, ``FilterID=Generic/Bessell.B``) would
  have taken SVO's ``energy``. A weaker resolution is now reported and not
  written: the command lists each kept file with SVO's answer and the
  reason, and the returned rows carry ``kept``, ``resolved_type`` and
  ``resolved_source``. ``user``, ``composed`` and ``component`` types are
  never replaced.
- Default runs re-resolve the same files as before. The one difference: an
  ``original`` or ``inferred`` tag (or a ``user``, ``composed`` or
  ``component`` one) with a ``FilterID`` but no ``WavelengthUnit`` or
  ``Components`` is now left as it is instead of taking SVO's type and
  fields.
- Documented ``original`` and ``inferred``, and corrected the description of
  ``assumed``: it marks an unverified type, which is ``energy`` only for files
  that had none (the original-source builders write the convention set for
  the instrument, ``photon`` by default).
- Argus/System and Argus/DR280 are labelled ``DetectorType=energy`` with
  ``DetectorTypeSource=user``, as the Argus team states. A band whose manifest
  records a ``user`` detector type keeps it in every composed product, with
  or without the QE.

Filter pickers, instrument names and NJM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- NJM is optional. Facility and instrument names, descriptions and NJM's
  map of SVO instruments to its own come from a catalogue bundled with the
  package (``sed_tools/instrument_catalogue.json``, regenerated by
  ``scripts/make_instrument_catalogue.py``), which a reachable NJM extends.
  Without NJM the other sources are listed, named and described as when it
  is reachable.
- ``FilterLibrary.published`` (``Filters.fetch``, ``Filters.compose``) tries,
  in order, an installed complete copy, NJM, the instrument's original-source
  builder and SVO. NJM and the builder are asked under their own spelling of
  the name, matched ignoring case and punctuation (``CTIO/S-PLUS`` is NJM's
  ``CTIO/S_PLUS``). Each source that cannot give a complete set is reported
  with a warning naming the next one tried; when none can, ``ComponentError``
  carries every source's reason.
- NJM requests use a 5 s connect timeout. A mirror that stops answering is
  not asked again for five minutes by anything in the process, with one
  warning naming the sources used instead (installed copies, the
  original-source builders and SVO).
- NJM's instruments are read from its ``index.json``; its directory listing
  is a last resort, keeps directories only, and is announced.
- Picker entries from ``FilterLibrary.facilities()`` and ``instruments()``
  gain ``description`` and ``fetch_key``. Labels and fetch keys use NJM's
  spelling, else the original builder's, else the installed directory's,
  else SVO's instrument name, never the text of SVO's link, which can be a
  filter ID (0.4.1 showed labels such as ``Swift/UVOT.UVW2_f``). ``key``
  equals ``fetch_key``.
- An SVO instrument that NJM lists as the same product (``svo_instruments``
  in ``index.json``) is shown once, under NJM's name and description, whether
  or not NJM is reachable: with NJM down, SVO's GAIA2, GAIA2r and GAIA3 are
  listed as DR2, DR2r and GAIA, and Euclid's NISP0 is part of NISP. An entry
  only SVO lists takes NJM's name and description from the bundled catalogue
  only when NJM says it is the same product.
- Builders that make only an instrument's separate components (30 of 84,
  e.g. 2MASS, HST, JWST, Euclid/NISP) are no longer listed or tagged
  ``original`` as a source of its curves.
- With NJM down, a product of NJM's is filled from the SVO instruments that
  hold its bands, under NJM's facility, instrument and band names
  (``Filters.fetch("GAIA", "GAIA")`` installs SVO's GAIA/GAIA3 curves in
  ``GAIA/GAIA``; SVO's ``GAIA/GAIA2.Gbp`` becomes ``GAIA/DR2/BP.dat``). Each
  curve's header names the SVO filter it came from. The bundled catalogue
  records each instrument's bands and which SVO filter holds which band.
- An instrument installed under another spelling (``Gemini/GMOS-N``, or
  ``PLANCK/HFI`` from 0.4.1 on a case-sensitive disk) is used, or completed,
  in that folder; no second copy is made beside it.
- ``Filters.info`` and ``Filters.properties`` list the superseded copies NJM
  keeps beside a set (``BP_svo``) under ``archived_bands``, not ``bands``;
  ``filters-info`` no longer shows them as bands. ``filters-cite`` prints
  each title on one line and a reference known only by its address as the
  address.
- The pickers show each row as label, description and the sources it can be
  downloaded from, in the order a download tries them
  (``[local,njm,original,svo]``), with a legend; NJM is left out of the tags
  when it cannot be reached. Long labels are cut with an ellipsis, pages fit the terminal,
  and the longest labels are shown in full (0.4.1 cut every label to 2
  characters fewer than the longest, so with SVO down GAIA's instruments
  showed as ``DR2 DR2 GAI GAI``).
  Downloading several facilities at once asks for confirmation with a count.
- The browser downloads with the library it lists from, so NJM is checked and
  reported once per session; each listing error is printed once; an
  unreachable NJM does not count as an incomplete instrument list. The
  composition prompt reads a site's conditions from the cached atmosphere
  grid and asks NJM only for a site not cached yet.
- ``builder_for``, ``status_for`` and SVO lookups match names ignoring case
  and punctuation.
- ``Filters.compose`` refuses an unknown band before asking any source.
- ``Filters.available`` describes an NJM instrument without a component
  manifest instead of reporting an error.
- Downloaded curves are written atomically.

Command line
~~~~~~~~~~~~

- Expected errors are printed as ``sed-tools: error: <message>`` on stderr,
  with the traceback only when ``SED_TOOLS_DEBUG=1``. Ctrl-C during a
  command prints ``sed-tools: interrupted`` instead of a traceback, and
  during a menu workflow returns to the menu.
- The interactive menu shows itself again after a workflow fails, is
  interrupted or exits, and says "Unknown option" for a choice it does not
  know. The session exits 1 if any workflow failed.
- Usage errors print the subcommand's own usage line and a plain message
  (no Python names or syntax) and exit 2.
- New options: ``--flux-unit`` (``grids``, ``rebuild``, ``import``);
  ``--variant ROLE=NAME`` (``filters``, ``filters-compose``; repeatable);
  ``grids --teff``, ``--logg``, ``--meta`` and ``--wavelength`` (each
  ``LOW HIGH``; NJM, MSG and MAST cut downloads, SVO refuses cuts);
  ``combine --grids NAME ...``; ``ml_generator --trials`` and ``--epochs``
  (with ``--auto``; defaults 20 and 100); the ``filters-atmospheres``
  command. From the 0.4.2 work: ``--atmosphere-site`` and ``--airmass
  unstated`` (``filters``, ``filters-compose``),
  ``filters-build-components --archive``, and the ``filters-search`` and
  ``filters-cite`` commands.
- ``grids`` without a terminal no longer asks: it needs ``--grids``, and
  takes the whole grid unless a cut option is given. In a terminal an
  unparseable range is asked again (0.4.1 took it as no cut).
- ``filters-info`` shows the variants and every atmosphere site, puts long
  band descriptions under their row, and summarises the licence;
  ``filters-info``, ``filters-cite`` and ``filters-search`` fit the terminal.
  NJM's catalogue never makes ``filters-info`` fail.
- The composition prompt offers the variants, asks again on a bad recipe,
  detector type, airmass or PWV, defaults the airmass to the site's first
  condition, and offers the bands that can be built when coverage fails.
- Atmosphere and coverage errors name the band and say what the site has,
  e.g. "Site ctio_2mass has no atmosphere for airmass 1.2; it has unstated
  airmass (the atmosphere as published)".
- ``sed-tools densify`` runs (it always failed with ``TypeError``). Options
  ``densify_grid`` cannot honour are refused as usage errors: ``--method
  interp``, ``--method ml`` without ``--ml-model``, ``--method blackbody``
  with ``--ml-model``, and ``--ml-gap-threshold`` other than its default.
- ``coverage`` says why the Anderson-Darling test was skipped (too few
  distinct values, with the count, or scipy not installed).
  ``filters-build-components --help`` lists every registered source.

Spectral grids, flux cubes and ML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``mesa_prepare`` reads ``variants_index.csv`` first: ``sed-tools rebuild``
  (strategy ``split``) writes one subgrid per variant, ``<grid>/<path>/``
  with its own flux cube and lookup table, and lists them there.
  ``fluxcube_library/`` is still read when present. Variants are named and
  counted by the axes that vary; columns with one value throughout are not
  variants, and nor are the provenance columns of combined and ML-extended
  grids. A uniform grid with its flux cube is reported as MESA-ready. The
  default output folder name replaces characters such as ``/`` and spaces
  in the variant label.
- ``combine`` leaves out earlier combine outputs: a grid whose lookup table
  has the ``source_model`` and ``original_file`` columns (which only a
  combine writes, so 0.4.x outputs under any name are found), or a folder
  named ``combined_models``. ``--grids`` may still name one, and
  ``SED.combine(catalogs=[...])`` combines any grid it is given. The menu's
  spectrum count is no longer one short.
- The ML completer's interactive extend step asks for the wavelength range,
  defaulting to the model's own (it was fixed at 100 Å - 1e5 Å).
  ``SEDCompleter.list_models`` no longer lists generator models (choosing
  one raised ``KeyError``).
- The densifier assembles the dense cube one block of wavelengths at a time
  and writes each block sequentially, with blocks sized to half the
  available RAM.
- New warnings, with no number changed: ``complete_sed`` when the input
  covers less than 80 % of the network's input window; a band average whose
  filter has transmission outside the SED's wavelength range (counted as
  zero, so biased low); and a flux-cube build that averages several lookup
  rows onto one (Teff, log g, [M/H]) node, naming the columns that differ.

Correct or stop
~~~~~~~~~~~~~~~

The whole package was reviewed so that a run either gives a correct result or
stops and says why. Tests that compare the outputs of fixed inputs byte for
byte (``tests/test_golden.py``) guard against unintended changes.

- Exception handlers catch only the errors they expect, so a programming
  error is no longer reported as a missing file, an unavailable source or "no
  answer".
- No placeholder results: a value or file that cannot be read raises instead
  of becoming 0, NaN, an empty array, a normalisation factor of 1 or default
  stellar parameters.
- Fallbacks are visible: falling back to another source, method or cached
  copy is logged as a warning naming what failed and what is used instead
  (they were printed, logged at debug level, or silent).
- Batch operations (downloads, cleaning, tagging, ``grids``, ``rebuild``,
  ``coverage``, ``combine``, ML extension) finish what they can, then raise
  one error listing every failure.
- Commands exit non-zero when anything failed.
- Long functions are split into named steps.

Packaging
~~~~~~~~~

- No new required dependency. psutil is not a dependency: available RAM is
  measured by psutil if it is installed, else from ``/proc/meminfo``
  (Linux), else as half the physical RAM (macOS); only if all three fail is
  2 GiB assumed, with a warning. It sizes memory use only; no result depends
  on it.
- New extras: ``extinction`` (``dust_extinction>=1.6``, for ``AfAv``) and
  ``ml-tune`` (``optuna>=3``, for ``ml_generator --auto``; not in ``full``).
  ``full`` gains ``xlrd>=2.0`` (Keck workbooks) and ``dust_extinction>=1.6``.
- ``combine_stellar_atm`` imports scipy only where it uses it, so the module
  imports without scipy (scipy is in ``full``).
- The CIE tables (``sed_tools/cie/*.csv``) and the instrument catalogue
  (``sed_tools/instrument_catalogue.json``, NJM's ``index.json`` of
  2026-10-28T02:36:36) are package data.

Behaviour changes
~~~~~~~~~~~~~~~~~

Each item says what 0.4.1 did, what 0.5.0 does, and when it happens.

**Command line**

- ``combine --non-interactive`` used to combine every local grid, including
  earlier combine outputs; now it leaves out the output grid and every
  earlier combine output (and ``--grids`` names the grids to combine).
- The interactive menu used to end the session on a failed workflow (with a
  traceback and exit 1) and exit 0 on an unknown choice or Enter; now it
  prints the error, shows the menu again, says "Unknown option", and the
  session exits 1 at quit if any workflow failed.
- Usage errors that used to exit 1 now exit 2: ``ml_generator --auto``
  without ``--library`` or ``--output``, ``filters-build-components`` with an
  unknown source, ``filters-build-components tabular`` without
  ``--instrument`` or with an unknown one, and the ``densify`` options listed
  under "Command line".
- ``python -m sed_tools.cli`` always exited 0; it now exits with the
  command's status. Every command handler returns 0 on success.
- ``sed-tools filters`` (the browser) printed "Cannot fetch" or "Cannot build"
  and exited 0; it now lists every failed download or build when browsing
  ends and exits 1. In a multi-facility download an instrument list left
  incomplete by a source error counts as a failure (an unreachable NJM does
  not).
- A non-numeric airmass or PWV in the composition prompt printed a message
  and built nothing; now the question is asked again.
- ``grids`` printed a message for a grid it could not complete and carried
  on to "Done." (exit 0); now it processes every other grid, then exits 1
  listing each grid not completed and every source's error. Triggers: grid
  not found, no source with metadata, nothing downloaded, nothing left after
  cleaning, missing files, no flux cube built, or a download, cleaning or
  build error. Spectra left unusable by cleaning are listed too, though the
  grid's products are built. With nothing discovered from any source it
  fails.
- ``grids`` without a terminal stopped at the axis-cut question (EOFError);
  now it asks nothing, takes the whole grid unless a cut option is given, and
  without ``--grids`` says to name the grids.
- ``rebuild`` stopped at the first grid that raised, and grids skipped for
  having no spectra still ended "Rebuild complete." (exit 0); now every grid
  that can be rebuilt is, then it exits 1 listing each one that was not
  (including ``--grids`` names that are not local grids). With no local grids
  it fails. Its interactive selection asks again on a bad answer (out-of-range
  numbers were dropped), and the selection is sorted with duplicates removed.
- ``combine`` with no local grids exited 0; it now fails.
- The combine menu combined every grid on an empty answer, a trailing comma
  or a non-number, dropped out-of-range numbers and combined a repeated
  number twice; now it asks again until the answer is ``all`` or distinct
  grid numbers. Grids are combined in the order typed.
- ``coverage`` with no local grids exited 0, and with several grids stopped
  at the first that raised; now it exits 1, covering every grid and printing
  each failure.
- ``mesa_prepare`` printed and exited 0 when there were no grids, a grid or
  variant choice was invalid, the lookup table or flux cube was missing, a
  grid without its cube offered a single variant, or the user declined to
  overwrite the output; each now exits 1 with "Error: ...". A uniform grid
  with its flux cube exits 0 and is reported as MESA-ready.
- ``filters-info`` exited 1 only when no source answered; now it also exits
  1 when an answering source reported errors (NJM failing while another
  source answered does not count).
- ``filters-detector`` printed a traceback and no summary when tagging
  raised; it now prints each file as it is tagged, then the error, and exits 1.
- ``python -m sed_tools.flux_cube_tool`` exited 0 when a filter's photometry
  failed; it now lists the failed filters at the end and exits 1.
- ``ml_generator --auto`` without optuna failed inside ``ml_optimized`` with
  ``ModuleNotFoundError``; it now stops before training and names
  ``pip install 'sed-tools[ml-tune]'``.
- The picker showed the whole list when a ``//regex`` filter was invalid;
  now it says so and keeps the previous filter.
- The picker's columns are wider, so labels are shown in full; occasionally
  one fewer column fits.
- ``run_combine_flow`` prints the same progress lines as ``SED.combine``.
- The interactive number questions of the command line, the ML menus and the
  densifier are one (``ui_utils.ask_number``): the airmass and PWV questions
  no longer accept ``nan`` or ``inf``, a blank answer where one is needed says
  "Please enter a number.", and a bad one "'x' is not a number; try again."
- ``combine --grids`` may name an earlier combine output, and
  ``--non-interactive`` judges earlier outputs as the combine menu does (the
  ``source_model`` and ``original_file`` columns, or the folder name
  ``combined_models``).

**Filters, NJM and SVO**

- ``FilterLibrary.facilities()`` and ``instruments()`` entries' ``key`` and
  ``fetch_key`` used SVO's spelling; they now use NJM's (``PLANCK`` ->
  ``Planck``, ``GMOS-N`` -> ``GMOS_N``, ``Pacs`` -> ``PACS``, ``S-PLUS`` ->
  ``S_PLUS``). The old spellings still work as input to ``Filters.fetch``,
  ``Filters.compose`` and the commands.
- An instrument requested under SVO's spelling skipped NJM and the original
  builder and was downloaded from SVO; it is now downloaded from NJM or built
  by the builder, into the directory named by NJM's or the builder's spelling
  (``Filters.fetch("CTIO", "S-PLUS")`` installs ``CTIO/S_PLUS``). A complete
  set already installed under the old name is used as it is.
- ``FilterLibrary.published``: when NJM gave only part of a set, the SVO
  fallback could write SVO's curves over NJM's under the same names; SVO is
  now never run into a set NJM wrote or confirmed curves in during the call,
  and ``ComponentError`` says to run again. Completeness is judged by NJM's
  index file; superseded copies NJM keeps beside a set are not expected.
- ``FilterLibrary.published`` used to try only ``tabular`` builders between
  NJM and SVO; every registered original-source builder that can publish
  curves is now tried.
- When no source gives a complete set, ``FilterLibrary.published`` raised
  whatever SVO raised (``RuntimeError``, ``SVOUnreachableError`` or
  ``ComponentError``); it now raises ``ComponentError`` (a ``ValueError``)
  giving every source's reason, and warns at each fallback.
- ``FilterLibrary.published`` ignored a failure after NJM's curves arrived
  (its components, say); it now returns the complete set with a warning.
- ``FilterLibrary.published``, SVO step: a failed row now raises
  ``ComponentError`` even when the requested directory is complete (the row
  was filed under another directory); it used to return that directory.
- ``FilterLibrary.published`` with NJM down: a product of NJM's that SVO
  holds under other names (GAIA/GAIA is SVO's GAIA/GAIA3) raised "SVO
  instrument not found"; it is now installed from those SVO instruments under
  NJM's names. When SVO has only some of its bands, those are installed and
  ``ComponentError`` then names the bands missing and why (GAIA/GAIA's
  Gbp_bright and Gbp_faint are NJM's only); the set stays incomplete, so the
  next fetch asks NJM again, which keeps the curves that are its own and
  replaces the others with a warning.
- ``FilterLibrary.published`` downloaded into a new folder spelled as NJM
  spells the instrument when it was installed under another spelling; it now
  completes the installed folder (the MESA index file takes that folder's
  name). Two installed spellings of one instrument raise ``ComponentError``
  naming both, unless one is spelled exactly as asked.
- When NJM provides every curve its index lists, bands the local
  ``manifest.json`` declares that the index does not list and that are not
  installed are removed from it, with a warning (they made the set look
  incomplete on every fetch).
- An unavailable NJM mirror was silent; it is now a warning with the reason,
  naming the sources used instead. The SSL "verification disabled" note is a
  warning instead of a print.
- NJM listings (``discover_instruments``, ``discover_filters``,
  ``discover_facilities``, ``list_atmosphere_sites``) returned ``[]`` on any
  failure; they now raise unless NJM answered 404. A fallback from
  ``index.json`` or an instrument index to the directory listing is a
  warning.
- ``NJMFilterGrabber.download_filters`` printed failures and returned a
  count; it now saves everything else, then raises ``RuntimeError`` naming
  each curve or component file that failed. ``download_components`` does the
  same. ``download_all_filters`` finishes every facility and instrument,
  including past failed listings and requests, then lists the incomplete
  ones.
- ``NJMFilterGrabber.fetch_manifest`` returned ``None`` on any failure; now
  only for 404. Other failures raise, and an unreadable manifest, or one
  naming another instrument, raises ``ValueError``.
- ``NJMFilterGrabber.download_atmosphere`` returned ``False`` for any failure;
  now only for 404, and it retries a 403 burst block like other requests.
- An SVO failure while checking a mirror curve's header during an NJM
  download was hidden (an untagged curve was written energy/assumed); the
  mirror's header is now kept, with a warning.
- ``FilterLibrary.instruments()`` reports "NJM unavailable" in its errors as
  ``facilities()`` did; ``inventory`` lists each error once.
- ``FilterLibrary.prepare_components`` and ``prepare_atmosphere`` warn, with
  the reason, when NJM cannot provide the components or the atmosphere,
  before trying the original source.
- ``FilterLibrary.atmospheres(remote=True)`` listed fewer sites silently when
  an NJM listing or site manifest failed (other than 404); it now raises.
- ``SVOFilterBrowser.download_filters`` printed each failure and returned; it
  now saves what it can, writes the index and manifest, then raises
  ``RuntimeError`` listing every filter not saved. The interactive bulk mode
  downloads every selected instrument, then raises one error listing the
  failed ones; a partial single-instrument download ends the session with
  exit 1.
- Detector tagging (``filters-detector``, ``Filters.tag_detector_types``,
  ``tag_filter_file``, ``resolve_detector_type``) treated an SVO timeout,
  HTTP error or unreadable answer as "SVO has no record" and wrote an
  untagged file as ``energy``/``assumed``; it now raises and leaves the file
  unchanged. "SVO has no such ID" still gives ``assumed``.
- ``Filters.tag_detector_types(facility=...)`` stopped at the first file SVO
  could not resolve; it, and ``tag_filter_tree``, now tag every file they
  can, printing each as it is tagged, then raise ``RuntimeError`` listing the
  files left unchanged. An unreachable SVO still stops at once.
- ``SVOMetadataClient.facility_filter_ids`` cached ``[]`` for the session on
  failure; it now raises.
- ``InstrumentManifest.from_headers`` warns, naming the band and the token,
  when a ``Components`` header cannot be parsed.
- ``Filters.query`` logged source discovery errors at info level; they are
  warnings.
- Composed products that include the QE were always labelled
  photon-counting; a band whose manifest records a ``user`` detector type
  now keeps it (only Argus does: its products are energy-counting).
- NJM's availability, its five-minute pause and the certificate fallback are
  now shared by the whole process: every ``FilterLibrary`` and
  ``FilterCatalogue`` used to check NJM again (5 s each with NJM silently
  down) and repeat the warning; now NJM is checked once, asked again only
  after the pause, and an outage is warned about once. The warning says NJM
  is not used for 5 minutes (it said "not used").
- An NJM download fetched every installed curve of the set again to check it
  was NJM's. It now records which installed curves it found to be NJM's
  (``<filter root>/../cache/njm/filters/<F>/<I>/verified_curves.json``, with
  each file's size and NJM's ``index.json`` record of the instrument), and
  skips those while neither has changed: a retry after a partial download
  fetches only the missing curves. A set installed by an earlier release is
  checked once.
- ``Filters.compose`` (``filters-compose``) refuses, before any download,
  bands for which the instrument's manifest declares none of the requested
  components while declaring them for other bands, naming both: composing
  Euclid/NISP from ``filter,optics,qe`` without ``--bands`` used to download
  ESA's tables and then fail on the grisms Blue and Red.
- Manifests keep ``archived_bands`` (superseded copies NJM keeps beside a set,
  such as Euclid/NISP's ``H_svo``): ``InstrumentManifest`` reads and saves it
  (saved only when set), so a rebuild by the JWST or Roman builder no longer
  drops it. ``InstrumentManifest.from_published`` takes as bands only the
  curves the MESA index file lists (when there is one), never an archived
  band, so the Euclid, HSC, GMOS and other builders no longer put archive
  copies back into ``bands``, and ``write_filter_index`` leaves archived bands
  out of the MESA index.
- ``Filters.query()``'s ``facility_key`` is the spelling to download with
  (NJM's where NJM has the facility); in 0.4.1 it was SVO's. Rows gain
  ``svo_facility_key`` (SVO's own spelling, or ``None``), and the
  ``facility`` argument matches any source's spelling, ignoring case. Picker
  entries gain ``names``, each source's own spelling.
- ``FilterLibrary.available()`` (``Filters.available``) gains
  ``sources_failed`` (failures by source) and ``fallback_from`` (the sources
  whose failure only made a later one answer); ``errors`` is unchanged.
  ``filters-info`` decides its exit status from them instead of matching
  ``"NJM"`` in the messages.
- ``FilterCatalogue.origin()`` says whether NJM catalogue data came from the
  copy installed beside the curves, the cache or NJM, and when NJM last
  confirmed it; ``filters-info`` prints "Catalogue: from the cached copy,
  checked <date> UTC; NJM could not be asked" when that happened, and
  ``--json`` gives the origin.
- Atmosphere site errors list sites as plain text ("observes from paranal,
  here_team; atmospheres from other sites (lsst_std) are not mixed in")
  instead of Python lists.

**Original-source builders**

- ``LSST/LSST`` took its components from lsst/throughputs v1.9.2; they now
  come from release 1.5, which its published (SVO) curves equal, and v1.9.2
  is the new ``LSST/LSSTv19``. Products composed for ``LSST/LSST`` change.
- SVO's ``uprime_filter`` ... ``zprime_filter`` were taken as the
  filter-only components of SDSS ``u`` ... ``z``; they are the USNO 40-inch
  filters of the primed system, in air, and are no longer used so. SDSS
  components come from the ``sdss`` builder.
- Paranal/NACO bands built from the original source are named ``IB_2p00``
  etc. instead of ``IB_2.00``, as on NJM.
- ``_common.ensure_published`` (DECam, VISTA, HSC, ZTF): when SVO had
  no record for one of a builder's IDs it warned, carried on and wrote a MESA
  index listing only the curves it had; it now writes those curves and raises
  ``RuntimeError`` naming the missing IDs, writing no index.
- ``_common.build_simple``: a filter or composite band without a published
  curve was skipped and the manifest saved; it now raises
  ``FileNotFoundError`` naming the bands before writing any component.
- ``hst.build`` and ``jwst.build`` skipped an instrument with no published
  curves with a warning and returned normally (``hst`` left an empty
  directory); they now build the others, then raise ``FileNotFoundError``
  listing those not built. ``filters-build-components hst``, ``jwst`` or
  ``all`` on a library lacking any of their instruments therefore fails, and
  ``FilterLibrary.prepare_components`` raises instead of returning ``False``
  for such an empty directory.
- ``hst.build`` ignored a published curve it could not read; it now raises
  ``ValueError`` naming the file. HST bands with no single synphot chain
  (ramp, quad and grism filters) are logged as a warning instead of at info
  level.
- ``tabular.build(raw=...)`` silently downloaded a file missing from the
  saved inventory; it still downloads it, with a warning naming it.
- SkyCalc: a failed request to delete ESO's temporary directory is a warning
  instead of a debug message, and "not a FITS file" names the URL.
- Original-source downloads (``Table``, ``Fetcher``, the team atmospheres,
  DECam, HSC, HST, LSST, JWST, SDSS, VISTA) wait 10 s for a connection, as
  SVO requests do; they used a single 120-600 s timeout or a 30 s connect
  timeout. ``SkyCalcClient``'s ``connect_timeout`` default is 10 s (was 30 s).
  Read timeouts are unchanged.
- ``tabular.build`` raised ``KeyError`` for an unknown instrument; it raises
  ``ValueError`` naming the instruments it has.

**Spectral grid downloads**

- ``SED.fetch`` stopped, or carried on with an empty or stale directory, when
  a source's metadata or download failed or downloaded nothing; it now
  reports that source and tries the next. When every source fails it raises
  ``ValueError`` giving each source's error. If a failed attempt has already
  changed the grid directory, no other source is tried: ``RuntimeError``
  says why. The ``grids`` command does the same.
- ``SED.fetch`` and ``SED.query`` caught every exception from a source and
  logged it at debug level; a failing source is now a warning, and only the
  errors a remote source raises are caught. ``SED.query`` raises
  ``RuntimeError`` when every requested remote source fails, and
  ``ValueError`` for an unknown source (it returned ``[]``).
- ``SED.fetch`` from MAST blocked on the MAST grabber's metallicity prompt;
  that source now fails, saying to use ``sed-tools grids --source mast``.
- ``SED.fetch(source="msg")`` found no metadata; it now reads the MSG index
  first.
- ``SED.fetch(...).grid.source`` was ``None`` when the source was chosen
  automatically; it is the source the spectra came from.
- NJM spectra: an unreachable mirror, a grid with no files, or a grid whose
  listing and lookup table both fail made ``get_model_metadata`` return an
  empty result; it now raises ``RuntimeError``. ``download_model_spectra``
  raises when the mirror is unreachable; with Teff, log g or [M/H] cuts and
  an unreadable lookup table it downloaded the whole grid and now raises;
  failed downloads or trims are listed in one ``RuntimeError`` after the rest
  are downloaded (it returned the full count). ``get_grid_info`` warns why it
  returns ``None``.
- SVO spectra: ``get_model_metadata`` returned ``[]`` when no method found a
  spectrum, silently truncated the FID list when the SSAP response reached
  its 50 MB cap (and cached it), and treated a failed probe as an absent FID;
  each now raises, and only a complete FID list is cached.
  ``download_model_spectra`` ignored Teff, log g, [M/H] and wavelength ranges
  and now raises ``ValueError`` for them; when a spectrum fails it now writes
  no ``lookup_table.csv`` (it wrote one for the partial grid) and raises
  listing each FID.
- MSG: ``get_model_metadata`` returned ``[]`` for an unknown grid, a failed or
  unreadable HDF5 file or one without spectra; it now raises. Spectra that
  failed to extract were dropped silently; now the rest are written, no
  lookup table is written, and ``RuntimeError`` lists them. A wavelength
  segment that could not be decoded truncated the spectrum; that layout is
  now abandoned for the others. ``discover_and_run`` finishes the other grids
  before raising. An axis role past the last axis raises instead of giving
  NaN.
- MAST (BOSZ): an unreadable download script dropped that metallicity from
  the grid, scripts listing no spectra and a missing wavelength grid gave 0
  spectra, and an unrecognised metallicity token was dropped (all of them
  dropped fetched every metallicity); each now raises. A failed spectrum is
  listed in one ``RuntimeError`` after the rest are written, with no lookup
  table written. URLs whose names cannot be parsed are kept, so their
  failure is reported. A failed first spectrum is fetched again with the
  rest instead of being skipped.
- Every grabber's ``discover_models`` reports failures and empty results as
  warnings (they were prints or silent).
- The NJM spectra grabber tried three ways (HTTPS, HEAD, plain HTTP) before
  reporting NJM down, about 26 s on a silent outage; no answer now means down
  at once (5 s). A certificate failure is still retried without verification,
  and plain HTTP is used only when the TLS handshake fails both ways.

**Reading and cleaning spectra**

- ``clean_model_dir``, ``SED.fetch``, ``SED.import_grid`` and the CLI
  converted spectra without a flux-unit header using a guess; they now raise
  ``ValueError`` before changing any file, or ask in a terminal (see "Flux
  units"). ``clean_spectrum_file`` on such a file raises ``ValueError``.
- Headers with text such as "low mass", or ``erg/s/cm2/arcsec2``, no longer
  count as declaring a unit, so cleaning stops and asks. Headers giving a
  quantity name and a unit use the unit (``F_nu (Jy)`` is Jansky). Headers
  declaring erg/s/cm²/Å as ``Ångström``, ``Angström`` or ``Angstroem``, or an
  SI unit with a prefix (``mW``, ``kW``, ``uW``), now stop and ask.
  W/m²/nm and W/m²/µm are converted with 1e2 and 0.1 instead of 1e3.
- ``clean_model_dir`` returned its summary as if the grid were clean when
  files were skipped as invalid, as index grids without recovery, or after an
  error, leaving them in their original units beside the converted ones; it
  now tries every file, rebuilds the lookup table, then raises
  ``RuntimeError`` listing each unconverted file by reason. ``SED.fetch``,
  ``SED.import_grid``, ``rebuild`` and ``grids`` receive it.
- A failed rewrite of a spectrum (an ``OSError``, e.g. disk full) was
  recorded and the batch carried on; it now stops at once. A wavelength and
  flux length mismatch truncated the file to its header; the file is now left
  intact and listed.
- An unreadable ``*.txt`` (a permission error, a directory) was treated as an
  empty spectrum; it now raises. A non-numeric wavelength dataset in an
  ``.h5`` file made the whole file be skipped; it is now listed per spectrum.
  An unreadable ``.h5`` during wavelength recovery is a warning instead of a
  debug message. ``clean_spectrum_file`` no longer returns ``'error'``.
- A spectrum whose integrated flux is more than 3 times a blackbody at its
  Teff (with at least 5 % of σT⁴ in range) is divided by that ratio and
  reported (see "Wrong results fixed").
- ``parse_header`` returned NaN parameters, and ``read_text_spectrum`` empty
  arrays, for a file that could not be opened; both now raise ``OSError``.
- ``regenerate_lookup_table`` wrote a ``nan`` column for a parameter no
  spectrum states; it now writes no column, with a note. If only some spectra
  state a parameter it raises ``ValueError`` naming those that do not, and
  writes no table.
- Grids converted file by file because the sample gave no grid-wide units
  now get a warning saying so.
- ``flux_unit=`` (``SED.fetch``, ``SED.import_grid``, ``clean_model_dir``) is
  used only for spectra that do not declare their flux unit, as
  ``--flux-unit`` is; a declared unit is never replaced, and a warning says
  when the given unit is not used.

**Flux cubes, combine, densify and mesa_prepare**

- ``precompute_flux_cube``: a NaN, infinite, blank or non-numeric Teff, log g
  or [M/H] became 0.0 or skipped every spectrum at that node; it now raises
  ``ValueError`` before writing. A lookup table with no Teff column put every
  spectrum on one node; it now raises (missing log g or [M/H] columns still
  read as one 0.0 node). Spectra missing from disk or holding no numeric data
  were skipped, leaving zero nodes; each now raises, listing them. ``split``
  with more than 25 variants in a non-interactive session returned as if it
  had succeeded; it now raises ``RuntimeError``.
- Callers printed "Built flux_cube.bin" whether or not one was written;
  ``SpectralGrid.write``, ``SED.import_grid``, ``grids`` and ``rebuild`` now
  raise ``RuntimeError`` when neither ``flux_cube.bin`` nor
  ``variants_index.csv`` was written.
- ``combine_stellar_atm``: a listed spectrum that was missing, empty or
  unreadable got a normalisation factor of 1 or left its node to be filled
  from a neighbour, and a row with a non-finite Teff, log g or [M/H]
  overwrote node 0; both now raise one ``ValueError`` listing them before any
  output is written. Copy failures were printed while the run reported
  success; they now raise one ``OSError`` after the cube and lookup table are
  written. With no sampled spectrum covering 3000-25000 Å it raised
  ``OverflowError``; now ``ValueError`` saying what to check. Spectra that
  keep their native scale are counted in a warning.
- ``combine`` (menu and ``--non-interactive``) leaves out earlier combine
  outputs; see "Command line" above.
- The densifier replaced a node whose ML generation failed with a scaled
  blackbody, logging a warning; it now raises ``RuntimeError`` naming the
  node. It holds one block of wavelengths in memory (up to half the
  available RAM) instead of using only memory maps; the output is no longer
  zero-filled first, so a failed run leaves the header and the finished
  blocks; progress is printed per block. Its menu rejects 0 and negative
  numbers (they picked from the end of the list).
- ``mesa_prepare.list_variants`` showed an unreadable cube header's
  wavelengths as "n/a"; it now raises. ``export_variant`` skipped missing
  spectra and still reported a MESA-ready folder; it now raises
  ``FileNotFoundError`` listing them, and refuses a variant without its flux
  cube, before writing anything. A uniform grid is no longer noted as
  averaging variants.
- ``sed-tools rebuild`` (``precompute_flux_cube``) took the provenance
  columns of a combined grid (``source_model``, ``original_file``) and of an
  ML-extended grid (``extended_by_ml``, ``wl_min``, ``wl_max``) as extra
  physical axes, so such a grid was split into one "variant" per spectrum and
  got no ``flux_cube.bin`` of its own; they are now bookkeeping columns and
  the grid is built as one cube.
- The rebuild's "Extra physical axis detected" message named every extra
  column (``alpha, lh, vtur`` for Kurucz); it now names only the columns that
  vary. Subgrid directory names are unchanged.

**Photometry and magnitudes**

- ``photometry(("a.dat", "b.dat"))`` with ``b.dat`` an existing file used to
  be two filters; it is now one filter, ``b.dat`` under the name ``a.dat``.
  Pass two arguments or a list for two filters.
- Vega magnitudes (``EvaluatedSED.photometry``, ``flux_cube_tool``) raised
  ``ValueError`` for zero or negative band flux or zero point; they now
  return ``inf`` for zero flux and NaN for negative flux (``-inf`` or NaN for a
  zero or negative zero point), with no warning. AB and bolometric magnitudes
  of negative flux were ``inf``; they are NaN. ``bolometric_magnitude`` with a
  zero or negative reference flux raised; it returns ``-inf`` or NaN.
  Positive-flux magnitudes are unchanged.
- A Vega zero-point key in filter metadata whose value is not a number was
  skipped (the magnitude then came from another key or the Vega spectrum);
  it now raises ``ValueError``. The first key present decides.
- ``load_filter_curve`` and ``load_spectrum`` skipped a non-numeric line
  inside the data, losing a point; it now raises ``ValueError`` naming the
  file and line. Lines before the first data row are still skipped.
- ``flux_cube_tool`` skipped directories it could not search for Vega
  spectra or flux cubes silently; it now warns.

**ML**

- On a Mac with MPS, ML trains and runs on MPS (float32 on the GPU), so
  results differ from a CPU run at float32 rounding level; before, it
  crashed.
- ``extend_catalog``: no ``lookup_table.csv`` now raises
  ``FileNotFoundError`` (it extended every file as Teff 5000, log g 4.5, [M/H]
  0); a missing file, Teff, log g or [M/H] column raises ``ValueError`` (it
  defaulted them, or matched rows to files by position); a row with an
  unusable value or a missing, malformed or rejected spectrum is listed in one
  ``RuntimeError`` after the others are written (it was counted as failed and
  printed only with ``verbose``); any other error stops the run at once.
  ``MLCompleter.extend`` raises ``RuntimeError`` when any spectrum failed and
  ``FileNotFoundError`` when there was nothing to extend (it discarded the
  summary).
- ``complete_sed`` returns the SED's own samples plus points beyond its ends,
  instead of 1,000 resampled points within the range.
- ``auto_train_generator`` returns the trained ``SEDGenerator`` (it returned
  the network module); ``AutoTrainer.train_best`` returns ``(SEDGenerator,
  best_params)``; ``AutoTrainer.objective`` lost its unused ``input_dim``
  parameter. The output directory holds ``model.pt`` and a ``config.json``
  ``SEDGenerator`` reads (it held ``model_best.pt``), is created only when
  training succeeds, and the best parameters are printed. A trial in which
  every batch has a NaN loss stops the search with ``RuntimeError``.
- Removed: ``ml_optimized.ResidualBlock`` and ``SEDModel``,
  ``ml_utils.create_dataloaders`` and ``ml_utils.FluxCubeDataset``, and the
  unused ``SEDCompleter._check_combined_grid`` and ``_build_coverage_map``.
- ``train_model`` ignored keywords other than ``learning_rate``,
  ``hidden_layers``, ``dropout``, ``val_split`` and ``patience``; all now go to
  ``SEDCompleter.train``, so an unknown one raises ``TypeError``.
- ``SEDCompleter.train`` and ``SEDGenerator.train`` saved a model when no
  epoch gave a usable loss (then crashed while plotting); they now raise
  ``RuntimeError`` before saving. ``SEDGenerator.train`` with no flux above
  1e-25 saved a NaN scaler; it now raises ``ValueError``. Non-finite grid
  parameters raise ``ValueError`` instead of ``AssertionError``. Dropped
  training samples are counted in a warning.
- The generator's menus took 0 or a negative number as an index from the end;
  they now say "Invalid selection". The ML menus warn when a lookup table
  cannot be read (a grid used to vanish from the list).
- ``list_models`` (both) catches only ``OSError`` and ``ValueError``; a
  ``config.json`` that is valid JSON but not an object raises.
- ``MLGenerator.parameter_ranges`` returned ``(0, 0)`` for ranges missing from
  a model's config; it now raises ``KeyError``.
- ``MLGenerator.generate_with_outputs`` writes to ``<model>/SED`` (it wrote
  to ``./output/SED``) and raises if the model directory is unknown.

**Python API**

- ``SED.query``: a local grid whose flux cube header or lookup table cannot be
  read was listed with unknown ranges, which passed every filter; it now
  raises ``ValueError`` naming the grid.
- ``SED.local`` and ``SED.fetch``: a spectrum listed in ``lookup_table.csv``
  that cannot be read was dropped at debug level; it now raises
  ``ValueError`` naming the file.
- ``SED.fetch`` returned the grid without spectra that cleaning left
  unusable; it now raises ``RuntimeError`` listing them. Its cleaning message
  reports the converted count (it always said 0).
- ``SpectralGrid.write`` printed a failed flux cube or HDF5 build as a
  warning and returned; the error now propagates, and an HDF5 bundle with no
  ``.txt`` spectra raises ``RuntimeError``.
- ``SED.import_grid`` reported files lacking Teff, log g or [M/H], or unusable
  after cleaning, and then said "done"; it now builds the rest of the grid,
  then raises ``RuntimeError`` listing them. Its "Load with" hint names
  ``model_root`` when the grid went elsewhere than the default root.
- ``SED.query`` and ``SED.fetch`` ignored the parameter bounds when every
  bound given was 0 or 0.0; they now apply them.
- ``clean_model_dir``, ``SED.fetch`` and ``SED.import_grid`` gain
  ``flux_unit=None`` as their last parameter.

**Configuration, memory and packaging**

- A malformed ``~/.sed_tools/config.toml`` was ignored (``data_dir`` fell back
  to the default); it now raises ``ValueError`` naming the file at
  ``import sed_tools``, until the file is fixed or removed. A config path that
  exists but cannot be read raises ``OSError``. On Python 3.10 without
  ``tomli`` (a declared dependency) every config file was ignored; the import
  now fails with ``ModuleNotFoundError``. ``load_toml(strict=True)`` raises
  ``ValueError`` with the path. ``config._load_toml`` is removed. A failing
  ``rmdir`` of an emptied directory when moving the data directory raises.
- Without psutil, available RAM was taken as 2 GiB on macOS, and on Linux
  without a ``MemAvailable`` line; it is now half the physical RAM, with no
  warning. This changes only memory use and chunk counts.
- An error other than ``ImportError`` while importing matplotlib or
  ``flux_cube_tool`` now surfaces as itself (it disabled plotting or was
  reported as a missing module).
- ``import sed_tools.combine_stellar_atm`` works without scipy; the functions
  that need it raise ``ImportError`` naming scipy.
- ``grid_coverage`` lists the representative spectra its terminal preview
  could not plot (they were left out silently).

Version 0.4.1 (October 2026)
----------------------------

Filter metadata robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Keep metadata keys this version does not know. 0.4.0 treated a column-row
  field containing any unknown key as a column name, so a file written by a
  newer SED_Tools read as untagged and re-tagging appended a second field.
  Unknown keys now read normally and are written back after the known ones.
- Remove the ``filters-migrate`` command and ``Filters.migrate_metadata()``.
  Converting a whole tree of 0.3.0 files is mirror maintenance and lives with
  the NJM server tooling; SED_Tools still converts a 0.3.0 file whenever it
  fetches a cached set, downloads, tags or combines it.
- Warn when a converted curve keeps comment or blank lines that released MESA
  counts as samples, instead of treating it as repaired; the lines are left
  for the user to remove. ``filter_metadata.released_mesa_reads()`` runs the
  same check.

Version 0.4.0 (October 2026)
----------------------------

Legacy-safe filter metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Filter metadata (``FilterID``, ``DetectorType``, ``Components`` and the
  rest) is written as a third field of the ``Wavelength,Transmission`` column
  row, e.g. ``Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon``,
  with no whitespace and percent-escaped values. Version 0.3.0 wrote it as
  leading ``#`` lines; released MESA (25.12.1, 26.4.1) sizes its arrays from
  every line after the first, so each of those lines left an uninitialised
  sample at the end of the curve and corrupted zero points without a nonzero
  exit. MESA discards the column row, so curves now load exactly as before.
- Files written by 0.3.0 are still read. Downloading, tagging, combining or
  fetching a cached set moves their metadata onto the column row, keeping
  every sample row byte for byte. ``sed-tools filters-migrate`` /
  ``Filters.migrate_metadata()`` repairs a whole tree offline. A curve with
  no column row is left as it is (or restored, if 0.3.0 tagged it), since
  MESA would otherwise read a sample it has always discarded.
- Regression tests transcribe MESA 26.4.1's ``load_filter`` and its
  ``plot_unit_test`` loader and check the output of every writer against them
  and against pandas, ``csv``, ``np.loadtxt`` and astropy.

Public terminology
~~~~~~~~~~~~~~~~~~

Use "spectral grids" for parameterized collections and "model spectrum"
for an individual synthetic spectrum across documentation, CLI help,
menus and API descriptions. Add ``sed-tools grids``, ``--grids`` and
``--grid`` as preferred CLI names, with the existing commands and options
retained as aliases. Python exposes ``SpectralGrid``, ``SpectralGridInfo``,
``sed.grid``, ``sed.available_grids()`` and ``find_grids`` alongside the existing names. Storage paths
and established keyword arguments remain compatible.

Filter availability and composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Accept comma-separated facility numbers and ranges directly in the facility
  picker. Multiple facilities download all their instruments without further
  selection, confirmation or component prompts; one facility opens its instruments.
- Remove duplicate validation and blanket exception wrappers. Handle network
  failures at source boundaries and let programming and file-write errors
  reach the caller. Atmosphere cache lookup no longer uses exceptions for
  ordinary cache misses.
- Consolidate source selection, component recovery and atmosphere recovery
  in a shared filter-library workflow used by the CLI and Python API.
- Add ``Filters.inventory()`` / ``sed-tools filters-inventory`` to discover
  the union of SVO, NJM, installed and registered instruments, retaining
  source errors and explicit ``unreviewed`` component status.
- Report missing component files and recipes available for individual bands;
  inspect SVO metadata when NJM availability information is unavailable.
- Support direct CLI instrument requests, selected bands, explicit detector
  conventions and opt-in large original-source downloads through both interfaces.
  Detector overrides create derived copies and preserve the original convention.
- Validate component contents, reject additions to unresolved instrument
  throughput, record physical factors separately from recipe selections, and
  preserve existing products when a replacement cannot be completed.
- Recover missing atmosphere data from registered original sources; restrict
  SkyCalc recovery to the requested site and conditions and reject ambiguous
  PWV selection. Track expected SVO bands across partial downloads.

Version 0.3.0 (October 2026)
---------------------------

Optical Path and Detector Awareness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passbands now track the physical elements included along the optical path:
the site's atmosphere, telescope/instrument optics, filter transmission and
detector quantum efficiency (QE). Previously, downloaded curves were used
as a single throughput and photometric integration treated every detector
as energy-counting. The new metadata distinguishes the curve's included
components from its photon- or energy-counting response convention.

``Components`` headers and instrument manifests describe published contents,
available factors and provenance; ``DetectorType`` and
``DetectorTypeSource`` record the integration convention and how it was
identified. Separate factors are available only where supported by the
source data; unresolved instrument contributions and unknown contents are
not presented as independently recoverable elements.

Existing collections can be inspected with
``sed-tools filters-detector --dry-run`` and tagged with
``sed-tools filters-detector``. Unidentified curves remain explicitly
``assumed`` energy-counting. Published passbands remain the default;
automatic discovery skips component and product subdirectories, and a
composed product must be selected explicitly. Tagging and composition do
not require rebuilding flux cubes, but photometry must be recomputed to
use corrected detector weights.

New Features
~~~~~~~~~~~~

**Photon- vs energy-counting filters** (``filter_metadata.py``, ``flux_cube_tool.py``)
   Filter files now carry a ``# DetectorType: energy|photon`` header (with
   ``FilterID``, ``DetectorTypeSource`` and ``WavelengthUnit``). SVO and NJM
   downloads write it; ``sed-tools filters-detector`` /
   ``Filters.tag_detector_types()`` backfill existing files by SVO filter ID or
   by matching the curve against SVO records. Band averages and computed Vega
   zero points weight photon-counting curves by ``T·λ`` (Bessell & Murphy 2012,
   eq. A9). Previously every curve was treated as energy-counting, e.g. a
   0.035 mag error in Gaia DR3 G. Synthetic Vega zero points now reproduce
   SVO's to <1e-4 mag for Gaia G/BP/RP, JWST NIRCam, Johnson U/V and TESS.

**Separable filter components** (``filter_components.py``, ``component_sources/``)
   Filter bandpass, optics, detector QE and telluric atmosphere can now be
   requested separately or combined (``Filters.fetch(..., components=...,
   atmosphere=...)``, ``sed-tools filters-info`` / ``filters-compose``).
   Published curves stay where MESA reads them; components live in
   ``components/``, composed sets in ``products/<recipe>/`` (each a MESA
   instrument directory), tellurics in ``atmospheres/<site>/``. Published
   curves record their contents from SVO (``# Components:``). Builders:
   Argus (reproduces ArgusSim to 2e-4 mag), Rubin/LSST (reproduces Rubin's
   hardware and total curves exactly) and an ESO SkyCalc telluric grid.
   Each instrument is composed only from its own components and its own
   site's atmosphere (never another instrument's or site's); composition
   also refuses to double count, to extrapolate components, and to add an
   atmosphere to space/airborne instruments. ``build_all_products`` writes
   every with/without variant.

**Component sources** (``component_sources/``)
   Builders for HST (synphot graph tables; ~340 bands), JWST NIRCam/NIRISS/
   MIRI (Pandeia 4.0), Subaru HSC, DECam (DES), VISTA (ESO) and ZTF, plus
   automatic use of SVO's filter-only sibling curves (57 instrument groups)
   and an explicit record of instruments whose teams publish totals only.
   Validated against SVO's curves (synthetic photometry of hot and cool
   black bodies); differences are documented (e.g. SVO truncates HST UV red
   leaks). Components are guarded against percent-scaled inputs, and each
   source's own out-of-range convention ("zero"/"edge") is recorded.

Bug Fixes
~~~~~~~~~

**Wavelength unit mis-inference** (``flux_cube_tool.py``)
   Files without a unit header were read as nanometres whenever their largest
   wavelength was below 50,000, so essentially every optical/NIR filter curve
   in Angstrom (Gaia, Johnson, 2MASS, TESS, ...) and any spectrum ending below
   5 microns was stretched by 10x in wavelength. Angstrom is now assumed
   unless the reddest point lies below 1000 (then nm, um, m). Unit headers
   accept common spellings (``Angstroms``, ``Å``, ``microns``, ``µm``); an
   unrecognised unit falls back to inference instead of being silently
   treated as Angstrom. ``load_filter_curve`` and ``load_spectrum`` take a
   ``wavelength_unit=`` override, and a warning is logged whenever the unit
   has to be guessed.

Version 0.1.0 (March–April 2026)
----------------------------------

Initial development sprint covering core pipeline construction, bug fixes,
audit infrastructure, and data quality corrections across all installed
stellar spectral grids.

Bug Fixes
~~~~~~~~~

**Flux cube axis mismatch** (``flux_cube_tool.py``)
   ``FluxCube.from_file`` was reshaping the flat binary buffer as
   ``(nt, nl, nm, nw)`` — the exact inverse of the ``(nw, nm, nl, nt)``
   layout written by ``precompute_flux_cube.py`` and ``combine_stellar_atm.py``.
   Every interpolation result in the codebase was operating on scrambled axes,
   producing wild oscillations and negative flux values at runtime.
   Fixed by reshaping as ``(nw, nm, nl, nt)`` and transposing back to
   ``(nt, nl, nm, nw)``.

**Double normalisation** (``combine_stellar_atm.py``)
   ``renorm_to_sigmaT4`` was being called on spectra already standardised by
   ``spectra_cleaner.py``, causing flux to be rescaled a second time.
   Removed the redundant call; ``spectra_cleaner.py`` is now the sole
   authority for unit standardisation.

**Grid index snapping** (``combine_stellar_atm.py``)
   Three uses of ``np.searchsorted`` (floor behaviour) replaced with
   ``np.argmin(np.abs(...))`` for correct nearest-neighbour snapping to
   grid points.

**Divergent unit conversion path** (``sed_unit_converter.py``)
   A ``/ wl_factor`` step absent from ``spectra_cleaner.py`` was incorrectly
   rescaling flux by factors of 10 or more depending on wavelength unit.
   Removed to align both conversion paths.

**``EvaluatedSED`` missing attribute properties** (``models.py``)
   Parameters were stored in a ``metadata`` dict rather than as direct
   attributes, causing ``AttributeError`` when accessing ``teff``, ``logg``,
   ``metallicity``, ``wl``, or ``fl`` on an interpolated spectrum.
   Added these as properties reading from ``self.metadata``.

**NextGen T ≥ 5000 K flux 100× too large** (``spectra_cleaner.py``)
   Confirmed as a bug in the SVO NextGen catalogue: hot-star entries were
   submitted with flux 100× too large relative to cool stars.
   Fixed by a post-conversion bolometric renormalisation step: if the
   bolometric ratio deviates more than 3× from the expected blackbody
   integral over the file's wavelength range — and that range covers ≥ 5 %
   of the total σT⁴ — the flux is rescaled by the measured ratio.

**``spectra_cleaner.py`` status accounting** (``spectra_cleaner.py``)
   The ``converted_renormed`` status produced by the renorm step was not
   present in the summary dictionary, causing renormed files to be counted
   as errors. Fixed by stripping the ``_renormed`` suffix before routing
   to the summary counter.

**MSG wavelength recovery accepting index sequences** (``msg_spectra_grabber.py``)
   ``_recover_wavelengths`` was falling through to a global HDF5 dataset
   scan and accepting integer index sequences (0, 1, 2, …) as wavelength
   arrays, producing pixel-index wavelength grids for ``sg-BSTAR2006-low``.
   Fixed by introducing ``_is_physical_wavelength()`` validation applied at
   every step of the recovery chain.  A new step 3 searches top-level HDF5
   datasets, where MSG sometimes stores the shared wavelength grid.  The
   global fallback scan now only accepts arrays that pass physical validation.

**Memory crash on large flux cubes** (``grid_densifier.py``)
   Full in-memory allocation of BT-Settl-scale flux cubes (~174 GiB) caused
   system lockups on RHEL.  Fixed by switching to ``np.memmap`` for both
   source and destination arrays, computing blackbody scale factors
   row-by-row, and writing the output file header first then memory-mapping
   the flux region directly on disk.

New Modules
~~~~~~~~~~~

**``sed_tools/_resample.py``**
   Single authoritative resampling function used by ``precompute_flux_cube.py``,
   ``combine_stellar_atm.py``, and ``grid_densifier.py``.  Replaces three
   separate ad-hoc ``np.interp`` calls with inconsistent edge behaviour.
   Within wavelength coverage: log-linear interpolation.
   Outside coverage: zero-fill (no invented flux).

**``sed_tools/grid_densifier.py``**
   Solves stair-stepping artefacts in MESA photometry caused by large Teff
   gaps between grid points (e.g. ~9 000 K in the tmap grid).  Acts as a
   thin wrapper around ``SEDGenerator``, iterating over new Teff values and
   calling ``generator.generate(teff, logg, meta)`` at each node.
   A blackbody fallback (Planck function normalised to the nearest real
   SED's bolometric flux) is used when no ML model is available or the
   requested Teff falls outside the generator's training range.

**``tests/audit_pipeline.py``**
   Comprehensive pipeline audit script.  For every installed spectral grid, checks:

   - ``units_standardized = True`` header flag on raw files
   - Correct wavelength unit (Å) and flux unit (erg cm⁻² s⁻¹ Å⁻¹)
   - Bolometric ratio of raw files against the blackbody integral over the
     file's actual wavelength coverage (coverage-aware, not full σT⁴)
   - Wien peak position sanity, with exemptions for UV/EUV opacity-dominated
     hot compact objects (``WIEN_UV_OPACITY_WL_THRESHOLD = 1500 Å``) and
     band-dominated cool stars (Teff < 1 500 K)
   - Negative flux counts
   - Wavelength coverage
   - Raw vs cube consistency using a tight-tolerance matched-pair comparison:
     for each raw wavelength the nearest cube grid point is found via
     ``np.searchsorted``; the pair is only included if the distance is less
     than 1 % of the minimum raw grid spacing — no interpolation on either
     side, eliminating false positives from molecular band features
   - Collision detection and ``fluxcube_library/`` strategy reporting
   - Sparse-grid detection (cube bolometric check skipped if < 50 % of
     sampled nodes are populated)

   Exempt sets: ``BOL_CHECK_EXEMPT = {grams_cgrid, grams_ogrid, bbody}``,
   ``WIEN_CHECK_EXEMPT = {grams_cgrid, grams_ogrid}``.

   Also generates ``make_raw_cube_diag_plots()``: a 3-panel diagnostic figure
   (SED overlay with bad points marked, relative error, distance from raw to
   nearest cube grid point) for any spectral grid with genuine raw-vs-cube failures.

Pipeline Changes
~~~~~~~~~~~~~~~~

**``precompute_flux_cube.py``**
   Rewritten to support extra axes.  With no extra axes: behaviour is
   identical to the original.  With extra axes present: groups files by
   unique combination, builds a mean cube, per-variant ``fluxcube_library/``
   cubes, and ``index.csv``.  Uses a running sum/count accumulator for the
   mean — single-pass and RAM-efficient.

Audit Results
~~~~~~~~~~~~~

All installed stellar spectral grids pass the full audit suite
(12 PASS / 0 FAIL).  Key findings from the audit process:

- **Husfeld and tmap2 Wien failures** confirmed as false positives:
  UV/EUV atmospheric opacity in hot compact objects physically shifts the
  apparent SED peak redward of the Planck prediction.  Not a unit error.
  Resolved by ``WIEN_UV_OPACITY_WL_THRESHOLD``.

- **NextGen2 raw vs cube mismatch** confirmed as an audit methodology
  artefact: interpolating onto a union wavelength grid caused disagreement
  at molecular band features.  Resolved by the tight-tolerance matched-pair
  comparison described above.

Outstanding Issues
~~~~~~~~~~~~~~~~~~

- **bbody**: Raw files store B\ :sub:`λ` (specific intensity) instead of
  π × B\ :sub:`λ` (surface flux).  Bolometric ratio is off by 1/π.
  Fix requires changes to the bbody generation code.

- **sg-BSTAR2006-low**: HDF5 file has no ``specsource`` group; the grabber
  returns no metadata.  Structure must be inspected before the grabber can
  be fixed.

- **``sed_tools/_resample.py`` docstring**: The module-level docstring
  incorrectly states that extrapolation outside wavelength coverage uses
  a blackbody-scaled fill.  The actual implementation uses zero-fill.
  Docstring needs correcting.
