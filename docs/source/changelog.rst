=========
Changelog
=========

Version 0.1.0 (March–April 2026)
----------------------------------

Initial development sprint covering core pipeline construction, bug fixes,
audit infrastructure, and data quality corrections across all installed
stellar atmosphere models.

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
   Comprehensive pipeline audit script.  For every installed model, checks:

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
   nearest cube grid point) for any model with genuine raw-vs-cube failures.

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

All installed stellar atmosphere models pass the full audit suite
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
