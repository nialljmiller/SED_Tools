#!/usr/bin/env python3
"""Standardize a grid's spectra for MESA: wavelength in Å, flux as F_lambda in erg/cm²/s/Å, a standard header.

A grid has one unit system, taken from a sample of its spectra; an undeclared flux unit is never guessed (pass
flux_unit). Each file is tagged '# units_standardized = True'. units.py and hdf5_wavelengths.py hold the rest.
"""

from __future__ import annotations

import glob
import logging
import os
import random  # noqa: F401 - tests patch spectra_cleaner.random.sample, which units.py uses
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._constants import C, SIGMA, planck_flam  # noqa: F401 - C is importable from here since 0.5.0
from .lookup import write_lookup_csv
from .hdf5_wavelengths import (  # noqa: F401 - the cleaner's whole surface stays importable from here
    WAVE_KEYS,
    is_index_grid,
    recover_wavelengths_from_hdf5,
)
from .spectrum_io import read_text_spectrum  # noqa: F401
from .units import (  # noqa: F401
    FLUX_GUESS_WHY,
    FLUX_UNITS,
    SPEED_OF_LIGHT_ANGSTROM,
    WAVELENGTH_FACTORS,
    SpectrumMeta,
    UnitInfo,
    _check_flux_unit_name,
    _load_header,
    _with_flux_unit,
    check_flux_unit,
    convert_to_standard_units,
    detect_catalog_units,
    detect_flux_unit_from_range,
    detect_units_from_header,
    detect_units_single_file,
    detect_wavelength_unit_from_range,
    read_spectrum_data,
    undeclared_flux_unit,
    validate_converted_spectrum,
)

logger = logging.getLogger("sed_tools.spectra_cleaner")


# ---- Constants ----

# A spectrum whose integrated flux is more than 3 times a blackbody's at its
# Teff, over the same wavelengths, has a wrong absolute flux scale (SVO's NextGen
# T >= 5000 K entries are about 100x too large). It is only checked when those
# wavelengths hold at least 5 % of sigma*T^4. Spectra far below the blackbody are
# left alone: cool giants' optical flux really is 2-10 % of it.
_RENORM_MAX_RATIO = 3.0
_RENORM_MIN_COVERAGE = 0.05

STANDARD_HEADER_TEMPLATE = """\
# source = {source}
# teff = {teff}
# logg = {logg}
# metallicity = {metallicity}
# wavelength_unit = Angstrom
# flux_unit = erg/cm2/s/A
# units_standardized = True
# original_wavelength_unit = {orig_wl_unit}
# original_flux_unit = {orig_flux_unit}
# conversion_confidence = {confidence}
"""


# ---- File writing ----

def write_standardized_spectrum(
    filepath: str,
    wl: np.ndarray,
    flux: np.ndarray,
    meta: SpectrumMeta,
    unit_info: UnitInfo,
) -> None:
    """Write standardized spectrum with proper header."""
    header = STANDARD_HEADER_TEMPLATE.format(
        source=meta.source,
        teff=meta.teff if np.isfinite(meta.teff) else 'nan',
        logg=meta.logg if np.isfinite(meta.logg) else 'nan',
        metallicity=meta.metallicity if np.isfinite(meta.metallicity) else 'nan',
        orig_wl_unit=unit_info.wavelength_unit,
        orig_flux_unit=unit_info.flux_type,
        confidence=unit_info.confidence,
    )
    
    if meta.spec_group:  # for HDF5 wavelength recovery
        header += f"# spec_group = {meta.spec_group}\n"
    for key, value in meta.extra.items():
        if key not in ('wavelength_unit', 'flux_unit', 'units_standardized',
                       'original_wavelength_unit', 'original_flux_unit', 'conversion_confidence'):
            header += f"# {key} = {value}\n"
    
    # Stack first so a length mismatch fails before the file is truncated.
    data = np.column_stack([wl, flux])
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
    with open(filepath, 'ab') as f:
        np.savetxt(f, data, fmt='%.6f %.8e')


# ---- Main cleaning logic ----

def clean_spectrum_file(
    filepath: str,
    catalog_units: Optional[UnitInfo] = None,
    try_h5_recovery: bool = True,
) -> Tuple[str, str]:
    """
    Clean and standardize a single spectrum file.
    
    Args:
        filepath: Path to spectrum file
        catalog_units: Pre-determined units for the spectral grid (from the sample).
                       If None, detects units per-file; a flux unit the header
                       does not declare then raises ValueError.
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
    
    Returns: (status, detail)
        status: 'skipped_already', 'skipped_index', 'skipped_invalid', 
                'converted', 'recovered', or 'converted_renormed' /
                'recovered_renormed' when the flux scale was also corrected
                (see _bolometric_renorm_ratio); detail then gives the factor.

    Raises RuntimeError if rewriting the file fails; the file may then hold
    only its new header.
    """
    status, detail, renorm_ratio = _clean_spectrum(filepath, catalog_units, try_h5_recovery)
    if renorm_ratio is not None:
        status += '_renormed'
    return status, detail


def _clean_spectrum(
    filepath: str,
    catalog_units: Optional[UnitInfo],
    try_h5_recovery: bool,
    flux_unit: Optional[str] = None,
) -> Tuple[str, str, Optional[float]]:
    """clean_spectrum_file's work, plus the factor the flux was divided by (or None); flux_unit is for per-file units."""
    meta, _ = _load_header(filepath)
    if meta.units_standardized:
        return 'skipped_already', 'already standardized', None
    wl, flux = read_spectrum_data(filepath)
    if wl.size == 0 or flux.size == 0:
        return 'skipped_invalid', 'empty data', None
    valid = np.isfinite(wl) & np.isfinite(flux) & (wl > 0)
    wl, flux = wl[valid], flux[valid]
    
    if wl.size < 2:
        return 'skipped_invalid', 'no valid data points', None
    
    wl, flux = _sorted_unique(wl, flux)
    
    if is_index_grid(wl):
        wl = _recovered_wavelengths(filepath, meta, len(flux), try_h5_recovery)
        if wl is None:
            return 'skipped_index', 'index grid without HDF5 recovery', None
        status = 'recovered'
    else:
        status = 'converted'
    
    # The file has at least two valid points by now, so per-file detection always succeeds
    if catalog_units is not None:
        unit_info = catalog_units
    else:
        unit_info = _with_flux_unit("this spectrum", detect_units_single_file(filepath), flux_unit, report=False)
    wl_std, flux_std = convert_to_standard_units(wl, flux, unit_info)
    valid, msg = validate_converted_spectrum(wl_std, flux_std)
    if not valid:
        return 'skipped_invalid', f'validation failed: {msg}', None

    renorm_ratio = _bolometric_renorm_ratio(wl_std, flux_std, meta.teff)
    if renorm_ratio is not None:
        flux_std = flux_std / renorm_ratio

    try:
        write_standardized_spectrum(filepath, wl_std, flux_std, meta, unit_info)
    except OSError as exc:
        raise RuntimeError(
            f"Failed while rewriting {filepath}: {exc}. The file may now hold only its new "
            "header, which marks it as standardized, so later runs would skip it. Restore "
            "it from the original download, then run the cleaning again."
        ) from exc
    
    detail = f"{wl_std[0]:.0f}-{wl_std[-1]:.0f}Å, {unit_info.wavelength_unit}→Å, conf={unit_info.confidence}"
    if renorm_ratio is not None:
        detail += f", flux divided by {renorm_ratio:.4g} (bolometric renormalisation)"
    return status, detail, renorm_ratio


def _bolometric_renorm_ratio(wl: np.ndarray, flux: np.ndarray, teff: float) -> Optional[float]:
    """The factor to divide the flux by when its absolute scale is wrong, else None.

    The factor is the integrated flux over pi*B_lambda(Teff) integrated on the same wavelengths: both
    integrals share one quadrature, so a correctly scaled spectrum gives about 1 however it is sampled.
    """
    if not (np.isfinite(teff) and teff > 0):
        return None
    expected = float(np.trapezoid(planck_flam(wl, teff), wl))
    if expected < _RENORM_MIN_COVERAGE * SIGMA * teff ** 4:
        return None
    ratio = float(np.trapezoid(np.maximum(flux, 0), wl)) / expected
    if ratio > _RENORM_MAX_RATIO:
        return ratio
    return None


def _sorted_unique(wl: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by wavelength and drop repeated wavelengths, keeping the first."""
    order = np.argsort(wl)
    wl, flux = wl[order], flux[order]
    unique_mask = np.concatenate([[True], np.diff(wl) > 0])
    return wl[unique_mask], flux[unique_mask]


def _recovered_wavelengths(filepath: str, meta: SpectrumMeta, n_points: int,
                           try_h5_recovery: bool) -> Optional[np.ndarray]:
    """Real wavelengths for an index-grid MSG spectrum from the grid's HDF5 files, or None."""
    if not (try_h5_recovery and meta.spec_group and 'msg' in meta.source.lower()):
        return None
    return recover_wavelengths_from_hdf5(os.path.dirname(filepath), meta.spec_group, n_points)


def rebuild_lookup_table(model_dir: str) -> str:
    """Rebuild lookup_table.csv from spectrum file headers."""
    txt_files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not txt_files:
        return ""
    
    rows = []
    all_keys = {'file_name'}
    
    for filepath in txt_files:
        meta, _ = _load_header(filepath)
        row = {'file_name': os.path.basename(filepath)}
        for param in ('teff', 'logg', 'metallicity'):
            if np.isfinite(getattr(meta, param)):
                row[param] = getattr(meta, param)
        row['source'] = meta.source
        row['units_standardized'] = str(meta.units_standardized)
        row.update(meta.extra)
        rows.append(row)
        all_keys.update(row.keys())
    
    out_path = os.path.join(model_dir, "lookup_table.csv")
    priority = ['file_name', 'teff', 'logg', 'metallicity', 'source', 'units_standardized']
    columns = [c for c in priority if c in all_keys]
    columns += sorted(k for k in all_keys if k not in priority)
    write_lookup_csv(rows, out_path, columns=columns)
    return out_path


# ---- Public API ----

def clean_model_dir(
    model_dir: str,
    try_h5_recovery: bool = True,
    rebuild_lookup: bool = True,
    sample_fraction: float = 0.10,
    flux_unit: Optional[str] = None,
) -> Dict:
    """
    Clean all spectra in a model directory.
    
    KEY: Detects units from a random sample first (default 10%), then applies uniformly.
    
    Args:
        model_dir: Directory containing *.txt spectrum files
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
        rebuild_lookup: Rebuild lookup_table.csv after cleaning
        sample_fraction: Fraction of files to sample for unit detection (default 10%)
        flux_unit: The spectra's flux unit, a name in FLUX_UNITS ('flam', 'fnu',
            'fnu_jy', 'flam_si', 'flam_si_nm', 'flam_si_um', 'normalized'),
            used only for spectra whose headers do not declare theirs, and
            required when the sampled ones declare none. A declared unit is
            never replaced; a warning says when flux_unit is not used.
    
    Returns:
        Summary dict with counts and file lists. When any spectrum's flux scale
        was corrected it also has 'renormed': {file name: factor its flux was
        divided by}; those files are listed under 'converted' or 'recovered' too.

    Raises:
        ValueError: before any file is changed, if flux_unit is not a FLUX_UNITS
            name, or is None while the sampled headers do not all declare the
            flux unit; the message gives the best guess.
        RuntimeError: if any spectrum is left unconverted (invalid, unrecoverable
            index grid, unreadable, or without a declared flux unit when no
            grid-wide units were found). Raised after every file has been tried
            and the lookup table rebuilt, listing each such file and why.
            Raised at once if rewriting a file fails.
    """
    _check_flux_unit_name(flux_unit)
    txt_files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    
    summary = {
        'total': len(txt_files),
        'converted': [],
        'recovered': [],
        'skipped_already': [],
        'skipped_index': [],
        'skipped_invalid': [],
        'error': [],
        'lookup_updated': False,
        'lookup_path': '',
        'catalog_units': None,
        'detection_stats': {},
    }
    
    if not txt_files:
        return summary
    catalog_units, detection_stats = detect_catalog_units(txt_files, sample_fraction)
    if catalog_units is not None:
        catalog_units = _with_flux_unit(model_dir, catalog_units, flux_unit)
    summary['detection_stats'] = detection_stats
    if catalog_units:
        summary['catalog_units'] = {
            'wavelength': catalog_units.wavelength_unit,
            'flux': catalog_units.flux_type,
            'confidence': catalog_units.confidence,
        }
    _print_detected_units(catalog_units, detection_stats)
    failures = _clean_all_files(txt_files, catalog_units, try_h5_recovery, summary, flux_unit)
    n_per_file = len(summary['converted']) + len(summary['recovered'])
    if catalog_units is None and n_per_file:
        logger.warning(
            "%s: the sampled files gave no grid-wide units (status: %s); %d spectra were "
            "converted using units detected file by file.",
            model_dir, detection_stats.get('status', 'unknown'), n_per_file,
        )
    elif catalog_units is None and detection_stats.get('status') != 'already_standardized':
        print(f"[clean] Warning: Could not determine spectral grid units. "
              f"Status: {detection_stats.get('status', 'unknown')}")
    if 'renormed' in summary:
        _warn_renormed(model_dir, summary['renormed'])
    if rebuild_lookup:
        lookup_path = rebuild_lookup_table(model_dir)
        if lookup_path:
            summary['lookup_updated'] = True
            summary['lookup_path'] = lookup_path
    
    if failures:
        raise RuntimeError(
            _unconverted_report(model_dir, len(txt_files), failures, summary['lookup_updated'])
        )
    return summary


# Outcomes that leave a spectrum in its original units.
_UNCONVERTED = ('skipped_invalid', 'skipped_index', 'error')


def _print_detected_units(catalog_units: Optional[UnitInfo], stats: Dict) -> None:
    """The grid's units, or that the sample was already standardized; undetermined units are reported after cleaning."""
    if catalog_units:
        print(f"[clean] Spectral grid units (from {stats.get('sample_size', '?')} samples): "
              f"λ={catalog_units.wavelength_unit} ({stats.get('wavelength_agreement', '?')}), "
              f"flux={catalog_units.flux_type} ({stats.get('flux_agreement', '?')})")
    elif stats.get('status') == 'already_standardized':
        print(f"[clean] Spectral grid already standardized (all {stats.get('sample_size', '?')} samples)")


def _clean_all_files(txt_files: List[str], catalog_units: Optional[UnitInfo],
                     try_h5_recovery: bool, summary: Dict,
                     flux_unit: Optional[str]) -> List[Tuple[str, str]]:
    """Clean every file into its status list in *summary* (renormalised ones also in summary['renormed']).

    Returns (file name, reason) for every spectrum left unconverted. A file that
    cannot be read is recorded and the batch goes on; a failed rewrite raises.
    """
    failures = []
    renormed = {}
    t_start = time.time()
    for i, filepath in enumerate(txt_files):
        filename = os.path.basename(filepath)
        try:
            status, detail, renorm_ratio = _clean_spectrum(filepath, catalog_units, try_h5_recovery,
                                                           flux_unit)
        except (OSError, ValueError) as exc:
            # Raised before the file is rewritten (rewrite failures raise RuntimeError).
            status, detail, renorm_ratio = 'error', f'{type(exc).__name__}: {exc}', None
        summary[status].append(filename)
        if renorm_ratio is not None:
            renormed[filename] = renorm_ratio
        if status in _UNCONVERTED:
            failures.append((filename, detail))
        _print_progress(i + 1, len(txt_files), t_start)
    print()
    if renormed:
        summary['renormed'] = renormed
    return failures


def _warn_renormed(model_dir: str, renormed: Dict[str, float]) -> None:
    ratios = list(renormed.values())
    logger.warning(
        "%s: %d spectra integrated to %.3g-%.3g times a blackbody at their Teff, so each "
        "one's flux was divided by its factor (bolometric renormalisation). The summary's "
        "'renormed' entry lists every file and factor.",
        model_dir, len(ratios), min(ratios), max(ratios),
    )


def _print_progress(done: int, total: int, t_start: float) -> None:
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 1.0
    eta = (total - done) / rate
    print(
        f"\r  [clean] {done}/{total} ({100*done/total:.1f}%)  "
        f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
        end="", flush=True,
    )


def _unconverted_report(model_dir: str, n_total: int, failures: List[Tuple[str, str]],
                        lookup_rebuilt: bool) -> str:
    """Error text listing every unconverted spectrum, grouped by reason."""
    by_reason = defaultdict(list)
    for filename, reason in failures:
        by_reason[reason].append(filename)
    lines = [f"{len(failures)} of {n_total} spectra in {model_dir} were not converted "
             "and are still in their original units:"]
    for reason, filenames in by_reason.items():
        lines.append(f"  {reason} ({len(filenames)}): {', '.join(filenames)}")
    if len(failures) < n_total:
        rebuilt = " and lookup_table.csv was rebuilt" if lookup_rebuilt else ""
        lines.append(f"The other spectra were standardized in place{rebuilt}; a re-run skips them.")
    lines.append("Fix or remove the listed files, then clean the directory again.")
    return "\n".join(lines)
