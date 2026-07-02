"""
sed_tools/collision_config.py

Loads and validates the on-collision configuration that controls how
precompute_flux_cube.py behaves when multiple SED files map to the same
(Teff, logg, [M/H]) node.

Config file format  (TOML):

    [on_collision]
    strategy = "all-warn"   # all-warn | all | mean | filter

    # Only needed when strategy = "filter"
    [on_collision.filter]
    alpha = "min"           # min | max | <exact float value as string>
    f_sed = "2.0"
    composition = "h-rich"  # string: case-insensitive exact match

Config resolution order (later wins):
    1. Built-in defaults  (strategy = "all-warn")
    2. <sed_tools_root>/sed_tools.defaults
    3. <model_dir>/mesa_config.toml

The global file is also copied into each model directory when a cube is
built so the exact configuration used is reproducible alongside the data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import load_defaults_config, load_toml, merge_config

# ---------------------------------------------------------------------------
# MESA axes — everything else in the lookup table is a candidate extra axis
# ---------------------------------------------------------------------------
# Physical parameter axes MESA cares about
_MESA_PARAM_AXES: Set[str] = {
    "teff", "Teff", "T_eff",
    "logg", "logG", "log g", "log_g", "log(g)", "LOGG",
    "metallicity", "meta", "feh",
    "z", "Z", "zh", "zmet",
}

# Bookkeeping columns written by spectra_cleaner / regenerate_lookup_table
_BOOKKEEPING_AXES: Set[str] = {
    "file_name", "filename", "file",
    "source",
    "units_standardized", "unitsstandardized",
    "wavelength_unit", "wavelengthunit",
    "flux_unit", "fluxunit", "flux_units",
    "original_wavelength_unit", "originalwavelengthunit",
    "original_flux_unit", "originalfluxunit",
    "conversion_confidence", "conversionconfidence",
    "spec_group", "specgroup",
    # common SVO/MSG provenance fields
    "url", "wave_grid", "columns", "res", "atmos",
    # teff, logg, meta variants used by various grabbers
    "t_eff", "log_g", "mh", "m_h", "feh",
    # spectra_cleaner bookkeeping
    "renormed",
    # common provenance / model-type labels
    "atmosphere", "model", "model_type", "grid", "library",
    # MAST BOSZ specific
    "vturb", "cm", "am",
}

MESA_AXES: Set[str] = _MESA_PARAM_AXES | _BOOKKEEPING_AXES

VALID_STRATEGIES = {"split", "all-warn", "all", "mean", "filter"}

# ---------------------------------------------------------------------------
# Fuzzy key normalisation
# ---------------------------------------------------------------------------

def _normalise_key(key: str) -> str:
    """Lowercase, strip non-alphanumeric, collapse runs.

    Examples:
        '[alpha/Fe]' -> 'alphafe'
        'f_sed'      -> 'fsed'
        'Alpha'      -> 'alpha'
        'v_turb'     -> 'vturb'
    """
    return re.sub(r"[^a-z0-9]+", "", key.strip().lower())


def _keys_match(a: str, b: str) -> bool:
    """Return True if two parameter key names refer to the same axis."""
    return _normalise_key(a) == _normalise_key(b)


def _find_column(target: str, columns: List[str]) -> Optional[str]:
    """Find the best-matching column name from a list, or None."""
    norm_target = _normalise_key(target)
    for col in columns:
        if _normalise_key(col) == norm_target:
            return col
    return None


# ---------------------------------------------------------------------------
# Fuzzy value matching
# ---------------------------------------------------------------------------

def _parse_float_safe(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def match_filter_value(
    spec: str,
    available: List[Any],
    param_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Match a filter spec string against the set of available values for a
    parameter.

    spec: "min" | "max" | a literal value string
    available: list of values present in the lookup table for this param
    param_name: used only in error messages

    Returns:
        (matched_value, error_message)
        One of the two will be None.
    """
    if not available:
        return None, f"No values found for parameter '{param_name}'"

    spec_stripped = spec.strip()
    spec_lower    = spec_stripped.lower()

    # --- min / max shortcuts ---
    float_vals = [v for v in available if _parse_float_safe(str(v)) is not None]
    str_vals   = [v for v in available if _parse_float_safe(str(v)) is None]

    if spec_lower == "min":
        if float_vals:
            return min(float_vals, key=float), None
        return min(str_vals, key=str), None

    if spec_lower == "max":
        if float_vals:
            return max(float_vals, key=float), None
        return max(str_vals, key=str), None

    # --- try numeric exact match ---
    spec_float = _parse_float_safe(spec_stripped)
    if spec_float is not None and float_vals:
        # Exact match within tight epsilon
        for v in float_vals:
            if abs(float(v) - spec_float) < 1e-9:
                return v, None
        # No exact match — report two nearest
        sorted_vals = sorted(float_vals, key=float)
        dists = [(abs(float(v) - spec_float), v) for v in sorted_vals]
        dists.sort()
        nearest = [str(d[1]) for d in dists[:2]]
        return None, (
            f"No exact match for {param_name}={spec_stripped}. "
            f"Available values include: {', '.join(str(v) for v in sorted_vals)}. "
            f"Nearest: {' and '.join(nearest)}"
        )

    # --- string exact match (case-insensitive, stripped) ---
    spec_norm = spec_lower.strip()
    for v in available:
        if str(v).strip().lower() == spec_norm:
            return v, None

    # No match
    return None, (
        f"No match for {param_name}='{spec_stripped}'. "
        f"Available values: {', '.join(str(v) for v in available)}"
    )


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class CollisionConfig:
    """Resolved on-collision configuration for one model."""

    def __init__(
        self,
        strategy: str = "all-warn",
        filter_specs: Optional[Dict[str, str]] = None,
        source: str = "default",
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of: {', '.join(sorted(VALID_STRATEGIES))}"
            )
        self.strategy    = strategy
        self.filter_specs: Dict[str, str] = filter_specs or {}
        self.source      = source   # for reporting

    def __repr__(self) -> str:
        return (f"CollisionConfig(strategy={self.strategy!r}, "
                f"filter_specs={self.filter_specs!r}, source={self.source!r})")

    def resolve_filter_values(
        self,
        extra_columns: Dict[str, List[Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Resolve filter_specs against the actual values present in the
        lookup table.

        extra_columns: {column_name: [list of unique values]}

        Returns:
            (resolved: {col_name: matched_value}, errors: [str])
        """
        resolved: Dict[str, Any] = {}
        errors:   List[str]      = []

        for spec_key, spec_val in self.filter_specs.items():
            # Find which actual column this spec refers to
            matched_col = _find_column(spec_key, list(extra_columns.keys()))
            if matched_col is None:
                errors.append(
                    f"Filter key '{spec_key}' does not match any extra axis. "
                    f"Available: {', '.join(extra_columns.keys())}"
                )
                continue

            value, err = match_filter_value(
                spec_val, extra_columns[matched_col], matched_col
            )
            if err:
                errors.append(err)
            else:
                resolved[matched_col] = value

        return resolved, errors


def _parse_config_dict(data: Dict[str, Any], source: str) -> CollisionConfig:
    """Parse a raw TOML dict into a CollisionConfig."""
    oc = data.get("on_collision", {})
    strategy = oc.get("strategy", "all-warn")

    filter_specs: Dict[str, str] = {}
    if strategy == "filter":
        raw_filter = oc.get("filter", {})
        if not isinstance(raw_filter, dict):
            raise ValueError(f"[on_collision.filter] must be a table in {source}")
        for k, v in raw_filter.items():
            filter_specs[k] = str(v)

    return CollisionConfig(strategy=strategy, filter_specs=filter_specs, source=source)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULTS_FILENAME  = "sed_tools.defaults"
_MODEL_CFG_FILENAME = "mesa_config.toml"


def load_config(
    model_dir:     Optional[Path] = None,
    root_dir:      Optional[Path] = None,
    override_dict: Optional[Dict[str, Any]] = None,
) -> CollisionConfig:
    """
    Load the on-collision config with proper precedence:

        built-in default
            ↓ overridden by
        <root_dir>/sed_tools.defaults
            ↓ overridden by
        <model_dir>/mesa_config.toml
            ↓ overridden by
        override_dict  (Python API)

    Any of the optional arguments may be None.
    """
    data: Dict[str, Any] = load_defaults_config()
    source = "packaged sed_tools.defaults"

    # Global defaults
    if root_dir is not None:
        global_path = Path(root_dir) / _DEFAULTS_FILENAME
        if global_path.exists():
            try:
                data = merge_config(data, load_toml(global_path, strict=True))
                source = str(global_path)
            except Exception as e:
                raise RuntimeError(f"Error reading {global_path}: {e}") from e

    # Per-model override
    if model_dir is not None:
        model_path = Path(model_dir) / _MODEL_CFG_FILENAME
        if model_path.exists():
            try:
                data = merge_config(data, load_toml(model_path, strict=True))
                source = str(model_path)
            except Exception as e:
                raise RuntimeError(f"Error reading {model_path}: {e}") from e

    # Python API dict override (highest priority)
    if override_dict is not None:
        data = merge_config(data, override_dict)
        source = "Python API dict"

    return _parse_config_dict(data, source)


def copy_global_config_to_model(root_dir: Path, model_dir: Path) -> None:
    """
    Copy <root_dir>/sed_tools.defaults into <model_dir>/ so the exact
    global config used to build the cube is preserved alongside the data.
    Does nothing if the global file does not exist.
    """
    src = Path(root_dir) / _DEFAULTS_FILENAME
    if src.exists():
        dst = Path(model_dir) / _DEFAULTS_FILENAME
        dst.write_bytes(src.read_bytes())


def write_default_config(root_dir: Path, overwrite: bool = False) -> Path:
    """
    Write a commented sed_tools.defaults file to root_dir if one does not
    already exist (or if overwrite=True).  Returns the path.
    """
    path = Path(root_dir) / _DEFAULTS_FILENAME
    if path.exists() and not overwrite:
        return path

    content = """\
# sed_tools.defaults
# Global on-collision configuration for precompute_flux_cube.
# Per-model overrides go in <model_dir>/mesa_config.toml (same schema).
#
# strategy options:
#   split     — (default) split each extra physical axis into a separate
#               MESA-ready subgrid directory.  Physically safe: no averaging.
#               Subdirs are named {Model}_{axis}_{value}/.
#               A variants_index.csv is written in the parent directory.
#   all-warn  — alias for split (backward compatible).
#   all       — alias for split (backward compatible).
#   mean      — collapse extra axes by averaging into a single cube.
#               Physically unsafe — produces a synthetic mean atmosphere.
#   filter    — filter to one specific extra-axis slice, then build.
#               Requires [on_collision.filter] section below.

[on_collision]
strategy = "split"

# Uncomment and fill in when strategy = "filter":
# [on_collision.filter]
# alpha = "min"          # min | max | exact value e.g. "0.4"
# f_sed = "2.0"
# composition = "h-rich" # string: case-insensitive
"""
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Extra-axis discovery from lookup table
# ---------------------------------------------------------------------------

def discover_extra_axes(lookup_columns: List[str]) -> List[str]:
    """
    Return the subset of lookup table column names that are NOT MESA axes.
    Uses fuzzy normalisation so variants like '[alpha/Fe]', 'alpha_fe',
    'conversion_confidence' etc. are correctly matched against MESA_AXES.
    """
    mesa_norms = {_normalise_key(k) for k in MESA_AXES}
    extra = []
    for col in lookup_columns:
        if _normalise_key(col) not in mesa_norms:
            extra.append(col)
    return extra
