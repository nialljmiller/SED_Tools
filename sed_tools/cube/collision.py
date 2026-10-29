"""The [on_collision] configuration: what precompute_flux_cube does when a grid's extra lookup columns
(alpha, vturb, ...) vary, so that several spectra map to one (Teff, logg, [M/H]) node. The TOML format:

    [on_collision]
    strategy = "all-warn"   # split | all-warn | all | mean | filter  (all-warn, all = split)

    # Only needed when strategy = "filter"
    [on_collision.filter]
    alpha = "min"           # min | max | <exact float value as string>
    f_sed = "2.0"
    composition = "h-rich"  # string: case-insensitive exact match

Later sources win: the packaged sed_tools.defaults (strategy = "all-warn"), <sed_tools_root>/sed_tools.defaults,
<model_dir>/mesa_config.toml, then the Python API's override_dict. The global file is copied into the model
directory when a cube is built, so the configuration used stays beside the data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import load_defaults_config, load_toml, merge_config

# Lookup columns that are not extra axes: MESA's three parameters, then bookkeeping and provenance.
_MESA_PARAM_AXES: Set[str] = {
    "teff", "Teff", "T_eff",
    "logg", "logG", "log g", "log_g", "log(g)", "LOGG",
    "metallicity", "meta", "feh",
    "z", "Z", "zh", "zmet",
}

# Written by the spectra cleaner and regenerate_lookup_table, unless noted
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
    # Provenance of a combined grid (combine_stellar_atm.save_combined_data) and of
    # an ML-extended one (Completer.extend_catalog): they differ row to row, but
    # are not physical axes, so such a grid is one cube, not one per row.
    "source_model", "original_file", "extended_by_ml", "wl_min", "wl_max",
}

MESA_AXES: Set[str] = _MESA_PARAM_AXES | _BOOKKEEPING_AXES

VALID_STRATEGIES = {"split", "all-warn", "all", "mean", "filter"}

def _normalise_key(key: str) -> str:
    """Lowercase with every non-alphanumeric removed: '[alpha/Fe]' -> 'alphafe', 'v_turb' -> 'vturb'."""
    return re.sub(r"[^a-z0-9]+", "", key.strip().lower())


def _keys_match(a: str, b: str) -> bool:
    """Whether two column names refer to the same axis."""
    return _normalise_key(a) == _normalise_key(b)


def _find_column(target: str, columns: List[str]) -> Optional[str]:
    """The first of *columns* naming the same axis as *target*, or None."""
    return next((col for col in columns if _keys_match(col, target)), None)


def _parse_float_safe(s: str) -> Optional[float]:
    """``float(s)``, or None when *s* is a string value such as "h-rich"."""
    try:
        return float(s)
    except ValueError:
        return None


def match_filter_value(
    spec: str,
    available: List[Any],
    param_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Match a filter spec ("min", "max" or a literal value) against a parameter's values in the lookup table.

    Returns ``(matched_value, None)`` or ``(None, error_message)``; param_name is used only in the messages.
    """
    if not available:
        return None, f"No values found for parameter '{param_name}'"

    spec_stripped = spec.strip()
    spec_lower    = spec_stripped.lower()

    float_vals = [v for v in available if _parse_float_safe(str(v)) is not None]
    str_vals   = [v for v in available if _parse_float_safe(str(v)) is None]

    if spec_lower in ("min", "max"):
        pick = min if spec_lower == "min" else max
        if float_vals:
            return pick(float_vals, key=float), None
        return pick(str_vals, key=str), None

    spec_float = _parse_float_safe(spec_stripped)
    if spec_float is not None and float_vals:
        for v in float_vals:
            if abs(float(v) - spec_float) < 1e-9:
                return v, None
        sorted_vals = sorted(float_vals, key=float)
        dists = [(abs(float(v) - spec_float), v) for v in sorted_vals]
        dists.sort()
        nearest = [str(d[1]) for d in dists[:2]]
        return None, (
            f"No exact match for {param_name}={spec_stripped}. "
            f"Available values include: {', '.join(str(v) for v in sorted_vals)}. "
            f"Nearest: {' and '.join(nearest)}"
        )

    for v in available:
        if str(v).strip().lower() == spec_lower:
            return v, None

    return None, (
        f"No match for {param_name}='{spec_stripped}'. "
        f"Available values: {', '.join(str(v) for v in available)}"
    )


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
        """Resolve filter_specs against the values in the lookup table, given as {column_name: [unique values]}.

        Returns ``(resolved, errors)``: {column_name: matched_value} and a list of messages.
        """
        resolved: Dict[str, Any] = {}
        errors:   List[str]      = []

        for spec_key, spec_val in self.filter_specs.items():
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
        filter_specs = {k: str(v) for k, v in raw_filter.items()}

    return CollisionConfig(strategy=strategy, filter_specs=filter_specs, source=source)


_DEFAULTS_FILENAME  = "sed_tools.defaults"
_MODEL_CFG_FILENAME = "mesa_config.toml"


def load_config(
    model_dir:     Optional[Path] = None,
    root_dir:      Optional[Path] = None,
    override_dict: Optional[Dict[str, Any]] = None,
) -> CollisionConfig:
    """Load the on-collision config: the packaged default, overridden by <root_dir>/sed_tools.defaults, then by
    <model_dir>/mesa_config.toml, then by override_dict (the Python API). Any argument may be None.
    """
    data: Dict[str, Any] = load_defaults_config()
    source = "packaged sed_tools.defaults"

    for folder, filename in ((root_dir, _DEFAULTS_FILENAME), (model_dir, _MODEL_CFG_FILENAME)):
        if folder is None:
            continue
        path = Path(folder) / filename
        if path.exists():
            try:
                data = merge_config(data, load_toml(path, strict=True))
                source = str(path)
            except Exception as e:
                raise RuntimeError(f"Error reading {path}: {e}") from e

    if override_dict is not None:
        data = merge_config(data, override_dict)
        source = "Python API dict"

    return _parse_config_dict(data, source)


def copy_global_config_to_model(root_dir: Path, model_dir: Path) -> None:
    """Copy <root_dir>/sed_tools.defaults into model_dir, so the global config a cube was built with stays beside
    it. Does nothing if the global file does not exist.
    """
    src = Path(root_dir) / _DEFAULTS_FILENAME
    if src.exists():
        dst = Path(model_dir) / _DEFAULTS_FILENAME
        dst.write_bytes(src.read_bytes())


def write_default_config(root_dir: Path, overwrite: bool = False) -> Path:
    """Write a commented sed_tools.defaults to root_dir, unless one exists and overwrite is False; returns its path."""
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


def discover_extra_axes(lookup_columns: List[str]) -> List[str]:
    """The lookup columns that are not MESA axes or bookkeeping (MESA_AXES), compared after _normalise_key."""
    mesa_norms = {_normalise_key(k) for k in MESA_AXES}
    return [col for col in lookup_columns if _normalise_key(col) not in mesa_norms]
