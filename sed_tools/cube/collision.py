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

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..config import load_defaults_config, load_toml, merge_config
from ..grids.lookup import _FILE_CANDIDATES, _LOGG_CANDIDATES, _META_CANDIDATES, _TEFF_CANDIDATES

# Lookup columns that are not extra axes: the file column and MESA's three parameters, under every name
# grids.lookup's find_*_column accept, then bookkeeping.
_MESA_PARAM_AXES: Set[str] = {*_FILE_CANDIDATES, *_TEFF_CANDIDATES, *_LOGG_CANDIDATES, *_META_CANDIDATES}

# Bookkeeping: the columns SED_Tools itself writes to describe a file (where it came from, its units and format,
# what was done to it), never a property of the model. Every other column is one (discover_extra_axes).
_BOOKKEEPING_AXES: Set[str] = {
    # the spectra cleaner (grids/clean.py)
    "source", "spec_group", "units_standardized", "wavelength_unit", "flux_unit", "original_wavelength_unit",
    "original_flux_unit", "conversion_confidence",
    # the MSG and MAST grabbers (grids/msg.py, grids/mast.py)
    "negative_flux_fraction", "columns", "url", "wave_grid", "res",
    # combine (cube/combine.py save_combined_data) and Completer.extend_catalog
    "source_model", "original_file", "extended_by_ml", "wl_min", "wl_max",
}

MESA_AXES: Set[str] = _MESA_PARAM_AXES | _BOOKKEEPING_AXES

VALID_STRATEGIES = {"split", "all-warn", "all", "mean", "filter"}

def _normalise_key(key: str) -> str:
    """Lowercase with every non-alphanumeric removed: '[alpha/Fe]' -> 'alphafe', 'v_turb' -> 'vturb'."""
    return re.sub(r"[^a-z0-9]+", "", key.strip().lower())


def _value(text: Any) -> Any:
    """An extra-axis value as variants are told apart by it: a number where it reads as one, so "0" and "0.0" are
    one value; otherwise the text, stripped and lower-cased."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        number = float("nan")
    return number if math.isfinite(number) else str(text).strip().lower()


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
        dists = sorted((abs(float(v) - spec_float), v) for v in sorted_vals)
        nearest = [str(v) for _, v in dists[:2]]
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

            value, err = match_filter_value(spec_val, extra_columns[matched_col], matched_col)
            if err:
                errors.append(err)
            else:
                resolved[matched_col] = value

        return resolved, errors


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
            data = merge_config(data, load_toml(path, strict=True))  # ValueError naming the file if it cannot be parsed
            source = str(path)

    if override_dict is not None:
        data = merge_config(data, override_dict)
        source = "Python API dict"

    oc = data.get("on_collision", {})
    strategy = oc.get("strategy", "all-warn")
    filter_specs: Dict[str, str] = {}
    if strategy == "filter":
        filter_specs = {k: str(v) for k, v in oc.get("filter", {}).items()}
    return CollisionConfig(strategy=strategy, filter_specs=filter_specs, source=source)


def copy_global_config_to_model(root_dir: Path, model_dir: Path) -> None:
    """Copy <root_dir>/sed_tools.defaults into model_dir, so the global config a cube was built with stays beside
    it. Does nothing if the global file does not exist.
    """
    src = Path(root_dir) / _DEFAULTS_FILENAME
    if src.exists():
        dst = Path(model_dir) / _DEFAULTS_FILENAME
        dst.write_bytes(src.read_bytes())


def discover_extra_axes(columns: Dict[str, Sequence], nodes: Sequence[Tuple]) -> List[str]:
    """The extra axes among a grid's lookup ``columns`` ({name: one value per row}), its rows being at ``nodes``
    (each row's Teff, log g and [M/H]).

    A column that is a file or parameter column or bookkeeping (MESA_AXES, compared after _normalise_key) is no
    axis. Every other column states something about the model. It is an axis where it tells apart rows at one node
    (BOSZ's [alpha/M]: the grid is split by it) or has one value (Kurucz's lh and vtur: it names the variants); one
    that varies only between nodes labels the spectra (hres's model code, Phoenix or Tlusty) and splits nothing.
    Values are compared as _value compares them.
    """
    mesa_norms = {_normalise_key(k) for k in MESA_AXES}
    axes = []
    for name, values in columns.items():
        if _normalise_key(name) in mesa_norms:
            continue
        at_node: Dict[Tuple, Set] = {}
        for node, value in zip(nodes, values):
            at_node.setdefault(tuple(node), set()).add(_value(value))
        varies = len(set().union(*at_node.values())) > 1
        if not varies or any(len(found) > 1 for found in at_node.values()):
            axes.append(name)
    return axes
