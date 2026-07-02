from __future__ import annotations
import os
import shutil
import warnings
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Union

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

_CONFIG_DIR  = Path("~/.sed_tools").expanduser()
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_DEFAULT_DATA_DIR = _CONFIG_DIR / "data"

# Sub-directories that hold movable data under a data dir.
_DATA_SUBDIRS = ("stellar_models", "filters")
_UI_MODES = {"auto", "always", "never"}


def load_toml(path: Union[str, Path], *, strict: bool = False) -> Dict[str, Any]:
    """Load a TOML mapping through the package's single TOML implementation."""
    if tomllib is None:
        if strict:
            raise ImportError("TOML support requires tomllib or tomli")
        return {}
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        if strict:
            raise
        return {}


_load_toml = load_toml  # private compatibility alias


def load_defaults_config() -> Dict[str, Any]:
    """Load the defaults shipped inside the installed package."""
    return load_toml(files("sed_tools").joinpath("sed_tools.defaults"))


def load_user_config() -> Dict[str, Any]:
    """Load the per-user configuration, returning an empty dict if absent."""
    return load_toml(_CONFIG_FILE)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


merge_config = _merge_dicts


def get_config() -> Dict[str, Any]:
    """Return shipped defaults merged with user and environment settings."""
    cfg = _merge_dicts(load_defaults_config(), load_user_config())
    if os.environ.get("SED_DATA_DIR"):
        cfg["data_dir"] = os.environ["SED_DATA_DIR"]
    return cfg


def get_ui_config() -> Dict[str, str]:
    ui = get_config().get("ui", {})
    return dict(ui) if isinstance(ui, dict) else {}


def get_ui_setting(name: str, default: str = "auto") -> str:
    value = str(get_ui_config().get(name, default)).strip().lower()
    if value not in _UI_MODES:
        warnings.warn(
            f"Invalid ui.{name} value {value!r}; using 'auto'.",
            UserWarning,
            stacklevel=2,
        )
        return "auto"
    return value


def get_data_dir() -> Path:
    data_dir = get_config().get("data_dir")
    if data_dir:
        return Path(data_dir).expanduser()
    return _DEFAULT_DATA_DIR


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_toml(path: Path, config: Dict[str, Any]) -> None:
    """Write the scalar and nested-table shapes supported by SED_Tools."""
    lines = []

    def emit_table(table: Dict[str, Any], prefix: str = "") -> None:
        scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        children = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in children:
            if lines and lines[-1] != "":
                lines.append("")
            section = f"{prefix}.{key}" if prefix else key
            lines.append(f"[{section}]")
            emit_table(value, section)

    emit_table(config)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def has_content(data_dir: Path) -> bool:
    """True if any of the movable data sub-directories exists and is non-empty."""
    data_dir = Path(data_dir)
    for sub in _DATA_SUBDIRS:
        p = data_dir / sub
        if p.is_dir() and any(p.iterdir()):
            return True
    return False


def _move_data(old: Path, new: Path) -> list[tuple[str, str]]:
    """
    Move stellar_models/ and filters/ from old to new, merging recursively.

    Never overwrites an existing target file. Old files that are byte-identical
    to their target are removed (duplicate cleanup); old files that differ from
    an existing target are left in place and returned as conflicts so the caller
    can report them. Emptied source directories are removed.

    Returns a list of (old_path, target_path) conflict tuples.
    """
    import filecmp

    old, new = Path(old), Path(new)
    new.mkdir(parents=True, exist_ok=True)
    conflicts: list[tuple[str, str]] = []

    def _merge(src: Path, dst: Path) -> None:
        if src.is_dir():
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  moved {src} -> {dst}")
                return
            dst.mkdir(parents=True, exist_ok=True)
            for child in list(src.iterdir()):
                _merge(child, dst / child.name)
            try:
                src.rmdir()  # only succeeds if fully drained
            except OSError:
                pass
            return
        # src is a file
        if not dst.exists():
            shutil.move(str(src), str(dst))
            return
        if filecmp.cmp(str(src), str(dst), shallow=False):
            src.unlink()
            print(f"  duplicate identical, removed old: {src}")
        else:
            conflicts.append((str(src), str(dst)))
            print(f"  CONFLICT differs, kept target, left old: {src}")

    for sub in _DATA_SUBDIRS:
        src = old / sub
        if src.is_dir():
            _merge(src, new / sub)

    return conflicts


def set_data_dir(path: str, move: bool | None = None, interactive: bool = False) -> None:
    """
    Set the data directory.

    Parameters
    ----------
    path : new data directory.
    move : if True, move existing stellar_models/ and filters/ from the current
           data dir into the new one. If None and interactive=True, the user is
           prompted (when there is data to move). If None and not interactive,
           nothing is moved (config is just repointed).
    interactive : allow an interactive y/n prompt when move is None.
    """
    old = get_data_dir()
    new = Path(path).expanduser()

    relocate = False
    if old.resolve() != new.resolve() and has_content(old):
        if move is None and interactive:
            ans = input(f"Move existing data from {old} to {new}? [y/N] ").strip().lower()
            relocate = ans.startswith("y")
        else:
            relocate = bool(move)

    if relocate:
        print(f"Moving existing data: {old} -> {new}")
        conflicts = _move_data(old, new)
        if conflicts:
            print(f"\n  {len(conflicts)} file(s) differ between old and target "
                  "and were NOT moved (target kept). Reconcile by hand:")
            for src_f, dst_f in conflicts:
                print(f"    old:    {src_f}")
                print(f"    target: {dst_f}")
        # Report what, if anything, is still sitting at the old location.
        leftover = []
        for sub in _DATA_SUBDIRS:
            p = old / sub
            if p.is_dir():
                leftover += [str(c) for c in p.rglob("*") if c.is_file()]
        if leftover:
            print(f"\n  {len(leftover)} file(s) remain under {old} "
                  "(conflicts and/or non-empty dirs). Nothing was overwritten.")
        else:
            print(f"\n  Old data location {old} fully drained.")

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_user_config()
    existing["data_dir"] = str(new)
    _write_toml(_CONFIG_FILE, existing)
    print(f"Data directory set to: {existing['data_dir']}")
    print(f"Config saved to: {_CONFIG_FILE}")
    if os.environ.get("SED_DATA_DIR"):
        print("Note: SED_DATA_DIR is set and overrides the config file until unset.")


def show_config() -> None:
    data_dir = get_data_dir()
    source = "default"
    if os.environ.get("SED_DATA_DIR"):
        source = "environment variable SED_DATA_DIR"
    elif _CONFIG_FILE.exists():
        source = f"config file ({_CONFIG_FILE})"
    print(f"Data directory : {data_dir}  [{source}]")
    print(f"Stellar models : {data_dir / 'stellar_models'}")
    print(f"Filters        : {data_dir / 'filters'}")
    print(f"Config file    : {_CONFIG_FILE}")
