from __future__ import annotations
import os
import shutil
from pathlib import Path

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


def get_data_dir() -> Path:
    env = os.environ.get("SED_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if _CONFIG_FILE.exists() and tomllib is not None:
        try:
            with open(_CONFIG_FILE, "rb") as f:
                cfg = tomllib.load(f)
            data_dir = cfg.get("data_dir")
            if data_dir:
                return Path(data_dir).expanduser()
        except Exception:
            pass
    return _DEFAULT_DATA_DIR


def has_content(data_dir: Path) -> bool:
    """True if any of the movable data sub-directories exists and is non-empty."""
    data_dir = Path(data_dir)
    for sub in _DATA_SUBDIRS:
        p = data_dir / sub
        if p.is_dir() and any(p.iterdir()):
            return True
    return False


def _move_data(old: Path, new: Path) -> None:
    """Move stellar_models/ and filters/ from old to new, merging if needed."""
    old, new = Path(old), Path(new)
    new.mkdir(parents=True, exist_ok=True)
    for sub in _DATA_SUBDIRS:
        src = old / sub
        if not src.is_dir():
            continue
        dst = new / sub
        if not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  moved {src} -> {dst}")
            continue
        # Merge child-by-child; never overwrite an existing target.
        dst.mkdir(parents=True, exist_ok=True)
        for child in list(src.iterdir()):
            target = dst / child.name
            if target.exists():
                print(f"  skip (exists in target): {target}")
                continue
            shutil.move(str(child), str(target))
            print(f"  moved {child} -> {target}")
        try:
            src.rmdir()
        except OSError:
            pass


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
        _move_data(old, new)

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if _CONFIG_FILE.exists() and tomllib is not None:
        try:
            with open(_CONFIG_FILE, "rb") as f:
                existing = tomllib.load(f)
        except Exception:
            pass
    existing["data_dir"] = str(new)
    with open(_CONFIG_FILE, "w") as f:
        for k, v in existing.items():
            f.write(f'{k} = "{v}"\n')
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
