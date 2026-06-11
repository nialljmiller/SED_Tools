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