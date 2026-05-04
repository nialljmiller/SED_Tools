from __future__ import annotations
import os
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


def set_data_dir(path: str) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if _CONFIG_FILE.exists() and tomllib is not None:
        try:
            with open(_CONFIG_FILE, "rb") as f:
                existing = tomllib.load(f)
        except Exception:
            pass
    existing["data_dir"] = str(Path(path).expanduser())
    with open(_CONFIG_FILE, "w") as f:
        for k, v in existing.items():
            f.write(f'{k} = "{v}"\n')
    print(f"Data directory set to: {existing['data_dir']}")
    print(f"Config saved to: {_CONFIG_FILE}")


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