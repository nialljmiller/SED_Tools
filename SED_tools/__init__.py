"""Compatibility layer exposing both programmatic and CLI entry points."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import sed_tools as _core
from sed_tools import *  # type: ignore[F403]

__all__ = list(getattr(_core, "__all__", ()))

_CLI_EXPORTS = (
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
    "menu",
    "run_filters_flow",
    "run_rebuild_flow",
    "run_spectra_flow",
    "run_tool",
    "ensure_deps",
    "TOOL_HELP",
    "TOOLS",
)


def __getattr__(name: str) -> Any:
    if name in _CLI_EXPORTS:
        cli = import_module("SED_tools.cli")
        value = getattr(cli, name)
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)
        return value
    raise AttributeError(f"module 'SED_tools' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
