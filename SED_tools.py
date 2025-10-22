#!/usr/bin/env python3
"""Compatibility entry point delegating to :mod:`SED_tools.cli`."""

from SED_tools.cli import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
