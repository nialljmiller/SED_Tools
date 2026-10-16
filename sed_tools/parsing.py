"""Shared parsers for command-line and programmatic inputs."""

from __future__ import annotations

import re
from typing import Optional, Tuple


def parse_numeric_range(raw: str) -> Optional[Tuple[float, float]]:
    """Parse two numbers separated by comma, whitespace, colon, ``..`` or dash."""
    text = raw.strip()
    if not text:
        return None
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:,|:|\.\.|\s+|-)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*",
        text,
    )
    if match is None:
        return None
    low, high = (float(value) for value in match.groups())
    return (low, high) if low <= high else (high, low)


def parse_multi_selection(spec: str, total: int) -> list[int]:
    """Parse 1-based IDs such as ``1,3-5`` into unique 0-based indexes."""
    if total < 0:
        raise ValueError("total must be non-negative")
    chosen: set[int] = set()
    for chunk in (part.strip() for part in spec.split(",")):
        if not chunk:
            continue
        if "-" in chunk:
            bounds = [part.strip() for part in chunk.split("-", 1)]
            if len(bounds) != 2 or not all(part.isdigit() for part in bounds):
                raise ValueError(f"Invalid range: {chunk}")
            start, end = map(int, bounds)
            if start > end:
                start, end = end, start
            if start < 1 or end > total:
                raise ValueError(f"Range out of bounds: {chunk}")
            chosen.update(range(start - 1, end))
        else:
            if not chunk.isdigit():
                raise ValueError(f"Invalid item: {chunk}")
            value = int(chunk)
            if value < 1 or value > total:
                raise ValueError(f"Selection out of bounds: {value}")
            chosen.add(value - 1)
    return sorted(chosen)
