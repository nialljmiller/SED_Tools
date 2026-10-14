#!/usr/bin/env python3
"""Update release metadata together: python3 scripts/bump_version.py minor.

Accepts major, minor, patch, or an explicit X.Y.Z version. Use --check to
verify consistency, optionally with --tag vX.Y.Z before publishing.
Requires Python 3.11 or newer; run from any directory.
"""

import argparse
import re
import tomllib
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, pattern, replacement):
    updated, count = re.subn(pattern, lambda match: replacement, text, flags=re.M)
    if count != 1:
        raise ValueError(f"Expected one match for {pattern!r}, found {count}")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bump", nargs="?", help="major, minor, patch, or X.Y.Z")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tag", help="require a matching release tag (with --check)")
    args = parser.parse_args()
    if args.check == bool(args.bump) or (args.tag and not args.check):
        parser.error("choose a version bump or --check; --tag requires --check")

    paths = ["pyproject.toml", "sed_tools/__init__.py", "CITATION.cff",
             "README.md", "docs/source/changelog.rst"]
    contents = {name: (ROOT / name).read_text() for name in paths}
    current = tomllib.loads(contents["pyproject.toml"])["project"]["version"]
    for name, pattern in [
        ("sed_tools/__init__.py", r'^__version__ = "([^"]+)"$'),
        ("CITATION.cff", r'^version: "([^"]+)"$'),
    ]:
        match = re.search(pattern, contents[name], re.M)
        if not match or match[1] != current:
            parser.error(f"{name} does not match package version {current}")
    if args.check:
        if args.tag and args.tag != f"v{current}":
            parser.error(f"tag {args.tag!r} does not match v{current}")
        print(f"Release versions agree: {current}")
        return

    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", current):
        parser.error(f"unsupported current version: {current}")
    parts = list(map(int, current.split(".")))
    if args.bump in ("major", "minor", "patch"):
        index = ("major", "minor", "patch").index(args.bump)
        parts[index] += 1
        parts[index + 1:] = [0] * (2 - index)
        target = ".".join(map(str, parts))
    else:
        target = args.bump
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", target):
        parser.error("version must be X.Y.Z with nonnegative integer components")
    if tuple(map(int, target.split("."))) <= tuple(map(int, current.split("."))):
        parser.error("new version must be greater than the current version")

    today = date.today()
    try:
        contents["pyproject.toml"] = replace_once(
            contents["pyproject.toml"], rf'^version = "{re.escape(current)}"$',
            f'version = "{target}"')
        contents["sed_tools/__init__.py"] = replace_once(
            contents["sed_tools/__init__.py"], r'^__version__ = "[^"]+"$',
            f'__version__ = "{target}"')
        contents["CITATION.cff"] = replace_once(
            contents["CITATION.cff"], r'^version: "[^"]+"$', f'version: "{target}"')
        contents["CITATION.cff"] = replace_once(
            contents["CITATION.cff"], r'^date-released: "[^"]+"$',
            f'date-released: "{today.isoformat()}"')
        contents["README.md"] = replace_once(
            contents["README.md"],
            r'^A version-specific Zenodo DOI for Version .* will be added following archival of this release\.$',
            f'A version-specific Zenodo DOI for Version {target} will be added following archival of this release.')
        heading = f"Version {target} ({today.strftime('%B %Y')})"
        changelog = contents["docs/source/changelog.rst"]
        if re.search(r'^Unreleased\n-+\n', changelog, re.M):
            changelog = replace_once(changelog, r'^Unreleased\n-+\n', heading + '\n' + '-' * len(heading) + '\n')
        else:
            marker = "Changelog\n=========\n\n"
            if changelog.count(marker) != 1:
                raise ValueError("Cannot locate changelog insertion point")
            changelog = changelog.replace(marker, marker + heading + '\n' + '-' * len(heading) + '\n\n', 1)
        contents["docs/source/changelog.rst"] = changelog
    except ValueError as exc:
        parser.error(str(exc))

    # Validate every replacement before writing any file. Archived citations
    # and older changelog entries deliberately retain their original versions.
    for name, content in contents.items():
        (ROOT / name).write_text(content)
    print(f"Updated {current} -> {target} in {len(contents)} files")


if __name__ == "__main__":
    main()
