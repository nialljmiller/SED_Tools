"""Write a stellar grid's manifest.json, properties.json and info.json from the guard: the publish step (design §4).

For each grid it first builds the flux cubes with SED_Tools' cube builder when they need it (build_cubes): none yet,
one lacking a spectrum beside it, or a layout other than SED_Tools' split rule gives (sed_tools.cube.collision.
discover_extra_axes: a grid whose extra axes tell apart spectra at one node is split into one variant folder per
value, each with its cube, lookup table and links to the spectra, and the grid's top-level flux_cube.bin is then
the all-axes cube of every variant). A split grid whose variant cubes need nothing but whose top-level flux_cube.bin
is not its all-axes cube (a 3-axis copy of a variant, or none) gets only that cube, assembled from the variant cubes
(sed_tools.cube.variants.write_all_axes_cube). Cube files it replaces are first copied to
backups/<UTC>-grid-publish/.
Then it runs grid_info.check_grids on data/stellar_models/<grid>/ and writes beside the spectra:
  manifest.json    the files clients download (and the variant cubes), with size, modification time and SHA-256,
                   plus the SHA-256 of each spectrum's data lines; axes, variants, cube layout, the archived
                   copies and aliases from the curated record, and the guard's verdict. Its status is
                   "published" only when every check passed, else "held".
  properties.json  values measured from the data: axes, filled and empty nodes, wavelength range and sampling,
                   flux unit, and every check's result
  info.json        grid_info/curated/<grid>.json: description, sources, terms quoted verbatim, references, credit
These are the grid counterparts of a filter instrument's manifest.json, properties.json and info.json. A record is
rewritten only when it would change other than in its run times (generated, computed, checked, previous), and only then is the
record there first copied to backups/<UTC>-grid-publish/data/stellar_models/<grid>/, outside the served tree.
generate_index.py lists a grid only if its manifest is "published" and its files still match it.

Usage (on NJM, from the server checkout, under the NJM lock):
  flock /tmp/sed-filter-info.lock python3 -m grid_info.publish GRID [GRID ...] [--data data] [--dry-run]
Exit status 1 if any grid is held. --dry-run says which grids' cubes it would build and builds none.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from grid_info.check_grids import (CURATED, DEFAULT_CACHE, ROOT, SCHEMA, HashCache, _jsonable, all_axes_cube_problem,
                                   check_grid, other_hashes, previous_manifest, read_lookup, served_paths,
                                   variant_links, variant_paths)

# Lookup columns that are bookkeeping, not axes (as SED_Tools' cube/collision.py lists them).
BOOKKEEPING = {"file_name", "source", "units_standardized", "wavelength_unit", "flux_unit", "original_wavelength_unit",
               "original_flux_unit", "conversion_confidence", "spec_group", "url", "wave_grid", "columns", "res",
               "atmos", "renormed"}
MAX_HOLES = 50000
CUBE_ORDER = "int32 nt, nl, nm, nw; float64 axes teff, logg, meta, wavelength; float64 flux[nw][nm][nl][nt] (Teff fastest)"

DEFINITIONS = {
    "spectra": "number of .txt spectra in the grid folder",
    "axes": "the distinct values of each parameter in the spectrum headers (teff in K, logg in log10 cm/s2, "
            "metallicity as the grid states it)",
    "extra_axes": "other parameters that vary between spectra; each combination is one variant with its own cube",
    "cubes": "per flux_cube.bin: dimensions, nodes holding a spectrum, empty nodes (holes) as [teff, logg, meta]",
    "wavelength": "over the sampled spectra: first and last wavelength (Angstrom), points per spectrum, and the "
                  "median resolving power lambda/delta-lambda of the sampling",
    "bolometric": "integral of F_lambda over each sampled spectrum (or cube node) divided by that of "
                  "pi*B_lambda(Teff) over the same wavelengths: 1 for a surface flux, 1/pi for B_lambda, 1/(4 pi) "
                  "for an Eddington flux H_lambda",
    "shape": "sampled spectra divided by the raw upstream files they came from, on common wavelengths",
    "guard": "the result of each check of grid_info/check_grids.py (design section 5)",
}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def axis_values(headers, rows, columns):
    axes, extra = {}, {}
    for p in ("teff", "logg", "metallicity"):
        v = sorted({h[p] for h in headers.values() if isinstance(h[p], float) and math.isfinite(h[p])})
        if v:
            axes[p] = {"n": len(v), "range": [v[0], v[-1]], "values": v}
    for col in columns or []:
        key = col.strip().lower()
        if key in BOOKKEEPING or key in ("teff", "logg", "metallicity", "meta", "feh", "z"):
            continue
        vals = sorted({r.get(col, "") for r in rows.values()})
        extra[col] = vals
    return axes, extra


def wavelength_facts(read):
    if not read:
        return None
    starts = [float(wl[0]) for _, wl, _ in read]
    ends = [float(wl[-1]) for _, wl, _ in read]
    npts = [len(wl) for _, wl, _ in read]
    res = [float(np.median(wl[1:] / np.diff(wl))) for _, wl, _ in read if len(wl) > 2]
    return {"min_angstrom": min(starts), "max_angstrom": max(ends), "first_angstrom_range": [min(starts), max(starts)],
            "last_angstrom_range": [min(ends), max(ends)], "points": [min(npts), max(npts)],
            "median_resolving_power": round(float(np.median(res)), 1) if res else None, "sampled_spectra": len(read)}


def cube_facts(grid, scans):
    out = {}
    for path, f in scans:
        if "filled" not in f:
            continue
        rel = str(path.relative_to(grid))
        empty = np.argwhere(~f["filled"])
        holes = [[float(f["teff"][it]), float(f["logg"][il]), float(f["meta"][im])] for im, il, it in empty]
        out[rel] = {"dims": {"nt": f["nt"], "nl": f["nl"], "nm": f["nm"], "nw": f["nw"]},
                    "header_bytes": f["offset"], "dtype": "<f8", "order": CUBE_ORDER,
                    "teff": [float(x) for x in f["teff"]], "logg": [float(x) for x in f["logg"]],
                    "meta": [float(x) for x in f["meta"]],
                    "wavelength_angstrom": [float(f["wavelengths"][0]), float(f["wavelengths"][-1])],
                    "nodes": int(f["filled"].size), "filled": int(f["filled"].sum()), "empty": len(holes),
                    "fill_fraction": round(float(f["filled"].mean()), 4),
                    "holes": holes if len(holes) <= MAX_HOLES else f"{len(holes)} (too many to list)"}
    return out


def file_records(grid, cache):
    """{relative path: size, mtime, sha256 (data_sha256 for spectra), role} for what clients and MESA read."""
    roles = {".txt": "spectrum", ".csv": "lookup", ".h5": "h5", ".bin": "cube"}
    out = {}
    for rel, p in served_paths(grid).items():
        hit = cache.get(p, p.suffix.lower() == ".txt")
        rec = {"size": hit["size"], "mtime": hit["mtime"], "ctime_ns": hit["stamp"][2], "sha256": hit["sha256"],
               "role": roles[p.suffix.lower()]}
        if "data_sha256" in hit:
            rec["data_sha256"] = hit["data_sha256"]
        if p.name == "variants_index.csv":
            rec["role"] = "variants"
        elif rel == "flux_cube.bin" and (grid / "variants_index.csv").is_file():
            rec["role"] = "cube (all axes: every variant; MESA reads a variant's)"
        elif p.name == "lookup_table_full.csv":
            rec["role"] = "lookup (whole grid, all variants)"
        out[rel] = rec
    return out


def build_records(grid, report, aux, curated, cache):
    headers, rows, cols = aux["headers"], aux["rows"], aux["columns"]
    axes, extra = axis_values(headers, rows, cols)
    files = file_records(grid, cache)
    spectra = {k: v for k, v in files.items() if v["role"] == "spectrum"}
    variants = []
    if (grid / "variants_index.csv").is_file():
        vcols, vrows = read_lookup(grid / "variants_index.csv")
        variants = vrows
    guard = {"passed": report["passed"], "checked": report["checked"], "tool": report["tool"],
             "errors": report["errors"], "warnings": report["warnings"],
             "terms_sha256": report["checks"]["7_provenance"].get("terms_sha256")}
    cubes = cube_facts(grid, aux["scans"])
    manifest = {
        "schema": SCHEMA, "grid": grid.name, "status": "published" if report["passed"] else "held",
        "generated": utc(), "description": curated.get("short_description"),
        "upstream": curated.get("upstream"),
        "spectra": {"count": len(spectra), "bytes": sum(v["size"] for v in spectra.values())},
        "axes": {k: {"n": v["n"], "range": v["range"]} for k, v in axes.items()},
        "extra_axes": extra, "variants": variants,
        "cubes": {k: {x: v[x] for x in ("dims", "header_bytes", "dtype", "order")} for k, v in cubes.items()},
        "default_variant": curated.get("default_variant"),
        "archived_copies": curated.get("archived_copies", {}), "aliases": curated.get("aliases", {}),
        "guard": guard, "variant_links": variant_links(grid), "files": files,
    }
    checks = {k: {x: v for x, v in c.items()} for k, c in report["checks"].items()}
    properties = {
        "schema": SCHEMA, "grid": grid.name, "computed": utc(), "definitions": DEFINITIONS,
        "spectra": len(spectra), "axes": axes, "extra_axes": extra, "cubes": cubes,
        "wavelength": wavelength_facts(aux["read"]),
        "wavelength_medium": curated.get("wavelength_medium", "not stated"),
        "flux": {"unit": "erg/s/cm2/Angstrom", "quantity": curated.get("flux_quantity",
                                                                       "surface flux F_lambda")},
        "completeness": curated.get("completeness"),
        "bolometric": report["checks"]["5_physics"].get("bolometric"),
        "shape": report["checks"]["5_physics"].get("shape"),
        "guard": {"passed": report["passed"], "checked": report["checked"], "checks": checks},
    }
    info = {"schema": SCHEMA, "grid": grid.name, **{k: v for k, v in curated.items()
                                                     if k not in ("archived_copies", "aliases", "upstream")}}
    return manifest, properties, info


VOLATILE = {"generated", "computed", "checked", "previous"}   # run times; records differing only in these are unchanged


def _stable(doc):
    """A record as JSON text without its run times, for comparing a new record with the one on disk."""
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in VOLATILE}
        if isinstance(d, list):
            return [strip(v) for v in d]
        return d
    return json.dumps(strip(_jsonable(doc)), sort_keys=True)


def write_json(path, doc):
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(_jsonable(doc), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _split_by_rule(grid):
    """Whether SED_Tools' cube builder splits this grid into variants: its extra axes (the generic rule of
    sed_tools.cube.collision.discover_extra_axes) give more than one combination of values."""
    from sed_tools.cube.build import _parse_axes, _read_lookup_table
    from sed_tools.cube.collision import discover_extra_axes
    from sed_tools.cube.variants import _group_by_extra_axes
    from sed_tools.grids.lookup import find_file_column

    path = grid / "lookup_table.csv"
    lookup, cols, _ = _read_lookup_table(str(path))
    file_col = find_file_column(cols) or cols[0]
    teff, logg, meta = _parse_axes(lookup, str(path), file_col)
    extra = discover_extra_axes({c: lookup[c] for c in lookup if c != file_col}, list(zip(teff, logg, meta)))
    return len(_group_by_extra_axes(lookup, extra, file_col, teff, logg, meta)) > 1


def cube_build_needed(grid):
    """Why the grid's flux cubes must be built with SED_Tools' builder, or None (see the module docstring)."""
    from sed_tools.cube.check import _flux_cube_file, _flux_cube_shortfall

    if not (grid / "lookup_table.csv").is_file() or not any(grid.glob("*.txt")):
        return None   # check 3 reports it; nothing to build from
    vindex = grid / "variants_index.csv"
    cube = _flux_cube_file(grid)
    if cube is None:
        return "no flux cube"
    shortfall = _flux_cube_shortfall(grid)
    if shortfall:
        return f"{cube.name} {shortfall}"
    split = _split_by_rule(grid)
    if split != vindex.is_file():
        return ("SED_Tools' rule splits it into variants, and it has one cube" if split
                else "it is split into variants, and SED_Tools' rule gives one cube")
    return None


def _cube_files(grid):
    """The cube files a build replaces or removes: the top-level cube and variants index, and each listed variant's
    cube and lookup table."""
    paths = [grid / n for n in ("flux_cube.bin", "variants_index.csv", "lookup_table_full.csv")]
    vindex = grid / "variants_index.csv"
    if vindex.is_file():
        for rel in variant_paths(vindex):
            paths += [grid / rel / "flux_cube.bin", grid / rel / "lookup_table.csv"]
    return [p for p in paths if p.is_file()]


def build_cubes(grid, stamp, dry_run=False):
    """Build the grid's cubes with SED_Tools' builder when cube_build_needed says why, or else only a split grid's
    all-axes cube when all_axes_cube_problem says why; returns that reason, or None. RuntimeError from the builder
    (more variants than it splits without asking) is returned as the reason with "not built: "; the check then holds
    the grid."""
    why = cube_build_needed(grid)
    top_only = why is None and (grid / "variants_index.csv").is_file() and all_axes_cube_problem(grid)
    if not (why or top_only) or dry_run:
        return why or top_only or None
    from sed_tools.cube.variants import write_all_axes_cube
    from sed_tools.grids.pipeline import _build_flux_cube

    backup = ROOT / "backups" / f"{stamp}-grid-publish" / grid.relative_to(ROOT)
    for path in _cube_files(grid):
        target = backup / path.relative_to(grid)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if top_only:
        write_all_axes_cube(str(grid))
        return top_only
    try:
        _build_flux_cube(grid)  # a split writes the all-axes cube too
    except RuntimeError as exc:
        return f"not built: {exc}"
    return why


def publish(grid, cache, stamp, dry_run=False):
    curated_path = CURATED / f"{grid.name}.json"
    curated = json.loads(curated_path.read_text()) if curated_path.is_file() else {}
    built = build_cubes(grid, stamp, dry_run)
    if built:
        print(f"{'would build' if dry_run else 'built'} the cubes of {grid.name}: {built}")
    report, aux = check_grid(grid, other_hashes(grid.parent, grid.name, cache), cache, previous_manifest(grid))
    cache.save()
    records = build_records(grid, report, aux, curated, cache)
    cache.save()
    changed = []
    for name, doc in zip(("manifest.json", "properties.json", "info.json"), records):
        try:
            old = _stable(json.loads((grid / name).read_text()))
        except (OSError, ValueError):
            old = None
        if old != _stable(doc):
            changed.append((name, doc))
    if dry_run:
        return report, records, [name for name, _ in changed]
    backup = ROOT / "backups" / f"{stamp}-grid-publish" / grid.relative_to(ROOT)
    for name, _ in changed:
        if (grid / name).is_file():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(grid / name, backup / name)
    for name, doc in changed:
        write_json(grid / name, doc)
    return report, records, [name for name, _ in changed]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grids", nargs="+")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dry-run", action="store_true", help="check and build the records, write nothing")
    ap.add_argument("--json", type=Path, help="write the guard reports here")
    args = ap.parse_args(argv)
    cache = HashCache(args.cache)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    held, reports = 0, {}
    for name in args.grids:
        grid = (args.data / "stellar_models" / name).resolve()
        report, _, changed = publish(grid, cache, stamp, args.dry_run)
        reports[name] = report
        held += not report["passed"]
        what = (f"would rewrite {', '.join(changed) or 'nothing'} (dry run: nothing written)" if args.dry_run
                else f"rewrote {', '.join(changed)}" if changed else "records unchanged, nothing written")
        print(f"{'published' if report['passed'] else 'held'} {name}: {len(report['errors'])} errors, "
              f"{len(report['warnings'])} warnings; {what}")
        for line in report["errors"] + [f"warn: {w}" for w in report["warnings"]]:
            print(f"     {line}")
    if args.json:
        args.json.write_text(json.dumps(_jsonable(reports), indent=1) + "\n")
    return 1 if held else 0


if __name__ == "__main__":
    sys.exit(main())
