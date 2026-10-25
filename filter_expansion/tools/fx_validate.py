#!/usr/bin/env python3
"""Validate staged filter instruments against the filter-expansion format contract.

Usage: fx_validate.py [STAGE_ROOT] [--mirror-index mirror_index.json] [--svo] [--json OUT]

Checks per instrument (stage/filters/<Facility>/<Instrument>/):
  * every .dat (published and components) is read correctly by released MESA;
  * the column row carries FilterID, DetectorType, DetectorTypeSource,
    WavelengthUnit=Angstrom, Components, ComponentsSource (components: Role);
  * wavelengths strictly increasing, transmission >= 0, >= 2 samples, finite;
  * filenames have exactly one dot; band stems unique ignoring case;
  * the index file lists exactly the published .dat files;
  * manifest.json loads as InstrumentManifest, its bands match, no missing files;
  * the instrument is not already on the mirror (index.json);
  * pivot matches a published value (provenance ref_pivot) within 1.5 %;
  * with --svo, pivot wavelength matches the SVO counterpart within 1.5 %.
Exit status 1 if any error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fxlib import ROOT, STAGE, pivot_wavelength, svo_curve  # noqa: E402
from sed_tools.filter_components import InstrumentManifest  # noqa: E402
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads  # noqa: E402

REQUIRED = ("FilterID", "DetectorType", "DetectorTypeSource", "WavelengthUnit", "Components", "ComponentsSource")


def read_samples(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    return lines[0], data[:, 0], data[:, 1]


def check_curve(path, errors, *, component=False):
    if path.name.count(".") != 1:
        errors.append(f"{path}: filename must contain exactly one dot")
    if not released_mesa_reads(path):
        errors.append(f"{path}: released MESA misreads this file")
    try:
        first, wl, tr = read_samples(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path}: unreadable samples ({exc})")
        return None
    if not first.startswith("Wavelength,Transmission,"):
        errors.append(f"{path}: row 1 is not the Wavelength,Transmission metadata row")
    hdr = read_filter_header(path)
    need = ("Role", "Components", "WavelengthUnit") if component else REQUIRED
    for key in need:
        if not hdr.get(key):
            errors.append(f"{path}: header lacks {key}")
    if hdr.get("WavelengthUnit") != "Angstrom":
        errors.append(f"{path}: WavelengthUnit is {hdr.get('WavelengthUnit')!r}")
    if not component and hdr.get("DetectorType") not in ("photon", "energy"):
        errors.append(f"{path}: DetectorType {hdr.get('DetectorType')!r}")
    if len(wl) < 2 or not np.all(np.isfinite(wl)) or not np.all(np.isfinite(tr)):
        errors.append(f"{path}: fewer than 2 samples or non-finite values")
    elif not np.all(np.diff(wl) > 0):
        errors.append(f"{path}: wavelengths not strictly increasing")
    if np.any(tr < 0):
        errors.append(f"{path}: negative transmission")
    if not np.any(tr > 0):
        errors.append(f"{path}: all-zero curve")
    if wl.min() < 50 or wl.max() > 2e8:
        errors.append(f"{path}: wavelength range {wl.min():g}-{wl.max():g} Å implausible (unit?)")
    return hdr, wl, tr


def validate_instrument(inst, mirror, use_svo, report):
    errors, warnings = [], []
    facility, instrument = inst.parent.name, inst.name
    published = sorted(p for p in inst.glob("*.dat"))
    stems = [p.stem for p in published]
    if len({s.lower() for s in stems}) != len(stems):
        errors.append(f"{inst}: band names collide ignoring case")
    index = inst / instrument
    if not index.is_file():
        errors.append(f"{inst}: missing index file {instrument}")
    else:
        listed = [ln.strip() for ln in index.read_text().splitlines() if ln.strip()]
        if sorted(listed) != [p.name for p in published]:
            errors.append(f"{inst}: index file does not list exactly the published .dat files")
    try:
        man = InstrumentManifest.load(inst)
        if not (inst / "manifest.json").is_file():
            errors.append(f"{inst}: no manifest.json")
        if sorted(man.bands) != sorted(stems):
            errors.append(f"{inst}: manifest bands {man.bands} != files {stems}")
        missing = man.missing_files(inst)
        if missing:
            errors.append(f"{inst}: manifest lists missing files {missing}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{inst}: manifest does not load ({exc})")
    fac = mirror.get(facility, {})
    if instrument in fac.get("instruments", {}):
        errors.append(f"{facility}/{instrument} already exists on the mirror")
    prov_path = inst.parents[2] / "provenance" / f"{facility}__{instrument}.json"
    prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    if not prov:
        warnings.append(f"{inst}: no provenance record")
    bands = {}
    for p in published:
        res = check_curve(p, errors)
        if res is None:
            continue
        hdr, wl, tr = res
        piv = pivot_wavelength(wl, tr, hdr.get("DetectorType", "photon"))
        row = {"pivot_A": round(piv, 2), "n": len(wl), "detector": hdr.get("DetectorType"),
               "components": hdr.get("Components"), "source": hdr.get("ComponentsSource")}
        pb = prov.get("bands", {}).get(p.stem, {})
        if pb.get("ref_pivot"):
            ref, cite = pb["ref_pivot"]
            diff = abs(piv - ref) / ref
            row["ref_diff_pct"] = round(100 * diff, 3)
            (errors if diff > 0.015 else warnings if diff > 0.005 else []).append(
                f"{p}: pivot {piv:.1f} Å vs published {ref:.1f} Å ({cite}; {100 * diff:.2f} %)")
        svo_id = pb.get("svo_id")
        if use_svo and svo_id:
            try:
                swl, str_, params = svo_curve(svo_id, "_svo_check")
                ref = float(params.get("WavelengthPivot") or pivot_wavelength(swl, str_, hdr.get("DetectorType")))
                row["svo_pivot_A"] = round(ref, 2)
                diff = abs(piv - ref) / ref
                row["svo_diff_pct"] = round(100 * diff, 3)
                (errors if diff > 0.015 else warnings if diff > 0.005 else []).append(
                    f"{p}: pivot {piv:.1f} Å vs SVO {svo_id} {ref:.1f} Å ({100 * diff:.2f} %)")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{p}: SVO check for {svo_id} failed ({exc.__class__.__name__}: {exc})")
        bands[p.stem] = row
    for p in sorted((inst / "components").rglob("*.dat")) if (inst / "components").is_dir() else []:
        check_curve(p, errors, component=True)
    report[f"{facility}/{instrument}"] = {"bands": bands, "errors": errors, "warnings": warnings}
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=STAGE)
    ap.add_argument("--mirror-index", type=Path, default=ROOT / "mirror_index.json")
    ap.add_argument("--svo", action="store_true", help="compare pivot wavelengths with SVO")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--only", nargs="*", default=None, help="Facility/Instrument to check")
    args = ap.parse_args(argv)
    mirror = json.loads(args.mirror_index.read_text())["filters"]["facilities_detailed"]
    report, n_err = {}, 0
    for inst in sorted(p for p in (args.root / "filters").glob("*/*") if p.is_dir()):
        key = f"{inst.parent.name}/{inst.name}"
        if args.only and key not in args.only:
            continue
        errors, warnings = validate_instrument(inst, mirror, args.svo, report)
        n_err += len(errors)
        status = "FAIL" if errors else "ok"
        print(f"{status:4} {key}: {len(report[key]['bands'])} bands, {len(warnings)} warnings")
        for line in errors + [f"warn: {w}" for w in warnings]:
            print(f"     {line}")
    for key in args.only or []:
        if key not in report:
            print(f"FAIL {key}: not staged")
            n_err += 1
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report)} instruments, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
