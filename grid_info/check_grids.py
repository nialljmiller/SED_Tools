"""Check a stellar grid before NJM lists it: the grid counterpart of check_manifests.py and fx_validate.py.

A grid (data/stellar_models/<grid>/) is listed only if all eight checks pass (design §5):
  1 upload      no rsync temp (dot) files; folders 755, files 644; no dangling symlinks
  2 spectra     every .txt declares teff, wavelength_unit, flux_unit = erg/cm2/s/A and units_standardized;
                sampled data are finite, strictly increasing in wavelength, two columns
  3 lookup      lookup_table.csv lists exactly the .txt present, and each row equals its spectrum's header
                re-read (no 999.9 placeholders, no lost minus signs, no missing columns)
  4 cube        each flux_cube.bin (or variant cube in variants_index.csv) has a header that agrees with its
                axes and size, is filled at exactly its lookup's nodes, and holds its spectra; the .h5 bundle
                holds the same spectra. A grid without a cube passes with a warning: clients build one
  5 physics     (a) integral of F_lambda over each sampled spectrum and every filled cube node is within
                0.5-2 of that of pi*B_lambda(Teff) on the same wavelengths (warning outside 0.9-1.1);
                (b) shape: sampled spectra equal the raw upstream files in source_evidence/grids/<grid>/raw/,
                or, without raw files, every header declares that no unit conversion was made
  6 duplicate   no spectrum's data (the lines after the header) and no cube occurs in another listed grid
  7 provenance  info.json (or grid_info/curated/<grid>.json) gives a description, sources, references and
                the terms, quoted verbatim and found in the archived evidence file, or "none found"
  8 shrinkage   no fewer spectra and no narrower axes than the grid's previous manifest.json

manifest_matches() is what generate_index.py asks before listing a grid: a manifest.json whose guard passed,
and a folder whose top-level .txt/.csv/.h5/.bin files (and variant cubes) are exactly those it records, by
name, size and modification time, so files changed after the check (a re-upload, say) delist the grid.

Usage: python3 -m grid_info.check_grids [DATA_DIR] [--only GRID ...] [--json OUT] [--cache FILE]
Prints one line per grid (ok/FAIL, then its errors and warnings) and exits 1 if any grid fails, else 0.
Reads only; never writes in DATA_DIR. Needs numpy; h5py for the .h5 check (a warning without it).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "grid_info" / "curated"
EVIDENCE = ROOT / "source_evidence" / "grids"
DEFAULT_CACHE = ROOT / "grid_info" / ".cache" / "hashes.json"
SCHEMA = 1

SIGMA = 5.670374419e-5                   # erg/s/cm2/K4
H, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
FLUX_UNIT = "erg/cm2/s/a"                # as SED_Tools' cleaner writes it, compared lower-case
SERVED_SUFFIXES = (".txt", ".csv", ".h5", ".bin")   # what clients download from a grid folder
BOLO_FAIL, BOLO_WARN = (0.5, 2.0), (0.9, 1.1)
NODE_TOL = 0.01                          # a node's integral against its own spectrum's, resampled onto the cube
FAULT_RATIOS = {"1/pi": 1 / math.pi, "1/(4 pi)": 1 / (4 * math.pi)}   # an exception range may not admit these
# 5c: the infrared slope d ln F_lambda / d ln lambda, fitted over 3-25 times the Wien peak wavelength of the spectrum's
# Teff (spectra whose wavelengths reach 20 times it). Atmospheres approach the Rayleigh-Jeans -4 there; a lambda^-2
# tilt (the BOSZ fault) gives about -6. SLOPE_RANGE is set from the real grids (SLOPE_NOTE).
SLOPE_WINDOW = (3.0, 25.0)
SLOPE_RANGE = (-4.6, -1.5)
SLOPE_NOTE = ("measured 30 Oct 2026 on 11,026 spectra that reach 20 times their Wien peak: every one of Kurucz2003all (6,409), "
              "NextGen2 (1,953), Husfeld (448) and bbody (583), and samples of Kurucz (633), bt-cond (400), raw SVO TLUSTY (300) "
              "and the 4 pi H BOSZ rebuild (300) give -3.95 to -2.10 (the flattest are NextGen2's coolest models); the BOSZ copy "
              "served tilted by lambda^-2 gives -5.84 to -5.67, and any of those spectra tilted by lambda^-2 at most -5.31. The "
              "upper bound is loose, because cooler grids (morley12, bt-settl) are not yet measured: it catches a lambda^+2 tilt "
              "only of spectra steeper than -3.5")
COVERED = 0.5                            # pi*B over the file's wavelengths / sigma T^4 needed to judge 5a
SAMPLE_ALL_BELOW = 1.5e9                 # bytes of .txt below which every spectrum's data is read
SAMPLE_SIZE = 120

# Header keys and the canonical name each is read as: SED_Tools' grids/header_parser.ALIASES.
ALIASES = {**dict.fromkeys(["teff", "t_eff", "t eff", "t-eff", "effective_temperature", "temperature", "temp"],
                           "teff"),
           **dict.fromkeys(["logg", "log_g", "log(g)", "log g", "surface_gravity", "gravity"], "logg"),
           **dict.fromkeys(["metallicity", "meta", "feh", "[fe/h]", "[m/h]", "m/h", "m_h", "mh", "z", "zh", "zmet",
                            "z/z0", "z/zsun"], "metallicity")}
PARAMS = ("teff", "logg", "metallicity")
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ---- reading ----

def read_header(path):
    """The '# key = value' header of a spectrum as SED_Tools' parse_header reads it, and its line count."""
    out = {p: math.nan for p in PARAMS}
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                break
            n += 1
            if "=" not in s:
                continue
            key, _, val = s.lstrip("#").partition("=")
            key, val = key.strip().lower(), val.split("(")[0].strip()
            key = ALIASES.get(key, key)
            if key in PARAMS:
                m = NUM.search(val)
                v = float(m.group(0)) if m else math.nan
                if not math.isnan(v) or math.isnan(out[key]):
                    out[key] = v
            else:
                out[key] = val
    return out, n


def read_data(path):
    """Wavelength and flux (the first two columns) of a text spectrum; rows without two numbers are skipped."""
    try:
        a = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2)
    except ValueError:  # a short or garbled row (e.g. a truncated raw download): keep the rows that parse
        rows = []
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2 and not line.lstrip().startswith("#"):
                    try:
                        rows.append((float(f[0]), float(f[1])))
                    except ValueError:
                        pass
        a = np.array(rows, dtype=float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def read_lookup(path):
    """(columns, rows as dicts) of a lookup table whose header row may start with '#'."""
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        cols = [c.strip() for c in fh.readline().lstrip("#").strip().split(",")]
        rows = [dict(zip(cols, (v.strip() for v in r))) for r in csv.reader(fh) if r and any(x.strip() for x in r)]
    return cols, rows


def file_column(cols):
    low = {c.lower(): c for c in cols}
    return next((low[c] for c in ("file_name", "filename", "file") if c in low), None)


def param_column(cols, canonical):
    low = {c.lower(): c for c in cols}
    return next((low[k] for k, v in ALIASES.items() if v == canonical and k in low), None)


def cube_header(path):
    """(nt, nl, nm, nw, teff, logg, meta, wavelengths, data offset) of a MESA flux_cube.bin."""
    with open(path, "rb") as fh:
        nt, nl, nm, nw = struct.unpack("4i", fh.read(16))
        if not all(0 < n < 10 ** 8 for n in (nt, nl, nm, nw)):
            raise ValueError(f"implausible dimensions {nt} x {nl} x {nm} x {nw}")
        ax = np.fromfile(fh, dtype="<f8", count=nt + nl + nm + nw)
    return (nt, nl, nm, nw, ax[:nt], ax[nt:nt + nl], ax[nt + nl:nt + nl + nm], ax[nt + nl + nm:],
            16 + 8 * (nt + nl + nm + nw))


def pi_b(lam, teff):
    """pi B_lambda(T) in erg/s/cm2/A at wavelengths in A."""
    x = np.minimum(H * C / (lam * 1e-8 * K * teff), 700.0)
    return math.pi * 2 * H * C ** 2 / (lam * 1e-8) ** 5 / np.expm1(x) * 1e-8


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0


def resample(wl, fl, target):
    """SED_Tools' cube rule (cube/build.resample_to_grid): log-linear inside the spectrum's range, 0 outside."""
    out = np.zeros(len(target))
    inside = (target >= wl[0]) & (target <= wl[-1])
    out[inside] = 10.0 ** np.interp(target[inside], wl, np.log10(np.maximum(fl, 1e-300)))
    return out


def data_hash(path):
    """SHA-256 of a spectrum's data: the bytes after its leading '#' lines (copies with other headers match)."""
    blob = Path(path).read_bytes()
    i = 0
    while blob[i:i + 1] == b"#":
        j = blob.find(b"\n", i)
        i = len(blob) if j < 0 else j + 1
    return hashlib.sha256(blob).hexdigest(), hashlib.sha256(blob[i:]).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HashCache:
    """SHA-256 of files keyed by path, size, mtime, ctime and inode, kept outside the served tree: a file whose
    content changed with its size and mtime restored still has a new ctime, so it is hashed again."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}
        self.dirty = False

    def get(self, path, spectrum):
        st = os.stat(path)
        key = str(Path(path).resolve())
        stamp = [st.st_size, int(st.st_mtime), st.st_ctime_ns, st.st_ino]
        hit = self.data.get(key)
        if hit and hit.get("stamp") == stamp and (not spectrum or "data_sha256" in hit):
            return hit
        if spectrum:
            whole, data = data_hash(path)
            hit = {"size": st.st_size, "mtime": int(st.st_mtime), "stamp": stamp, "sha256": whole, "data_sha256": data}
        else:
            hit = {"size": st.st_size, "mtime": int(st.st_mtime), "stamp": stamp, "sha256": file_hash(path)}
        self.data[key] = hit
        self.dirty = True
        return hit

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".part")
            tmp.write_text(json.dumps(self.data))
            os.replace(tmp, self.path)


# ---- the checks; each returns {"ok", "errors", "warnings", ...} ----

def _result(errors, warnings, **facts):
    return {"ok": not errors, "errors": errors, "warnings": warnings, **facts}


def _few(names, n=3):
    names = sorted(names)
    return ", ".join(names[:n]) + (f" and {len(names) - n} more" if len(names) > n else "")


def check_upload(grid):
    errors, warnings, dots, modes, dangling = [], [], [], [], []
    for dirpath, dirs, files in os.walk(grid):
        if stat.S_IMODE(os.lstat(dirpath).st_mode) != 0o755:
            modes.append(f"{os.path.relpath(dirpath, grid)}/ is {oct(stat.S_IMODE(os.lstat(dirpath).st_mode))}")
        for name in files + [d for d in dirs if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, grid)
            if name.startswith("."):
                dots.append(rel)
            if os.path.islink(p):
                if not os.path.exists(p):
                    dangling.append(rel)
            elif stat.S_IMODE(os.lstat(p).st_mode) != 0o644:
                modes.append(f"{rel} is {oct(stat.S_IMODE(os.lstat(p).st_mode))}")
    if dots:
        errors.append(f"{len(dots)} dot file(s), e.g. rsync temp files: {_few(dots)}")
    if modes:
        errors.append(f"{len(modes)} path(s) not 755/644: {_few(modes)}")
    if dangling:
        errors.append(f"{len(dangling)} dangling symlink(s): {_few(dangling)}")
    return _result(errors, warnings)


def check_spectra(grid, txt):
    errors, warnings, headers = [], [], {}
    for p in txt:
        headers[p.name], _ = read_header(p)
    if not txt:
        return _result(["no .txt spectra"], warnings), headers, []
    need = {"teff": lambda h: math.isfinite(h["teff"]),
            "wavelength_unit": lambda h: str(h.get("wavelength_unit", "")).lower() == "angstrom",
            "flux_unit": lambda h: str(h.get("flux_unit", "")).lower() == FLUX_UNIT,
            "units_standardized": lambda h: str(h.get("units_standardized", "")).lower() == "true"}
    for key, ok in need.items():
        bad = [n for n, h in headers.items() if not ok(h)]
        if bad:
            errors.append(f"{len(bad)} of {len(txt)} spectra lack {key}"
                          f"{' = erg/cm2/s/A' if key == 'flux_unit' else ''}: {_few(bad)}")
    for param in ("logg", "metallicity"):
        stated = sum(math.isfinite(h[param]) for h in headers.values())
        if 0 < stated < len(txt):
            errors.append(f"{len(txt) - stated} of {len(txt)} spectra do not state {param} (the others do)")
    total = sum(p.stat().st_size for p in txt)
    sample = txt if total < SAMPLE_ALL_BELOW else [txt[i] for i in np.unique(
        np.linspace(0, len(txt) - 1, SAMPLE_SIZE).astype(int))]
    bad_data, negative, read = [], [], []
    for p in sample:
        try:
            wl, fl = read_data(p)
        except ValueError as exc:
            bad_data.append(f"{p.name} ({exc})")
            continue
        if len(wl) < 2 or not (np.all(np.isfinite(wl)) and np.all(np.isfinite(fl))):
            bad_data.append(f"{p.name} (fewer than 2 rows or non-finite values)")
        elif not np.all(np.diff(wl) > 0) or wl[0] <= 0:
            bad_data.append(f"{p.name} (wavelengths not positive and strictly increasing)")
        else:
            if np.any(fl < 0):
                negative.append(f"{p.name} ({int((fl < 0).sum())})")
            read.append((p, wl, fl))
    if bad_data:
        errors.append(f"{len(bad_data)} of {len(sample)} sampled spectra unusable: {_few(bad_data)}")
    if negative:
        warnings.append(f"{len(negative)} sampled spectra hold negative flux values: {_few(negative)}")
    return _result(errors, warnings, spectra=len(txt), sampled=len(sample),
                   sampling="all" if len(sample) == len(txt) else f"{len(sample)} spread over the name order"), \
        headers, read


def _same(header_value, cell, numeric):
    if numeric:
        try:
            a, b = float(header_value), float(cell)
        except (TypeError, ValueError):
            return False
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return str(header_value).strip() == str(cell).strip()


def check_lookup(lookup_path, names, headers):
    """Lookup rows against the spectra on disk and their headers; also the rows as (name -> row)."""
    errors, warnings = [], []
    if not lookup_path.is_file():
        return _result([f"no {lookup_path.name}"], warnings), None, {}
    cols, rows = read_lookup(lookup_path)
    fcol = file_column(cols)
    if not fcol:
        return _result([f"{lookup_path.name} has no file_name column ({cols})"], warnings), cols, {}
    listed = [r.get(fcol, "") for r in rows]
    by_name = {r.get(fcol, ""): r for r in rows}
    if len(set(listed)) != len(listed):
        errors.append(f"{len(listed) - len(set(listed))} duplicate row(s)")
    missing, extra = set(names) - set(listed), set(listed) - set(names)
    if missing:
        errors.append(f"{len(missing)} spectra on disk are not in it: {_few(missing)}")
    if extra:
        errors.append(f"{len(extra)} rows name spectra that are not on disk: {_few(extra)}")
    header_keys = set().union(*(h.keys() for h in headers.values())) if headers else set()
    for p in PARAMS:  # a parameter no spectrum states gets no column (SED_Tools' regenerate_lookup_table)
        if all(math.isnan(h[p]) for h in headers.values()):
            header_keys.discard(p)
    canon = {c: ALIASES.get(c.lower(), c.lower()) for c in cols if c != fcol}
    lacking = sorted(header_keys - set(canon.values()))
    if lacking:
        errors.append(f"no column for header key(s) {lacking}")
    unknown = sorted(c for c, k in canon.items() if k not in header_keys)
    if unknown:
        errors.append(f"column(s) {unknown} that no spectrum header carries")
    for col in (param_column(cols, "teff"), param_column(cols, "logg"), param_column(cols, "metallicity")):
        if col:
            bad = [r[fcol] for r in rows if not _finite(r.get(col))]
            if bad:
                errors.append(f"{len(bad)} rows have no finite {col}: {_few(bad)}")
    placeholders = sorted({c for c in cols if any(r.get(c) == "999.9" for r in rows)})
    mismatch = {}
    for name, h in headers.items():
        row = by_name.get(name)
        if row is None:
            continue
        for col, key in canon.items():
            if key in h and not _same(h[key], row.get(col, ""), key in PARAMS):
                mismatch.setdefault(col, []).append(name)
    for col, bad in sorted(mismatch.items()):
        errors.append(f"column {col} differs from the spectrum header in {len(bad)} rows: {_few(bad)}")
    if placeholders:
        errors.append(f"999.9 placeholders in column(s) {placeholders}")
    return _result(errors, warnings, rows=len(rows), columns=cols), cols, by_name


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def scan_cube(path):
    """Read a cube once, plane by plane: its header and, per node, whether it holds flux and its integral."""
    nt, nl, nm, nw, te, lg, me, wl, off = cube_header(path)
    size, expected = path.stat().st_size, off + 8 * nt * nl * nm * nw
    facts = {"nt": nt, "nl": nl, "nm": nm, "nw": nw, "bytes": size, "expected_bytes": expected,
             "teff": te, "logg": lg, "meta": me, "wavelengths": wl, "offset": off}
    if size != expected:
        return facts
    nodes = nt * nl * nm
    filled = np.zeros(nodes, bool)
    nan = np.zeros(nodes, bool)
    integral = np.zeros(nodes)
    with open(path, "rb") as fh:
        fh.seek(off)
        prev = None
        for i in range(nw):
            plane = np.fromfile(fh, dtype="<f8", count=nodes)
            nan |= ~np.isfinite(plane)
            plane = np.where(np.isfinite(plane), plane, 0.0)
            filled |= plane > 0
            if prev is not None:
                integral += 0.5 * (plane + prev) * (wl[i] - wl[i - 1])
            prev = plane
    # plane index = im*(nl*nt) + il*nt + it (Teff fastest), MESA's order
    facts.update(filled=filled.reshape(nm, nl, nt), nan=nan.reshape(nm, nl, nt),
                 integral=integral.reshape(nm, nl, nt))
    return facts


def node_spectrum(path, facts, it, il, im):
    nt, nl, nm, nw = (facts[k] for k in ("nt", "nl", "nm", "nw"))
    mm = np.memmap(path, dtype="<f8", mode="r", offset=facts["offset"], shape=(nw, nm, nl, nt))
    return np.array(mm[:, im, il, it])


def check_one_cube(cube_path, lookup_cols, rows, folder, label):
    """One flux_cube.bin against the lookup rows it should hold."""
    errors, warnings = [], []
    try:
        f = scan_cube(cube_path)
    except (OSError, ValueError, struct.error) as exc:
        return _result([f"{label}: unreadable ({exc})"], warnings), None
    if f["bytes"] != f["expected_bytes"]:
        return _result([f"{label}: {f['bytes']} bytes, its header says {f['expected_bytes']}"], warnings), f
    for ax in ("teff", "logg", "meta", "wavelengths"):
        a = f[ax]
        if not np.all(np.isfinite(a)) or (len(a) > 1 and not np.all(np.diff(a) > 0)):
            errors.append(f"{label}: {ax} axis is not finite and strictly increasing")
    if f["nan"].any():
        errors.append(f"{label}: {int(f['nan'].sum())} nodes hold NaN or inf")
    if not f["filled"].any():
        errors.append(f"{label}: every node is empty")
    tcol, gcol, mcol = (param_column(lookup_cols, p) for p in PARAMS)
    want, off_axis = {}, []
    for name, r in rows.items():
        t = float(r[tcol]) if tcol and _finite(r.get(tcol)) else math.nan
        g = float(r[gcol]) if gcol and _finite(r.get(gcol)) else 0.0
        m = float(r[mcol]) if mcol and _finite(r.get(mcol)) else 0.0
        idx = tuple(_index(f[a], v) for a, v in (("teff", t), ("logg", g), ("meta", m)))
        if None in idx:
            off_axis.append(name)
        else:
            want.setdefault(idx, []).append(name)
    if off_axis:
        errors.append(f"{label}: {len(off_axis)} lookup rows are not on the cube's axes: {_few(off_axis)}")
    filled = {(it, il, im) for im, il, it in np.argwhere(f["filled"])}
    empty_wanted = sorted(n for idx, ns in want.items() if idx not in filled for n in ns)
    stale = filled - set(want)
    if empty_wanted:
        errors.append(f"{label}: holds no flux for {len(empty_wanted)} lookup rows: {_few(empty_wanted)}")
    if stale:
        errors.append(f"{label}: {len(stale)} filled nodes have no lookup row (stale or synthetic nodes)")
    for ax, name in (("teff", tcol), ("logg", gcol), ("meta", mcol)):
        used = {k[("teff", "logg", "meta").index(ax)] for k in want}
        unused = len(f[ax]) - len(used)
        if unused and rows:
            warnings.append(f"{label}: {unused} {ax} node value(s) carry no spectrum")
    averaged = sum(1 for ns in want.values() if len(ns) > 1)
    if averaged:
        errors.append(f"{label}: {averaged} nodes average several spectra (unsplit extra axes)")
    # every node holding one spectrum: its integral against that spectrum's, resampled as SED_Tools builds cubes
    worst, off = 0.0, []
    wlc = f["wavelengths"]
    for (it, il, im), ns in want.items():
        if len(ns) != 1 or (it, il, im) not in filled:
            continue
        try:
            wl, fl = read_data(folder / ns[0])
        except (OSError, ValueError):
            continue
        ref = trapz(resample(wl, fl, wlc), wlc)
        node = f["integral"][im, il, it]
        dev = abs(node / ref - 1) if ref > 0 else math.inf
        worst = max(worst, dev)
        if dev > NODE_TOL:
            off.append(f"{ns[0]} ({node / ref if ref > 0 else math.inf:.4g})")
    if off:
        errors.append(f"{label}: {len(off)} nodes integrate to more than {NODE_TOL:.0%} away from their own "
                      f"spectrum: {_few(off)}")
    # node values against their spectra, resampled as SED_Tools builds cubes
    compared = []
    for idx, ns in list(want.items()):
        if len(compared) >= 3 or len(ns) != 1 or idx not in filled:
            continue
        p = folder / ns[0]
        try:
            wl, fl = read_data(p)
        except (OSError, ValueError):
            continue
        node = node_spectrum(cube_path, f, *idx)
        # at the wavelengths the spectrum itself samples, a node must equal it whatever the builder's interpolation
        _, ic, isp = np.intersect1d(f["wavelengths"], wl, return_indices=True)
        good = (fl[isp] > 0) & (node[ic] > 0)
        if good.sum() < 10:
            errors.append(f"{label}: node of {ns[0]} shares under 10 sampled wavelengths with it")
            continue
        r = node[ic][good] / fl[isp][good]
        compared.append(f"{ns[0]}: {int(good.sum())} common wavelengths, ratio median {np.median(r):.6f}, "
                        f"range {r.min():.6f}-{r.max():.6f}")
        if abs(np.median(r) - 1) > 1e-4 or r.min() < 0.999 or r.max() > 1.001:
            errors.append(f"{label}: node of {ns[0]} differs from the spectrum (ratio median {np.median(r):.6g})")
    nodes = sorted(want)
    holes = int(f["nt"] * f["nl"] * f["nm"] - len(filled))
    return _result(errors, warnings, label=label, dims=[f["nt"], f["nl"], f["nm"], f["nw"]], bytes=f["bytes"],
                   filled_nodes=len(filled), empty_nodes=holes, nodes_checked_against_spectra=compared,
                   nodes_vs_own_spectrum_worst=round(worst, 6)), \
        {**f, "want": want, "nodes": nodes}


def _index(axis, value):
    if not math.isfinite(value):
        return None
    hit = np.where(np.isclose(axis, value, rtol=0, atol=1e-6 * max(1.0, abs(value))))[0]
    return int(hit[0]) if len(hit) else None


def variant_paths(index):
    cols, rows = read_lookup(index)
    return [r["path"] for r in rows if r.get("path")]


def check_cubes(grid, cols, rows, read, record=None):
    """Check 4 for the top-level cube or every variant cube; returns the scans for check 5.

    A split grid may also hold a top-level flux_cube.bin for clients that read only the top level (SED_Tools
    0.4.1, MESA), if it is a byte copy of the cube of the variant its curated record names as default_variant."""
    errors, warnings, details, scans = [], [], [], []
    top, vindex = grid / "flux_cube.bin", grid / "variants_index.csv"
    default_copy = None
    if top.is_file() and not vindex.is_file():
        res, scan = check_one_cube(top, cols or [], rows, grid, "flux_cube.bin")
        details.append(res)
        errors += res["errors"]
        warnings += res["warnings"]
        if scan:
            scans.append((top, scan))
    if vindex.is_file() and top.is_file():
        default = (record or {}).get("default_variant")
        dcube = grid / default / "flux_cube.bin" if default else None
        if dcube is not None and dcube.is_file() and file_hash(top) == file_hash(dcube):
            default_copy = default
        else:
            errors.append("flux_cube.bin beside variants_index.csv is not a byte copy of the cube of the curated "
                          f"default_variant ({default!r})")
    if vindex.is_file():
        covered = set()
        for rel in variant_paths(vindex):
            vdir = grid / rel
            vlookup, vcube = vdir / "lookup_table.csv", vdir / "flux_cube.bin"
            if not (vlookup.is_file() and vcube.is_file()):
                errors.append(f"variant {rel}: lacks lookup_table.csv or flux_cube.bin")
                continue
            vcols, vrows = read_lookup(vlookup)
            vf = file_column(vcols)
            vby = {r[vf]: r for r in vrows}
            wrong = [n for n in vby if not (vdir / n).is_symlink() or os.readlink(vdir / n) != os.path.join("..", n)]
            if wrong:
                errors.append(f"variant {rel}: {len(wrong)} spectra are not links to the same file in the grid: "
                              f"{_few(wrong)}")
            outside = set(vby) - set(rows)
            if outside:
                errors.append(f"variant {rel}: lists {len(outside)} spectra the grid's lookup does not")
            if covered & set(vby):
                errors.append(f"variant {rel}: shares spectra with another variant")
            covered |= set(vby)
            res, scan = check_one_cube(vcube, vcols, vby, grid, rel)
            details.append(res)
            errors += res["errors"]
            warnings += res["warnings"]
            if scan:
                scans.append((vcube, scan))
        lacking = set(rows) - covered
        if lacking:
            errors.append(f"the variants hold {len(covered)} of {len(rows)} spectra; missing {_few(lacking)}")
    if not top.is_file() and not vindex.is_file():
        warnings.append("no flux cube: clients build one from the spectra")
    h5 = grid / f"{grid.name}.h5"
    if h5.is_file():
        try:
            import h5py
            with h5py.File(h5, "r") as fh:
                held = set(fh["spectra"].keys())
            names = {p.name for p in grid.glob("*.txt")}
            if held != names:
                errors.append(f"{h5.name} holds {len(held & names)} of the {len(names)} spectra"
                              + (f" and {len(held - names)} others" if held - names else ""))
        except ImportError:
            warnings.append(f"{h5.name} not checked (no h5py)")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{h5.name} unreadable ({exc})")
    allowed = {"lookup_table.csv", "flux_cube.bin", f"{grid.name}.h5"}
    if vindex.is_file():
        allowed |= {"variants_index.csv", "lookup_table_full.csv"}
    stray = sorted(p.name for p in grid.iterdir() if p.is_file() and p.suffix.lower() in SERVED_SUFFIXES[1:]
                   and p.name not in allowed)
    if stray:
        warnings.append(f"old clients also download {stray}")
    return _result(errors, warnings, cubes=details, top_level_default_variant=default_copy), scans


def ir_slope(wl, fl, teff):
    """5c: d ln F / d ln lambda over SLOPE_WINDOW times the Wien peak of teff; None if the spectrum stops short."""
    peak = 2.8977719e7 / teff
    k = (wl >= SLOPE_WINDOW[0] * peak) & (wl <= SLOPE_WINDOW[1] * peak) & (fl > 0)
    if k.sum() < 10 or wl[k][-1] < 0.8 * SLOPE_WINDOW[1] * peak:
        return None
    return float(np.polyfit(np.log(wl[k]), np.log(fl[k]), 1)[0])


_PINS = {}


def _pinned(rule, name, ratio):
    """Whether an exception admits this spectrum: with a "pinned" file ({file name: ratio} under ROOT), only a
    listed spectrum whose ratio is unchanged (within 0.1 %); without one, any spectrum."""
    if not rule.get("pinned"):
        return True
    if rule["pinned"] not in _PINS:
        try:
            _PINS[rule["pinned"]] = json.loads((ROOT / rule["pinned"]).read_text())
        except (OSError, ValueError):
            _PINS[rule["pinned"]] = {}
    pin = _PINS[rule["pinned"]].get(name)
    return pin is not None and abs(ratio / pin - 1) <= 1e-3


def check_physics(grid, headers, read, scans, raw_dir, exceptions=(), record=None):
    """Check 5. exceptions: the curated record's bolometric_exceptions, each {"logg_max", "range", "why"}: a
    spectrum with logg at most logg_max whose ratio lies in range is reported as a warning, not an error. A range
    may not admit 1/pi or 1/(4 pi), and the exceptions may not cover most of the judged spectra."""
    errors, warnings = [], []
    ratios, uncovered, excepted, shapes = [], 0, [], []
    for e in exceptions:
        lo, hi = e["range"]
        bad = [k for k, v in FAULT_RATIOS.items() if lo <= v * 1.03 and v / 1.03 <= hi]
        if bad and not e.get("pinned"):
            errors.append(f"5a: bolometric_exceptions range {lo}-{hi} admits {', '.join(bad)} (a unit fault) and pins "
                          "no spectrum's ratio")
    for p, wl, fl in read:
        t = headers[p.name]["teff"]
        if not (math.isfinite(t) and t > 0):
            continue
        bb = trapz(pi_b(wl, t), wl)
        if bb / (SIGMA * t ** 4) < COVERED:
            uncovered += 1
            continue
        r = trapz(fl, wl) / bb
        slope = ir_slope(wl, fl, t)
        if slope is not None:
            shapes.append((slope, p.name))
        rule = next((e for e in exceptions if not BOLO_FAIL[0] <= r <= BOLO_FAIL[1]
                     and headers[p.name]["logg"] <= e["logg_max"] and e["range"][0] <= r <= e["range"][1]
                     and _pinned(e, p.name, r)), None)
        if rule:
            excepted.append(f"{p.name} ({r:.3g}, logg {headers[p.name]['logg']:g})")
            continue
        ratios.append((r, p.name))
    node_ratios = []
    for cube_path, f in scans:
        if "integral" not in f:
            continue
        wl = f["wavelengths"]
        for it, t in enumerate(f["teff"]):
            if t <= 0:
                continue
            bb = trapz(pi_b(wl, t), wl)
            if bb / (SIGMA * t ** 4) < COVERED:
                continue
            col = f["integral"][:, :, it]
            for im, il in np.argwhere(f["filled"][:, :, it]):
                node_ratios.append((col[im, il] / bb, f"{cube_path.parent.name} T={t:g} logg={f['logg'][il]:g} "
                                                      f"[M/H]={f['meta'][im]:g}"))
    summary = {}
    for label, rs in (("spectra", ratios), ("cube_nodes", node_ratios)):
        if not rs:
            continue
        v = np.array([r for r, _ in rs])
        summary[label] = {"n": len(rs), "min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
                          "max": round(float(v.max()), 4)}
        bad = [f"{n} ({r:.3g})" for r, n in rs if not BOLO_FAIL[0] <= r <= BOLO_FAIL[1]]
        odd = [f"{n} ({r:.3g})" for r, n in rs if not BOLO_WARN[0] <= r <= BOLO_WARN[1]]
        if bad:
            errors.append(f"5a: {len(bad)} of {len(rs)} {label.replace('_', ' ')} integrate to under 0.5 or over 2 "
                          f"times pi*B_lambda(Teff): {_few(bad)}")
        elif odd:
            warnings.append(f"5a: {len(odd)} of {len(rs)} {label.replace('_', ' ')} lie outside 0.9-1.1 of "
                            f"pi*B_lambda(Teff): {_few(odd)}")
    narrow = not ratios and not node_ratios and uncovered > 0 and not excepted
    if not ratios and not node_ratios and not narrow:
        errors.append("5a: no spectrum or node could be compared with sigma*T^4 outside the curated exceptions")
    if excepted and len(excepted) > len(ratios):
        errors.append(f"5a: the curated exceptions cover {len(excepted)} of the {len(excepted) + len(ratios)} judged "
                      "spectra: an exception must stay an exception")
    if shapes:
        v = np.array([x for x, _ in shapes])
        summary["infrared_slope"] = {"n": len(shapes), "min": round(float(v.min()), 3),
                                     "median": round(float(np.median(v)), 3), "max": round(float(v.max()), 3),
                                     "allowed": list(SLOPE_RANGE), "basis": SLOPE_NOTE}
        odd = [f"{n} ({x:.2f})" for x, n in shapes if not SLOPE_RANGE[0] <= x <= SLOPE_RANGE[1]]
        if odd:
            errors.append(f"5c: {len(odd)} of {len(shapes)} spectra have an infrared slope (d ln F/d ln lambda at "
                          f"{SLOPE_WINDOW[0]:g}-{SLOPE_WINDOW[1]:g} times the Wien peak) outside {SLOPE_RANGE[0]} to "
                          f"{SLOPE_RANGE[1]}: a tilted spectrum (the BOSZ fault gives about -6): {_few(odd)}")
    elif ratios:
        warnings.append("5c: no sampled spectrum reaches 20 times its Wien peak, so the infrared slope was not judged")
    if narrow:  # e.g. an optical-only grid: sigma*T^4 cannot be judged, so 5b must use raw upstream files
        warnings.append("5a: not applicable: every sampled spectrum covers under half of sigma*T^4")
    if uncovered:
        warnings.append(f"5a: {uncovered} sampled spectra cover under half of sigma*T^4 and were not judged")
    if excepted:
        summary["excepted"] = excepted
        warnings.append(f"5a: {len(excepted)} sampled spectra lie outside 0.5-2 but within the curated record's "
                        f"bolometric_exceptions ({'; '.join(e['why'] for e in exceptions)}): {_few(excepted)}")
    shape = check_shape(headers, raw_dir, grid, record)
    if narrow and shape["method"].startswith("declared"):
        errors.append("5b: a grid too narrow for 5a needs raw upstream files in source_evidence/grids/<grid>/raw/")
    errors += shape.pop("errors")
    warnings += shape.pop("warnings")
    return _result(errors, warnings, bolometric=summary, shape=shape)


def raw_params(path):
    """Parameters from a raw SVO text file's header ('# teff = 3500 K (...)')."""
    h, _ = read_header(path)
    return h


def check_shape(headers, raw_dir, grid, record=None):
    """5b: the grid's spectra against raw upstream files, else declared identity.

    The curated record may declare a conversion {"factor", "original_flux_unit", "header_conversion"} (BOSZ: F = 4 pi H):
    raw files must then equal the spectra divided by factor, and without raw files the headers must declare that
    unit and conversion. raw_match_keys adds header keys to the parameter match; raw_wavelength_file gives the
    wavelengths of raw files that hold only flux columns; a raw file without parameters matches by file name."""
    errors, warnings = [], []
    conv = (record or {}).get("conversion") or {}
    factor = float(conv.get("factor", 1.0))
    extra = tuple((record or {}).get("raw_match_keys", ()))
    rawwl = (record or {}).get("raw_wavelength_file")
    raws = sorted(p for p in raw_dir.iterdir() if p.name.endswith((".txt", ".txt.gz"))) \
        if raw_dir and raw_dir.is_dir() else []
    if raws:
        index = {}
        for name, h in headers.items():
            index.setdefault(_param_key(h, extra), []).append(name)
        done = []
        for raw in raws:
            rh = raw_params(raw)
            plain = raw.name[:-3] if raw.name.endswith(".gz") else raw.name
            hits = index.get(_param_key(rh, extra), []) if math.isfinite(rh["teff"]) else []
            if len(hits) != 1:  # no parameters, or none that match: the same file name (MAST names carry parameters)
                hits = [n for n in (plain, plain.replace("+", "_")) if n in headers][:1]
            if len(hits) != 1:
                warnings.append(f"5b: raw {raw.name} matches {len(hits)} spectra")
                continue
            wl, fl = read_data(grid / hits[0])
            if rawwl:
                rfl = np.loadtxt(raw, comments="#", usecols=(0,), ndmin=1)
                rwl = np.loadtxt(ROOT / rawwl, comments="#", usecols=(0,), ndmin=1)
            else:
                rwl, rfl = read_data(raw)
            rfl = rfl * factor
            common, ia, ib = np.intersect1d(wl, rwl, return_indices=True)
            good = rfl[ib] > 0
            if good.sum() < 10:
                errors.append(f"5b: {hits[0]} shares under 10 wavelengths with raw {raw.name}")
                continue
            r = fl[ia][good] / rfl[ib][good]
            p1, med, p99 = np.percentile(r, [1, 50, 99])
            done.append({"spectrum": hits[0], "raw": raw.name, "common_points": int(good.sum()),
                         "ratio_p1": round(float(p1), 6), "ratio_median": round(float(med), 6),
                         "ratio_p99": round(float(p99), 6)})
            if abs(med - 1) > 0.01 or p1 < 0.98 or p99 > 1.02:
                errors.append(f"5b: {hits[0]} differs in shape from raw {raw.name} "
                              f"(ratio p1/median/p99 {p1:.4g}/{med:.4g}/{p99:.4g})")
        if not done:
            errors.append("5b: no raw upstream file could be matched")
        return {"method": "raw upstream files" + (f" times the declared factor {factor:g}" if factor != 1 else ""),
                "raw_dir": str(raw_dir.relative_to(ROOT)) if raw_dir else None,
                "compared": done, "errors": errors, "warnings": warnings}
    units = {"flam", FLUX_UNIT} | ({str(conv["original_flux_unit"]).lower()} if conv.get("original_flux_unit") else set())
    converted = sorted(n for n, h in headers.items()
                       if str(h.get("original_flux_unit", "")).lower() not in units
                       or str(h.get("original_wavelength_unit", "angstrom" if conv else "")).lower() not in ("angstrom", "a")
                       or (conv.get("header_conversion") and str(h.get("original_flux_unit", "")).lower() not in
                           ("flam", FLUX_UNIT) and h.get("conversion") != conv["header_conversion"]))
    if converted:
        errors.append(f"5b: no raw upstream files, and {len(converted)} spectra do not declare an unconverted "
                      f"original unit (original_flux_unit flam, original_wavelength_unit angstrom): {_few(converted)}")
    return {"method": "declared identity conversion (no raw upstream files archived)",
            "spectra_declaring_identity": len(headers) - len(converted), "errors": errors, "warnings": warnings}


def _param_key(h, extra=()):
    key = []
    for p in PARAMS:
        v = h.get(p, math.nan)
        key.append(None if not isinstance(v, float) or math.isnan(v) else round(v, 4))
    for name in ("alpha", "yhe"):
        v = h.get(name)
        if v is not None:
            m = NUM.search(str(v))
            key.append(round(float(m.group(0)), 4) if m else None)
    key.extend(str(h.get(k, "")).strip() for k in extra)
    return tuple(key)


def check_duplicates(grid, txt, others, cache):
    """Check 6. Spectra with the same data have data parts of the same length, so only those are hashed."""
    errors, warnings = [], []
    by_size = {}
    for p in txt:
        by_size.setdefault(data_size(p), []).append(p)
    other_sizes = set().union(*(o["sizes"] for o in others.values())) if others else set()
    mine = {}  # data hash -> names, for the spectra whose data size occurs twice here or in another grid
    for size, ps in by_size.items():
        if len(ps) > 1 or size in other_sizes:
            for p in ps:
                mine.setdefault(cache.get(p, True)["data_sha256"], []).append(p.name)
    within = [ns for ns in mine.values() if len(ns) > 1]
    if within:
        warnings.append(f"{len(within)} sets of spectra with identical data: {_few(['='.join(ns) for ns in within])}")
    shared = {}
    for name, other in others.items():
        theirs = {cache.get(q, True)["data_sha256"] for size in set(by_size) & set(other["sizes"])
                  for q in other["sizes"][size]}
        n = sum(len(mine[h]) for h in set(mine) & theirs)
        if n:
            shared[name] = n
    for cube in [grid / "flux_cube.bin"] + [p for p in grid.glob("*/flux_cube.bin") if not p.is_symlink()]:
        if not cube.is_file():
            continue
        sha = None
        for name, other in others.items():
            same_size = other["cubes"].get(cube.stat().st_size, [])
            if same_size:
                sha = sha or cache.get(cube, False)["sha256"]
                if any(cache.get(q, False)["sha256"] == sha for q in same_size):
                    errors.append(f"{cube.relative_to(grid)} is byte-identical to a cube of {name}")
    for name, n in sorted(shared.items()):
        errors.append(f"{n} of {len(txt)} spectra have the same data as spectra of {name}: an alias, not a grid")
    return _result(errors, warnings, shared_with=shared)


def data_size(path):
    """Bytes of a spectrum after its leading '#' header lines (only the header is read)."""
    header = 0
    with open(path, "rb") as fh:
        for line in fh:
            if not line.startswith(b"#"):
                break
            header += len(line)
    return os.path.getsize(path) - header


def grid_hashes(grid, cache):
    """Another grid's spectra by data size and its cubes by file size, for check 6 (hashed only on a size match)."""
    sizes, cubes = {}, {}
    for p in sorted(grid.glob("*.txt")):
        sizes.setdefault(data_size(p), []).append(p)
    for p in grid.rglob("flux_cube*.bin"):
        if not p.is_symlink():
            cubes.setdefault(p.stat().st_size, []).append(p)
    return {"sizes": sizes, "cubes": cubes}


def check_provenance(grid, previous):
    errors, warnings = [], []
    info_path = grid / "info.json"
    source = info_path if info_path.is_file() else CURATED / f"{grid.name}.json"
    if not source.is_file():
        return _result([f"no info.json and no {source.relative_to(ROOT)}"], warnings), None
    try:
        info = json.loads(source.read_text())
    except ValueError as exc:
        return _result([f"{source.name} is not JSON ({exc})"], warnings), None
    for key in ("description", "sources", "references"):
        if not info.get(key):
            errors.append(f"{source.name} has no {key}")
    licence = info.get("licence") or {}
    status = licence.get("terms_status")
    if status not in ("verbatim", "none found"):
        errors.append(f"{source.name} licence.terms_status is {status!r} (verbatim or none found)")
    terms = licence.get("terms") or []
    if status == "verbatim":
        if not terms:
            errors.append(f"{source.name}: terms_status verbatim but no terms quoted")
        for t in terms:
            ev = ROOT / str(t.get("evidence", ""))
            if not t.get("evidence") or not ev.is_file():
                errors.append(f"terms quote for {t.get('for')!r} has no archived evidence file ({t.get('evidence')})")
            elif _norm(t.get("text", "")) not in _norm(ev.read_text(encoding="utf-8", errors="ignore")):
                errors.append(f"terms quote for {t.get('for')!r} is not in {t.get('evidence')}: review it")
    elif status == "none found" and not licence.get("searched"):
        errors.append(f"{source.name}: terms 'none found' without saying where was searched")
    digest = hashlib.sha256(json.dumps(terms, sort_keys=True).encode()).hexdigest()
    old = (previous or {}).get("terms_sha256")
    if old and old != digest:
        warnings.append("the terms text changed since the previous manifest: review it")
    return _result(errors, warnings, record=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT)
                   else str(source), terms_sha256=digest), info


def _norm(text):
    text = re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ").replace("&quot;", '"').replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def check_shrinkage(previous, count, axes, grid=None, cache=None):
    """Check 8: no fewer spectra, no narrower axes, and no file whose content changed while its size and
    modification time stayed as the previous manifest records them."""
    errors, warnings = [], []
    if not previous:
        return _result([], ["no previous manifest: first version"], previous=None)
    silent = []
    if grid is not None and cache is not None:
        for rel, rec in (previous.get("files") or {}).items():
            path = Path(grid) / rel
            if not path.is_file() or "sha256" not in rec:
                continue
            st = path.stat()
            if st.st_size == rec.get("size") and int(st.st_mtime) == rec.get("mtime") and \
                    cache.get(path, rel.endswith(".txt"))["sha256"] != rec["sha256"]:
                silent.append(rel)
    if silent:
        errors.append(f"{len(silent)} file(s) changed content with the size and time the previous manifest records: "
                      f"{_few(silent)}")
    if count < previous.get("spectra", 0):
        errors.append(f"{count} spectra, the previous version had {previous['spectra']}")
    for ax, (lo, hi) in axes.items():
        old = (previous.get("axes") or {}).get(ax)
        if old and old.get("range") and (lo > old["range"][0] + 1e-9 or hi < old["range"][1] - 1e-9):
            errors.append(f"{ax} range {lo:g}-{hi:g} is narrower than the previous {old['range']}")
    return _result(errors, warnings, previous=previous.get("generated"))


# ---- one grid ----

def axes_of(headers):
    out = {}
    for p in PARAMS:
        v = sorted({h[p] for h in headers.values() if isinstance(h[p], float) and math.isfinite(h[p])})
        if v:
            out[p] = (v[0], v[-1])
    return out


def check_grid(grid, others, cache, previous=None):
    """The guard's report for one grid folder (see the module docstring)."""
    grid = Path(grid)
    txt = sorted(p for p in grid.iterdir() if p.suffix.lower() == ".txt" and p.is_file() and not p.name.startswith("."))
    checks = {"1_upload": check_upload(grid)}
    checks["2_spectra"], headers, read = check_spectra(grid, txt)
    checks["3_lookup"], cols, rows = check_lookup(grid / "lookup_table.csv", [p.name for p in txt], headers)
    record = load_record(grid)
    checks["4_cube"], scans = check_cubes(grid, cols, rows, read, record)
    checks["5_physics"] = check_physics(grid, headers, read, scans, EVIDENCE / grid.name / "raw",
                                        record.get("bolometric_exceptions", []), record)
    checks["6_duplicate"] = check_duplicates(grid, txt, others, cache)
    checks["7_provenance"], info = check_provenance(grid, previous)
    checks["8_shrinkage"] = check_shrinkage(previous, len(txt), axes_of(headers), grid, cache)
    errors = [f"{k[2:]}: {e}" for k, c in checks.items() for e in c["errors"]]
    warnings = [f"{k[2:]}: {w}" for k, c in checks.items() for w in c["warnings"]]
    return {"grid": grid.name, "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool_version(), "passed": not errors, "errors": errors, "warnings": warnings,
            "checks": checks}, {"headers": headers, "scans": scans, "rows": rows, "columns": cols, "info": info,
                                "read": read}


def load_record(grid):
    """The grid's served info.json updated by its curated record (grid_info/curated/<grid>.json); {} without either."""
    record = {}
    for path in (Path(grid) / "info.json", CURATED / f"{Path(grid).name}.json"):
        try:
            record.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return record


def tool_version():
    """The guard that ran: its last commit, its SHA-256, and whether the file is as committed there."""
    rel = "grid_info/check_grids.py"
    digest = file_hash(ROOT / rel)
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", rel],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True, timeout=10).stdout
        same = bool(blob) and hashlib.sha256(blob).hexdigest() == digest
    except (OSError, subprocess.SubprocessError):
        sha, same = None, False
    return {"script": rel, "commit": sha, "sha256": digest, "as_committed": same}


def previous_manifest(grid):
    try:
        m = json.loads((Path(grid) / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    return {"generated": m.get("generated"), "spectra": (m.get("spectra") or {}).get("count", 0),
            "axes": m.get("axes"), "terms_sha256": (m.get("guard") or {}).get("terms_sha256"),
            "files": m.get("files") or {}}


def grid_folders(stellar_dir):
    return sorted(p for p in Path(stellar_dir).iterdir() if p.is_dir() and not p.name.startswith("."))


_SIZE_MAPS = {}


def other_hashes(stellar_dir, exclude, cache):
    """Size maps (grid_hashes) of every other grid folder holding spectra or a cube, read once per run."""
    out = {}
    for g in grid_folders(stellar_dir):
        if g.name == exclude or not os.access(g, os.R_OK | os.X_OK):
            continue
        if g not in _SIZE_MAPS:
            _SIZE_MAPS[g] = grid_hashes(g, cache) if any(g.glob("*.txt")) or any(g.rglob("flux_cube*.bin")) else None
        if _SIZE_MAPS[g]:
            out[g.name] = _SIZE_MAPS[g]
    return out


# ---- what generate_index.py asks ----

SMALL = 64 << 20   # files up to this size (lookups, small cubes) are also compared by SHA-256 at listing time


def served_paths(grid):
    """Top-level files clients download, plus every variant cube and lookup, as {relative path: Path}."""
    grid = Path(grid)
    out = {p.name: p for p in sorted(grid.iterdir()) if p.is_file() and p.suffix.lower() in SERVED_SUFFIXES}
    vindex = grid / "variants_index.csv"
    if vindex.is_file():
        for rel in variant_paths(vindex):
            for name in ("lookup_table.csv", "flux_cube.bin"):
                if (grid / rel / name).is_file():
                    out[f"{rel}/{name}"] = grid / rel / name
    return out


def served_files(grid):
    """served_paths as {relative path: (size, mtime, ctime_ns)}: a write since the check changes ctime even when
    size and mtime are put back."""
    return {k: (p.stat().st_size, int(p.stat().st_mtime), p.stat().st_ctime_ns) for k, p in served_paths(grid).items()}


def variant_links(grid):
    """{variant folder: SHA-256 of its sorted 'name -> link target' lines}: re-pointed spectrum links change it."""
    grid = Path(grid)
    vindex = grid / "variants_index.csv"
    out = {}
    if vindex.is_file():
        for rel in variant_paths(vindex):
            d = grid / rel
            lines = sorted(f"{q.name} -> {os.readlink(q) if q.is_symlink() else 'file'}"
                           for q in d.iterdir() if q.suffix.lower() == ".txt") if d.is_dir() else []
            out[rel] = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return out


def manifest_matches(grid):
    """(True, "") when <grid>/manifest.json says the guard passed and the folder still holds exactly the files
    it records: by name, size, modification time and ctime (any write since the check changes ctime), by SHA-256
    for files up to 64 MB other than spectra, and with the same variant links; else (False, why)."""
    grid = Path(grid)
    try:
        m = json.loads((grid / "manifest.json").read_text())
    except OSError:
        return False, "no manifest.json"
    except ValueError as exc:
        return False, f"manifest.json is not JSON ({exc})"
    if m.get("schema") != SCHEMA or m.get("grid") != grid.name:
        return False, "manifest.json is for another grid or schema"
    if m.get("status") != "published" or not (m.get("guard") or {}).get("passed"):
        return False, f"status {m.get('status')!r}: {'; '.join((m.get('guard') or {}).get('errors', [])[:2])}"
    files = m.get("files") or {}
    recorded = {k: (v["size"], v["mtime"], v.get("ctime_ns")) for k, v in files.items()
                if k.lower().endswith(SERVED_SUFFIXES)}
    present = served_files(grid)
    changed = sorted(k for k in set(recorded) | set(present) if recorded.get(k) != present.get(k))
    if changed:
        return False, f"{len(changed)} file(s) differ from manifest.json (size, time or ctime): {_few(changed)}"
    paths = served_paths(grid)
    altered = sorted(k for k, p in paths.items() if not k.endswith(".txt") and p.stat().st_size <= SMALL
                     and file_hash(p) != files[k].get("sha256"))
    if altered:
        return False, f"{len(altered)} file(s) differ from manifest.json by SHA-256: {_few(altered)}"
    if variant_links(grid) != (m.get("variant_links") or {}):
        return False, "variant spectrum links differ from manifest.json"
    return True, ""


# ---- command line ----

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", type=Path, default=ROOT / "data")
    ap.add_argument("--only", nargs="*", help="grid folder names (default: every folder that holds files)")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="SHA-256 cache, outside the served tree")
    args = ap.parse_args(argv)
    stellar = args.data / "stellar_models"
    if not stellar.is_dir():
        print(f"check_grids: {stellar} is not a directory", file=sys.stderr)
        return 2
    cache = HashCache(args.cache)
    names = args.only or [g.name for g in grid_folders(stellar) if os.access(g, os.R_OK) and any(g.iterdir())]
    report, failed = {}, 0
    for name in names:
        grid = stellar / name
        if not grid.is_dir():
            print(f"FAIL {name}: no such grid folder")
            failed += 1
            continue
        rep, _ = check_grid(grid, other_hashes(stellar, name, cache), cache, previous_manifest(grid))
        cache.save()
        report[name] = rep
        failed += not rep["passed"]
        print(f"{'ok  ' if rep['passed'] else 'FAIL'} {name}: {len(rep['errors'])} errors, "
              f"{len(rep['warnings'])} warnings")
        for line in rep["errors"] + [f"warn: {w}" for w in rep["warnings"]]:
            print(f"     {line}")
    if args.json:
        args.json.write_text(json.dumps(_jsonable(report), indent=1) + "\n")
    print(f"check_grids: {len(report)} grids, {failed} failing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
