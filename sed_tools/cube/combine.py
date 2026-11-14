"""Combine several stellar spectral grids into one grid folder with a unified flux cube.
Each spectrum keeps the flux in its file. A grid that does not state log g or [M/H] (a blackbody grid) holds the
same spectrum at every value of that axis, so its spectra are copied to every value the other grids give it; where
spectra meet at a node, the one placed by more of its grid's own values wins, then the grid named later. Extra axes
(alpha, [alpha/Fe], C/O, helium, ...) split the combination into one subgrid per variant, as the cube builder splits
one grid, and a grid without such an axis is in every variant of it (:func:`write_combined_grid`).
:func:`find_stellar_models` leaves out earlier combine outputs, so combining "all" never folds their copies
back in: a grid whose lookup table has the ``source_model`` and ``original_file`` columns (only
:func:`save_combined_data` writes them), or whose folder is ``combined_models``. One named explicitly is combined.
"""

import itertools
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .build import compute_common_wavelengths, resample_to_grid
from .collision import _normalise_key, _value, discover_extra_axes
from .fileio import write_header
from .variants import _link_spectra, _remove_variants, _subdir_names
from ..ui_utils import banner
from ..grids.lookup import find_logg_column, find_metallicity_column, find_teff_column, write_lookup_csv
from ..grids.spectrum_io import read_text_spectrum


# The lookup columns only save_combined_data writes, and the default output name.
_COMBINED_COLUMNS = {"source_model", "original_file"}
_DEFAULT_COMBINED_NAME = "combined_models"


def find_stellar_models(base_dir=None):
    """``(name, path)`` of every grid in base_dir with a lookup table, except earlier
    combine outputs (see the module docstring), which are named on stdout."""
    if base_dir is None:
        from .model import STELLAR_DIR_DEFAULT
        base_dir = str(STELLAR_DIR_DEFAULT)

    model_dirs = []
    combined = []
    for item, item_path in _grids_with_lookup(base_dir):
        if _is_combined_output(item, os.path.join(item_path, "lookup_table.csv")):
            combined.append(item)
        else:
            model_dirs.append((item, item_path))
    if combined:
        print(f"Leaving out {', '.join(sorted(combined))}: written by an earlier combine. "
              "To combine one again, name it explicitly.")
    return model_dirs


def _grids_with_lookup(base_dir):
    """``(name, path)`` of every grid in base_dir with a lookup table, combine outputs included."""
    paths = [(item, os.path.join(base_dir, item)) for item in os.listdir(base_dir)]
    return [(item, path) for item, path in paths
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "lookup_table.csv"))]


def _is_combined_output(name, lookup_file):
    """True for a grid an earlier combine wrote (see the module docstring)."""
    if name == _DEFAULT_COMBINED_NAME:
        return True
    try:
        with open(lookup_file) as f:
            header = f.readline()
    except (OSError, UnicodeDecodeError):
        return False  # offered as a grid; loading it reports the problem
    return _COMBINED_COLUMNS <= {c.strip() for c in header.strip().lstrip("#").split(",")}


def _count_spectra(lookup_file):
    """Rows of a lookup table as load_model_data reads them: after the header, not blank or '#'."""
    with open(lookup_file) as f:
        next(f, None)
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def select_models_interactive(model_dirs):
    """Present spectral grid options and ask until the answer is 'all', valid grid numbers, or q/b.

    The grids are returned in the order typed; q or b (back) returns [].
    """
    print("\nAvailable stellar spectral grids:")
    print("-" * 50)
    for idx, (name, path) in enumerate(model_dirs, start=1):
        lookup_file = os.path.join(path, "lookup_table.csv")
        try:
            n_models = _count_spectra(lookup_file)
        except (OSError, UnicodeDecodeError):
            n_models = "?"  # only the menu count; loading the grid reports the problem
        print(f"{idx}. {name} ({n_models} model spectra)")

    print("\nEnter spectral grid numbers (comma-separated) or 'all'; q or b to go back:")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "all":
            return model_dirs
        if user_input.lower() in ("q", "quit", "exit", "b", "back"):  # as in the pickers
            return []
        tokens = [x.strip() for x in user_input.split(",")]
        valid = all(t.isdecimal() and 1 <= int(t) <= len(model_dirs) for t in tokens)
        if valid and len(set(map(int, tokens))) == len(tokens):
            return [model_dirs[int(t) - 1] for t in tokens]
        print(f"Invalid selection {user_input!r}: enter distinct numbers from 1 to "
              f"{len(model_dirs)}, comma-separated, or 'all'.")


def load_model_data(model_path):
    """The grid's lookup table: ``files``, ``teff``, ``logg`` and ``meta``, ``model_dir``, ``unstated``, the axes
    (logg, meta) it has no column for, which read as NaN, and ``extra``, {column: values as text} of its extra axes
    (:func:`~sed_tools.cube.collision.discover_extra_axes`). Raises ValueError for a table without a Teff column."""
    lookup_file = os.path.join(model_path, "lookup_table.csv")

    with open(lookup_file, "r") as f:
        header = f.readline().strip().lstrip("#").strip()
        columns = [col.strip() for col in header.split(",")]

    df = pd.read_csv(lookup_file, comment="#", names=columns, skiprows=1)

    file_col = columns[0]
    teff_col, logg_col, meta_col = (find_teff_column(columns), find_logg_column(columns),
                                    find_metallicity_column(columns))
    if teff_col is None:
        raise ValueError(f"{lookup_file} has no Teff column (teff, t_eff, temperature or temp); "
                         "a grid cannot be combined without it.")
    extra = discover_extra_axes({col: df[col].values for col in columns
                                 if col not in (file_col, teff_col, logg_col, meta_col)},
                                list(zip(*(df[col].values for col in (teff_col, logg_col, meta_col) if col))))
    return {
        "files": df[file_col].values,
        "teff": df[teff_col].values,
        "logg": df[logg_col].values if logg_col else np.full(len(df), np.nan),
        "meta": df[meta_col].values if meta_col else np.full(len(df), np.nan),
        "model_dir": model_path,
        "unstated": {axis for axis, col in (("logg", logg_col), ("meta", meta_col)) if col is None},
        "extra": {col: df[col].astype(str).values for col in extra},
    }


def prepare_sed(filepath):
    """Load a spectrum already in standard units (Angstrom, erg/cm²/s/Å, from the spectra cleaner), unscaled.

    Raises OSError if the file cannot be read, and ValueError if it gives fewer than two readable points.
    """
    wl, flux = read_text_spectrum(filepath)
    if wl.size < 2:
        raise ValueError(f"{filepath}: fewer than 2 readable wavelength/flux rows "
                         "(empty or not a spectrum)")
    return wl, flux


def create_unified_grid(all_models_data):
    """Create unified parameter grids from all spectral grids: each axis holds every distinct finite value of the
    grids that state it, compared exactly, as the cube builder compares one grid's; or 0.0 alone when none does (a
    lone blackbody grid's log g and [M/H], as the cube builder reads them)."""
    def axis(key):
        stated = [np.asarray(d[key], dtype=float) for d in all_models_data if key not in d.get("unstated", ())]
        if not stated:
            return np.array([0.0])
        values = np.concatenate(stated)
        return np.unique(values[np.isfinite(values)])

    return axis("teff"), axis("logg"), axis("meta")


def create_common_wavelength_grid(all_models_data):
    """The combined cube's wavelengths: the union of every grid's own, each grid's taken as the cube builder takes
    them (:func:`~sed_tools.cube.build.compute_common_wavelengths`), so the combination keeps each grid's whole
    coverage, and the same grids always give the same axis."""
    print("\nWavelengths of each grid:")
    axes = [compute_common_wavelengths(d["model_dir"], list(d["files"])) for d in all_models_data]
    wavelengths = np.unique(np.concatenate(axes))
    print(f"Combined wavelength grid: {len(wavelengths)} points ({wavelengths[0]:.1f}–{wavelengths[-1]:.1f} Å)")
    return wavelengths


def _placements(all_models_data, teff_grid, logg_grid, meta_grid):
    """``(placed, failed)``: placed is {node index: (grid, row)}, the spectrum that fills each node; failed is
    {(grid, row): why} for every row with a non-finite value of an axis its grid states.

    A spectrum goes to its own Teff, and to its own log g and [M/H] where its grid states them, else to every value
    of that axis ("any value": a blackbody is the same at every log g). Where spectra meet at a node, the one placed
    by more of its grid's own values (``_rank``) wins, and between equals the grid named later.
    """
    placed, ranks, failed = {}, {}, {}
    for g, data in enumerate(all_models_data):
        unstated = data.get("unstated", set())
        rank = _rank(data)
        for i, (file, teff, logg, meta) in enumerate(zip(data["files"], data["teff"], data["logg"], data["meta"])):
            stated = [teff] + [value for axis, value in (("logg", logg), ("meta", meta)) if axis not in unstated]
            if not np.all(np.isfinite(stated)):
                failed[g, i] = (f"{os.path.join(data['model_dir'], file)}: non-finite parameters "
                                f"(teff={teff}, logg={logg}, meta={meta})")
                continue
            axes = ([_nearest_idx(teff_grid, teff)],
                    range(len(logg_grid)) if "logg" in unstated else [_nearest_idx(logg_grid, logg)],
                    range(len(meta_grid)) if "meta" in unstated else [_nearest_idx(meta_grid, meta)])
            for node in itertools.product(*axes):
                if rank >= ranks.get(node, -1):
                    placed[node], ranks[node] = (g, i), rank
    return placed, failed


def _rank(data):
    """How many of the combined axes a grid states besides Teff: log g, [M/H] and the extra axes whose values
    split the combination into variants (set by :func:`_variants`; one value throughout is no axis)."""
    return 2 - len(data.get("unstated", ())) + data.get("variant_axes", 0)


def build_combined_flux_cube(all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid):
    """Build the combined flux cube from all spectral grids.

    Each spectrum keeps the absolute flux in its file (the cleaner ties it to the spectrum's own sigma T^4): no
    spectrum is scaled to another grid's. Each fills the nodes :func:`_placements` gives it; a node no spectrum
    fills stays at zero, as in the cube builder. Raises ValueError, before anything is written, listing every
    spectrum that cannot be read or has non-finite parameters.
    """
    n_teff, n_logg, n_meta, n_lambda = len(teff_grid), len(logg_grid), len(meta_grid), len(wavelength_grid)

    flux_cube = np.zeros((n_teff, n_logg, n_meta, n_lambda))
    source_map = np.full((n_teff, n_logg, n_meta), -1, dtype=int)

    print(f"\nBuilding flux cube: {flux_cube.shape}")
    print(f"Memory: {flux_cube.nbytes / (1024**2):.1f} MB")

    placed, failed = _placements(all_models_data, teff_grid, logg_grid, meta_grid)
    failures = list(failed.values())
    nodes_of = {}
    for node, spectrum in placed.items():
        nodes_of.setdefault(spectrum, []).append(node)

    for model_idx, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        print(f"\nProcessing {model_name}...")
        for i, file in enumerate(tqdm(model_data["files"], desc=f"Spectral grid {model_idx + 1}")):
            if (model_idx, i) in failed:
                continue
            filepath = os.path.join(model_data["model_dir"], file)
            try:
                wl, fl = prepare_sed(filepath)  # every spectrum is read, so every unreadable one is listed
            except (OSError, ValueError) as exc:
                failures.append(str(exc))
                continue
            nodes = nodes_of.get((model_idx, i), [])
            if nodes:
                resampled = resample_to_grid(wl, fl, wavelength_grid)
                for node in nodes:
                    flux_cube[node] = resampled
                    source_map[node] = model_idx

    if failures:
        raise ValueError(
            f"Cannot place {len(failures)} spectra in the combined cube; nothing was written. "
            "Fix them or remove their rows from the lookup table:\n  " + "\n  ".join(failures)
        )

    empty_count = int(np.sum(source_map < 0))
    if empty_count > 0:
        print(f"\n{empty_count} grid points hold no spectrum and stay at zero, as the cube builder leaves them.")

    return flux_cube, source_map


def save_combined_data(output_dir, teff_grid, logg_grid, meta_grid,
                       wavelength_grid, flux_cube, all_models_data):
    """Save the combined flux cube, a copy of every spectrum, and the lookup table of the cube's filled nodes.

    The lookup table has one row per node a spectrum fills (:func:`_placements`), at the spectrum's own values and,
    for an axis its grid does not state, the node's; so a copied spectrum has a row at each of its nodes, and a
    spectrum that fills no node has none. Raises OSError after writing the lookup table if any copy failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    grids = (teff_grid, logg_grid, meta_grid)
    _write_cube(os.path.join(output_dir, "flux_cube.bin"), grids, wavelength_grid, flux_cube)
    names = _copy_names(all_models_data)
    lookup_df = _lookup_rows(all_models_data, grids, names)
    copy_failures = _copy_spectra(output_dir, all_models_data, names, set(lookup_df["file_name"]))
    _write_lookup(os.path.join(output_dir, "lookup_table.csv"), lookup_df)
    _raise_copy_failures(output_dir, copy_failures)
    return lookup_df


def _write_cube(path, grids, wavelength_grid, flux_cube):
    with open(path, "wb") as f:
        write_header(f, *grids, wavelength_grid)
        # Transpose to (wavelength, meta, logg, teff) for Fortran column-major order
        flux_cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)
    print(f"\nSaved flux cube: {path}")


def _write_lookup(path, lookup_df):
    with open(path, "w") as f:
        f.write("#" + ",".join(lookup_df.columns) + "\n")
        lookup_df.to_csv(f, index=False, header=False)
    print(f"Saved lookup table: {path}")
    print(f"Total model spectra: {len(lookup_df)}")


def _raise_copy_failures(output_dir, copy_failures):
    if copy_failures:
        raise OSError(
            f"Could not copy {len(copy_failures)} spectra into {output_dir}; its lookup table "
            "lists them but the files are missing. Fix the cause and run the combine again "
            "(copies already made are kept):\n  " + "\n  ".join(copy_failures)
        )


def _copy_names(all_models_data):
    """{(grid, row): <grid>_<n>.txt}, numbered through all grids in order."""
    names, counter = {}, 0
    for g, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        for row in _row_ids(model_data):
            names[g, row] = f"{model_name}_{counter:06d}.txt"
            counter += 1
    return names


def _copy_spectra(output_dir, all_models_data, names, listed):
    """Copy into output_dir every spectrum whose new name the lookup table lists (one that fills no node is left
    out, so the folder holds what its cubes hold); returns the copies that failed."""
    print("\nCopying SED files...")
    failures = []
    for g, model_data in enumerate(tqdm(all_models_data, desc="Copying")):
        for row, orig_file in zip(_row_ids(model_data), model_data["files"]):
            if names[g, row] not in listed:
                continue
            src_path = os.path.join(model_data["model_dir"], orig_file)
            try:
                shutil.copy2(src_path, os.path.join(output_dir, names[g, row]))
            except OSError as exc:
                failures.append(f"{src_path}: {exc}")
    return failures


def _remove_previous_output(output_dir):
    """Remove what an earlier combine wrote in output_dir, so none of it is left beside the new combination or read
    in its place: the spectra its lookup table lists, its flux_cube.bin, and its variants (variants_index.csv and
    the subgrids it lists). A folder whose lookup table is not a combine's is left alone."""
    lookup = os.path.join(output_dir, "lookup_table.csv")
    if not os.path.isfile(lookup) or not _is_combined_output(None, lookup):
        return
    listed = pd.read_csv(lookup, comment=None, dtype=str).iloc[:, 0]
    for name in set(listed.dropna()):
        path = os.path.join(output_dir, name)
        if os.path.basename(name) == name and os.path.isfile(path):
            os.remove(path)
    _remove_variants(output_dir)
    if os.path.isfile(os.path.join(output_dir, "flux_cube.bin")):
        os.remove(os.path.join(output_dir, "flux_cube.bin"))


def _row_ids(model_data):
    """Each row's number in its grid's lookup table (a variant holds some of them)."""
    return model_data.get("rows", range(len(model_data["files"])))


def _lookup_rows(all_models_data, grids, names, extra=None):
    """The combined lookup table: a row per filled node, in grid and row order (see :func:`save_combined_data`),
    with ``extra``, {column: value}, on every row."""
    placed, _ = _placements(all_models_data, *grids)
    nodes_of = {}
    for node, spectrum in sorted(placed.items()):
        nodes_of.setdefault(spectrum, []).append(node)
    rows = []
    for g, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        unstated = model_data.get("unstated", set())
        for i, (row, orig_file) in enumerate(zip(_row_ids(model_data), model_data["files"])):
            for node in nodes_of.get((g, i), []):
                values = {axis: grid[k] if axis in unstated else model_data[axis][i]
                          for axis, grid, k in zip(("teff", "logg", "meta"), grids, node)}
                rows.append({"file_name": names[g, row], **values, **(extra or {}), "source_model": model_name,
                             "original_file": orig_file})
    return pd.DataFrame(rows, columns=["file_name", "teff", "logg", "meta", *(extra or {}), "source_model",
                                       "original_file"])


def _variants(all_models_data):
    """The combination's variants: [(key, grids)], key being ((column, value), ...) sorted by column, as the cube
    builder names one grid's variants, and grids each grid's rows in that variant (all of them, () with no extra
    axis). A variant is a value of every extra axis: the values one grid's rows give together, with every value the
    grids give an axis that grid does not state. A row is in every variant that agrees with the extra-axis values its
    grid states, so a grid without an axis is in every variant of it ("any value", as for log g and [M/H])."""
    columns, texts, row_keys = {}, {}, []
    for data in all_models_data:
        axes = {_normalise_key(col): col for col in data.get("extra", {})}
        for axis, col in axes.items():
            columns.setdefault(axis, col)
        keys = []
        for i in range(len(data["files"])):
            key = {}
            for axis, col in axes.items():
                text = str(data["extra"][col][i]).strip()
                key[axis] = _value(text)
                texts.setdefault((axis, key[axis]), text)
            keys.append(key)
        row_keys.append(keys)
    if not columns:
        return [((), all_models_data)]

    values = {axis: [value for a, value in texts if a == axis] for axis in columns}
    varying = {axis for axis in columns if len(values[axis]) > 1}
    full = set()
    for keys in row_keys:
        for partial in {tuple(key.items()) for key in keys if key}:
            others = [axis for axis in columns if axis not in dict(partial)]
            for chosen in itertools.product(*(values[axis] for axis in others)):
                full.add(tuple(sorted({**dict(partial), **dict(zip(others, chosen))}.items(), key=str)))

    variants = []
    for assignment in map(dict, full):
        grids = []
        for data, keys in zip(all_models_data, row_keys):
            states = {_normalise_key(col) for col in data.get("extra", {})} & varying
            rows = np.array([i for i, key in enumerate(keys) if all(assignment[a] == v for a, v in key.items())],
                            dtype=int)
            grids.append({**data, "rows": rows, "variant_axes": len(states), **{name: np.asarray(data[name])[rows]
                                                    for name in ("files", "teff", "logg", "meta")},
                          "extra": {col: np.asarray(v)[rows] for col, v in data.get("extra", {}).items()}})
        key = tuple(sorted((columns[axis], texts[axis, value]) for axis, value in assignment.items()))
        variants.append((key, grids))
    return sorted(variants, key=lambda variant: variant[0])


def write_combined_grid(all_models_data, output_dir, visualization=True):
    """Combine the grids into output_dir on one wavelength grid, the union of theirs.

    With no extra axis, or one value of each, output_dir holds the combined flux cube, a copy of every spectrum and
    the lookup table (:func:`save_combined_data`). Otherwise the combination is split as the cube builder splits one
    grid (:func:`_variants`): output_dir holds the copies, the lookup table of every variant's rows with their
    extra-axis values, and variants_index.csv, and each variant is a subgrid folder with its lookup table, links to
    its spectra and its flux_cube.bin, so no spectrum collides with or overwrites another variant's. Every cube is
    built before anything is written: a spectrum that cannot be read or placed stops the combine with nothing written.
    What an earlier combine wrote in output_dir is then replaced (:func:`_remove_previous_output`).
    """
    wavelength_grid = create_common_wavelength_grid(all_models_data)
    built = []
    for key, grids in _variants(all_models_data):
        if key:
            print("\nVariant " + ", ".join(f"{column} = {value}" for column, value in key))
        axes = create_unified_grid(grids)
        flux_cube, source_map = build_combined_flux_cube(grids, *axes, wavelength_grid)
        built.append((key, grids, axes, flux_cube, source_map))

    _remove_previous_output(output_dir)
    if len(built) == 1:
        _, grids, axes, flux_cube, source_map = built[0]
        save_combined_data(output_dir, *axes, wavelength_grid, flux_cube, grids)
        if visualization:
            visualize_parameter_space(*axes, source_map, grids, output_dir, wavelength_grid=wavelength_grid,
                                      flux_cube=flux_cube)
        return

    os.makedirs(output_dir, exist_ok=True)
    names = _copy_names(all_models_data)
    subdirs = _subdir_names(os.path.basename(os.path.abspath(output_dir)), [key for key, *_ in built])
    tables, index = [], []
    for key, grids, axes, flux_cube, source_map in built:
        name = subdirs[key]
        subdir = os.path.join(output_dir, name)
        os.makedirs(subdir, exist_ok=True)
        print(f"\n  [{name}]")
        _write_cube(os.path.join(subdir, "flux_cube.bin"), axes, wavelength_grid, flux_cube)
        table = _lookup_rows(grids, axes, names, dict(key))
        _write_lookup(os.path.join(subdir, "lookup_table.csv"), table)
        _link_spectra(subdir, sorted(set(table["file_name"])))
        if visualization:
            visualize_parameter_space(*axes, source_map, grids, subdir, wavelength_grid=wavelength_grid,
                                      flux_cube=flux_cube)
        tables.append(table)
        index.append({"variant_name": name, **dict(key), "n_spectra": table["file_name"].nunique(), "path": name})
    table = pd.concat(tables, ignore_index=True)
    copy_failures = _copy_spectra(output_dir, all_models_data, names, set(table["file_name"]))
    _write_lookup(os.path.join(output_dir, "lookup_table.csv"), table)
    index_path = os.path.join(output_dir, "variants_index.csv")
    write_lookup_csv(index, index_path, columns=["variant_name", *(column for column, _ in built[0][0]), "n_spectra",
                                                 "path"])
    print(f"Variants index: {index_path}")
    _raise_copy_failures(output_dir, copy_failures)


def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def _plot_meta_index(meta_grid):
    """[M/H] index to plot: the one nearest 0 if the grid spans 0, else the middle."""
    if meta_grid.min() <= 0.0 <= meta_grid.max():
        return _nearest_idx(meta_grid, 0.0)
    return len(meta_grid) // 2


def _sed_sample_points(teff_grid, logg_grid, meta_grid, meta_idx, flux_cube, source_map, model_names):
    """Up to 15 nodes (5 Teff x 3 log g quantiles) at meta_idx that have positive flux."""
    def spread(grid, count):
        return np.unique(grid[np.round(np.linspace(0, len(grid) - 1, count)).astype(int)])

    sample_points = []
    for t_val in spread(teff_grid, 5):
        for g_val in spread(logg_grid, 3):
            ti = _nearest_idx(teff_grid, t_val)
            gi = _nearest_idx(logg_grid, g_val)
            flux = flux_cube[ti, gi, meta_idx, :]
            if not np.any(flux > 0):
                continue
            src_idx = int(source_map[ti, gi, meta_idx])
            src_name = model_names[src_idx] if 0 <= src_idx < len(model_names) else "unknown"
            sample_points.append((teff_grid[ti], logg_grid[gi], meta_grid[meta_idx], flux, src_name, src_idx))
    return sample_points


def plot_sed_samples(teff_grid, logg_grid, meta_grid, wavelength_grid,
                     flux_cube, source_map, all_models_data, output_dir):
    """Plot sample SEDs across the combined parameter space, coloured by source grid, as output_dir/sed_samples.png."""
    model_names = [os.path.basename(d["model_dir"]) for d in all_models_data]
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_names), 1)))

    meta_idx = _plot_meta_index(meta_grid)
    sample_points = _sed_sample_points(teff_grid, logg_grid, meta_grid, meta_idx,
                                       flux_cube, source_map, model_names)
    if not sample_points:
        print("  Warning: no valid SED samples found — skipping sed_samples.png")
        return

    # Cap at 9 panels (3×3)
    if len(sample_points) > 9:
        indices = np.round(np.linspace(0, len(sample_points) - 1, 9)).astype(int)
        sample_points = [sample_points[i] for i in indices]

    n = len(sample_points)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (teff, logg, meta, flux, src_name, src_idx) in enumerate(sample_points):
        ax = axes[i]
        color = cmap_colors[src_idx % len(cmap_colors)]
        valid = flux > 0
        if valid.sum() > 1:
            ax.loglog(wavelength_grid[valid], flux[valid], color=color, linewidth=0.8, label=src_name)
        ax.set_title(f"Teff={teff:.0f} K  log g={logg:.2f}  [M/H]={meta:.2f}\nsource: {src_name}", fontsize=8)
        ax.set_xlabel("Wavelength (Å)", fontsize=7)
        ax.set_ylabel("Flux (erg/cm²/s/Å)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"SED Samples Across Combined Parameter Space  ([M/H] ≈ {meta_grid[meta_idx]:.2f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "sed_samples.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")


def visualize_parameter_space(teff_grid, logg_grid, meta_grid, source_map,
                              all_models_data, output_dir,
                              wavelength_grid=None, flux_cube=None):
    """Create parameter space visualization."""
    print("\nCreating visualization...")

    model_names = [os.path.basename(d["model_dir"]) for d in all_models_data]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig = plt.figure(figsize=(15, 10))

    panels = (("teff", "logg", "Teff (K)", "log g"), ("teff", "meta", "Teff (K)", "[M/H]"),
              ("logg", "meta", "log g", "[M/H]"))
    for position, (x_key, y_key, x_label, y_label) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, position)
        for idx, (name, data) in enumerate(zip(model_names, all_models_data)):
            ax.scatter(data[x_key], data[y_key], c=[colors[idx]], label=name, alpha=0.6, s=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if position == 1:
            ax.invert_yaxis()  # log g downwards, as in an HR diagram
        ax.grid(True, alpha=0.3)
        if position == 1:
            ax.legend(fontsize=8)  # one legend for the three panels

    ax4 = fig.add_subplot(2, 2, 4)
    density = np.zeros((len(teff_grid), len(logg_grid)))
    for i, j in np.ndindex(density.shape):
        density[i, j] = len(np.unique(source_map[i, j, :][source_map[i, j, :] >= 0]))

    im = ax4.imshow(density.T, origin="lower", aspect="auto",
                    extent=[teff_grid.min(), teff_grid.max(), logg_grid.min(), logg_grid.max()], cmap="YlOrRd")
    ax4.set_xlabel("Teff (K)")
    ax4.set_ylabel("log g")
    ax4.set_title("Model spectrum density per grid point")
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "parameter_space.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")

    _print_combination_summary(teff_grid, logg_grid, meta_grid, source_map, model_names)

    if wavelength_grid is not None and flux_cube is not None:
        plot_sed_samples(teff_grid, logg_grid, meta_grid, wavelength_grid,
                         flux_cube, source_map, all_models_data, output_dir)


def _print_combination_summary(teff_grid, logg_grid, meta_grid, source_map, model_names):
    """Print the combined grid's extent and each source grid's share of its nodes."""
    banner("SUMMARY")
    print(f"Grid: {len(teff_grid)} × {len(logg_grid)} × {len(meta_grid)} = "
          f"{len(teff_grid) * len(logg_grid) * len(meta_grid):,} points")
    print(f"Teff: {teff_grid.min():.0f} - {teff_grid.max():.0f} K")
    print(f"logg: {logg_grid.min():.2f} - {logg_grid.max():.2f}")
    print(f"[M/H]: {meta_grid.min():.2f} - {meta_grid.max():.2f}")
    print("\nContributions:")
    for idx, name in enumerate(model_names):
        n = np.sum(source_map == idx)
        print(f"  {name}: {n:,} ({100 * n / source_map.size:.1f}%)")
