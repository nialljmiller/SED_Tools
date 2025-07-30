# stellar_colors/cube/builder.py

import os
import struct
import numpy as np
from tqdm import tqdm


class DataCubeBuilder:
    def __init__(self, wavelengths, teff_grid, logg_grid, meta_grid, flux_cube):
        self.wavelengths = wavelengths
        self.teff_grid = teff_grid
        self.logg_grid = logg_grid
        self.meta_grid = meta_grid
        self.flux_cube = flux_cube

    def save(self, output_file):
        """Save flux cube and metadata to binary file."""
        with open(output_file, "wb") as f:
            f.write(struct.pack(
                "4i",
                len(self.teff_grid),
                len(self.logg_grid),
                len(self.meta_grid),
                len(self.wavelengths),
            ))
            self.teff_grid.astype(np.float64).tofile(f)
            self.logg_grid.astype(np.float64).tofile(f)
            self.meta_grid.astype(np.float64).tofile(f)
            self.wavelengths.astype(np.float64).tofile(f)
            self.flux_cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)

    @classmethod
    def from_directory(cls, model_dir):
        lookup = os.path.join(model_dir, "lookup_table.csv")
        names, teffs, loggs, metals = load_lookup_table(lookup)

        teff_grid = get_unique_sorted(teffs)
        logg_grid = get_unique_sorted(loggs)
        meta_grid = get_unique_sorted(metals)

        ref_wave, _ = load_sed(os.path.join(model_dir, names[0]))
        n_lambda = len(ref_wave)
        cube = np.zeros((len(teff_grid), len(logg_grid), len(meta_grid), n_lambda))
        filled = np.zeros((len(teff_grid), len(logg_grid), len(meta_grid)), dtype=bool)

        for fname, teff, logg, meta in tqdm(zip(names, teffs, loggs, metals), total=len(names)):
            i, j, k = nearest_index(teff, teff_grid), nearest_index(logg, logg_grid), nearest_index(meta, meta_grid)
            path = os.path.join(model_dir, fname)
            try:
                wave, flux = load_sed(path)
                if len(wave) == len(ref_wave) and np.allclose(wave, ref_wave):
                    cube[i, j, k, :] = flux
                else:
                    cube[i, j, k, :] = np.interp(ref_wave, wave, flux, left=flux[0], right=flux[-1])
                filled[i, j, k] = True
            except Exception as e:
                print(f"[!] Failed {fname}: {e}")

        return cls(ref_wave, teff_grid, logg_grid, meta_grid, cube)


def load_lookup_table(path):
    names, teffs, loggs, metals = [], [], [], []
    with open(path, "r") as f:
        header = next(f).strip().split(",")
        col_map = {k.lower(): i for i, k in enumerate(header)}

        for line in f:
            vals = line.strip().split(",")
            try:
                names.append(vals[col_map["filename"]].strip())
                teffs.append(float(vals[col_map["teff"]]))
                loggs.append(float(vals[col_map["logg"]]))
                metals.append(float(vals[col_map.get("meta", col_map.get("feh"))]))
            except Exception:
                continue

    return names, np.array(teffs), np.array(loggs), np.array(metals)


def load_sed(path):
    wave, flux = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip().startswith("#"):
                try:
                    w, fval = map(float, line.strip().split())
                    wave.append(w)
                    flux.append(fval)
                except:
                    continue
    return np.array(wave), np.array(flux)


def get_unique_sorted(vals, tol=1e-8):
    sorted_vals = np.sort(vals)
    unique = [sorted_vals[0]]
    for v in sorted_vals[1:]:
        if abs(v - unique[-1]) > tol:
            unique.append(v)
    return np.array(unique)


def nearest_index(val, grid):
    idx = np.searchsorted(grid, val)
    if idx == len(grid):
        return idx - 1
    elif idx > 0 and abs(grid[idx - 1] - val) < abs(grid[idx] - val):
        return idx - 1
    return idx


def build_flux_cube(model_dir: str, output_file: str = "flux_cube.bin"):
    """
    Load atmosphere models from model_dir, build flux cube, and save to output_file.
    This replicates the behavior of precompute_flux_cube.py.
    """
    import os
    import struct
    import numpy as np
    from tqdm import tqdm

    def load_lookup_table(path):
        names, teffs, loggs, metas = [], [], [], []
        with open(path, "r") as f:
            header = f.readline().strip().split(",")
            col = {name.lower(): i for i, name in enumerate(header)}

            for line in f:
                vals = line.strip().split(",")
                try:
                    names.append(vals[col["filename"]])
                    teffs.append(float(vals[col["teff"]]))
                    loggs.append(float(vals[col["logg"]]))
                    metas.append(float(vals.get("meta", vals.get("feh"))))
                except Exception:
                    continue

        return names, np.array(teffs), np.array(loggs), np.array(metas)

    def load_sed(path):
        wave, flux = [], []
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    try:
                        w, fval = map(float, line.strip().split())
                        wave.append(w)
                        flux.append(fval)
                    except:
                        continue
        return np.array(wave), np.array(flux)

    def get_unique_sorted(values, tol=1e-6):
        sorted_vals = np.sort(values)
        unique = [sorted_vals[0]]
        for v in sorted_vals[1:]:
            if abs(v - unique[-1]) > tol:
                unique.append(v)
        return np.array(unique)

    def nearest_index(val, grid):
        idx = np.searchsorted(grid, val)
        if idx == len(grid):
            return idx - 1
        elif idx > 0 and abs(grid[idx - 1] - val) < abs(grid[idx] - val):
            return idx - 1
        return idx

    lookup_file = os.path.join(model_dir, "lookup_table.csv")
    names, teffs, loggs, metas = load_lookup_table(lookup_file)

    teff_grid = get_unique_sorted(teffs)
    logg_grid = get_unique_sorted(loggs)
    meta_grid = get_unique_sorted(metas)

    ref_wave, _ = load_sed(os.path.join(model_dir, names[0]))
    n_lambda = len(ref_wave)

    cube = np.zeros((len(teff_grid), len(logg_grid), len(meta_grid), n_lambda))

    for fname, teff, logg, meta in tqdm(zip(names, teffs, loggs, metas), total=len(names)):
        i = nearest_index(teff, teff_grid)
        j = nearest_index(logg, logg_grid)
        k = nearest_index(meta, meta_grid)

        wave, flux = load_sed(os.path.join(model_dir, fname))
        if len(wave) != len(ref_wave) or not np.allclose(wave, ref_wave):
            flux = np.interp(ref_wave, wave, flux)

        cube[i, j, k, :] = flux

    with open(output_file, "wb") as f:
        f.write(struct.pack("4i", len(teff_grid), len(logg_grid), len(meta_grid), len(ref_wave)))
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        ref_wave.astype(np.float64).tofile(f)
        cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)




class FluxCube:
    def __init__(self, data, wavelengths, parameters, filters):
        self.data = data
        self.wavelengths = wavelengths
        self.parameters = parameters
        self.filters = filters

    def save(self, filename):
        import h5py
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=self.data)
            f.create_dataset('wavelengths', data=self.wavelengths)
            f.create_dataset('parameters', data=self.parameters)
            f.create_dataset('filters', data=self.filters)

    @classmethod
    def load(cls, filename):
        import h5py
        with h5py.File(filename, 'r') as f:
            data = f['data'][()]
            wavelengths = f['wavelengths'][()]
            parameters = f['parameters'][()]
            filters = f['filters'][()]
        return cls(data, wavelengths, parameters, filters)
