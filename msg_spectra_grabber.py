#!/usr/bin/env python3
"""
MSG Grids Stellar Spectra Grabber
Downloads and extracts spectra from MSG HDF5 grids into individual files
"""

import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import h5py
import numpy as np
from itertools import product
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class MSGSpectraGrabber:
    def __init__(self, base_dir="../data/stellar_models/", max_workers=5):
        self.base_dir = base_dir
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (MESA Colors Module)"})

        # Add retries
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # MSG endpoints
        self.index_url = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"

        self.model_urls = {}

        os.makedirs(base_dir, exist_ok=True)

    def inspect_hdf5_structure(self, h5_path):
        """Inspect and print the structure of an HDF5 file for debugging."""
        print(f"Inspecting HDF5 file: {h5_path}")
        try:
            with h5py.File(h5_path, "r") as f:
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj)}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                        if obj.size < 10:
                            print(f"    Values: {obj[:]}")
                        else:
                            print(f"    First few values: {obj[:5]}")
                f.visititems(print_structure)
        except Exception as e:
            print(f"Error inspecting HDF5 file: {e}")

    def discover_models(self):
        """Discover all available stellar atmosphere models."""
        print("Discovering available models from MSG grids...")

        try:
            response = self.session.get(self.index_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching model index: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        models = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if '/msg/grids/' in href and href.endswith('.h5'):
                model_name = os.path.basename(href).rstrip('.h5')
                if model_name not in [m['name'] for m in models]:
                    models.append({'name': model_name, 'url': urljoin(self.index_url, href)})

        self.model_urls = {m['name']: m['url'] for m in models}

        return sorted([m['name'] for m in models])

    def get_model_metadata(self, model_name):
        """Get metadata about available spectra for a specific model."""
        print(f"  Fetching metadata for {model_name}...")

        output_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        h5_filename = f"{model_name}.h5"
        h5_path = os.path.join(output_dir, h5_filename)

        model_url = self.model_urls.get(model_name)
        if not model_url:
            print(f"    No URL found for {model_name}")
            return []

        if not os.path.exists(h5_path):
            print(f"    Downloading {h5_filename}...")
            try:
                response = self.session.get(model_url, stream=True, timeout=30)
                response.raise_for_status()
                with open(h5_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.RequestException as e:
                print(f"Error downloading {h5_filename}: {e}")
                return []

        # Inspect HDF5 structure for debugging
        self.inspect_hdf5_structure(h5_path)

        # Extract parameter axes from HDF5
        try:
            with h5py.File(h5_path, "r") as f:
                if model_name == "sg-SPHINX":
                    # SPHINX structure
                    param_names = ["microturbulence", "Teff", "metallicity", "logg"]
                    param_values = []
                    for i in range(1, 5):
                        axis_key = f"vgrid/axes[{i}]/x"
                        if axis_key not in f:
                            raise KeyError(f"Parameter axis {axis_key} not found")
                        param_values.append(f[axis_key][:])
                    v_lin_seq = f["vgrid/v_lin_seq"][:]

                    # Verify v_lin_seq indices
                    max_spec_idx = max(v_lin_seq)
                    min_spec_idx = min(v_lin_seq)
                    if max_spec_idx > 5292 or min_spec_idx < 1:
                        raise ValueError(f"v_lin_seq indices out of range (min: {min_spec_idx}, max: {max_spec_idx}, expected 1-5292)")

                    spectra_info = []
                    for fid, spec_idx in enumerate(v_lin_seq, 1):
                        # Adjust spec_idx to map to specsource/specints[0-5291]
                        adjusted_spec_idx = spec_idx - 1
                        if adjusted_spec_idx < 0 or adjusted_spec_idx >= 5292:
                            print(f"    Warning: Invalid spec_idx {adjusted_spec_idx} for FID {fid}, skipping")
                            continue
                        idx = np.unravel_index(adjusted_spec_idx, [len(v) for v in param_values])
                        meta = {
                            param_names[j]: float(param_values[j][idx[j]])
                            for j in range(len(param_names))
                        }
                        spectra_info.append({"fid": fid, "spec_idx": adjusted_spec_idx, "meta": meta})

                    print(f"    Found {len(spectra_info)} spectra for {model_name}")
                    return spectra_info
                else:
                    # Standard MSG structure
                    wavelength_candidates = ['lambda', 'wavelength', 'wave']
                    data_candidates = ['intensity', 'flux', 'specific_intensity']

                    wavelength_key = None
                    for candidate in wavelength_candidates:
                        if candidate in f and isinstance(f[candidate], h5py.Dataset):
                            wavelength_key = candidate
                            break
                    if not wavelength_key:
                        raise KeyError("No wavelength dataset found (tried: {})".format(', '.join(wavelength_candidates)))

                    data_key = None
                    for candidate in data_candidates:
                        if candidate in f and isinstance(f[candidate], h5py.Dataset):
                            data_key = candidate
                            break
                    if not data_key:
                        raise KeyError("No data dataset found (tried: {})".format(', '.join(data_candidates)))

                    param_names = [
                        k for k in f
                        if isinstance(f[k], h5py.Dataset)
                        and len(f[k].shape) == 1
                        and k not in [wavelength_key, 'mu', data_key]
                        and f[k].dtype.kind == "f"
                    ]
                    param_values = [f[k][:] for k in param_names]

                    if not param_names:
                        print(f"    No parameter axes found for {model_name}, assuming single spectrum")
                        return [{"fid": 1, "indices": (), "meta": {}}]

                    axes_ranges = [range(len(v)) for v in param_values]
                    spectra_info = []
                    fid = 1
                    for indices in product(*axes_ranges):
                        meta = {
                            param_names[j]: float(param_values[j][indices[j]])
                            for j in range(len(param_names))
                        }
                        spectra_info.append({"fid": fid, "indices": indices, "meta": meta})
                        fid += 1

                    print(f"    Found {len(spectra_info)} spectra for {model_name}")
                    return spectra_info
        except Exception as e:
            print(f"    Error reading HDF5 metadata: {e}")
            return []

    def download_spectrum(self, model_name, spectrum, output_path):
        """Extract a single spectrum from HDF5."""
        try:
            output_dir = os.path.dirname(output_path)
            h5_filename = f"{model_name}.h5"
            h5_path = os.path.join(output_dir, h5_filename)

            with h5py.File(h5_path, "r") as f:
                if model_name == "sg-SPHINX":
                    spec_idx = spectrum["spec_idx"]
                    spec_group = f"specsource/specints[{spec_idx}]"
                    if spec_group not in f:
                        raise KeyError(f"Spectrum group {spec_group} not found")
                    lambda_ = f[f"{spec_group}/range/x"][:]
                    spec_data = f[f"{spec_group}/c"][:, 0]  # Shape (1324,)
                    meta = spectrum["meta"]

                    # Handle shape mismatch (1325 wavelengths vs 1324 fluxes)
                    if len(lambda_) == len(spec_data) + 1:
                        lambda_ = lambda_[:-1]  # Truncate last wavelength
                    elif len(lambda_) != len(spec_data):
                        raise ValueError(f"Wavelength ({len(lambda_)}) and flux ({len(spec_data)}) length mismatch")

                    spec_flux = spec_data
                else:
                    wavelength_candidates = ['lambda', 'wavelength', 'wave']
                    data_candidates = ['intensity', 'flux', 'specific_intensity']

                    wavelength_key = None
                    for candidate in wavelength_candidates:
                        if candidate in f and isinstance(f[candidate], h5py.Dataset):
                            wavelength_key = candidate
                            break
                    if not wavelength_key:
                        raise KeyError("No wavelength dataset found")

                    data_key = None
                    for candidate in data_candidates:
                        if candidate in f and isinstance(f[candidate], h5py.Dataset):
                            data_key = candidate
                            break
                    if not data_key:
                        raise KeyError("No data dataset found")

                    lambda_ = f[wavelength_key][:]
                    indices = spectrum["indices"]
                    meta = spectrum["meta"]

                    has_mu = "mu" in f and isinstance(f["mu"], h5py.Dataset)
                    if has_mu:
                        mu = f["mu"][:]

                    dset = f[data_key]
                    spec_data = dset[indices] if indices else dset[()]

                    if has_mu:
                        if spec_data.shape[-1] != len(mu):
                            raise ValueError("Shape mismatch for mu")
                        i_mu = spec_data * mu[np.newaxis, :]
                        spec_flux = 2 * np.pi * np.trapz(i_mu, x=mu, axis=-1)
                    else:
                        if len(spec_data.shape) > 1 or (len(spec_data.shape) == 1 and spec_data.shape[0] != len(lambda_)):
                            raise ValueError("Shape mismatch for flux")
                        spec_flux = spec_data

            with open(output_path, "w", encoding="utf-8") as file:
                for k, v in meta.items():
                    file.write(f"# {k} = {v}\n")
                for w, fl in zip(lambda_, spec_flux):
                    file.write(f"{w:.6e} {fl:.6e}\n")

            return True

        except Exception as e:
            print(f"    Error extracting spectrum FID {spectrum['fid']}: {e}")
            return False

    def parse_metadata(self, file_path):
        """Parse metadata from a downloaded spectrum file."""
        metadata = {}
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("#") and "=" in line:
                        try:
                            key, value = line.split("=", 1)
                            key = key.strip("# ").strip()
                            value = value.strip()

                            # Clean numerical values
                            if key != "file_name":
                                match = re.search(r"[-+]?\d*\.?\d+", value)
                                value = match.group() if match else "999.9"

                            metadata[key] = value
                        except ValueError:
                            continue
                    elif not line.startswith("#"):
                        break  # Stop at first data line
        except Exception as e:
            print(f"    Error parsing metadata in {file_path}: {e}")

        return metadata

    def download_model_spectra(self, model_name, spectra_info):
        """Extract all spectra for a given model."""
        output_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Extracting {len(spectra_info)} spectra for {model_name}...")

        metadata_rows = []
        successful_downloads = 0

        # Extract spectra in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_fid = {}

            for spectrum in spectra_info:
                fid = spectrum["fid"]
                filename = f"{model_name}_fid{fid}.txt"
                output_path = os.path.join(output_dir, filename)

                # Skip if already exists
                if os.path.exists(output_path):
                    metadata = self.parse_metadata(output_path)
                    metadata["file_name"] = filename
                    metadata_rows.append(metadata)
                    successful_downloads += 1
                    continue

                future = executor.submit(
                    self.download_spectrum, model_name, spectrum, output_path
                )
                future_to_fid[future] = (fid, filename, output_path)

            # Process completed extractions
            for future in tqdm(
                as_completed(future_to_fid),
                total=len(future_to_fid),
                desc=f"Extracting {model_name}",
            ):
                fid, filename, output_path = future_to_fid[future]

                try:
                    success = future.result()
                    if success:
                        metadata = self.parse_metadata(output_path)
                        metadata["file_name"] = filename
                        metadata_rows.append(metadata)
                        successful_downloads += 1
                    else:
                        # Clean up failed extraction
                        if os.path.exists(output_path):
                            os.remove(output_path)
                except Exception as e:
                    print(f"    Error processing FID {fid}: {e}")

        # Create lookup table
        if metadata_rows:
            self._create_lookup_table(output_dir, metadata_rows)
            print(f"  Successfully extracted {successful_downloads} spectra")
        else:
            print(f"  No spectra extracted for {model_name}.")

        return successful_downloads

    def _create_lookup_table(self, output_dir, metadata_rows):
        """Create lookup table CSV for the model."""
        lookup_table_path = os.path.join(output_dir, "lookup_table.csv")

        # Get all unique metadata keys
        all_keys = set()
        for row in metadata_rows:
            all_keys.update(row.keys())

        # Define column order
        header = ["file_name"] + sorted(all_keys - {"file_name"})

        # Write CSV
        with open(
            lookup_table_path, mode="w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_file.write("#" + ", ".join(header) + "\n")
            writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
            writer.writerows(metadata_rows)

        print(f"    Lookup table saved: {lookup_table_path}")

    def select_models_interactive(self, available_models):
        """Allow user to select which models to download."""
        print("\nAvailable stellar atmosphere models:")
        print("-" * 50)
        for idx, model in enumerate(available_models, start=1):
            print(f"{idx:2d}. {model}")

        print("\nEnter model numbers to download (comma-separated):")
        print("Example: 1,3,5 or 'all' for all models")

        user_input = input("> ").strip()

        if user_input.lower() == "all":
            return available_models

        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected = [
                available_models[i] for i in indices if 0 <= i < len(available_models)
            ]
            return selected
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
            return self.select_models_interactive(available_models)

    def run(self, selected_models=None):
        """Main execution function."""
        print("MSG Grids Stellar Atmosphere Model Downloader")
        print("=" * 50)

        # Discover available models
        available_models = self.discover_models()
        if not available_models:
            print("No models found on MSG grids page!")
            return

        # Select models to process
        if selected_models is None:
            selected_models = self.select_models_interactive(available_models)

        if not selected_models:
            print("No models selected.")
            return

        print(f"\nSelected models: {', '.join(selected_models)}")
        print("=" * 50)

        total_spectra = 0

        # Process each model
        for model_name in selected_models:
            print(f"\nProcessing model: {model_name}")
            print("-" * 30)

            # Get metadata about available spectra
            spectra_info = self.get_model_metadata(model_name)

            if not spectra_info:
                print(f"  No spectra found for {model_name}")
                continue

            # Extract spectra
            downloaded = self.download_model_spectra(model_name, spectra_info)
            total_spectra += downloaded

        print("\n" + "=" * 50)
        print("Processing complete!")
        print(f"Total spectra extracted: {total_spectra}")
        print(f"Models processed: {len(selected_models)}")
        print(f"Output directory: {self.base_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and extract stellar spectra from MSG grids")
    parser.add_argument(
        "--output",
        type=str,
        default="../data/stellar_models/",
        help="Output directory for downloaded spectra",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel extraction workers"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Specific models to process (if not provided, interactive selection)",
    )

    args = parser.parse_args()

    # Create grabber
    grabber = MSGSpectraGrabber(base_dir=args.output, max_workers=args.workers)

    # Run grabber
    grabber.run(selected_models=args.models)


if __name__ == "__main__":
    main()