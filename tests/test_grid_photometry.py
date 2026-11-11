"""grid_info.photometry and grid_info.compare on a tiny synthetic grid and instrument (server-only)."""

import json
import math

import h5py
import numpy as np

from grid_info import check_grids, compare, photometry
from sed_tools.filters.properties import ST_FLAM
from sed_tools.photometry import band_average_flux_lambda, load_filter_curve

H, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
BANDS = {"blue": (4000, 5000), "red": (6000, 8000), "far": (1e5, 3e5)}   # far: beyond every spectrum


def planck(lam, t):
    x = lam * 1e-8
    return math.pi * 2 * H * C ** 2 / x ** 5 / np.expm1(H * C / (x * K * t)) * 1e-8


def make_grid(data, name, teffs, hole=None):
    folder = data / "stellar_models" / name
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"axes": {"teff": {}, "logg": {}, "metallicity": {}},
                                                      "extra_axes": {}, "variants": [], "generated": "test"}))
    with h5py.File(folder / f"{name}.h5", "w") as h5:
        group = h5.create_group("spectra")
        for i, (t, g) in enumerate((t, g) for t in teffs for g in (4.0, 5.0)):
            if (t, g) == hole:
                continue
            lam = np.geomspace(100, 5e4, 3000 + 7 * (i % 2))     # two wavelength grids
            item = group.create_group(f"{name}_{i}.txt")
            item.create_dataset("lambda", data=lam)
            item.create_dataset("flux", data=planck(lam, t))
            item.attrs.update({"teff": t, "logg": g, "feh": 0.0})


def make_data(tmp_path):
    data = tmp_path / "data"
    inst = data / "filters" / "Fac" / "Inst"
    inst.mkdir(parents=True)
    for band, (lo, hi) in BANDS.items():
        lam = np.linspace(lo, hi, 41)
        rows = "".join(f"{x},{y}\n" for x, y in zip(lam, np.sin(np.linspace(0, np.pi, 41))))
        (inst / f"{band}.dat").write_text(f"Wavelength,Transmission,FilterID=Fac/Inst.{band};DetectorType=photon;"
                                          f"WavelengthUnit=Angstrom\n{rows}")
    (data / "vega").mkdir()
    lam = np.geomspace(900, 5e4, 4000)        # ends before the far band
    (data / "vega" / "vega_flam.csv").write_text(
        "WAVELENGTH,FLUX\n" + "".join(f"{x},{y}\n" for x, y in zip(lam, 1e-21 * planck(lam, 9600.0))))
    make_grid(data, "G", (4000.0, 5000.0, 6000.0), hole=(5000.0, 5.0))
    make_grid(data, "G2", (4000.0, 6000.0))
    return data


def test_tables_match_sed_tools_photometry(tmp_path):
    data = make_data(tmp_path)
    out = tmp_path / "out"
    photometry.main([str(data), str(out), "--grids", "G", "G2", "--instruments", "Fac/Inst"])
    points = json.loads((out / "G" / "points.json").read_text())
    table = json.loads((out / "G" / "Fac" / "Inst.json").read_text())["bands"]
    with h5py.File(data / "stellar_models" / "G" / "G.h5") as h5:
        for i, name in enumerate(points["points"]["file"]):
            item = h5["spectra"][name]
            ref = band_average_flux_lambda(item["lambda"][()], item["flux"][()],
                                           load_filter_curve(str(data / "filters/Fac/Inst/blue.dat")))
            assert abs(table["blue"]["m_st"][i] + 2.5 * math.log10(ref / ST_FLAM)) <= 5e-5
    assert table["far"]["outside"] == [1.0] * 5 and table["far"]["m_st"] == [None] * 5
    assert table["far"]["Vega_minus_ST"] is None and table["blue"]["Vega_minus_ST"] is not None
    assert table["blue"]["outside"] is None
    assert points["variants"][0]["nodes"].count(None) == 1                 # the hole at 5000 K, logg 5
    assert all(abs(r - 1) < 0.05 for r in points["points"]["fbol_ratio"])
    assert set(points["points"]["fbol_flag"]) == {None} and min(points["points"]["planck_cover"]) > 0.9
    # --points-only rewrites the same points (apart from the stamp) without the band tables
    photometry.main([str(data), str(out), "--grids", "G", "--points-only"])
    again = json.loads((out / "G" / "points.json").read_text())
    assert again["points"] == points["points"] and again["variants"] == points["variants"]
    index = json.loads((out / "index.json").read_text())
    assert list(index["instruments"]["Fac/Inst"]) == ["G", "G2"] and list(index["grids"]["G"]["instruments"]) == ["Fac/Inst"]
    assert index["grids"]["G"]["points_bytes"] == (out / "G" / "points.json").stat().st_size
    # Guard check 9 on the same bands: "far" lies beyond the spectra (recorded, a warning only when within them);
    # a cube axis with a hole where its spectra sample "red" loses that band (an error).
    grid, lam = data / "stellar_models" / "G", np.geomspace(100, 5e4, 3000)
    read = [(grid / "G_0.txt", lam, planck(lam, 5000.0))]
    for axis, errors in ((lam, []), (lam[(lam < 5500) | (lam > 8500)], ["Fac/Inst/red"])):
        res = check_grids.check_bands(grid, [(grid / "flux_cube.bin", {"wavelengths": axis})], read)
        assert [e.rsplit(": ", 1)[1] for e in res["errors"]] == errors
        assert res["not_sampled"]["flux_cube.bin"]["Fac/Inst/far"] == {"inside": 0, "outside": 1.0, "coarse": 0.0}
        assert ("Fac/Inst/red" in res["not_sampled"]["flux_cube.bin"]) == bool(errors)


def test_compare_coinciding_and_interpolated(tmp_path):
    data = make_data(tmp_path)
    out = tmp_path / "out"
    photometry.main([str(data), str(out), "--grids", "G", "G2", "--instruments", "Fac/Inst"])
    same = compare.compare(out, tmp_path / "cmp", "G", "G", detail=["Fac/Inst"], summary={"Fac/Inst": ["blue", "red"]})
    assert same["B_at_A_points"] == {"coinciding": 5, "interpolated": 0, "not_covered": 0}
    assert same["bands"]["Fac/Inst:blue"]["coinciding"]["median_abs"] == 0
    assert "Fac/Inst:far" in same["bands"] and same["bands"]["Fac/Inst:far"]["coinciding"] is None
    across = compare.compare(out, tmp_path / "cmp", "G", "G2")
    assert across["B_at_A_points"] == {"coinciding": 4, "interpolated": 1, "not_covered": 0}
    # Linear interpolation of a Planck band flux across 4000-6000 K: not exact, but bounded.
    assert 0 < across["bands"]["Fac/Inst:blue"]["interpolated"]["median_abs"] < 0.5


def test_slice_and_label_axis(tmp_path):
    data = make_data(tmp_path)
    # A grid whose extra axis only labels Teff ranges (never two values at one node) is not split into variants.
    folder = data / "stellar_models" / "G"
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["extra_axes"] = {"code": ["cool", "hot"]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with h5py.File(folder / "G.h5", "r+") as h5:
        for item in h5["spectra"].values():
            item.attrs["raw:code"] = "cool" if item.attrs["teff"] < 5500 else "hot"
    out = tmp_path / "out"
    photometry.main([str(data), str(out), "--grids", "G", "G2", "--instruments", "Fac/Inst"])
    points = json.loads((out / "G" / "points.json").read_text())
    assert points["extra_axes"] == [] and points["label_axes"] == ["code"] and len(points["variants"]) == 1
    # A node-axis value takes a slice: logg = 4 of G against logg = 4 of G2, on Teff alone.
    r = compare.compare(out, tmp_path / "cmp", "G:logg=4", "G2:logg=4")
    assert r["A"] == "G:logg=4" and r["points_A"] == 3
    assert r["B_at_A_points"] == {"coinciding": 2, "interpolated": 1, "not_covered": 0}
