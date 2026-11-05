"""photometry() takes a (name, path) tuple as one filter, several such tuples, and plain
paths and sequences of them as before.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import numpy as np
import pytest

from sed_tools.models import SEDModel

from helpers_grids import no_network  # noqa: F401 (an autouse fixture)


# ---------------------------------------------------------------- 8. photometry(("V", path))
@pytest.fixture
def evaluated(tmp_path, cube_writer):
    teff, logg, meta = [4000.0, 6000.0], [4.0, 5.0], [-0.5, 0.5]
    wave = np.arange(3000.0, 8001.0, 100.0)
    shape = np.exp(-((wave - 5500.0) / 2000.0) ** 2) + 0.1
    flux = np.array([[[t / 5000.0 * shape for _ in meta] for _ in logg] for t in teff])
    cube_writer(tmp_path / "flux_cube.bin", teff, logg, meta, wave, flux)
    return SEDModel(name="tiny", flux_cube_path=tmp_path / "flux_cube.bin")(5000.0, 4.5, 0.0)


def filter_file(path, lo, hi):
    rows = [(lo - 100, 0.0), (lo, 1.0), (hi, 1.0), (hi + 100, 0.0)]
    path.write_text("# WavelengthUnit = Angstrom\n" + "".join(f"{w:.1f} {t}\n" for w, t in rows))
    return path


@pytest.fixture
def bands(tmp_path):
    return filter_file(tmp_path / "johnson_v.dat", 5000, 6000), filter_file(tmp_path / "johnson_b.dat", 4000, 5000)


def magnitudes(results):
    return {name: result.magnitude for name, result in results.items()}


def test_photometry_takes_a_single_name_path_tuple_as_one_filter(evaluated, bands):
    v, _ = bands
    plain = magnitudes(evaluated.photometry(v))

    assert magnitudes(evaluated.photometry(("V", v))) == {"V": plain["johnson_v"]}
    assert magnitudes(evaluated.photometry(("V", str(v)))) == {"V": plain["johnson_v"]}


def test_photometry_takes_several_name_path_tuples(evaluated, bands):
    v, b = bands
    plain = magnitudes(evaluated.photometry(v, b))

    expected = {"V": plain["johnson_v"], "B": plain["johnson_b"]}
    assert magnitudes(evaluated.photometry(("V", v), ("B", b))) == expected
    assert magnitudes(evaluated.photometry((("V", v), ("B", b)))) == expected  # one tuple of specs


def test_a_lone_pair_of_names_stays_two_filters(bands):
    # ("V", "B") must not become filter "B" labelled "V": only a path makes a (name, path) spec.
    from sed_tools.models import _normalize_filter_sequence

    v, _ = bands
    assert _normalize_filter_sequence([("V", "B")]) == ["V", "B"]
    assert _normalize_filter_sequence([("V", str(v))]) == [("V", str(v))]
    assert _normalize_filter_sequence([("V", v)]) == [("V", v)]


def test_photometry_plain_paths_and_sequences_of_them_are_unchanged(evaluated, bands):
    v, b = bands
    plain = magnitudes(evaluated.photometry(str(v), str(b)))

    assert set(plain) == {"johnson_v", "johnson_b"}
    assert plain["johnson_v"] != plain["johnson_b"]
    assert magnitudes(evaluated.photometry(v, b)) == plain
    assert magnitudes(evaluated.photometry([str(v), str(b)])) == plain
    assert magnitudes(evaluated.photometry((v, b))) == plain  # Paths, so not a (name, path) spec
