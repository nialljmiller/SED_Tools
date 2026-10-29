"""Old import path; the code is in sed_tools/filters/compose.py."""

from . import _old_path
from .filters import compose as _module

# What `from sed_tools.filter_components import *` gave in 0.5.0.
__all__ = """
ATMOSPHERE ATMOSPHERE_KINDS AUXILIARY_DIRS AtmosphereGrid AtmosphereSpec COMPONENTS_DIR COMPOSABLE_ROLES
ComponentCoverageError ComponentError Curve DERIVED Dict ENERGY FILTER FILTER_SIBLING_SUFFIXES
INSTRUMENT InstrumentManifest Iterable List MANIFEST_NAME MANIFEST_SCHEMA Mapping OPTICS Optional PHOTON
PRODUCTS_DIR PUBLISHED Path PathLike QE ROLES SHARED Sequence TEAM Tuple Union add_component annotations
archived_bands asdict atmosphere_file atmosphere_grid build_all_products build_product compose_band
dataclass deepcopy field find_atmosphere index_listing instrument_sites is_auxiliary_path
is_ground_based json logger logging math_isclose multiply_curves normalize_detector_type np os own_site
parse_components platform product_detector_type product_name re read_curve read_filter_header
recipe_name resolve_atmosphere tempfile write_filter_file
""".split()

_old_path.forward(
    __name__, _module,
    asdict="dataclasses:asdict", deepcopy="copy:deepcopy", field="dataclasses:field", json="json", os="os", re="re")
