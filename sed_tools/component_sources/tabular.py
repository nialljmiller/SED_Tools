"""Original filter scans with independently enumerable tabular sources.

These are measured *filters*, not a claim that the full optical path is known.
The table lists are explicit so a new upstream file cannot silently change a
band, unit conversion, or observing mode. Multi-band or multi-temperature
tables carry separate named channels.
"""

from __future__ import annotations

from pathlib import Path

from ._common import SimpleSource, Table, build_simple

ESO = "https://www.eso.org/sci/facilities/paranal/instruments"


def _tables(path, names, *, wl_scale, tr_scale, suffix=".dat"):
    return {name: Table(f"{ESO}/{path}/{name}{suffix}", wl_scale=wl_scale,
                        tr_scale=tr_scale) for name in names}


HAWKI_NAMES = (
    "BrGamma", "CH4", "H", "H2", "J", "Knew", "Kold", "NB1060",
    "NB1190", "NB2090", "Y",
)
HAWKI = SimpleSource(
    facility="Paranal", instrument="HAWKI",
    filters={name: Table(f"{ESO}/hawki/inst/filters/hawki_{name}.dat",
                         wl_scale=10, tr_scale=0.01,
                         note="cryogenic filter scan; see original header") for name in HAWKI_NAMES},
    published_from_filters=True, default_site="paranal",
    references={"eso": f"{ESO}/hawki/inst/filters"},
    notes="ESO cryogenic filter transmissions only; Knew and Kold retain the source's separate labels. No detector QE or instrument optics are included.",
)

NACO_NAMES = (
    "H_CONICA", *(f"IB_{v}" for v in ("2.00", "2.03", "2.06", "2.09", "2.12",
                                       "2.15", "2.18", "2.21", "2.24", "2.27",
                                       "2.30", "2.33", "2.36", "2.39", "2.42",
                                       "2.45", "2.48")),
    "J_CONICA", "Ks_CONICA", "L_prime", "M_prime",
    *(f"NB_{v}" for v in ("1.04", "1.08", "1.09", "1.24", "1.26", "1.28",
                              "1.64", "1.75", "2.12", "2.17", "3.74")),
)
NACO = SimpleSource(
    facility="Paranal", instrument="NACO",
    filters=_tables("naco/inst/filters", NACO_NAMES, wl_scale=10000, tr_scale=1),
    published_from_filters=True, default_site="paranal",
    references={"eso": f"{ESO}/naco/inst/filters.html"},
    notes="ESO CONICA filter transmissions in microns. NB_4.05 is excluded: its file labels the wavelength axis as inverse centimetres but its values conflict with the named band.",
)

IRDIS_SINGLE = ("B_H", "B_J", "B_Ks", "B_Y", "N_BrG", "N_CO", "N_CntH",
                "N_CntJ", "N_CntK1", "N_CntK2", "N_FeII", "N_H2", "N_HeI", "N_PaB")
IRDIS_DUAL = {"H23": ("H2", "H3"), "H34": ("H3", "H4"),
              "J23": ("J2", "J3"), "K12": ("K1", "K2"), "Y23": ("Y2", "Y3")}
_IRDIS_PATH = "sphere/inst/filters"
_irdis = {name: Table(f"{ESO}/{_IRDIS_PATH}/SPHERE_IRDIS_{name}.dat", wl_scale=10)
          for name in IRDIS_SINGLE}
for pair, channels in IRDIS_DUAL.items():
    url = f"{ESO}/{_IRDIS_PATH}/SPHERE_IRDIS_D_{pair}.dat"
    for column, channel in enumerate(channels, start=1):
        _irdis[f"D_{pair}_{channel}"] = Table(url, wl_scale=10, cols=(0, column),
                                             note=f"{pair}, channel {channel}")

SPHERE = SimpleSource(
    facility="Paranal", instrument="SPHERE", filters=_irdis,
    published_from_filters=True, default_site="paranal",
    references={"eso": f"{ESO}/{_IRDIS_PATH}.html"},
    notes="ESO SPHERE/IRDIS filter transmissions, including separately indexed dual-band channels. ZIMPOL, neutral-density and instrument optics remain unmodelled; these curves are filter-only.",
)

GEMINI = "https://www.gemini.edu/sciops/instruments"
GSAOI_FILES = (
    "CH4long_ED151", "CH4short_ED144", "CO_ED187", "FeII_ED150",
    "H2O_ED188", "H2_1-0_ED162", "H2_2-1_S1_ED164", "HIBrGamma_ED166",
    "HIPaBeta_ED231", "HIPaGamma_ED186", "H_ED169-1", "Hcont_ED141",
    "HeI2p2s_ED165", "HeI_ED185", "J_ED191-1", "Jcont_ED190",
    "K_ED192-1", "Klong_cont_ED163", "Kprime_ED278-2", "Kshort_ED196-1",
    "Kshort_cont_ED168", "Z_ED205-1",
)
GSAOI = SimpleSource(
    facility="Gemini", instrument="GSAOI",
    filters={name: Table(f"{GEMINI}/gsaoi/data/{name}.dat", wl_scale=10,
                         tr_scale=0.01, note="measured filter transmission")
             for name in GSAOI_FILES},
    published_from_filters=True,
    references={"gemini": "https://www.gemini.edu/instrumentation/gsaoi/components"},
    notes="Gemini GSAOI filter scans in nm and percent; optical train and detector QE are not supplied here. No site atmosphere is assumed.",
)

_nifs = {}
for name in ("HK", "JH", "ZJ"):
    url = f"{GEMINI}/nifs/{name}.dat"
    for temperature, column in (("298K", 1), ("77K", 2)):
        _nifs[f"{name}_{temperature}"] = Table(url, wl_scale=10000,
                                                tr_scale=0.01, cols=(0, column),
                                                note=f"witness measurement at {temperature}")
NIFS = SimpleSource(
    facility="Gemini", instrument="NIFS", filters=_nifs,
    published_from_filters=True,
    references={"gemini": "https://www.gemini.edu/instrumentation/nifs/components"},
    notes="Gemini NIFS ZJ/JH/HK witness filter scans. The 298 K and 77 K columns are separate variants; no optics, QE or site atmosphere is inferred.",
)

SOURCES = {spec.instrument: spec for spec in (HAWKI, NACO, SPHERE, GSAOI, NIFS)}
RAW_GROUPS = {
    "HAWKI": "eso2/eso_hawki", "NACO": "eso/eso_naco", "SPHERE": "eso/eso_sphere",
    "GSAOI": "pages2/pg_www_gemini_edu_instrumentation_gsaoi_components",
    "NIFS": "pages2/pg_www_gemini_edu_instrumentation_nifs_components",
}


def build(filter_root, *, instrument, raw=None, session=None, **kwargs) -> Path:
    """Build one instrument; ``raw`` may hold the saved research inventory."""
    spec = SOURCES[instrument]
    if raw is not None:
        from requests import Response, Session

        class SavedSession:
            def __init__(self, root, fallback):
                self.root, self.fallback = Path(root), fallback

            def get(self, url, **options):
                filename = url.rsplit("/", 1)[-1]
                path = self.root / RAW_GROUPS[instrument] / filename
                if path.is_file():
                    response = Response()
                    response.status_code = 200
                    response.url = url
                    response._content = path.read_bytes()
                    return response
                return self.fallback.get(url, **options)

        session = SavedSession(raw, session or Session())
    return build_simple(spec, filter_root, session=session, **kwargs)
