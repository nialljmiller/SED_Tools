# Citing

## SED_Tools

Please cite the archived software through Zenodo. The concept DOI covers all
versions and resolves to the latest release:

- all versions: [10.5281/zenodo.18321943](https://doi.org/10.5281/zenodo.18321943)

Each release also has its own DOI, listed on the Zenodo record under
"Versions"; cite that one to name the exact version you used (0.6.0:
[10.5281/zenodo.24117340](https://doi.org/10.5281/zenodo.24117340)).
Machine-readable metadata is in
[`CITATION.cff`](https://github.com/nialljmiller/SED_Tools/blob/main/CITATION.cff),
which GitHub turns into a "Cite this repository" entry.

SED_Tools is the data-preparation pipeline for MESA's `colors` module; see
[colors/README.rst](https://github.com/MESAHub/mesa/blob/main/colors/README.rst)
in the MESA repository.

## The data

Filter curves and spectral grids belong to the teams that published them.
Please cite them as well, and give the acknowledgements they ask for. For
filters, NJM's catalogue holds the references, the acknowledgement texts and
the licence of each instrument:

```python
from sed_tools.api import Filters

print(Filters.cite("GAIA/GAIA", "2MASS/2MASS"))                 # text: acknowledgements, licences, references
print(Filters.cite("GAIA/GAIA", format="bibtex"))              # a .bib file
refs = Filters.citations("Generic/Johnson")                     # the same as a dict
```

```bash
sed-tools filters-info GAIA/GAIA        # licence and the number of references
```

For the services and grids SED_Tools reads, cite:

- the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/)
  (Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024) for filters from SVO;
- the [SVO Theory Server](https://svo2.cab.inta-csic.es/theory/newov2/) and
  each grid's own paper for SVO grids;
- Rich Townsend's [MSG](https://madstar.richtownsend.dev/static.php?ref=msg-grids)
  grids and the papers of the underlying models;
- the [BOSZ](https://archive.stsci.edu/prepds/bosz/) library (MAST) for BOSZ
  grids;
- [dust_extinction](https://doi.org/10.21105/joss.07023) (Gordon 2024) and
  Fitzpatrick (1999) when you use `AfAv`.
