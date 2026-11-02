# Golden tests

`tests/test_golden.py` reruns every case in `cases.py` on the fixed inputs in
`inputs/` and requires the outputs to match `expected/` byte for byte. The
expected outputs were recorded from integration/0.4.2 (d622dcc), so any
refactor that changes a result, even in the last digit, fails.

The cases cover filter headers, filter properties (including A(λ)/A(V), which
needs `dust_extinction`), plotting colours, component composition with an
atmosphere and with QE variants, combining filter sets, cleaning a spectral
grid, building and densifying a flux cube, regenerating a lookup table,
interpolation and synthetic photometry (AB and Vega), band averages, writing a
filtered grid, combining two grids, the MESA export, and the public API
signatures and CLI help.

They do not cover anything that needs the network: the NJM, SVO, MSG and MAST
downloaders, the original-source builders' downloads, or SkyCalc.

**Running.** The outputs depend on the platform and on the numpy and scipy
versions, which are stored in `expected/environment.json`. On any other
environment the cases skip and say why.

**Rules.**
- A failure means the code now gives a different answer. Find out why; do not
  edit `expected/`.
- `record.py` refuses to overwrite a recorded case and refuses to record a case
  whose two runs differ. Re-recording is a deliberate decision, made only by
  the maintainer.
