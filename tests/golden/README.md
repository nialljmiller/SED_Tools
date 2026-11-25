# Golden tests

`tests/test_golden.py` reruns every case in `cases.py` on the fixed inputs in
`inputs/` and requires the outputs to match `expected/` byte for byte, so any
refactor that changes a result, even in the last digit, fails. The expected
outputs were first recorded from integration/0.4.2 (d622dcc); a case whose
output a change meant to alter was re-recorded in the commit that explains the
change (`git log -- tests/golden/expected/<case>` lists them).

The cases cover filter headers, filter properties (including A(λ)/A(V), which
needs `dust_extinction`), plotting colours, component composition with an
atmosphere and with QE variants, combining filter sets, cleaning a spectral
grid, building and densifying a flux cube, regenerating a lookup table,
interpolation and synthetic photometry (AB and Vega), band averages, writing a
filtered grid, combining two grids, the MESA export, and the public API
signatures and CLI help.

They do not cover anything that needs the network: the NJM, SVO, MSG and MAST
downloaders, the original-source builders' downloads, or SkyCalc.

**Running.** The last bits of some outputs depend on the platform and on the
numpy and scipy versions; the environment they were recorded on is stored in
`expected/environment.json`. There every file must match byte for byte. On any
other environment (CI runs Linux with Python 3.10 and 3.12) every file must
still match byte for byte except those listed in `TOLERANT` in
`tests/test_golden.py` (flux cubes, interpolated spectra, band quantities):
their headers and text must be identical and their numbers equal to within a
relative 1e-9 (`RTOL`). Linux differs from the recording by at most 1.7e-14,
while the smallest error the cases must catch (the AB zero point 3631 Jy read
as 3630 Jy) moves a magnitude by 8e-6. `public_interface` records Python's own
text (argparse help and `inspect` signatures), which differs between Python
versions, so it runs only on the recorded Python version.

**Rules.**
- A failure means the code now gives a different answer. Find out why; do not
  edit `expected/`.
- `record.py` refuses to overwrite a recorded case and refuses to record a case
  whose two runs differ. Re-recording is a deliberate decision, made only by
  the maintainer.
