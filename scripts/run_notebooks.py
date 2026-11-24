"""Execute the notebooks in jupyter_notebooks/ against this checkout, in a scratch HOME: a manual check before a release.

    python scripts/run_notebooks.py                      # every notebook, in a new scratch HOME
    python scripts/run_notebooks.py --home DIR           # reuse DIR (its downloads are kept between runs)
    python scripts/run_notebooks.py 04 06                # only the notebooks whose names start with these
    python scripts/run_notebooks.py --write-outputs      # also store the outputs in the notebooks

Needs nbclient, nbformat and ipykernel (``pip install nbclient ipykernel``) and the package's ``full`` extra; none is
a dependency of SED_Tools. The notebooks download about 150 MB (bbody, Husfeld, part of Kurucz2003all, two filter
sets and Vega) into the scratch HOME's data directory; nothing outside it is written. Each notebook runs with the
scratch HOME as its home, ``sed_tools`` imported from this checkout, and ``sed-tools`` from this interpreter's
folder. Every cell runs, also after an error; the run prints each failing cell and exits 1 if any failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "jupyter_notebooks"
PROBE = ("import os, sed_tools\n"
         "assert sed_tools.__file__.startswith({repo!r}), sed_tools.__file__\n"
         "assert os.path.expanduser('~') == {home!r}, os.path.expanduser('~')")


def scratch_environment(home: Path) -> None:
    """Point this process (and so the kernels it starts) at ``home`` and this checkout."""
    for name in ("SED_DATA_DIR", "SED_STELLAR_DIR", "SED_FILTER_DIR", "SED_VEGA_SPECTRUM", "MPLBACKEND"):
        os.environ.pop(name, None)
    bin_dir = str(Path(sys.executable).parent)
    os.environ.update(HOME=str(home), JUPYTER_PLATFORM_DIRS="1", JUPYTER_DATA_DIR=str(home / "jupyter"),
                      PYTHONPATH=os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])]),
                      PATH=os.pathsep.join([bin_dir, os.environ["PATH"]]))
    # The kernel is this interpreter, whatever kernels the user has installed.
    spec = home / "jupyter" / "kernels" / "python3"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3", "language": "python"}))


def terminal_text(text: str) -> str:
    """Stream text as a terminal shows it: of each line rewritten with carriage returns, the last version."""
    return "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))


def merged_streams(outputs):
    """The outputs with each run of same-stream text joined (progress bars arrive in pieces), as a terminal shows it."""
    merged = []
    for out in outputs:
        if out.output_type == "stream" and merged and merged[-1].output_type == "stream" \
                and merged[-1].name == out.name:
            merged[-1].text += out.text
        else:
            merged.append(out)
    for out in merged:
        if out.output_type == "stream":
            out.text = terminal_text(out.text)
    return merged


def run(path: Path, home: Path, timeout: int, write_outputs: bool):
    """Execute one notebook; (seconds, [(cell number, error)])."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    probe = nbformat.v4.new_code_cell(PROBE.format(repo=str(REPO), home=str(home)))
    nb.cells.insert(0, probe)
    work = home / "work" / path.stem
    work.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(work)}})
    start = time.time()
    client.execute()
    seconds = time.time() - start
    nb.cells.pop(0)
    errors = [(n, f"{out.ename}: {out.evalue}".splitlines()[0])
              for n, cell in enumerate(nb.cells) if cell.cell_type == "code"
              for out in cell.outputs if out.output_type == "error"]
    if any(out.output_type == "error" for out in probe.outputs):
        errors.insert(0, (-1, "the probe cell failed: sed_tools or HOME is not the one this run set up"))
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code":
            cell.outputs = merged_streams(cell.outputs)
    executed = home / "executed" / path.name
    executed.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, executed)
    if write_outputs and not errors:
        nbformat.write(nb, path)
    return seconds, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefixes", nargs="*", help="run only notebooks whose names start with these")
    parser.add_argument("--home", type=Path, help="scratch HOME to use (default: a new folder, left in place)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per cell (default 1800)")
    parser.add_argument("--write-outputs", action="store_true",
                        help="store the outputs in each notebook that ran without an error")
    args = parser.parse_args(argv)
    try:
        import ipykernel  # noqa: F401
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except ImportError as missing:
        parser.error(f"{missing.name} is not installed: pip install nbclient ipykernel")

    home = (args.home or Path(tempfile.mkdtemp(prefix="sed_tools_notebooks_"))).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    scratch_environment(home)
    paths = [p for p in sorted(NOTEBOOKS.glob("*.ipynb"))
             if not args.prefixes or any(p.name.startswith(prefix) for prefix in args.prefixes)]
    print(f"Scratch HOME: {home}  (left in place; executed copies in {home / 'executed'})")
    failed = 0
    for path in paths:
        seconds, errors = run(path, home, args.timeout, args.write_outputs)
        print(f"{path.name}: {seconds:.0f} s, {'ok' if not errors else f'{len(errors)} failing cell(s)'}")
        for n, error in errors:
            print(f"    cell {n}: {error}")
        failed += bool(errors)
    print(f"{len(paths) - failed} of {len(paths)} notebooks ran without an error")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
