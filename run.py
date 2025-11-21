#!/usr/bin/env python3
"""
Self-bootstrapping runner for spectraa tools.
- Installs/updates Python deps (no venv; uses --user when not in a venv)
- Adds a light prompt for the 'flux' tool so you don’t hit argparse errors
"""

import os, sys, subprocess, importlib, shutil, textwrap

REQUIRED = {
    "numpy":             "numpy>=1.22",
    "pandas":            "pandas>=1.5",
    "tqdm":              "tqdm>=4.66",
    "requests":          "requests>=2.31",
    "bs4":               "beautifulsoup4>=4.12",
    "lxml":              "lxml>=4.9",
    "h5py":              "h5py>=3.10",
    "astropy":           "astropy>=6.0",
    "astroquery":        "astroquery>=0.4.7",
    "matplotlib":        "matplotlib>=3.7",
}

TOOLs_DIR = os.path.abspath(os.path.dirname(__file__)) + '/sed_tools'

TOOLS = {
    "svo":        os.path.join(TOOLs_DIR, "svo_spectra_grabber.py"),
    "msg":        os.path.join(TOOLs_DIR, "msg_spectra_grabber.py"),
    "filters":    os.path.join(TOOLs_DIR, "svo_filter_grabber.py"),
    "regen":      os.path.join(TOOLs_DIR, "svo_regen_spectra_lookup.py"),
    "flux":       os.path.join(TOOLs_DIR, "precompute_flux_cube.py"),
    "lookfilter": os.path.join(TOOLs_DIR, "svo_spectra_filter.py"),
    #"fluxcube":   os.path.join(TOOLs_DIR, "flux_cube_tool.py"),
}

def in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV")
    )

def pip_install(spec: str) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    if not in_venv():
        cmd.append("--user")
    print(f"[pip] {' '.join(cmd)}")
    return subprocess.call(cmd)

def ensure_deps():
    print("Checking Python dependencies…")
    missing = []
    for import_name, spec in REQUIRED.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(spec)

    if not missing:
        print("All dependencies already present.")
        return

    print("Installing missing dependencies:")
    for spec in missing:
        print(f"  - {spec}")


    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if not in_venv():
        cmd.append("--user")
    cmd.extend(missing)
    print(f"[pip] {' '.join(cmd)}")
    resp = input("Install missing dependencies? [y/N] ").strip().lower()
    if resp in ("y", "yes"):
        rc = subprocess.call(cmd)
        if rc != 0:
            print("Bulk install failed; retrying one-by-one…")
            for spec in missing:
                rc_i = pip_install(spec)
                if rc_i != 0:
                    print(textwrap.dedent(
                        f"""
                        ERROR: Failed to install '{spec}'.
                        Some wheels (e.g., h5py/numpy) may need system headers:
                          - Debian/Ubuntu: sudo apt-get install python3-dev build-essential libhdf5-dev
                          - Fedora/RHEL:   sudo dnf install python3-devel gcc hdf5-devel
                          - macOS (brew):  brew install hdf5
                        """
                    ).strip())
                    sys.exit(1)

        for import_name in REQUIRED:
            importlib.import_module(import_name)
        print("Dependencies installed OK.\n")



def _maybe_prompt_flux(args):
    # If user didn’t pass --model_dir, give a quick prompt here so argparse doesn’t blow up.
    joined = " ".join(args) if args else ""
    if "--model_dir" in joined:
        return args
    ans = input("Enter --model_dir for flux cube (Enter to let the tool ask interactively): ").strip()
    if ans:
        args = list(args) if args else []
        args += ["--model_dir", ans]
    return args

def run_tool(tool_key: str, extra_args=None):
    script = TOOLS.get(tool_key)
    if not script or not os.path.exists(script):
        print(f"Tool '{tool_key}' not found at {script}")
        sys.exit(1)

    args = [sys.executable, script]
    if extra_args:
        args += extra_args

    if tool_key == "flux":
        args = [args[0], args[1], *(_maybe_prompt_flux(args[2:]))]

    print(f"\n[run] {' '.join(args)}\n")
    os.execv(sys.executable, args)

def main():
    ensure_deps()

    from sed_tools.cli import menu, run_filters_flow, run_rebuild_flow, run_spectra_flow
    from sed_tools import STELLAR_DIR_DEFAULT, FILTER_DIR_DEFAULT

    argv = sys.argv[1:]
    if argv:
        sub = argv[0].lower()
        if sub in ("quit", "exit"):
            sys.exit(0)
        if sub in TOOLS:
            run_tool(sub, argv[1:])
        else:
            print(f"Unknown subcommand '{sub}'. Valid: {', '.join(TOOLS)}")
            sys.exit(2)

    while True:
        choice = menu()

        if choice == "filters":
            run_filters_flow(base_dir=FILTER_DIR_DEFAULT)
        elif choice == "rebuild":
            run_rebuild_flow(base_dir=STELLAR_DIR_DEFAULT)
        elif choice == "spectra":
            run_spectra_flow(source="all",
                             base_dir=STELLAR_DIR_DEFAULT,
                             models=None,
                             workers=5,
                             force_bundle_h5=True,
                             build_flux_cube=True)
        #elif choice == "fluxcube":
        #    run_tool("fluxcube")
        else:
            exit()


if __name__ == "__main__":
    main()
