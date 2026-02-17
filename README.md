# SED_Tools — Stellar Atmosphere Data Archive

This repository backs the **SED_Tools data server**: a hosted archive of stellar atmosphere models and photometric filter profiles, pre-processed for direct use with MESA’s `colors` module (and general synthetic photometry workflows).

The live archive is served from:

http://nillmill.ddns.net/sed_tools

It provides:

- Pre-built stellar atmosphere libraries
- Flux cubes + lookup tables
- Filter transmission curves
- A machine-readable `index.json`
- A simple HTML browser

This is intended to remove duplicated effort across projects and groups.

---

## What’s on the Server

### Stellar Atmosphere Models

Each model directory contains (when complete):

```

ModelName/
├── flux_cube.bin
├── lookup_table.csv
├── ModelName.h5
└── *.txt   (individual spectra)

```

These can be copied directly into:

```

$MESA_DIR/colors/data/stellar_models/

````

Then referenced in your MESA inlist as:

```fortran
stellar_atm = 'ModelName/'
````

---

### Photometric Filters

Filters are organised by facility and instrument:

```
filters/
└── Facility/
    └── Instrument/
        ├── Filter1.dat
        ├── Filter2.dat
        └── Instrument
```

Copy into:

```
$MESA_DIR/colors/data/filters/
```

And reference in MESA as:

```fortran
instrument = 'Facility/Instrument'
```

---

## Machine Index

The archive automatically generates:

```
data/index.json
```

This file contains:

* Available stellar models
* Spectrum counts
* File sizes
* Flux cube / lookup availability
* Filter facilities + instruments
* Total archive statistics

If you’re building pipelines, scrape this instead of guessing directory contents.

Example:

```bash
curl http://nillmill.ddns.net/sed_tools/index.json
```

---

## Regenerating the Index (Server Side)

On the server:

```bash
python generate_index.py
```

This scans:

```
data/stellar_models/
data/filters/
```

and regenerates `index.json` automatically.

---

## Purpose

This archive exists to:

* Centralise stellar SED resources
* Avoid repeated atmosphere downloads across groups
* Provide MESA-ready datasets
* Enable reproducible synthetic photometry
* Support automated pipelines via JSON

It is intentionally simple: static files + generated index.

No accounts. No API keys. Just data.

---

## Repository Structure

```
SED_Tools/
├── data/                # Server archive (models + filters)
├── generate_index.py   # Builds index.json
├── index.html          # Browser UI
├── sed_tools/          # Python package (optional client tooling)
└── README.md
```

---

## Notes

* This server lives in a garage in Wyoming.
* Bandwidth is finite.
* Please don’t hammer it unnecessarily.
* Mirrors welcome.

---

## Links

GitHub:
[https://github.com/nialljmiller/SED_Tools](https://github.com/nialljmiller/SED_Tools)

Raw index:
[http://nillmill.ddns.net/sed_tools/index.json](http://nillmill.ddns.net/sed_tools/index.json)

Archive browser:
[http://nillmill.ddns.net/sed_tools](http://nillmill.ddns.net/sed_tools)

---

Maintained by Niall J. Miller.
