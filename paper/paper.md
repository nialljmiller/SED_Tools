---
title: 'SED_Tools: A Unified Pipeline for Acquiring, Standardizing, and Interpolating Stellar Atmosphere Spectral Energy Distributions'
tags:
  - Python
  - astronomy
  - stellar astrophysics
  - spectral energy distributions
  - stellar atmospheres
  - MESA
authors:
  - name: Niall J. Miller
    orcid: 0000-0002-3780-0592
    affiliation: 1
  - name: Meredith Joyce
    orcid: 0000-0002-8717-127X
    affiliation: 1
affiliations:
  - name: "University of Wyoming, 1000 E University Ave, Laramie, WY, United States"
    index: 1
date:
bibliography: paper.bib
---

# Summary

`SED_Tools` is an open-source Python package for acquiring, validating, and standardizing stellar atmosphere spectral energy distribution (SED) grids from public archives.
It converts heterogeneous spectra and filter curves into common per-model files, HDF5 archives, binary flux cubes, and lookup tables for use by stellar-evolution and synthetic-photometry software.
Supported sources include the Spanish Virtual Observatory (SVO), the MAST/BOSZ synthetic library [@meszaros2024bosz], and stellar atmosphere grids distributed by the MSG (Multidimensional Spectral Grids) project [@townsend2023], including model families such as Kurucz/ATLAS9 [@castelli2003], BT-Settl, TLUSTY, and TMAP.
The package provides command-line and Python interfaces for acquisition, grid construction, interpolation-based densification, coverage diagnostics, and import of externally sourced grids.
Two PyTorch-based components can extend incomplete wavelength coverage or generate an SED from (*T*eff, log *g*, [M/H]).
Centralized header parsing and unit normalization produce a common representation across the supported source archives.

# Statement of need

Stellar-evolution and synthetic-photometry codes require precomputed stellar atmosphere spectra spanning effective temperature, surface gravity, metallicity, and, in some cases, additional physical parameters.
Available grids are distributed across independent archives that use different file formats, header conventions, wavelength and flux units, wavelength sampling, and axis ordering.
Some archives encode parameters such as sedimentation efficiency, helium fraction, or alpha enhancement as separate files rather than explicit grid dimensions, which can produce duplicate or colliding nodes during grid construction.
Combining grids from different sources therefore requires archive-specific parsing, unit conversion, collision handling, and validation, and errors in these steps can propagate into downstream interpolation and synthetic photometry.
`SED_Tools` automates the acquisition of spectra and filter curves, validates inputs against expected physical ranges, resolves hidden-axis collisions, and writes standardized products as per-model spectra, HDF5 archives, binary flux cubes, and lookup tables.
The package is intended for astronomers and scientific-software developers preparing atmosphere libraries for stellar evolution, synthetic photometry, SED fitting, or population-synthesis workflows.
It directly supports the MESA `colors` module, whose documentation identifies `SED_Tools` as the data-preparation tool for its standard input products [@mesa_colors_readme].

# State of the field

Several existing packages address related parts of this workflow.
`synphot` and its predecessor `pysynphot` perform synthetic photometry using prepared source spectra and bandpasses, but their documented scope does not include acquisition and standardization of heterogeneous atmosphere archives into interpolation-ready grids [@synphot; @pysynphot].
`specutils` provides general-purpose representations and analysis tools for spectroscopic data, rather than an archive-to-grid construction pipeline [@specutils].
`gollum` provides programmatic access to selected precomputed synthetic grids and tools for spectral inspection and model-data comparison [@shankar2024].
MSG provides compiled libraries and associated tools for interpolating, converting, and packaging stellar spectral grids in its HDF5 data model [@townsend2023].
`SED_Tools` instead focuses on archive-specific acquisition, header and unit standardization, hidden-axis collision handling, validation, and production of multiple downstream representations.
Archive services such as the SVO Filter Profile Service [@rodrigo2020] and MAST provide access to source data, while preparation for a particular interpolation or stellar-evolution workflow remains a downstream task.
The principal software contribution of `SED_Tools` is to combine these acquisition and standardization steps in a reproducible, versioned pipeline spanning SVO, MAST/BOSZ, MSG-hosted grids, and externally supplied models.

# Software design

`SED_Tools` centralizes transformations that must remain consistent across source archives.
A header-parsing module resolves historical header-key variants against a common alias table and prevents a later unparseable value from overwriting an already resolved value.
A spectra-cleaning module converts each spectrum once to a common wavelength and flux representation (angstroms; erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$), after which downstream stages operate on the normalized data.

A collision-handling module addresses physical parameters that are encoded as separate files rather than explicit grid dimensions.
It can construct a mean flux cube, preserve each colliding variant in a separate cube, or select a configured variant.
This retains the information associated with hidden axes while making the chosen collision policy explicit and reproducible for each model family.

Large expanded grids are constructed using memory-mapped source and destination arrays rather than requiring the full dataset to reside in memory.
Core operations are exposed through both the `sed-tools` command-line interface and the Python API, allowing the same processing stages to be used interactively, from scripts, or in shell-based pipelines.
Coverage diagnostics report the sampled parameter space before interpolation, while the optional ML modules extend wavelength coverage or generate spectra from atmospheric parameters.
Malformed inputs raise errors rather than being silently skipped.

Validation is performed at two levels.
An automated `pytest` suite exercises configuration, standardization utilities, the command-line interface, and flux-cube loading and evaluation.
A separate audit pipeline applies physical consistency checks to standardized products by comparing spectral peak locations with Wien-law expectations, comparing integrated fluxes with blackbody reference integrals over the available wavelength range, and matching raw spectra to their corresponding flux-cube nodes.

# Research impact statement

`SED_Tools` is the designated data-preparation pipeline for the MESA `colors` module, which was first shipped in release candidate r25.10.1-rc1 and is included in stable MESA releases from r25.12.1 onward [@mesa_release_r25101rc1].
MESA is an open-source stellar-evolution code used across asteroseismology, stellar-population studies, and binary-evolution research [@paxton2011; @paxton2013; @paxton2015; @paxton2018; @paxton2019; @jermyn2023].
The `colors` module documentation names `SED_Tools` as the tool that prepares its input data and identifies its precomputed flux-cube products as generated by `SED_Tools` [@mesa_colors_readme].
The Kurucz2003 atmosphere grids distributed for the module were also prepared with `SED_Tools`.
Consequently, users of the standard atmosphere products distributed for the MESA `colors` module consume data products prepared by this package even when they do not run `SED_Tools` directly.

# AI usage disclosure

OpenAI Codex and Claude Code assisted with formatting during a codebase refactor, construction of portions of the automated test suite, and investigation of flux-cube construction performance.
Claude and OpenAI ChatGPT assisted with drafting and editing portions of this manuscript.
All generated code and prose were reviewed by the authors, and code changes were checked using the automated test suite and relevant numerical validation procedures before merging.

# Acknowledgements

Computations were performed using the University of Wyoming (UW) Advance Research Computing Center MedicineBow HPC, a UW managed computational resource available to UW researchers including faculty, staff, students, and collaborators (<https://doi.org/10.15786/M2FY47>).
This research has made use of the SVO Filter Profile Service "Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00 [@rodrigo2020].
This research has also made use of theoretical stellar spectra provided through the Spanish Virtual Observatory (<https://svo.cab.inta-csic.es>).
This work has made use of data obtained from the Mikulski Archive for Space Telescopes (MAST), operated by the Space Telescope Science Institute (STScI), which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS5-26555.
This work makes use of stellar atmosphere grids packaged and distributed by the MSG project [@townsend2023], and we thank R. H. D. Townsend for maintaining this resource.
The `SED_Tools` data mirror is currently hosted on computing infrastructure operated by N. J. Miller.

# References
