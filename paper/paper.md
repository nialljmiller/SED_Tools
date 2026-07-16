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
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Meridith Joyce
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: "REPLACE WITH INSTITUTION AND DEPARTMENT"
    index: 1
  - name: "REPLACE WITH INSTITUTION AND DEPARTMENT"
    index: 2
date: 13 July 2026
bibliography: paper.bib
---

# Summary

`SED_Tools` is an open-source Python package that downloads, validates, and standardizes stellar
atmosphere spectral energy distribution (SED) grids from heterogeneous public archives, and
converts them into the binary flux-cube and lookup-table format required for rapid runtime
interpolation by stellar evolution codes. It ships a command-line interface (`sed-tools`) and a
matching Python API (`SED`, `Catalog`, `Spectrum`, `Filters`, `CatalogInfo`) that together cover
the full pipeline: spectra acquisition, filter transmission-curve acquisition and combination,
flux-cube construction and rebuilding, grid densification via interpolation, target-directory
preparation for stellar evolution codes, coverage diagnostics, and import of externally sourced
grids. Supported source archives include the Spanish Virtual Observatory (SVO), the MAST/BOSZ
synthetic library [@meszaros2024bosz], and the Munari Stellar Grid (MSG) collection, spanning
model families such as Kurucz/ATLAS9 [@castelli2003], NextGen/NextGen2, BT-Settl, TLUSTY
(OSTAR2002/BSTAR2006), TMAP, GRAMS, SPHINX, Husfeld, and blackbody references. Two PyTorch-based
components extend the tool beyond format conversion: an SED completer that extends incomplete
wavelength coverage, and an SED generator that synthesizes an SED directly from
(*T*eff, log *g*, [M/H]) without requiring an existing grid node. All standardization funnels
through two single-authority modules — a header parser and a unit-cleaning routine — that
guarantee a consistent output representation regardless of the conventions used by the source
archive.

# Statement of need

Stellar evolution and synthetic-photometry codes need pre-computed grids of stellar atmosphere
spectra spanning effective temperature, surface gravity, and metallicity. In practice, these
grids are scattered across independent archives that differ in file formats, header-keyword
conventions, flux and wavelength units, wavelength sampling, and axis conventions; some archives
additionally encode extra physical axes — sedimentation efficiency, helium fraction, alpha
enhancement — as separate files rather than as grid dimensions, which silently produces
duplicate or colliding nodes if not accounted for. A researcher who wants to combine, for
example, a hot-star TLUSTY grid with a cool-star BT-Settl grid for a single evolutionary track
must hand-reconcile all of these inconsistencies before any interpolation is possible, and
errors introduced during that reconciliation — unit mismatches, axis-order bugs, double-counted
nodes — are difficult to detect after the fact and can silently bias downstream synthetic
photometry rather than produce an obvious failure. This is not a hypothetical risk: during
`SED_Tools`'s own development, a read/write axis-order mismatch in the flux-cube binary format
produced exactly this kind of silent corruption, one that would have propagated undetected into
every downstream synthetic photometric calculation had it not been caught by dedicated
validation. `SED_Tools` exists to remove this class of manual, error-prone reconciliation from
the workflow entirely. It automates the download of raw grids and filter curves, validates each
file against its expected physical ranges, resolves duplicate or colliding grid nodes introduced
by hidden axes, and writes a single standardized product — as flat per-model spectral files, an
HDF5 archive, and a pre-computed binary flux cube with an accompanying lookup table — that a
stellar evolution code can consume directly, with no additional user-side reformatting. This
directly serves the MESA `colors` module, which requires exactly this product and provides no
other supported route to generating it, but the standardized output is equally usable by any
downstream synthetic photometry, spectral energy distribution fitting, or population synthesis
workflow that needs a self-consistent, science-ready stellar atmosphere grid.

# State of the field

Several existing tools address related but distinct problems. `synphot`/`pysynphot` (STScI)
perform synthetic photometry given an already-prepared SED and bandpass, but do not download,
standardize, or reconcile heterogeneous source archives into interpolation-ready grids.
`specutils`, part of the Astropy ecosystem, provides general-purpose spectral-object handling and
analysis but is not oriented toward multi-source stellar atmosphere grid curation or production
of runtime-optimized binary grids for external codes; it operates one spectrum at a time rather
than as a grid-construction pipeline. MSG ships its own compiled grid-access library together
with a defined set of packaged grids, but does not provide a general mechanism for ingesting and
standardizing grids sourced from other archives such as SVO or MAST/BOSZ — a user who wants an
MSG-format grid built from BOSZ or SVO data has no supported path to do so without custom
scripting. Archive-side services such as the SVO Filter Profile Service [@rodrigo2020] and MAST
provide raw data access but explicitly leave unit standardization, header reconciliation,
collision handling, and target-code packaging to the user. To the authors' knowledge, no existing
public tool spans SVO, MAST/BOSZ, and MSG-hosted grids simultaneously behind a single collision-
aware standardization layer. `SED_Tools`'s distinguishing contribution is treating grid
acquisition, unit and header standardization, hidden-axis collision resolution, and stellar-
evolution-code-ready packaging as a single reproducible, versioned pipeline spanning multiple
source archives at once, rather than as a one-off per-grid conversion script maintained privately
by individual research groups — which is the status quo this package was built to replace. The
machine-learning-based SED completion and generation components are also, to the authors'
knowledge, not offered by any comparable publicly available tool in this space.

# Software design

`SED_Tools` separates concerns along two axes: a data-flow axis and an interface axis. On the
data-flow axis, two modules act as sole authorities over specific transformations, so that
behavior does not depend on call order or on which grid is processed first. A header-parsing
module resolves the many historical header-key spellings used across archives against a single
alias table, using a nan-guard so that a later, unparseable value can never silently overwrite an
already-resolved one. A spectra-cleaning module performs the one-time conversion of every
spectrum to a common wavelength/flux unit system (angstroms; erg s⁻¹ cm⁻² Å⁻¹), after which no
downstream stage is permitted to renormalize — this constraint is enforced by design rather than
by convention, so that a unit bug can only be introduced, and only needs to be fixed, in one
place. Hidden physical axes that some archives encode as separate files rather than as grid
dimensions are resolved by a dedicated collision-handling module that builds both a mean flux
cube and a library of per-variant cubes, so that information distinguishing the colliding
variants is preserved rather than discarded during standardization; the on-collision strategy
(average everything into one cube, keep every variant, or filter to a specific variant) is a
per-model, reproducible configuration rather than an implicit default. Large grids — the BT-Settl
grid alone occupies roughly 174 GiB once expanded to a common wavelength grid — are handled via
memory-mapped array allocation for both source and destination flux arrays, rather than in-RAM
allocation, specifically to avoid exhausting system memory during cube construction. On the
interface axis, every capability is available identically through the `sed-tools` command-line
interface and the Python `SED`/`Catalog`/`Spectrum` API, so the package is usable both as an
interactive or scriptable library and as a stage in shell-based data pipelines. A dedicated
coverage-diagnostics command lets a user inspect the completeness of a standardized grid in
parameter space before relying on it for interpolation, and the ML-based completer and generator
modules provide a fallback in regions where empirical grid coverage is insufficient. Error
handling throughout the pipeline is intentionally non-defensive: malformed input is designed to
raise immediately rather than be silently caught and skipped, so that data-quality problems
surface at the point of ingestion rather than as a difficult-to-trace downstream artifact in a
stellar model.

Correctness is checked at two levels. An automated `pytest` suite exercises the standardization,
configuration, CLI, and flux-cube-construction code paths directly. Separately, a dedicated audit
pipeline performs physically motivated consistency checks on the standardized output itself:
comparing each spectrum's flux-weighted peak wavelength against the Wien's-law prediction from
its catalogued effective temperature, comparing bolometric flux against the blackbody integral
over each file's actual wavelength coverage, and matching raw files against their corresponding
flux-cube grid node under a tight distance tolerance to avoid false positives from molecular-band
features. As of the most recent audit run, all installed models pass (12/12). The audit process
has also been useful for separating genuine data problems from methodology artifacts: an
initially suspected raw-versus-cube mismatch in the NextGen2 grid was traced to the comparison
methodology itself (disagreement at molecular-band features introduced by interpolating onto a
union wavelength grid) rather than to the data, and apparent Wien-peak violations in the Husfeld
and tmap2 grids were confirmed to be a real, physically expected redward shift caused by UV/EUV
atmospheric opacity in hot compact objects, not a unit error. Two smaller items remain open and
are tracked rather than silently patched: a known 1/π flux-normalization discrepancy in the
blackbody-reference generator, and a structural incompatibility between one MSG-hosted grid
(`sg-BSTAR2006-low`) and the current HDF5 reader.

# Research impact statement

The most concrete evidence of `SED_Tools`'s research impact is its role as the designated
data-preparation pipeline for the MESA `colors` module, a new module (co-authored by N. J.
Miller) that was merged into the official Modules for Experiments in Stellar Astrophysics (MESA)
code base and shipped starting with release r25.10.1-rc1
[@mesa_release_r25101rc1]. MESA is a widely used, actively maintained open-source stellar
evolution code with a large user base spanning asteroseismology, stellar population synthesis,
and binary evolution research [@paxton2011; @paxton2013; @paxton2015; @paxton2018; @paxton2019;
@jermyn2023]. The `colors` module's own source documentation states explicitly that its required
input files — a lookup table and a pre-computed flux cube — are produced by `SED_Tools`, and
provides no other supported route to generating them [@mesa_colors_readme]. This means that
every MESA user who enables synthetic photometry via the `colors` module is, in practice, a
downstream consumer of `SED_Tools`'s output, independent of whether they interact with the
`SED_Tools` repository directly. This is a stronger and more durable form of research impact than
citation count alone, because MESA's release and documentation infrastructure make the
dependency independently verifiable rather than self-reported. As a secondary, softer indicator,
the repository has accumulated organic community interest — 34 GitHub stars at the time of
writing — without any promotional effort. We consider the MESA integration the primary evidence
for this section and the star count a minor supporting signal.

# AI usage disclosure

OpenAI Codex and Claude Code assisted with delinting/formatting during a codebase refactor, with
constructing portions of the automated test suite, and with debugging flux-cube-construction
performance issues; all such changes were reviewed and validated by the author before merging.
All architectural decisions described in Software Design were made by the author, not the AI
tools. This manuscript was drafted with Claude's assistance from author interviews and reviewed
and edited by the author before submission.

# Acknowledgements

*REPLACE: list any funding, grants, or institutional support here, if applicable. Leave this
section out entirely if there is none to disclose.*

# References
