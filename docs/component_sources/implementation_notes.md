# Implementation handoff: independent source acquisition

These notes describe constraints observed in the current code while the source
research is being assembled. They are inputs to a later coding task, not changes
to the acquisition pipeline.

## Independence must include discovery

Downloading original component files alone does not remove the SVO dependency:

- [`build_simple`](../../sed_tools/component_sources/_common.py) first fills
  missing published curves through `ensure_published` and skips component bands
  without a corresponding published file.
- The [LSST builder](../../sed_tools/component_sources/lsst.py) requests SVO's
  `LSST/LSST.*` curves before constructing its original-source components.
- The [HST](../../sed_tools/component_sources/hst.py) and
  [JWST](../../sed_tools/component_sources/jwst.py) builders enumerate bands from
  published local `.dat` files. Their source graph/configuration data could
  instead be the basis of independent discovery once the research establishes
  the correct mapping.

For every proposed source, the research therefore needs an original-source
route for band enumeration, variant selection, response metadata and retrieval.
SVO identifiers should remain useful aliases and fallback references. They
should not be required to discover a band or name a component.

## Preserve the physical quantity and its provenance

The implementation needs enough information to distinguish:

- Dimensionless optical throughput, detector QE and their product.
- Percentage-valued tables versus fractional tables.
- Effective area, which includes collecting area.
- Published response functions that may already include wavelength weighting.
- Normalized empirical passbands, which may not factor into physical components.
- Individually measured factors, combined measurements and derived factors.

A photon-counting label alone is insufficient to choose how to integrate an
arbitrary published response. The source's definition of its columns and any
weighting already applied must be retained. Swapping a counting convention and
replacing a detector's QE are separate operations.

## Map releases and observing modes explicitly

Treat source versions, filters, detector chips/coatings and observing modes as
part of the identity. A newer file from the same team is not automatically the
curve represented by an older SVO entry. Where a public archive offers both
current and historical versions, retain their relationship rather than replacing
one silently.

For ground-based instruments, keep the actual site and atmosphere conditions
with the curves. A standard atmosphere supplied by the instrument team and a
newly generated atmospheric model are distinct products. Do not interchange
them merely because they use the same observatory name.

## Keep common acquisition rules readable

The research should identify reusable source patterns: two-column tables,
multi-column component tables, archive member lists, FITS calibration chains,
configuration manifests and official APIs. Straightforward sources can share
declarative parsing and retrieval machinery. A custom Python adapter should be
reserved for a source that requires distinct discovery or interpretation logic.

For each source, retain the original bytes or an immutable upstream reference,
retrieval date, source version, checksums where available, explanatory
documentation and processing instructions. The final source report supplies
evidence for these choices; it is not automatically executable configuration.

## Acceptance checks for the later coding task

1. Discover and acquire supported original-source instruments with SVO disabled.
2. Enumerate all documented bands/modes for the selected source version.
3. Preserve native units and meanings before applying recorded conversions.
4. Compare composed throughput against the matching official total when the
   source actually supplies the necessary factors. Record valid comparison
   limits, rather than assuming every product must reproduce every passband.
5. Keep incomplete component coverage explicit and avoid replacing a missing
   factor with another instrument's or detector's curve.
6. Exercise NJM unavailability and original-source failures with an explicit,
   provenance-preserving SVO fallback.
7. Confirm that CLI and Python API expose the same source choices, available
   components, variants and limitations.

Research findings and verification issues must be resolved or explicitly carried
as limitations before the relevant adapter is enabled.
