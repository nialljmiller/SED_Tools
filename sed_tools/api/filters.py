"""The ``Filters`` API: filter curves from installed copies, NJM, the original sources or SVO, their
composition from components, and NJM's catalogue (properties, descriptions, search, citations, plots).
sed_tools.api re-exports it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from requests import RequestException

from ..cube.model import FILTER_DIR_DEFAULT
from ..filters.njm_catalogue import (CatalogueError, FilterCatalogue, collect_citations, format_bibtex,
                                     format_citations, parse_target)
from ..njm_mirror import NJM_NOT_LISTED

logger = logging.getLogger("sed_tools.api")


def _listing_failures(errors: List[str]) -> List[str]:
    """The catalogue listing errors that leave a list incomplete (an unreachable NJM does not)."""
    return [error for error in errors if error != NJM_NOT_LISTED]


class Filters:
    """API for working with photometric filter profiles.

    Usage::

        filters = Filters.query()
        filters = Filters.query(facility='HST')
        Filters.fetch('Generic', 'Johnson')
        Filters.search(wavelength=21500, fwhm=(1000, 5000))   # NJM's catalogue, no SVO
        Filters.info('2MASS', '2MASS'); Filters.properties('2MASS', '2MASS')
        print(Filters.cite('2MASS/2MASS', 'GAIA/GAIA/G'))
        Filters.plot('2MASS/2MASS', normalize='peak', vega=True)
    """
    
    _filter_root: Path = FILTER_DIR_DEFAULT
    
    @classmethod
    def query(
        cls,
        facility: Optional[str] = None,
        include_local: bool = True,
        include_remote: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query available filter profiles."""
        results = []
        
        if include_local:
            results.extend(cls._query_local(facility))
        
        if include_remote:
            results.extend(cls._query_remote(facility))
        
        return results
    
    @classmethod
    def _query_local(cls, facility: Optional[str] = None) -> List[Dict[str, Any]]:
        """Installed instruments holding at least one .dat curve (a failed download can leave only an index file)."""
        results = []
        if not cls._filter_root.exists():
            return results
        for fac_dir in sorted(cls._filter_root.iterdir()):
            if not fac_dir.is_dir() or (facility and fac_dir.name.lower() != facility.lower()):
                continue
            for inst_dir in sorted(fac_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                filters = sorted(inst_dir.glob("*.dat"))
                if filters:
                    results.append({'facility': fac_dir.name, 'instrument': inst_dir.name,
                                    'n_filters': len(filters), 'is_local': True})
        return results

    @classmethod
    def _query_remote(cls, facility: Optional[str] = None) -> List[Dict[str, Any]]:
        """A row per remote source of each facility: ``facility_key`` is the spelling to download with (NJM's,
        else the original builder's, else SVO's), ``svo_facility_key`` SVO's own, or None if SVO lacks it."""
        from ..filters.library import FilterLibrary

        report = FilterLibrary(cls._filter_root).facilities()
        for error in _listing_failures(report["errors"]):
            logger.warning("Filter discovery: %s", error)
        rows = []
        for row in report["entries"]:
            names = row.get("names", {})
            if facility and facility.casefold() not in {n.casefold() for n in (row["key"], row["label"], *names.values())}:
                continue
            rows.extend({"facility": row["label"], "facility_key": row["key"], "svo_facility_key": names.get("svo"),
                         "source": source, "is_local": False}
                        for source in row["sources"] if source != "local")
        return rows

    @classmethod
    def fetch(
        cls,
        facility: str,
        instrument: str,
        filter_root: Optional[Union[str, Path]] = None,
        components: Optional[Union[str, Sequence[str]]] = None,
        atmosphere: Optional[Dict[str, Any]] = None,
        *,
        bands: Optional[Sequence[str]] = None,
        detector_type: Optional[str] = None,
        allow_heavy: bool = False,
        allow_partial: bool = False,
        variants: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Download filter profiles, optionally composing them from components.

        With no ``components``/``atmosphere`` this returns the directory of the
        published curves (the one MESA reads).  Otherwise the published
        curves are downloaded, the instrument's components fetched, and a
        product built under ``products/<recipe>/``, whose path is returned.
        While NJM is unreachable, a product of NJM's that SVO holds only some
        bands of is installed with those bands and a warning naming the
        others; :meth:`available` then shows it incomplete, and fetching it
        again with NJM reachable completes it.

        Parameters
        ----------
        components : str or list of str, optional
            Any of ``filter``, ``optics``, ``qe``, or ``published`` (the
            published curve as the base), e.g. ``["filter", "optics", "qe"]``
            or ``"filter"`` for the bare bandpass.  See :meth:`available`.
        atmosphere : dict, optional
            Observing conditions, e.g. ``{"airmass": 1.2, "pwv": 2.5}``, to fold
            in the telluric transmission at the instrument's *own* site
            (ground-based instruments with a site only).  Components are never
            taken from another instrument or site.
        bands : sequence of str, optional
            Compose only these bands, useful when components cover part of a set.
        detector_type : str, optional
            Explicit energy/photon integration convention for a derived copy.
            This changes weighting, not the physical QE component.
        allow_heavy : bool
            Permit large original-source downloads such as JWST reference data.
        allow_partial : bool
            Explicitly allow zero transmission outside component coverage.
        variants : dict, optional
            Alternative component curves to use, e.g. ``{"qe": "single_layer"}``
            for a detector coating or epoch; see ``variants`` in
            :meth:`available`.  The choice is part of the product name.
        """
        instrument_dir = cls._fetch_published(facility, instrument, filter_root=filter_root)
        if (components is None and atmosphere is None and detector_type is None and bands is None
                and not variants):
            return instrument_dir
        return cls.compose(instrument_dir, components or ["published"], atmosphere=atmosphere,
                           filter_root=filter_root, bands=bands, detector_type=detector_type,
                           allow_heavy=allow_heavy, allow_partial=allow_partial, variants=variants)

    @classmethod
    def _fetch_published(cls, facility, instrument, filter_root=None) -> Path:
        from ..filters.library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).published(facility, instrument)

    @classmethod
    def tag_detector_types(
        cls,
        path: Optional[Union[str, Path]] = None,
        *,
        facility: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ensure every filter file under ``path`` declares its detector type.

        Each ``.dat`` file gets ``DetectorType=energy|photon`` on its column
        row (plus ``FilterID`` and ``DetectorTypeSource``), resolved from SVO by
        filter ID or by matching the curve against SVO's records. Files with no
        ``DetectorType`` that cannot be identified are written as ``energy``
        with source ``assumed``; files that already have one keep it. An
        existing type is replaced only by a resolution whose source has equal
        or higher precedence (``original`` > ``inferred`` > ``svo`` >
        ``svo-curve-match`` > ``assumed``); a weaker resolution is reported
        (``kept``, ``resolved_type``, ``resolved_source``, ``note``) and not
        written. See :mod:`sed_tools.filter_metadata`.

        Parameters
        ----------
        path : str or Path, optional
            Filter root, a facility/instrument directory, or a single file.
            Defaults to the configured filter root.
        facility : str, optional
            SVO facility name, when ``path`` is a single instrument directory.
        force : bool
            Re-resolve files that are already tagged. The precedence still
            applies, so ``original`` and ``inferred`` types are kept.
        dry_run : bool
            Report what would be written without modifying files.

        Each file is reported as it is tagged.  Files that cannot be tagged are
        left unchanged and listed in a RuntimeError once the others are done;
        an unreachable SVO raises
        :class:`~sed_tools.filter_metadata.SVOUnreachableError` at once.
        """
        from ..filters.curve_metadata import (SVOMetadataClient, SVOUnreachableError, iter_filter_files,
                                      tag_filter_file, tag_filter_tree)

        rows: List[Dict[str, Any]] = []

        def record(r) -> None:
            rows.append({
                "path": str(r.path),
                "detector_type": r.detector_type,
                "source": r.source,
                "filter_id": r.filter_id,
                "changed": r.changed,
                "migrated": r.migrated,
                "note": r.note,
                "kept": r.kept,
                "resolved_type": r.resolved_type,
                "resolved_source": r.resolved_source,
            })
            if verbose or r.kept or (r.changed and r.source == "assumed"):
                flag = "*" if r.changed else " "
                note = f"  ({r.note})" if r.note and (r.kept or r.source == "assumed") else ""
                print(f"[filters]{flag} {r.path}: {r.detector_type} [{r.source}]{note}")

        target = Path(path) if path is not None else cls._filter_root
        if facility is None:
            tag_filter_tree(target, force=force, dry_run=dry_run, progress=record)
            return rows

        client = SVOMetadataClient()
        failed = []
        for p in iter_filter_files(target):
            try:
                record(tag_filter_file(p, client=client, facility=facility, force=force, dry_run=dry_run))
            except SVOUnreachableError:
                raise  # every later lookup would fail too
            except (RequestException, ValueError) as exc:
                failed.append(f"{p}: {exc}")
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(failed) + len(rows)} filter files could not be tagged "
                               "and were left unchanged:\n  " + "\n  ".join(failed))
        return rows

    @classmethod
    def available(
        cls,
        facility: str,
        instrument: str,
        filter_root: Optional[Union[str, Path]] = None,
        *,
        include_remote: bool = True,
    ) -> Dict[str, Any]:
        """Describe what can be requested for an instrument.

        Returns the published curve's contents (e.g. ``filter+instrument+atmosphere``),
        the instrument's own separable components, the with/without variants
        that can be built from them, whether (and at which site) a telluric
        component can be added, and the detector type.  Uses the local copy if present,
        otherwise the NJM manifest or SVO metadata. ``include_remote=False``
        inspects installed data and source registration without network access.

        For an installed set, ``bands`` are the bands installed and
        ``expected_bands`` every band it declares; ``missing_files`` names the
        files of the others.  A set :meth:`fetch` could install only in part
        (from SVO while NJM was unreachable) is one where they differ.
        """
        from ..filters.library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).available(
            facility, instrument, remote=include_remote)

    @classmethod
    def inventory(cls, *, filter_root=None, include_remote=True, facility=None):
        """List all discovered instruments and source-review status.

        Includes SVO, NJM, local data and registered original-source builders.
        Individual passbands are not downloaded. Discovery errors are returned
        alongside the rows; unreviewed instruments are never called total-only.
        """
        from ..filters.library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).inventory(
            remote=include_remote, facility=facility)

    @classmethod
    def atmospheres(cls, filter_root=None, *, include_remote=True):
        """Describe cached and mirrored atmosphere grids, without generating models."""
        from ..filters.library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).atmospheres(remote=include_remote)

    @classmethod
    def compose(
        cls,
        instrument: Union[str, Path],
        components: Union[str, Sequence[str]],
        *,
        atmosphere: Optional[Dict[str, Any]] = None,
        filter_root: Optional[Union[str, Path]] = None,
        allow_partial: bool = False,
        bands: Optional[Sequence[str]] = None,
        detector_type: Optional[str] = None,
        allow_heavy: bool = False,
        variants: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Build ``products/<recipe>/`` for a local instrument directory.

        ``instrument`` is a path or ``"Facility/Instrument"`` under the filter
        root.  Raises :class:`sed_tools.filter_components.ComponentError` with
        an explanation when the request cannot be met (component not
        published for this instrument, double counting, space-based
        instrument with an atmosphere, ...).
        """
        from ..filters.library import FilterLibrary

        out = FilterLibrary(filter_root or cls._filter_root).compose(
            instrument, components, atmosphere=atmosphere, bands=bands,
            detector_type=detector_type, allow_heavy=allow_heavy, allow_partial=allow_partial,
            variants=variants)
        print(f"[filters] Built {out}")
        return out

    @classmethod
    def combine(
        cls,
        output: Union[str, Path],
        *inputs: Union[str, Path],
        filter_root: Optional[Union[str, Path]] = None,
        facility: Optional[str] = None,
        instrument: Optional[str] = None,
        on_conflict: str = "rename",
    ) -> Path:
        """Combine existing filter sets into one MESA-compatible set
        (:func:`sed_tools.combine_filters.combine_filter_sets`)."""
        from ..filters.combine import combine_filter_sets

        return combine_filter_sets(
            output,
            inputs,
            filter_root=filter_root or cls._filter_root,
            facility=facility,
            instrument=instrument,
            on_conflict=on_conflict,
        )

    # NJM catalogue: properties, descriptions, search, citations, plots
    _catalogues: Dict[Tuple[str, bool], Any] = {}

    @classmethod
    def _catalogue(cls, filter_root=None, *, offline: bool = False):
        """One catalogue client per filter root, so the mirror is checked once per session."""
        key = (str(Path(filter_root or cls._filter_root).expanduser().resolve()), offline)
        if key not in cls._catalogues:
            cls._catalogues[key] = FilterCatalogue(key[0], offline=offline)
        return cls._catalogues[key]

    @classmethod
    def properties(
        cls,
        facility: str,
        instrument: str,
        *,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """An instrument's ``properties.json`` as NJM serves it.

        ``{"bands": {band: {"WavelengthPivot": ..., "FWHM": ..., "ZeroPointVega_Jy": ...,
        "ABminusVega": ..., "DetectorType": ..., "DisplayColor": ...}}, "definitions": {...}}``,
        computed by :mod:`sed_tools.filter_properties` (units under ``definitions``).

        The copy saved beside installed curves is used first, since it describes
        those curves; otherwise NJM's current file, cached under
        ``<filter root>/../cache/njm/``.  ``refresh=True`` asks NJM again;
        ``offline=True`` uses installed and cached copies only.  Raises
        :class:`sed_tools.filter_catalogue.CatalogueError` when neither has it.
        """
        return cls._catalogue(filter_root, offline=offline).properties(facility, instrument, refresh=refresh)

    @classmethod
    def info(
        cls,
        facility: str,
        instrument: str,
        *,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """An instrument's ``info.json`` as NJM serves it.

        Descriptions, links, notes, the references to cite (``bibliography``, with
        BibTeX where available), the providers' ``acknowledgements``, the
        ``licence``, and per band the SVO-style fields (``Description``,
        ``PhotSystem``, ``MagSys``, ...), owner-published values and what SVO
        recorded.  Sources are chosen as in :meth:`properties`.  Bands NJM keeps
        beside the set but does not serve as part of it (superseded copies such
        as ``BP_svo``) are under ``archived_bands``, not ``bands``; so in
        :meth:`properties`.
        """
        return cls._catalogue(filter_root, offline=offline).info(facility, instrument, refresh=refresh)

    @classmethod
    def search(
        cls,
        text: Optional[str] = None,
        *,
        facility: Optional[str] = None,
        instrument: Optional[str] = None,
        band: Optional[str] = None,
        wavelength: Any = None,
        pivot: Optional[Sequence[Optional[float]]] = None,
        fwhm: Optional[Sequence[Optional[float]]] = None,
        detector_type: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        properties: bool = True,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Find bands in NJM's catalogue, one row per band, without SVO.

        Parameters
        ----------
        text : str, optional
            Case-insensitive substring of ``Facility/Instrument/Band``.
        facility, instrument, band : str, optional
            Case-insensitive substrings of each name.
        wavelength : float or (low, high), optional
            Å (or an astropy quantity).  A single value keeps bands that cover it
            (``WavelengthMin <= λ <= WavelengthMax``, where T is at least 1 % of its
            peak); a range keeps bands that overlap it.
        pivot, fwhm : (low, high), optional
            Ranges in Å for ``WavelengthPivot`` and ``FWHM``; ``None`` leaves an end open.
        detector_type : {"energy", "photon"}, optional
        where : dict, optional
            Any other ``properties.json`` column: a ``(low, high)`` range or a value
            to equal (strings ignore case), e.g. ``{"ABminusVega": (None, 0.5)}``.
            A band whose value is missing (Vega quantities where Vega does not
            cover it) never matches a criterion.
        properties : bool
            ``False`` matches names only, from ``index.json``, without fetching
            any ``properties.json``.

        Names are matched in ``index.json`` first, so ``properties.json`` is
        fetched only for instruments that can still match (at most six requests
        at a time; cached).  Returns a DataFrame sorted by facility, instrument and
        pivot, with ``facility``, ``instrument``, ``band`` and the property
        columns; ``.attrs["units"]`` gives units, ``.attrs["errors"]`` lists
        instruments whose properties could not be read, and ``.attrs["gaps"]``
        lists instruments NJM serves no ``properties.json`` for.
        """
        return cls._catalogue(filter_root, offline=offline).search(
            text, facility=facility, instrument=instrument, band=band, wavelength=wavelength,
            pivot=pivot, fwhm=fwhm, detector_type=detector_type, where=where,
            properties=properties, refresh=refresh)

    @classmethod
    def _targets(cls, targets, filter_root) -> List[Tuple[str, str, Optional[List[str]], Optional[Path]]]:
        """``(facility, instrument, bands or None, directory or None)`` per target (the forms :meth:`plot`
        takes); no targets means every installed instrument."""
        root = Path(filter_root or cls._filter_root)
        items: List[Tuple[str, str, Optional[str], Optional[Path]]] = []
        if not targets:
            for directory in sorted(root.glob("*/*")):
                if directory.is_dir() and any(directory.glob("*.dat")):
                    items.append((directory.parent.name, directory.name, None, directory))
        for target in targets:
            if isinstance(target, pd.DataFrame):
                items.extend((r.facility, r.instrument, r.band, None)
                             for r in target[["facility", "instrument", "band"]].itertuples())
            elif isinstance(target, (tuple, list)):
                fac, inst, *rest = target
                items.append((fac, inst, rest[0] if rest else None, None))
            elif isinstance(target, Path) or (Path(str(target)).exists() and (
                    Path(str(target)).suffix == ".dat" or any(Path(str(target)).glob("*.dat")))):
                path = Path(target)
                if path.is_file():
                    items.append((path.parent.parent.name, path.parent.name, path.stem, path.parent))
                else:
                    items.append((path.parent.name, path.name, None, path))
            else:
                items.append(parse_target(str(target)) + (None,))
        grouped: Dict[Tuple[str, str], List[Any]] = {}
        for fac, inst, band, directory in items:
            entry = grouped.setdefault((fac, inst), [fac, inst, [], directory])
            if band is None:
                entry[2] = None
            elif entry[2] is not None and band not in entry[2]:
                entry[2].append(band)
        return [tuple(v) for v in grouped.values()]

    @classmethod
    def citations(
        cls,
        *targets: Any,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """What to cite and acknowledge for instruments or bands, from their ``info.json``.

        Targets are as in :meth:`plot`; with none, every installed instrument.
        Returns ``{"instruments", "references", "acknowledgements", "errors"}``:
        references merged across instruments (one per work, following
        ``same_as`` to the bibcode form, each with ``bibtex`` where NJM has it and
        ``used_by``), and the providers' acknowledgement texts, merged.  For a
        subset of bands, the instrument-wide references plus those bands' own.
        """
        catalogue = cls._catalogue(filter_root, offline=offline)
        infos, errors = [], []
        for fac, inst, bands, directory in cls._targets(targets, filter_root):
            try:
                local = directory / "info.json" if directory is not None else None
                if local is not None and local.is_file() and not refresh:
                    info = catalogue._json_object(local.read_text(encoding="utf-8"))
                else:
                    info = catalogue.info(fac, inst, refresh=refresh)
            except (CatalogueError, OSError, ValueError) as exc:
                errors.append(f"{fac}/{inst}: {exc}")
                continue
            if bands is not None:
                known = {**(info.get("archived_bands") or {}), **info.get("bands", {})}
                unknown = [b for b in bands if b not in known]
                errors.extend(f"{fac}/{inst}: no band {b!r} in info.json" for b in unknown)
                bands = [b for b in bands if b not in unknown]
                if not bands:
                    continue
            infos.append((f"{info.get('facility', fac)}/{info.get('instrument', inst)}", info, bands))
        result = collect_citations(infos)
        result["errors"] = errors
        return result

    @classmethod
    def cite(
        cls,
        *targets: Any,
        format: str = "text",
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> str:
        """:meth:`citations` as text (``format="text"``: acknowledgements, licences,
        references and BibTeX) or as a ``.bib`` file (``format="bibtex"``).

        References NJM holds no BibTeX for appear in the ``.bib`` output as ``%``
        comments with what is known; nothing is made up.  The text lists any
        target whose ``info.json`` could not be read; the ``.bib`` file cannot,
        so then :class:`~sed_tools.filter_catalogue.CatalogueError` is raised
        (:meth:`citations` returns what was found).
        """
        result = cls.citations(*targets, filter_root=filter_root, refresh=refresh, offline=offline)
        if format == "bibtex":
            if result["errors"]:
                raise CatalogueError("Cannot cite every target: " + "; ".join(result["errors"]))
            return format_bibtex(result)
        if format == "text":
            return format_citations(result)
        raise ValueError("format must be 'text' or 'bibtex'")

    @classmethod
    def plot(
        cls,
        *targets: Any,
        normalize: Union[None, bool, str] = None,
        vega: Any = False,
        ax=None,
        legend: bool = True,
        log_wavelength: Optional[bool] = None,
        filter_root: Optional[Union[str, Path]] = None,
        fetch: bool = True,
    ):
        """Plot filter curves in their :mod:`sed_tools.filter_colors` colours; returns the Axes.

        Targets: ``"Facility/Instrument"``, ``"Facility/Instrument/Band"``, a
        tuple of those parts, an instrument directory or ``.dat`` file, or a
        :meth:`search` result.  Instruments that are not installed are fetched
        (``fetch=False`` refuses instead).  Bands keep their instrument's colours
        (true colours for visible bands, ordered false colours otherwise) when
        they all come from one instrument; a selection across instruments is
        coloured as one set.  ``normalize`` is ``None``, ``"peak"`` or
        ``"area"``.  ``vega=True`` overlays NJM's Vega spectrum
        (``vega/vega_flam.csv``) scaled to the curves; a path or a
        ``(wavelength Å, f_λ)`` pair overlays another spectrum.  Needs matplotlib.
        """
        from ..filters.display import instrument_colors
        from ..filters.plot import plot_curves
        from ..filters.properties import load_vega
        from ..photometry import load_filter_curve

        if not targets:
            raise ValueError("nothing to plot: give Facility/Instrument[/Band] targets or a search result")
        catalogue = cls._catalogue(filter_root)
        chosen = cls._plot_bands(targets, filter_root, catalogue, fetch)

        curves, labels = {}, {}
        for fac, inst, band, directory in chosen:
            key = f"{fac}/{inst}/{band}"
            curve = load_filter_curve(str(directory / f"{band}.dat"))
            curves[key] = (curve.wavelength, curve.transmission)
        directories = {directory for *_, directory in chosen}
        colors, title = None, None
        if len(directories) == 1:
            directory = directories.pop()
            served = instrument_colors(sorted(directory.glob("*.dat")))
            colors = {f"{fac}/{inst}/{band}": served[band]["DisplayColor"] for fac, inst, band, _ in chosen}
            labels = {f"{fac}/{inst}/{band}": band for fac, inst, band, _ in chosen}
            title = f"{chosen[0][0]}/{chosen[0][1]}"
        else:
            labels = {f"{fac}/{inst}/{band}": f"{fac}/{inst} {band}" for fac, inst, band, _ in chosen}

        spectrum = None
        if vega is True:
            spectrum = load_vega(catalogue.vega())
        elif isinstance(vega, (str, Path)):
            spectrum = load_vega(vega)
        elif vega is not None and vega is not False:
            spectrum = vega
        return plot_curves(curves, colors=colors, labels=labels, normalize=normalize, vega=spectrum,
                           ax=ax, legend=legend, log_wavelength=log_wavelength, title=title)

    @classmethod
    def _plot_bands(cls, targets, filter_root, catalogue, fetch) -> List[Tuple[str, str, str, Path]]:
        """``(facility, instrument, band, directory)`` for every band the plot targets select."""
        chosen: List[Tuple[str, str, str, Path]] = []
        for fac, inst, bands, directory in cls._targets(targets, filter_root):
            if directory is None:
                fac, inst, directory = cls._installed_or_fetched(fac, inst, filter_root, catalogue, fetch)
            files = {p.stem: p for p in sorted(Path(directory).glob("*.dat"))}
            for band in (files if bands is None else bands):
                if band not in files:
                    raise FileNotFoundError(f"{fac}/{inst} has no band {band!r} in {directory}")
                chosen.append((fac, inst, band, Path(directory)))
        return chosen

    @classmethod
    def _installed_or_fetched(cls, fac, inst, filter_root, catalogue, fetch) -> Tuple[str, str, Path]:
        """``(facility, instrument, directory)`` of an installed instrument (names ignore case), else fetched."""
        root = Path(filter_root or cls._filter_root)
        for d in (root / fac / inst, *root.glob("*/*")):
            if (d.is_dir() and (d.parent.name.casefold(), d.name.casefold()) == (fac.casefold(), inst.casefold())
                    and any(d.glob("*.dat"))):
                return fac, inst, d
        if not fetch:
            raise FileNotFoundError(f"{fac}/{inst} is not installed under {root}")
        try:
            served = catalogue.resolve(fac, inst)
        except CatalogueError as exc:
            logger.warning("%s; fetching %s/%s under the name given", exc, fac, inst)
        else:
            fac, inst = served["facility"], served["instrument"]
        return fac, inst, cls._fetch_published(fac, inst, filter_root=filter_root)
