from __future__ import annotations

import csv
import gzip
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parseMetadataSpec(spec):
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError("`cellMetadata` must be a mapping.")
    allowedKeys = {"path", "barcodeColumn", "sampleColumn", "groupColumn"}
    if set(spec) - allowedKeys:
        raise ValueError(
            f"Unsupported cell metadata keys: {sorted(set(spec) - allowedKeys)}"
        )
    metadataPath = spec.get("path")
    if not isinstance(metadataPath, (str, Path)) or not str(metadataPath):
        raise ValueError("`cellMetadata.path` must be a non-empty path.")
    metadataPath = Path(metadataPath).resolve()
    if not str(metadataPath).endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".h5ad")):
        raise ValueError("Cell metadata must use CSV, TSV, or H5AD format.")
    result = {"path": str(metadataPath)}
    for key in ("barcodeColumn", "sampleColumn", "groupColumn"):
        value = spec.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"`cellMetadata.{key}` must be a non-empty column name.")
        result[key] = value
    if metadataPath.suffix != ".h5ad" and result["barcodeColumn"] is None:
        raise ValueError("Tabular cell metadata requires `barcodeColumn`.")
    return result


def parseFits(fits):
    if fits is None:
        return None
    if not isinstance(fits, list) or not fits:
        raise ValueError("`scParams.fits` must be a non-empty list.")
    result = []
    fitNames = set()
    for fit in fits:
        if not isinstance(fit, Mapping) or set(fit) != {"name", "selectGroups"}:
            raise ValueError("Each fit requires `name` and `selectGroups`.")
        name = fit["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
        ):
            raise ValueError(
                "Fit names must use letters, digits, underscores, or hyphens."
            )
        if name.casefold() in fitNames:
            raise ValueError(f"Duplicate fit name: {name}")
        fitNames.add(name.casefold())
        groups = fit["selectGroups"]
        if (
            not isinstance(groups, list)
            or not groups
            or any(not isinstance(group, str) or not group for group in groups)
        ):
            raise ValueError(f"Fit `{name}` requires a non-empty `selectGroups` list.")
        if len(set(groups)) != len(groups):
            raise ValueError(f"Fit `{name}` contains duplicate group names.")
        result.append({"name": name, "selectGroups": tuple(groups)})
    return tuple(result)


def _readMetadata(metadataPath):
    if metadataPath.endswith(".h5ad"):
        try:
            import h5py
            from anndata.io import read_elem
        except ImportError as error:
            raise ImportError(
                "H5AD metadata requires `consenrich[singlecell]`."
            ) from error
        with h5py.File(metadataPath, "r") as fileHandle:
            if "obs" not in fileHandle:
                raise ValueError(f"H5AD metadata lacks `.obs`: {metadataPath}")
            frame = read_elem(fileHandle["obs"])
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"H5AD `.obs` must be a table: {metadataPath}")
        return frame
    delimiter = "," if metadataPath.endswith((".csv", ".csv.gz")) else "\t"
    openFile = gzip.open if metadataPath.endswith(".gz") else open
    with openFile(metadataPath, "rt", encoding="utf-8", newline="") as fileHandle:
        reader = csv.reader(fileHandle, delimiter=delimiter, strict=True)
        try:
            columns = next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty cell metadata file: {metadataPath}") from error
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate cell metadata column names: {metadataPath}")
        records = []
        for rowNumber, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(
                    f"Invalid cell metadata row {rowNumber}: {metadataPath}"
                )
            records.append(row)
    return pd.DataFrame(records, columns=columns)


class MetadataSelections:
    def __init__(self):
        self.tables = {}
        self.catalogs = {}
        self.allowLists = {}

    def _catalog(self, spec):
        metadataPath = spec["path"]
        key = tuple(
            spec.get(field)
            for field in ("path", "barcodeColumn", "sampleColumn", "groupColumn")
        )
        if key in self.catalogs:
            return self.catalogs[key]
        if metadataPath not in self.tables:
            self.tables[metadataPath] = _readMetadata(metadataPath)
        frame = self.tables[metadataPath]
        if not frame.columns.is_unique:
            raise ValueError(f"Duplicate cell metadata column names: {metadataPath}")
        for column in key[1:]:
            if column is not None and column not in frame.columns:
                raise ValueError(
                    f"Missing cell metadata column `{column}`: {metadataPath}"
                )
        barcodeValues = (
            frame.index
            if spec["barcodeColumn"] is None
            else frame[spec["barcodeColumn"]]
        )
        sampleValues = (
            frame[spec["sampleColumn"]]
            if spec["sampleColumn"] is not None
            else [None] * len(frame)
        )
        groupValues = (
            frame[spec["groupColumn"]]
            if spec["groupColumn"] is not None
            else [None] * len(frame)
        )
        identities = set()
        catalog = {}
        for barcode, sample, group in zip(barcodeValues, sampleValues, groupValues):
            if (
                not isinstance(barcode, str)
                or not barcode
                or any(char.isspace() for char in barcode)
            ):
                raise ValueError(
                    f"Invalid barcode identifier in {metadataPath}: {barcode!r}"
                )
            for column, value in (
                (spec["sampleColumn"], sample),
                (spec["groupColumn"], group),
            ):
                if column is not None and (not isinstance(value, str) or not value):
                    raise ValueError(
                        f"Invalid identifier in cell metadata column `{column}`: {value!r}"
                    )
            # barcodes are consistent across donors and can repeat within them
            identity = (sample, barcode)
            if identity in identities:
                raise ValueError(
                    f"Duplicate sample/barcode identifier in {metadataPath}: {identity!r}"
                )
            identities.add(identity)
            catalog.setdefault(sample, {}).setdefault(group, set()).add(barcode)
        self.catalogs[key] = catalog
        return catalog

    def prepareSources(
        self, sources, sharedMetadata=None, selectGroups=None, grouped=False
    ):
        sources = list(sources)
        for index, source in enumerate(sources):
            if source.sourceKind == "H5AD" and (
                source.barcodeAllowListFile is not None or source.barcodeGroupMapFile is not None
            ):
                logger.warning("Ignoring barcodeAllowListFile and barcodeGroupMapFile for H5AD input %s", source.path)
                sources[index] = source._replace(barcodeAllowListFile=None, barcodeGroupMapFile=None)
        sharedMetadata = parseMetadataSpec(sharedMetadata)
        specs = [
            (
                parseMetadataSpec(source.cellMetadata)
                if source.cellMetadata is not None
                else sharedMetadata
            )
            for source in sources
        ]
        metadataSources = {}
        for source, spec in zip(sources, specs):
            if grouped and source.selectGroups is not None:
                raise ValueError(
                    "Per-source `selectGroups` cannot be used with `scParams.fits`."
                )
            if spec is None:
                continue
            if source.barcodeGroupMapFile is not None:
                raise ValueError(
                    "Cell metadata and `barcodeGroupMapFile` cannot be used together."
                )
            if source.sourceKind == "BEDGRAPH":
                raise ValueError(
                    "Cell metadata selection requires BAM or fragments inputs."
                )
            sample = (
                source.metadataSample
                if source.metadataSample is not None
                else source.sampleName
            )
            if spec["sampleColumn"] is not None and (
                not isinstance(sample, str) or not sample
            ):
                raise ValueError(
                    "A metadata sample column requires source `name` or `metadataSample`."
                )
            metadataSources.setdefault(spec["path"], set()).add(
                sample if sample is not None else source.path
            )
        for spec in specs:
            if (
                spec is not None
                and len(metadataSources[spec["path"]]) > 1
                and spec["sampleColumn"] is None
            ):
                raise ValueError("Shared cell metadata requires `sampleColumn`.")
        result = []
        for source, spec in zip(sources, specs):
            if spec is None:
                result.append(source)
                continue
            groups = selectGroups if grouped else source.selectGroups
            if groups is not None and (
                not isinstance(groups, (list, tuple))
                or any(not isinstance(group, str) or not group for group in groups)
            ):
                raise ValueError("`selectGroups` must contain non-empty group names.")
            if groups is not None and spec["groupColumn"] is None:
                raise ValueError("Metadata group selection requires `groupColumn`.")
            sample = (
                source.metadataSample
                if source.metadataSample is not None
                else source.sampleName
            )
            sampleKey = sample if spec["sampleColumn"] is not None else None
            groupCells = self._catalog(spec).get(sampleKey, {})
            barcodes = set()
            for group, cells in groupCells.items():
                if groups is None or group in groups:
                    barcodes.update(cells)
            if source.barcodeAllowListFile is not None:
                allowPath = str(Path(source.barcodeAllowListFile).resolve())
                if allowPath not in self.allowLists:
                    from .core import _loadBarcodeAllowSet

                    self.allowLists[allowPath] = _loadBarcodeAllowSet(allowPath)
                barcodes.intersection_update(self.allowLists[allowPath])
            result.append(
                source._replace(
                    cellMetadata=spec,
                    metadataSample=sample,
                    selectedBarcodes=frozenset(barcodes),
                    selectedCellCount=len(barcodes),
                )
            )
        return result
