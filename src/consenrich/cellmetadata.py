from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path


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
