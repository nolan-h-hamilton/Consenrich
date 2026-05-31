"""Runtime reader for Consenrich genome covariate caches."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FEATURES = ("gc", "low_mappability_frac", "repeat_frac")
KNOWN_FEATURES = FEATURES
FEATURE_ALIASES = {"gc_dev": "gc"}
SCHEMA_VERSION = "consenrich-genome-covariates-v1"


@dataclass(frozen=True)
class CacheChromosome:
    name: str
    length: int
    bins: int
    array: str


@dataclass(frozen=True)
class GenomeCovariateCacheValidation:
    manifest: dict[str, Any]
    bin_size_bp: int
    features: tuple[str, ...]
    feature_index: dict[str, int]
    chromosomes: dict[str, CacheChromosome]

    def validate_request(
        self,
        *,
        required_features: Sequence[str] | None = None,
        interval_size_bp: int | None = None,
        requested_chromosomes: Sequence[str] | None = None,
        required_features_label: str = "requested features",
    ) -> None:
        _validate_cache_request(
            self,
            required_features=required_features,
            interval_size_bp=interval_size_bp,
            requested_chromosomes=requested_chromosomes,
            required_features_label=required_features_label,
        )


def _feature_key(name: Any) -> str:
    return str(name).strip().replace("-", "_").lower()


def normalize_genome_covariate_feature_name(
    name: Any,
    *,
    available_features: Sequence[str] | None = None,
) -> str:
    """Normalize a requested feature name while preserving manifest spelling."""

    raw = str(name).strip()
    key = _feature_key(raw)
    alias = FEATURE_ALIASES.get(key)
    if alias is not None:
        return alias
    if available_features is None:
        return key
    available = tuple(str(feature) for feature in available_features)
    if raw in available:
        return raw
    matches = [feature for feature in available if _feature_key(feature) == key]
    if len(matches) == 1:
        return matches[0]
    return key


def resolve_genome_covariate_feature_config(
    value: Any,
    *,
    default_features: Sequence[str],
    available_features: Sequence[str] | None = None,
    config_name: str = "features",
) -> tuple[str, ...]:
    """Resolve config feature selectors into manifest feature names.

    ``"all"`` expands to all manifest features when ``available_features`` is
    provided; otherwise it expands to the supplied defaults.
    """

    if value is None:
        raw_items = list(default_features)
    elif value is True:
        raw_items = ["all"]
    elif value is False:
        raw_items = []
    elif isinstance(value, str):
        text = value.strip()
        raw_items = [] if not text else [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raise ValueError(
            f"`{config_name}` must be 'all', a comma-separated string, "
            "a list, boolean, or null."
        )

    source_features = (
        available_features if available_features is not None else default_features
    )
    all_features = tuple(str(feature) for feature in source_features)
    features: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        name = str(item).strip()
        if not name:
            continue
        if _feature_key(name) == "all":
            candidates = all_features
        else:
            candidates = (
                normalize_genome_covariate_feature_name(
                    name,
                    available_features=available_features,
                ),
            )
        for candidate in candidates:
            if candidate not in seen:
                features.append(candidate)
                seen.add(candidate)
    return tuple(features)


def validate_genome_covariate_cache(
    cache_dir: str | os.PathLike[str],
    *,
    required_features: Sequence[str] | None = None,
    interval_size_bp: int | None = None,
    requested_chromosomes: Sequence[str] | None = None,
    required_features_label: str = "requested features",
) -> GenomeCovariateCacheValidation:
    """Validate a local genome covariate cache manifest and array headers."""

    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    with open(manifest_path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("genome covariate cache manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported genome covariate cache schema "
            f"{manifest.get('schema')!r}; expected {SCHEMA_VERSION!r}"
        )

    try:
        bin_size_bp = int(manifest["bin_size_bp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "genome covariate cache bin_size_bp must be an integer"
        ) from exc
    if bin_size_bp <= 0:
        raise ValueError("genome covariate cache bin_size_bp must be positive")

    raw_features = manifest.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("genome covariate cache features must be a list")
    features = tuple(str(feature).strip() for feature in raw_features)
    if not features:
        raise ValueError("genome covariate cache must define at least one feature")
    if any(not feature for feature in features):
        raise ValueError("genome covariate cache features must be non-empty strings")
    if len(set(features)) != len(features):
        raise ValueError("genome covariate cache features must be unique")
    feature_index = {name: i for i, name in enumerate(features)}

    raw_chromosomes = manifest.get("chromosomes")
    if not isinstance(raw_chromosomes, list):
        raise ValueError("genome covariate cache chromosomes must be a list")
    chromosomes: dict[str, CacheChromosome] = {}
    for idx, row in enumerate(raw_chromosomes):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"genome covariate cache chromosome row {idx} must be an object"
            )
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError(
                "genome covariate cache chromosome names must be non-empty"
            )
        if name in chromosomes:
            raise ValueError(
                f"genome covariate cache chromosome names must be unique: {name}"
            )
        try:
            length = int(row["length"])
            bins = int(row["bins"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"genome covariate cache chromosome {name} length/bins "
                "must be integers"
            ) from exc
        if length < 0 or bins < 0:
            raise ValueError(
                f"genome covariate cache chromosome {name} length/bins "
                "must be nonnegative"
            )
        array = str(row.get("array", "")).strip()
        if not array:
            raise ValueError(
                f"genome covariate cache chromosome {name} must define an array path"
            )
        chromosomes[name] = CacheChromosome(
            name=name,
            length=length,
            bins=bins,
            array=array,
        )

        path = cache_path / array
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except OSError as exc:
            raise ValueError(
                f"could not read genome covariate array for {name}: {path}"
            ) from exc
        if arr.shape != (bins, len(features)):
            raise ValueError(
                f"genome covariate array shape mismatch for {name}: "
                f"expected {(bins, len(features))}, got {arr.shape}"
            )
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"genome covariate array for {name} must contain numeric values"
            )
    if not chromosomes:
        raise ValueError("genome covariate cache must define at least one chromosome")

    validation = GenomeCovariateCacheValidation(
        manifest=manifest,
        bin_size_bp=bin_size_bp,
        features=features,
        feature_index=feature_index,
        chromosomes=chromosomes,
    )
    validation.validate_request(
        required_features=required_features,
        interval_size_bp=interval_size_bp,
        requested_chromosomes=requested_chromosomes,
        required_features_label=required_features_label,
    )
    return validation


def _validate_cache_request(
    validation: GenomeCovariateCacheValidation,
    *,
    required_features: Sequence[str] | None = None,
    interval_size_bp: int | None = None,
    requested_chromosomes: Sequence[str] | None = None,
    required_features_label: str = "requested features",
) -> None:
    if interval_size_bp is not None:
        try:
            interval_size = int(interval_size_bp)
        except (TypeError, ValueError) as exc:
            raise ValueError("interval_size_bp must be an integer") from exc
        if interval_size <= 0:
            raise ValueError("interval_size_bp must be positive")
        if interval_size % validation.bin_size_bp != 0:
            raise ValueError(
                "genome covariate cache bin size must divide Consenrich interval size"
            )

    if required_features is not None:
        required = resolve_genome_covariate_feature_config(
            tuple(required_features),
            default_features=validation.features,
            available_features=validation.features,
            config_name=required_features_label,
        )
        missing = [
            feature for feature in required if feature not in validation.feature_index
        ]
        if missing:
            raise ValueError(
                f"genome covariate cache is missing {required_features_label}: "
                + ", ".join(missing)
            )

    if requested_chromosomes is not None:
        requested = tuple(
            str(chrom).strip()
            for chrom in requested_chromosomes
            if str(chrom).strip()
        )
        missing_chromosomes = [
            chrom
            for chrom in dict.fromkeys(requested)
            if chrom not in validation.chromosomes
        ]
        if missing_chromosomes:
            raise ValueError(
                "genome covariate cache is missing requested chromosomes: "
                + ", ".join(missing_chromosomes)
            )


class ConsenrichGenomeCovariateCache:
    """Fast mmap-backed reader for genome covariate caches."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        mmap: bool = True,
        required_features: Sequence[str] | None = None,
        interval_size_bp: int | None = None,
        requested_chromosomes: Sequence[str] | None = None,
        required_features_label: str = "requested features",
    ):
        self.cache_dir = Path(cache_dir)
        self._validation = validate_genome_covariate_cache(
            self.cache_dir,
            required_features=required_features,
            interval_size_bp=interval_size_bp,
            requested_chromosomes=requested_chromosomes,
            required_features_label=required_features_label,
        )
        self.manifest = self._validation.manifest
        self.bin_size_bp = self._validation.bin_size_bp
        self.features = self._validation.features
        self.feature_index = self._validation.feature_index
        self.chromosomes = self._validation.chromosomes
        self._mmap = bool(mmap)
        self._arrays: dict[str, np.ndarray] = {}

    def validate_request(
        self,
        *,
        required_features: Sequence[str] | None = None,
        interval_size_bp: int | None = None,
        requested_chromosomes: Sequence[str] | None = None,
        required_features_label: str = "requested features",
    ) -> None:
        self._validation.validate_request(
            required_features=required_features,
            interval_size_bp=interval_size_bp,
            requested_chromosomes=requested_chromosomes,
            required_features_label=required_features_label,
        )

    def chrom_array(self, chrom: str) -> np.ndarray:
        if chrom not in self.chromosomes:
            raise KeyError(f"chromosome not in genome covariate cache: {chrom}")
        if chrom not in self._arrays:
            row = self.chromosomes[chrom]
            path = self.cache_dir / row.array
            self._arrays[chrom] = np.load(path, mmap_mode="r" if self._mmap else None)
            if self._arrays[chrom].shape != (row.bins, len(self.features)):
                raise ValueError(
                    f"genome covariate array shape mismatch for {chrom}: "
                    f"expected {(row.bins, len(self.features))}, "
                    f"got {self._arrays[chrom].shape}"
                )
        return self._arrays[chrom]

    def fetch(
        self,
        chrom: str,
        *,
        start: int = 0,
        end: int | None = None,
        feature_names: Sequence[str] | None = None,
        interval_size_bp: int | None = None,
    ) -> np.ndarray:
        """Fetch interval-aligned covariates as float32 rows.

        ``interval_size_bp`` may equal the cache bin size or be an integer
        multiple of it. Other sizes are rejected to avoid hidden interpolation.
        """

        if chrom not in self.chromosomes:
            raise KeyError(f"chromosome not in genome covariate cache: {chrom}")
        row = self.chromosomes[chrom]
        start = max(0, int(start))
        end = row.length if end is None else min(int(end), row.length)
        names = resolve_genome_covariate_feature_config(
            feature_names,
            default_features=self.features,
            available_features=self.features,
            config_name="feature_names",
        )
        if end <= start:
            return np.empty((0, len(names)), dtype=np.float32)
        missing = [name for name in names if name not in self.feature_index]
        if missing:
            raise KeyError(
                "features not in genome covariate cache: " + ", ".join(missing)
            )

        interval_size = (
            self.bin_size_bp if interval_size_bp is None else int(interval_size_bp)
        )
        if interval_size <= 0:
            raise ValueError("interval_size_bp must be positive")
        if start % self.bin_size_bp != 0:
            raise ValueError(
                "genome covariate cache start must be aligned to cache bin size"
            )
        if interval_size % self.bin_size_bp != 0:
            raise ValueError(
                "genome covariate cache bin size must divide Consenrich interval size"
            )

        indices = [self.feature_index[name] for name in names]
        arr = self.chrom_array(chrom)
        first = start // self.bin_size_bp
        last = (end + self.bin_size_bp - 1) // self.bin_size_bp
        base = np.asarray(arr[first:last, :][:, indices], dtype=np.float32)
        if interval_size == self.bin_size_bp:
            return base

        group = interval_size // self.bin_size_bp
        n_out = (end - start + interval_size - 1) // interval_size
        out = np.full((n_out, len(indices)), np.nan, dtype=np.float32)
        for idx in range(n_out):
            lo = idx * group
            hi = min((idx + 1) * group, base.shape[0])
            block = np.asarray(base[lo:hi, :], dtype=np.float32)
            finite = np.isfinite(block)
            counts = np.sum(finite, axis=0)
            if np.any(counts > 0):
                sums = np.sum(np.where(finite, block, 0.0), axis=0)
                out[idx, counts > 0] = sums[counts > 0] / counts[counts > 0]
        return out
