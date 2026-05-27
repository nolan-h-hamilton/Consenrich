"""Local builders for Consenrich genome covariate caches."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .genome_covariates import SCHEMA_VERSION


__all__ = [
    "GenomeCovariateBuildResult",
    "GenomeCovariateBuildSpec",
    "build_genome_covariate_cache",
]


ChromSizesInput = (
    str
    | os.PathLike[str]
    | Mapping[str, int]
    | Sequence[tuple[str, int] | Sequence[Any]]
)
PathInput = str | os.PathLike[str]


@dataclass(frozen=True)
class GenomeCovariateBuildSpec:
    output_dir: PathInput
    bin_size_bp: int
    chrom_sizes: ChromSizesInput
    repeat_masker: PathInput | None = None
    repeat_bed: PathInput | None = None
    chromosomes: Sequence[str] | None = None
    assembly: str | None = None
    force: bool = False
    features: Sequence[str] = ("repeat_frac",)


@dataclass(frozen=True)
class GenomeCovariateBuildResult:
    output_dir: Path
    manifest_path: Path
    bin_size_bp: int
    features: tuple[str, ...]
    chromosomes: tuple[str, ...]
    arrays: dict[str, Path]
    manifest: dict[str, Any]


def build_genome_covariate_cache(
    spec: GenomeCovariateBuildSpec,
) -> GenomeCovariateBuildResult:
    """Build a repeat-fraction-only genome covariate cache from local files."""

    output_dir = Path(spec.output_dir)
    bin_size_bp = int(spec.bin_size_bp)
    if bin_size_bp <= 0:
        raise ValueError("bin_size_bp must be positive")

    features = tuple(str(feature) for feature in spec.features)
    if features != ("repeat_frac",):
        raise ValueError("genome covariate builder v1 only supports repeat_frac")

    if output_dir.exists() and not spec.force:
        raise FileExistsError(
            f"genome covariate output already exists: {output_dir}"
        )

    chrom_sizes = _load_chrom_sizes(spec.chrom_sizes)
    chromosomes = _select_chromosomes(chrom_sizes, spec.chromosomes)
    if not chromosomes:
        raise ValueError("no chromosomes selected for genome covariate build")

    repeat_sources = _repeat_sources(spec)
    if not repeat_sources:
        raise ValueError("repeat_masker or repeat_bed must be provided")

    intervals_by_chrom: dict[str, list[tuple[int, int]]] = {
        chrom: [] for chrom in chromosomes
    }
    selected = set(chromosomes)
    source_summaries: list[dict[str, Any]] = []
    for source_type, path in repeat_sources:
        parser = (
            _iter_rmsk_intervals
            if source_type == "repeat_masker"
            else _iter_bed3_intervals
        )
        source_summary = {
            "type": source_type,
            "path": str(path),
            "intervals_read": 0,
            "intervals_used": 0,
            "skipped_unknown_chromosome": 0,
            "skipped_empty_after_clipping": 0,
        }
        for chrom, start, end in parser(path):
            source_summary["intervals_read"] += 1
            if chrom not in selected:
                source_summary["skipped_unknown_chromosome"] += 1
                continue
            clipped = _clip_interval(start, end, chrom_sizes[chrom])
            if clipped is None:
                source_summary["skipped_empty_after_clipping"] += 1
                continue
            intervals_by_chrom[chrom].append(clipped)
            source_summary["intervals_used"] += 1
        source_summaries.append(source_summary)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        arrays_dir = staging_dir / "arrays"
        arrays_dir.mkdir()

        chrom_rows: list[dict[str, Any]] = []
        arrays: dict[str, Path] = {}
        all_values: list[np.ndarray] = []
        total_bins = 0
        total_covered_bases = 0
        total_eligible_bases = 0

        for chrom in chromosomes:
            _validate_array_chromosome_name(chrom)
            chrom_length = chrom_sizes[chrom]
            merged_intervals = _merge_intervals(intervals_by_chrom[chrom])
            array, covered_bases = _build_repeat_frac_array(
                chrom_length, bin_size_bp, merged_intervals
            )
            bins = int(array.shape[0])
            array_relpath = f"arrays/{chrom}.npy"
            array_path = arrays_dir / f"{chrom}.npy"
            np.save(array_path, array, allow_pickle=False)

            chrom_rows.append(
                {
                    "name": chrom,
                    "length": int(chrom_length),
                    "bins": bins,
                    "array": array_relpath,
                }
            )
            arrays[chrom] = output_dir / array_relpath
            total_bins += bins
            total_covered_bases += int(covered_bases)
            total_eligible_bases += int(chrom_length)
            if array.size:
                all_values.append(array[:, 0])

        feature_stats = {
            "repeat_frac": _feature_stats(
                all_values,
                total_bins=total_bins,
                covered_bases=total_covered_bases,
                eligible_bases=total_eligible_bases,
            )
        }
        manifest: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "bin_size_bp": bin_size_bp,
            "features": list(features),
            "chromosomes": chrom_rows,
            "assembly": spec.assembly,
            "sources": {
                "chrom_sizes": _source_name(spec.chrom_sizes),
                "repeat_intervals": source_summaries,
            },
            "parameters": {
                "features": list(features),
                "coordinate_system": "0-based half-open",
                "interval_clipping": "clip to chromosome bounds",
                "interval_merge": "union per chromosome before binning",
                "partial_bin_denominator": "actual bin width",
            },
            "feature_stats": feature_stats,
        }
        with open(staging_dir / "manifest.json", "wt", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        _publish_staging_dir(staging_dir, output_dir, force=spec.force)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return GenomeCovariateBuildResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        bin_size_bp=bin_size_bp,
        features=features,
        chromosomes=tuple(chromosomes),
        arrays=arrays,
        manifest=manifest,
    )


def _repeat_sources(spec: GenomeCovariateBuildSpec) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    if spec.repeat_masker is not None:
        sources.append(("repeat_masker", Path(spec.repeat_masker)))
    if spec.repeat_bed is not None:
        sources.append(("repeat_bed", Path(spec.repeat_bed)))
    return sources


def _open_text(path: Path) -> Iterator[str]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from handle
    else:
        with open(path, "rt", encoding="utf-8") as handle:
            yield from handle


def _iter_rmsk_intervals(path: Path) -> Iterator[tuple[str, int, int]]:
    for line_number, raw_line in enumerate(_open_text(path), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(
                f"{path}:{line_number}: RepeatMasker rmsk row has fewer than 8 fields"
            )
        if fields[5] == "genoName" and fields[6] == "genoStart":
            continue
        try:
            start = int(fields[6])
            end = int(fields[7])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid rmsk genoStart/genoEnd"
            ) from exc
        yield fields[5], start, end


def _iter_bed3_intervals(path: Path) -> Iterator[tuple[str, int, int]]:
    for line_number, raw_line in enumerate(_open_text(path), start=1):
        line = raw_line.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("track ")
            or line.startswith("browser ")
        ):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(
                f"{path}:{line_number}: BED row has fewer than 3 fields"
            )
        if fields[0].lower() in {"chrom", "chromosome"} and fields[1].lower() in {
            "start",
            "chromstart",
        }:
            continue
        try:
            start = int(fields[1])
            end = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid BED start/end") from exc
        yield fields[0], start, end


def _load_chrom_sizes(chrom_sizes: ChromSizesInput) -> dict[str, int]:
    if isinstance(chrom_sizes, (str, os.PathLike)):
        return _load_chrom_sizes_file(Path(chrom_sizes))

    rows = (
        chrom_sizes.items()
        if isinstance(chrom_sizes, Mapping)
        else chrom_sizes
    )
    parsed: dict[str, int] = {}
    for row in rows:
        try:
            chrom, length = row[0], row[1]  # type: ignore[index]
        except (IndexError, TypeError) as exc:
            raise ValueError("chrom_sizes rows must contain chromosome and length") from exc
        _add_chrom_size(parsed, chrom, length, source="chrom_sizes")
    return parsed


def _load_chrom_sizes_file(path: Path) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for line_number, raw_line in enumerate(_open_text(path), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(
                f"{path}:{line_number}: chrom sizes row has fewer than 2 fields"
            )
        if fields[0].lower() in {"chrom", "chromosome"} and fields[1].lower() in {
            "size",
            "length",
        }:
            continue
        try:
            _add_chrom_size(
                parsed,
                fields[0],
                fields[1],
                source=f"{path}:{line_number}",
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid chrom size") from exc
    return parsed


def _add_chrom_size(
    parsed: dict[str, int],
    chrom: Any,
    length: Any,
    *,
    source: str,
) -> None:
    chrom_name = str(chrom)
    if not chrom_name:
        raise ValueError(f"{source}: chromosome name must be non-empty")
    if chrom_name in parsed:
        raise ValueError(f"{source}: duplicate chromosome {chrom_name!r}")
    try:
        chrom_length = int(length)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: chromosome length must be an integer") from exc
    if chrom_length < 0:
        raise ValueError(f"{source}: chromosome length must be non-negative")
    parsed[chrom_name] = chrom_length


def _select_chromosomes(
    chrom_sizes: Mapping[str, int],
    requested: Sequence[str] | None,
) -> list[str]:
    if requested is None:
        return list(chrom_sizes)
    chromosomes: list[str] = []
    seen: set[str] = set()
    for chrom in requested:
        chrom_name = str(chrom)
        if chrom_name in seen:
            raise ValueError(f"duplicate requested chromosome {chrom_name!r}")
        if chrom_name not in chrom_sizes:
            raise KeyError(f"requested chromosome not in chrom_sizes: {chrom_name}")
        seen.add(chrom_name)
        chromosomes.append(chrom_name)
    return chromosomes


def _clip_interval(
    start: int,
    end: int,
    chrom_length: int,
) -> tuple[int, int] | None:
    clipped_start = max(0, int(start))
    clipped_end = min(int(end), int(chrom_length))
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _build_repeat_frac_array(
    chrom_length: int,
    bin_size_bp: int,
    intervals: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    bins = (int(chrom_length) + int(bin_size_bp) - 1) // int(bin_size_bp)
    array = np.zeros((bins, 1), dtype=np.float32)
    if bins == 0:
        return array, 0

    covered = np.zeros(bins, dtype=np.int64)
    for start, end in intervals:
        first_bin = start // bin_size_bp
        last_bin = (end - 1) // bin_size_bp
        if first_bin == last_bin:
            covered[first_bin] += end - start
            continue
        first_bin_end = (first_bin + 1) * bin_size_bp
        last_bin_start = last_bin * bin_size_bp
        covered[first_bin] += first_bin_end - start
        covered[last_bin] += end - last_bin_start
        if last_bin > first_bin + 1:
            covered[first_bin + 1 : last_bin] += bin_size_bp

    array[:, 0] = covered.astype(np.float32) / np.float32(bin_size_bp)
    last_width = chrom_length - (bins - 1) * bin_size_bp
    if last_width != bin_size_bp:
        array[-1, 0] = np.float32(covered[-1]) / np.float32(last_width)
    return array, int(covered.sum())


def _feature_stats(
    values_by_chrom: Sequence[np.ndarray],
    *,
    total_bins: int,
    covered_bases: int,
    eligible_bases: int,
) -> dict[str, Any]:
    if values_by_chrom:
        values = np.concatenate(values_by_chrom).astype(np.float32, copy=False)
        finite = np.isfinite(values)
        finite_values = values[finite]
    else:
        finite_values = np.asarray([], dtype=np.float32)
        finite = np.asarray([], dtype=bool)

    if finite_values.size:
        min_value: float | None = float(np.min(finite_values))
        max_value: float | None = float(np.max(finite_values))
        mean_value: float | None = float(np.mean(finite_values, dtype=np.float64))
    else:
        min_value = None
        max_value = None
        mean_value = None

    weighted_mean: float | None
    if eligible_bases > 0:
        weighted_mean = float(covered_bases / eligible_bases)
    else:
        weighted_mean = None

    return {
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "weighted_mean": weighted_mean,
        "finite_bins": int(finite.sum()),
        "nan_bins": int(total_bins - finite.sum()),
        "covered_bases": int(covered_bases),
        "eligible_bases": int(eligible_bases),
    }


def _source_name(chrom_sizes: ChromSizesInput) -> str:
    if isinstance(chrom_sizes, (str, os.PathLike)):
        return str(Path(chrom_sizes))
    return "in_memory"


def _validate_array_chromosome_name(chrom: str) -> None:
    if chrom in {"", ".", ".."} or "/" in chrom or "\\" in chrom:
        raise ValueError(
            f"chromosome name cannot be used as an array filename: {chrom!r}"
        )


def _publish_staging_dir(staging_dir: Path, output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"genome covariate output already exists: {output_dir}"
            )
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    os.replace(staging_dir, output_dir)
