"""Command-line tools for Consenrich genome covariate caches."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class CliError(Exception):
    """User-facing CLI validation error."""


BUILDER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("consenrich.genome_covariate_builder", "build_genome_covariate_cache"),
    ("consenrich.genome_covariate_builder", "build_cache"),
    ("consenrich.genome_covariates_builder", "build_genome_covariate_cache"),
    ("consenrich.genome_covariates_builder", "build_cache"),
    ("consenrich.genome_covariates", "build_genome_covariate_cache"),
    ("consenrich.genome_covariates", "build_cache"),
)


def _comma_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise CliError("expected a non-empty comma-separated list")
    return items


def _chromosomes_arg(
    value: str | None,
    *,
    allow_primary: bool,
    chrom_sizes: Path | None = None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    text = value.strip()
    if allow_primary and text.lower() == "primary":
        if chrom_sizes is None:
            raise CliError("'primary' requires --chrom-sizes")
        return _primary_chromosomes(chrom_sizes)
    return _comma_list(text)


def _primary_chromosomes(chrom_sizes: Path) -> tuple[str, ...]:
    names: list[str] = []
    with open(chrom_sizes, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            chrom = fields[0]
            try:
                int(fields[1])
            except ValueError:
                continue
            key = chrom.lower()
            if key.startswith("chr"):
                key = key[3:]
            if (
                "_" in chrom
                or "." in chrom
                or key in {"un", "ebv"}
                or key.endswith("random")
                or key.endswith("alt")
                or key.endswith("fix")
                or key.endswith("hap")
            ):
                continue
            names.append(chrom)
    if not names:
        raise CliError(f"no primary chromosomes found in --chrom-sizes: {chrom_sizes}")
    return tuple(dict.fromkeys(names))


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def convert(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return parsed

    return convert


def _existing_file(path: str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise CliError(f"{label} does not exist: {candidate}")
    if not candidate.is_file():
        raise CliError(f"{label} is not a file: {candidate}")
    return candidate


def _load_builder() -> Callable[..., Any]:
    missed: list[str] = []
    for module_name, attr_name in BUILDER_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if exc.name == module_name:
                missed.append(f"{module_name}.{attr_name}")
                continue
            raise
        builder = getattr(module, attr_name, None)
        if callable(builder):
            return builder
        missed.append(f"{module_name}.{attr_name}")
    expected = ", ".join(missed)
    raise CliError(
        "genome covariate builder API is not available yet; expected one of: "
        f"{expected}"
    )


def _filter_kwargs(
    callable_obj: Callable[..., Any], values: dict[str, Any]
) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return {key: value for key, value in values.items() if value is not None}

    aliases = {
        "out": ("out", "out_dir", "output_dir", "cache_dir"),
        "chrom_sizes": ("chrom_sizes", "chrom_sizes_path", "chrom_sizes_file"),
        "repeat_masker": (
            "repeat_masker",
            "repeat_masker_path",
            "repeat_masker_file",
            "rmsk_path",
        ),
        "repeat_bed": ("repeat_bed", "repeat_bed_path", "repeat_bed_file"),
        "assembly": ("assembly", "assembly_name"),
        "chromosomes": ("chromosomes",),
        "bin_size_bp": ("bin_size_bp", "bin_size"),
        "force": ("force", "overwrite"),
    }
    kwargs: dict[str, Any] = {}
    available = set(signature.parameters)
    for canonical, names in aliases.items():
        if values.get(canonical) is None:
            continue
        for name in names:
            if name in available:
                kwargs[name] = values[canonical]
                break
    return kwargs


def _call_builder(builder: Callable[..., Any], values: dict[str, Any]) -> Any:
    if inspect.isclass(builder):
        instance = builder()
        build_method = getattr(instance, "build", None)
        if not callable(build_method):
            raise CliError("builder class does not expose a callable build() method")
        return build_method(**_filter_kwargs(build_method, values))
    signature = inspect.signature(builder)
    required_params = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required_params) == 1 and required_params[0].name == "spec":
        module = importlib.import_module(builder.__module__)
        spec_type = getattr(module, "GenomeCovariateBuildSpec", None)
        if spec_type is None:
            raise CliError(
                f"{builder.__module__}.build_genome_covariate_cache expects "
                "spec, but GenomeCovariateBuildSpec is not available"
            )
        spec = spec_type(**_filter_kwargs(spec_type, values))
        return builder(spec)
    return builder(**_filter_kwargs(builder, values))


def _build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if out_dir.exists() and not out_dir.is_dir():
        raise CliError(f"--out exists and is not a directory: {out_dir}")

    chrom_sizes = _existing_file(args.chrom_sizes, "--chrom-sizes")
    if args.repeat_masker is None and args.repeat_bed is None:
        raise CliError("one of --repeat-masker or --repeat-bed is required")
    values = {
        "out": out_dir,
        "chrom_sizes": chrom_sizes,
        "repeat_masker": (
            _existing_file(args.repeat_masker, "--repeat-masker")
            if args.repeat_masker is not None
            else None
        ),
        "repeat_bed": (
            _existing_file(args.repeat_bed, "--repeat-bed")
            if args.repeat_bed is not None
            else None
        ),
        "assembly": args.assembly,
        "chromosomes": _chromosomes_arg(
            args.chromosomes,
            allow_primary=True,
            chrom_sizes=chrom_sizes,
        ),
        "bin_size_bp": int(args.bin_size_bp),
        "force": bool(args.force),
    }
    builder = _load_builder()
    result = _call_builder(builder, values)
    print(f"Built cache: {out_dir}")
    manifest_path = getattr(result, "manifest_path", None)
    if manifest_path is not None:
        print(f"Manifest: {manifest_path}")
    result_features = getattr(result, "features", None)
    if result_features:
        print(f"Features: {', '.join(str(feature) for feature in result_features)}")
    result_chromosomes = getattr(result, "chromosomes", None)
    if result_chromosomes:
        print(
            "Chromosomes: "
            + ", ".join(str(chromosome) for chromosome in result_chromosomes)
        )
    return 0


def _validate_cache(
    cache_dir: str,
    *,
    required_features: Sequence[str] | None = None,
    interval_size_bp: int | None = None,
    requested_chromosomes: Sequence[str] | None = None,
):
    from consenrich.genome_covariates import validate_genome_covariate_cache

    return validate_genome_covariate_cache(
        cache_dir,
        required_features=required_features,
        interval_size_bp=interval_size_bp,
        requested_chromosomes=requested_chromosomes,
    )


def _resolve_features(
    value: str | None, default_features: Sequence[str]
) -> tuple[str, ...]:
    from consenrich.genome_covariates import resolve_genome_covariate_feature_config

    return resolve_genome_covariate_feature_config(
        value,
        default_features=default_features,
        available_features=default_features,
        config_name="--features",
    )


def _inspect(args: argparse.Namespace) -> int:
    validation = _validate_cache(args.cache_dir)
    manifest = validation.manifest
    assembly = manifest.get("assembly") or manifest.get("build_parameters", {}).get(
        "assembly", "unknown"
    )
    total_bins = sum(row.bins for row in validation.chromosomes.values())
    total_bp = sum(row.length for row in validation.chromosomes.values())

    print(f"Cache: {Path(args.cache_dir)}")
    print(f"Schema: {manifest.get('schema')}")
    print(f"Assembly: {assembly}")
    print(f"Bin size bp: {validation.bin_size_bp}")
    print(f"Features: {', '.join(validation.features)}")
    print(f"Chromosomes: {len(validation.chromosomes)}")
    print(f"Total bp: {total_bp}")
    print(f"Total bins: {total_bins}")
    for row in validation.chromosomes.values():
        print(f"  {row.name}\tlength={row.length}\tbins={row.bins}\tarray={row.array}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    chromosomes = _comma_list(args.chromosomes) if args.chromosomes is not None else None
    validation = _validate_cache(
        args.cache_dir,
        required_features=(
            _comma_list(args.features) if args.features is not None else None
        ),
        interval_size_bp=args.interval_size_bp,
        requested_chromosomes=chromosomes,
    )
    feature_names = _resolve_features(args.features, validation.features)
    checked = (
        len(chromosomes) if chromosomes is not None else len(validation.chromosomes)
    )

    print(
        "OK: validated "
        f"{checked} chromosome(s), {len(feature_names)} feature(s), "
        f"bin_size_bp={validation.bin_size_bp}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consenrich-cache",
        description="Build, inspect, and validate Consenrich genome covariate caches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="build a genome covariate cache",
        description="Build a Consenrich genome covariate cache.",
    )
    build_parser.add_argument(
        "--out", required=True, metavar="CACHE_DIR", help="Output cache directory."
    )
    build_parser.add_argument(
        "--chrom-sizes", required=True, metavar="PATH", help="Chromosome sizes file."
    )
    build_parser.add_argument(
        "--repeat-masker", metavar="PATH", help="RepeatMasker rmsk file."
    )
    build_parser.add_argument(
        "--repeat-bed", metavar="PATH", help="Repeat interval BED file."
    )
    build_parser.add_argument(
        "--assembly", metavar="NAME", help="Assembly label recorded in the cache."
    )
    build_parser.add_argument(
        "--chromosomes",
        metavar="LIST|primary",
        help="Comma-separated chromosomes, or primary chromosomes from --chrom-sizes.",
    )
    build_parser.add_argument(
        "--bin-size-bp",
        type=_int_at_least(5),
        default=50,
        metavar="N",
        help="Cache bin size in bp, minimum 5. Default: 50.",
    )
    build_parser.add_argument(
        "--force", action="store_true", help="Allow replacing existing cache output."
    )
    build_parser.set_defaults(func=_build)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="summarize a cache",
        description="Print a concise Consenrich genome covariate cache summary.",
    )
    inspect_parser.add_argument(
        "cache_dir", metavar="CACHE_DIR", help="Cache directory to inspect."
    )
    inspect_parser.set_defaults(func=_inspect)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate a cache",
        description="Validate a Consenrich genome covariate cache manifest and arrays.",
    )
    validate_parser.add_argument(
        "cache_dir", metavar="CACHE_DIR", help="Cache directory to validate."
    )
    validate_parser.add_argument(
        "--interval-size-bp",
        type=_int_at_least(1),
        metavar="N",
        help="Optional Consenrich interval size; must be a cache-bin multiple.",
    )
    validate_parser.add_argument(
        "--features",
        metavar="LIST",
        help="Comma-separated feature names to validate. Default: all cache features.",
    )
    validate_parser.add_argument(
        "--chromosomes",
        metavar="LIST",
        help="Comma-separated chromosomes to validate. Default: all cache chromosomes.",
    )
    validate_parser.set_defaults(func=_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except CliError as exc:
        parser.exit(2, f"consenrich-cache: error: {exc}\n")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"consenrich-cache: error: {exc}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
