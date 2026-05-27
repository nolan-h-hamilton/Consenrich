import contextlib
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from consenrich.genome_covariates import ConsenrichGenomeCovariateCache


def _write_repeat_frac_fixtures(tmp_path):
    chrom_sizes = tmp_path / "mini.chrom.sizes"
    chrom_sizes.write_text(
        "\n".join(
            [
                "chrB\t70",
                "chrA\t125",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rmsk = tmp_path / "mini.rmsk.txt"
    rmsk.write_text(
        "\n".join(
            [
                "585\t1000\t0\t0\t0\tchrA\t10\t40\t-85\t+\tR1\tSINE\tAlu\t0\t30\t0\t1",
                "585\t1000\t0\t0\t0\tchrA\t30\t60\t-65\t+\tR2\tLINE\tL1\t0\t30\t0\t2",
                "585\t1000\t0\t0\t0\tchrA\t115\t130\t0\t+\tR3\tLTR\tERV\t0\t15\t0\t3",
                "585\t1000\t0\t0\t0\tchrB\t0\t10\t-60\t+\tR4\tSimple_repeat\tSimple_repeat\t0\t10\t0\t4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bed = tmp_path / "mini.repeats.bed"
    bed.write_text(
        "\n".join(
            [
                "chrA\t45\t75\tbed_overlap",
                "chrA\t110\t120\tterminal_overlap",
                "chrB\t40\t85\tclipped_past_chrom",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return SimpleNamespace(chrom_sizes=chrom_sizes, rmsk=rmsk, bed=bed)


def _run_cache_cli(argv, *, cwd=None):
    argv = [str(arg) for arg in argv]
    module_name = "consenrich.cache_cli"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        result = subprocess.run(
            [sys.executable, "-m", module_name, *argv],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return SimpleNamespace(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    main = getattr(module, "main", None)
    if main is None:
        result = subprocess.run(
            [sys.executable, "-m", module_name, *argv],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return SimpleNamespace(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exit_code = main(argv)
        except SystemExit as exc:
            exit_code = exc.code
    if exit_code is None:
        exit_code = 0
    if not isinstance(exit_code, int):
        exit_code = 1
    return SimpleNamespace(
        returncode=exit_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _build_repeat_frac_cache(tmp_path):
    fixtures = _write_repeat_frac_fixtures(tmp_path)
    cache_dir = tmp_path / "repeat_frac_cache"
    result = _run_cache_cli(
        [
            "build",
            "--out",
            cache_dir,
            "--chrom-sizes",
            fixtures.chrom_sizes,
            "--bin-size-bp",
            "50",
            "--repeat-masker",
            fixtures.rmsk,
            "--repeat-bed",
            fixtures.bed,
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return cache_dir


def test_consenrich_cache_cli_import_does_not_load_core_runtime(tmp_path):
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(src_dir)
        if not env.get("PYTHONPATH")
        else str(src_dir) + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import consenrich; "
                "assert 'consenrich.core' not in sys.modules; "
                "import consenrich.cache_cli; "
                "assert 'consenrich.core' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_repeat_frac_builder_v1_unions_rmsk_and_bed_intervals(tmp_path):
    cache_dir = _build_repeat_frac_cache(tmp_path)

    cache = ConsenrichGenomeCovariateCache(cache_dir, mmap=False)

    assert cache.features == ("repeat_frac",)
    assert list(cache.chromosomes) == ["chrB", "chrA"]
    np.testing.assert_allclose(
        cache.fetch(
            "chrA",
            start=0,
            end=125,
            feature_names=("repeat_frac",),
            interval_size_bp=50,
        ).ravel(),
        np.asarray([0.8, 0.5, 0.6], dtype=np.float32),
    )
    np.testing.assert_allclose(
        cache.fetch(
            "chrB",
            start=0,
            end=70,
            feature_names=("repeat_frac",),
            interval_size_bp=50,
        ).ravel(),
        np.asarray([0.4, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        cache.fetch(
            "chrA",
            start=0,
            end=125,
            feature_names=("repeat_frac",),
            interval_size_bp=100,
        ).ravel(),
        np.asarray([0.65, 0.6], dtype=np.float32),
    )


def test_consenrich_cache_cli_help_build_inspect_validate(tmp_path):
    help_result = _run_cache_cli(["--help"], cwd=tmp_path)

    assert help_result.returncode == 0, help_result.stderr or help_result.stdout
    help_text = help_result.stdout + help_result.stderr
    assert "build" in help_text
    assert "inspect" in help_text
    assert "validate" in help_text

    cache_dir = _build_repeat_frac_cache(tmp_path)

    inspect_result = _run_cache_cli(["inspect", cache_dir], cwd=tmp_path)
    assert inspect_result.returncode == 0, (
        inspect_result.stderr or inspect_result.stdout
    )
    inspect_text = inspect_result.stdout + inspect_result.stderr
    assert "repeat_frac" in inspect_text
    assert "chrB" in inspect_text
    assert "chrA" in inspect_text

    validate_result = _run_cache_cli(["validate", cache_dir], cwd=tmp_path)
    assert validate_result.returncode == 0, (
        validate_result.stderr or validate_result.stdout
    )


def test_consenrich_cache_cli_validate_rejects_bad_manifest(tmp_path):
    cache_dir = tmp_path / "bad_cache"
    arrays_dir = cache_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    np.save(arrays_dir / "chrA.npy", np.zeros((2, 1), dtype=np.float32))
    (cache_dir / "manifest.json").write_text(
        """{
  "schema": "consenrich-genome-covariates-v1",
  "bin_size_bp": 50,
  "features": ["custom_numeric"],
  "chromosomes": [
    {"name": "chrA", "length": 100, "bins": 3, "array": "arrays/chrA.npy"}
  ]
}
""",
        encoding="utf-8",
    )

    validate_result = _run_cache_cli(["validate", cache_dir], cwd=tmp_path)

    assert validate_result.returncode != 0
    assert "shape" in (validate_result.stdout + validate_result.stderr).lower()
