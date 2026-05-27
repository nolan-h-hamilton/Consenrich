import json

import numpy as np
import pytest

from consenrich.genome_covariates import (
    FEATURES,
    SCHEMA_VERSION,
    ConsenrichGenomeCovariateCache,
)


def _write_cache(tmp_path, chrom="chrTest", bin_size=50, features=FEATURES):
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    full = np.asarray(
        [
            [0.40, 0.00, 0.10],
            [0.60, 0.20, 0.30],
            [np.nan, 0.50, 0.00],
            [0.50, 1.00, 0.20],
        ],
        dtype=np.float32,
    )
    feature_index = {name: idx for idx, name in enumerate(FEATURES)}
    arr = full[:, [feature_index[name] for name in features]]
    np.save(arrays_dir / f"{chrom}.npy", arr, allow_pickle=False)
    manifest = {
        "schema": SCHEMA_VERSION,
        "bin_size_bp": int(bin_size),
        "features": list(features),
        "chromosomes": [
            {
                "name": chrom,
                "length": int(arr.shape[0] * bin_size),
                "bins": int(arr.shape[0]),
                "array": f"arrays/{chrom}.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return arr


def _write_numeric_cache(tmp_path, *, features, values, chrom="chrTest", bin_size=50):
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    arr = np.asarray(values, dtype=np.float32)
    np.save(arrays_dir / f"{chrom}.npy", arr, allow_pickle=False)
    manifest = {
        "schema": SCHEMA_VERSION,
        "bin_size_bp": int(bin_size),
        "features": list(features),
        "chromosomes": [
            {
                "name": chrom,
                "length": int(arr.shape[0] * bin_size),
                "bins": int(arr.shape[0]),
                "array": f"arrays/{chrom}.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return arr


def test_genome_covariate_cache_fetch_exact_and_aggregate(tmp_path):
    arr = _write_cache(tmp_path)
    cache = ConsenrichGenomeCovariateCache(tmp_path)

    exact = cache.fetch("chrTest", start=50, end=150, interval_size_bp=50)
    np.testing.assert_allclose(exact, arr[1:3], equal_nan=True)

    aggregated = cache.fetch("chrTest", start=0, end=200, interval_size_bp=100)
    expected = np.asarray(
        [
            np.nanmean(arr[0:2], axis=0),
            np.nanmean(arr[2:4], axis=0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(aggregated, expected, equal_nan=True)


def test_genome_covariate_cache_rejects_incompatible_interval_size(tmp_path):
    _write_cache(tmp_path)
    with pytest.raises(ValueError, match="bin size must divide"):
        ConsenrichGenomeCovariateCache(tmp_path, interval_size_bp=75)

    cache = ConsenrichGenomeCovariateCache(tmp_path)
    with pytest.raises(ValueError, match="bin size must divide"):
        cache.validate_request(interval_size_bp=75)
    with pytest.raises(ValueError, match="bin size must divide"):
        cache.fetch("chrTest", start=0, end=100, interval_size_bp=75)


def test_genome_covariate_cache_supports_feature_selection_and_partial_fetch(tmp_path):
    arr = _write_cache(tmp_path)
    cache = ConsenrichGenomeCovariateCache(tmp_path)

    selected = cache.fetch(
        "chrTest",
        start=50,
        end=175,
        feature_names=("repeat_frac", "gc"),
        interval_size_bp=100,
    )

    expected = np.asarray(
        [
            [
                np.nanmean(arr[1:3, 2]),
                np.nanmean(arr[1:3, 0]),
            ],
            [
                np.nanmean(arr[3:4, 2]),
                np.nanmean(arr[3:4, 0]),
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(selected, expected, equal_nan=True)


def test_genome_covariate_cache_accepts_arbitrary_numeric_feature_names(tmp_path):
    arr = _write_numeric_cache(
        tmp_path,
        features=("gc", "custom_signal_z", "blacklist_frac"),
        values=[
            [0.40, 1.0, 0.0],
            [0.45, 2.0, 0.5],
            [0.50, 3.0, 1.0],
        ],
    )

    cache = ConsenrichGenomeCovariateCache(tmp_path)

    assert cache.features == ("gc", "custom_signal_z", "blacklist_frac")
    cache.validate_request(
        required_features=("gc_dev", "custom_signal_z"),
        interval_size_bp=100,
    )
    np.testing.assert_allclose(
        cache.fetch(
            "chrTest",
            start=0,
            end=150,
            feature_names=("custom_signal_z",),
            interval_size_bp=100,
        ),
        np.asarray([[1.5], [3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        cache.fetch(
            "chrTest",
            start=50,
            end=150,
            feature_names=("blacklist_frac", "custom_signal_z"),
            interval_size_bp=50,
        ),
        arr[1:3, [2, 1]],
    )
    np.testing.assert_allclose(
        cache.fetch(
            "chrTest",
            start=0,
            end=150,
            feature_names=("gc_dev",),
            interval_size_bp=50,
        ),
        arr[:, [0]],
    )


def test_genome_covariate_cache_reports_missing_chromosome_and_feature(tmp_path):
    _write_cache(tmp_path, features=("gc",))
    cache = ConsenrichGenomeCovariateCache(tmp_path)

    with pytest.raises(KeyError, match="chromosome"):
        cache.fetch("chrMissing", start=0, end=50, interval_size_bp=50)
    with pytest.raises(KeyError, match="features"):
        cache.fetch(
            "chrTest",
            start=0,
            end=50,
            feature_names=("repeat_frac",),
            interval_size_bp=50,
        )


def test_genome_covariate_cache_mmap_toggle(tmp_path):
    _write_cache(tmp_path)
    mmap_cache = ConsenrichGenomeCovariateCache(tmp_path, mmap=True)
    eager_cache = ConsenrichGenomeCovariateCache(tmp_path, mmap=False)

    assert isinstance(mmap_cache.chrom_array("chrTest"), np.memmap)
    assert not isinstance(eager_cache.chrom_array("chrTest"), np.memmap)
    np.testing.assert_allclose(
        mmap_cache.fetch("chrTest", start=0, end=100, interval_size_bp=50),
        eager_cache.fetch("chrTest", start=0, end=100, interval_size_bp=50),
        equal_nan=True,
    )
