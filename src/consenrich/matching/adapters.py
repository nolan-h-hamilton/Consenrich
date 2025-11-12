from __future__ import annotations
import numpy as np
import numpy.typing as npt

from .. import core as _core
from .. import cconsenrich as _cc


def bed_mask_adapter(
    chrom: str,
    bed_file: str | None,
    intervals: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint8]:
    r"""Wrapper around `consenrich.core.getBedMask`.
    :param chrom: Chromosome name.
    :param bed_file: Path to BED file. If None, returns an array of zeros
    :param intervals: Array of intervals (u32).
    :return: Exclusion mask (u8).
    """

    if bed_file is None:
        return np.zeros(len(intervals), dtype=np.uint8)
    return _core.getBedMask(chrom, bed_file, intervals).astype(
        np.uint8
    )


def sample_block_stats_adapter(
    intervals_u32: npt.NDArray[np.uint32],
    response_f64: npt.NDArray[np.float64],
    rel_window_bins: int,
    nsamp: int,
    seed: int,
    exclude_mask_u8: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    r"""Wrapper around cconsenrich.csampleBlockStats.

    Note the expected types in cconsenrich.csampleBlockStats.

    :param intervals_u32: Array of intervals (u32).
    :param response_f64: Array of response values (f64).
    :param rel_window_bins: Relative window size in (units: intervals, not base pairs)
    :param nsamp: Number of block samples to draw.
    :param seed: Random seed.
    :param exclude_mask_u8: Exclusion mask (u8).
    :return: Array of sampled block statistics (f32).
    """
    try:
        out = _cc.csampleBlockStats(
            intervals_u32,
            response_f64,
            int(rel_window_bins),
            int(nsamp),
            int(seed),
            exclude_mask_u8,
        )
    except ValueError as _ve:
        raise ValueError(
            f"\n\t{_ve}\n"
            f"\n\tEnsure typing is corrrect: `intervals` array must be u32 and `response` array must be f64, etc."
        ) from _ve

    return np.asarray(out, dtype=np.float32)
