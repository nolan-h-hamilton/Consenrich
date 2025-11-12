from __future__ import annotations
import numpy as np
import numpy.typing as npt

from .. import core as _core
from .. import cconsenrich as _cc

def bed_mask_adapter(chrom: str,
                     bed_file: str | None,
                     intervals: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint8]:
    if bed_file is None:
        return np.zeros(len(intervals), dtype=np.uint8)
    return _core.getBedMask(chrom, bed_file, intervals).astype(np.uint8)

def sample_block_stats_adapter(intervals_u32: npt.NDArray[np.uint32],
                               response_f32: npt.NDArray[np.float32],
                               rel_window_bins: int,
                               nsamp: int,
                               seed: int,
                               exclude_mask_u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    out = _cc.csampleBlockStats(intervals_u32,
                                response_f32,
                                int(rel_window_bins),
                                int(nsamp),
                                int(seed),
                                exclude_mask_u8)
    return np.asarray(out, dtype=np.float32)
