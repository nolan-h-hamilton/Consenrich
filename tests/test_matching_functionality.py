# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import os
import consenrich.matching as matching

def _pure_python_bed_mask(chrom, bed_file, intervals_u32):
    return np.zeros(len(intervals_u32), dtype=np.uint8)

def _pure_python_sample_block_stats(intervals_u32, response_f32,
                                    rel_window_bins, nsamp, seed, exclude_mask_u8):
    """
    Lightweight stand-in for csampleBlockStats:
    draw 'nsamp' random windows from the allowed (non-excluded) indices and
    return their window maxima. Keeps the distributional intent of the null.
    """
    rng = np.random.default_rng(int(seed))
    n = len(response_f32)
    half = int(rel_window_bins)
    allowed = np.where(exclude_mask_u8 == 0)[0]
    if n == 0 or len(allowed) == 0:
        return np.array([], dtype=np.float32)
    out = []
    for _ in range(int(nsamp)):
        c = int(rng.choice(allowed))
        s = max(0, c - half)
        e = min(n, c + half + 1)
        out.append(float(np.max(response_f32[s:e])))
    return np.asarray(out, dtype=np.float32)

def test_matchWavelet_smoke_injected_adapters(toy_track):
    starts, vals = toy_track
    # make sure evenly spaced
    assert np.all(np.diff(starts) == 10)
    df = matching.matchWavelet(
        chromosome="chr2",
        intervals=starts,
        values=vals,
        templateNames=["haar"],
        cascadeLevels=[3],
        iters=1500,
        alpha=0.2,
        minMatchLengthBP=50,
        minSignalAtMaxima="q:0.70",
        recenterAtPointSource=True,
        useScalingFunction=True,
        # Crucial: inject pure-Python adapters so we don’t depend on C helpers
        get_bed_mask=_pure_python_bed_mask,
        sample_block_stats=_pure_python_sample_block_stats,
    )
    # a dataframe return with those expected columns
    assert list(df.columns) == ["chromosome","start","end","name","score","strand",
                                "signal","pValue","qValue","pointSource"]
    # some candidates should pull out
    assert len(df) >= 1

def test_matchExistingBedGraph_roundtrip(toy_bedgraph, tmp_path):
    out = matching.matchExistingBedGraph(
        bedGraphFile=str(toy_bedgraph),
        templateName="haar",
        cascadeLevel=4,
        alpha=0.2,
        minSignalAtMaxima="q:0.70",
        minMatchLengthBP=50,
        merge=False,
    )
    assert out is not None and os.path.isfile(out)
    with open(out) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert len(lines) >= 1
    fields = lines[0].split("\t")
    assert len(fields) == 10
