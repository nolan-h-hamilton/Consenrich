# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture
def toy_track():
    # 10 kbp region with 10 bp bins for total of 1000 bins
    starts = np.arange(0, 10_000, 10, dtype=np.int32)
    # create fake signal with periodic bumps
    rng = np.random.default_rng(42)
    bg = rng.poisson(1.0, size=len(starts)).astype(float)
    bumps = (np.sin(np.linspace(0, 20*np.pi, len(starts))) > 0.95).astype(float) * 6.0
    vals = (bg + bumps).astype(float)
    return starts, vals

@pytest.fixture
def toy_bedgraph(tmp_path, toy_track):
    starts, vals = toy_track
    df = pd.DataFrame({
        "chromosome": ["chr2"] * len(starts),
        "start": starts,
        "end": starts + 10,
        "value": vals,
    })
    p = Path(tmp_path) / "toy.bedGraph"
    df.to_csv(p, sep="\t", header=False, index=False)
    return p
