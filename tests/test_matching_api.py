# -*- coding: utf-8 -*-
import importlib
import types

def test_public_api_shapes():
    mod = importlib.import_module("consenrich.matching")
    # __all__ gate
    assert isinstance(getattr(mod, "__all__", []), list)
    for name in ["matchWavelet", "matchExistingBedGraph", "mergeMatches",
                 "bed_mask_adapter", "sample_block_stats_adapter"]:
        assert hasattr(mod, name), f"missing {name} in consenrich.matching"
