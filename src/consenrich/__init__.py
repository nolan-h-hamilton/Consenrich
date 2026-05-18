# -*- coding: utf-8 -*-
"""Public package API for Consenrich."""

from ._version import __version__
from .config import readConfig
from .core import runConsenrich
from .io import convertBedGraphToBigWig
from .peaks import solveRocco
from .uncertainty import calibrateChromosomeStateUncertainty

__all__ = [
    "__version__",
    "calibrateChromosomeStateUncertainty",
    "convertBedGraphToBigWig",
    "readConfig",
    "runConsenrich",
    "solveRocco",
]
