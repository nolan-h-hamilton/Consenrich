# -*- coding: utf-8 -*-

__version__ = "0.8.12rc1"
from importlib import import_module

cconsenrich = import_module(__name__ + ".cconsenrich")
from .cconsenrich import *
from . import core, misc_util, constants, detrorm, matching, mergeNarrowPeaks
from .core import *
from .misc_util import *
from .constants import *
from .detrorm import *
from .matching import *
from .mergeNarrowPeaks import *