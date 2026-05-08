# -*- coding: utf-8 -*-

__version__ = "0.10.5a0"

from importlib import import_module

cconsenrich = import_module(__name__ + ".cconsenrich")
ccounts = import_module(__name__ + ".ccounts")

from .cconsenrich import *
from .ccounts import *
from . import (
    config,
    constants,
    core,
    detrorm,
    misc_util,
    munc,
    peaks,
    regions,
    state_space,
)
from .core import *
from .misc_util import *
from .constants import *
from .detrorm import *
from .peaks import *
from .state_space import *
from .config import *
from .munc import *
from .regions import *

_optional_import_errors = {}
try:
    cuncertainty = import_module(__name__ + ".cuncertainty")
except ImportError as exc:
    if exc.name != __name__ + ".cuncertainty":
        raise
    _optional_import_errors["cuncertainty"] = exc
else:
    try:
        from . import uncertainty
        from .uncertainty import *
    except ImportError as exc:
        if exc.name != __name__ + ".cuncertainty":
            raise
        _optional_import_errors["uncertainty"] = exc

__all__ = sorted(
    name
    for name, value in globals().items()
    if not name.startswith("_") and name not in {"import_module"} and value is not None
)
