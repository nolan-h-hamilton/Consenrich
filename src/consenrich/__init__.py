# -*- coding: utf-8 -*-
"""Public package API for Consenrich."""

from importlib import import_module
from typing import Any

from ._version import __version__

_LAZY_EXPORTS = {
    "calibrateChromosomeStateUncertainty": (
        ".uncertainty",
        "calibrateChromosomeStateUncertainty",
    ),
    "convertBedGraphToBigWig": (".io", "convertBedGraphToBigWig"),
    "readConfig": (".config", "readConfig"),
    "runConsenrich": (".core", "runConsenrich"),
    "solveRocco": (".peaks", "solveRocco"),
}

_PRIVATE_MODULES = frozenset({"_logging", "_normalization", "_runtime"})

__all__ = [
    "__version__",
    "calibrateChromosomeStateUncertainty",
    "convertBedGraphToBigWig",
    "readConfig",
    "runConsenrich",
    "solveRocco",
]


def __getattr__(name: str) -> Any:
    if name in _PRIVATE_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PRIVATE_MODULES})
