"""YAML/default parsing and CLI configuration contracts."""

from __future__ import annotations


def _delegate(name: str, *args, **kwargs):
    from . import consenrich as cli

    return getattr(cli, name)(*args, **kwargs)


def loadConfig(*args, **kwargs):
    return _delegate("loadConfig", *args, **kwargs)


def _cfgGet(*args, **kwargs):
    return _delegate("_cfgGet", *args, **kwargs)


def getInputArgs(*args, **kwargs):
    return _delegate("getInputArgs", *args, **kwargs)


def getOutputArgs(*args, **kwargs):
    return _delegate("getOutputArgs", *args, **kwargs)


def getUncertaintyCalibrationArgs(*args, **kwargs):
    return _delegate("getUncertaintyCalibrationArgs", *args, **kwargs)


def getGenomeArgs(*args, **kwargs):
    return _delegate("getGenomeArgs", *args, **kwargs)


def getStateArgs(*args, **kwargs):
    return _delegate("getStateArgs", *args, **kwargs)


def getCountingArgs(*args, **kwargs):
    return _delegate("getCountingArgs", *args, **kwargs)


def getScArgs(*args, **kwargs):
    return _delegate("getScArgs", *args, **kwargs)


def readConfig(*args, **kwargs):
    return _delegate("readConfig", *args, **kwargs)


__all__ = [
    "_cfgGet",
    "getCountingArgs",
    "getGenomeArgs",
    "getInputArgs",
    "getOutputArgs",
    "getScArgs",
    "getStateArgs",
    "getUncertaintyCalibrationArgs",
    "loadConfig",
    "readConfig",
]
