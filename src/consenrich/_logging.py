"""Small structured logging utilities shared by Consenrich modules."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np


def log_field_name(value: Any) -> str:
    text = str(value).strip().lower()
    chars: list[str] = []
    last_underscore = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_underscore = False
        elif not last_underscore:
            chars.append("_")
            last_underscore = True
    return "".join(chars).strip("_") or "value"


def log_event_name(value: Any) -> str:
    text = str(value).strip().lower()
    parts: list[str] = []
    token: list[str] = []
    for char in text:
        if char.isalnum():
            token.append(char)
        elif token:
            parts.append("".join(token))
            token = []
    if token:
        parts.append("".join(token))
    return ".".join(parts) or "event"


def quote_log_string(value: str) -> str:
    if value == "":
        return '""'
    if all(char.isalnum() or char in "._:/@%+-" for char in value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def format_log_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        return f"{value_float:.6g}" if np.isfinite(value_float) else "NA"
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size > 12:
            shape = "x".join(str(int(dim)) for dim in value.shape)
            return f"array[{shape}]"
        return quote_log_string(",".join(format_log_value(item) for item in flat))
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            return f"list[{len(value)}]"
        return quote_log_string(",".join(format_log_value(item) for item in value))
    if isinstance(value, Mapping):
        return f"mapping[{len(value)}]"
    text = str(value)
    if "\n" in text:
        text = " ".join(text.split())
    return quote_log_string(text)


def _field_items(fields: Mapping[str, Any] | Sequence[tuple[str, Any]] | None):
    if fields is None:
        return ()
    if isinstance(fields, Mapping):
        return tuple(fields.items())
    return tuple(fields)


def format_log_event(
    event: str,
    fields: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> str:
    parts = [f"event={log_event_name(event)}"]
    for key, value in _field_items(fields):
        parts.append(f"{log_field_name(key)}={format_log_value(value)}")
    return " ".join(parts)


def log_event(
    logger: logging.Logger,
    event: str,
    fields: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    *,
    level: int = logging.INFO,
    stacklevel: int = 2,
) -> None:
    logger.log(level, format_log_event(event, fields), stacklevel=stacklevel)


def progress_enabled(stderr: Any | None = None) -> bool:
    import sys

    stream = sys.stderr if stderr is None else stderr
    return bool(getattr(stream, "isatty", lambda: False)())


def atomic_write(path: str, writer: Callable[[str], None]) -> None:
    """Write a path via a same-directory temporary file then replace atomically."""

    target = os.path.abspath(str(path))
    directory = os.path.dirname(target) or "."
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix="consenrich_write_",
            suffix=".tmp",
            delete=False,
            dir=directory,
        ) as handle:
            temp_path = handle.name
        writer(temp_path)
        os.replace(temp_path, target)
        temp_path = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def log_file_written(
    logger: logging.Logger,
    *,
    event: str,
    path: str,
    fields: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    level: int = logging.INFO,
) -> None:
    payload = list(_field_items(fields))
    payload.extend(
        [
            ("path", str(path)),
            ("bytes", os.path.getsize(path) if os.path.exists(path) else None),
        ]
    )
    log_event(logger, event, payload, level=level, stacklevel=3)


__all__ = [
    "atomic_write",
    "format_log_event",
    "format_log_value",
    "log_event",
    "log_event_name",
    "log_field_name",
    "log_file_written",
    "progress_enabled",
    "quote_log_string",
]
