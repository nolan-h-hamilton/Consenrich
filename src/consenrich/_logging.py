"""Small structured logging utilities shared by Consenrich modules."""

from __future__ import annotations

import logging
import os
import tempfile
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import structlog


def _reject_json_default(value: Any) -> Any:
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    logger.log(
        level,
        format_log_event(event, fields),
        extra={
            "consenrich_event": log_event_name(event),
            "consenrich_fields": dict(_field_items(fields)),
        },
        stacklevel=stacklevel,
    )


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


def strictJsonValue(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return strictJsonValue(value.tolist())
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON mappings must use string keys")
            out[key] = strictJsonValue(item)
        return out
    if isinstance(value, (list, tuple)):
        return [strictJsonValue(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


_JSON_RENDERER = structlog.processors.JSONRenderer(
    serializer=lambda obj, **kwargs: json.dumps(
        obj,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_reject_json_default,
    )
)


def strictJsonDumps(record: Mapping[str, Any]) -> str:
    if not isinstance(record, Mapping):
        raise TypeError("JSON records must be mappings")
    return _JSON_RENDERER(None, None, strictJsonValue(record))


def _jsonl_open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def init_jsonl_log(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _write_empty(temp_path: str) -> None:
        tempTarget = Path(temp_path)
        if str(target).endswith(".gz"):
            with gzip.open(tempTarget, "wt", encoding="utf-8", newline=""):
                pass
        else:
            tempTarget.write_text("", encoding="utf-8")

    atomic_write(str(target), _write_empty)
    return target


def sizeAdvisoryRecord(
    path: str | os.PathLike[str],
    *,
    bytesWritten: int,
    sizeAdvisoryBytes: int,
    event: str = "artifact.size",
    artifact: str | None = None,
) -> dict[str, Any]:
    if isinstance(bytesWritten, bool) or not isinstance(bytesWritten, int):
        raise TypeError("bytesWritten must be an integer")
    if isinstance(sizeAdvisoryBytes, bool) or not isinstance(sizeAdvisoryBytes, int):
        raise TypeError("sizeAdvisoryBytes must be an integer")
    if sizeAdvisoryBytes <= 0:
        raise ValueError("sizeAdvisoryBytes must be positive")
    record = {
        "record_type": "size_advisory",
        "event": event,
        "path": str(path),
        "bytes": bytesWritten,
        "size_advisory_bytes": sizeAdvisoryBytes,
        "exceeds_advisory": bool(bytesWritten > sizeAdvisoryBytes),
    }
    if artifact is not None:
        record["artifact"] = str(artifact)
    return record


class JsonlWriter:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        append: bool = True,
        sizeAdvisoryBytes: int | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if sizeAdvisoryBytes is not None:
            if isinstance(sizeAdvisoryBytes, bool) or not isinstance(
                sizeAdvisoryBytes,
                int,
            ):
                raise TypeError("sizeAdvisoryBytes must be an integer")
            if sizeAdvisoryBytes <= 0:
                raise ValueError("sizeAdvisoryBytes must be positive")
        self.sizeAdvisoryBytes = sizeAdvisoryBytes
        self.bytesWritten = (
            self.path.stat().st_size
            if append and self.path.exists() and self.sizeAdvisoryBytes is not None
            else 0
        )
        self._advisoryEmitted = False
        self._handle = _jsonl_open(self.path, "at" if append else "wt")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def write(self, record: Mapping[str, Any]) -> int:
        line = strictJsonDumps(record) + "\n"
        self._handle.write(line)
        self.bytesWritten += len(line.encode("utf-8"))
        rowsWritten = 1
        if (
            self.sizeAdvisoryBytes is not None
            and self.bytesWritten > self.sizeAdvisoryBytes
            and not self._advisoryEmitted
        ):
            self._advisoryEmitted = True
            advisory = sizeAdvisoryRecord(
                self.path,
                bytesWritten=self.bytesWritten,
                sizeAdvisoryBytes=self.sizeAdvisoryBytes,
            )
            advisoryLine = strictJsonDumps(advisory) + "\n"
            self._handle.write(advisoryLine)
            self.bytesWritten += len(advisoryLine.encode("utf-8"))
            rowsWritten += 1
        return rowsWritten

    def write_many(self, records: Sequence[Mapping[str, Any]]) -> int:
        rowsWritten = 0
        for record in records:
            rowsWritten += self.write(record)
        return rowsWritten


def append_jsonl_log(
    path: str | os.PathLike[str],
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | Any,
    *,
    sizeAdvisoryBytes: int | None = None,
) -> int:
    import pandas as pd

    if records is None:
        raise TypeError("records must not be None")
    if isinstance(records, Mapping):
        recordList = [records]
    elif isinstance(records, pd.DataFrame):
        recordList = records.to_dict(orient="records")
    else:
        recordList = list(records)
    if not recordList:
        return 0
    with JsonlWriter(path, append=True, sizeAdvisoryBytes=sizeAdvisoryBytes) as writer:
        return writer.write_many(recordList)


__all__ = [
    "JsonlWriter",
    "append_jsonl_log",
    "atomic_write",
    "format_log_event",
    "format_log_value",
    "init_jsonl_log",
    "log_event",
    "log_event_name",
    "log_field_name",
    "log_file_written",
    "quote_log_string",
    "sizeAdvisoryRecord",
    "strictJsonDumps",
    "strictJsonValue",
]
