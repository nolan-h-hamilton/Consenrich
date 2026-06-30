"""Runtime resource helpers shared by IO and fitting code."""

from __future__ import annotations

import logging
import os
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Iterable

_MEMORY_UNSET = object()


def get_small_worker_count(task_count: int, max_workers: int = 4) -> int:
    cpu_count = os.cpu_count() or 1
    return min(int(task_count), max(1, cpu_count // 2), int(max_workers))


def get_available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * page_size
        if total > 0:
            return total // 4
    except (AttributeError, OSError, ValueError):
        pass

    return None


def get_munc_worker_count(
    num_samples: int,
    num_intervals: int,
    shared_arrays: Iterable[Any] = (),
    available_memory_bytes: Any = _MEMORY_UNSET,
    logger: logging.Logger | None = None,
) -> int:
    num_samples = int(num_samples)
    num_intervals = int(num_intervals)
    if num_samples <= 0:
        return 1

    cpu_count = os.cpu_count() or 1
    cpu_workers = min(num_samples, max(1, cpu_count // 2))
    if cpu_workers <= 1:
        return 1

    if available_memory_bytes is _MEMORY_UNSET:
        available_memory_bytes = get_available_memory_bytes()
    if available_memory_bytes is None:
        return cpu_workers

    try:
        available_bytes = int(available_memory_bytes)
    except (TypeError, ValueError):
        return cpu_workers
    if available_bytes <= 0:
        return cpu_workers

    shared_bytes = 0
    for arr in shared_arrays:
        if arr is not None:
            shared_bytes += int(getattr(arr, "nbytes", 0) or 0)

    scratch_bytes = max(
        int(64 * max(num_intervals, 0)) + (64 * 1024 * 1024),
        128 * 1024 * 1024,
    )
    memory_budget = int(max(0, available_bytes - shared_bytes) * 0.5)
    memory_workers = max(1, memory_budget // scratch_bytes)
    workers = max(1, min(cpu_workers, memory_workers))
    if workers < cpu_workers and logger is not None:
        logger.info(
            "munc matrix calculation: reducing workers from %d to %d based on memory estimate "
            "(available=%.2f GiB, shared=%.2f GiB, perWorker=%.2f GiB).",
            int(cpu_workers),
            int(workers),
            available_bytes / (1024**3),
            shared_bytes / (1024**3),
            scratch_bytes / (1024**3),
        )
    return workers


def thread_map(
    items: Iterable[Any],
    func: Callable[[Any], Any],
    label: str,
    *,
    logger: logging.Logger | None = None,
    allow_threads: bool = True,
    min_items: int = 4,
    max_workers: int = 4,
) -> list[Any]:
    item_list = list(items)
    if not item_list:
        return []

    worker_count = get_small_worker_count(len(item_list), max_workers=max_workers)
    use_pool = allow_threads and len(item_list) >= int(min_items) and worker_count > 1
    if use_pool:
        if logger is not None:
            logger.info(
                "%s: using ThreadPool with %d workers (n=%d).",
                label,
                int(worker_count),
                int(len(item_list)),
            )
        with ThreadPool(processes=int(worker_count)) as pool:
            return list(pool.map(func, item_list))

    return [func(item) for item in item_list]
