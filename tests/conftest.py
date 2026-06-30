import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def contract_case():
    def run(label, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from exc

    return run
