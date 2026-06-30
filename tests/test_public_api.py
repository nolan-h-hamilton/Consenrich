from pathlib import Path
import re
import subprocess
import sys

import pytest

import consenrich
import consenrich.config
import consenrich.core
import consenrich.io
import consenrich.peaks
import consenrich.uncertainty


def _case_public_api_exports_curated_entrypoints():
    assert consenrich.runConsenrich is consenrich.core.runConsenrich
    assert consenrich.readConfig is consenrich.config.readConfig
    assert consenrich.convertBedGraphToBigWig is consenrich.io.convertBedGraphToBigWig
    assert consenrich.solveRocco is consenrich.peaks.solveRocco
    assert (
        consenrich.calibrateChromosomeStateUncertainty
        is consenrich.uncertainty.calibrateChromosomeStateUncertainty
    )
    assert not hasattr(consenrich, "getMuncTrack")
    assert not hasattr(consenrich, "state_space")


def _case_private_helpers_are_not_package_wildcard_exports():
    assert "_formatDiagValue" not in consenrich.__all__
    assert "_convertSingleBedGraphToBigWig" not in consenrich.__all__
    assert "_logging" not in consenrich.__all__
    assert "_normalization" not in consenrich.__all__
    assert "_runtime" not in consenrich.__all__


def _case_private_helper_modules_are_lazily_available():
    assert consenrich._logging.__name__ == "consenrich._logging"
    assert consenrich._normalization.__name__ == "consenrich._normalization"
    assert consenrich._runtime.__name__ == "consenrich._runtime"


def test_public_api_contract(contract_case):
    contract_case("curated entrypoints", _case_public_api_exports_curated_entrypoints)
    contract_case(
        "private helper wildcard exports",
        _case_private_helpers_are_not_package_wildcard_exports,
    )
    contract_case(
        "private helper modules",
        _case_private_helper_modules_are_lazily_available,
    )


@pytest.mark.parametrize(
    "moduleName",
    (
        "consenrich.cconsenrich",
        "consenrich.ccounts",
        "consenrich.cuncertainty",
    ),
)
def test_native_module_import_fails_when_extension_import_fails(moduleName):
    srcDir = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib
import importlib.abc
import sys

moduleName = sys.argv[1]
sys.path.insert(0, sys.argv[2])

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == moduleName:
            raise ImportError(f"blocked {moduleName}")
        return None

sys.meta_path.insert(0, Blocker())
try:
    importlib.import_module(moduleName)
except ImportError as exc:
    if f"blocked {moduleName}" not in str(exc):
        raise
else:
    raise SystemExit(f"{moduleName} import unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, moduleName, str(srcDir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_src_does_not_probe_required_native_symbols():
    srcDir = Path(__file__).resolve().parents[1] / "src" / "consenrich"
    probePattern = re.compile(
        r"\b(?:hasattr|getattr)\(\s*"
        r"(?:cconsenrich|ccounts|cuncertainty|_cuncertainty)\s*,"
    )
    wordingNeedles = (
        "optional native",
        "optional-native",
        "native extension is optional",
        "native extensions are optional",
    )
    hits = []
    for sourcePath in sorted(srcDir.rglob("*")):
        if sourcePath.suffix not in {".py", ".pyx"}:
            continue
        text = sourcePath.read_text(encoding="utf-8")
        loweredText = text.lower()
        if probePattern.search(text):
            hits.append(f"{sourcePath.relative_to(srcDir)} probes native symbols")
        for needle in wordingNeedles:
            if needle in loweredText:
                hits.append(f"{sourcePath.relative_to(srcDir)} contains {needle!r}")

    assert hits == []
