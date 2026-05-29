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
