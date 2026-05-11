import consenrich


def _case_public_api_exports_curated_entrypoints():
    assert consenrich.runConsenrich is consenrich.state_space.runConsenrich
    assert consenrich.readConfig is consenrich.config.readConfig
    assert consenrich.getMuncTrack is consenrich.munc.getMuncTrack
    assert consenrich.chooseDependenceLength is consenrich.regions.chooseDependenceLength
    assert (
        consenrich.calibrateChromosomeStateUncertainty
        is consenrich.uncertainty.calibrateChromosomeStateUncertainty
    )
    assert consenrich.uncertaintyCalibrationParams is consenrich.core.uncertaintyCalibrationParams


def _case_private_helpers_are_not_package_wildcard_exports():
    assert "_formatDiagValue" not in consenrich.__all__
    assert "_convertSingleBedGraphToBigWig" not in consenrich.__all__


def test_public_api_contract(contract_case):
    contract_case("curated entrypoints", _case_public_api_exports_curated_entrypoints)
    contract_case(
        "private helper wildcard exports",
        _case_private_helpers_are_not_package_wildcard_exports,
    )
