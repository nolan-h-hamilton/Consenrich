# -*- coding: utf-8 -*-

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import consenrich.constants as constants
import consenrich.io as consenrich_io
import consenrich.peaks as peaks


def _toyChromState(seed: int = 7, n: int = 512):
    rng = np.random.default_rng(seed)
    noise = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(scale=0.35)

    state = noise.copy()
    state[90:120] += np.hanning(30) * 4.0
    state[300:340] += np.hanning(40) * 3.0
    uncertainty = 0.8 + 0.15 * np.sin(np.linspace(0.0, 3.0 * np.pi, n)) ** 2
    return state, uncertainty.astype(np.float64, copy=False)


def _writeToyBedGraphs(tmp_path: Path):
    state1, unc1 = _toyChromState(seed=7, n=512)
    state2, unc2 = _toyChromState(seed=17, n=384)
    state2 = state2 * 0.7
    unc2 = unc2 * 1.2

    stateRows = []
    uncRows = []
    for chrom, state, unc in [("chr19", state1, unc1), ("chr22", state2, unc2)]:
        starts = np.arange(0, state.size * 50, 50, dtype=np.int64)
        ends = starts + 50
        for start, end, x, u in zip(starts, ends, state, unc):
            stateRows.append((chrom, int(start), int(end), float(x)))
            uncRows.append((chrom, int(start), int(end), float(u)))

    statePath = tmp_path / "toy_state.bedGraph"
    uncPath = tmp_path / "toy_uncertainty.bedGraph"
    pd.DataFrame(stateRows).to_csv(statePath, sep="\t", header=False, index=False)
    pd.DataFrame(uncRows).to_csv(uncPath, sep="\t", header=False, index=False)
    return statePath, uncPath


def _writeSingleChromBedGraphs(
    tmp_path: Path,
    state: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    chrom: str = "chr1",
    step: int = 25,
    stem: str = "single",
):
    starts = np.arange(0, int(state.size) * int(step), int(step), dtype=np.int64)
    ends = starts + int(step)
    statePath = tmp_path / f"{stem}_state.bedGraph"
    pd.DataFrame(
        [
            (str(chrom), int(start), int(end), float(x))
            for start, end, x in zip(starts, ends, state)
        ]
    ).to_csv(statePath, sep="\t", header=False, index=False)
    if uncertainty is None:
        return statePath, None

    uncPath = tmp_path / f"{stem}_uncertainty.bedGraph"
    pd.DataFrame(
        [
            (str(chrom), int(start), int(end), float(x))
            for start, end, x in zip(starts, ends, uncertainty)
        ]
    ).to_csv(uncPath, sep="\t", header=False, index=False)
    return statePath, uncPath


def _assertNoBoundaryGammaMetadata(value):
    if isinstance(value, dict):
        for key, child in value.items():
            keyLower = str(key).lower()
            assert "boundary_gamma" not in keyLower
            assert "per_boundary_gamma" not in keyLower
            _assertNoBoundaryGammaMetadata(child)
    elif isinstance(value, list):
        for child in value:
            _assertNoBoundaryGammaMetadata(child)


@pytest.mark.correctness
def _caseEmpiricalMirroredNullStrengthensThreshold():
    rng = np.random.default_rng(5)
    scoreTrack = rng.normal(loc=0.0, scale=1.0, size=1024)
    scoreTrack[::37] -= 5.0

    nullCenter, nullScale, threshold, details = peaks._estimateEmpiricalMirroredNullForROCCO(
        scoreTrack,
        thresholdZ=2.5,
    )

    assert details["null_method"] == "stationary_null_dwb"
    assert details["null_calibration_method"] == "stationary_null_dwb"
    assert threshold == pytest.approx(details["threshold"])
    assert details["threshold_offset"] >= 0.0
    assert nullScale >= details["core_null_scale"]
    assert details["bootstrap_upper_tail_offset"] > 0.0
    assert details["tail_method"] == "stationary_null_dwb"


@pytest.mark.correctness
def _caseEstimateGammaForROCCOUsesLowerContextBound(monkeypatch):
    scoreTrack = np.linspace(-0.5, 3.5, 256, dtype=np.float64)

    def _fakeChooseFeatureLength(vals, minSpan=3, maxSpan=64):
        return 12, 7, 20, {"method": "feature_peak_width_random_effects"}

    monkeypatch.setattr(peaks.core, "chooseFeatureLength", _fakeChooseFeatureLength)
    gamma, details = peaks.estimateROCCOGamma(
        scoreTrack,
        gamma=-1.0,
        gammaScale=0.5,
        returnDetails=True,
    )

    positiveMedian = float(np.median(scoreTrack[scoreTrack > 0.0]))
    assert details["dependence_span"] == 12
    assert details["gamma_span"] == 7
    assert details["dependence_span_lower"] == 7
    assert details["dependence_span_upper"] == 20
    assert details["context_span_lower"] == 7
    assert details["context_span_upper"] == 20
    assert details["method"] == "dependence_span_lower_score_scale"
    assert np.isclose(gamma, 0.5 * 7.0 * positiveMedian)


@pytest.mark.correctness
def _caseGetBudgetForROCCOUsesDirectConsenrichState():
    state, uncertainty = _toyChromState()
    uncertaintyHi = uncertainty.copy()
    uncertaintyHi[90:120] = uncertaintyHi[90:120] * 3.0
    uncertaintyHi[300:340] = uncertaintyHi[300:340] * 2.5

    budgetBase, _ = peaks.getROCCOBudget(
        state,
        uncertainty=uncertainty,
        numBootstrap=48,
        dependenceSpan=8,
        thresholdZ=2.0,
        nullQuantile=0.80,
        randomSeed=11,
        returnDetails=True,
    )
    budgetNoUnc, _ = peaks.getROCCOBudget(
        state,
        uncertainty=None,
        numBootstrap=48,
        dependenceSpan=8,
        thresholdZ=2.0,
        nullQuantile=0.80,
        randomSeed=11,
        returnDetails=True,
    )
    budgetHiUnc, details = peaks.getROCCOBudget(
        state,
        uncertainty=uncertaintyHi,
        numBootstrap=48,
        dependenceSpan=8,
        thresholdZ=2.0,
        nullQuantile=0.80,
        randomSeed=11,
        returnDetails=True,
    )
    budgetExplicitState, explicitDetails = peaks.getROCCOBudget(
        state,
        uncertainty=uncertaintyHi,
        uncertaintyScoreMode="state",
        numBootstrap=48,
        dependenceSpan=8,
        thresholdZ=2.0,
        nullQuantile=0.80,
        randomSeed=11,
        returnDetails=True,
    )

    assert details["score_mode"] == "consenrich_state"
    assert details["uncertainty_score_mode"] == "state"
    assert details["uncertainty_score_z"] == pytest.approx(
        constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
    )
    assert details["budget_model"] == "dwb_tail_occupancy"
    assert details["method"] == "stationary_null_dwb"
    assert details["null_calibration_method"] == "stationary_null_dwb"
    assert details["threshold_z"] == pytest.approx(2.0)
    assert details["dwb_panel_reused"] is True
    assert details["se_mode"] == "ignored"
    assert details["uncertainty_available"] is True
    assert details["uncertainty_used"] is False
    assert explicitDetails["uncertainty_score_mode"] == "state"
    assert budgetNoUnc == pytest.approx(budgetBase)
    assert budgetHiUnc == pytest.approx(budgetBase)
    assert budgetExplicitState == pytest.approx(budgetBase)
    expectedRawBudget = max(
        0.0,
        float(details["observed_tail_occupancy"])
        - float(details["null_tail_occupancy_calibrated"]),
    )
    assert details["budget_raw"] == pytest.approx(expectedRawBudget)
    assert details["budget_occupancy_raw"] == pytest.approx(expectedRawBudget)


@pytest.mark.correctness
def _caseLowerConfidenceROCCOScoreUsesUncertainty():
    state = np.zeros(128, dtype=np.float64)
    state[20:30] = 5.0
    state[80:90] = 5.0
    uncertainty = np.full(128, 0.1, dtype=np.float64)
    uncertainty[80:90] = 4.0

    defaultPrepared = peaks._prepareROCCOBaseScore(state, uncertainty=uncertainty)
    lowerPrepared = peaks._prepareROCCOBaseScore(
        state,
        uncertainty=uncertainty,
        uncertaintyScoreMode="lower_confidence",
        uncertaintyScoreZ=1.0,
    )

    expected = state - uncertainty
    assert np.allclose(lowerPrepared["score_track"], expected)
    assert np.allclose(defaultPrepared["score_track"], state)

    scoreMeta = lowerPrepared["score_meta"]
    assert scoreMeta["score_mode"] == "lower_confidence"
    assert scoreMeta["uncertainty_score_mode"] == "lower_confidence"
    assert scoreMeta["uncertainty_score_z"] == pytest.approx(1.0)
    assert scoreMeta["uncertainty_used"] is True
    assert scoreMeta["se_mode"] == "used"
    assert scoreMeta["lower_confidence_score_floor_hits"] == 0
    assert (
        lowerPrepared["score_track"][20:30].max()
        > lowerPrepared["score_track"][80:90].max()
    )

    _budget, details = peaks.getROCCOBudget(
        state,
        uncertainty=uncertainty,
        uncertaintyScoreMode="lower_confidence",
        uncertaintyScoreZ=1.0,
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )
    assert details["score_mode"] == "lower_confidence"
    assert details["uncertainty_used"] is True


@pytest.mark.correctness
def _caseLowerConfidenceROCCORequiresUncertainty():
    state = np.zeros(64, dtype=np.float64)
    with pytest.raises(ValueError, match="lower_confidence.*uncertainty"):
        peaks.getROCCOBudget(
            state,
            uncertainty=None,
            uncertaintyScoreMode="lower_confidence",
        )


@pytest.mark.correctness
def _caseGetBudgetForROCCOAppliesSmallPositiveBudgetFloor():
    lowState = np.zeros(512, dtype=np.float64)
    lowUncertainty = np.ones(512, dtype=np.float64)
    budgetLow, detailsLow = peaks.getROCCOBudget(
        lowState,
        uncertainty=lowUncertainty,
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )

    highState, highUncertainty = _toyChromState()
    highState[60:220] += 8.0
    budgetHigh, detailsHigh = peaks.getROCCOBudget(
        highState,
        uncertainty=highUncertainty,
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )

    assert np.isclose(budgetLow, 0.001)
    assert detailsLow["budget_clipped"] is True
    assert detailsLow["budget_raw"] == pytest.approx(0.0)
    assert np.isclose(budgetHigh, constants.ROCCO_BUDGET_MAX)
    assert detailsHigh["budget_clipped"] is True
    assert detailsHigh["budget_raw"] >= constants.ROCCO_BUDGET_MAX


@pytest.mark.correctness
def _caseAutosomalNullFloorHelperStillRuns():
    chr1State, chr1Unc = _toyChromState(seed=7, n=512)
    chr1State[::19] -= 3.0
    chrYState = np.zeros(384, dtype=np.float64)
    chrYUnc = np.full(384, 0.8, dtype=np.float64)

    basePreparedByChrom = {
        "chr1": peaks._prepareROCCOBaseScore(chr1State, uncertainty=chr1Unc),
        "chrY": peaks._prepareROCCOBaseScore(chrYState, uncertainty=chrYUnc),
    }
    pooledNullFloor = peaks._estimateAutosomalNullFloorForROCCO(
        basePreparedByChrom,
        thresholdZ=2.5,
    )
    localPrepared = peaks._prepareROCCOScoreAndNull(
        chrYState,
        uncertainty=chrYUnc,
        thresholdZ=2.5,
    )
    pooledPrepared = peaks._prepareROCCOScoreAndNull(
        chrYState,
        uncertainty=chrYUnc,
        thresholdZ=2.5,
        pooledNullFloor=pooledNullFloor,
    )

    assert pooledNullFloor["source"] == "autosomal_pool"
    assert pooledPrepared["threshold"] >= localPrepared["threshold"]
    assert pooledPrepared["null_scale"] >= localPrepared["null_scale"]
    assert pooledPrepared["pooled_null_floor"]["source"] == "autosomal_pool"


@pytest.mark.correctness
def _caseROCCONullFallbackAndEBShrinkage():
    state, _ = _toyChromState()
    nullCenter, nullScale, details = peaks.estimateROCCONull(np.abs(state) + 3.0)

    assert details["null_method"] == "mode_centered_central_support"
    assert details["center_method"] == "lower_bulk_half_sample_mode"
    assert np.isfinite(nullCenter)
    assert np.isfinite(nullScale)
    assert nullScale > 0.0
    assert details["provisional_center"] > 2.5
    assert details["null_center"] > 2.5
    assert details["support_fraction"] > 0.25
    assert details["support_radius"] > 0.0

    shrunk, meta = peaks.shrinkROCCOBudgets(
        {"chr1": 2.0, "chr2": 40.0, "chr3": 15.0},
        {"chr1": 100.0, "chr2": 100.0, "chr3": 100.0},
    )

    assert 0.0 < meta["genome_wide_budget"] < 1.0
    assert meta["min_prior_concentration"] >= 8.0
    assert meta["posterior_estimator"] == "mean"
    assert meta["posterior_quantile"] is None
    assert shrunk["chr1"] < shrunk["chr3"] < shrunk["chr2"]


@pytest.mark.correctness
def _caseRunROCCOAlgorithmFromBedGraphs(tmp_path):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "toy_rocco.narrowPeak"
    metaPath = tmp_path / "toy_rocco.narrowPeak.json"

    resultPath = peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        numBootstrap=24,
        dependenceSpan=8,
        randSeed=11,
        outPath=str(outPath),
        metaPath=str(metaPath),
        stateDiagnosticsByChromosome={
            "chr19": {
                "state_roughness": {
                    "overall_mean_abs_diff": 1.25,
                },
            },
        },
    )

    assert Path(resultPath).is_file()
    assert metaPath.is_file()
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["metadata_detail"] == "compact"
    assert meta["settings"]["metadata_detail"] == "compact"
    assert meta["settings"]["budget_method"] == "dwb_tail_occupancy"
    assert meta["settings"]["null_calibration_method"] == "stationary_null_dwb"
    assert meta["pooled_null_floor"] is None
    assert meta["budget_shrinkage"] is None
    assert meta["chromosomes"]["chr19"]["state_diagnostics"]["state_roughness"][
        "overall_mean_abs_diff"
    ] == pytest.approx(1.25)
    assert meta["chromosomes"]["chr22"]["state_diagnostics"] == {}
    for chrom in meta["chromosomes"].values():
        assert "budget_pre_shrink" in chrom["budget_details"]
        assert "budget_post_shrink" in chrom["budget_details"]
        assert chrom["budget_details"]["budget_pre_shrink"] == pytest.approx(
            chrom["budget_details"]["budget_post_shrink"]
        )
        assert chrom["budget_details"]["budget_shrinkage_meta"] is None
        assert "peak_details" not in chrom
        assert "candidate_details" not in chrom
        assert chrom["peak_details_omitted"] >= chrom["num_segments"]
        assert chrom["candidate_details_omitted"] >= chrom["num_segments"]
    lines = outPath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    assert lines[0].startswith("chr")


def _caseRunROCCOLowerConfidenceRecordsMetadata(tmp_path):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "toy_lcb_rocco.narrowPeak"
    metaPath = tmp_path / "toy_lcb_rocco.narrowPeak.json"

    resultPath = peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        uncertaintyScoreMode="lower_confidence",
        uncertaintyScoreZ=1.0,
        numBootstrap=24,
        dependenceSpan=8,
        nestedRoccoIters=0,
        massiveSubpeakCleanup=False,
        randSeed=11,
        outPath=str(outPath),
        metaPath=str(metaPath),
    )

    assert resultPath == str(outPath)
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["settings"]["uncertainty_score_mode"] == "lower_confidence"
    assert meta["settings"]["uncertainty_score_z"] == pytest.approx(1.0)
    assert meta["settings"]["dwb_null_enabled"] is True
    chromDetails = next(iter(meta["chromosomes"].values()))
    assert chromDetails["budget_details"]["score_mode"] == "lower_confidence"
    assert chromDetails["budget_details"]["uncertainty_used"] is True


def _caseSolveRoccoReturnsSummaryInventoryAndLogs(tmp_path, caplog):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "summary_rocco.narrowPeak"
    metaPath = tmp_path / "summary_rocco.narrowPeak.json"
    detailPath = Path(f"{outPath}.nested_rocco_subproblems.jsonl")

    with caplog.at_level(logging.INFO, logger="consenrich.peaks"):
        resultPath, summary = peaks.solveRocco(
            str(statePath),
            uncertaintyBedGraphFile=str(uncPath),
            numBootstrap=24,
            dependenceSpan=8,
            randSeed=11,
            outPath=str(outPath),
            metaPath=str(metaPath),
            verbose=True,
            returnSummary=True,
        )

    assert resultPath == str(outPath)
    assert summary["narrowPeak_path"] == str(outPath)
    assert summary["metadata_json_path"] == str(metaPath)
    assert summary["nested_jsonl_path"] == str(detailPath)
    assert summary["exported_peak_count"] >= 1
    assert summary["total_peak_bp"] > 0
    assert summary["min_width_bp"] <= summary["median_width_bp"]
    assert summary["median_width_bp"] <= summary["max_width_bp"]
    assert summary["blacklist"]["dropped"] == 0
    assert summary["blacklist"]["kept"] == summary["exported_peak_count"]
    assert set(summary["per_chrom"]) == {"chr19", "chr22"}
    assert summary["nested_rocco"]["requested_iters"] == constants.MATCHING_DEFAULT_NESTED_ROCCO_ITERS
    assert summary["nested_rocco"]["diagnostics"] is True
    assert summary["nested_rocco"]["subproblem_details"] == str(detailPath)
    assert summary["nested_rocco"]["stops"]["chr19"]["completed_iters"] >= 1
    assert summary["files"][0]["kind"] == "narrowPeak"
    assert summary["files"][0]["bytes"] == outPath.stat().st_size
    assert summary["files"][1]["kind"] == "metadata_json"
    assert summary["files"][1]["bytes"] == metaPath.stat().st_size
    assert summary["files"][2]["kind"] == "nested_rocco_subproblems_jsonl"
    assert summary["files"][2]["bytes"] == detailPath.stat().st_size
    assert "rocco.summary peaks=" in caplog.text
    assert "output.inventory" in caplog.text


def _caseSolveRoccoMetadataCapFallsBackToBounded(tmp_path, caplog):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "bounded_rocco.narrowPeak"
    metaPath = tmp_path / "bounded_rocco.narrowPeak.json"

    with caplog.at_level(logging.WARNING, logger="consenrich.peaks"):
        peaks.solveRocco(
            str(statePath),
            uncertaintyBedGraphFile=str(uncPath),
            numBootstrap=24,
            dependenceSpan=8,
            randSeed=11,
            outPath=str(outPath),
            metaPath=str(metaPath),
            metadataDetail="full",
            maxNonTrackFileBytes=1,
        )

    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["metadata_detail"] == "bounded"
    assert meta["settings"]["metadata_requested_detail"] == "full"
    assert meta["settings"]["metadata_byte_cap"] == 1
    for chrom in meta["chromosomes"].values():
        assert "peak_details" not in chrom
        assert "candidate_details" not in chrom
        assert chrom["peak_details_omitted"] >= chrom["num_segments"]
        assert chrom["candidate_details_omitted"] >= chrom["num_segments"]
    assert "ROCCO metadata bounded" in caplog.text


def _caseSolveRoccoCutoffReportWritesSweeps(tmp_path, monkeypatch):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    monkeypatch.setattr(
        peaks,
        "_ROCCO_CUTOFF_REPORT_SWEEPS",
        (
            (
                "thresholdZ",
                "matchingParams.thresholdZ",
                "thresholdZ",
                (1.5,),
                {},
                False,
            ),
            (
                "uncertaintyScoreZ",
                "matchingParams.uncertaintyScoreZ",
                "uncertaintyScoreZ",
                (0.5,),
                {"uncertaintyScoreMode": "lower_confidence"},
                True,
            ),
            (
                "nestedRoccoBudgetScale",
                "matchingParams.nestedRoccoBudgetScale",
                "nestedRoccoBudgetScale",
                (0.5,),
                {},
                False,
            ),
        ),
    )

    reportDir = Path(
        peaks.solveRoccoCutoffReport(
            str(statePath),
            uncertaintyBedGraphFile=str(uncPath),
            numBootstrap=12,
            dependenceSpan=8,
            randSeed=13,
        )
    )
    assert reportDir.name.endswith("_rocco_cutoff_analysis")
    assert ("m" + "acs") not in reportDir.name.lower()

    summaryPath = reportDir / "cutoff_summary.tsv"
    assert summaryPath.exists()
    summary = pd.read_csv(summaryPath, sep="\t").fillna("")
    assert list(summary["sweep"]) == [
        "baseline",
        "thresholdZ",
        "uncertaintyScoreZ",
        "nestedRoccoBudgetScale",
    ]
    assert set(summary["parameter"]) == {
        "",
        "matchingParams.thresholdZ",
        "matchingParams.uncertaintyScoreZ",
        "matchingParams.nestedRoccoBudgetScale",
    }
    assert set(summary["uncertaintyScoreMode"]) == {"state", "lower_confidence"}
    for pathText in summary["narrowPeak_path"]:
        assert Path(pathText).exists()
    assert "metadata_json_path" not in summary.columns
    assert list(reportDir.glob("*.narrowPeak.json")) == []


@pytest.mark.correctness
def _caseRunROCCOAlgorithmKeepsShortFlatEnrichment(tmp_path):
    n = 80
    starts = np.arange(0, n * 50, 50, dtype=np.int64)
    ends = starts + 50
    state = np.zeros(n, dtype=np.float64)
    state[37:46] = 10.0
    uncertainty = np.ones(n, dtype=np.float64)
    statePath = tmp_path / "flat_state.bedGraph"
    uncPath = tmp_path / "flat_uncertainty.bedGraph"
    outPath = tmp_path / "flat_rocco.narrowPeak"
    metaPath = tmp_path / "flat_rocco.narrowPeak.json"
    pd.DataFrame(
        [("chr1", int(start), int(end), float(x)) for start, end, x in zip(starts, ends, state)]
    ).to_csv(statePath, sep="\t", header=False, index=False)
    pd.DataFrame(
        [("chr1", int(start), int(end), float(x)) for start, end, x in zip(starts, ends, uncertainty)]
    ).to_csv(uncPath, sep="\t", header=False, index=False)

    peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        numBootstrap=24,
        dependenceSpan=8,
        outPath=str(outPath),
        metaPath=str(metaPath),
    )

    lines = outPath.read_text(encoding="utf-8").strip().splitlines()
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    chromMeta = meta["chromosomes"]["chr1"]
    fields = lines[0].split("\t")

    assert len(lines) == 1
    assert int(fields[1]) >= int(starts[37])
    assert int(fields[2]) <= int(ends[45])
    assert isinstance(chromMeta["solve_details"]["budget_fallback_window"], bool)
    assert chromMeta["solve_details"]["first_pass_selected_count"] > 0
    assert chromMeta["solve_details"]["final_selected_count"] > 0
    assert (
        chromMeta["nested_rocco_details"]["history"][0][
            "num_budget_fallback_windows"
        ]
        >= 0
    )


@pytest.mark.correctness
def _caseRunROCCODropsBlacklistOverlapsAndRecordsMetadata(tmp_path):
    n = 80
    starts = np.arange(0, n * 50, 50, dtype=np.int64)
    ends = starts + 50
    state = np.zeros(n, dtype=np.float64)
    state[37:46] = 10.0
    uncertainty = np.ones(n, dtype=np.float64)
    statePath = tmp_path / "blacklist_state.bedGraph"
    uncPath = tmp_path / "blacklist_uncertainty.bedGraph"
    blacklistPath = tmp_path / "blacklist.bed"
    outPath = tmp_path / "blacklist_rocco.narrowPeak"
    metaPath = tmp_path / "blacklist_rocco.narrowPeak.json"
    pd.DataFrame(
        [
            ("chr1", int(start), int(end), float(x))
            for start, end, x in zip(starts, ends, state)
        ]
    ).to_csv(statePath, sep="\t", header=False, index=False)
    pd.DataFrame(
        [
            ("chr1", int(start), int(end), float(x))
            for start, end, x in zip(starts, ends, uncertainty)
        ]
    ).to_csv(uncPath, sep="\t", header=False, index=False)
    blacklistPath.write_text(
        f"chr1\t{int(starts[40])}\t{int(ends[40])}\n",
        encoding="utf-8",
    )

    peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        blacklistBedFile=str(blacklistPath),
        numBootstrap=24,
        dependenceSpan=8,
        outPath=str(outPath),
        metaPath=str(metaPath),
    )

    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert outPath.read_text(encoding="utf-8").strip() == ""
    assert meta["settings"]["blacklist_bed"] == str(blacklistPath)
    assert meta["blacklist_filter"]["dropped"] == 1
    assert meta["blacklist_filter"]["kept"] == 0
    assert meta["chromosomes"]["chr1"]["export_details"]["blacklist_filter"][
        "dropped"
    ] == 1


@pytest.mark.correctness
def _caseSolveRoccoAppliesMinPeakSignalFilter(tmp_path):
    n = 128
    state = np.zeros(n, dtype=np.float64)
    state[18:26] = 8.0
    state[58:68] = 5.0
    state[95:105] = 4.0
    uncertainty = np.ones(n, dtype=np.float64)
    statePath, uncPath = _writeSingleChromBedGraphs(
        tmp_path,
        state,
        uncertainty,
        stem="min_peak_score",
    )
    outPath = tmp_path / "min_peak_score_rocco.narrowPeak"
    metaPath = tmp_path / "min_peak_score_rocco.narrowPeak.json"

    resultPath, summary = peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        numBootstrap=16,
        dependenceSpan=6,
        randSeed=19,
        gamma=0.0,
        nestedRoccoIters=0,
        massiveSubpeakCleanup=False,
        minPeakScore=7.0,
        outPath=str(outPath),
        metaPath=str(metaPath),
        metadataDetail="full",
        returnSummary=True,
    )

    rows = [
        line.split("\t")
        for line in outPath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    chromMeta = meta["chromosomes"]["chr1"]
    exportDetails = chromMeta["export_details"]

    assert resultPath == str(outPath)
    assert rows
    assert all(float(row[6]) >= 7.0 for row in rows)
    assert summary["exported_peak_count"] == len(rows)
    assert summary["exported_peak_count"] == chromMeta["num_segments"]
    assert summary["settings"]["min_peak_score"] == pytest.approx(7.0)
    assert meta["settings"]["min_peak_score"] == pytest.approx(7.0)
    assert exportDetails["min_peak_score"] == pytest.approx(7.0)
    assert exportDetails["min_peak_score_field"] == "signalValue"
    assert exportDetails["min_peak_score_narrowpeak_column"] == 7
    assert exportDetails["min_peak_score_filter_active"] is True
    assert exportDetails["num_segments_min_peak_score_evaluated"] == (
        len(rows) + exportDetails["num_segments_dropped_min_peak_score"]
    )
    assert exportDetails["num_segments_dropped_min_peak_score"] >= 1
    assert all("dwb_empirical_q" in peak for peak in chromMeta["peak_details"])


@pytest.mark.correctness
def _caseIntegratedBudgetUsesExcessTailGrid():
    rng = np.random.default_rng(3)
    state = np.zeros(1024, dtype=np.float64)
    for i in range(1, state.size):
        state[i] = 0.95 * state[i - 1] + rng.normal(scale=0.05)
    state[300:340] += 0.01
    prepared = peaks._prepareROCCOScoreAndNull(
        state,
        uncertainty=np.ones_like(state),
        thresholdZ=2.0,
        thresholdZGrid=(1.5, 2.0, 2.5, 3.0),
    )

    thresholdZGrid = (1.5, 2.0, 2.5, 3.0)
    budgetOcc, detailsOcc = peaks._estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic="occupancy",
        numBootstrap=24,
        dependenceSpan=16,
        randomSeed=11,
        budgetMax=1.0,
        thresholdZGrid=thresholdZGrid,
        returnDetails=True,
    )
    budgetIntegrated, detailsIntegrated = peaks._estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic="integrated",
        numBootstrap=24,
        dependenceSpan=16,
        randomSeed=11,
        budgetMax=1.0,
        thresholdZGrid=thresholdZGrid,
        returnDetails=True,
    )

    assert detailsIntegrated["threshold_z_grid"] == pytest.approx(thresholdZGrid)
    assert len(detailsIntegrated["threshold_metrics"]) == len(thresholdZGrid)
    assert detailsIntegrated["dwb_panel_reused"] is True
    expectedIntegrated = float(
        np.mean(
            [
                float(metrics["budget_soft_raw"])
                for metrics in detailsIntegrated["threshold_metrics"].values()
            ]
        )
    )
    assert budgetIntegrated == pytest.approx(expectedIntegrated)
    assert budgetIntegrated != pytest.approx(budgetOcc)
    assert detailsIntegrated["budget_integrated_raw"] == pytest.approx(budgetIntegrated)
    assert detailsOcc["budget_occupancy_raw"] == pytest.approx(budgetOcc)


@pytest.mark.correctness
def _casePreparedStationaryNullDWBUsesSharedPanelAndMonotoneThresholds():
    state, uncertainty = _toyChromState(seed=9, n=512)
    prepared = peaks._prepareROCCOScoreAndNull(
        state,
        uncertainty=uncertainty,
        thresholdZ=2.0,
        thresholdZGrid=(1.5, 2.0, 2.5, 3.0),
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=11,
        nullQuantile=0.80,
    )
    _budget, details = peaks._estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic="integrated",
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )

    thresholds = [
        float(prepared["threshold_views"][peaks._thresholdZKey(z)]["threshold"])
        for z in (1.5, 2.0, 2.5, 3.0)
    ]
    panelIds = {
        str(view["null_meta"]["dwb_panel_id"])
        for view in prepared["threshold_views"].values()
    }

    assert thresholds == sorted(thresholds)
    assert len(panelIds) == 1
    assert details["dwb_panel_id"] in panelIds
    assert all(
        metric["dwb_panel_id"] == details["dwb_panel_id"]
        for metric in details["threshold_metrics"].values()
    )


@pytest.mark.correctness
def _caseSolveRoccoAnnotatesPeakLevelDwbEmpiricalPQ(tmp_path):
    n = 96
    state = np.zeros(n, dtype=np.float64)
    state[18:26] = 6.0
    state[58:68] = 4.5
    uncertainty = np.ones(n, dtype=np.float64)
    statePath, uncPath = _writeSingleChromBedGraphs(
        tmp_path,
        state,
        uncertainty,
        stem="peak_pq",
    )
    outPath = tmp_path / "peak_pq_rocco.narrowPeak"
    metaPath = tmp_path / "peak_pq_rocco.narrowPeak.json"

    peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        numBootstrap=16,
        dependenceSpan=6,
        randSeed=19,
        gamma=0.0,
        nestedRoccoIters=0,
        massiveSubpeakCleanup=False,
        outPath=str(outPath),
        metaPath=str(metaPath),
        metadataDetail="full",
    )

    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    chromMeta = meta["chromosomes"]["chr1"]
    peakDetails = chromMeta["peak_details"]
    assert peakDetails
    assert chromMeta["budget_details"]["null_calibration_method"] == "stationary_null_dwb"
    scoring = chromMeta["export_details"]["dwb_peak_scoring"]
    assert scoring["p_value"] == "empirical_replay_segment_tail"
    assert scoring["q_value"] == "dwb_replay_fdr_candidate_segments"
    assert scoring["primary_metric"] == "width_adjusted_mass"
    candidateDetails = chromMeta["candidate_details"]
    assert len(candidateDetails) >= len(peakDetails)
    assert chromMeta["candidate_significance"]["num_candidates"] == len(candidateDetails)
    assert all("exported_peak" in peak["candidate_sources"] for peak in peakDetails)
    assert all(peak["candidate_scale_bins"] for peak in peakDetails)
    for peak in peakDetails:
        assert peak["dwb_empirical_method"] == "stationary_null_dwb_peak_replay"
        assert peak["dwb_empirical_panel_id"] == chromMeta["budget_details"]["dwb_panel_id"]
        assert peak["dwb_empirical_null_replays"] >= 8
        assert 0.0 <= float(peak["dwb_empirical_p"]) <= 1.0
        assert 0.0 <= float(peak["dwb_empirical_q"]) <= 1.0
        assert float(peak["dwb_empirical_q"]) >= float(peak["dwb_empirical_p"])
        assert np.isfinite(float(peak["dwb_empirical_statistic"]))
        for metric in ("summit_excess", "integrated_excess", "width_adjusted_mass"):
            assert np.isfinite(float(peak[metric]))
            assert 0.0 <= float(peak[f"{metric}_p"]) <= 1.0
            assert 0.0 <= float(peak[f"{metric}_q"]) <= 1.0
            assert float(peak[f"{metric}_q"]) >= float(peak[f"{metric}_p"])

    sortedByStatistic = sorted(
        (
            float(peak["dwb_empirical_statistic"]),
            float(peak["dwb_empirical_q"]),
        )
        for peak in peakDetails
    )[::-1]
    qByStatistic = [q for _statistic, q in sortedByStatistic]
    assert all(
        left <= right + 1.0e-12
        for left, right in zip(qByStatistic, qByStatistic[1:])
    )
    narrowRows = [
        line.split("\t")
        for line in outPath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(narrowRows) == len(peakDetails)
    for row, peak in zip(narrowRows, peakDetails):
        expectedP = min(-np.log10(max(float(peak["dwb_empirical_p"]), 1.0e-300)), 1000.0)
        expectedQ = (
            1000.0
            if float(peak["dwb_empirical_q"]) <= 0.0
            else min(-np.log10(float(peak["dwb_empirical_q"])), 1000.0)
        )
        assert float(row[7]) == pytest.approx(expectedP)
        assert float(row[8]) == pytest.approx(expectedQ)
        assert float(row[8]) <= float(row[7]) + 1.0e-12
    _assertNoBoundaryGammaMetadata(peakDetails)


@pytest.mark.correctness
def _caseReplayFDRModeratePanelsStaySubquadratic():
    empiricalP = getattr(peaks, "_empiricalReplaySegmentPValues", None)
    replayQ = getattr(peaks, "_replayFDRQValues", None)
    assert empiricalP is not None
    assert replayQ is not None

    rng = np.random.default_rng(271)
    observed = rng.gamma(shape=2.5, scale=1.0, size=6000)
    nullDraws = [
        rng.gamma(shape=2.2, scale=1.0, size=3000)
        for _ in range(32)
    ]

    started = time.perf_counter()
    pValues = empiricalP(observed, nullDraws)
    qValues = np.maximum(replayQ(observed, nullDraws), pValues)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert pValues.shape == observed.shape
    assert qValues.shape == observed.shape
    assert np.all(np.isfinite(pValues))
    assert np.all(np.isfinite(qValues))
    assert np.all((0.0 <= pValues) & (pValues <= 1.0))
    assert np.all((0.0 <= qValues) & (qValues <= 1.0))
    assert np.all(qValues + 1.0e-12 >= pValues)
    assert np.any(qValues > pValues + 1.0e-6)

    order = np.argsort(-observed, kind="mergesort")
    pByStatistic = pValues[order]
    qByStatistic = qValues[order]
    assert np.all(pByStatistic[:-1] <= pByStatistic[1:] + 1.0e-12)
    assert np.all(qByStatistic[:-1] <= qByStatistic[1:] + 1.0e-12)


@pytest.mark.correctness
def _caseDWBPeakScoringModerateReplayStaysBoundedAndSane():
    addScoring = getattr(peaks, "_addDWBPeakScoringToPeakMeta", None)
    assert addScoring is not None

    rng = np.random.default_rng(503)
    n = 3500
    scores = rng.normal(size=n)
    peakStarts = list(range(100, 3400, 300))
    for start in peakStarts:
        width = 25
        scores[start : start + width] += np.hanning(width) * 4.0

    thresholdViews = {
        f"z{z}": {
            "threshold_z": float(z),
            "threshold": float(z),
            "null_scale": 1.0,
            "null_center": 0.0,
        }
        for z in (-0.25, 0.0, 0.25, 0.5)
    }
    prepared = {
        "threshold_views": thresholdViews,
        "template": rng.normal(size=n),
        "dwb_calibration": {
            "dependence_span": 16,
            "dependence_span_lower": 8,
            "dependence_span_upper": 24,
            "kernel": "bartlett",
            "num_bootstrap": 12,
            "random_seed": 127,
            "dwb_panel_id": "synthetic-performance-regression",
        },
    }
    peakMeta = []
    for idx, start in enumerate(peakStarts, start=1):
        end = start + 24
        peakMeta.append(
            {
                "name": f"synthetic_peak_{idx}",
                "start_idx": int(start),
                "end_idx": int(end),
                "start": int(start * 25),
                "end": int((end + 1) * 25),
                "summit_idx": int(start + np.argmax(scores[start : end + 1])),
            }
        )
    intervals = np.arange(n, dtype=np.int64) * 25
    ends = intervals + 25
    exportDetails = {}

    started = time.perf_counter()
    summary = addScoring(
        peakMeta,
        scores,
        prepared,
        exportDetails=exportDetails,
        minRunBins=1,
        intervals=intervals,
        ends=ends,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 4.0
    assert summary["enabled"] is True
    assert summary["num_bootstrap"] == 12
    assert summary["num_candidate_regions"] >= len(peakMeta)
    assert summary["num_candidate_regions"] <= 10000 + len(peakMeta)
    assert summary["null_replay_candidate_count_q95"] >= summary[
        "null_replay_candidate_count_mean"
    ]
    assert exportDetails["candidate_significance"]["num_candidates"] == summary[
        "num_candidate_regions"
    ]
    diagnostics = exportDetails["null_replay_false_segment_diagnostics"]
    assert diagnostics["num_replays"] == 12
    assert diagnostics["observed_segment_count"] == len(peakMeta)
    assert diagnostics["observed_candidate_count"] == summary["num_candidate_regions"]

    qByStatistic = []
    for peak in peakMeta:
        assert 0.0 <= float(peak["dwb_empirical_p"]) <= 1.0
        assert 0.0 <= float(peak["dwb_empirical_q"]) <= 1.0
        assert float(peak["dwb_empirical_q"]) >= float(peak["dwb_empirical_p"])
        assert peak["candidate_scale_bins"]
        qByStatistic.append(
            (
                float(peak["dwb_empirical_statistic"]),
                float(peak["dwb_empirical_q"]),
            )
        )
    qByStatistic.sort(reverse=True)
    qValues = [q for _stat, q in qByStatistic]
    assert any(
        float(peak["dwb_empirical_q"]) > float(peak["dwb_empirical_p"]) + 1.0e-6
        for peak in peakMeta
    )
    assert all(left <= right + 1.0e-12 for left, right in zip(qValues, qValues[1:]))


@pytest.mark.correctness
def _caseGetBudgetForROCCOIsStableUnderFixedSeed():
    state, uncertainty = _toyChromState(seed=13, n=384)
    budget1, details1 = peaks.getROCCOBudget(
        state,
        uncertainty=uncertainty,
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=21,
        returnDetails=True,
    )
    budget2, details2 = peaks.getROCCOBudget(
        state,
        uncertainty=uncertainty,
        numBootstrap=24,
        dependenceSpan=8,
        randomSeed=21,
        returnDetails=True,
    )

    assert budget1 == pytest.approx(budget2)
    assert details1["threshold"] == pytest.approx(details2["threshold"])
    assert details1["dwb_panel_id"] == details2["dwb_panel_id"]


@pytest.mark.correctness
def _caseEstimateGammaForROCCOUsesCenteredExcessWhenAvailable():
    scoreTrack = np.linspace(1.0, 5.0, 256, dtype=np.float64)

    gammaRaw, detailsRaw = peaks.estimateROCCOGamma(
        scoreTrack,
        dependenceSpan=8,
        gamma=-1.0,
        gammaScale=0.5,
        returnDetails=True,
    )
    gammaCentered, detailsCentered = peaks.estimateROCCOGamma(
        scoreTrack,
        dependenceSpan=8,
        gamma=-1.0,
        gammaScale=0.5,
        threshold=3.0,
        returnDetails=True,
    )

    assert detailsRaw["reference_method"] == "zero"
    assert detailsCentered["reference_method"] == "threshold"
    assert gammaCentered < gammaRaw


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsSplitsSubpeaks():
    n = 100
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.zeros(n, dtype=np.float64)
    state[20:35] += np.hanning(15) * 6.0
    state[50:65] += np.hanning(15) * 5.0
    state[15:70] += 1.5
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[15:70] = 1

    rows, rowMeta = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="splitTest",
        nullScale=0.5,
        subpeakSelectionPenalty=2.0,
        subpeakBoundaryCost=0.25,
    )

    assert len(rows) == 2
    assert len(rowMeta) == 2
    assert all(meta["split_from_parent"] for meta in rowMeta)
    assert all(meta["num_subpeaks"] == 2 for meta in rowMeta)
    assert rowMeta[0]["summit"] < rowMeta[1]["summit"]


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsSplitsSelectedCoordinateGaps():
    intervals = np.asarray([0, 100, 1000, 1100], dtype=np.int64)
    ends = np.asarray([100, 200, 1100, 1200], dtype=np.int64)
    state = np.asarray([3.0, 3.0, 4.0, 4.0], dtype=np.float64)
    scores = state.copy()
    solution = np.ones(state.size, dtype=np.uint8)

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="gapSplitTest",
        nullScale=0.25,
        splitSubpeaks=False,
        trimScoreFloor=0.0,
        returnExportDetails=True,
    )

    assert len(rows) == 2
    assert len(rowMeta) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [0, 200]
    assert [int(rows[1][1]), int(rows[1][2])] == [1000, 1200]
    assert all(int(row[2]) - int(row[1]) == 200 for row in rows)
    assert exportDetails["num_coordinate_gap_splits"] == 1


@pytest.mark.correctness
def _caseMultiscaleCandidateGenerationUsesMultipleScales():
    scores = np.zeros(64, dtype=np.float64)
    scores[8:12] = 3.0
    scores[24:42] = 1.15
    scores[29:34] = 3.5
    intervals = np.arange(0, scores.size * 25, 25, dtype=np.int64)
    ends = intervals + 25

    generate = getattr(peaks, "_generateROCCOMultiscaleCandidateSegments", None)
    assert generate is not None
    candidates, details = generate(
        scores,
        intervals=intervals,
        ends=ends,
        threshold=1.0,
        scales=(1, 3, 9),
        minRunBins=2,
        returnDetails=True,
    )

    assert details["method"] == "multiscale_rocco_candidates"
    assert details["scales"] == [1, 3, 9]
    assert details["num_candidates"] == len(candidates)
    assert details["num_candidates"] >= 2
    assert set(details["candidate_scales"]).issuperset({1, 3, 9})
    candidateScales = {int(candidate["scale_bins"]) for candidate in candidates}
    assert 1 in candidateScales
    assert any(scale > 1 for scale in candidateScales)
    for candidate in candidates:
        startIdx = int(candidate["start_idx"])
        endIdx = int(candidate["end_idx"])
        assert 0 <= startIdx <= endIdx < scores.size
        assert int(candidate["start"]) == int(intervals[startIdx])
        assert int(candidate["end"]) == int(ends[endIdx])
        assert np.isfinite(float(candidate["score_statistic"]))
        assert "boundary_gamma" not in candidate
        assert "per_boundary_gamma" not in candidate


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsSplitsObviousSubpeaksWhenContextIsWide():
    n = 60
    intervals = np.arange(0, n * 50, 50, dtype=np.int64)
    ends = intervals + 50
    state = np.full(n, 0.1, dtype=np.float64)
    state[10:15] = 6.0
    state[42:47] = 5.0
    scores = state.copy()
    solution = np.ones(n, dtype=np.uint8)

    rows, rowMeta = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="wideContextSplitTest",
        nullScale=0.25,
        trimScoreFloor=1.0,
        subpeakSelectionPenalty=1.0,
        subpeakBoundaryCost=0.25,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[10]), int(ends[14])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[42]), int(ends[46])]
    assert all(meta["split_from_parent"] for meta in rowMeta)


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsDoesNotLetDominantPeakHideSubpeak():
    n = 100
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.full(n, 0.1, dtype=np.float64)
    state[20:30] = 20.0
    state[65:75] = 3.0
    scores = state.copy()
    solution = np.ones(n, dtype=np.uint8)

    rows, rowMeta = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="dominantSplitTest",
        nullScale=0.25,
        trimScoreFloor=1.0,
        subpeakSelectionPenalty=1.0,
        subpeakBoundaryCost=0.25,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[20]), int(ends[29])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[65]), int(ends[74])]
    assert rowMeta[1]["max_state"] == pytest.approx(3.0)


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsTrimsChildFlanksAroundSummit():
    n = 80
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.full(n, 0.2, dtype=np.float64)
    state[20:30] = 3.0
    state[50:60] = 2.8
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[10:70] = 1

    rows, rowMeta = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="trimTest",
        nullScale=0.25,
        trimScoreFloor=1.0,
        subpeakSelectionPenalty=1.0,
        subpeakBoundaryCost=0.25,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[20]), int(ends[29])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[50]), int(ends[59])]
    assert all(meta["split_from_parent"] for meta in rowMeta)
    assert not any(meta["trimmed_from_parent"] for meta in rowMeta)
    assert rowMeta[0]["untrimmed_start"] == int(intervals[20])
    assert rowMeta[1]["untrimmed_end"] == int(ends[59])
    assert all(meta["trim_score_floor"] == pytest.approx(1.0) for meta in rowMeta)


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsRefinesAllNegativeChildToBestBin():
    n = 40
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = -0.5 * np.ones(n, dtype=np.float64)
    state[20] = -0.1
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[10:30] = 1

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="negativeTrimTest",
        nullScale=0.25,
        trimScoreFloor=0.0,
        returnExportDetails=True,
    )

    assert rows == []
    assert rowMeta == []
    assert exportDetails["min_peak_bp"] == 200
    assert exportDetails["num_candidate_segments"] == 1
    assert exportDetails["num_segments_dropped_min_peak_bp"] == 1


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsDropsMedianBelowNegativeScaledLocalMedianP():
    n = 50
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.zeros(n, dtype=np.float64)
    state[5:15] = -2.0
    state[25:35] = 2.0
    state[40:45] = -3.5
    scores = state.copy()
    uncertainty = np.ones(n, dtype=np.float64)
    solution = np.zeros(n, dtype=np.uint8)
    solution[5:15] = 1
    solution[25:35] = 1
    solution[40:45] = 1

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="stdFilterTest",
        nullScale=0.25,
        uncertainty=uncertainty,
        trimScoreFloor=0.0,
        subpeakSelectionPenalty=-10.0,
        subpeakBoundaryCost=0.0,
        returnExportDetails=True,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[5]), int(ends[14])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[25]), int(ends[34])]
    assert rowMeta[0]["median_state"] == pytest.approx(-2.0)
    assert rowMeta[0]["local_median_p"] == pytest.approx(1.0)
    assert rowMeta[0]["median_signal_threshold"] == pytest.approx(
        -constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert rowMeta[1]["median_state"] == pytest.approx(2.0)
    assert rowMeta[1]["local_median_p"] == pytest.approx(1.0)
    assert rowMeta[1]["median_signal_threshold"] == pytest.approx(
        -constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert exportDetails["median_signal_local_p_multiplier"] == pytest.approx(
        constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert exportDetails["median_signal_local_p_filter_active"] is True
    assert exportDetails["num_candidate_segments"] == 3
    assert exportDetails["num_segments_dropped_median_signal_local_p"] == 1
    assert exportDetails["num_segments_kept"] == 2


@pytest.mark.correctness
def _caseMassiveSubpeakWidthPolicyRequiresSeparatedTailCluster():
    widths = np.concatenate(
        [
            np.linspace(500.0, 5000.0, 1000),
            np.asarray([12000.0, 14000.0, 16000.0]),
            np.asarray([60000.0, 70000.0]),
        ]
    )

    policy = peaks._learnMassiveSubpeakWidthPolicy(widths)

    assert policy["active"] is True
    assert policy["gap_width_threshold_bp"] == 60000
    assert policy["width_threshold_bp"] == policy["width_cap_bp"]
    assert 10000 <= policy["width_threshold_bp"] < 20000
    assert policy["num_width_tail_gap_candidates"] == 2
    assert policy["num_width_cluster_candidates"] == 3


@pytest.mark.correctness
def _caseMassiveSubpeakWidthPolicyCapsExtremeTailGap():
    widths = np.concatenate(
        [
            np.linspace(500.0, 5000.0, 4500),
            np.asarray([12000.0, 15000.0, 18000.0, 22000.0]),
            np.asarray([33000.0, 46600.0, 70000.0]),
        ]
    )

    policy = peaks._learnMassiveSubpeakWidthPolicy(widths)

    assert policy["active"] is True
    assert policy["gap_width_threshold_bp"] >= 30000
    assert policy["width_threshold_bp"] == policy["width_cap_bp"]
    assert 10000 <= policy["width_threshold_bp"] <= 20000


@pytest.mark.correctness
def _caseMassiveSubpeakWidthPolicyDoesNotFlagSmoothTailWithoutGap():
    widths = np.linspace(500.0, 16000.0, 300)

    policy = peaks._learnMassiveSubpeakWidthPolicy(widths)

    assert policy["active"] is False
    assert policy["width_threshold_bp"] is None


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsForcesMassiveSplittableDomain():
    n = 700
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.full(n, 1.4, dtype=np.float64)
    state[150:230] = 3.0
    state[300:380] = 0.8
    state[470:550] = 2.8
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[100:600] = 1
    policy = {
        "active": True,
        "width_threshold_bp": 10000,
        "null_center": float(np.log(1000.0)),
        "null_scale": 1.0,
    }

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="massiveSplitTest",
        nullScale=0.25,
        trimScoreFloor=0.0,
        subpeakSelectionPenalty=0.0,
        subpeakBoundaryCost=0.25,
        massiveSubpeakCleanup=True,
        massiveSubpeakWidthPolicy=policy,
        returnExportDetails=True,
    )

    assert len(rows) == 2
    assert exportDetails["num_massive_subpeak_splits"] == 1
    assert exportDetails["num_massive_subpeak_contracts"] == 0
    assert all(meta["massive_subpeak_cleanup_applied"] for meta in rowMeta)
    assert all(
        int(row[2]) - int(row[1]) < policy["width_threshold_bp"] for row in rows
    )
    assert rowMeta[0]["end"] <= int(intervals[300])
    assert rowMeta[1]["start"] >= int(ends[379])


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsContractsMassiveSmoothDomain():
    n = 700
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = np.ones(n, dtype=np.float64)
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[100:600] = 1
    policy = {
        "active": True,
        "width_threshold_bp": 10000,
        "null_center": float(np.log(1000.0)),
        "null_scale": 1.0,
    }

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="massiveSmoothTest",
        nullScale=0.25,
        trimScoreFloor=0.0,
        subpeakSelectionPenalty=0.0,
        subpeakBoundaryCost=0.25,
        massiveSubpeakCleanup=True,
        massiveSubpeakWidthPolicy=policy,
        returnExportDetails=True,
    )

    assert len(rows) == 1
    assert len(rowMeta) == 1
    assert exportDetails["num_massive_subpeak_candidates"] == 1
    assert exportDetails["num_massive_subpeak_splits"] == 0
    assert exportDetails["num_massive_subpeak_contracts"] == 1
    assert exportDetails["num_segments_kept"] == 1
    assert rowMeta[0]["massive_subpeak_cleanup_applied"] is True
    assert rowMeta[0]["massive_subpeak_cleanup_mode"] == "width_capped_core"
    assert int(rows[0][2]) - int(rows[0][1]) < policy["width_threshold_bp"]


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsContractsToMinBpCap():
    n = 2200
    intervals = np.arange(0, n * 100, 100, dtype=np.int64)
    ends = intervals + 100
    state = np.ones(n, dtype=np.float64)
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[100:2100] = 1
    policy = {
        "active": True,
        "width_threshold_bp": 60000,
        "min_bp": 10000,
        "contract_width_bp": 10000,
        "null_center": float(np.log(1000.0)),
        "null_scale": 1.0,
    }

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="massiveCapTest",
        nullScale=0.25,
        trimScoreFloor=0.0,
        subpeakSelectionPenalty=0.0,
        subpeakBoundaryCost=0.25,
        massiveSubpeakCleanup=True,
        massiveSubpeakWidthPolicy=policy,
        returnExportDetails=True,
    )

    assert len(rows) == 1
    assert exportDetails["num_massive_subpeak_contracts"] == 1
    assert rowMeta[0]["massive_subpeak_cleanup_mode"] == "width_capped_core"
    assert int(rows[0][2]) - int(rows[0][1]) < policy["contract_width_bp"]


@pytest.mark.correctness
def _caseSolutionToChromNarrowPeakRowsContractsOversizedSplitChildren():
    n = 750
    intervals = np.arange(0, n * 100, 100, dtype=np.int64)
    ends = intervals + 100
    state = np.ones(n, dtype=np.float64)
    state[100:430] = 3.0
    state[430:500] = 0.2
    state[500:590] = 2.8
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[50:650] = 1
    policy = {
        "active": True,
        "width_threshold_bp": 46600,
        "min_bp": 10000,
        "contract_width_bp": 10000,
        "null_center": float(np.log(1000.0)),
        "null_scale": 1.0,
    }

    rows, rowMeta, exportDetails = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="massiveChildCapTest",
        nullScale=0.25,
        trimScoreFloor=0.0,
        subpeakSelectionPenalty=0.0,
        subpeakBoundaryCost=0.25,
        massiveSubpeakCleanup=True,
        massiveSubpeakWidthPolicy=policy,
        returnExportDetails=True,
    )

    assert len(rows) == 2
    assert exportDetails["num_massive_subpeak_splits"] == 1
    assert exportDetails["num_massive_subpeak_contracts"] == 2
    assert all(int(row[2]) - int(row[1]) < policy["contract_width_bp"] for row in rows)
    assert any(
        meta["massive_subpeak_cleanup_mode"] == "width_capped_core"
        for meta in rowMeta
    )


@pytest.mark.correctness
def _caseNestedROCCORefinementShrinksWithinParentRegions():
    scores = np.full(80, 0.2, dtype=np.float64)
    scores[20:30] = 2.0
    scores[45:55] = 2.0
    firstPass = np.zeros(80, dtype=np.uint8)
    firstPass[10:70] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.5,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=1.0,
    )

    assert np.all(refined <= firstPass)
    assert int(np.sum(refined)) < int(np.sum(firstPass))
    assert np.all(refined[20:30] == 1)
    assert np.all(refined[45:55] == 1)
    assert np.all(refined[10:20] == 0)
    assert np.all(refined[30:45] == 0)
    assert np.all(refined[55:70] == 0)
    assert details["local_gamma"] == pytest.approx(0.125)
    assert details["budget_scale"] == pytest.approx(1.0)
    assert details["completed_iters"] == 2
    assert details["stop_reason"] == "mask_equal"
    assert details["history"][0]["num_parent_peaks"] == 1
    assert details["history"][0]["num_parent_peaks_after"] == 2
    assert details["history"][0]["num_budget_constrained_regions"] == 0
    assert np.isfinite(details["history"][0]["objective"])
    assert details["history"][0]["jaccard"] == pytest.approx(20.0 / 60.0)


@pytest.mark.correctness
def _caseNestedROCCORefinementStopsOnJaccardThreshold():
    scores = np.full(1000, 2.0, dtype=np.float64)
    scores[500] = 0.0
    firstPass = np.ones(1000, dtype=np.uint8)

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.0,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=1.0,
        jaccardThreshold=0.999,
    )

    assert np.all(refined <= firstPass)
    assert int(np.sum(refined)) == 999
    assert refined[500] == 0
    assert details["completed_iters"] == 1
    assert details["stop_reason"] == "jaccard"
    assert details["jaccard_threshold"] == pytest.approx(0.999)
    assert details["history"][0]["jaccard"] >= 0.999
    assert details["history"][0]["num_parent_peaks"] == 1
    assert details["history"][0]["num_parent_peaks_after"] == 2
    assert np.isfinite(details["history"][0]["objective"])


@pytest.mark.correctness
def _caseNestedROCCORefinementCanApplyBudgetScale():
    scores = np.full(100, 1.2, dtype=np.float64)
    scores[35:65] = 4.0
    firstPass = np.ones(100, dtype=np.uint8)

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.0,
        selectionPenalty=1.0,
        nestedRoccoIters=1,
        nestedRoccoBudgetScale=0.5,
    )

    assert int(np.sum(refined)) <= 50
    assert int(np.sum(refined)) < int(np.sum(firstPass))
    assert details["budget_scale"] == pytest.approx(0.5)
    assert details["budget_policy"] == "soft_selection_penalty"
    assert details["history"][0]["budget_scale"] == pytest.approx(0.5)
    assert details["history"][0]["num_budget_constrained_regions"] == 1
    assert details["history"][0]["num_soft_budget_penalty_regions"] == 1
    assert details["history"][0]["budget_policy"] == "soft_selection_penalty"


@pytest.mark.correctness
def _caseNestedROCCORefinementDoesNotEraseFlatPositivePlateau():
    scores = np.zeros(80, dtype=np.float64)
    scores[37:44] = 10.0
    firstPass = np.zeros(80, dtype=np.uint8)
    firstPass[37:44] = 1
    intervals = np.arange(0, 80 * 50, 50, dtype=np.int64)
    ends = intervals + 50

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.5,
        selectionPenalty=0.01,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=0.5,
        intervals=intervals,
        ends=ends,
        minRegionBP=250,
    )

    selected = np.flatnonzero(refined)
    assert selected.size == 7
    assert selected[0] >= 37
    assert selected[-1] <= 43
    assert details["history"][0]["num_empty_local_solutions"] == 0
    assert details["history"][0]["num_budget_fallback_windows"] == 0
    assert details["history"][0]["local_penalty_extra_mean"] >= 0.0


@pytest.mark.correctness
def _caseNestedROCCORefinementAppliesBudgetOnlyOnFirstNestedPass():
    scores = np.full(80, 0.2, dtype=np.float64)
    scores[20:30] = 2.0
    scores[45:55] = 2.0
    firstPass = np.zeros(80, dtype=np.uint8)
    firstPass[10:70] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.5,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=0.5,
    )

    assert int(np.sum(refined)) == 20
    assert np.all(refined[20:30] == 1)
    assert np.all(refined[45:55] == 1)
    assert details["completed_iters"] == 2
    assert details["history"][0]["budget_scale"] == pytest.approx(0.5)
    assert details["history"][0]["num_budget_constrained_regions"] == 1
    assert details["history"][1]["budget_scale"] == pytest.approx(1.0)
    assert details["history"][1]["num_budget_constrained_regions"] == 0


@pytest.mark.correctness
def _caseNestedROCCORefinementSatisfiesRequiredBinMonotonicityContract():
    scores = np.full(120, -0.4, dtype=np.float64)
    scores[15:35] = -0.2
    scores[72:82] = 3.0
    scores[95:105] = 2.7
    firstPass = np.zeros(120, dtype=np.uint8)
    firstPass[10:40] = 1
    firstPass[65:110] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.2,
        selectionPenalty=1.0,
        nestedRoccoIters=1,
        nestedRoccoBudgetScale=0.5,
        minRegionBins=5,
    )

    parentRuns = peaks._selectedRunBounds(firstPass)
    childRuns = peaks._selectedRunBounds(refined)
    assert np.all(refined <= firstPass)
    assert int(np.sum(refined)) <= int(np.sum(firstPass))
    assert len(childRuns) >= len(parentRuns)
    assert len(childRuns) == 3
    assert any(start <= 15 <= end for start, end in childRuns)
    assert any(start <= 72 <= end for start, end in childRuns)
    assert details["history"][0]["num_parent_peaks"] == 2
    assert details["history"][0]["num_parent_peaks_after"] == 3
    assert details["history"][0]["num_parent_erasure_violations"] == 0
    assert details["history"][0]["num_required_bin_violations"] == 0
    assert details["history"][0]["num_peak_count_monotonicity_violations"] == 0
    assert details["history"][0]["num_coverage_expansion_violations"] == 0


@pytest.mark.correctness
def _caseNestedROCCORefinementKeepsRequiredBinMinRunWhenPeakIsNarrow():
    scores = np.zeros(40, dtype=np.float64)
    scores[20] = 8.0
    firstPass = np.zeros(40, dtype=np.uint8)
    firstPass[10:30] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.0,
        selectionPenalty=4.0,
        nestedRoccoIters=1,
        nestedRoccoBudgetScale=1.0,
        minRegionBins=5,
    )

    selected = np.flatnonzero(refined)
    assert selected.size == 5
    assert 20 in selected
    assert selected[0] >= 10
    assert selected[-1] <= 29
    assert details["history"][0]["num_empty_local_solutions"] == 0
    assert details["history"][0]["num_parent_peaks"] == 1
    assert details["history"][0]["num_parent_peaks_after"] == 1
    assert details["history"][0]["num_parent_erasure_violations"] == 0
    assert details["history"][0]["num_required_bin_violations"] == 0
    assert details["history"][0]["num_short_child_runs_expanded"] == 0
    assert details["history"][0]["num_short_child_bins_added"] == 0


@pytest.mark.correctness
def _caseNestedROCCORefinementWritesSubproblemDiagnostics(caplog, tmp_path):
    scores = np.full(60, 0.2, dtype=np.float64)
    scores[20:40] = 3.0
    firstPass = np.zeros(60, dtype=np.uint8)
    firstPass[10:50] = 1
    detailPath = tmp_path / "nested_rocco_subproblems.jsonl"

    with caplog.at_level(logging.INFO, logger="consenrich.peaks"):
        refined, details = peaks._refineNestedROCCOSolution(
            scores,
            firstPass,
            gamma=0.5,
            selectionPenalty=1.0,
            nestedRoccoIters=1,
            nestedRoccoBudgetScale=0.5,
            diagnostics=True,
            diagnosticLabel="chrTest",
            diagnosticDetailPath=detailPath,
        )

    assert np.all(refined <= firstPass)
    assert details["completed_iters"] == 1
    assert "nested ROCCO subproblem chrTest" not in caplog.text
    detailRows = [
        json.loads(line)
        for line in detailPath.read_text(encoding="utf-8").splitlines()
    ]
    assert len(detailRows) == 1
    assert detailRows[0]["chromosome"] == "chrTest"
    assert detailRows[0]["event"] == "subproblem"
    assert detailRows[0]["status"] == "solved"
    assert detailRows[0]["mode"] == "parent_conditioned_min_run_soft_budget"
    assert detailRows[0]["budget_policy"] == "soft_selection_penalty"
    assert detailRows[0]["soft_budget_target"] == detailRows[0]["budget_target"]
    assert detailRows[0]["nonpos_selected"] >= 0
    assert detailRows[0]["min_child_bins"] == 5
    assert detailRows[0]["required_selected"] is True


def _caseSolveRoccoVerboseWritesNullReplayFalseSegmentDiagnostics(tmp_path, caplog):
    n = 96
    state = np.zeros(n, dtype=np.float64)
    state[30:38] = 5.5
    uncertainty = np.ones(n, dtype=np.float64)
    statePath, uncPath = _writeSingleChromBedGraphs(
        tmp_path,
        state,
        uncertainty,
        stem="null_replay",
        step=50,
    )
    outPath = tmp_path / "null_replay_rocco.narrowPeak"
    metaPath = tmp_path / "null_replay_rocco.narrowPeak.json"
    detailPath = Path(f"{outPath}.nested_rocco_subproblems.jsonl")

    with caplog.at_level(logging.INFO, logger="consenrich.peaks"):
        peaks.solveRocco(
            str(statePath),
            uncertaintyBedGraphFile=str(uncPath),
            numBootstrap=16,
            dependenceSpan=6,
            randSeed=23,
            gamma=0.0,
            nestedRoccoIters=0,
            massiveSubpeakCleanup=False,
            outPath=str(outPath),
            metaPath=str(metaPath),
            verbose=True,
        )

    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    chromMeta = meta["chromosomes"]["chr1"]
    diagnostics = chromMeta["null_replay_false_segment_diagnostics"]
    assert diagnostics["method"] == "stationary_null_dwb_null_replay"
    assert diagnostics["num_replays"] >= 8
    assert diagnostics["observed_segment_count"] == chromMeta["num_segments"]
    assert diagnostics["false_segment_count_mean"] >= 0.0
    assert diagnostics["false_segment_count_q95"] >= diagnostics["false_segment_count_mean"]
    assert 0.0 <= diagnostics["false_segment_fdr_estimate"] <= 1.0
    _assertNoBoundaryGammaMetadata(diagnostics)

    detailRows = [
        json.loads(line)
        for line in detailPath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replayRows = [
        row for row in detailRows if row.get("event") == "null_replay_false_segments"
    ]
    assert replayRows
    assert replayRows[0]["chromosome"] == "chr1"
    assert replayRows[0]["num_replays"] == diagnostics["num_replays"]
    assert "null replay false-segment diagnostics chr1" in caplog.text


@pytest.mark.correctness
def _caseNestedROCCORefinementSkipsShortParentRegions():
    scores = np.zeros(20, dtype=np.float64)
    firstPass = np.zeros(20, dtype=np.uint8)
    firstPass[8:12] = 1
    intervals = np.arange(0, 20 * 25, 25, dtype=np.int64)
    ends = intervals + 25

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.0,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        intervals=intervals,
        ends=ends,
        minRegionBP=5 * 25,
    )

    assert np.array_equal(refined, firstPass)
    assert details["completed_iters"] == 1
    assert details["stop_reason"] == "mask_equal"
    assert details["history"][0]["num_skipped_short_regions"] == 1


def _caseSolveRoccoVerboseWritesSubproblemDiagnostics(tmp_path, caplog):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "toy_rocco_verbose.narrowPeak"
    metaPath = tmp_path / "toy_rocco_verbose.narrowPeak.json"
    detailPath = Path(f"{outPath}.nested_rocco_subproblems.jsonl")

    with caplog.at_level(logging.INFO, logger="consenrich.peaks"):
        peaks.solveRocco(
            str(statePath),
            uncertaintyBedGraphFile=str(uncPath),
            outPath=str(outPath),
            metaPath=str(metaPath),
            verbose=True,
        )

    assert "writing nested ROCCO subproblem solving details to" in caplog.text
    assert "nested ROCCO subproblem chr" not in caplog.text
    assert detailPath.exists()
    detailRows = [
        json.loads(line)
        for line in detailPath.read_text(encoding="utf-8").splitlines()
    ]
    assert detailRows
    assert {row["event"] for row in detailRows} == {"subproblem"}
    assert any(row["status"] == "solved" for row in detailRows)
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["settings"]["nested_rocco_subproblem_details"] == str(detailPath)


@pytest.mark.correctness
def _caseNestedROCCOAllNegativeParentStillEmitsRequiredBinChild():
    scores = np.full(40, -1.0, dtype=np.float64)
    firstPass = np.zeros(40, dtype=np.uint8)
    firstPass[10:30] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.0,
        selectionPenalty=0.0,
        nestedRoccoIters=1,
        nestedRoccoBudgetScale=1.0,
        minRegionBins=5,
    )

    selected = np.flatnonzero(refined)
    assert selected.size == 5
    assert selected[0] == 10
    assert selected[-1] == 14
    assert details["history"][0]["num_empty_local_solutions"] == 0
    assert details["history"][0]["num_parent_peaks"] == 1
    assert details["history"][0]["num_parent_peaks_after"] == 1
    assert details["history"][0]["num_parent_erasure_violations"] == 0
    assert details["history"][0]["num_required_bin_violations"] == 0


@pytest.mark.correctness
def _caseNestedROCCOWithoutLocalBudgetDoesNotEraseCoherentParentRegion():
    scores = np.ones(100, dtype=np.float64)
    firstPass = np.ones(100, dtype=np.uint8)

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.5,
        selectionPenalty=-0.5,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=1.0,
    )

    assert np.array_equal(refined, firstPass)
    assert details["budget_scale"] == pytest.approx(1.0)
    assert details["stop_reason"] == "mask_equal"


def _assertNestedPolicyMetadataIsUseful(details):
    assert details["subproblem_mode"]
    assert details["required_bin_policy"]
    assert details["min_child_bins"] >= 1
    assert details["initial_selected_count"] >= details["final_selected_count"]
    assert details["history"]
    for step in details["history"]:
        assert step["selected_count_after"] <= step["selected_count_before"]
        assert step["num_coverage_expansion_violations"] == 0
        assert step["num_parent_erasure_violations"] == 0
        assert step["num_required_bin_violations"] == 0
        assert np.isfinite(step["objective"])
        assert np.isfinite(step["objective_delta"])
        assert 0.0 <= step["jaccard"] <= 1.0


@pytest.mark.correctness
def _caseNestedROCCOAdaptivePolicyRetainsBroadCoherentPlateau():
    scores = np.zeros(140, dtype=np.float64)
    scores[25:105] = 4.0
    firstPass = np.zeros(140, dtype=np.uint8)
    firstPass[25:105] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.4,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=0.5,
        minRegionBins=5,
    )

    runs = peaks._selectedRunBounds(refined)
    assert np.all(refined <= firstPass)
    assert len(runs) == 1
    assert int(np.sum(refined[25:105])) >= 0.9 * int(np.sum(firstPass[25:105]))
    assert details["history"][0]["num_parent_peaks_after"] == 1
    assert details["history"][0]["selected_count_delta"] >= -8
    assert details["history"][0]["local_penalty_extra_mean"] == pytest.approx(0.0)
    _assertNestedPolicyMetadataIsUseful(details)


@pytest.mark.correctness
def _caseNestedROCCOAdaptivePolicySplitsOnlyAcrossRealValleys():
    scores = np.zeros(140, dtype=np.float64)
    scores[25:50] = 4.0
    scores[50:60] = -0.2
    scores[60:95] = 3.8
    firstPass = np.zeros(140, dtype=np.uint8)
    firstPass[25:95] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.4,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=1.0,
        minRegionBins=5,
    )

    runs = peaks._selectedRunBounds(refined)
    assert np.all(refined <= firstPass)
    assert len(runs) == 2
    assert np.all(refined[25:50] == 1)
    assert np.all(refined[60:95] == 1)
    assert int(np.sum(refined[50:60])) == 0
    assert details["history"][0]["num_parent_peaks_after"] > details["history"][0][
        "num_parent_peaks"
    ]
    assert details["history"][0]["selected_count_delta"] <= -10
    _assertNestedPolicyMetadataIsUseful(details)


@pytest.mark.correctness
def _caseNestedROCCOAdaptivePolicySuppressesDiffuseShoulders():
    scores = np.zeros(150, dtype=np.float64)
    scores[30:50] = 3.8
    scores[50:105] = 0.45
    firstPass = np.zeros(150, dtype=np.uint8)
    firstPass[30:105] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.4,
        selectionPenalty=1.0,
        nestedRoccoIters=3,
        nestedRoccoBudgetScale=0.5,
        minRegionBins=5,
    )

    runs = peaks._selectedRunBounds(refined)
    assert np.all(refined <= firstPass)
    assert len(runs) == 1
    assert np.all(refined[30:50] == 1)
    assert int(np.sum(refined[50:105])) <= 5
    assert details["history"][0]["selected_count_delta"] <= -50
    assert details["history"][0]["local_penalty_extra_mean"] > 0.0
    _assertNestedPolicyMetadataIsUseful(details)


@pytest.mark.correctness
def _caseNestedROCCOChildSummitContracts():
    scores = np.zeros(160, dtype=np.float64)
    scores[10:70] = 0.15
    scores[20:32] = 4.0
    scores[32:43] = -0.35
    scores[43:55] = 3.7
    noise = np.array(
        [0.0, 0.08, -0.03, 0.05, -0.04, 0.03, -0.02, 0.04, -0.01, 0.02],
        dtype=np.float64,
    )
    scores[90:140] = 2.0 + np.resize(noise, 50)
    firstPass = np.zeros(160, dtype=np.uint8)
    firstPass[10:70] = 1
    firstPass[90:140] = 1

    refined, details = peaks._refineNestedROCCOSolution(
        scores,
        firstPass,
        gamma=0.4,
        selectionPenalty=1.0,
        nestedRoccoIters=1,
        nestedRoccoBudgetScale=1.0,
        minRegionBins=5,
    )

    runs = peaks._selectedRunBounds(refined)
    assert np.all(refined <= firstPass)
    assert [run for run in runs if 10 <= run[0] and run[1] <= 69] == [
        (20, 31),
        (43, 54),
    ]
    assert [run for run in runs if 90 <= run[0] and run[1] <= 139] == [(90, 139)]
    assert all(
        bool(refined[idx])
        for idx in (
            10 + int(np.argmax(scores[10:70])),
            90 + int(np.argmax(scores[90:140])),
        )
    )
    assert details["history"][0]["num_parent_peaks"] == 2
    assert details["history"][0]["num_parent_peaks_after"] == 3
    assert details["history"][0]["num_required_bin_violations"] == 0
    _assertNestedPolicyMetadataIsUseful(details)

    splitScores = np.array(
        [
            3.0,
            3.0,
            3.0,
            -0.1,
            -0.1,
            2.8,
            2.8,
            2.8,
            -0.1,
            -0.1,
            2.6,
            2.6,
            2.6,
        ],
        dtype=np.float64,
    )
    lowMask, _lowObjective, lowDetails = peaks._solveParentConditionedSubpeaks(
        splitScores,
        boundaryCosts=0.05,
        selectionPenalty=0.0,
        minRunBins=2,
        requiredIndex=int(np.argmax(splitScores)),
    )
    highMask, _highObjective, highDetails = peaks._solveParentConditionedSubpeaks(
        splitScores,
        boundaryCosts=0.05,
        selectionPenalty=0.0,
        minRunBins=2,
        requiredIndex=int(np.argmax(splitScores)),
        runPenalty=0.4,
    )

    assert len(peaks._selectedRunBounds(lowMask)) == 3
    assert len(peaks._selectedRunBounds(highMask)) == 1
    assert lowDetails["run_penalty_total"] == pytest.approx(0.0)
    assert highDetails["run_penalty_total"] == pytest.approx(0.4)
    assert lowDetails["required_selected"] is True
    assert highDetails["required_selected"] is True


@pytest.mark.correctness
def _caseCheckMatchingEnabledHonorsEnabledFlag():
    matchingArgs = type(
        "MatchingArgs",
        (),
        {
            "enabled": True,
        },
    )()

    assert consenrich_io.checkMatchingEnabled(matchingArgs) is True


def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_rocco_score_null_gamma_and_budget_contracts(monkeypatch, tmp_path, contract_case):
    for label, func, args in (
        ("empirical mirrored null", _caseEmpiricalMirroredNullStrengthensThreshold, ()),
        (
            "gamma lower context bound",
            _run_with_monkeypatch,
            (monkeypatch, _caseEstimateGammaForROCCOUsesLowerContextBound),
        ),
        ("direct state budget", _caseGetBudgetForROCCOUsesDirectConsenrichState, ()),
        (
            "lower-confidence score uses uncertainty",
            _caseLowerConfidenceROCCOScoreUsesUncertainty,
            (),
        ),
        (
            "lower-confidence score requires uncertainty",
            _caseLowerConfidenceROCCORequiresUncertainty,
            (),
        ),
        ("small positive budget floor", _caseGetBudgetForROCCOAppliesSmallPositiveBudgetFloor, ()),
        ("autosomal null floor helper", _caseAutosomalNullFloorHelperStillRuns, ()),
        ("ROCCO null fallback and EB shrinkage", _caseROCCONullFallbackAndEBShrinkage, ()),
        ("integrated budget tail grid", _caseIntegratedBudgetUsesExcessTailGrid, ()),
        ("stationary DWB shared panel", _casePreparedStationaryNullDWBUsesSharedPanelAndMonotoneThresholds, ()),
        ("peak-level DWB empirical p/q", _caseSolveRoccoAnnotatesPeakLevelDwbEmpiricalPQ, (tmp_path,)),
        ("replay FDR moderate panel performance", _caseReplayFDRModeratePanelsStaySubquadratic, ()),
        ("DWB peak scoring moderate replay performance", _caseDWBPeakScoringModerateReplayStaysBoundedAndSane, ()),
        ("budget fixed-seed stability", _caseGetBudgetForROCCOIsStableUnderFixedSeed, ()),
        ("centered excess gamma", _caseEstimateGammaForROCCOUsesCenteredExcessWhenAvailable, ()),
    ):
        contract_case(label, func, *args)


def test_rocco_bedgraph_solver_contracts(tmp_path, contract_case):
    contract_case("ROCCO bedGraph algorithm", _caseRunROCCOAlgorithmFromBedGraphs, tmp_path)
    contract_case(
        "ROCCO lower-confidence metadata",
        _caseRunROCCOLowerConfidenceRecordsMetadata,
        tmp_path,
    )
    contract_case(
        "short flat enrichment retained",
        _caseRunROCCOAlgorithmKeepsShortFlatEnrichment,
        tmp_path,
    )
    contract_case(
        "blacklist export drop",
        _caseRunROCCODropsBlacklistOverlapsAndRecordsMetadata,
        tmp_path,
    )
    contract_case(
        "min signal export floor",
        _caseSolveRoccoAppliesMinPeakSignalFilter,
        tmp_path,
    )


def test_rocco_subpeak_policy_contracts(contract_case):
    for label, func in (
        ("subpeak splitting", _caseSolutionToChromNarrowPeakRowsSplitsSubpeaks),
        (
            "selected coordinate gaps split",
            _caseSolutionToChromNarrowPeakRowsSplitsSelectedCoordinateGaps,
        ),
        ("multiscale candidate generation", _caseMultiscaleCandidateGenerationUsesMultipleScales),
        (
            "wide-context splitting",
            _caseSolutionToChromNarrowPeakRowsSplitsObviousSubpeaksWhenContextIsWide,
        ),
        (
            "dominant peak exposes subpeak",
            _caseSolutionToChromNarrowPeakRowsDoesNotLetDominantPeakHideSubpeak,
        ),
        ("child flank trimming", _caseSolutionToChromNarrowPeakRowsTrimsChildFlanksAroundSummit),
        ("all-negative child refinement", _caseSolutionToChromNarrowPeakRowsRefinesAllNegativeChildToBestBin),
        (
            "negative local-median drop",
            _caseSolutionToChromNarrowPeakRowsDropsMedianBelowNegativeScaledLocalMedianP,
        ),
    ):
        contract_case(label, func)


def test_rocco_massive_domain_policy_contracts(contract_case):
    for label, func in (
        ("separated tail cluster required", _caseMassiveSubpeakWidthPolicyRequiresSeparatedTailCluster),
        ("extreme tail gap capped", _caseMassiveSubpeakWidthPolicyCapsExtremeTailGap),
        ("smooth tail not flagged", _caseMassiveSubpeakWidthPolicyDoesNotFlagSmoothTailWithoutGap),
        ("massive domain forced split", _caseSolutionToChromNarrowPeakRowsForcesMassiveSplittableDomain),
        (
            "massive smooth domain contracted",
            _caseSolutionToChromNarrowPeakRowsContractsMassiveSmoothDomain,
        ),
        (
            "massive contraction uses min-bp cap",
            _caseSolutionToChromNarrowPeakRowsContractsToMinBpCap,
        ),
        (
            "massive split children contract to cap",
            _caseSolutionToChromNarrowPeakRowsContractsOversizedSplitChildren,
        ),
    ):
        contract_case(label, func)


def test_rocco_nested_refinement_contracts(contract_case):
    for label, func in (
        ("nested shrink within parents", _caseNestedROCCORefinementShrinksWithinParentRegions),
        ("nested stop on Jaccard", _caseNestedROCCORefinementStopsOnJaccardThreshold),
        ("nested budget scale", _caseNestedROCCORefinementCanApplyBudgetScale),
        ("flat positive plateau retained", _caseNestedROCCORefinementDoesNotEraseFlatPositivePlateau),
        ("first-pass budget only", _caseNestedROCCORefinementAppliesBudgetOnlyOnFirstNestedPass),
        ("short parent skipped", _caseNestedROCCORefinementSkipsShortParentRegions),
        ("all-negative parent emits child", _caseNestedROCCOAllNegativeParentStillEmitsRequiredBinChild),
        ("coherent parent retained", _caseNestedROCCOWithoutLocalBudgetDoesNotEraseCoherentParentRegion),
        (
            "adaptive plateau retained",
            _caseNestedROCCOAdaptivePolicyRetainsBroadCoherentPlateau,
        ),
        (
            "adaptive real valley split",
            _caseNestedROCCOAdaptivePolicySplitsOnlyAcrossRealValleys,
        ),
        (
            "adaptive diffuse shoulder suppressed",
            _caseNestedROCCOAdaptivePolicySuppressesDiffuseShoulders,
        ),
        ("nested child summits", _caseNestedROCCOChildSummitContracts),
    ):
        contract_case(label, func)


def test_rocco_required_bin_min_run_contracts(contract_case):
    contract_case(
        "required_bin monotonicity",
        _caseNestedROCCORefinementSatisfiesRequiredBinMonotonicityContract,
    )
    contract_case(
        "required_bin min-run for narrow peak",
        _caseNestedROCCORefinementKeepsRequiredBinMinRunWhenPeakIsNarrow,
    )


def test_rocco_diagnostics_contracts(tmp_path, caplog, contract_case):
    caplog.clear()
    contract_case(
        "nested subproblem diagnostics",
        _caseNestedROCCORefinementWritesSubproblemDiagnostics,
        caplog,
        tmp_path,
    )
    caplog.clear()
    contract_case(
        "verbose solve diagnostics",
        _caseSolveRoccoVerboseWritesSubproblemDiagnostics,
        tmp_path,
        caplog,
    )
    caplog.clear()
    contract_case(
        "null replay false-segment diagnostics",
        _caseSolveRoccoVerboseWritesNullReplayFalseSegmentDiagnostics,
        tmp_path,
        caplog,
    )
    caplog.clear()
    contract_case(
        "ROCCO summary inventory",
        _caseSolveRoccoReturnsSummaryInventoryAndLogs,
        tmp_path,
        caplog,
    )
    caplog.clear()
    contract_case(
        "ROCCO metadata cap fallback",
        _caseSolveRoccoMetadataCapFallsBackToBounded,
        tmp_path,
        caplog,
    )


def test_rocco_cutoff_report_contract(tmp_path, monkeypatch, contract_case):
    contract_case(
        "ROCCO cutoff report",
        _caseSolveRoccoCutoffReportWritesSweeps,
        tmp_path,
        monkeypatch,
    )


def test_rocco_matching_enabled_contract(contract_case):
    contract_case("matching enabled flag", _caseCheckMatchingEnabledHonorsEnabledFlag)
