# -*- coding: utf-8 -*-

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import consenrich.consenrich as consenrich_cli
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


@pytest.mark.correctness
def testBuildStudentizedScoreTrackUsesTau0():
    state = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    uncertainty = np.array([0.0, 1.0, 3.0], dtype=np.float64)

    scoreTrack, details = peaks.studentizedScoreTrack(
        state,
        uncertainty=uncertainty,
        tau0=1.0,
        returnDetails=True,
    )

    expected = state / np.sqrt(uncertainty * uncertainty + 1.0)
    assert np.allclose(scoreTrack, expected)
    assert details["tau0"] == 1.0
    assert details["se_mode"] == "uncertainty"


@pytest.mark.correctness
def testBuildShrinkageScoreTrackUsesNullCenterAndUncertainty():
    state = np.array([0.0, 2.0, 6.0], dtype=np.float64)
    uncertainty = np.array([0.0, 1.0, 3.0], dtype=np.float64)

    scoreTrack, details = peaks.shrinkageScoreTrack(
        state,
        uncertainty=uncertainty,
        nullCenter=1.0,
        tau0=2.0,
        returnDetails=True,
    )

    expectedWeights = 4.0 / (4.0 + uncertainty * uncertainty)
    expected = expectedWeights * (state - 1.0)
    assert np.allclose(scoreTrack, expected)
    assert details["score_mode"] == "posterior_mean_shrinkage"
    assert details["prior_variance"] == 4.0
    assert details["null_center_input"] == 1.0


@pytest.mark.correctness
def testEmpiricalMirroredNullStrengthensThreshold():
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
def testEstimateGammaForROCCOUsesLowerContextBound(monkeypatch):
    scoreTrack = np.linspace(-0.5, 3.5, 256, dtype=np.float64)

    def _fakeGetContextSize(vals, minSpan=3, maxSpan=64):
        return 12, 7, 20

    monkeypatch.setattr(peaks.core, "getContextSize", _fakeGetContextSize)
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
def testEstimateGammaForROCCOUsesFixedDefault():
    scoreTrack = np.linspace(-0.5, 3.5, 256, dtype=np.float64)
    gamma, details = peaks.estimateROCCOGamma(scoreTrack, returnDetails=True)

    assert gamma == pytest.approx(0.5)
    assert details["method"] == "fixed"


@pytest.mark.correctness
def testGetBudgetForROCCOUsesDirectConsenrichStateByDefault():
    state, uncertainty = _toyChromState()
    uncertaintyHi = uncertainty.copy()
    uncertaintyHi[90:120] = uncertaintyHi[90:120] * 3.0
    uncertaintyHi[300:340] = uncertaintyHi[300:340] * 2.5

    budgetBase, _ = peaks.getROCCOBudget(
        state,
        uncertainty=uncertainty,
        numBootstrap=48,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )
    budgetNoUnc, _ = peaks.getROCCOBudget(
        state,
        uncertainty=None,
        numBootstrap=48,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )
    budgetHiUnc, details = peaks.getROCCOBudget(
        state,
        uncertainty=uncertaintyHi,
        numBootstrap=48,
        dependenceSpan=8,
        randomSeed=11,
        returnDetails=True,
    )

    assert details["score_mode"] == "consenrich_state"
    assert details["budget_model"] == "dwb_integrated_excess_tail"
    assert details["method"] == "stationary_null_dwb"
    assert details["null_calibration_method"] == "stationary_null_dwb"
    assert details["budget_min"] == pytest.approx(0.001)
    assert details["budget_max"] == 0.10
    assert details["null_quantile"] == pytest.approx(0.80)
    assert details["threshold_z"] == pytest.approx(2.0)
    assert details["threshold_z_grid"] == pytest.approx([1.5, 2.0, 2.5, 3.0])
    assert details["dwb_panel_reused"] is True
    assert details["se_mode"] == "ignored"
    assert details["uncertainty_available"] is True
    assert details["uncertainty_used"] is False
    assert details["tau0_used"] is False
    assert budgetNoUnc == pytest.approx(budgetBase)
    assert budgetHiUnc == pytest.approx(budgetBase)


@pytest.mark.correctness
def testGetBudgetForROCCOAppliesSmallPositiveBudgetFloor():
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
    assert np.isclose(budgetHigh, 0.10)
    assert detailsHigh["budget_clipped"] is True
    assert detailsHigh["budget_raw"] >= 0.10


@pytest.mark.correctness
def testLegacyAutosomalNullFloorHelperStillRuns():
    chr1State, chr1Unc = _toyChromState(seed=7, n=512)
    chr1State[::19] -= 3.0
    chrYState = np.zeros(384, dtype=np.float64)
    chrYUnc = np.full(384, 0.8, dtype=np.float64)

    basePreparedByChrom = {
        "chr1": peaks._prepareROCCOBaseScore(chr1State, uncertainty=chr1Unc, tau0=1.0),
        "chrY": peaks._prepareROCCOBaseScore(chrYState, uncertainty=chrYUnc, tau0=1.0),
    }
    pooledNullFloor = peaks._estimateAutosomalNullFloorForROCCO(
        basePreparedByChrom,
        thresholdZ=2.5,
    )
    localPrepared = peaks._prepareROCCOScoreAndNull(
        chrYState,
        uncertainty=chrYUnc,
        tau0=1.0,
        thresholdZ=2.5,
    )
    pooledPrepared = peaks._prepareROCCOScoreAndNull(
        chrYState,
        uncertainty=chrYUnc,
        tau0=1.0,
        thresholdZ=2.5,
        pooledNullFloor=pooledNullFloor,
    )

    assert pooledNullFloor["source"] == "autosomal_pool"
    assert pooledPrepared["threshold"] >= localPrepared["threshold"]
    assert pooledPrepared["null_scale"] >= localPrepared["null_scale"]
    assert pooledPrepared["pooled_null_floor"]["source"] == "autosomal_pool"


@pytest.mark.correctness
def testROCCONullFallbackAndEBShrinkage():
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
def testRunROCCOAlgorithmFromBedGraphs(tmp_path):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "toy_rocco.narrowPeak"
    metaPath = tmp_path / "toy_rocco.narrowPeak.json"

    resultPath = peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        tau0=1.0,
        numBootstrap=24,
        dependenceSpan=8,
        randSeed=11,
        outPath=str(outPath),
        metaPath=str(metaPath),
    )

    assert Path(resultPath).is_file()
    assert metaPath.is_file()
    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["settings"]["budget_method"] == "dwb_integrated_excess_tail"
    assert meta["settings"]["null_calibration_method"] == "stationary_null_dwb"
    assert meta["pooled_null_floor"] is None
    assert meta["budget_shrinkage"] is None
    for chrom in meta["chromosomes"].values():
        assert "budget_pre_shrink" in chrom["budget_details"]
        assert "budget_post_shrink" in chrom["budget_details"]
        assert chrom["budget_details"]["budget_pre_shrink"] == pytest.approx(
            chrom["budget_details"]["budget_post_shrink"]
        )
        assert chrom["budget_details"]["budget_shrinkage_meta"] is None
        assert "peak_details" in chrom
    lines = outPath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    assert lines[0].startswith("chr")


@pytest.mark.correctness
def testRunROCCOAlgorithmKeepsShortFlatEnrichment(tmp_path):
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
    assert chromMeta["solve_details"]["budget_fallback_window"] is True
    assert chromMeta["solve_details"]["first_pass_selected_count"] > 0
    assert chromMeta["solve_details"]["final_selected_count"] > 0
    assert chromMeta["nested_rocco_details"]["history"][0]["num_budget_fallback_windows"] == 0


@pytest.mark.correctness
def testIntegratedBudgetUsesExcessTailGrid():
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

    budgetOcc, detailsOcc = peaks._estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic="occupancy",
        numBootstrap=24,
        dependenceSpan=16,
        randomSeed=11,
        budgetMax=1.0,
        returnDetails=True,
    )
    budgetIntegrated, detailsIntegrated = peaks._estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic="integrated",
        numBootstrap=24,
        dependenceSpan=16,
        randomSeed=11,
        budgetMax=1.0,
        returnDetails=True,
    )

    assert detailsIntegrated["threshold_z_grid"] == pytest.approx([1.5, 2.0, 2.5, 3.0])
    assert len(detailsIntegrated["threshold_metrics"]) == 4
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
def testPreparedStationaryNullDWBUsesSharedPanelAndMonotoneThresholds():
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
def testGetBudgetForROCCOIsStableUnderFixedSeed():
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
def testEstimateGammaForROCCOUsesCenteredExcessWhenAvailable():
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
def testSolutionToChromNarrowPeakRowsSplitsSubpeaks():
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
        contextSpan=8,
    )

    assert len(rows) == 2
    assert len(rowMeta) == 2
    assert all(meta["split_from_parent"] for meta in rowMeta)
    assert all(meta["num_subpeaks"] == 2 for meta in rowMeta)
    assert rowMeta[0]["summit"] < rowMeta[1]["summit"]


@pytest.mark.correctness
def testSolutionToChromNarrowPeakRowsSplitsObviousSubpeaksWhenContextIsWide():
    n = 60
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
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
        contextSpan=64,
        trimScoreFloor=1.0,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[10]), int(ends[14])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[42]), int(ends[46])]
    assert all(meta["split_from_parent"] for meta in rowMeta)


@pytest.mark.correctness
def testSolutionToChromNarrowPeakRowsDoesNotLetDominantPeakHideSubpeak():
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
        contextSpan=8,
        trimScoreFloor=1.0,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[20]), int(ends[29])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[65]), int(ends[74])]
    assert rowMeta[1]["max_state"] == pytest.approx(3.0)


@pytest.mark.correctness
def testSolutionToChromNarrowPeakRowsTrimsChildFlanksAroundSummit():
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
        contextSpan=8,
        trimScoreFloor=1.0,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[20]), int(ends[29])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[50]), int(ends[59])]
    assert all(meta["trimmed_from_parent"] for meta in rowMeta)
    assert rowMeta[0]["untrimmed_start"] == int(intervals[10])
    assert rowMeta[1]["untrimmed_end"] == int(ends[69])
    assert all(meta["trim_score_floor"] == pytest.approx(1.0) for meta in rowMeta)


@pytest.mark.correctness
def testSolutionToChromNarrowPeakRowsDoesNotCollapseAllNegativeChild():
    n = 40
    intervals = np.arange(0, n * 25, 25, dtype=np.int64)
    ends = intervals + 25
    state = -0.5 * np.ones(n, dtype=np.float64)
    state[20] = -0.1
    scores = state.copy()
    solution = np.zeros(n, dtype=np.uint8)
    solution[10:30] = 1

    rows, rowMeta = peaks._solutionToChromNarrowPeakRows(
        "chr1",
        intervals,
        ends,
        state,
        scores,
        solution,
        prefix="negativeTrimTest",
        nullScale=0.25,
        contextSpan=8,
        trimScoreFloor=0.0,
    )

    assert len(rows) == 1
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[10]), int(ends[29])]
    assert rowMeta[0]["trimmed_from_parent"] is False
    assert rowMeta[0]["untrimmed_start"] == int(intervals[10])
    assert rowMeta[0]["untrimmed_end"] == int(ends[29])


@pytest.mark.correctness
def testSolutionToChromNarrowPeakRowsDropsMedianBelowNegativeScaledLocalMedianP():
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
        contextSpan=8,
        uncertainty=uncertainty,
        trimScoreFloor=0.0,
        returnExportDetails=True,
    )

    assert len(rows) == 2
    assert [int(rows[0][1]), int(rows[0][2])] == [int(intervals[5]), int(ends[14])]
    assert [int(rows[1][1]), int(rows[1][2])] == [int(intervals[25]), int(ends[34])]
    assert rowMeta[0]["median_state"] == pytest.approx(-2.0)
    assert rowMeta[0]["local_median_p"] == pytest.approx(1.0)
    assert rowMeta[0]["median_signal_threshold"] == pytest.approx(-2.5)
    assert rowMeta[1]["median_state"] == pytest.approx(2.0)
    assert rowMeta[1]["local_median_p"] == pytest.approx(1.0)
    assert rowMeta[1]["median_signal_threshold"] == pytest.approx(-2.5)
    assert exportDetails["median_signal_local_p_multiplier"] == pytest.approx(2.5)
    assert exportDetails["median_signal_local_p_filter_active"] is True
    assert exportDetails["num_candidate_segments"] == 3
    assert exportDetails["num_segments_dropped_median_signal_local_p"] == 1
    assert exportDetails["num_segments_kept"] == 2


@pytest.mark.correctness
def testNestedROCCORefinementShrinksWithinParentRegions():
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
def testNestedROCCORefinementStopsOnJaccardThreshold():
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
def testNestedROCCORefinementCanApplyBudgetScale():
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
    assert details["history"][0]["budget_scale"] == pytest.approx(0.5)
    assert details["history"][0]["num_budget_constrained_regions"] == 1


@pytest.mark.correctness
def testNestedROCCORefinementDoesNotEraseFlatPositivePlateau():
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
def testNestedROCCORefinementAppliesBudgetOnlyOnFirstNestedPass():
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
def testNestedROCCORefinementSatisfiesAnchoredMonotonicityContract():
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
    assert details["history"][0]["num_anchor_survival_violations"] == 0
    assert details["history"][0]["num_peak_count_monotonicity_violations"] == 0
    assert details["history"][0]["num_coverage_expansion_violations"] == 0


@pytest.mark.correctness
def testNestedROCCORefinementKeepsAnchoredMinRunWhenPeakIsNarrow():
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
    assert details["history"][0]["num_anchor_survival_violations"] == 0
    assert details["history"][0]["num_short_child_runs_expanded"] == 0
    assert details["history"][0]["num_short_child_bins_added"] == 0


@pytest.mark.correctness
def testNestedROCCORefinementWritesSubproblemDiagnostics(caplog, tmp_path):
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
    assert detailRows[0]["mode"] == "anchored_min_run_soft_budget"
    assert detailRows[0]["nonpos_selected"] >= 0
    assert detailRows[0]["min_child_bins"] == 5
    assert detailRows[0]["anchor_selected"] is True


@pytest.mark.correctness
def testNestedROCCORefinementSkipsShortParentRegions():
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


@pytest.mark.correctness
def testSolveRoccoDefaultsMatchConfig(tmp_path):
    statePath, uncPath = _writeToyBedGraphs(tmp_path)
    outPath = tmp_path / "toy_rocco_default.narrowPeak"
    metaPath = tmp_path / "toy_rocco_default.narrowPeak.json"

    peaks.solveRocco(
        str(statePath),
        uncertaintyBedGraphFile=str(uncPath),
        outPath=str(outPath),
        metaPath=str(metaPath),
    )

    meta = json.loads(metaPath.read_text(encoding="utf-8"))
    assert meta["settings"]["num_bootstrap"] == 128
    assert meta["settings"]["threshold_z"] == pytest.approx(2.0)
    assert meta["settings"]["nested_rocco_iters"] == 3
    assert meta["settings"]["nested_rocco_budget_scale"] == pytest.approx(0.5)
    assert meta["settings"]["nested_rocco_jaccard"] == pytest.approx(0.999)
    assert meta["settings"]["nested_rocco_min_parent_steps"] == 5
    assert meta["settings"]["nested_rocco_min_child_steps"] == 5
    assert meta["settings"]["export_trim_score_floor"] == pytest.approx(0.0)
    assert meta["settings"]["export_filter"] == "drop_median_signal_below_negative_local_median_p"
    assert meta["settings"]["export_filter_threshold"] == "-2.5 * median(local_uncertainty)"
    assert meta["settings"]["export_filter_uncertainty_multiplier"] == pytest.approx(2.5)


@pytest.mark.correctness
def testSolveRoccoVerboseWritesSubproblemDiagnostics(tmp_path, caplog):
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
def testNestedROCCOAllNegativeParentStillEmitsAnchoredChild():
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
    assert details["history"][0]["num_anchor_survival_violations"] == 0


@pytest.mark.correctness
def testNestedROCCOWithoutLocalBudgetDoesNotEraseCoherentParentRegion():
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


@pytest.mark.correctness
def testCheckMatchingEnabledDefaultsToEnabled():
    matchingArgs = type(
        "MatchingArgs",
        (),
        {
            "enabled": True,
        },
    )()

    assert consenrich_cli.checkMatchingEnabled(matchingArgs) is True
