# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import consenrich
import consenrich.io as consenrich_io
import consenrich.peaks as peaks


def _writeSingleChromBedGraphs(
    tmp_path: Path,
    state: np.ndarray,
    uncertainty: np.ndarray | None = None,
    *,
    chrom: str = "chr1",
    step: int = 25,
    stem: str = "single",
):
    starts = np.arange(state.size, dtype=np.int64) * int(step)
    rows = zip(np.repeat(chrom, state.size), starts, starts + step, state)
    statePath = tmp_path / f"{stem}_state.bedGraph"
    pd.DataFrame(rows).to_csv(statePath, sep="\t", header=False, index=False)
    if uncertainty is None:
        return statePath, None
    rows = zip(np.repeat(chrom, state.size), starts, starts + step, uncertainty)
    uncertaintyPath = tmp_path / f"{stem}_uncertainty.bedGraph"
    pd.DataFrame(rows).to_csv(
        uncertaintyPath, sep="\t", header=False, index=False
    )
    return statePath, uncertaintyPath


def _caseBroadMergePolicyContracts():
    starts = np.arange(0, 700, 100, dtype=np.int64)
    runs = [(0, 1), (4, 5)]
    blacklist = {"chr1": np.asarray([(200, 400)], dtype=np.int64)}
    cases = (
        (-0.5, {}, [(0, 5)], ("num_gaps_merged", 1)),
        (-2.0, {}, runs, ("num_gaps_merged", 0)),
        (-1.0, {}, [(0, 5)], None),
        (-0.5, blacklist, runs, ("num_gaps_blocked_by_blacklist", 1)),
    )
    for gapScore, blacklistByChrom, expected, detail in cases:
        scores = np.asarray([5.0, 5.0, gapScore, gapScore, 5.0, 5.0, 0.0])
        merged, details = peaks._mergeBroadRunsByObjective(
            runs,
            scores,
            starts,
            starts + 100,
            "chr1",
            selectionPenalty=0.0,
            boundaryCost=1.0,
            mergeToleranceBP=300,
            maxRegionBP=700,
            blacklistByChrom=blacklistByChrom,
        )
        assert merged == expected
        if detail is not None:
            assert details[detail[0]] == detail[1]


def _caseRunROCCOBothModeWritesNarrowAndGapped(tmp_path, monkeypatch, caplog):
    state = 0.2 * np.sin(np.linspace(0.0, 30.0, 1_000))
    state[20:70], state[35:40] = 6.0, 15.0
    state[105:145], state[118:123] = 5.0, 12.0
    statePath, uncertaintyPath = _writeSingleChromBedGraphs(
        tmp_path,
        state,
        np.full(state.size, 0.25),
        step=100,
        stem="both_mode",
    )
    narrowPath = tmp_path / "combined.narrowPeak"
    metadataPath = tmp_path / "combined.json"
    plotCalls = []

    def capturePlot(rows, path, *, dpi=400):
        plotCalls.append((tuple(rows), str(path), dpi))
        Path(path).write_bytes(b"png")
        return True

    with monkeypatch.context() as patch:
        patch.setattr(peaks, "_plotROCCONullCalibrationDiagnostics", capturePlot)
        artifacts = peaks.solveRocco(
            str(statePath),
            500,
            1000,
            uncertaintyBedGraphFile=str(uncertaintyPath),
            peakMode="both",
            broadWeakThresholdZ=3.0,
            thresholdZ=2.0,
            mergeToleranceBP=500,
            maxRegionBP=18000,
            numBootstrap=8,
            numRegionReplays=2,
            gamma=0.25,
            nestedRoccoIters=1,
            randSeed=11,
            outPath=str(narrowPath),
            metaPath=str(metadataPath),
        )
    diagnosticPath = Path(f"{narrowPath}.nullCalibration.png")
    assert isinstance(artifacts, consenrich.peakArtifacts)
    assert artifacts == consenrich.peakArtifacts(
        str(narrowPath),
        str(narrowPath.with_suffix(".gappedPeak")),
        str(metadataPath),
        str(diagnosticPath),
    )
    assert diagnosticPath.read_bytes() == b"png"
    assert len(plotCalls) == 1
    assert plotCalls[0][1:] == (str(diagnosticPath), 400)
    assert tuple(plotCalls[0][0][0]) == (
        "chromosome",
        "nullTailOccupancyDraws",
        "nullTailOccupancy",
        "thresholdZ",
        "tailAlpha",
        "signedTailExcess",
        "budgetLocal",
        "budget",
    )
    for outputPath, width in ((artifacts.narrowPeak, 10), (artifacts.gappedPeak, 15)):
        rows = [line.split("\t") for line in Path(outputPath).read_text().splitlines()]
        assert rows and all(len(row) == width for row in rows)
    metadata = json.loads(metadataPath.read_text())
    assert set(metadata) == {
        "inputs", "outputs", "settings", "chromosomes", "budgetShrinkage",
        "qValues", "counts",
    }
    assert metadata["outputs"] == {
        "narrowPeak": artifacts.narrowPeak,
        "gappedPeak": artifacts.gappedPeak,
        "nullCalibrationDiagnostics": artifacts.nullCalibrationDiagnostics,
    }
    assert metadata["settings"]["peakMode"] == "both"
    assert metadata["settings"]["thresholdZ"] == 2.0
    assert metadata["settings"]["broadWeakThresholdZ"] == 2.0
    assert metadata["settings"]["useLocalBootStrapRadius"] is True
    assert metadata["settings"]["plotNullCalibrationDiagnostics"] is True
    assert metadata["settings"]["localBootstrapRadiusLimitBP"] == 1_000_000
    radiusFit = metadata["chromosomes"]["chr1"]["fit"]
    assert {
        name: radiusFit[name]
        for name in (
            "useLocalBootStrapRadius",
            "localBootstrapRadiusLimitBP",
            "bootstrapBinBP",
            "maxLocalRadiusIntervals",
            "bootstrapLocalRadiusMinBins",
            "bootstrapLocalRadiusMaxBins",
            "bootstrapLocalRadiusLimitHitSegmentCount",
        )
    } == {
        "useLocalBootStrapRadius": True,
        "localBootstrapRadiusLimitBP": 1_000_000,
        "bootstrapBinBP": 100,
        "maxLocalRadiusIntervals": 10_000,
        "bootstrapLocalRadiusMinBins": 71,
        "bootstrapLocalRadiusMaxBins": 71,
        "bootstrapLocalRadiusLimitHitSegmentCount": 0,
    }
    assert (
        "broadWeakThresholdZ=3 exceeds thresholdZ=2. "
        "Setting broadWeakThresholdZ to 2."
    ) in caplog.messages
    assert {
        mode: (member["scope"], member["method"])
        for mode, member in metadata["qValues"].items()
    } == {
        "narrow": ("chromosome", "stationaryBootstrapCandidateReplay"),
        "broad": ("chromosome", "stationaryBootstrapCandidateReplay"),
    }


def _caseNullCalibrationDiagnosticPanel(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = (
        {
            "chromosome": "chr1",
            "nullTailOccupancyDraws": np.asarray([0.02, 0.03, 0.04, 0.05]),
            "nullTailOccupancy": 0.035,
            "thresholdZ": 2.0,
            "tailAlpha": 0.022750131948179195,
            "signedTailExcess": 0.025,
            "budgetLocal": 0.025,
            "budget": 0.030,
        },
        {
            "chromosome": "chr2",
            "nullTailOccupancyDraws": np.asarray([0.01, 0.02, 0.03, 0.04]),
            "nullTailOccupancy": 0.025,
            "thresholdZ": 2.0,
            "tailAlpha": 0.022750131948179195,
            "signedTailExcess": -0.005,
            "budgetLocal": 0.001,
            "budget": 0.001,
        },
    )
    plotPath = tmp_path / "null_panel.png"
    openFigures = set(plt.get_fignums())
    with monkeypatch.context() as patch:
        patch.setattr(plt, "close", lambda _figure: None)
        assert peaks._plotROCCONullCalibrationDiagnostics(rows, plotPath)
        figureNumbers = set(plt.get_fignums()) - openFigures
        assert len(figureNumbers) == 1
        figure = plt.figure(figureNumbers.pop())

    assert plotPath.stat().st_size > 0
    np.testing.assert_allclose(figure.get_size_inches(), [14.8, 9.2])
    assert figure._suptitle.get_text() == (
        "ROCCO Null Calibration and Chromosome Budgets"
    )
    assert [axis.get_title() for axis in figure.axes] == [
        "Bootstrap Null Tail Occupancy",
        "Chromosome Signed Excess and Budgets",
        "Local and Shrunk Chromosome Budgets",
        "Bootstrap Null and Signed Excess",
    ]
    assert [(axis.get_xlabel(), axis.get_ylabel()) for axis in figure.axes] == [
        ("tail occupancy fraction", "bootstrap draws"),
        ("tail occupancy fraction", "chromosomes"),
        ("local budget fraction", "shrunk budget fraction"),
        ("chromosome", "tail occupancy fraction"),
    ]
    excessLabels = figure.axes[1].get_legend_handles_labels()[1]
    pairedLabels = figure.axes[3].get_legend_handles_labels()[1]
    assert figure.axes[0].get_legend_handles_labels()[1] == [
        "pooled median",
        "normal tail target ($z=2$)",
    ]
    assert excessLabels == ["signed excess", "shrunk budget"]
    assert pairedLabels == ["bootstrap null mean", "signed excess"]
    np.testing.assert_allclose(
        figure.axes[2].collections[0].get_offsets(),
        [[0.025, 0.030], [0.001, 0.001]],
    )
    pairedBars = figure.axes[3].patches
    assert len(pairedBars) == 4
    for nullBar, excessBar in zip(pairedBars[:2], pairedBars[2:]):
        assert nullBar.get_x() + nullBar.get_width() == pytest.approx(
            excessBar.get_x()
        )
    plt.close(figure)


def _caseReplayFDRModeratePanelsStaySubquadratic():
    empiricalP = peaks._empiricalReplaySegmentPValues
    replayQ = peaks._replayFDRQValues
    rng = np.random.default_rng(271)
    observed = rng.gamma(shape=2.5, scale=1.0, size=6000)
    nullDraws = [rng.gamma(shape=2.2, scale=1.0, size=3000) for _ in range(32)]
    pValues = empiricalP(observed, nullDraws)
    qValues = np.maximum(replayQ(observed, nullDraws), pValues)
    assert pValues.shape == qValues.shape == observed.shape
    assert np.all((0.0 <= pValues) & (pValues <= qValues) & (qValues <= 1.0))
    assert np.any(qValues > pValues + 1.0e-6)
    order = np.argsort(-observed, kind="mergesort")
    assert np.all(np.diff(pValues[order]) >= -1.0e-12)
    assert np.all(np.diff(qValues[order]) >= -1.0e-12)


def _caseStationaryBootstrapNativeContracts():
    template = np.arange(20, dtype=np.float64)
    offsets = np.asarray([0, 10, 20], dtype=np.int64)
    weights = np.arange(1.0, 21.0)
    nativeDraw = peaks.cconsenrich.cStationaryNullBootstrapDraw
    restartIndices = np.asarray([0, 1, 2, 9, 10, 12, 14, 15, 19])

    expectedLocalSource = np.asarray(
        [5, 1, 3, 4, 5, 6, 7, 8, 9, 8, 11, 12, 16, 17, 17, 14, 15, 16, 17, 16],
        dtype=np.float64,
    )
    localDraw = nativeDraw(
        template,
        offsets,
        weights,
        2.0,
        np.random.Generator(np.random.PCG64(0)),
        5,
    )
    localShift = expectedLocalSource[0] - localDraw[0]
    np.testing.assert_allclose(localDraw + localShift, expectedLocalSource)
    assert localDraw.dtype == np.float64 and localDraw.flags.c_contiguous
    assert np.average(localDraw, weights=weights) == pytest.approx(0.0)
    assert np.all(np.abs(expectedLocalSource[restartIndices] - restartIndices) <= 5)
    assert np.all(expectedLocalSource[:10] < 10)
    assert np.all((expectedLocalSource[10:] >= 10) & (expectedLocalSource[10:] < 20))
    assert expectedLocalSource[2:9].size > 5
    np.testing.assert_array_equal(np.diff(expectedLocalSource[2:9]), np.ones(6))

    expectedCappedSource = np.asarray(
        [2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 14, 17, 18, 19, 10, 19],
        dtype=np.float64,
    )
    cappedDraw = nativeDraw(
        template,
        offsets,
        weights,
        2.0,
        np.random.Generator(np.random.PCG64(0)),
        2,
    )
    cappedShift = expectedCappedSource[0] - cappedDraw[0]
    np.testing.assert_allclose(cappedDraw + cappedShift, expectedCappedSource)
    assert np.all(np.abs(expectedCappedSource[restartIndices] - restartIndices) <= 2)
    assert np.any(expectedCappedSource != expectedLocalSource)

    implicitRNG = np.random.Generator(np.random.PCG64(0))
    explicitRNG = np.random.Generator(np.random.PCG64(0))
    implicitGlobal = nativeDraw(template, offsets, weights, 2.0, implicitRNG)
    explicitGlobal = nativeDraw(template, offsets, weights, 2.0, explicitRNG, -1)
    np.testing.assert_array_equal(implicitGlobal, explicitGlobal)
    assert (
        implicitRNG.bit_generator.random_raw() == explicitRNG.bit_generator.random_raw()
    )
    expectedGlobalSource = np.asarray(
        [11, 4, 11, 12, 13, 14, 15, 16, 17, 12, 19, 10, 14, 15, 17, 4, 5, 6, 7, 18],
        dtype=np.float64,
    )
    globalShift = expectedGlobalSource[0] - implicitGlobal[0]
    np.testing.assert_allclose(implicitGlobal + globalShift, expectedGlobalSource)

    for edgeOffsets, meanBlockLength, radiusLimit in (
        (np.arange(template.size + 1, dtype=np.int64), 2.0, 100),
        (offsets, 1.0, 0),
    ):
        edgeDraw = nativeDraw(
            template,
            edgeOffsets,
            weights,
            meanBlockLength,
            np.random.Generator(np.random.PCG64(7)),
            radiusLimit,
        )
        edgeShift = template[0] - edgeDraw[0]
        np.testing.assert_allclose(edgeDraw + edgeShift, template)

    saturatedLocalRNG = np.random.Generator(np.random.PCG64(0))
    saturatedGlobalRNG = np.random.Generator(np.random.PCG64(0))
    saturatedLocal = nativeDraw(
        template,
        np.asarray([0, template.size], dtype=np.int64),
        weights,
        np.finfo(np.float64).max,
        saturatedLocalRNG,
        template.size,
    )
    saturatedGlobal = nativeDraw(
        template,
        np.asarray([0, template.size], dtype=np.int64),
        weights,
        np.finfo(np.float64).max,
        saturatedGlobalRNG,
    )
    np.testing.assert_array_equal(saturatedLocal, saturatedGlobal)
    assert np.all(np.isin(np.diff(saturatedLocal), (1.0, -19.0)))
    assert np.count_nonzero(np.diff(saturatedLocal) == -19.0) == 1
    assert (
        saturatedLocalRNG.bit_generator.random_raw()
        == saturatedGlobalRNG.bit_generator.random_raw()
    )
    with pytest.raises(ValueError, match="maxLocalRadiusIntervals"):
        nativeDraw(template, offsets, weights, 2.0, np.random.default_rng(0), -2)


def _caseReflectedTemplateAndMeanOccupancy(monkeypatch):
    negative = -np.linspace(0.1, 10.0, 1_000)
    positive = np.linspace(0.05, 25.0, 1_000)
    positive[::100] = 5.0
    scores = np.empty(2_000)
    scores[0::2], scores[1::2] = negative, positive
    weights = np.linspace(1.0, 3.0, scores.size)
    template, metadata = peaks._prepareNullResidualTemplate(scores, 0.0, weights)
    positiveMask = scores > 0.0
    ranks = peaks.stats.rankdata(scores[positiveMask], method="average")
    reflected = scores.copy()
    reflected[positiveMask] = np.quantile(
        -scores[scores < 0.0],
        (ranks - 0.5) / ranks.size,
        method="interpolated_inverted_cdf",
    )
    caps = np.quantile(reflected, (0.001, 0.999), method="interpolated_inverted_cdf")
    expected = np.clip(reflected, *caps)
    expected -= np.average(expected, weights=weights)
    lowerScale = np.median(-scores[scores < 0.0]) / peaks.stats.norm.ppf(0.75)
    expected *= lowerScale / np.sqrt(np.average(expected**2, weights=weights))
    np.testing.assert_allclose(template, expected)
    assert (metadata["template_clip_lower"], metadata["template_clip_upper"]) == pytest.approx(caps)
    assert caps[0] > np.min(reflected) and caps[1] < np.max(reflected)

    scoreTrack = np.asarray([-5, -4, -3, -2, 0.5, 0.5, 0.5, 0.5], dtype=float)
    calibrationTemplate = np.asarray([-2, -1, 0, 0, 0, 0, 1, 2], dtype=float)
    calibrationWeights = np.arange(1.0, 9.0)
    counts = np.arange(1, 9)
    drawIndex = [0]
    radiusCalls = []

    def fakeDraw(*args):
        radiusCalls.append(args[5])
        if drawIndex[0] >= counts.size:
            return calibrationTemplate.copy()
        draw = np.full(8, -2.0)
        draw[-counts[drawIndex[0]] :] = 2.0
        drawIndex[0] += 1
        return draw

    monkeypatch.setattr(peaks.cconsenrich, "cStationaryNullBootstrapDraw", fakeDraw)
    calibration = peaks._calibrateStationaryNullBootstrap(
        scoreTrack,
        calibrationTemplate,
        0.0,
        1.0,
        3,
        segmentOffsets=np.asarray([0, 4, 8]),
        coverageWeights=calibrationWeights,
        thresholdZ=0.0,
        numBootstrap=8,
    )
    assert drawIndex == [8]
    assert radiusCalls == [calibration["max_local_radius_intervals"]] * 8
    key = peaks._thresholdZKey(0.0)
    metrics = calibration["threshold_metrics"][key]
    totalWeight = np.sum(calibrationWeights)
    nullOccupancies = np.asarray(
        [np.sum(calibrationWeights[-count:]) / totalWeight for count in counts]
    )
    trackOccupancy = np.sum(calibrationWeights[4:]) / totalWeight
    nullOccupancy = np.mean(nullOccupancies)
    expectedMetrics = {
        "track_tail_occupancy": trackOccupancy,
        "null_tail_occupancy": nullOccupancy,
        "null_tail_occupancy_sd": np.std(nullOccupancies, ddof=1),
        "null_tail_mc_se": np.std(nullOccupancies, ddof=1) / np.sqrt(8.0),
        "signed_tail_excess": trackOccupancy - nullOccupancy,
        "budget_occupancy_raw": trackOccupancy - nullOccupancy,
    }
    assert {name: metrics[name] for name in expectedMetrics} == pytest.approx(expectedMetrics)
    prepared = {
        "score_track": scoreTrack,
        "template": calibrationTemplate,
        "threshold_views": calibration["threshold_views"],
        "threshold_metrics": calibration["threshold_metrics"],
        "null_calibration": {**calibration, "random_seed": 0},
    }
    assert peaks._estimateBudgetForPreparedROCCOScore(prepared) == pytest.approx(
        expectedMetrics["signed_tail_excess"]
    )
    peaks._scorePeakRecords(
        (),
        scoreTrack,
        prepared,
        np.arange(scoreTrack.size, dtype=np.int64),
        np.arange(1, scoreTrack.size + 1, dtype=np.int64),
        featureSpanBins=3,
        numRegionReplays=2,
    )
    assert radiusCalls == [calibration["max_local_radius_intervals"]] * 10
    radiusCalls.clear()
    disabledPrepared = {
        **prepared,
        "null_calibration": {
            **prepared["null_calibration"],
            "use_local_bootstrap_radius": False,
            "max_local_radius_intervals": -1,
        },
    }
    peaks._scorePeakRecords(
        (),
        scoreTrack,
        disabledPrepared,
        np.arange(scoreTrack.size, dtype=np.int64),
        np.arange(1, scoreTrack.size + 1, dtype=np.int64),
        featureSpanBins=3,
        numRegionReplays=2,
    )
    assert radiusCalls == [-1, -1]
    assert "null_quantile" not in calibration


def _caseSyntheticTailGates():
    rng = np.random.default_rng(20260831)
    size, rho = 8_192, 0.75
    iidGaussian = rng.normal(size=size)
    arGaussian = signal.lfilter(
        [np.sqrt(1.0 - rho**2)], [1.0, -rho], rng.normal(size=size)
    )

    def response(nullTrack, dose, seed):
        plantedTrack = nullTrack.copy()
        plantedTrack[: int(dose * size)] += 7.0
        prepared = peaks._prepareROCCOScoreAndNull(
            plantedTrack, 32, thresholdZ=2.0, numBootstrap=64, randomSeed=seed
        )
        return prepared["threshold_metrics"][peaks._thresholdZKey(2.0)][
            "signed_tail_excess"
        ]

    for nullTrack in (iidGaussian, arGaussian):
        medians = [
            np.median([response(nullTrack, dose, seed) for seed in (17, 31, 59)])
            for dose in (0.0, 0.01, 0.04)
        ]
        assert abs(medians[0]) <= 0.005
        assert medians[0] <= medians[1] <= medians[2]
        assert medians[2] - medians[0] >= 0.025


def _caseTrackTailVarianceAndChromosomeShrinkage():
    tail = np.r_[np.asarray([1, 0, 0, 0, 1], dtype=np.uint8), np.zeros(125)]
    weights = np.r_[np.asarray([40.0, 10.0, 10.0, 20.0, 20.0]), np.ones(125)]
    offsets = np.asarray([0, 3, 5, 130], dtype=np.int64)
    variance, details = peaks._estimateROCCOTrackTailVariance(
        tail, offsets, weights, minBlockMass=50.0
    )
    residuals = np.asarray([24.0, 28 / 3, -40 / 3, -40 / 3, -20 / 3])
    expectedVariance = (5 / 4) * np.dot(residuals, residuals) / 225**2
    assert variance == pytest.approx(expectedVariance)
    assert {
        name: details[name]
        for name in ("block_count", "underfilled_block_count", "merged_terminal_count")
    } == {"block_count": 5, "underfilled_block_count": 2, "merged_terminal_count": 1}

    occupancy, nullOccupancy, nullMCSE = 4 / 15, 0.08, 0.04
    prepared = {
        "score_track": tail,
        "threshold_views": {"z2": {"threshold": 0.5, "null_center": 0.0, "null_scale": 1.0}},
        "threshold_metrics": {"z2": {
            "track_tail_occupancy": occupancy,
            "null_tail_occupancy": nullOccupancy,
            "null_tail_mc_se": nullMCSE,
            "signed_tail_excess": occupancy - nullOccupancy,
            "budget_occupancy_raw": occupancy - nullOccupancy,
        }},
        "null_calibration": {
            "bootstrap_method": "stationary_bootstrap",
            "residual_span_intervals": 5,
            "segment_offsets": offsets,
            "coverage_weights": weights,
            "primary_key": "z2",
        },
    }
    budget, fit = peaks._estimateBudgetForPreparedROCCOScore(
        prepared, returnDetails=True, minBlockMass=50.0
    )
    assert budget == pytest.approx(occupancy - nullOccupancy)
    assert fit["track_tail_variance"] == pytest.approx(variance)
    assert fit["signed_tail_excess_se"] ** 2 == pytest.approx(variance + nullMCSE**2)

    observations = {
        "scaffoldA": (0.80, 0.03), "chr2": (0.10, 0.02),
        "chrM": (0.30, 0.40), "chr1": (-0.20, 0.00), "chr10": (0.00, 0.01),
    }
    names = sorted(observations)
    values = np.asarray([observations[name][0] for name in names])
    variances = np.square([observations[name][1] for name in names])
    pooledMean = np.mean(values)
    tauSquared = max(np.var(values, ddof=1) - np.mean(variances), 0.0)
    expectedRetention = np.maximum(
        np.divide(tauSquared, tauSquared + variances, out=np.ones(5), where=variances > 0),
        0.90,
    )
    expectedBudgets = np.clip(
        np.maximum(pooledMean + expectedRetention * (values - pooledMean), 0.0),
        0.001,
        0.25,
    )
    budgets, retention, metadata = peaks._shrinkROCCOChromosomeBudgets(observations)
    assert budgets == pytest.approx(dict(zip(names, expectedBudgets)))
    assert retention == pytest.approx(dict(zip(names, expectedRetention)))
    assert metadata["applied"] is True and metadata["method"] == "normalNormalMomentEB"
    assert peaks._shrinkROCCOChromosomeBudgets(dict(reversed(tuple(observations.items())))) == (
        budgets, retention, metadata
    )
    localBudgets = {name: float(np.clip(max(value, 0.0), 0.001, 0.25)) for name, (value, _) in observations.items()}
    for enabled, penalty, reason in ((False, None, "disabled"), (True, 0.4, "selectionPenaltyOverride")):
        skippedBudgets, skippedRetention, skipped = peaks._shrinkROCCOChromosomeBudgets(
            observations, enabled=enabled, selectionPenalty=penalty
        )
        assert skippedBudgets == pytest.approx(localBudgets)
        assert skippedRetention == {name: 1.0 for name in names}
        assert skipped["applied"] is False and skipped["skipReason"] == reason


def _caseSharedSeedAndSpoolCleanup(tmp_path, monkeypatch, caplog):
    caplog.clear()
    caplog.set_level(logging.INFO, logger=peaks.__name__)
    rows = []
    for chromosome, phase in (("chr21", 0.0), ("chr22", 0.4)):
        state = 0.2 * np.sin(np.linspace(phase, 8.0 + phase, 256))
        state[35:65] += 3.0
        state[145:175] += 4.0
        rows.extend((chromosome, 50 * i, 50 * (i + 1), value) for i, value in enumerate(state))
    statePath = tmp_path / "shared_seed_state.bedGraph"
    pd.DataFrame(rows).to_csv(statePath, sep="\t", header=False, index=False)
    seedCalls, drawCalls, mappedCalls, raiseOnSolve = [], [], [], [False]
    prepareScore = peaks._prepareROCCOScoreAndNull
    nativeDraw = peaks.cconsenrich.cStationaryNullBootstrapDraw

    def captureSeed(*args, **kwargs):
        seedCalls.append(kwargs["randomSeed"])
        return prepareScore(*args, **kwargs)

    def captureDraw(*args):
        drawCalls.append(args[5])
        return nativeDraw(*args)

    def captureSolve(scores, budget=None, gamma=None, selectionPenalty=None, returnDetails=False, **kwargs):
        backing = scores
        while backing is not None and not isinstance(backing, np.memmap):
            backing = getattr(backing, "base", None)
        mappedCalls.append(backing is not None)
        if raiseOnSolve[0]:
            raise RuntimeError("injected segmentation failure")
        solution = (np.asarray(scores) > 0.0).astype(np.uint8)
        details = {"selection_penalty": 0.0 if selectionPenalty is None else selectionPenalty}
        return (solution, 0.0, details) if returnDetails else (solution, 0.0)

    spoolRoot = tmp_path / "spool"
    spoolRoot.mkdir()
    sharedArgs = {
        "residualDurationBP": 400,
        "featureDurationBP": 500,
        "numBootstrap": 8,
        "numRegionReplays": 1,
        "gamma": 0.25,
        "nestedRoccoIters": 0,
        "randSeed": 73,
        "writeMetadata": False,
        "plotNullCalibrationDiagnostics": False,
    }
    with monkeypatch.context() as patch:
        patch.setattr(peaks.tempfile, "tempdir", str(spoolRoot))
        patch.setattr(peaks, "_prepareROCCOScoreAndNull", captureSeed)
        patch.setattr(peaks.cconsenrich, "cStationaryNullBootstrapDraw", captureDraw)
        patch.setattr(peaks, "solveChromROCCO", captureSolve)
        patch.setattr(peaks, "_scorePeakRecords", lambda records, *args, **kwargs: (list(records), ()))
        jointPath = tmp_path / "joint.narrowPeak"
        soloPath = tmp_path / "solo.narrowPeak"
        jointArtifacts = peaks.solveRocco(
            str(statePath), chromosomes=("chr21", "chr22"), outPath=str(jointPath), **sharedArgs
        )
        soloArtifacts = peaks.solveRocco(
            str(statePath), chromosomes=("chr22",), outPath=str(soloPath), **sharedArgs
        )
        assert jointArtifacts.nullCalibrationDiagnostics is None
        assert soloArtifacts.nullCalibrationDiagnostics is None
        assert not Path(f"{jointPath}.nullCalibration.png").exists()
        assert not Path(f"{soloPath}.nullCalibration.png").exists()
        assert seedCalls == [73, 73, 73]
        assert drawCalls == [20_000] * (3 * sharedArgs["numBootstrap"])
        jointChr22 = [line for line in jointPath.read_text().splitlines() if line.startswith("chr22\t")]
        assert jointChr22 and jointChr22 == soloPath.read_text().splitlines()
        assert all(mappedCalls) and not tuple(spoolRoot.iterdir())
        for chromosomeLabel in (
            "[1/2 chr21]",
            "[2/2 chr22]",
            "[1/1 chr22]",
        ):
            assert any(
                message
                == (
                    f"ROCCO {chromosomeLabel}: solving ROCCO constrained "
                    "optimization problem... γ=0.25"
                )
                for message in caplog.messages
            )
            assert any(
                message.startswith(f"ROCCO {chromosomeLabel}: λ=0 --> ")
                and message.endswith(", γ=0.25")
                for message in caplog.messages
            )

        seedCalls.clear()
        drawCalls.clear()
        raiseOnSolve[0] = True
        sharedArgs["useLocalBootStrapRadius"] = False
        with pytest.raises(RuntimeError, match="injected segmentation failure"):
            peaks.solveRocco(
                str(statePath),
                chromosomes=("chr21", "chr22"),
                outPath=str(tmp_path / "failure.narrowPeak"),
                **sharedArgs,
            )
        assert seedCalls == [73, 73]
        assert drawCalls == [-1] * (2 * sharedArgs["numBootstrap"])
        assert not tuple(spoolRoot.iterdir())


def _caseCheckMatchingEnabledHonorsEnabledFlag():
    matchingArgs = type("MatchingArgs", (), {"enabled": True})()
    assert consenrich_io.checkMatchingEnabled(matchingArgs) is True


def test_rocco_score_null_gamma_and_budget_contracts(monkeypatch, contract_case):
    contract_case("stationary bootstrap native contracts", _caseStationaryBootstrapNativeContracts)
    contract_case("synthetic null and planted tails", _caseSyntheticTailGates)
    contract_case("reflected template and mean occupancy", _caseReflectedTemplateAndMeanOccupancy, monkeypatch)
    contract_case("track-tail variance and chromosome shrinkage", _caseTrackTailVarianceAndChromosomeShrinkage)
    contract_case("candidate replay p and q", _caseReplayFDRModeratePanelsStaySubquadratic)


def test_rocco_bedgraph_solver_contracts(
    tmp_path, monkeypatch, contract_case, caplog
):
    caplog.set_level(logging.WARNING, logger=peaks.__name__)
    contract_case(
        "paired output",
        _caseRunROCCOBothModeWritesNarrowAndGapped,
        tmp_path,
        monkeypatch,
        caplog,
    )
    contract_case(
        "shared seed and spool cleanup",
        _caseSharedSeedAndSpoolCleanup,
        tmp_path,
        monkeypatch,
        caplog,
    )


def test_rocco_null_calibration_diagnostic_panel(tmp_path, monkeypatch, contract_case):
    contract_case(
        "null calibration diagnostic panel",
        _caseNullCalibrationDiagnosticPanel,
        tmp_path,
        monkeypatch,
    )


def test_rocco_subpeak_policy_contracts(contract_case):
    contract_case("broad merge policy", _caseBroadMergePolicyContracts)


def test_rocco_matching_enabled_contract(contract_case):
    contract_case("matching enabled flag", _caseCheckMatchingEnabledHonorsEnabledFlag)
