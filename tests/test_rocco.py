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

    empty, emptyDetails = peaks._mergeBroadRunsByObjective(
        [],
        np.zeros(3),
        np.arange(3, dtype=np.int64) * 10,
        np.arange(1, 4, dtype=np.int64) * 10,
        "chr1",
        selectionPenalty=0.0,
        boundaryCost=0.0,
        mergeToleranceBP=10,
        maxRegionBP=30,
        blacklistByChrom={},
    )
    assert empty == [] and emptyDetails["retained_utility"] == 0.0

    starts = np.arange(3, dtype=np.int64) * 10
    ends = starts + 10
    exactWidth, _ = peaks._mergeBroadRunsByObjective(
        [(0, 0), (2, 2)],
        np.asarray([1.0, 0.2, 1.0]),
        starts,
        ends,
        "chr1",
        selectionPenalty=0.0,
        boundaryCost=0.0,
        mergeToleranceBP=10,
        maxRegionBP=30,
        blacklistByChrom={"chr1": np.asarray([(20, 30)], dtype=np.int64)},
    )
    assert exactWidth == [(0, 2)]
    zeroGain, zeroDetails = peaks._mergeBroadRunsByObjective(
        [(0, 0), (2, 2)],
        np.asarray([1.0, 0.0, 1.0]),
        starts,
        ends,
        "chr1",
        selectionPenalty=0.0,
        boundaryCost=0.0,
        mergeToleranceBP=10,
        maxRegionBP=30,
        blacklistByChrom={},
    )
    assert zeroGain == [(0, 0), (2, 2)]
    assert zeroDetails["num_gaps_blocked_by_gain"] == 1
    singleton, _ = peaks._mergeBroadRunsByObjective(
        [(1, 1)],
        np.ones(3),
        starts,
        ends,
        "chr1",
        selectionPenalty=0.0,
        boundaryCost=0.0,
        mergeToleranceBP=10,
        maxRegionBP=10,
        blacklistByChrom={},
    )
    assert singleton == [(1, 1)]
    with pytest.raises(ValueError, match="ordered and disjoint"):
        peaks._mergeBroadRunsByObjective(
            [(2, 2), (0, 0)],
            np.ones(3),
            starts,
            ends,
            "chr1",
            selectionPenalty=0.0,
            boundaryCost=0.0,
            mergeToleranceBP=10,
            maxRegionBP=30,
            blacklistByChrom={},
        )

    def canonicalPartition(runs, edgeGains, intervals, ends, maxRegionBP):
        runCount = len(runs)
        edgePrefix = np.empty(runCount, dtype=np.float64)
        edgePrefix[0] = 0.0
        for index, gain in enumerate(edgeGains, start=1):
            edgePrefix[index] = float(edgePrefix[index - 1] + gain)
        utilities = np.full(runCount + 1, -np.inf)
        groups = np.full(runCount + 1, runCount + 1, dtype=np.int64)
        predecessors = np.full(runCount + 1, -1, dtype=np.int64)
        utilities[0] = 0.0
        groups[0] = 0
        for stop in range(1, runCount + 1):
            feasible = [
                start
                for start in range(stop)
                if int(ends[runs[stop - 1][1]])
                - int(intervals[runs[start][0]])
                <= maxRegionBP
            ]
            predecessor = max(
                feasible,
                key=lambda start: (
                    float(utilities[start] - edgePrefix[start]),
                    -int(groups[start]),
                    -start,
                ),
            )
            utilities[stop] = float(
                edgePrefix[stop - 1]
                + float(utilities[predecessor] - edgePrefix[predecessor])
            )
            groups[stop] = int(groups[predecessor] + 1)
            predecessors[stop] = predecessor
        partition = []
        stop = runCount
        while stop > 0:
            start = int(predecessors[stop])
            partition.append((runs[start][0], runs[stop - 1][1]))
            stop = start
        return list(reversed(partition)), float(utilities[runCount])

    rng = np.random.default_rng(835)
    for runCount in range(1, 9):
        for _ in range(32):
            size = 2 * runCount - 1
            widths = rng.integers(2, 12, size=size, dtype=np.int64)
            gaps = rng.integers(0, 4, size=max(size - 1, 0), dtype=np.int64)
            intervals = np.empty(size, dtype=np.int64)
            ends = np.empty(size, dtype=np.int64)
            cursor = 0
            for index in range(size):
                intervals[index] = cursor
                ends[index] = cursor + int(widths[index])
                if index + 1 < size:
                    cursor = int(ends[index] + gaps[index])
            runs = [(2 * index, 2 * index) for index in range(runCount)]
            gapScores = rng.choice(
                np.asarray([0.1, 0.2, 0.3]),
                size=max(runCount - 1, 0),
            )
            scores = np.zeros(size, dtype=np.float64)
            scores[1::2] = gapScores
            minimumWidth = max(
                int(ends[end] - intervals[start]) for start, end in runs
            )
            chromosomeWidth = int(ends[-1] - intervals[0])
            maxRegionBP = int(rng.integers(minimumWidth, chromosomeWidth + 1))
            expectedRuns, expectedUtility = canonicalPartition(
                runs,
                gapScores,
                intervals,
                ends,
                maxRegionBP,
            )
            merged, details = peaks._mergeBroadRunsByObjective(
                runs,
                scores,
                intervals,
                ends,
                "chrRandom",
                selectionPenalty=0.0,
                boundaryCost=0.0,
                mergeToleranceBP=chromosomeWidth,
                maxRegionBP=maxRegionBP,
                blacklistByChrom={},
            )
            assert merged == expectedRuns
            assert details["retained_utility"] == expectedUtility

    runCount = 8
    intervals = np.arange(2 * runCount - 1, dtype=np.int64) * 10
    ends = intervals + 10
    runs = [(2 * index, 2 * index) for index in range(runCount)]
    equalGains = np.full(runCount - 1, 0.2)
    scores = np.zeros(intervals.size)
    scores[1::2] = equalGains
    expectedRuns, expectedUtility = canonicalPartition(
        runs,
        equalGains,
        intervals,
        ends,
        50,
    )
    merged, details = peaks._mergeBroadRunsByObjective(
        runs,
        scores,
        intervals,
        ends,
        "chrTied",
        selectionPenalty=0.0,
        boundaryCost=0.0,
        mergeToleranceBP=10,
        maxRegionBP=50,
        blacklistByChrom={},
    )
    assert merged == expectedRuns
    assert details["retained_utility"] == expectedUtility


def _caseRunROCCOBothModeWritesNarrowAndGapped(tmp_path, monkeypatch, caplog):
    caplog.clear()
    caplog.set_level(logging.INFO, logger=peaks.__name__)
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
    replayDrawCalls = []
    candidateScanCalls = []
    nativeDraw = peaks.cconsenrich.cStationaryNullBootstrapDraw
    candidateScan = peaks._multiscaleCandidateSegments

    def capturePlot(rows, path, *, dpi=400):
        plotCalls.append((tuple(rows), str(path), dpi))
        Path(path).write_bytes(b"png")
        return True

    def captureReplayDraw(*args):
        replayDrawCalls.append(args[5])
        return nativeDraw(*args)

    def captureCandidateScan(*args, **kwargs):
        candidateScanCalls.append(1)
        return candidateScan(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(peaks, "_plotROCCONullCalibrationDiagnostics", capturePlot)
        patch.setattr(
            peaks.cconsenrich,
            "cStationaryNullBootstrapDraw",
            captureReplayDraw,
        )
        patch.setattr(
            peaks,
            "_multiscaleCandidateSegments",
            captureCandidateScan,
        )
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
    assert len(replayDrawCalls) == 8 + 2
    assert len(candidateScanCalls) == 2 + 1
    replayMessages = [
        message for message in caplog.messages if "candidate replay" in message
    ]
    assert replayMessages == [
        "ROCCO [1/1 chr1]: candidate replay 1/2",
        "ROCCO [1/1 chr1]: candidate replay 2/2",
    ]
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
    assert metadata["chromosomes"]["chr1"]["narrow"][
        "replayNullCandidateCounts"
    ] == metadata["chromosomes"]["chr1"]["broad"][
        "replayNullCandidateCounts"
    ]


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
    nullDraws = [
        np.sort(rng.gamma(shape=2.2, scale=1.0, size=3000))
        for _ in range(32)
    ]
    for draw in nullDraws:
        draw.setflags(write=False)
    pooledNull = np.sort(np.concatenate(nullDraws))
    pooledNull.setflags(write=False)
    nullCopies = tuple(draw.copy() for draw in nullDraws)
    pValues = empiricalP(observed, pooledNull)
    qValues = np.maximum(replayQ(observed, nullDraws), pValues)
    assert pValues.shape == qValues.shape == observed.shape
    assert np.all((0.0 <= pValues) & (pValues <= qValues) & (qValues <= 1.0))
    assert np.any(qValues > pValues + 1.0e-6)
    order = np.argsort(-observed, kind="mergesort")
    assert np.all(np.diff(pValues[order]) >= -1.0e-12)
    assert np.all(np.diff(qValues[order]) >= -1.0e-12)
    for draw, expected in zip(nullDraws, nullCopies):
        np.testing.assert_array_equal(draw, expected)
        assert not draw.flags.writeable

    tiedObserved = np.asarray([3.0, 1.0, 3.0, 0.0])
    tiedNullDraws = (
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([1.0, 1.0, 3.0]),
    )
    tiedPooledNull = np.sort(np.concatenate(tiedNullDraws))
    expectedP = np.asarray(
        [
            (1.0 + np.count_nonzero(tiedPooledNull >= statistic))
            / float(tiedPooledNull.size + 1)
            for statistic in tiedObserved
        ]
    )
    order = np.argsort(-tiedObserved, kind="mergesort")
    rawFDR = np.ones(tiedObserved.size)
    for rank, index in enumerate(order):
        statistic = tiedObserved[index]
        observedCount = np.count_nonzero(tiedObserved >= statistic)
        expectedNullCount = np.mean(
            [np.count_nonzero(draw >= statistic) for draw in tiedNullDraws]
        )
        rawFDR[rank] = np.clip(
            (expectedNullCount + 1.0 / 3.0) / observedCount,
            0.0,
            1.0,
        )
    expectedQ = np.ones(tiedObserved.size)
    expectedQ[order] = np.minimum.accumulate(rawFDR[::-1])[::-1]
    np.testing.assert_array_equal(
        empiricalP(tiedObserved, tiedPooledNull),
        expectedP,
    )
    np.testing.assert_array_equal(
        replayQ(tiedObserved, tiedNullDraws),
        expectedQ,
    )
    assert empiricalP([], tiedPooledNull).size == 0
    assert replayQ([], tiedNullDraws).size == 0
    np.testing.assert_array_equal(empiricalP(tiedObserved, []), np.ones(4))


def _caseCoordinateRunNativeContracts():
    nativeRuns = peaks.cconsenrich.cSelectedCoordinateRunBounds
    rng = np.random.default_rng(28041)
    for size in (0, 1, 2, 9, 64):
        for panel in range(8):
            widths = rng.integers(1, 12, size=size, dtype=np.int64)
            gaps = rng.integers(0, 3, size=size, dtype=np.int64)
            starts = np.cumsum(widths + gaps, dtype=np.int64) - widths
            ends = starts + widths
            mask = rng.random(size) < (panel / 7.0)
            for stride in (1, 2, -1):
                maskView, startsView, endsView = (
                    array[::stride] for array in (mask, starts, ends)
                )
                connected = (
                    maskView[:-1]
                    & maskView[1:]
                    & (endsView[:-1] == startsView[1:])
                )
                expected = list(zip(
                    np.flatnonzero(maskView & ~np.r_[False, connected][:maskView.size]),
                    np.flatnonzero(maskView & ~np.r_[connected, False][:maskView.size]),
                ))
                copies = tuple(array.copy() for array in (maskView, startsView, endsView))
                for array in (maskView, startsView, endsView):
                    array.setflags(write=False)
                assert nativeRuns(maskView, startsView, endsView) == expected
                assert peaks._selectedCoordinateRunBounds(maskView, startsView, endsView) == expected
                for array, copy in zip((maskView, startsView, endsView), copies):
                    np.testing.assert_array_equal(array, copy)

    starts = np.asarray([-(2**63), -(2**63) + 1, 2**63 - 3, 2**63 - 2], dtype=np.int64)
    ends = starts + 1
    mask = np.ones(4, dtype=bool)
    assert nativeRuns(mask, starts, ends) == [(0, 1), (2, 3)]
    assert peaks._selectedCoordinateRunBounds(
        [1, 1, 0, 1], [[0, 2], [4, 6]], [[2, 4], [6, 8]]
    ) == [(0, 1), (3, 3)]
    for function in (nativeRuns, peaks._selectedCoordinateRunBounds):
        for args in ((mask, starts[:-1], ends), (mask, starts, ends[:-1])):
            with pytest.raises(ValueError, match="match length"):
                function(*args)
        with pytest.raises(ValueError):
            function(mask.reshape(2, 2), starts, ends)
    for args in (
        (mask, starts.reshape(2, 2), ends),
        (mask, starts, ends.astype(np.int32)),
        (None, starts, ends),
    ):
        with pytest.raises((TypeError, ValueError)):
            nativeRuns(*args)


def _caseSubpeakDPNativeContracts():
    nativeSolve = peaks.cconsenrich.cSolveParentConditionedSubpeaks
    rng = np.random.default_rng(88104)
    cases = []
    for size in range(9):
        for panel in range(6):
            scores = rng.normal((-2.0, 0.0, 2.0)[panel % 3], 0.4, size)
            costs = rng.uniform(0.0, 0.5, size + 1)
            selectionPenalty = (-0.2, 0.0, 0.3)[panel % 3]
            runPenalty = (-0.1, 0.0, 0.4)[panel // 2]
            requiredIndex = None if size == 0 or panel < 3 else (0, size // 2, size - 1)[panel - 3]
            for minRunBins in {-2, 1, 2, size, size + 3}:
                minimum = min(max(minRunBins, 1), size)
                candidates = []
                for encoded in range(1 << size):
                    mask = ((encoded >> np.arange(size)) & 1).astype(bool)
                    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
                    if np.any(edges[1::2] - edges[::2] < minimum):
                        continue
                    if requiredIndex is not None and not mask[requiredIndex]:
                        continue
                    boundaryPenalty = sum(float(costs[index]) for index in edges)
                    objective = float(np.sum(scores[mask]) - boundaryPenalty - runPenalty * (edges.size // 2))
                    candidates.append((objective - selectionPenalty * int(mask.sum()), -int(mask.sum()), -encoded, mask))
                expected = max(candidates, key=lambda item: item[:3])[3]
                cases.append((scores, costs, selectionPenalty, minRunBins, requiredIndex, runPenalty, expected))

    randomCaseCount = len(cases)
    for scores, minimum, requiredIndex, expected in (
        ([0.0], 1, None, [False]),
        ([5.0e-13], 1, None, [True]),
        ([5.0e-13, -1.0], 1, None, [False, False]),
        ([2.0e-12, -1.0], 1, None, [True, False]),
        ([-2.0, -2.0, -2.0], 2, 1, [True, True, False]),
        ([1.0, -1.0, 1.0], 1, None, [True, False, True]),
    ):
        cases.append((np.asarray(scores), np.zeros(len(scores) + 1), 0.0, minimum, requiredIndex, 0.0, np.asarray(expected)))

    for caseIndex, (scores, costs, selectionPenalty, minimum, requiredIndex, runPenalty, expected) in enumerate(cases):
        scoreStorage = np.repeat(scores, 2)
        costStorage = np.repeat(costs, 2)
        scores = scoreStorage[::2]
        costs = costStorage[::2]
        reverse = caseIndex < randomCaseCount and caseIndex % 2
        if reverse:
            scores = scores[::-1]
            costs = costs[::-1]
            expected = expected[::-1]
            requiredIndex = None if requiredIndex is None else scores.size - 1 - requiredIndex
        scores.setflags(write=False)
        costs.setflags(write=False)
        args = (scores, costs, selectionPenalty, minimum, requiredIndex, runPenalty)
        actual, objective, details = peaks._solveParentConditionedSubpeaks(*args)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(nativeSolve(*args), expected)
        assert actual.dtype == np.bool_ and actual.shape == scores.shape
        edges = np.flatnonzero(np.diff(np.r_[False, expected, False]))
        boundaryPenalty = 0.0
        for index in edges:
            boundaryPenalty += float(costs[index])
        runCount = edges.size // 2
        runPenaltyTotal = float(runPenalty * runCount)
        selectedCount = int(expected.sum())
        expectedObjective = float(np.sum(scores[expected]) - boundaryPenalty - runPenaltyTotal)
        assert objective == expectedObjective
        assert details == {
            "mode": "parent_conditioned_min_run_dp",
            "penalized_objective": float(expectedObjective - selectionPenalty * selectedCount),
            "selected_count": selectedCount,
            "selected_fraction": float(selectedCount / max(scores.size, 1)),
            "selection_penalty": selectionPenalty,
            "run_penalty": runPenalty,
            "run_penalty_total": runPenaltyTotal,
            "boundary_cost_min": float(np.min(costs)),
            "boundary_cost_max": float(np.max(costs)),
            "boundary_penalty": boundaryPenalty,
            "min_run_bins": min(max(minimum, 1), scores.size),
            "num_runs": runCount,
            "required_index": requiredIndex,
            "required_selected": True,
            "required_fallback_window": False,
        }
        np.testing.assert_array_equal(scoreStorage[::2], scores[::(-1 if reverse else 1)])
        np.testing.assert_array_equal(costStorage[::2], costs[::(-1 if reverse else 1)])

    for function in (nativeSolve, peaks._solveParentConditionedSubpeaks):
        for scores, costs, requiredIndex, message in (
            (np.ones((2, 2)), np.ones(5), None, "one-dimensional"),
            (np.ones(2), np.ones(2), None, "length"),
            (np.ones(2), np.ones(4), None, "length"),
            (np.ones(2), np.ones(3), -1, "requiredIndex"),
            (np.ones(2), np.ones(3), 2, "requiredIndex"),
            (np.empty(0), np.ones(1), 0, "requiredIndex"),
        ):
            with pytest.raises(ValueError, match=message):
                function(scores, costs, 0.0, 1, requiredIndex)
        with pytest.raises(RuntimeError, match="no feasible path"):
            function(np.asarray([-np.inf]), np.zeros(2), 0.0, 1, 0)
        result = function([2.0, 2.0], [0.0, 0.0, 0.0], 0.0, 10**100)
        np.testing.assert_array_equal(result if function is nativeSolve else result[0], [True, True])
    with pytest.raises(ValueError, match="one-dimensional"):
        nativeSolve(np.ones(3), np.ones((2, 2)), 0.0, 1)
    flattened = peaks._solveParentConditionedSubpeaks(np.ones(3), np.zeros((2, 2)), 0.0, 1)
    np.testing.assert_array_equal(flattened[0], [True, True, True])


def _caseRegionalSignalPrefixAndRunSweeps():
    intervals = np.asarray([0, 8, 31, 38, 75, 89, 130], dtype=np.int64)
    ends = np.asarray([5, 19, 35, 47, 82, 103, 133], dtype=np.int64)
    widths = ends - intervals
    signalValues = np.asarray(
        [1.25, -4.0, 9.5, 2.75, -1.5, 6.25, 3.0],
        dtype=np.float64,
    )
    signalPrefix = peaks._buildRegionalSignalPrefix(widths, signalValues)
    maxAbsSignal = float(np.max(np.abs(signalValues)))
    for startIdx, endIdx in ((0, 0), (3, 3), (0, 6), (1, 4), (4, 6)):
        startBP = int(intervals[startIdx])
        endBP = int(ends[endIdx])
        overlap = np.maximum(
            np.minimum(ends, endBP) - np.maximum(intervals, startBP),
            0,
        ).astype(np.float64)
        expected = float(np.dot(overlap, signalValues) / np.sum(overlap))
        np.testing.assert_allclose(
            peaks._regionalMeanSignal(signalPrefix, startIdx, endIdx),
            expected,
            rtol=1.0e-10,
            atol=1.0e-12 * max(1.0, maxAbsSignal),
        )

    with pytest.raises(ValueError, match="non-finite"):
        peaks._buildRegionalSignalPrefix(widths, np.r_[signalValues[:-1], np.nan])
    with pytest.raises(ValueError, match="positive"):
        peaks._buildRegionalSignalPrefix(np.r_[widths[:-1], 0], signalValues)
    hugePrefix = peaks._buildRegionalSignalPrefix(
        np.asarray([2**53, 1], dtype=np.int64),
        np.asarray([1.0, 2.0]),
    )
    assert peaks._regionalMeanSignal(hugePrefix, 1, 1) == 2.0
    with pytest.raises(ValueError, match="indices"):
        peaks._regionalMeanSignal(signalPrefix, 2, signalValues.size)

    rng = np.random.default_rng(1741)
    for _ in range(64):
        size = int(rng.integers(4, 48))
        randomWidths = rng.integers(1, 13, size=size, dtype=np.int64)
        randomGaps = rng.integers(0, 4, size=size - 1, dtype=np.int64)
        randomStarts = np.empty(size, dtype=np.int64)
        randomEnds = np.empty(size, dtype=np.int64)
        cursor = 0
        for index in range(size):
            randomStarts[index] = cursor
            randomEnds[index] = cursor + int(randomWidths[index])
            if index + 1 < size:
                cursor = int(randomEnds[index] + randomGaps[index])
        candidateRuns = peaks._selectedCoordinateRunBounds(
            rng.integers(0, 2, size=size, dtype=np.uint8),
            randomStarts,
            randomEnds,
        )
        referenceRuns = peaks._selectedCoordinateRunBounds(
            rng.integers(0, 2, size=size, dtype=np.uint8),
            randomStarts,
            randomEnds,
        )
        expected = [
            candidateRun
            for candidateRun in candidateRuns
            if any(
                int(randomEnds[candidateRun[1]])
                >= int(randomStarts[referenceRun[0]])
                and int(randomStarts[candidateRun[0]])
                <= int(randomEnds[referenceRun[1]])
                for referenceRun in referenceRuns
            )
        ]
        assert peaks._runsTouchingReferenceRuns(
            candidateRuns,
            referenceRuns,
            randomStarts,
            randomEnds,
        ) == expected

    touchingStarts = np.asarray([0, 10, 21], dtype=np.int64)
    touchingEnds = np.asarray([10, 20, 31], dtype=np.int64)
    assert peaks._runsTouchingReferenceRuns(
        [(0, 0), (2, 2)],
        [(1, 1)],
        touchingStarts,
        touchingEnds,
    ) == [(0, 0)]
    assert peaks._runsTouchingReferenceRuns(
        [],
        [(1, 1)],
        touchingStarts,
        touchingEnds,
    ) == []
    with pytest.raises(ValueError, match="ordered and disjoint"):
        peaks._runsTouchingReferenceRuns(
            [(2, 2), (0, 0)],
            [(1, 1)],
            touchingStarts,
            touchingEnds,
        )


def _caseBroadRecordSweepContracts():
    intervals = np.asarray(
        [0, 5, 12, 16, 27, 50, 56, 64, 80, 100],
        dtype=np.int64,
    )
    ends = np.asarray(
        [5, 12, 16, 27, 30, 56, 64, 69, 89, 104],
        dtype=np.int64,
    )
    signalValues = np.arange(1.0, 11.0)
    scores = np.linspace(-1.0, 2.0, signalValues.size)
    signalPrefix = peaks._buildRegionalSignalPrefix(
        ends - intervals,
        signalValues,
    )
    parentRuns = [(0, 4), (5, 9)]
    supportRuns = [(0, 4), (5, 9)]
    blockRuns = [(1, 6), (7, 7), (8, 8)]
    records = peaks._broadRecordsFromRuns(
        "chrSweep",
        intervals,
        ends,
        signalValues,
        scores,
        parentRuns,
        supportRuns,
        blockRuns,
        signalPrefix,
    )

    literalRecords = []
    for parentStart, parentEnd in parentRuns:
        overlappingSupport = [
            (start, end)
            for start, end in supportRuns
            if end >= parentStart and start <= parentEnd
        ]
        startBP = int(intervals[overlappingSupport[0][0]])
        endBP = int(ends[overlappingSupport[-1][1]])
        firstIdx = min(start for start, _end in overlappingSupport)
        lastIdx = max(end for _start, end in overlappingSupport)
        blockCoordinates = sorted(
            (
                int(intervals[max(start, parentStart)]),
                int(ends[min(end, parentEnd)]),
            )
            for start, end in blockRuns
            if end >= parentStart and start <= parentEnd
        )
        mergedBlocks = []
        for blockStart, blockEnd in blockCoordinates:
            if mergedBlocks and blockStart <= mergedBlocks[-1][1]:
                mergedBlocks[-1] = (
                    mergedBlocks[-1][0],
                    max(mergedBlocks[-1][1], blockEnd),
                )
            else:
                mergedBlocks.append((blockStart, blockEnd))
        if len(mergedBlocks) <= 1:
            mergedBlocks = [(startBP, endBP)]
        else:
            mergedBlocks[0] = (startBP, mergedBlocks[0][1])
            mergedBlocks[-1] = (mergedBlocks[-1][0], endBP)
        summitIdx = int(
            firstIdx + np.argmax(signalValues[firstIdx : lastIdx + 1])
        )
        literalRecords.append(
            peaks._peakRecord(
                "chrSweep",
                startBP,
                endBP,
                int(
                    intervals[summitIdx]
                    + (int(ends[summitIdx]) - int(intervals[summitIdx])) // 2
                ),
                "broad",
                float(np.mean(scores[firstIdx : lastIdx + 1])),
                float(
                    np.average(
                        signalValues[firstIdx : lastIdx + 1],
                        weights=(ends - intervals)[firstIdx : lastIdx + 1],
                    )
                ),
                1.0,
                1.0,
                tuple(mergedBlocks),
            )
        )
    for record, expectedRecord in zip(records, literalRecords):
        assert record._replace(signalValue=0.0) == expectedRecord._replace(
            signalValue=0.0
        )
        assert record.signalValue == pytest.approx(
            expectedRecord.signalValue,
            rel=1.0e-10,
            abs=1.0e-12 * max(1.0, abs(expectedRecord.signalValue)),
        )

    assert [(record.startBP, record.endBP) for record in records] == [
        (0, 30),
        (50, 104),
    ]
    assert [record.blocks for record in records] == [
        ((0, 30),),
        ((50, 69), (80, 104)),
    ]
    assert [record.summitBP for record in records] == [28, 102]
    for record, (startIdx, endIdx) in zip(records, ((0, 4), (5, 9))):
        expectedSignal = np.average(
            signalValues[startIdx : endIdx + 1],
            weights=(ends - intervals)[startIdx : endIdx + 1],
        )
        assert record.signalValue == pytest.approx(expectedSignal)
        assert record.rawScore == pytest.approx(
            np.mean(scores[startIdx : endIdx + 1])
        )

    assert peaks._broadRecordsFromRuns(
        "chrSweep",
        intervals,
        ends,
        signalValues,
        scores,
        [],
        [],
        [],
        signalPrefix,
    ) == []
    with pytest.raises(RuntimeError, match="no support"):
        peaks._broadRecordsFromRuns(
            "chrSweep",
            intervals,
            ends,
            signalValues,
            scores,
            [(0, 1)],
            [(3, 4)],
            [],
            signalPrefix,
        )
    with pytest.raises(ValueError, match="ordered and disjoint"):
        peaks._broadRecordsFromRuns(
            "chrSweep",
            intervals,
            ends,
            signalValues,
            scores,
            [(5, 9), (0, 4)],
            [(0, 4), (5, 9)],
            [],
            signalPrefix,
        )


def _caseSharedCandidateReplayContracts(monkeypatch):
    scores = np.asarray([-2.0, 0.5, 2.0, -1.0, 1.5, 3.0])
    intervals = np.arange(scores.size, dtype=np.int64) * 100
    ends = intervals + 100
    thresholdViews = {
        "primary": {
            "threshold_z": 1.0,
            "threshold": 0.25,
            "null_center": 0.0,
            "null_scale": 1.0,
        }
    }
    prepared = {
        "template": np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]),
        "threshold_views": thresholdViews,
        "null_calibration": {
            "bootstrap_method": "stationary_bootstrap",
            "bootstrap_block_length": 2,
            "use_local_bootstrap_radius": False,
            "max_local_radius_intervals": -1,
            "random_seed": 41,
            "segment_offsets": np.asarray([0, scores.size], dtype=np.int64),
            "coverage_weights": np.ones(scores.size, dtype=np.float64),
        },
    }
    nativeDraw = peaks.cconsenrich.cStationaryNullBootstrapDraw
    candidateScan = peaks._multiscaleCandidateSegments
    drawCalls = []
    candidateScanCalls = []

    def captureDraw(*args):
        drawCalls.append(args[5])
        return nativeDraw(*args)

    def captureCandidateScan(*args, **kwargs):
        candidateScanCalls.append(1)
        return candidateScan(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            peaks.cconsenrich,
            "cStationaryNullBootstrapDraw",
            captureDraw,
        )
        patch.setattr(
            peaks,
            "_multiscaleCandidateSegments",
            captureCandidateScan,
        )
        replayData = peaks._buildCandidateReplayData(
            scores,
            prepared,
            featureSpanBins=3,
            numRegionReplays=4,
        )
    assert drawCalls == [-1] * 4
    assert len(candidateScanCalls) == 4 + 1
    assert len(replayData.nullStatsByDraw) == 4
    assert replayData.nullCandidateCounts == tuple(
        draw.size for draw in replayData.nullStatsByDraw
    )
    assert not replayData.pooledNullStats.flags.writeable
    assert all(
        not draw.flags.writeable and np.all(np.diff(draw) >= 0.0)
        for draw in replayData.nullStatsByDraw
    )
    assert np.all(np.diff(replayData.pooledNullStats) >= 0.0)
    replayCopies = tuple(draw.copy() for draw in replayData.nullStatsByDraw)
    pooledCopy = replayData.pooledNullStats.copy()

    narrowRecord = peaks._peakRecord(
        "chrReplay", 0, 200, 50, "narrow", 2.0, 1.0, 1.0, 1.0, ((0, 200),)
    )
    broadRecord = peaks._peakRecord(
        "chrReplay", 200, 600, 550, "broad", 1.5, 1.0, 1.0, 1.0,
        ((200, 400), (500, 600)),
    )
    narrowFirst = peaks._scorePeakRecords(
        [narrowRecord], scores, replayData, intervals, ends
    )
    broadSecond = peaks._scorePeakRecords(
        [broadRecord], scores, replayData, intervals, ends
    )
    broadFirst = peaks._scorePeakRecords(
        [broadRecord], scores, replayData, intervals, ends
    )
    narrowSecond = peaks._scorePeakRecords(
        [narrowRecord], scores, replayData, intervals, ends
    )
    assert narrowFirst == narrowSecond
    assert broadFirst == broadSecond
    for draw, expected in zip(replayData.nullStatsByDraw, replayCopies):
        np.testing.assert_array_equal(draw, expected)
        assert not draw.flags.writeable
    np.testing.assert_array_equal(replayData.pooledNullStats, pooledCopy)
    assert not replayData.pooledNullStats.flags.writeable

    emptyNull = np.asarray([], dtype=np.float64)
    emptyNull.setflags(write=False)
    emptyReplay = peaks._candidateReplayData(
        thresholdViews,
        (),
        (emptyNull,),
        emptyNull,
        (0,),
    )
    duplicateScores = peaks._scorePeakRecords(
        [narrowRecord, narrowRecord],
        scores,
        emptyReplay,
        intervals,
        ends,
    )
    assert [(record.pValue, record.qValue) for record in duplicateScores] == [
        (1.0, 1.0),
        (1.0, 1.0),
    ]

    familyScores = np.asarray([1.0, 4.0, 2.0, 3.0, 0.5])
    familyIntervals = np.arange(familyScores.size, dtype=np.int64) * 10
    familyEnds = familyIntervals + 10
    familyViews = {
        "primary": {
            "threshold_z": 0.0,
            "threshold": 0.0,
            "null_center": 0.0,
            "null_scale": 1.0,
        }
    }
    familyNullDraws = (
        np.asarray([0.5, 1.5, 2.5, 3.5, 4.5]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    for draw in familyNullDraws:
        draw.setflags(write=False)
    familyPooledNull = np.sort(np.concatenate(familyNullDraws))
    familyPooledNull.setflags(write=False)
    familyReplay = peaks._candidateReplayData(
        familyViews,
        (((0, 4), 5.0),),
        familyNullDraws,
        familyPooledNull,
        (5, 4),
    )
    familyNarrowRecords = [
        peaks._peakRecord(
            "chrFamily", 0, 10, 5, "narrow", 1.0, 1.0, 1.0, 1.0,
            ((0, 10),),
        ),
        peaks._peakRecord(
            "chrFamily", 30, 40, 35, "narrow", 3.0, 3.0, 1.0, 1.0,
            ((30, 40),),
        ),
    ]
    familyBroadRecords = [
        peaks._peakRecord(
            "chrFamily", 10, 30, 15, "broad", 4.0, 3.0, 1.0, 1.0,
            ((10, 30),),
        )
    ]
    scoredNarrow = peaks._scorePeakRecords(
        familyNarrowRecords,
        familyScores,
        familyReplay,
        familyIntervals,
        familyEnds,
    )
    scoredBroad = peaks._scorePeakRecords(
        familyBroadRecords,
        familyScores,
        familyReplay,
        familyIntervals,
        familyEnds,
    )
    assert [(record.pValue, record.qValue) for record in scoredNarrow] == [
        (0.9, 1.0),
        (0.5, 1.0),
    ]
    assert [(record.pValue, record.qValue) for record in scoredBroad] == [
        (0.2, (0.5 + 1.0 / 3.0) / 2.0)
    ]
    pooledFamilyScores = peaks._scorePeakRecords(
        familyNarrowRecords + familyBroadRecords,
        familyScores,
        familyReplay,
        familyIntervals,
        familyEnds,
    )
    assert pooledFamilyScores[1].qValue == pytest.approx(7.0 / 9.0)
    assert pooledFamilyScores[1].qValue != scoredNarrow[1].qValue


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
    peaks._buildCandidateReplayData(
        scoreTrack,
        prepared,
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
    peaks._buildCandidateReplayData(
        scoreTrack,
        disabledPrepared,
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
        patch.setattr(
            peaks,
            "_scorePeakRecords",
            lambda records, *args, **kwargs: list(records),
        )
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
        assert drawCalls == [20_000] * (
            3
            * (
                sharedArgs["numBootstrap"]
                + sharedArgs["numRegionReplays"]
            )
        )
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
    contract_case(
        "coordinate run native contracts",
        _caseCoordinateRunNativeContracts,
    )
    contract_case(
        "regional signal prefix and run sweeps",
        _caseRegionalSignalPrefixAndRunSweeps,
    )
    contract_case(
        "shared candidate replay",
        _caseSharedCandidateReplayContracts,
        monkeypatch,
    )


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
    contract_case("subpeak DP native contracts", _caseSubpeakDPNativeContracts)
    contract_case("broad merge policy", _caseBroadMergePolicyContracts)
    contract_case("broad record sweep", _caseBroadRecordSweepContracts)


def test_rocco_matching_enabled_contract(contract_case):
    contract_case("matching enabled flag", _caseCheckMatchingEnabledHonorsEnabledFlag)
