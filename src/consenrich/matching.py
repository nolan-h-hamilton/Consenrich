# -*- coding: utf-8 -*-
r"""Module implementing (experimental) 'structured peak detection' features using wavelet-based templates."""

import logging
import os
import math
from typing import List, Optional
import pandas as pd
import pywt as pw
import numpy as np
import numpy.typing as npt

from scipy import signal, stats

from . import cconsenrich
from . import core as core
from . import __version__
from . import mergeNarrowPeaks as mergeNarrowPeaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _FDR(pVals: np.ndarray, method: str | None = "bh") -> np.ndarray:
    if method is None:
        return pVals
    return stats.false_discovery_control(pVals, method=method.lower())


def autoMinLengthIntervals(
    values: np.ndarray,
    initLen: int = 5,
    maxLen: int = 25,
) -> int:
    try:
        hlen = int(
            core.getContextSize(
                values,
            )[0]
            / 2.0
        )
    except Exception:
        logger.warning(
            "autoMinLengthIntervals: could not compute context size, using default length."
        )
        hlen = initLen
    return min(max(int(hlen), initLen), maxLen)


def castableToFloat(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        if value.lower().replace(" ", "") in [
            "nan",
            "inf",
            "-inf",
            "infinity",
            "-infinity",
            "",
            " ",
        ]:
            return False

    try:
        float(value)
        if np.isfinite(float(value)):
            return True
    except Exception:
        return False
    return False


def matchWavelet(
    chromosome: str,
    intervals: npt.NDArray[int],
    values: npt.NDArray[np.float64],
    templateNames: List[str],
    cascadeLevels: List[int],
    iters: int,
    alpha: float = 0.05,
    minMatchLengthBP: Optional[int] = -1,
    maxNumMatches: Optional[int] = 100_000,
    minSignalAtMaxima: Optional[float | str] = 0.1,
    randSeed: int = 42,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    weights: Optional[npt.NDArray[np.float64]] = None,
    eps: float = 1.0e-3,
    methodFDR: str | None = None,
) -> pd.DataFrame:
    r"""Detect structured peaks in Consenrich tracks by matching wavelet- or scaling-function-based templates.

    :param chromosome: Chromosome name for the input intervals and values.
    :type chromosome: str
    :param values: A 1D array of signal-like values. In this documentation, we refer to values derived from Consenrich,
        but other continuous-valued tracks at evenly spaced genomic intervals may be suitable, too.
    :type values: npt.NDArray[np.float64]
    :param templateNames: A list of str values -- each entry references a mother wavelet (or its corresponding scaling function). e.g., `[haar, db2]`
    :type templateNames: List[str]
    :param cascadeLevels: Number of cascade iterations used to approximate each template (wavelet or scaling function).
        Must have the same length as `templateNames`, with each entry aligned to the
        corresponding template. e.g., given templateNames `[haar, db2]`, then `[2,2]` would use 2 cascade levels for both templates.
    :type cascadeLevels: List[int]
    :param iters: Number of random blocks to sample in the response sequence while building
        an empirical null to test significance within chromosomes. See :func:`cconsenrich.csampleBlockStats`.
    :type iters: int
    :param alpha: Primary significance threshold on detected matches. Specifically, the
        minimum corrected empirical p-value approximated from randomly sampled blocks in the
        response sequence.
    :type alpha: float
    :param minMatchLengthBP: Within a window of `minMatchLengthBP` length (bp), relative maxima in
        the signal-template convolution must be greater in value than others to qualify as matches.
        If set to a value less than 1, the minimum length is determined via :func:`consenrich.matching.autoMinLengthIntervals`,
        a simple wrapper around :func:`consenrich.core.getContextSize`. If set to `None`, defaults to 250 bp.
    :type minMatchLengthBP: Optional[int]
    :param minSignalAtMaxima: Secondary significance threshold coupled with :math:`\alpha`. Requires the *signal value*
        at relative maxima in the response sequence to be greater than a threshold :math:`\pm \epsilon`.
    :type minSignalAtMaxima: Optional[str | float]
    :param useScalingFunction: If True, use (only) the scaling function to build the matching template.
        If False, use (only) the wavelet function.
    :type useScalingFunction: bool
    :param excludeRegionsBedFile: A BED file with regions to exclude from matching
    :type excludeRegionsBedFile: Optional[str]
    :param recenterAtPointSource: If True, recenter detected matches at the point source (max value)
    :type recenterAtPointSource: bool
    :param weights: Optional weights to apply to `values` prior to matching. Must have the same length as `values`.
    :type weights: Optional[npt.NDArray[np.float64]]
    :param eps: Tolerance parameter for relative maxima detection in the response sequence. Set to zero to enforce strict
        inequalities when identifying discrete relative maxima.
    :type eps: float
    :seealso: :class:`consenrich.core.matchingParams`, :func:`cconsenrich.csampleBlockStats`, :ref:`matching`
    :return: A pandas DataFrame with detected matches
    :rtype: pd.DataFrame
    """

    rng = np.random.default_rng(int(randSeed))
    if len(intervals) < 5:
        raise ValueError("`intervals` must be at least length 5")

    if len(values) != len(intervals):
        raise ValueError("`values` must have the same length as `intervals`")

    if len(templateNames) != len(cascadeLevels):
        raise ValueError(
            "\n\t`templateNames` and `cascadeLevels` must have the same length."
            "\n\tSet products are not supported, i.e., each template needs an explicitly defined cascade level."
            "\t\ne.g., for `templateNames = [haar, db2]`, use `cascadeLevels = [2, 2]`, not `[2]`.\n"
        )

    intervalLengthBp = intervals[1] - intervals[0]

    if minMatchLengthBP is not None and minMatchLengthBP < 1:
        minMatchLengthBP = autoMinLengthIntervals(
            values,
        ) * int(intervalLengthBp)
    elif minMatchLengthBP is None or minMatchLengthBP == 0:
        minMatchLengthBP = 250

    logger.info(f"\n\tUsing minMatchLengthBP: {minMatchLengthBP}")

    if not np.all(np.abs(np.diff(intervals)) == intervalLengthBp):
        raise ValueError("`intervals` must be evenly spaced.")

    chromStart = int(intervals[0])
    chromEnd = int(intervals[-1]) + int(intervalLengthBp)

    if weights is not None:
        if len(weights) != len(values):
            logger.warning(
                f"`weights` length {len(weights)} does not match `values` length {len(values)}. Ignoring..."
            )
        else:
            values = values * weights

    values_ = values.astype(np.float32)
    nz_values_ = values_  # NOTE: null/thresholds are now with respect to ALL values (including zeros/negatives)

    iters = max(int(iters), 1000)
    defQuantile = 0.50
    splitFolds = 4
    excludeMaskGlobal = np.zeros(len(intervals), dtype=np.uint8)
    if excludeRegionsBedFile is not None:
        excludeMaskGlobal = core.getBedMask(
            chromosome, excludeRegionsBedFile, intervals
        ).astype(np.uint8)
    allRows = []

    def parseMinSignalThreshold(val):
        if val is None:
            return -1e6
        if isinstance(val, str):
            if val.startswith("q:"):
                qVal = float(val.split("q:")[-1])
                if not (0 <= qVal <= 1):
                    raise ValueError(f"Quantile {qVal} is out of range")
                return float(
                    np.quantile(
                        nz_values_,
                        qVal,
                        method="interpolated_inverted_cdf",
                    )
                )
            elif castableToFloat(val):
                v = float(val)
                return -1e6 if v < 0 else v
            else:
                return float(
                    np.quantile(
                        nz_values_,
                        defQuantile,
                        method="interpolated_inverted_cdf",
                    )
                )
        if isinstance(val, (float, int)):
            v = float(val)
            return -1e6 if v < 0 else v
        return float(
            np.quantile(
                nz_values_,
                defQuantile,
                method="interpolated_inverted_cdf",
            )
        )

    def relativeMaxima(
        resp: np.ndarray, orderBins: int, eps: float = None
    ) -> np.ndarray:
        order_: int = max(int(orderBins), 1)
        if eps is None:
            eps = np.finfo(resp.dtype).eps * 10

        def ge_with_tol(a, b):
            return a > (b - eps)

        # get initial set using loosened criterion
        idx = signal.argrelextrema(resp, comparator=ge_with_tol, order=order_)[0]
        if idx.size == 0:
            return idx

        if eps > 0.0:
            groups = []
            start, prev = idx[0], idx[0]
            for x in idx[1:]:
                # case: still contiguous
                if x == prev + 1:
                    prev = x
                else:
                    # case: a gap --> break off from previous group
                    groups.append((start, prev))
                    start = x
                    prev = x
            groups.append((start, prev))

            centers: list[int] = []
            for s, e in groups:
                if s == e:
                    centers.append(s)
                else:
                    # for each `group` of tied indices, picks the center
                    centers.append((s + e) // 2)

            return np.asarray(centers, dtype=np.intp)

        return idx

    def sampleBlockMaxima(
        resp: np.ndarray,
        subsetMask: np.ndarray,
        relWindowBins: int,
        nsamp: int,
        seed: int,
        eps: float,
    ):
        exMask = excludeMaskGlobal.astype(np.uint8).copy()
        exMask |= (~subsetMask).astype(np.uint8)
        vals = np.array(
            cconsenrich.csampleBlockStats(
                intervals.astype(np.uint32),
                resp,
                int(relWindowBins),
                int(nsamp),
                int(seed),
                exMask.astype(np.uint8),
                np.float64(eps if eps is not None else 0.0),
            ),
            dtype=float,
        )
        finiteMask = np.isfinite(vals)
        vals = vals[finiteMask]
        if vals.size >= 100:
            upperCut = np.quantile(
                vals,
                0.995,
                method="interpolated_inverted_cdf",
            )
            vals = vals[vals <= upperCut]
        return vals

    def cauchyCombinePVals(
        pVals: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        p = np.asarray(pVals, dtype=np.float64)
        if p.size == 0:
            return 1.0
        if p.size == 1:
            return float(np.clip(p[0], 1.0e-12, 1.0))
        p = np.clip(p, 1.0e-12, 1.0 - 1.0e-12)
        if weights is None:
            w = np.full(p.size, 1.0 / float(p.size), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.size != p.size:
                raise ValueError("weights must match pVals length")
            w = np.clip(w, 0.0, np.inf)
            if float(np.sum(w)) <= 0.0:
                w = np.full(p.size, 1.0 / float(p.size), dtype=np.float64)
            else:
                w = w / float(np.sum(w))
        t = np.tan(np.pi * (0.5 - p))
        return float(
            np.clip(
                0.5 - (np.arctan(np.sum(w * t)) / np.pi),
                1.0e-12,
                1.0,
            )
        )

    def buildBlockedFoldAssignments(
        resp: np.ndarray,
        relWindowBins: int,
        seed: int,
        numFolds: int,
    ) -> np.ndarray:
        # block-level split, reduce leakage
        n = int(resp.size)
        if n < 10:
            return np.full(n, -1, dtype=np.int16)

        eligible = excludeMaskGlobal == 0
        blockLenBins = max(int(8 * relWindowBins), 128)
        blockStarts = np.arange(0, n, blockLenBins, dtype=np.intp)

        blocks = []
        blockMedian = []
        blockIQR = []
        minEligible = max(8, relWindowBins)
        for s in blockStarts:
            e = min(n, int(s + blockLenBins))
            localEligible = eligible[s:e]
            if int(np.count_nonzero(localEligible)) < int(minEligible):
                continue
            localVals = resp[s:e][localEligible]
            if localVals.size < 4:
                continue
            q25, q75 = np.quantile(localVals, [0.25, 0.75])
            blocks.append((int(s), int(e)))
            blockMedian.append(float(np.median(localVals)))
            blockIQR.append(float(q75 - q25))

        if len(blocks) < 4:
            return np.full(n, -1, dtype=np.int16)

        blockMedianArr = np.asarray(blockMedian, dtype=np.float64)
        blockIQRArr = np.asarray(blockIQR, dtype=np.float64)
        medCut = float(np.median(blockMedianArr))
        iqrCut = float(np.median(blockIQRArr))
        strata = 2 * (blockMedianArr > medCut).astype(np.int8) + (
            blockIQRArr > iqrCut
        ).astype(np.int8)

        foldAssign = np.full(n, -1, dtype=np.int16)
        rngSplit = np.random.default_rng(int(seed))
        numFolds = max(int(numFolds), 2)
        for st in range(4):
            stIdx = np.where(strata == st)[0]
            if stIdx.size == 0:
                continue
            perm = rngSplit.permutation(stIdx)
            for localFoldIndex, bIdx in enumerate(perm):
                s, e = blocks[int(bIdx)]
                foldAssign[s:e] = int(localFoldIndex % numFolds)

        # guard zones around split boundaries
        if relWindowBins > 0:
            boundaries = np.where(foldAssign[1:] != foldAssign[:-1])[0] + 1
            for b in boundaries:
                lo = max(0, int(b - relWindowBins))
                hi = min(n, int(b + relWindowBins + 1))
                foldAssign[lo:hi] = -1

        foldAssign[~eligible] = -1
        return foldAssign

    wavelet_set = set(pw.wavelist(kind="discrete"))
    for templateName, cascadeLevel in zip(templateNames, cascadeLevels):
        if templateName not in wavelet_set:
            logger.warning(f"Skipping unknown wavelet template: {templateName}")
            continue

        wav = pw.Wavelet(str(templateName))
        scalingFunc, waveletFunc, _ = wav.wavefun(level=int(cascadeLevel))
        template = np.array(
            scalingFunc if useScalingFunction else waveletFunc,
            dtype=np.float64,
        )
        template /= max(np.linalg.norm(template), np.finfo(np.float64).tiny)
        logger.info(
            f"\n\tMatching template: {templateName}"
            f"\n\tcascade level: {cascadeLevel}"
            f"\n\ttemplate length: {len(template)}"
        )

        # efficient FFT-based cross-correlation
        # (OA may be better for smaller templates, TODO add a check)
        response = signal.fftconvolve(values, template[::-1], mode="same")
        thisMinMatchBp = minMatchLengthBP
        if thisMinMatchBp is None or thisMinMatchBp < 1:
            thisMinMatchBp = len(template) * intervalLengthBp
        if thisMinMatchBp % intervalLengthBp != 0:
            thisMinMatchBp += intervalLengthBp - (thisMinMatchBp % intervalLengthBp)
        relWindowBins = int(((thisMinMatchBp / intervalLengthBp) / 2) + 1)
        relWindowBins = max(relWindowBins, 1)
        natThreshold = parseMinSignalThreshold(minSignalAtMaxima)
        candidateIdx = relativeMaxima(response, relWindowBins, eps=eps)
        candidateMask = (
            (candidateIdx >= relWindowBins)
            & (candidateIdx < len(response) - relWindowBins)
            & (excludeMaskGlobal[candidateIdx] == 0)
            & (values_[candidateIdx] >= natThreshold)
        )
        candidateIdx = candidateIdx[candidateMask]
        if len(candidateIdx) == 0:
            continue
        if maxNumMatches is not None and len(candidateIdx) > maxNumMatches:
            candidateIdx = candidateIdx[
                np.argsort(values_[candidateIdx])[-maxNumMatches:]
            ]

        foldAssign = buildBlockedFoldAssignments(
            response,
            relWindowBins,
            seed=int(rng.integers(1, 10_000_000)),
            numFolds=int(splitFolds),
        )
        pooledNullMask = foldAssign >= 0
        if int(np.count_nonzero(pooledNullMask)) < max(8, relWindowBins):
            pooledNullMask = excludeMaskGlobal == 0

        pooledBlockMaxima = sampleBlockMaxima(
            response,
            pooledNullMask,
            relWindowBins,
            nsamp=max(iters, 1000),
            seed=int(rng.integers(1, 10_000_000)),
            eps=eps,
        )
        if len(pooledBlockMaxima) < 25:
            continue

        pooledEcdfSf = stats.ecdf(pooledBlockMaxima).sf
        pooledP = np.clip(
            pooledEcdfSf.evaluate(response[candidateIdx]),
            1.0 / (float(len(pooledBlockMaxima)) + 1.0),
            1.0,
        )

        pByCandidate: dict[int, list[float]] = {int(i): [] for i in candidateIdx}
        pooledPByCandidate: dict[int, float] = {
            int(i): float(pVal) for i, pVal in zip(candidateIdx, pooledP)
        }
        for foldIndex in np.unique(foldAssign[foldAssign >= 0]):
            testMask = foldAssign == int(foldIndex)
            nullMask = (foldAssign >= 0) & (~testMask)
            if (not np.any(nullMask)) or (not np.any(testMask)):
                continue
            splitCandidateMask = testMask[candidateIdx]
            if not np.any(splitCandidateMask):
                continue

            blockMaxima = sampleBlockMaxima(
                response,
                nullMask,
                relWindowBins,
                nsamp=max(iters, 1000),
                seed=int(rng.integers(1, 10_000_000)),
                eps=eps,
            )
            if len(blockMaxima) < 25:
                continue

            ecdfSf = stats.ecdf(blockMaxima).sf
            splitCandidateIdx = candidateIdx[splitCandidateMask]
            pEmp = np.clip(
                ecdfSf.evaluate(response[splitCandidateIdx]),
                1.0 / (float(len(blockMaxima)) + 1.0),
                1.0,
            )
            for idxVal, pVal in zip(splitCandidateIdx, pEmp):
                pByCandidate[int(idxVal)].append(float(pVal))

        keepCandidateIdx = candidateIdx.astype(np.int64, copy=False)

        startsIdx = np.maximum(keepCandidateIdx - relWindowBins, 0)
        endsIdx = np.minimum(len(values) - 1, keepCandidateIdx + relWindowBins)
        pointSourcesIdx = []
        for s, e in zip(startsIdx, endsIdx):
            pointSourcesIdx.append(np.argmax(values[s : e + 1]) + s)
        pointSourcesIdx = np.array(pointSourcesIdx, dtype=np.int64)
        starts = intervals[startsIdx]
        ends = intervals[endsIdx]
        pointSourcesAbs = (intervals[pointSourcesIdx]) + max(1, intervalLengthBp // 2)
        if recenterAtPointSource:
            starts = pointSourcesAbs - (relWindowBins * intervalLengthBp)
            ends = pointSourcesAbs + (relWindowBins * intervalLengthBp)

        starts = np.maximum(starts.astype(np.int64, copy=False), chromStart)
        ends = np.minimum(ends.astype(np.int64, copy=False), chromEnd)

        pointSourcesRel = (
            intervals[pointSourcesIdx].astype(np.int64, copy=False) - starts
        ) + max(1, intervalLengthBp // 2)
        sqScores = (1 + response[keepCandidateIdx]) ** 2
        minR, maxR = float(np.min(sqScores)), float(np.max(sqScores))
        rangeR = max(maxR - minR, 1.0)
        scores = (250 + 750 * (sqScores - minR) / rangeR).astype(int)
        for i, idxVal in enumerate(keepCandidateIdx):
            allRows.append(
                {
                    "chromosome": chromosome,
                    "start": int(starts[i]),
                    "end": int(ends[i]),
                    "name": f"{templateName}_{cascadeLevel}_{idxVal}_S",
                        "score": int(scores[i]),
                        "strand": ".",
                        "signal": float(values[idxVal]),
                        "p_raw": cauchyCombinePVals(
                            np.asarray(
                                [pooledPByCandidate[int(idxVal)]]
                                + pByCandidate[int(idxVal)],
                                dtype=np.float64,
                            ),
                            weights=(
                                np.asarray([1.0], dtype=np.float64)
                                if len(pByCandidate[int(idxVal)]) == 0
                                else np.asarray(
                                    [0.25]
                                    + [
                                        0.75 / float(len(pByCandidate[int(idxVal)]))
                                        for _ in pByCandidate[int(idxVal)]
                                    ],
                                    dtype=np.float64,
                                )
                            ),
                        ),
                        "pointSource": int(pointSourcesRel[i]),
                        "templateName": str(templateName),
                        "cascadeLevel": int(cascadeLevel),
                    "tag": "S",
                }
            )

    if not allRows:
        logger.warning("No matches detected, returning empty DataFrame.")

        return pd.DataFrame(
            columns=[
                "chromosome",
                "start",
                "end",
                "name",
                "score",
                "strand",
                "signal",
                "pValue",
                "qValue",
                "pointSource",
            ]
        )

    df = pd.DataFrame(allRows)

    groupCols = ["chromosome", "templateName", "cascadeLevel"]
    qVals = np.empty(len(df), dtype=float)
    for _, groupIdx in df.groupby(groupCols, sort=False).groups.items():
        p = df.loc[groupIdx, "p_raw"].values.astype(float, copy=False)
        if methodFDR is None:
            qVals[groupIdx] = p
        elif isinstance(methodFDR, str):
            logger.info(
                f"Applying FDR correction method: {methodFDR} wrt {groupCols}: {len(p)} tests"
            )
            qVals[groupIdx] = _FDR(p, method=methodFDR)

    df["pValue"] = -np.log10(
        np.clip(df["p_raw"].values.astype(float), np.finfo(np.float32).tiny, 1.0)
    )
    df["qValue"] = -np.log10(np.clip(qVals, np.finfo(np.float32).tiny, 1.0))
    df.drop(columns=["p_raw"], inplace=True)
    df = df[qVals <= alpha].copy()

    df["chromosome"] = df["chromosome"].astype(str)
    df.sort_values(by=["chromosome", "start", "end"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    keepCols = [
        "chromosome",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signal",
        "pValue",
        "qValue",
        "pointSource",
    ]
    df = df[keepCols]
    return df, minMatchLengthBP


def runMatchingAlgorithm(
    bedGraphFile: str,
    templateNames: List[str],
    cascadeLevels: List[int],
    iters: int,
    alpha: float = 0.05,
    minMatchLengthBP: Optional[int] = 250,
    maxNumMatches: Optional[int] = 100_000,
    minSignalAtMaxima: Optional[float | str] = 0.1,
    randSeed: int = 42,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    weightsBedGraph: str | None = None,
    eps: float = 1.0e-3,
    mergeGapBP: int | None = -1,
    merge: bool = True,
    massQuantileCutoff: float = -1.0,
    methodFDR: str | None = None,
):
    r"""Wraps :func:`matchWavelet` for genome-wide matching given a bedGraph file"""
    cols = ["chromosome", "start", "end", "value"]
    bedGraphDF = pd.read_csv(
        bedGraphFile,
        sep="\t",
        header=None,
        names=cols,
        dtype={
            "chromosome": str,
            "start": np.uint32,
            "end": np.uint32,
            "value": np.float64,
        },
    )
    bedGraphDF.sort_values(
        by=["chromosome", "start", "end"],
        kind="mergesort",
        inplace=True,
    )
    bedGraphDF.reset_index(drop=True, inplace=True)
    chromosomes = bedGraphDF["chromosome"].unique().tolist()

    weightsDF = None
    if weightsBedGraph is not None and os.path.exists(weightsBedGraph):
        try:
            weightsDF = pd.read_csv(
                weightsBedGraph,
                sep="\t",
                header=None,
                names=cols,
                dtype={
                    "chromosome": str,
                    "start": np.uint32,
                    "end": np.uint32,
                    "value": np.float64,
                },
            )
        except Exception as ex:
            logger.warning(
                f"Failed to parse weights from {weightsBedGraph}. Ignoring weights....\n{ex}"
            )
            weightsDF = None

    minMatchLengths = []
    gapByChrom: dict[str, int] = {}
    tmpFiles = []
    for c_, chromosome_ in enumerate(chromosomes):
        chromBedGraphDF = bedGraphDF[bedGraphDF["chromosome"] == chromosome_]
        chromIntervals = chromBedGraphDF["start"].to_numpy()
        chromValues = chromBedGraphDF["value"].to_numpy()

        weights = np.ones_like(chromValues, dtype=np.float64)
        if weightsDF is not None:
            try:
                wChr = weightsDF[weightsDF["chromosome"] == chromosome_]
                if len(wChr) == len(chromValues):
                    weights = 1 / np.sqrt(wChr["value"].to_numpy() + 1.0)
                else:
                    logger.warning(
                        f"Weights length {len(wChr)} does not match values length {len(chromValues)} on {chromosome_}. Ignoring weights for this chrom...."
                    )
            except Exception as ex:
                logger.warning(
                    f"Failed to parse weights from {weightsBedGraph}. Ignoring weights....\n{ex}"
                )

        if minMatchLengthBP is not None and minMatchLengthBP < 1:
            minMatchLengthBP_ = autoMinLengthIntervals(
                chromValues,
            ) * int(chromIntervals[1] - chromIntervals[0])
        else:
            minMatchLengthBP_ = minMatchLengthBP

        df__, minMatchLengthBP__ = matchWavelet(
            chromosome_,
            chromIntervals,
            chromValues,
            templateNames,
            cascadeLevels,
            iters,
            alpha,
            minMatchLengthBP_,
            maxNumMatches,
            minSignalAtMaxima,
            randSeed,
            recenterAtPointSource,
            useScalingFunction,
            excludeRegionsBedFile,
            weights,
            eps,
            methodFDR,
        )

        if df__.empty:
            logger.info(f"No matches detected on {chromosome_}.")
            continue
        gapByChrom[chromosome_] = int(minMatchLengthBP__)
        stepSize_ = np.float32(chromIntervals[1] - chromIntervals[0])
        lengths = (
            df__["end"].to_numpy(dtype=np.int64)
            - df__["start"].to_numpy(dtype=np.int64)
        ).astype(np.float32)

        signals = df__["signal"].to_numpy(dtype=np.float32)

        massProxy = ((lengths * signals) / stepSize_).astype(np.float32)
        massQuantileCutoff_ = min(massQuantileCutoff, 0.995)
        if massQuantileCutoff_ > 0 and massProxy.size > 0:
            cutoff = np.quantile(
                massProxy,
                float(massQuantileCutoff_),
                method="interpolated_inverted_cdf",
            )

            logger.info(
                f"Applying mass cutoff: {cutoff:.3f} on chromosome {chromosome_}"
            )
            df__ = df__[massProxy >= cutoff].copy()
        else:
            df__ = df__.copy()

        df__.to_csv(
            f"{bedGraphFile}_{chromosome_}_matches.narrowPeak".replace(".bedGraph", ""),
            sep="\t",
            index=False,
            header=False,
        )
        tmpFiles.append(
            f"{bedGraphFile}_{chromosome_}_matches.narrowPeak".replace(".bedGraph", "")
        )
        minMatchLengths.append(minMatchLengthBP__)
    if merge and len(tmpFiles) > 0:
        mergeNarrowPeaks.mergeAndSortNarrowPeaks(
            tmpFiles,
            outPath=f"{bedGraphFile}_matches.mergedMatches.narrowPeak".replace(
                ".bedGraph", ""
            ),
            defaultGapBP=(
                mergeGapBP if mergeGapBP is not None and mergeGapBP > 0 else 250
            ),
            gapByChrom=gapByChrom,
        )

        for tf in tmpFiles:
            try:
                os.remove(tf)
            except Exception:
                logger.warning(f"Could not remove temporary file: {tf}")
                pass
    return f"{bedGraphFile}_matches.mergedMatches.narrowPeak".replace(".bedGraph", "")
