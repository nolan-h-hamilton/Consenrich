# -*- coding: utf-8 -*-
r"""Module implementing (experimental) 'structured peak detection' features using wavelet-based templates."""

import logging
import os
import math
from pybedtools import BedTool
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
    # can use bh or the more conservative Benjamini-Yekutieli to
    # ... control FDR under arbitrary dependencies between tests
    if method is None:
        return pVals
    return stats.false_discovery_control(pVals, method=method.lower())


def autoMinLengthIntervals(
    values: np.ndarray,
    initLen: int = 5,
    maxLen: int = 25,
    cutoffQuantile: float|None = None,
) -> int:
    hlen = core.getContextSize(
        values,
    )[1]
    return min(max(int(hlen), initLen), maxLen)


def scalarClip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


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
    minSignalAtMaxima: Optional[float | str] = 0.01,
    randSeed: int = 42,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    weights: Optional[npt.NDArray[np.float64]] = None,
    eps: float = 1.0e-3,
) -> pd.DataFrame:
    r"""Detect structured peaks in Consenrich tracks by matching wavelet- or scaling-function–based templates.

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
        If set to a value less than 1, the minimum length is determined via :func:`consenrich.matching.autoMinLengthIntervals`.
        If set to `None`, defaults to 250 bp.
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
    chromMin = int(intervals[0])
    chromMax = int(intervals[-1])
    chromMid = chromMin + (chromMax - chromMin) // 2  # for split
    halfLeftMask = intervals < chromMid
    halfRightMask = ~halfLeftMask
    excludeMaskGlobal = np.zeros(len(intervals), dtype=np.uint8)
    if excludeRegionsBedFile is not None:
        excludeMaskGlobal = core.getBedMask(chromosome, excludeRegionsBedFile, intervals).astype(np.uint8)
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

    def relativeMaxima(resp: np.ndarray, orderBins: int, eps: float = None) -> np.ndarray:
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
        halfMask: np.ndarray,
        relWindowBins: int,
        nsamp: int,
        seed: int,
        eps: float,
    ):
        exMask = excludeMaskGlobal.astype(np.uint8).copy()
        exMask |= (~halfMask).astype(np.uint8)
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
        if len(vals) == 0:
            return vals
        low = np.quantile(vals, 0.0001)
        high = np.quantile(vals, 0.9999)
        return vals[(vals > low) & (vals < high)]

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
        for nullMask, testMask, tag in [
            (halfLeftMask, halfRightMask, "R"),
            (halfRightMask, halfLeftMask, "L"),
        ]:
            blockMaxima = sampleBlockMaxima(
                response,
                nullMask,
                relWindowBins,
                nsamp=max(iters, 1000),
                seed=rng.integers(1, 10_000),
                eps=eps,
            )
            if len(blockMaxima) < 25:
                pooledMask = ~excludeMaskGlobal.astype(bool)
                blockMaxima = sampleBlockMaxima(
                    response,
                    pooledMask,
                    relWindowBins,
                    nsamp=max(iters, 1000),
                    seed=rng.integers(1, 10_000),
                    eps=eps,
                )

            ecdfSf = stats.ecdf(blockMaxima).sf
            candidateIdx = relativeMaxima(response, relWindowBins, eps=eps)

            candidateMask = (
                (candidateIdx >= relWindowBins)
                & (candidateIdx < len(response) - relWindowBins)
                & (testMask[candidateIdx])
                & (excludeMaskGlobal[candidateIdx] == 0)
                & (values_[candidateIdx] >= natThreshold)
            )

            candidateIdx = candidateIdx[candidateMask]
            if len(candidateIdx) == 0:
                continue
            if maxNumMatches is not None and len(candidateIdx) > maxNumMatches:
                candidateIdx = candidateIdx[np.argsort(values_[candidateIdx])[-maxNumMatches:]]
            pEmp = np.clip(
                ecdfSf.evaluate(response[candidateIdx]),
                np.finfo(np.float32).tiny,
                1.0,
            )
            startsIdx = np.maximum(candidateIdx - relWindowBins, 0)
            endsIdx = np.minimum(len(values) - 1, candidateIdx + relWindowBins)
            pointSourcesIdx = []
            for s, e in zip(startsIdx, endsIdx):
                pointSourcesIdx.append(np.argmax(values[s : e + 1]) + s)
            pointSourcesIdx = np.array(pointSourcesIdx)
            starts = intervals[startsIdx]
            ends = intervals[endsIdx]
            pointSourcesAbs = (intervals[pointSourcesIdx]) + max(1, intervalLengthBp // 2)
            if recenterAtPointSource:
                starts = pointSourcesAbs - (relWindowBins * intervalLengthBp)
                ends = pointSourcesAbs + (relWindowBins * intervalLengthBp)

            starts = np.maximum(starts.astype(np.int64, copy=False), chromStart)
            ends = np.minimum(ends.astype(np.int64, copy=False), chromEnd)

            pointSourcesRel = (intervals[pointSourcesIdx].astype(np.int64, copy=False) - starts) + max(
                1, intervalLengthBp // 2
            )
            sqScores = (1 + response[candidateIdx]) ** 2
            minR, maxR = (
                float(np.min(sqScores)),
                float(np.max(sqScores)),
            )
            rangeR = max(maxR - minR, 1.0)
            scores = (250 + 750 * (sqScores - minR) / rangeR).astype(int)
            for i, idxVal in enumerate(candidateIdx):
                allRows.append(
                    {
                        "chromosome": chromosome,
                        "start": int(starts[i]),
                        "end": int(ends[i]),
                        "name": f"{templateName}_{cascadeLevel}_{idxVal}_{tag}",
                        "score": int(scores[i]),
                        "strand": ".",
                        "signal": float(values[idxVal]),
                        "p_raw": float(pEmp[i]),
                        "pointSource": int(pointSourcesRel[i]),
                        "templateName": str(templateName),
                        "cascadeLevel": int(cascadeLevel),
                        "tag": str(tag),
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
        # FDR is wrt chromosome and the wavelet/scaling function template
        p = df.loc[groupIdx, "p_raw"].values.astype(float, copy=False)
        qVals[groupIdx] = p

    df["pValue"] = -np.log10(np.clip(df["p_raw"].values.astype(float), np.finfo(np.float32).tiny, 1.0))
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
    minSignalAtMaxima: Optional[float | str] = 0.01,
    randSeed: int = 42,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    weightsBedGraph: str | None = None,
    eps: float = 1.0e-3,
    mergeGapBP: int | None = -1,
    methodFDR: str | None = None,
    merge: bool = True,
    massQuantileCutoff: float = -1.0,
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
            logger.warning(f"Failed to parse weights from {weightsBedGraph}. Ignoring weights....\n{ex}")
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
                logger.warning(f"Failed to parse weights from {weightsBedGraph}. Ignoring weights....\n{ex}")

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
        )

        if df__.empty:
            logger.info(f"No matches detected on {chromosome_}.")
            continue
        gapByChrom[chromosome_] = int(minMatchLengthBP__)
        stepSize_ = np.float32(chromIntervals[1] - chromIntervals[0])
        lengths = (df__["end"].to_numpy(dtype=np.int64) - df__["start"].to_numpy(dtype=np.int64)).astype(
            np.float32
        )

        signals = df__["signal"].to_numpy(dtype=np.float32)

        massProxy = ((lengths * signals) / stepSize_).astype(np.float32)
        massQuantileCutoff_ = min(massQuantileCutoff, 0.995)
        if massQuantileCutoff_ > 0 and massProxy.size > 0:
            cutoff = np.quantile(
                massProxy,
                float(massQuantileCutoff_),
                method="interpolated_inverted_cdf",
            )

            logger.info(f"Applying mass cutoff: {cutoff:.3f} on chromosome {chromosome_}")
            df__ = df__[massProxy >= cutoff].copy()
        else:
            df__ = df__.copy()

        df__.to_csv(
            f"{bedGraphFile}_{chromosome_}_matches.narrowPeak".replace(".bedGraph", ""),
            sep="\t",
            index=False,
            header=False,
        )
        tmpFiles.append(f"{bedGraphFile}_{chromosome_}_matches.narrowPeak".replace(".bedGraph", ""))
        minMatchLengths.append(minMatchLengthBP__)


    mergeNarrowPeaks.mergeAndSortNarrowPeaks(
        tmpFiles,
        outPath=f"{bedGraphFile}_matches.mergedMatches.narrowPeak".replace(".bedGraph", ""),
        defaultGapBP=250,
        gapByChrom=gapByChrom,
    )

    for tf in tmpFiles:
        try:
            os.remove(tf)
        except Exception:
            logger.warning(f'Could not remove temporary file: {tf}')
            pass
    return f"{bedGraphFile}_matches.mergedMatches.narrowPeak".replace(".bedGraph", "")