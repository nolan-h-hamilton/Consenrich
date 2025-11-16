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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def autoMinLengthIntervals(
    values: np.ndarray, initLen: int = 3
) -> int:
    r"""Determines a minimum matching length (in interval units) based on the input signal values.

    Returns the mean length of non-zero contiguous segments in a log-scaled/centered version of `values`

    :param values: A 1D array of signal-like values.
    :type values: np.ndarray
    :param initLen: Initial minimum length (in intervals). Defaults to 3.
    :type initLen: int
    :return: Estimated minimum matching length (in intervals)
    :rtype: int

    """
    trValues = np.asinh(values) - signal.medfilt(
        np.asinh(values),
        kernel_size=max(
            (2 * initLen) + 1,
            2 * (int(len(values) * 0.005)) + 1,
        ),
    )
    nz = trValues[trValues > 0]
    if len(nz) == 0:
        return initLen
    thr = np.quantile(nz, 0.90, method="interpolated_inverted_cdf")
    mask = nz >= thr
    if not np.any(mask):
        return initLen
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    runs = idx.reshape(-1, 2)
    widths = runs[:, 1] - runs[:, 0]
    widths = widths[widths >= initLen]
    if len(widths) == 0:
        return initLen
    return int(np.mean(widths))


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


def matchExistingBedGraph(
    bedGraphFile: str,
    templateName: str,
    cascadeLevel: int,
    alpha: float = 0.05,
    minMatchLengthBP: Optional[int] = 250,
    iters: int = 25_000,
    minSignalAtMaxima: Optional[float | str] = "q:0.75",
    maxNumMatches: Optional[int] = 100_000,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    mergeGapBP: Optional[int] = None,
    merge: bool = True,
    weights: Optional[npt.NDArray[np.float64]] = None,
    randSeed: int = 42,
) -> Optional[str]:
    r"""Match discrete templates in a bedGraph file of Consenrich estimates

    This function is a simple wrapper. See :func:`consenrich.matching.matchWavelet` for details on parameters.

    :param bedGraphFile: A bedGraph file with 'consensus' signal estimates derived from multiple samples, e.g., from Consenrich. The suffix '.bedGraph' is required.
    :type bedGraphFile: str

    :seealso: :func:`consenrich.matching.matchWavelet`, :class:`consenrich.core.matchingParams`, :ref:`matching`
    """
    if not os.path.isfile(bedGraphFile):
        raise FileNotFoundError(f"Couldn't access {bedGraphFile}")
    if not bedGraphFile.endswith(".bedGraph"):
        raise ValueError(
            f"Please use a suffix '.bedGraph' for `bedGraphFile`, got: {bedGraphFile}"
        )

    if mergeGapBP is None:
        mergeGapBP = (
            (minMatchLengthBP // 2) + 1
            if minMatchLengthBP is not None
            else 75
        )

    allowedTemplates = [
        x for x in pw.wavelist(kind="discrete") if "bio" not in x
    ]
    if templateName not in allowedTemplates:
        raise ValueError(
            f"Unknown wavelet template: {templateName}\nAvailable templates: {allowedTemplates}"
        )

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

    outPaths: List[str] = []
    outPathsMerged: List[str] = []
    outPathAll: Optional[str] = None
    outPathMergedAll: Optional[str] = None

    for chrom_ in sorted(bedGraphDF["chromosome"].unique()):
        df_ = bedGraphDF[bedGraphDF["chromosome"] == chrom_]
        if len(df_) < 5:
            logger.info(f"Skipping {chrom_}: less than 5 intervals.")
            continue

        try:
            df__ = matchWavelet(
                chrom_,
                df_["start"].to_numpy(),
                df_["value"].to_numpy(),
                [templateName],
                [cascadeLevel],
                iters,
                alpha,
                minMatchLengthBP,
                maxNumMatches,
                recenterAtPointSource=recenterAtPointSource,
                useScalingFunction=useScalingFunction,
                excludeRegionsBedFile=excludeRegionsBedFile,
                weights=weights,
                minSignalAtMaxima=minSignalAtMaxima,
                randSeed=randSeed,
            )
        except Exception as ex:
            logger.info(
                f"Skipping {chrom_} due to error in matchWavelet: {ex}"
            )
            continue

        if df__.empty:
            logger.info(f"No matches detected on {chrom_}.")
            continue

        perChromOut = bedGraphFile.replace(
            ".bedGraph",
            f".{chrom_}.matched.{templateName}_lvl{cascadeLevel}.narrowPeak",
        )
        df__.to_csv(perChromOut, sep="\t", index=False, header=False)
        logger.info(f"Matches written to {perChromOut}")
        outPaths.append(perChromOut)

        if merge:
            mergedPath = mergeMatches(
                perChromOut, mergeGapBP=mergeGapBP
            )
            if mergedPath is not None:
                logger.info(f"Merged matches written to {mergedPath}")
                outPathsMerged.append(mergedPath)

    if len(outPaths) == 0 and len(outPathsMerged) == 0:
        raise ValueError("No matches were detected.")

    if len(outPaths) > 0:
        outPathAll = (
            f"{bedGraphFile.replace('.bedGraph', '')}"
            f".allChroms.matched.{templateName}_lvl{cascadeLevel}.narrowPeak"
        )
        with open(outPathAll, "w") as outF:
            for path_ in outPaths:
                if os.path.isfile(path_):
                    with open(path_, "r") as inF:
                        for line in inF:
                            outF.write(line)
        logger.info(f"All unmerged matches written to {outPathAll}")

    if merge and len(outPathsMerged) > 0:
        outPathMergedAll = (
            f"{bedGraphFile.replace('.bedGraph', '')}"
            f".allChroms.matched.{templateName}_lvl{cascadeLevel}.mergedMatches.narrowPeak"
        )
        with open(outPathMergedAll, "w") as outF:
            for path in outPathsMerged:
                if os.path.isfile(path):
                    with open(path, "r") as inF:
                        for line in inF:
                            outF.write(line)
        logger.info(
            f"All merged matches written to {outPathMergedAll}"
        )

    for path_ in outPaths + outPathsMerged:
        try:
            if os.path.isfile(path_):
                os.remove(path_)
        except Exception:
            pass

    if merge and outPathMergedAll:
        return outPathMergedAll
    if outPathAll:
        return outPathAll
    logger.warning("No matches were detected...returning `None`")
    return None


def matchWavelet(
    chromosome: str,
    intervals: npt.NDArray[int],
    values: npt.NDArray[np.float64],
    templateNames: List[str],
    cascadeLevels: List[int],
    iters: int,
    alpha: float = 0.05,
    minMatchLengthBP: Optional[int] = 250,
    maxNumMatches: Optional[int] = 100_000,
    minSignalAtMaxima: Optional[float | str] = "q:0.75",
    randSeed: int = 42,
    recenterAtPointSource: bool = True,
    useScalingFunction: bool = True,
    excludeRegionsBedFile: Optional[str] = None,
    weights: Optional[npt.NDArray[np.float64]] = None,
    eps: float = 1.0e-2,
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
        an empirical null to test significance. See :func:`cconsenrich.csampleBlockStats`.
    :type iters: int
    :param alpha: Primary significance threshold on detected matches. Specifically, the
        minimum corr. empirical p-value approximated from randomly sampled blocks in the
        response sequence.
    :type alpha: float
    :param minMatchLengthBP: Within a window of `minMatchLengthBP` length (bp), relative maxima in
        the signal-template convolution must be greater in value than others to qualify as matches.
        If set to a value less than 1, the minimum length is determined via :func:`consenrich.matching.autoMinLengthIntervals`.
        If set to `None`, defaults to 250 bp.
    :type minMatchLengthBP: Optional[int]
    :param minSignalAtMaxima: Secondary significance threshold coupled with `alpha`. Requires the *signal value*
        at relative maxima in the response sequence to be greater than this threshold. Comparisons are made in log-scale
        to temper genome-wide dynamic range. If a `float` value is provided, the minimum signal value must be greater
        than this (absolute) value. *Set to a negative value to disable the threshold*.
        If a `str` value is provided, looks for 'q:quantileValue', e.g., 'q:0.90'. The
        threshold is then set to the corresponding quantile of the non-zero signal estimates.
        Defaults to str value 'q:0.75' --- the 75th percentile of signal values.
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
        raise ValueError(
            "`values` must have the same length as `intervals`"
        )

    if len(templateNames) != len(cascadeLevels):
        raise ValueError(
            "\n\t`templateNames` and `cascadeLevels` must have the same length."
            "\n\tSet products are not supported, i.e., each template needs an explicitly defined cascade level."
            "\t\ne.g., for `templateNames = [haar, db2]`, use `cascadeLevels = [2, 2]`, not `[2]`.\n"
        )

    intervalLengthBp = intervals[1] - intervals[0]

    if minMatchLengthBP is not None and minMatchLengthBP < 1:
        minMatchLengthBP = autoMinLengthIntervals(values) * int(
            intervalLengthBp
        )
    elif minMatchLengthBP is None:
        minMatchLengthBP = 250

    logger.info(f"\n\tUsing minMatchLengthBP: {minMatchLengthBP}")

    if not np.all(np.abs(np.diff(intervals)) == intervalLengthBp):
        raise ValueError("`intervals` must be evenly spaced.")

    if weights is not None:
        if len(weights) != len(values):
            logger.warning(
                f"`weights` length {len(weights)} does not match `values` length {len(values)}. Ignoring..."
            )
        else:
            values = values * weights

    asinhValues = np.asinh(values, dtype=np.float32)
    asinhNonZeroValues = asinhValues[asinhValues > 0]

    iters = max(int(iters), 1000)
    defQuantile = 0.75
    chromMin = int(intervals[0])
    chromMax = int(intervals[-1])
    chromMid = chromMin + (chromMax - chromMin) // 2  # for split
    halfLeftMask = intervals < chromMid
    halfRightMask = ~halfLeftMask
    excludeMaskGlobal = np.zeros(len(intervals), dtype=np.uint8)
    if excludeRegionsBedFile is not None:
        excludeMaskGlobal = core.getBedMask(
            chromosome, excludeRegionsBedFile, intervals
        ).astype(np.uint8)
    allRows = []

    def bhFdr(p: np.ndarray) -> np.ndarray:
        m = len(p)
        order = np.argsort(p, kind="mergesort")
        ranked = np.arange(1, m + 1, dtype=float)
        q = (p[order] * m) / ranked
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q)
        out[order] = q
        return np.clip(out, 0.0, 1.0)

    def parseMinSignalThreshold(val):
        if val is None:
            return -1e6
        if isinstance(val, str):
            if val.startswith("q:"):
                qVal = float(val.split("q:")[-1])
                if not (0 <= qVal <= 1):
                    raise ValueError(
                        f"Quantile {qVal} is out of range"
                    )
                return float(
                    np.quantile(
                        asinhNonZeroValues,
                        qVal,
                        method="interpolated_inverted_cdf",
                    )
                )
            elif castableToFloat(val):
                v = float(val)
                return -1e6 if v < 0 else float(np.asinh(v))
            else:
                return float(
                    np.quantile(
                        asinhNonZeroValues,
                        defQuantile,
                        method="interpolated_inverted_cdf",
                    )
                )
        if isinstance(val, (float, int)):
            v = float(val)
            return -1e6 if v < 0 else float(np.asinh(v))
        return float(
            np.quantile(
                asinhNonZeroValues,
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
        idx = signal.argrelextrema(
            resp, comparator=ge_with_tol, order=order_
        )[0]
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
        low = np.quantile(vals, 0.001)
        high = np.quantile(vals, 0.999)
        return vals[(vals > low) & (vals < high)]

    for templateName, cascadeLevel in zip(
        templateNames, cascadeLevels
    ):
        if templateName not in pw.wavelist(kind="discrete"):
            logger.warning(
                f"Skipping unknown wavelet template: {templateName}"
            )
            continue

        wav = pw.Wavelet(str(templateName))
        scalingFunc, waveletFunc, _ = wav.wavefun(
            level=int(cascadeLevel)
        )
        template = np.array(
            scalingFunc if useScalingFunction else waveletFunc,
            dtype=np.float64,
        )
        template /= np.linalg.norm(template)

        logger.info(
            f"\n\tMatching template: {templateName}"
            f"\n\tcascade level: {cascadeLevel}"
            f"\n\ttemplate length: {len(template)}"
        )

        # efficient FFT-based cross-correlation
        # (OA may be better for smaller templates, TODO add a check)
        response = signal.fftconvolve(
            values, template[::-1], mode="same"
        )
        thisMinMatchBp = minMatchLengthBP
        if thisMinMatchBp is None or thisMinMatchBp < 1:
            thisMinMatchBp = len(template) * intervalLengthBp
        if thisMinMatchBp % intervalLengthBp != 0:
            thisMinMatchBp += intervalLengthBp - (
                thisMinMatchBp % intervalLengthBp
            )
        relWindowBins = int(
            ((thisMinMatchBp / intervalLengthBp) / 2) + 1
        )
        relWindowBins = max(relWindowBins, 1)
        asinhThreshold = parseMinSignalThreshold(minSignalAtMaxima)
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
            candidateIdx = relativeMaxima(
                response, relWindowBins, eps=eps
            )

            candidateMask = (
                (candidateIdx >= relWindowBins)
                & (candidateIdx < len(response) - relWindowBins)
                & (testMask[candidateIdx])
                & (excludeMaskGlobal[candidateIdx] == 0)
                & (asinhValues[candidateIdx] > asinhThreshold)
            )

            candidateIdx = candidateIdx[candidateMask]
            if len(candidateIdx) == 0:
                continue
            if (
                maxNumMatches is not None
                and len(candidateIdx) > maxNumMatches
            ):
                candidateIdx = candidateIdx[
                    np.argsort(asinhValues[candidateIdx])[
                        -maxNumMatches:
                    ]
                ]
            pEmp = np.clip(
                ecdfSf.evaluate(response[candidateIdx]),
                1.0e-10,
                1.0,
            )
            startsIdx = np.maximum(candidateIdx - relWindowBins, 0)
            endsIdx = np.minimum(
                len(values) - 1, candidateIdx + relWindowBins
            )
            pointSourcesIdx = []
            for s, e in zip(startsIdx, endsIdx):
                pointSourcesIdx.append(
                    np.argmax(values[s : e + 1]) + s
                )
            pointSourcesIdx = np.array(pointSourcesIdx)
            starts = intervals[startsIdx]
            ends = intervals[endsIdx]
            pointSourcesAbs = (intervals[pointSourcesIdx]) + max(
                1, intervalLengthBp // 2
            )
            if recenterAtPointSource:
                starts = pointSourcesAbs - (
                    relWindowBins * intervalLengthBp
                )
                ends = pointSourcesAbs + (
                    relWindowBins * intervalLengthBp
                )
            pointSourcesRel = (
                intervals[pointSourcesIdx] - starts
            ) + max(1, intervalLengthBp // 2)
            sqScores = (1 + response[candidateIdx]) ** 2
            minR, maxR = (
                float(np.min(sqScores)),
                float(np.max(sqScores)),
            )
            rangeR = max(maxR - minR, 1.0)
            scores = (250 + 750 * (sqScores - minR) / rangeR).astype(
                int
            )
            for i, idxVal in enumerate(candidateIdx):
                allRows.append(
                    {
                        "chromosome": chromosome,
                        "start": int(starts[i]),
                        "end": int(ends[i]),
                        "name": f"{templateName}_{cascadeLevel}_{idxVal}_{tag}",
                        "score": int(scores[i]),
                        "strand": ".",
                        "signal": float(response[idxVal]),
                        "p_raw": float(pEmp[i]),
                        "pointSource": int(pointSourcesRel[i]),
                    }
                )

    if not allRows:
        logger.warning(
            "No matches detected, returning empty DataFrame."
        )

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
    qVals = bhFdr(df["p_raw"].values.astype(float))
    df["pValue"] = -np.log10(
        np.clip(df["p_raw"].values, 1.0e-10, 1.0)
    )
    df["qValue"] = -np.log10(np.clip(qVals, 1.0e-10, 1.0))
    df.drop(columns=["p_raw"], inplace=True)
    df = df[qVals <= alpha].copy()
    df["chromosome"] = df["chromosome"].astype(str)
    df.sort_values(by=["chromosome", "start", "end"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[
        [
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
    ]
    return df


def mergeMatches(
    filePath: str,
    mergeGapBP: Optional[int],
) -> Optional[str]:
    r"""Merge overlapping or nearby structured peaks ('matches') in a narrowPeak file.

    The harmonic mean of p-values and q-values is computed for each merged region within `mergeGapBP` base pairs.
    The fourth column (name) of each merged peak contains information about the number of features that were merged
    and the range of q-values among them.

    Expects a `narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_ file as input (all numeric columns, '.' for strand if unknown).

    :param filePath: narrowPeak file containing matches detected with :func:`consenrich.matching.matchWavelet`
    :type filePath: str
    :param mergeGapBP: Maximum gap size (in base pairs) to consider for merging. Defaults to 75 bp if `None` or less than 1.
    :type mergeGapBP: Optional[int]

    :seealso: :ref:`matching`, :class:`consenrich.core.matchingParams`
    """

    if mergeGapBP is None or mergeGapBP < 1:
        mergeGapBP = 75

    MAX_NEGLOGP = 10.0
    MIN_NEGLOGP = 1.0e-10

    if not os.path.isfile(filePath):
        logger.warning(f"Couldn't access {filePath}...skipping merge")
        return None
    bed = None
    try:
        bed = BedTool(filePath)
    except Exception as ex:
        logger.warning(
            f"Couldn't create BedTool for {filePath}:\n{ex}\n\nskipping merge..."
        )
        return None
    if bed is None:
        logger.warning(
            f"Couldn't create BedTool for {filePath}...skipping merge"
        )
        return None

    bed = bed.sort()
    clustered = bed.cluster(d=mergeGapBP)
    groups = {}
    for f in clustered:
        fields = f.fields
        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        score = float(fields[4])
        signal = float(fields[6])
        pLog10 = float(fields[7])
        qLog10 = float(fields[8])
        peak = int(fields[9])
        clusterID = fields[-1]
        if clusterID not in groups:
            groups[clusterID] = {
                "chrom": chrom,
                "sMin": start,
                "eMax": end,
                "scSum": 0.0,
                "sigSum": 0.0,
                "n": 0,
                "maxS": float("-inf"),
                "peakAbs": -1,
                "pMax": float("-inf"),
                "pTail": 0.0,
                "pHasInf": False,
                "qMax": float("-inf"),
                "qMin": float("inf"),
                "qTail": 0.0,
                "qHasInf": False,
            }
        g = groups[clusterID]
        if start < g["sMin"]:
            g["sMin"] = start
        if end > g["eMax"]:
            g["eMax"] = end
        g["scSum"] += score
        g["sigSum"] += signal
        g["n"] += 1

        if math.isinf(pLog10) or pLog10 >= MAX_NEGLOGP:
            g["pHasInf"] = True
        else:
            if pLog10 > g["pMax"]:
                if g["pMax"] == float("-inf"):
                    g["pTail"] = 1.0
                else:
                    g["pTail"] = (
                        g["pTail"] * (10 ** (g["pMax"] - pLog10))
                        + 1.0
                    )
                g["pMax"] = pLog10
            else:
                g["pTail"] += 10 ** (pLog10 - g["pMax"])

        if (
            math.isinf(qLog10)
            or qLog10 >= MAX_NEGLOGP
            or qLog10 <= MIN_NEGLOGP
        ):
            g["qHasInf"] = True
        else:
            if qLog10 < g["qMin"]:
                if qLog10 < MIN_NEGLOGP:
                    g["qMin"] = MIN_NEGLOGP
                else:
                    g["qMin"] = qLog10

            if qLog10 > g["qMax"]:
                if g["qMax"] == float("-inf"):
                    g["qTail"] = 1.0
                else:
                    g["qTail"] = (
                        g["qTail"] * (10 ** (g["qMax"] - qLog10))
                        + 1.0
                    )
                g["qMax"] = qLog10
            else:
                g["qTail"] += 10 ** (qLog10 - g["qMax"])

        if signal > g["maxS"]:
            g["maxS"] = signal
            g["peakAbs"] = start + peak if peak >= 0 else -1

    items = []
    for clusterID, g in groups.items():
        items.append((g["chrom"], g["sMin"], g["eMax"], g))
    items.sort(key=lambda x: (str(x[0]), x[1], x[2]))

    outPath = f"{filePath.replace('.narrowPeak', '')}.mergedMatches.narrowPeak"
    lines = []
    i = 0
    for chrom, sMin, eMax, g in items:
        i += 1
        avgScore = g["scSum"] / g["n"]
        if avgScore < 0:
            avgScore = 0
        if avgScore > 1000:
            avgScore = 1000
        scoreInt = int(round(avgScore))
        sigAvg = g["sigSum"] / g["n"]

        if g["pHasInf"]:
            pHMLog10 = MAX_NEGLOGP
        else:
            if (
                g["pMax"] == float("-inf")
                or not (g["pTail"] > 0.0)
                or math.isnan(g["pTail"])
            ):
                pHMLog10 = MIN_NEGLOGP
            else:
                pHMLog10 = -math.log10(g["n"]) + (
                    g["pMax"] + math.log10(g["pTail"])
                )
                pHMLog10 = max(
                    MIN_NEGLOGP, min(pHMLog10, MAX_NEGLOGP)
                )

        if g["qHasInf"]:
            qHMLog10 = MAX_NEGLOGP
        else:
            if (
                g["qMax"] == float("-inf")
                or not (g["qTail"] > 0.0)
                or math.isnan(g["qTail"])
            ):
                qHMLog10 = MIN_NEGLOGP
            else:
                qHMLog10 = -math.log10(g["n"]) + (
                    g["qMax"] + math.log10(g["qTail"])
                )
                qHMLog10 = max(
                    MIN_NEGLOGP, min(qHMLog10, MAX_NEGLOGP)
                )

        pointSource = (
            g["peakAbs"] - sMin
            if g["peakAbs"] >= 0
            else (eMax - sMin) // 2
        )

        qMinLog10 = g["qMin"]
        qMaxLog10 = g["qMax"]
        if math.isfinite(qMinLog10) and qMinLog10 < MIN_NEGLOGP:
            qMinLog10 = MIN_NEGLOGP
        if math.isfinite(qMaxLog10) and qMaxLog10 > MAX_NEGLOGP:
            qMaxLog10 = MAX_NEGLOGP
        elif (
            not math.isfinite(qMaxLog10)
            or not math.isfinite(qMinLog10)
        ) or (qMaxLog10 < MIN_NEGLOGP):
            qMinLog10 = 0.0
            qMaxLog10 = 0.0

        # informative+parsable name
        # e.g., regex: ^consenrichPeak\|i=(?P<i>\d+)\|gap=(?P<gap>\d+)bp\|ct=(?P<ct>\d+)\|qRange=(?P<qmin>\d+\.\d{3})_(?P<qmax>\d+\_\d{3})$
        name = f"consenrichPeak|i={i}|gap={mergeGapBP}bp|ct={g['n']}|qRange={qMinLog10:.3f}_{qMaxLog10:.3f}"
        lines.append(
            f"{chrom}\t{int(sMin)}\t{int(eMax)}\t{name}\t{scoreInt}\t.\t{sigAvg:.3f}\t{pHMLog10:.3f}\t{qHMLog10:.3f}\t{int(pointSource)}"
        )

    with open(outPath, "w") as outF:
        outF.write("\n".join(lines) + ("\n" if lines else ""))
    logger.info(f"Merged matches written to {outPath}")
    return outPath
