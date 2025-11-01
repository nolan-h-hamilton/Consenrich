# -*- coding: utf-8 -*-
r"""Module implementing (experimental) 'structured peak detection' features using wavelet-based templates."""

import logging
import os
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
            logger.info(f"Skipping {chrom_}: fewer than 5 rows.")
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
) -> pd.DataFrame:
    r"""Detect structured peaks by cross-correlating Consenrich tracks with wavelet- or scaling-function templates.

    :param chromosome: Chromosome name for the input intervals and values.
    :type chromosome: str
    :param values: A 1D array of signal-like values. In this documentation, we refer to values derived from Consenrich,
        but other continuous-valued tracks at evenly spaced genomic intervals may be suitable, too.
    :type values: npt.NDArray[np.float64]
    :param templateNames: A list of str values -- wavelet bases used for matching, e.g., `[haar, db2, sym4]`
    :type templateNames: List[str]
    :param cascadeLevels: A list of int values -- the number of cascade iterations used for approximating
        the scaling/wavelet functions.
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
    :type minMatchLengthBP: int
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

    :seealso: :class:`consenrich.core.matchingParams`, :func:`cconsenrich.csampleBlockStats`, :ref:`matching`
    :return: A pandas DataFrame with detected matches
    :rtype: pd.DataFrame
    """
    if len(intervals) < 5:
        raise ValueError("`intervals` must be at least length 5")
    if len(values) != len(intervals):
        raise ValueError(
            "`values` must have the same length as `intervals`"
        )
    intervalLengthBp = intervals[1] - intervals[0]
    if not np.all(np.abs(np.diff(intervals)) == intervalLengthBp):
        raise ValueError("`intervals` must be evenly spaced.")
    rng = np.random.default_rng(int(randSeed))
    cascadeLevels = sorted(list(set(cascadeLevels)))
    if weights is not None and len(weights) == len(values):
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
        resp: np.ndarray, orderBins: int
    ) -> np.ndarray:
        return signal.argrelmax(resp, order=max(int(orderBins), 1))[0]

    def sampleBlockMaxima(
        resp: np.ndarray,
        halfMask: np.ndarray,
        relWindowBins: int,
        nsamp: int,
        seed: int,
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
            ),
            dtype=float,
        )
        if len(vals) == 0:
            return vals
        low = np.quantile(vals, 0.001)
        high = np.quantile(vals, 0.999)
        return vals[(vals > low) & (vals < high)]

    for cascadeLevel in cascadeLevels:
        for templateName in templateNames:
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
            asinhThreshold = parseMinSignalThreshold(
                minSignalAtMaxima
            )
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
                )
                if len(blockMaxima) < 25:
                    pooledMask = ~excludeMaskGlobal.astype(bool)
                    blockMaxima = sampleBlockMaxima(
                        response,
                        pooledMask,
                        relWindowBins,
                        nsamp=max(iters, 1000),
                        seed=rng.integers(1, 10_000),
                    )
                ecdfSf = stats.ecdf(blockMaxima).sf
                candidateIdx = relativeMaxima(response, relWindowBins)

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
                    ecdfSf.evaluate(response[candidateIdx]), 1.0e-10, 1.0
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
                scores = (
                    250 + 750 * (sqScores - minR) / rangeR
                ).astype(int)
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


def mergeMatches(filePath: str, mergeGapBP: int = 75):
    r"""Merge overlapping or nearby structured peaks (matches) in a narrowPeak file.

    Where an overlap occurs within `mergeGapBP` base pairs, the feature with the greatest signal defines the new summit/pointSource

    :param filePath: narrowPeak file containing matches detected with :func:`consenrich.matching.matchWavelet`
    :type filePath: str
    :param mergeGapBP: Maximum gap size (in base pairs) to consider for merging
    :type mergeGapBP: int

    :seealso: :class:`consenrich.core.matchingParams`
    """
    if not os.path.isfile(filePath):
        logger.info(f"Couldn't access {filePath}...skipping merge")
        return None
    bed = None
    try:
        bed = BedTool(filePath)
    except Exception as ex:
        logger.info(
            f"Couldn't create BedTool for {filePath}:\n{ex}\n\nskipping merge..."
        )
        return None
    if bed is None:
        logger.info(
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
        pval = float(fields[7])
        qval = float(fields[8])
        peak = int(fields[9])
        clId = fields[-1]
        if clId not in groups:
            groups[clId] = {
                "chrom": chrom,
                "sMin": start,
                "eMax": end,
                "scSum": 0.0,
                "sigSum": 0.0,
                "pSum": 0.0,
                "qSum": 0.0,
                "n": 0,
                "maxS": float("-inf"),
                "peakAbs": -1,
            }
        g = groups[clId]
        if start < g["sMin"]:
            g["sMin"] = start
        if end > g["eMax"]:
            g["eMax"] = end
        g["scSum"] += score
        g["sigSum"] += signal
        g["pSum"] += pval
        g["qSum"] += qval
        g["n"] += 1
        # scan for largest signal, FFR: consider using the p-val in the future
        if signal > g["maxS"]:
            g["maxS"] = signal
            g["peakAbs"] = start + peak if peak >= 0 else -1
    items = []
    for clId, g in groups.items():
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
        pAvg = g["pSum"] / g["n"]
        qAvg = g["qSum"] / g["n"]
        pointSource = g["peakAbs"] - sMin if g["peakAbs"] >= 0 else -1
        name = f"consenrichStructuredPeak{i}"
        lines.append(
            f"{chrom}\t{int(sMin)}\t{int(eMax)}\t{name}\t{scoreInt}\t.\t{sigAvg:.3f}\t{pAvg:.3f}\t{qAvg:.3f}\t{int(pointSource)}"
        )
    with open(outPath, "w") as outF:
        outF.write("\n".join(lines) + ("\n" if lines else ""))
    logger.info(f"Merged matches written to {outPath}")
    return outPath


def textNullCDF(
    nullBlockMaximaSFVals: npt.NDArray[np.float64],
    binCount: int = 20,
    barWidth: int = 50,
    barChar="\u25a2",
    normalize: bool = False,
) -> str:
    r"""Plot a histogram of the distribution 1 - ECDF(nullBlockMaxima)

    Called by :func:`consenrich.matching.matchWavelet`. Ideally resembles
    a uniform(0,1) distribution.

    :seealso: :func:`consenrich.matching.matchWavelet`, :ref:`cconsenrich.csampleBlockStats`
    """
    valueLower, valueUpper = (
        min(nullBlockMaximaSFVals),
        max(nullBlockMaximaSFVals),
    )
    binCount = max(1, int(binCount))
    binStep = (valueUpper - valueLower) / binCount
    binEdges = [
        valueLower + indexValue * binStep
        for indexValue in range(binCount)
    ]
    binEdges.append(valueUpper)
    binCounts = [0] * binCount
    for numericValue in nullBlockMaximaSFVals:
        binIndex = int((numericValue - valueLower) / binStep)
        if binIndex == binCount:
            binIndex -= 1
        binCounts[binIndex] += 1
    valueSeries = (
        [
            countValue / len(nullBlockMaximaSFVals)
            for countValue in binCounts
        ]
        if normalize
        else binCounts[:]
    )
    valueMaximum = max(valueSeries) if valueSeries else 0
    widthScale = (barWidth / valueMaximum) if valueMaximum > 0 else 0
    edgeFormat = f"{{:.{2}f}}"
    rangeLabels = [
        f"[{edgeFormat.format(binEdges[indexValue])},{edgeFormat.format(binEdges[indexValue + 1])})"
        for indexValue in range(binCount)
    ]
    labelWidth = max(len(textValue) for textValue in rangeLabels)
    lines = ['Histogram: "1 - ECDF(nullBlockMaxima)"']
    for rangeLabel, seriesValue, countValue in zip(
        rangeLabels, valueSeries, binCounts
    ):
        barString = barChar * int(round(seriesValue * widthScale))
        trailingText = (
            f"({countValue}/{len(nullBlockMaximaSFVals)})\t\t"
        )
        lines.append(
            f"{rangeLabel.rjust(labelWidth)} | {barString}{trailingText.ljust(10)}"
        )
    return "\n".join(lines)
