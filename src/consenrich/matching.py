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
    mergeGapBP: int = 50,
    merge: bool = True,
    weights: Optional[npt.NDArray[np.float64]] = None,
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
            )
        except Exception as ex:
            logger.info(f"Skipping {chrom_} due to error in matchWavelet: {ex}")
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
            mergedPath = mergeMatches(perChromOut, mergeGapBP=mergeGapBP)
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
        logger.info(f"All merged matches written to {outPathMergedAll}")

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

    See :ref:`matching` for an overview of the approach.

    :param chromosome: Chromosome name for the input intervals and values.
    :type chromosome: str
    :param values: 'Consensus' signal estimates derived from multiple samples, e.g., from Consenrich.
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
        :math:`1 - \alpha` quantile of an empirical null distribution. The empirical null
        distribution is built from cross-correlation values over randomly sampled blocks.
    :type alpha: float
    :param minMatchLengthBP: Within a window of `minMatchLengthBP` length (bp), relative maxima in
        the signal-template convolution must be greater in value than others to qualify as matches.
        *Set to a negative value to disable this filter*.
    :type minMatchLengthBP: int
    :param minSignalAtMaxima: Secondary significance threshold coupled with `alpha`. Require the *signal value*
        at relative maxima in the response sequence to be greater than this threshold. Comparisons are made in log-scale.
        If a `float` value is provided, the minimum signal value must be greater than this (absolute) value. *Set to a
        negative value to disable the threshold*.
        If a `str` value is provided, looks for 'q:quantileValue', e.g., 'q:0.75'. The
        threshold is then set to the corresponding quantile of the non-zero signal estimates.
        Defaults to str value 'q:0.75' --- the 75th percentile of signal values.
    :type minSignalAtMaxima: Optional[str | float]
    :param useScalingFunction: If True, use (only) the scaling function to build the matching template.
        If False, use (only) the wavelet function.
    :type useScalingFunction: bool
    :param excludeRegionsBedFile: A BED file with regions to exclude from matching
    :type excludeRegionsBedFile: Optional[str]

    :seealso: :class:`consenrich.core.matchingParams`, :func:`cconsenrich.csampleBlockStats`, :ref:`matching`
    """

    if len(intervals) < 5:
        raise ValueError("`intervals` must be at least length 5")
    if len(values) != len(intervals):
        raise ValueError("`values` must have the same length as `intervals`")
    intervalLengthBP = intervals[1] - intervals[0]
    if not np.all(np.abs(np.diff(intervals)) == intervalLengthBP):
        # FFR: don't change this exception message without updating tests
        # --'spaced' is matched in tests
        raise ValueError("`intervals` must be evenly spaced.")

    randSeed_: int = int(randSeed)
    cols = [
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
    matchDF = pd.DataFrame(columns=cols)
    minMatchLengthBPCopy: Optional[int] = minMatchLengthBP
    cascadeLevels = sorted(list(set(cascadeLevels)))
    if weights is not None and len(weights) == len(values):
        values = values * weights
    asinhValues = np.asinh(values, dtype=np.float32)
    asinhNonZeroValues = asinhValues[asinhValues > 0]
    iters = max(iters, 1000)
    defQuantile: float = 0.75
    for l_, cascadeLevel in enumerate(cascadeLevels):
        for t_, templateName in enumerate(templateNames):
            try:
                templateName = str(templateName)
                cascadeLevel = int(cascadeLevel)
            except ValueError:
                logger.info(
                    f"Skipping invalid templateName or cascadeLevel: {templateName}, {cascadeLevel}"
                )
                continue
            if templateName not in pw.wavelist(kind="discrete"):
                logger.info(
                    f"\nSkipping unknown wavelet template: {templateName}\nAvailable templates: {pw.wavelist(kind='discrete')}"
                )
                continue

            wav = pw.Wavelet(templateName)
            scalingFunc, waveletFunc, x = wav.wavefun(level=cascadeLevel)
            template = np.array(waveletFunc, dtype=np.float64) / np.linalg.norm(
                waveletFunc
            )

            if useScalingFunction:
                template = np.array(
                    scalingFunc, dtype=np.float64
                ) / np.linalg.norm(scalingFunc)

            logger.info(
                f"Matching: template: {templateName}, cascade level: {cascadeLevel}, template length: {len(template)}, scaling: {useScalingFunction}, wavelet: {not useScalingFunction}"
            )

            responseSequence: npt.NDArray[np.float64] = signal.fftconvolve(
                values, template[::-1], mode="same"
            )

            minMatchLengthBP = minMatchLengthBPCopy
            if minMatchLengthBP is None or minMatchLengthBP < 1:
                minMatchLengthBP = len(template) * intervalLengthBP
            if minMatchLengthBP % intervalLengthBP != 0:
                minMatchLengthBP += intervalLengthBP - (
                    minMatchLengthBP % intervalLengthBP
                )

            relativeMaximaWindow = int(
                ((minMatchLengthBP / intervalLengthBP) / 2) + 1
            )
            relativeMaximaWindow = max(relativeMaximaWindow, 1)

            excludeMask = np.zeros(len(intervals), dtype=np.uint8)
            if excludeRegionsBedFile is not None:
                excludeMask = core.getBedMask(
                    chromosome,
                    excludeRegionsBedFile,
                    intervals,
                )

            logger.info(
                f"\nSampling {iters} block maxima for template {templateName} at cascade level {cascadeLevel} with (expected) relative maxima window size {relativeMaximaWindow}.\n"
            )
            blockMaxima = np.array(
                cconsenrich.csampleBlockStats(
                    intervals.astype(np.uint32),
                    responseSequence,
                    relativeMaximaWindow,
                    iters * 2,
                    randSeed_,
                    excludeMask.astype(np.uint8),
                ),
                dtype=float,
            )
            blockMaximaCheck = blockMaxima.copy()[iters:]
            blockMaxima = blockMaxima[:iters]
            blockMaxima = blockMaxima[
                (blockMaxima > np.quantile(blockMaxima, 0.005))
                & (blockMaxima < np.quantile(blockMaxima, 0.995))
            ]

            ecdfBlockMaximaSF = stats.ecdf(blockMaxima).sf

            responseThreshold = float(1e6)
            arsinhSignalThreshold = float(1e6)
            try:
                # we use 'interpolated_inverted_cdf' in a few spots
                # --- making sure it's supported here, at its first use
                responseThreshold = np.quantile(
                    blockMaxima, 1 - alpha, method="interpolated_inverted_cdf"
                )
            except (TypeError, ValueError, KeyError) as err_:
                logger.warning(
                    f"\nError computing response threshold  with alpha={alpha}:\n{err_}\n"
                    f"\nIs `blockMaxima` empty?"
                    f"\nIs NumPy older than 1.22.0 (~May 2022~)?"
                    f"\nIs `alpha` in (0,1)?\n"
                )
                raise

            if minSignalAtMaxima is None or (
                isinstance(minSignalAtMaxima, (float, int))
                and minSignalAtMaxima < 0.0
            ):
                arsinhSignalThreshold = -float(1e6)
            elif isinstance(minSignalAtMaxima, str):
                if minSignalAtMaxima.startswith("q:"):
                    qVal = float(minSignalAtMaxima.split("q:")[-1])
                    if qVal < 0 or qVal > 1:
                        raise ValueError(f"Quantile {qVal} is out of range")
                    arsinhSignalThreshold = float(
                        np.quantile(
                            asinhNonZeroValues,
                            qVal,
                            method="interpolated_inverted_cdf",
                        )
                    )

                else:
                    logger.info(
                        f"Unrecognized string format for minSignalAtMaxima: {minSignalAtMaxima}\n"
                        f"Using default instead...."
                    )
                    arsinhSignalThreshold = float(
                        np.quantile(
                            asinhNonZeroValues,
                            defQuantile,
                            method="interpolated_inverted_cdf",
                        )
                    )
            elif isinstance(minSignalAtMaxima, (float, int)):
                arsinhSignalThreshold = np.asinh(float(minSignalAtMaxima))

            relativeMaximaIndices = signal.argrelmax(
                responseSequence, order=relativeMaximaWindow
            )[0]

            relativeMaximaIndices = relativeMaximaIndices[
                (responseSequence[relativeMaximaIndices] > responseThreshold)
                & (asinhValues[relativeMaximaIndices] > arsinhSignalThreshold)
            ]

            if len(relativeMaximaIndices) == 0:
                logger.info(
                    f"no matches were detected using for template {templateName} at cascade level {cascadeLevel}...skipping matching"
                )
                continue

            if maxNumMatches is not None:
                if len(relativeMaximaIndices) > maxNumMatches:
                    # take the greatest maxNumMatches (by 'signal')
                    relativeMaximaIndices = relativeMaximaIndices[
                        np.argsort(asinhValues[relativeMaximaIndices])[
                            -maxNumMatches:
                        ]
                    ]

            ecdfSFCheckVals: npt.NDArray[np.float64] = (
                ecdfBlockMaximaSF.evaluate(blockMaximaCheck)
            )
            testKS, _ = stats.kstest(
                ecdfSFCheckVals,
                stats.uniform.cdf,
                alternative="two-sided",
            )

            logger.info(
                f"\n\tDetected {len(relativeMaximaIndices)} matches (alpha={alpha}, useScalingFunction={useScalingFunction}): {templateName}: level={cascadeLevel}.\n"
                f"\tResponse threshold: {responseThreshold:.3f}, Signal Threshold: {arsinhSignalThreshold:.3f}\n"
                f"\t~KS_Statistic~ [ePVals, uniformCDF]: {testKS:.4f}\n"
                f"\n\n{textNullCDF(ecdfSFCheckVals)}\n\n" # lil text-plot histogram of approx. null CDF
            )

            # starts
            startsIdx = np.maximum(
                relativeMaximaIndices - relativeMaximaWindow, 0
            )
            # ends
            endsIdx = np.minimum(
                len(values) - 1, relativeMaximaIndices + relativeMaximaWindow
            )
            # point source
            pointSourcesIdx = []
            for start_, end_ in zip(startsIdx, endsIdx):
                pointSourcesIdx.append(
                    np.argmax(values[start_ : end_ + 1]) + start_
                )
            pointSourcesIdx = np.array(pointSourcesIdx)
            starts = intervals[startsIdx]
            ends = intervals[endsIdx]
            pointSources = (intervals[pointSourcesIdx]) + max(
                1, intervalLengthBP // 2
            )
            if (
                recenterAtPointSource
            ):  # recenter at point source (signal maximum)
                starts = pointSources - (
                    relativeMaximaWindow * intervalLengthBP
                )
                ends = pointSources + (relativeMaximaWindow * intervalLengthBP)
            pointSources = (intervals[pointSourcesIdx] - starts) + max(
                1, intervalLengthBP // 2
            )
            # (ucsc browser) score [0,1000]
            sqScores = (1 + responseSequence[relativeMaximaIndices]) ** 2
            minResponse = np.min(sqScores)
            maxResponse = np.max(sqScores)
            rangeResponse = max(maxResponse - minResponse, 1.0)
            scores = (
                250 + 750 * (sqScores - minResponse) / rangeResponse
            ).astype(int)
            # feature name
            names = [
                f"{templateName}_{cascadeLevel}_{i}"
                for i in relativeMaximaIndices
            ]
            # strand
            strands = ["." for _ in range(len(scores))]
            # p-values in -log10 scale per convention
            pValues = -np.log10(
                np.clip(
                    ecdfBlockMaximaSF.evaluate(
                        responseSequence[relativeMaximaIndices]
                    ),
                    1e-10,
                    1.0,
                )
            )
            # q-values (ignored)
            qValues = np.array(np.ones_like(pValues) * -1.0)

            tempDF = pd.DataFrame(
                {
                    "chromosome": [chromosome] * len(relativeMaximaIndices),
                    "start": starts.astype(int),
                    "end": ends.astype(int),
                    "name": names,
                    "score": scores,
                    "strand": strands,
                    "signal": responseSequence[relativeMaximaIndices],
                    "pValue": pValues,
                    "qValue": qValues,
                    "pointSource": pointSources.astype(int),
                }
            )

            if matchDF.empty:
                matchDF = tempDF
            else:
                matchDF = pd.concat([matchDF, tempDF], ignore_index=True)
            randSeed_ += 1

    if matchDF.empty:
        logger.info("No matches detected, returning empty DataFrame.")
        return matchDF
    matchDF.sort_values(by=["chromosome", "start", "end"], inplace=True)
    matchDF.reset_index(drop=True, inplace=True)
    return matchDF


def mergeMatches(filePath: str, mergeGapBP: int = 50):
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
        logger.info(f"Couldn't create BedTool for {filePath}...skipping merge")
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
        name = f"mergedPeak{i}"
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
        valueLower + indexValue * binStep for indexValue in range(binCount)
    ]
    binEdges.append(valueUpper)
    binCounts = [0] * binCount
    for numericValue in nullBlockMaximaSFVals:
        binIndex = int((numericValue - valueLower) / binStep)
        if binIndex == binCount:
            binIndex -= 1
        binCounts[binIndex] += 1
    valueSeries = (
        [countValue / len(nullBlockMaximaSFVals) for countValue in binCounts]
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
        trailingText = f"({countValue}/{len(nullBlockMaximaSFVals)})\t\t"
        lines.append(
            f"{rangeLabel.rjust(labelWidth)} | {barString}{trailingText.ljust(10)}"
        )
    return "\n".join(lines)
