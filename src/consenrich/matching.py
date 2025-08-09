# -*- coding: utf-8 -*-
r"""Module implementing spatial feature recognition (localization) in Consenrich-estimated genomic signals.
"""

import logging
from typing import List, Optional

import pandas as pd
import pywt as pw
import numpy as np
import numpy.typing as npt

from scipy import signal, stats

from . import cconsenrich

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def matchWavelet(chromosome: str,
    intervals: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    templateNames: List[str],
    cascadeLevels: List[int],
    iters: int,
    alpha: float,
    minMatchLengthBP: Optional[int],
    maxNumMatches: Optional[int] = 10_000,
    minSignalAtMaxima: Optional[float] = None,
    randSeed: int = 42
    ) -> pd.DataFrame:
    r"""(Experimental) Detect genomic regions with **signal enrichment** and **relevant, non-random structure** reinforced in multiple samples.

    Employs a matched filtering-based approach on the sequence of Consenrich signal estimates using wavelet-based templates approximated using the cascade algorithm.

    At a given genomic interval :math:`i \in \{1, \ldots, n\}`, a 'match' with the given template occurs if:

    - The filter response sequence (the convolution of the signal with a time-reversed template) is a local maximum within a window defined by the template (:math:`\psi`) length.

    - The maximum exceeds a significance cutoff determined by the :math:`1 - \alpha` quantile of an approximated null distribution (See :func:`cconsenrich.csampleBlockStats`).

    :param chromosome: Chromosome name
    :type chromosome: str
    :param intervals: Index set of genomic intervals (e.g., [0, 25, 50, ...] nucleotides).
    :type intervals: npt.NDArray[np.float64]
    :param values: Multi-sample signal estimates, e.g., from Consenrich.
    :type values: npt.NDArray[np.float64]
    :param templateNames: List of wavelet template names to use for matching, e.g. `[haar, db2, coif4]`
    :type templateNames: List[str]
    :param cascadeLevels: List of cascade 'levels' used for each template, e.g., `[1, 1, 1]` for `[haar, db2, coif4]`
    :type cascadeLevels: List[int]
    :param iters: Number of random blocks to sample while building the empirical distribution.
    :type iters: int
    :param alpha: Significance level for thresholding the response sequence.
    :type alpha: float

    :seealso: :class:`consenrich.core.matchingParams`, :func:`cconsenrich.csampleBlockStats`
    """
    randSeed_: int = int(randSeed)
    intervalLengthBP = intervals[1] - intervals[0]
    cols = ['chromosome', 'start', 'end', 'name', 'score', 'strand', 'signal', 'pValue', 'qValue', 'pointSource']
    matchDF = pd.DataFrame(columns=cols)

    for templateName, cascadeLevel in zip(templateNames, cascadeLevels):
        try:
            templateName = str(templateName)
            cascadeLevel = int(cascadeLevel)
        except ValueError:
            logger.warning(f"Skipping invalid templateName or cascadeLevel: {templateName}, {cascadeLevel}")
            continue
        if templateName not in pw.wavelist(kind='discrete'):
            logger.warning(f"\nSkipping unknown wavelet template: {templateName}\nAvailable templates: {pw.wavelist(kind='discrete')}")
            continue

        wav = pw.Wavelet(templateName)
        scalingFunc, waveletFunc, x = wav.wavefun(level=cascadeLevel)
        template = np.array(waveletFunc, dtype=np.float64)/np.linalg.norm(waveletFunc)
        responseSequence: npt.NDArray[np.float64] = signal.fftconvolve(values, template[::-1], mode='same')

        if minMatchLengthBP is None:
            minMatchLengthBP = len(template)*intervalLengthBP*2
        relativeMaximaWindow = int(minMatchLengthBP // intervalLengthBP)

        if relativeMaximaWindow % 2 == 0 or relativeMaximaWindow < 1:
            relativeMaximaWindow += 1

        logger.info(f"\nSampling {iters} block maxima for template {templateName} at cascade level {cascadeLevel} with relative maxima window size {relativeMaximaWindow}.")
        blockMaxima = cconsenrich.csampleBlockStats(responseSequence, relativeMaximaWindow, iters, randSeed_)
        responseThreshold = np.quantile(blockMaxima, 1-alpha)
        ecdfBlockMaximaSF = stats.ecdf(blockMaxima).sf
        logger.info(f"Done. Sampled {len(blockMaxima)} block maxima to determine threshold: {responseThreshold:.3f}.\n")

        signalThreshold: float = 0.0
        if minSignalAtMaxima is not None:
            signalThreshold = np.quantile(values, 0.75)

        relativeMaximaIndices = signal.argrelmax(responseSequence, order=relativeMaximaWindow)[0]
        relativeMaximaIndices = relativeMaximaIndices[(responseSequence[relativeMaximaIndices] > responseThreshold) & (values[relativeMaximaIndices] > signalThreshold)]

        if maxNumMatches is not None:
            if len(relativeMaximaIndices) > maxNumMatches:
                relativeMaximaIndices = relativeMaximaIndices[np.argsort(responseSequence[relativeMaximaIndices])[-maxNumMatches:]]

        if len(relativeMaximaIndices) == 0:
            logger.warning(f"no matches were detected using for template {templateName} at cascade level {cascadeLevel}.")
            continue

        starts = intervals[relativeMaximaIndices] - minMatchLengthBP
        ends = intervals[relativeMaximaIndices] + minMatchLengthBP
        pointSources = ((starts + ends) // 2) - starts
        minResponse = np.min(responseSequence[relativeMaximaIndices])
        if minResponse < 0:
            responseSequence[relativeMaximaIndices] -= minResponse
            minResponse = 0.0
        maxResponse = np.max(responseSequence[relativeMaximaIndices])
        rangeResponse = max(maxResponse - minResponse, 1.0)
        scores = 250 + 750*(responseSequence[relativeMaximaIndices] - minResponse) / rangeResponse
        names = [f'{templateName}_{cascadeLevel}_{i}' for i in relativeMaximaIndices]
        strands = ['.' for _ in range(len(scores))]
        pValues = (-np.log10(np.clip(ecdfBlockMaximaSF.evaluate(responseSequence[relativeMaximaIndices]), 1e-8, 1.0))).astype(np.float32)
        qValues = np.array(np.ones_like(pValues)*-1, dtype=np.float32)

        tempDF = pd.DataFrame({
            'chromosome': [chromosome]*len(relativeMaximaIndices),
            'start': starts.astype(int),
            'end': ends.astype(int),
            'name': names,
            'score': scores,
            'strand': strands,
            'signal': responseSequence[relativeMaximaIndices],
            'pValue': pValues,
            'qValue': qValues,
            'pointSource': pointSources.astype(int)
            })
        if matchDF.empty:
            matchDF = tempDF
        else:
            matchDF = pd.concat([matchDF, tempDF], ignore_index=True)
        randSeed_ += 1

    if matchDF.empty:
        logger.warning(f"No matches detected, returning empty DataFrame.")
        return matchDF
    matchDF.sort_values(by=['chromosome', 'start', 'end'], inplace=True)
    matchDF.reset_index(drop=True, inplace=True)
    return matchDF


