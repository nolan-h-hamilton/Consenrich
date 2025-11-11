# -*- coding: utf-8 -*-
r"""Module implementing (experimental) 'structured peak detection' features using wavelet-based templates."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
import pywt as pw
from scipy import signal, stats
import logging
from typing import List, Optional
import os, math
from pybedtools import BedTool

from .adapters import bed_mask_adapter, sample_block_stats_adapter

logger = logging.getLogger(__name__)

# this is a new helper that returns candidate min lengths
def autoMinLengthCandidates(
    values: np.ndarray,
    initLen: int = 3,
    quantiles: tuple = (0.50, 0.75, 0.90),  # typical, longer, high-end
    weight_by_intensity: bool = True,       # this favors better runs
) -> List[int]:
    """
    Infer candidate minimum run lengths from a signal.

    The method applies asinh transform, subtracts a median filter, keeps positive residuals,
    thresholds at a high quantile, groups contiguous runs, and returns run-length candidates
    weighted by area if requested.

    :param values: One dimensional array of signal values at uniform genomic bins.
    :param initLen: Hard lower bound on minimum run length in bins.
    :param quantiles: Cumulative weight cut points used to pick representative run lengths.
    :param weight_by_intensity: If true, weight each run by width times mean residual.
    :return: Sorted unique list of run length candidates in bins.
    """
    
    #same transformation, but only compute once and take a true median
    tr = np.asanyarray(values, dtype = np.float64)
    tr = np.asinh(tr)
    ks_target = max(2*int(initLen) + 1, 2*int(len(tr)*0.005) + 1)
    ks_odd = ks_target | 1
    ks_max = len(tr) if (len(tr) % 2 == 1) else (len(tr) - 1)
    ksize = min(ks_odd, ks_max) if len(tr) >= 3 else 3

    trValues = tr - signal.medfilt(tr, kernel_size=ksize)
   
    #only ones that pass threshold
    nz = trValues[trValues > 0]
    if len(nz) == 0:
        return [int(initLen)]

    #high quantile to keep super strong ones
    thr = np.quantile(nz, 0.90, method="interpolated_inverted_cdf")
    mask = trValues >= thr
    if not np.any(mask):
        return [int(initLen)]

    #checker
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    runs = idx.reshape(-1, 2)
    widths = runs[:, 1] - runs[:, 0]

    #remove super short runs
    keep = widths >= int(initLen)
    if not np.any(keep):
        return [int(initLen)]
    runs = runs[keep]
    widths = widths[keep]

    #weight again to favor clearer and stronger signals from each run
    if weight_by_intensity:
        means = np.array([float(trValues[s:e].mean()) for s, e in runs])
        wts = widths * means  # area weighting
    else:
        wts = np.ones_like(widths, dtype=float)

    #get cands
    out: List[int] = []
    order = np.argsort(widths)
    w = wts[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        out = [int(np.median(widths))]
    else:
        cw = cw / cw[-1]
        for q in quantiles:
            qf = float(q)
            j = int(np.searchsorted(cw, np.clip(qf, 0.0, 1.0), side="left"))
            j = int(np.clip(j, 0, len(widths) - 1))
            out.append(int(widths[order][j]))

    #make sure it is minimum, unique and sorted in an ascending manner
    out = sorted({int(x) for x in out if int(x) >= int(initLen)})
    return out or [int(initLen)]
    

def autoMinLengthIntervals(values: np.ndarray, initLen: int = 3) -> int:
    """
    Backward compatible wrapper over :func:`autoMinLengthCandidates` that returns only the first candidate.

    :param values: Signal values at uniform bins.
    :param initLen: Lower bound on the run length in bins.
    :return: A single run length in bins.
    """
    return int(autoMinLengthCandidates(values, initLen=initLen)[0])


def scalarClip(value: float, low: float, high: float) -> float:
    """
    Clip a scalar to the closed interval [low, high].

    :param value: Input value.
    :param low: Lower bound.
    :param high: Upper bound.
    :return: Clipped value.
    """
    return low if value < low else high if value > high else value


def castableToFloat(value) -> bool:
    """
    Return True if value can be safely cast to a finite float, False otherwise.

    Strings like nan inf and empty are treated as non castable.

    :param value: Object to test.
    :return: Boolean.
    """
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
    # new params
    adaptiveScale: bool = False,            
    scalesBp: Optional[List[int]] = None,   
    scaleStepBp: int = 10000,              
    refineNeighbors: bool = True,           
) -> Optional[str]:
    r"""
    Run wavelet matching on a bedGraph file and write outputs.

    This is a convenience wrapper around :func:`consenrich.matching.elet`.
    It processes each chromosome block in the input bedGraph and writes
    per chromosome narrowPeak files, optionally merges nearby matches,
    then writes a single concatenated output.

    :param bedGraphFile: Path to a four column bedGraph with chromosome start end value.
    :type bedGraphFile: str
    :param templateName: Discrete wavelet name available in PyWavelets wavelist kind discrete.
    :type templateName: str
    :param cascadeLevel: Cascade level used to build the template.
    :type cascadeLevel: int
    :param alpha: FDR threshold used to filter detected matches.
    :type alpha: float
    :param minMatchLengthBP: Minimum span in base pairs for local maxima competition.
        If None uses 250. If less than 1 triggers automatic selection in bins, then converted to base pairs.
    :type minMatchLengthBP: Optional[int]
    :param iters: Number of blocks to draw when estimating the empirical tail.
    :type iters: int
    :param minSignalAtMaxima: Absolute value or a string of the form q:quantile for the asinh signal threshold.
    :type minSignalAtMaxima: Optional[str or float]
    :param maxNumMatches: Upper bound on the number of candidates retained per pass.
    :type maxNumMatches: Optional[int]
    :param recenterAtPointSource: If true recenters each interval on the local point source before writing.
    :type recenterAtPointSource: bool
    :param useScalingFunction: If true builds the template from the scaling function else from the wavelet.
    :type useScalingFunction: bool
    :param excludeRegionsBedFile: Optional BED file whose regions are masked during matching.
    :type excludeRegionsBedFile: Optional[str]
    :param mergeGapBP: Gap in base pairs for merging overlapping or nearby matches. Defaults to half the minimum span.
    :type mergeGapBP: Optional[int]
    :param merge: If true merges matches within mergeGapBP and returns the merged file.
    :type merge: bool
    :param weights: Optional per bin weights aligned to values.
    :type weights: Optional[numpy.ndarray]
    :param randSeed: Random seed for reproducibility.
    :type randSeed: int
    :param adaptiveScale: Enable adaptive span selection per candidate using a sparse scale map.
    :type adaptiveScale: bool
    :param scalesBp: Candidate template spans in base pairs used in adaptive mode. Defaults to 120 180 240 320 480 if not provided.
    :type scalesBp: Optional[List[int]]
    :param scaleStepBp: Step in base pairs between evaluation centers when building the sparse scale map.
    :type scaleStepBp: int
    :param refineNeighbors: If true also tries adjacent scales and keeps the better response.
    :type refineNeighbors: bool
    :return: Path to the final output narrowPeak file. Returns the merged file if merge is true otherwise the unmerged concatenated file. Returns None if nothing was detected.
    :rtype: Optional[str]
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
                #new 
                adaptiveScale=adaptiveScale,          
                scalesBp=scalesBp,                    
                scaleStepBp=scaleStepBp,            
                refineNeighbors=refineNeighbors,      
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

# here is another thought that can automate the whole thing
def makeTemplateAtBins(base_template: np.ndarray, target_bins: int) -> np.ndarray:
    """
    Resample a base template to the target length in bins and L2 normalize it.

    :param base_template: Unit norm 1D template at arbitrary length.
    :param target_bins: Desired odd length in bins, minimum of three.
    :return: Resampled and normalized template.
    """
    #Resample a base template to target length in bins and L2 normalize.
    tb = int(max(3, target_bins))
    if tb == len(base_template):
        t = base_template.astype(np.float64, copy=True)
    else:
        t = signal.resample(base_template, tb).astype(np.float64, copy=False)
    nrm = np.linalg.norm(t)
    if not np.isfinite(nrm) or nrm <= 0:
        return np.zeros(tb, dtype=np.float64)
    return t / nrm

def responseAt(values: np.ndarray, center_idx: int, tmpl_rev: np.ndarray) -> float:
    """
    Compute dot product response of a reversed template centered at a given index with zero padding at edges.

    :param values: Signal values.
    :param center_idx: Center index in values.
    :param tmpl_rev: Reversed template vector.
    :return: Dot product response as float.
    """
    L = len(tmpl_rev)
    half = L // 2
    s = max(0, center_idx - half)
    e = min(len(values), center_idx + half + 1)
    w = values[s:e].astype(np.float64, copy=False)
    if len(w) < L:
        pad_left = max(0, half - (center_idx - s))
        pad_right = max(0, (center_idx + half + 1) - e)
        w = np.pad(w, (pad_left, pad_right), mode="constant", constant_values=0.0)
    return float(np.dot(w, tmpl_rev))

def buildScaleMapSparse(values: np.ndarray,
                            interval_bp: int,
                            base_template: np.ndarray,
                            scalesBp: List[int],
                            step_bp: int) -> np.ndarray:
    """
    Build a sparse map of locally best template spans.

    Evaluate reversed template dot products at regularly spaced centers for each candidate span,
    select the winning span per center, then fill gaps by forward and backward propagation and
    optional median filtering.

    :param values: Signal values array.
    :param interval_bp: Bin size in base pairs.
    :param base_template: Unit-norm base template at arbitrary length.
    :param scalesBp: Candidate spans in base pairs.
    :param step_bp: Step in base pairs between sampled centers.
    :return: Integer index per bin selecting the winning span.
    """
    n = len(values)
    step_bins = max(1, int(round(step_bp / interval_bp)))
    centers = np.arange(step_bins // 2, n, step_bins, dtype=int)
    if len(centers) == 0:
        return np.zeros(n, dtype=np.int32)

    is_center = np.zeros(n, dtype=np.bool_)
    is_center[centers] = True

    tmpl_per_scale = []
    for sbp in scalesBp:
        Lb = int(max(3, round(sbp / interval_bp)))
        if Lb % 2 == 0:
            Lb += 1
        tmpl_per_scale.append(makeTemplateAtBins(base_template, Lb))

    energy = np.zeros((len(scalesBp), len(centers)), dtype=np.float64)
    for k, tmpl in enumerate(tmpl_per_scale):
        half = len(tmpl) // 2
        for j, c in enumerate(centers):
            s = max(0, c - half)
            e = min(n, c + half + 1)
            w = values[s:e]
            if len(w) < len(tmpl):
                pad_left = max(0, half - (c - s))
                pad_right = max(0, (c + half + 1) - e)
                w = np.pad(w, (pad_left, pad_right), mode="constant", constant_values=0.0)
            r = float(np.dot(w, tmpl[::-1]))
            energy[k, j] = r * r

    winner_idx = np.argmax(energy, axis=0)

    scale_map = np.zeros(n, dtype=np.int32)
    scale_map[centers] = winner_idx

    last = -1
    for i in range(n):
        if is_center[i]:
            last = scale_map[i]
        elif last >= 0:
            scale_map[i] = last

    last = -1
    for i in range(n - 1, -1, -1):
        if is_center[i]:
            last = scale_map[i]
        elif last >= 0 and scale_map[i] == 0:
            scale_map[i] = last

    if n >= 5:
        scale_map = signal.medfilt(scale_map, kernel_size=5).astype(np.int32)
    return scale_map
                                

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
    # NEW!!
    adaptiveScale: bool = False,            
    scalesBp: Optional[List[int]] = None,   
    scaleStepBp: int = 10000,              
    refineNeighbors: bool = True,
    # NEWEST!!
    get_bed_mask = bed_mask_adapter,
    sample_block_stats = sample_block_stats_adapter,
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
    :param adaptiveScale: choose a template length per candidate using a sparse scale map
    :type adaptiveScale: bool
    :param scalesBp: list of candidate template spans in base pairs
    :type scalesBp: Optional[List[int]]
    :param scaleStepBp: step in base pairs for sparse scale map centers
    :type scaleStepBp: int
    :param refineNeighbors: also test adjacent scales and keep the best
    :type refineNeighbors: bool

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
    #new code
    if minMatchLengthBP is not None and minMatchLengthBP < 1:
        cand_bins = autoMinLengthCandidates(values, initLen=3)
        cand_bp = [int(b * intervalLengthBp) for b in cand_bins]
    elif minMatchLengthBP is None:
        cand_bp = [250]
    else:
        cand_bp = [int(minMatchLengthBP)]
    logger.info(f"\n\tUsing minMatchLengthBP candidates: {cand_bp}")

    if not np.all(np.abs(np.diff(intervals)) == intervalLengthBp):
        raise ValueError("`intervals` must be evenly spaced.")
        
    if weights is not None:
        if len(weights) != len(values):
            logger.warning(
                f"`weights` length {len(weights)} does not match `values` length {len(values)}. Ignoring..."
            )
        else:
            values = values * weights
    
    asinhValues = np.asanyarray(values, dtype=np.float32)
    asinhValues = np.asinh(asinhValues)
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
      excludeMaskGlobal = get_bed_mask(chromosome,
                                     excludeRegionsBedFile,
                                     intervals).astype(np.uint8)
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
          sample_block_stats(
            intervals.astype(np.uint32),
            resp.astype(np.float32, copy=False),
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

        # base template is the normalized template above
        base_template = template
        interval_bp = int(intervalLengthBp)

        if adaptiveScale:
            local_scalesBp = list(scalesBp) if scalesBp is not None else None
            if not local_scalesBp:
                local_scalesBp = [120, 180, 240, 320, 480]
            local_scalesBp = sorted({int(max(3, s)) for s in local_scalesBp})

            tmpl_per_scale = []
            tmpl_per_scale_rev = []
            for sbp in local_scalesBp:
                Lb = int(max(3, round(sbp / interval_bp)))
                if Lb % 2 == 0:
                    Lb += 1
                t = makeTemplateAtBins(base_template, Lb)
                tmpl_per_scale.append(t)
                tmpl_per_scale_rev.append(t[::-1])

            scale_map = buildScaleMapSparse(
                values=values,
                interval_bp=interval_bp,
                base_template=base_template,
                scalesBp=local_scalesBp,
                step_bp=int(scaleStepBp),
            )
            logger.info(f"\n\tAdaptive scale enabled with scalesBp={local_scalesBp}")
        else:
            tmpl_per_scale = None
            scale_map = None

        
        # seed response with the base template for candidate discovery
        response = signal.fftconvolve(values, base_template[::-1], mode="same")

        for thisMinMatchBp in cand_bp:
            if thisMinMatchBp is None or thisMinMatchBp < 1:
                thisMinMatchBp = len(base_template) * interval_bp
            if thisMinMatchBp % interval_bp != 0:
                thisMinMatchBp += interval_bp - (thisMinMatchBp % interval_bp)

            relWindowBins = int(((thisMinMatchBp / interval_bp) / 2) + 1)
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
                if len(blockMaxima) == 0:
                    continue

                ecdfSf = stats.ecdf(blockMaxima).sf
                candidateIdx = relativeMaxima(response, relWindowBins)

                candidateMask = (
                    (candidateIdx >= relWindowBins)
                    & (candidateIdx < len(response) - relWindowBins)
                    & (testMask[candidateIdx])
                    & ((excludeRegionsBedFile is None) | (excludeMaskGlobal[candidateIdx] == 0))
                    & (asinhValues[candidateIdx] > asinhThreshold)
                )
                candidateIdx = candidateIdx[candidateMask]
                if len(candidateIdx) == 0:
                    continue

                if (maxNumMatches is not None) and (len(candidateIdx) > maxNumMatches):
                    candidateIdx = candidateIdx[
                        np.argsort(asinhValues[candidateIdx])[-maxNumMatches:]
                    ]

                if adaptiveScale:
                    refined_resp = np.empty(len(candidateIdx), dtype=np.float64)
                    refined_halfbins = np.empty(len(candidateIdx), dtype=np.int32)

                    for i, idxVal in enumerate(candidateIdx):
                        k = int(scale_map[idxVal])
                        if tmpl_per_scale is None:
                            k = 0
                        else:
                            k = int(np.clip(k, 0, len(tmpl_per_scale) - 1))
                        tmpl_k_rev = tmpl_per_scale_rev[k]
                        refined_resp[i] = responseAt(values, idxVal, tmpl_k_rev)
                        refined_halfbins[i] = len(tmpl_per_scale[k]) // 2

                    if refineNeighbors and tmpl_per_scale is not None and len(tmpl_per_scale) > 1:
                        for i, idxVal in enumerate(candidateIdx):
                            k = int(scale_map[idxVal])
                            k = int(np.clip(k, 0, len(tmpl_per_scale) - 1))
                            best_r = refined_resp[i]
                            best_k = k
                            for kk in (k - 1, k + 1):
                                if 0 <= kk < len(tmpl_per_scale):
                                    r_kk = responseAt(values, idxVal, tmpl_per_scale_rev[kk])
                                    if r_kk > best_r:
                                        best_r = r_kk
                                        best_k = kk
                            refined_resp[i] = best_r
                            refined_halfbins[i] = len(tmpl_per_scale[best_k]) // 2

                    pEmp = np.clip(ecdfSf(refined_resp), 1.0e-10, 1.0)
                    startsIdx = np.maximum(candidateIdx - refined_halfbins, 0)
                    endsIdx = np.minimum(len(values) - 1, candidateIdx + refined_halfbins)
                    use_resp = refined_resp
                else:
                    pEmp = np.clip(ecdfSf(response[candidateIdx]), 1.0e-10, 1.0)
                    startsIdx = np.maximum(candidateIdx - relWindowBins, 0)
                    endsIdx = np.minimum(len(values) - 1, candidateIdx + relWindowBins)
                    use_resp = response[candidateIdx]

                pointSourcesIdx = []
                for s, e in zip(startsIdx, endsIdx):
                    pointSourcesIdx.append(np.argmax(values[s : e + 1]) + s)
                pointSourcesIdx = np.array(pointSourcesIdx)

                starts = intervals[startsIdx]
                ends = intervals[endsIdx]
                pointSourcesAbs = intervals[pointSourcesIdx] + max(1, interval_bp // 2)

                if recenterAtPointSource:
                    if adaptiveScale:
                        halfbins = endsIdx - candidateIdx
                    else:
                        halfbins = np.full(len(candidateIdx), relWindowBins, dtype=int)
                    starts = pointSourcesAbs - (halfbins * interval_bp)
                    ends = pointSourcesAbs + (halfbins * interval_bp)
                left_bound = int(intervals[0])
                right_bound = int(intervals[-1] + interval_bp)
                starts = np.clip(starts, left_bound, right_bound - interval_bp)
                ends = np.clip(ends, left_bound + interval_bp, right_bound)

                pointSourcesRel = (intervals[pointSourcesIdx] - starts) + max(1, interval_bp // 2)

                sqScores = (1.0 + use_resp) ** 2
                minR, maxR = float(np.min(sqScores)), float(np.max(sqScores))
                rangeR = max(maxR - minR, 1.0)
                scores = (250 + 750 * (sqScores - minR) / rangeR).astype(int)

                for i, idxVal in enumerate(candidateIdx):
                    # optionally include chosen span in name when adaptive to aid QC
                    if adaptiveScale:
                        chosen_bp = int(ends[i] - starts[i])
                        name_field = f"{templateName}_{cascadeLevel}_{idxVal}_{tag}|sbp={chosen_bp}"
                    else:
                        name_field = f"{templateName}_{cascadeLevel}_{idxVal}_{tag}"

                    allRows.append(
                        {
                            "chromosome": chromosome,
                            "start": int(starts[i]),
                            "end": int(ends[i]),
                            "name": name_field,
                            "score": int(scores[i]),
                            "strand": ".",
                            "signal": float(use_resp[i]),
                            "p_raw": float(pEmp[i]),
                            "pointSource": int(pointSourcesRel[i]),
                        }
                    )
    if not allRows:
        logger.warning("No matches detected, returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "chromosome", "start", "end", "name", "score", "strand",
                "signal", "pValue", "qValue", "pointSource",
            ]
        )
    df = pd.DataFrame(allRows)
    # Deduplicate events possibly found by multiple window candidates
    df.drop_duplicates(subset=["chromosome", "start", "end", "name"], inplace=True)
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
