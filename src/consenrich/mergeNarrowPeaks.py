from __future__ import annotations

from math import log10
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple, Optional, Union


PathLike = Union[str, Path]


class Peak(NamedTuple):
    chrom: str
    start: int
    end: int
    score: float
    signal: float
    pLog10: Optional[float]
    qLog10: Optional[float]


def mergeAndSortNarrowPeaks(
    paths: Iterable[PathLike],
    outPath: Optional[PathLike] = None,
    defaultGapBP: int = 147,
    gapByChrom: Optional[dict[str, int]] = None,
    gapFunc: Optional[Callable[[str], int]] = None,
) -> List[str]:
    def getGap(chrom: str) -> int:
        if gapFunc is not None:
            g = int(gapFunc(chrom))
            return g if g > 0 else 0
        return 250

    allPeaks: List[Peak] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for rawLine in f:
                line = rawLine.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])

                score = float(parts[4]) if len(parts) > 4 and parts[4] not in ("", ".") else 0.0
                signal = float(parts[6]) if len(parts) > 6 and parts[6] not in ("", ".") else 0.0

                pLog10: Optional[float] = None
                if len(parts) > 7 and parts[7] not in ("", "."):
                    try:
                        pLog10 = float(parts[7])
                    except ValueError:
                        pLog10 = None

                qLog10: Optional[float] = None
                if len(parts) > 8 and parts[8] not in ("", "."):
                    try:
                        qLog10 = float(parts[8])
                    except ValueError:
                        qLog10 = None

                allPeaks.append(Peak(chrom, start, end, score, signal, pLog10, qLog10))

    allPeaks.sort(key=lambda x: (x.chrom, x.start, x.end))

    mergedLines: List[str] = []
    idx = 1

    currChrom: Optional[str] = None
    currStart = 0
    currEnd = 0
    scoreSum = 0.0
    signalSum = 0.0
    n = 0
    pVals: List[float] = []
    qVals: List[float] = []

    def flush() -> None:
        nonlocal idx, currChrom, currStart, currEnd, scoreSum, signalSum, n, pVals, qVals, mergedLines
        if currChrom is None or n == 0:
            return

        meanScore = scoreSum / float(n)
        scoreInt = int(round(meanScore))
        meanSignal = signalSum / float(n)

        pOut = float("nan")
        if pVals:
            pMean = sum(10.0 ** (-v) for v in pVals) / float(len(pVals))
            pOut = max(pVals) if pMean <= 0.0 else -log10(pMean)

        qOut = float("nan")
        if qVals:
            qMean = sum(10.0 ** (-v) for v in qVals) / float(len(qVals))
            qOut = max(qVals) if qMean <= 0.0 else -log10(qMean)

        center = (currStart + currEnd) // 2
        peakOffset = center - currStart

        def fmt(x: float) -> str:
            return "." if x != x else f"{x:.6g}"

        mergedLines.append(
            "\t".join(
                [
                    currChrom,
                    str(currStart),
                    str(currEnd),
                    f"consenrichPeak_merged_{idx}",
                    str(scoreInt),
                    ".",
                    fmt(meanSignal),
                    fmt(pOut),
                    fmt(qOut),
                    str(peakOffset),
                ]
            )
        )

        idx += 1
        currChrom = None
        scoreSum = 0.0
        signalSum = 0.0
        n = 0
        pVals = []
        qVals = []

    for peak_ in allPeaks:
        if currChrom is None:
            currChrom, currStart, currEnd = peak_.chrom, peak_.start, peak_.end
            scoreSum = peak_.score
            signalSum = peak_.signal
            n = 1
            pVals = [peak_.pLog10] if peak_.pLog10 is not None else []
            qVals = [peak_.qLog10] if peak_.qLog10 is not None else []
            continue

        if peak_.chrom != currChrom:
            flush()
            currChrom, currStart, currEnd = peak_.chrom, peak_.start, peak_.end
            scoreSum = peak_.score
            signalSum = peak_.signal
            n = 1
            pVals = [peak_.pLog10] if peak_.pLog10 is not None else []
            qVals = [peak_.qLog10] if peak_.qLog10 is not None else []
            continue

        if peak_.start > currEnd + getGap(currChrom):
            flush()
            currChrom, currStart, currEnd = peak_.chrom, peak_.start, peak_.end
            scoreSum = peak_.score
            signalSum = peak_.signal
            n = 1
            pVals = [peak_.pLog10] if peak_.pLog10 is not None else []
            qVals = [peak_.qLog10] if peak_.qLog10 is not None else []
            continue

        if peak_.end > currEnd:
            currEnd = peak_.end
        scoreSum += peak_.score
        signalSum += peak_.signal
        n += 1
        if peak_.pLog10 is not None:
            pVals.append(peak_.pLog10)
        if peak_.qLog10 is not None:
            qVals.append(peak_.qLog10)

    flush()

    if outPath is not None:
        with open(outPath, "w", encoding="utf-8") as out:
            out.write("\n".join(mergedLines) + ("\n" if mergedLines else ""))

    return mergedLines
