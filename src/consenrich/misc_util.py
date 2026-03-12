# -*- coding: utf-8 -*-
r"""
==============================================================================
`consenrich.misc_util` -- Miscellaneous utility functions
==============================================================================

"""

import os
from typing import List, Optional, Tuple
import logging
import re
import numpy as np
import pandas as pd

from scipy import signal, ndimage
from . import ccounts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def checkAlignmentFile(alignmentFile: str) -> bool:
    r"""Check that an alignment file exists and is indexed

    Assumes the bam file is sorted by coordinates
    """
    if not os.path.exists(alignmentFile):
        raise FileNotFoundError(f"Could not find {alignmentFile}")
    return bool(
        ccounts.ccounts_checkAlignmentPath(
            alignmentFile,
            sourceKind="CRAM" if str(alignmentFile).lower().endswith(".cram") else "BAM",
            buildIndex=True,
        )
    )


def checkBamFile(bamFile: str) -> bool:
    r"""Backward-compatible alias for alignment file checks"""

    return checkAlignmentFile(bamFile)


def alignmentFilesArePairedEnd(
    alignmentFiles: List[str], maxReads: int = 1_000
) -> List[bool]:
    """
    Take a list of alignment files, return a list indicating whether
    each file contains paired-end reads

    :param alignmentFiles: List of paths to alignment files
    :type alignmentFiles: List[str]
    :param maxReads: Maximum number of reads to check in each file
    :type maxReads: int
    :return: List of booleans corresponding to each input file
    :rtype: List[bool]
    """

    results = []
    for path in alignmentFiles:
        results.append(
            bool(
                ccounts.ccounts_isAlignmentPairedEnd(
                    path,
                    maxReads=maxReads,
                    sourceKind="CRAM" if str(path).lower().endswith(".cram") else "BAM",
                )
            )
        )
    return results


def bamsArePairedEnd(bamFiles: List[str], maxReads: int = 1_000) -> List[bool]:
    r"""Backward-compatible alias for alignment file detection"""

    return alignmentFilesArePairedEnd(bamFiles, maxReads=maxReads)


def getChromSizesDict(
    sizes_file: str,
    excludeRegex: str = r"^chr[A-Za-z0-9]+$",
    excludeChroms: Optional[List[str]] = None,
) -> dict:
    r"""The function getChromSizesDict is a helper to get chromosome sizes file as a dictionary.
    :param sizes_file: Path to a genome assembly's chromosome sizes file
    :param exclude_regex: Regular expression to exclude chromosomes. Default: all non-standard chromosomes.
    :param exclude_chroms: List of chromosomes to exclude.
    :return: Dictionary of chromosome sizes. Formatted as `{chromosome_name: size}`
    """
    if excludeChroms is None:
        excludeChroms = []
    return {
        k: v
        for k, v in pd.read_csv(
            sizes_file,
            sep="\t",
            header=None,
            index_col=0,
            names=["chrom", "size"],
        )["size"]
        .to_dict()
        .items()
        if re.search(excludeRegex, k) is not None and k not in excludeChroms
    }
