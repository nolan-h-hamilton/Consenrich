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
import pybedtools as bed
import pysam as sam

from scipy import signal, ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def checkBamFile(bamFile: str) -> bool:
    r"""Check that the bam file exists and is indexed

    Assumes the bam file is sorted by coordinates
    """
    has_index = False
    if not os.path.exists(bamFile):
        raise FileNotFoundError(f"Could not find {bamFile}")
    try:
        bamfile = sam.AlignmentFile(bamFile, "rb")
        has_index = bamfile.check_index()
        bamfile.close()
    except AttributeError as aex:
        logger.info(f"Alignments must be in BAM format:\n{aex}")
        raise
    except ValueError as vex:
        has_index = False
        pass

    if not has_index:
        try:
            logger.info(
                f"Could not find index file for {bamFile}. Calling pysam.index()"
            )
            sam.index(bamFile)
            has_index = True
        except Exception as ex:
            logger.warning(
                f"Encountered the following exception\n{ex}\nCould not create index file for {bamFile}: is it sorted?"
            )

    return has_index


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
