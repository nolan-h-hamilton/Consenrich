# -*- coding: utf-8 -*-
r"""
==============================================================================
`consenrich.misc_util` -- Miscellaneous utility functions
==============================================================================

"""

from typing import List, Optional
from collections.abc import Mapping

import ccounts as counts
from . import constants


def isStandardAutosomalChromosome(chromosome: str) -> bool:
    chrom = str(chromosome).strip()
    if chrom in constants.NONSTANDARD_CHROMOSOME_NAMES:
        return False
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if not chrom.isdigit():
        return False
    chromNumber = int(chrom)
    return 1 <= chromNumber <= 22


def getChromSizesDict(
    sizesFile: str | Mapping[str, int],
    excludeChroms: Optional[List[str]] = None,
) -> dict:
    r"""The function getChromSizesDict is a helper to get chromosome sizes file as a dictionary.
    :param sizes_file: Path to a genome assembly's chromosome sizes file
    :param exclude_regex: Regular expression selecting chromosome names to keep.
    :param exclude_chroms: List of chromosomes to exclude.
    :return: Dictionary of chromosome sizes. Formatted as `{chromosome_name: size}`
    """
    excluded = set(excludeChroms or ())
    sizes = sizesFile if isinstance(sizesFile, Mapping) else counts.readChromSizes(sizesFile)
    return {name: length for name, length in sizes.items()
            if name not in excluded}
