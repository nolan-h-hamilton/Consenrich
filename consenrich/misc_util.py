r"""
Consenrich Utilities `misc_util` Documentation
===============================================================

The `misc_util` module contains utility functions for Consenrich.

"""

import logging
import os
import re

import numpy as np
import pandas as pd
import pybedtools as pbt
import pysam

from scipy import signal, ndimage, stats

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_shape(matrix: np.ndarray) -> tuple:
    r"""The function `get_shape` is a helper to get the shape of a matrix that returns a two-element tuple for both 1D and 2D input"""
    if len(matrix.shape) != 2:
        return (1, len(matrix))
    return matrix.shape

def get_step(intervals: np.ndarray) -> int:
    r"""The function `get_step` is a helper to determine step of `intervals` array and enforce uniform spacing."""
    if len(np.unique(np.diff(intervals))) != 1:
        raise ValueError(f"Track must be evenly spaced, found the following distinct interval sizes:\
            \n{np.unique(np.diff(intervals))}")
    return intervals[1] - intervals[0]


def match_lengths(intervals: np.ndarray, values: np.ndarray) -> int:
    r"""The function `match_lengths` is a helper to ensure `intervals` and `values` are of equal length.

    :param intervals: Numpy array of intervals (genomic positions).
    :param values: Numpy array of values (Typically some function increasing with the number of sequence alignments at each interval).
    :return: Length of `intervals` and `values` if they are equal.

    """
    if len(intervals) != len(values):
        raise ValueError(f"Length of intervals and values must be the same:\
            \nFound {len(intervals)} intervals and {len(values)} values")
    return len(intervals)


def wrap_index(bam_file: str) -> bool:
    """The function `wrap_index` checks if an index file (.bai) exists for `bam_file`. If not, it tries to invoke `pysam.index`"""
    has_index = False
    if not os.path.exists(bam_file):
        raise FileNotFoundError(f'Could not find {bam_file}')
    try:
        bamfile = pysam.AlignmentFile(bam_file, "rb")
        has_index = bamfile.check_index()
        bamfile.close()
    except AttributeError as aex:
        logger.info(f'Alignments must be in BAM format:\n{aex}')
        raise
    except ValueError as vex:
        has_index = False
        pass

    if not has_index:
        try:
            logger.info(f'Could not find index file for {bam_file}...calling pysam.index()')
            pysam.index(bam_file)
            has_index = True
        except Exception as ex:
            logger.warning(f'Encountered the following exception\n{ex}\nCould not create index file for {bam_file}: is it sorted?')

    return has_index


def get_chromsizes_dict(sizes_file: str,
                        exclude_regex: str=r'^chr[A-Za-z0-9]+$',
                        exclude_chroms: list=['chrM', 'chrEBV']) -> dict:
    r"""The function `get_chromsizes_dict` is a helper to get chromosome sizes file as a dictionary.
    
    :param sizes_file: Path to sizes file OR the name of a genome supported by  `pybedtools <https://daler.github.io/pybedtools/>`_
    :param exclude_regex: Regular expression to exclude chromosomes. Default excludes all non-standard chromosomes.
    :param exclude_chroms: List of chromosomes to exclude.
    :return: Dictionary of chromosome sizes. Formatted as `{chromosome_name: size}`, e.g., `{'chr1': 248956422, 'chr2': 242193529, ...}`

    """
    genome_ = None
    # if sizes_file is not a file, assume it is a genome name
    if not os.path.exists(sizes_file):
        logger.info(f"Could not find file {sizes_file}, assuming it is a genome name and calling pybedtools.chromsizes()")
        genome_ = sizes_file
        return {k: v[1] for k, v in pbt.chromsizes(genome_).items() if re.search(exclude_regex, k) is not None and k not in exclude_chroms}
    return {k: v for k, v in pd.read_csv(sizes_file, sep='\t', header=None, index_col=0, names=['chrom','size'])['size'].to_dict().items() if re.search(exclude_regex, k) is not None and k not in exclude_chroms}


def get_first_read(chromosome: str,
                   bam_file: str,
                   sizes_file: str,
                   exclude_flag: int=3840,
                   min_mapq: float=0.0,
                   step: int=50) -> int:
    r"""The function `get_first_read` returns the first read position (interval/step) for a given chromosome and BAM file that meets the specified criteria.
    
    :param chromosome: Chromosome name.
    :param bam_file: Path to BAM file.
    :param sizes_file: Path to sizes file.
    :param exclude_flag: SAM flag to exclude reads.
    :param min_mapq: Minimum mapping quality.
    :param step: Step size for intervals.
    :return: First read position.
    """


    sizes_dict = get_chromsizes_dict(sizes_file=sizes_file)
    start_ = 0
    stop_ = sizes_dict[chromosome] - (sizes_dict[chromosome] % step)
    first = None
    with pysam.AlignmentFile(bam_file,'rb') as bam:
        for read in bam.fetch(chromosome, start=start_, stop=stop_):
            if not read.flag & exclude_flag and read.mapping_quality >= min_mapq:
                return read.reference_start - (read.reference_start % step)
    return first


def get_last_read(chromosome: str,
                bam_file: str,
                sizes_file: str,
                exclude_flag: int=3840,
                min_mapq: int=0,
                step: int=50,
                backshift: int=None) -> int:
    r"""The function `get_last_read` returns the last read position (interval/step) for a given chromosome and BAM file that meets the specified criteria.
    
    :param chromosome: Chromosome name.
    :param bam_file: Path to BAM file.
    :param sizes_file: Path to sizes file.
    :param exclude_flag: SAM flag to exclude reads.
    :param min_mapq: Minimum mapping quality.
    :param step: Step size for intervals.
    :param backshift: Number of base pairs to backshift the last read.
    :return: Last read position.

    """

    sizes_dict = get_chromsizes_dict(sizes_file=sizes_file)
    stop_=sizes_dict[chromosome] - (sizes_dict[chromosome] % step)
    if backshift is None:
        backshift = (0.025*stop_)
        backshift = backshift - (backshift % step)
    start_ = max(0,stop_ - backshift)
    last = None
    with pysam.AlignmentFile(bam_file,'rb') as bam:
        for read in bam.fetch(chromosome, start=start_, stop=stop_):
            if not read.flag & exclude_flag and read.mapping_quality >= min_mapq:
                last = read.reference_start - (read.reference_start % step)
    if last is None:
        last = stop_ - (stop_ % step)
    return last


def write_bigwig(tsv_file, sizes_file, chrom_list, outfile_name, stat='signal', abs_residuals=False):
    r"""Write a bigWig file from a TSV file and sizes file.
    
    :param tsv_file: The five-column BedGraph-like TSV files generated by `run_consenrich`.
    :param sizes_file: Path to sizes file.
    :param stat: Statistic to write. Default is 'signal'.
    
    :raises ImportError: If pyBigWig is not installed or was compiled without numpy support.

    """

    try:
        import pyBigWig as pbw
    except ImportError:
        raise ImportError('Optional dependency pyBigWig not found. Try `pip install pybigwig`. Skipping bigWig output.')
    if pbw.numpy != 1:
        raise ImportError('pyBigWig was not compiled with numpy support. Try reinstalling pyBigWig.')
    
    if stat.lower() not in ['signal', 'ptrace', 'rtrace', 'residuals_ivw', 'residuals', 'ratio']:
        raise ValueError('stat must be either "signal", "ptrace", "rtrace", "residuals_ivw", "ratio".')

    if os.path.exists(outfile_name):
        logger.warning(f'Overwriting existing bigWig file: {outfile_name}')
        os.remove(outfile_name)

    sizes_dict = get_chromsizes_dict(sizes_file)
    chrom_list = sorted(chrom_list, key=lambda x: (x.lower(), x[3:]))

    pbw_out = pbw.open(outfile_name, 'w')
    pbw_out.addHeader([(chrom, size) for chrom, size in sizes_dict.items() if chrom in chrom_list])
    tsv_df = pd.read_csv(tsv_file, sep='\t', names=['chrom','start','end','signal', 'Ptrace', 'Rtrace', 'residual_ivw'])
    for chrom in chrom_list:
        chrom_df = tsv_df[tsv_df['chrom'] == chrom]
        chroms = np.array(chrom_df['chrom'], dtype='str')
        starts = np.array(chrom_df['start'], dtype='int')
        ends = np.array(chrom_df['end'], dtype='int')
        if stat.lower() == 'signal':
            sig_values = np.array(chrom_df['signal'], dtype='float')
            pbw_out.addEntries(chroms, starts, ends=ends, values=sig_values)
        elif stat.lower() == 'ptrace':
            ptrace_values = np.array(chrom_df['Ptrace'], dtype='float')
            pbw_out.addEntries(chroms, starts, ends=ends, values=ptrace_values)
        elif stat.lower() == 'rtrace':
            rtrace_values = np.array(chrom_df['Rtrace'], dtype='float')
            pbw_out.addEntries(chroms, starts, ends=ends, values=rtrace_values)
        elif stat.lower() == 'residuals_ivw':
            res_values = np.array(chrom_df['residual_ivw'], dtype='float')
            if abs_residuals:
                res_values = np.abs(res_values)
            pbw_out.addEntries(chroms, starts, ends=ends, values=res_values)
        elif stat.lower() == 'ratio':
            sq_signal_values = (np.array(chrom_df['signal'], dtype='float')**2) + 1
            sq_res_values = (np.array(chrom_df['residual_ivw'], dtype='float')**2) + 1
            ratio_vals = np.round(np.log2((sq_signal_values)/(sq_res_values)),3)
            pbw_out.addEntries(chroms, starts, ends=ends, values=ratio_vals)
    pbw_out.close()
    logger.info(f'Wrote bigWig file for {stat} to {outfile_name}')
    return outfile_name


def chrom_lexsort(chromosomes, sizes_file=None):
    r"""Sorts `chromosomes` in lexicographical order (e.g., '11' precedes '2').
    """
    if sizes_file is not None:
        sizes_dict = get_chromsizes_dict(sizes_file)
        chromosomes = [chrom for chrom in chromosomes if chrom in sizes_dict]
    return sorted(chromosomes, key=lambda x: (x.lower(), x[3:]))