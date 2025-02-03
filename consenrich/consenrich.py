r"""
Primary Module: `consenrich` Documentation
===============================================================

The `consenrich` module contains the primary functions for Consenrich.

"""

# standard library imports
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import sys
import uuid

from datetime import datetime
from typing import List, Tuple

# third-party package imports
import numpy as np
import pandas as pd
import pybedtools as pbt
import pysam

from scipy import ndimage, signal, stats

# module imports
from consenrich.misc_util import *


logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_genome_resource(filename, dir_='refdata'):
    file_path = os.path.join(os.path.dirname(__file__), os.path.join(dir_, filename))
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
    return file_path


def create_sparsebed(active_bed: str, sizes_file: str,
                     blacklist_file: str=None,
                     min_length: int=500,
                     max_length: int=5000,
                     num_windows: int=5,
                     min_gap: int=100,
                     outfile: str='sparse.bed'):
    r"""The function `create_sparsebed` generates an annotation of 'sparse segments' (genomic regions) from the complement of an 'inflated' BED file of known active regions.
    
    Briefly, given an annotation of previously identified, active regulatory regions
    (e.g., BED features: ENCODE cCRES, DNase I peaks etc.) widen or 'inflate' each by
    `min_gap` (`bedtools slop`). Then, take the *complement of the inflated active regions*
    to further avoid overlap with the potentially active regions. Finally, only retain
    regions satisfying length-based requirements. If a `blacklist_file` is provided
    subtract the blacklist regions from generated BED file (recommended).

    Active Region Annotations: `active_bed`
    ----------------------------------------
    ENCODE's collection of candidate cis-regulatory elements (cCREs) can be found at `<https://screen.encodeproject.org/>`_
    for mice (mm10,mm39,etc.) and humans (hg19,hg38,etc). The cCREs are available as BED files. These are likely suitable for use
    as the `active_bed` file in this function.

    For other organisms, the Table Browser  at `<https://genome.ucsc.edu/cgi-bin/hgTables>`_ can be used to integrate regulatory
    annotations from various sources (ORegAnno, ReMap Chip-seq, etc.) and export as BED files.
    

    Chromosome Sizes Files: `sizes_file`
    ------------------------------------
    The sizes file for a given `ASSEMBLY` can be obtained at `<https://hgdownload.soe.ucsc.edu/goldenPath/ASSEMBLY/bigZips/ASSEMBLY.chrom.sizes>`_
    For example, for the fruit fly assembly `dm6`: `<https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.chrom.sizes>`_


    Blacklist Regions: `blacklist_file`
    -----------------------------------
    The blacklisted genomic regions in BED format for several organisms can be found on the `Blacklist GitHub repository <https://github.com/Boyle-Lab/Blacklist/tree/61a04d2c5e49341d76735d485c61f0d1177d08a8/lists>`_.

    Example Use
    ----------------

    .. code-block:: python

        known_active_regions = 'hg38_cCREs.bed'
        hg38_sizes = 'hg38.sizes'
        hg38_blacklist= 'hg38_blacklist.bed'
        create_sparsebed(active_bed=known_active_regions,
          sizes_file=hg38_sizes,
          blacklist_file=hg38_blacklist,
          outfile='hg38_sparse.bed')

    :param active_bed: Path to BED file of active regions.
    :param sizes_file: Path to sizes file.
    :param blacklist_file: Path to blacklist file.
    :param min_length: Minimum length of regions to retain.
    :param max_length: Maximum length of regions to retain.
    :param num_windows: Number of windows to generate.
    :param min_gap: Desired minimum gap between regions.
    :param outfile: Path to write sparse BED file.
    :return: Path to sparse BED file.

    .. note::
        If you do not have access to a sufficient annotation of relevant regions, try running Consenrich with `--no_sparsebed` instead of creating a sparsebed.
    """
    sizes_df = get_chromsizes_dict(sizes_file=sizes_file)
    tmp_fname = f'tmp_gwide_sparse_{str(uuid.uuid4())[:5]}.bed'
    
    sparse_bed_init = pbt.BedTool(active_bed).slop(b=int(2*min_gap), g=sizes_file).sort(g=sizes_file).complement(g=sizes_file).filter(lambda x: len(x) >= min_length).filter(lambda x: len(x) <= max_length).subtract(b=active_bed, A=True)

    if blacklist_file is not None:
        sparse_bed_init = sparse_bed_init.subtract(b=blacklist_file, A=True)
        
    # save intermediate file for debugging
    sparse_bed_init.saveas(tmp_fname)
    if not os.path.exists(tmp_fname):
        raise FileNotFoundError(f"Could not find/generate file: {tmp_fname}")
    sparse_bed = pbt.BedTool(tmp_fname)

    with open(outfile, 'w') as f:
        for chromosome_ in sizes_df:
            logger.info(f"Creating sparse bed for chromosome {chromosome_}")
            chrom_sparse_fname = f'tmp_{chromosome_}_sparse.bed'
            sparse_bed.filter(lambda x: x.chrom == chromosome_).saveas(f'{chrom_sparse_fname}')
            if not os.path.exists(chrom_sparse_fname):
                raise FileNotFoundError(f"Could not find/generate {chrom_sparse_fname}")

            for feature in pbt.BedTool().window_maker(b=chrom_sparse_fname, n=num_windows):
                f.write(str(feature))
            try:
                os.remove(chrom_sparse_fname)
            except:
                logger.warning(f"Could not remove temporary file: {chrom_sparse_fname}")
    pbt.BedTool(tmp_fname).slop(b=-min_gap, g=sizes_file).subtract(b=active_bed).saveas(outfile)
    try:
        os.remove(tmp_fname)
    except:
        logger.warning(f"Could not remove temporary/intermediate file: {tmp_fname}")

    return outfile


def check_read(read: pysam.AlignedSegment,
               include_flag: int=None,
               exclude_flag: int=3840,
               min_mapq: float=10) -> bool:
    """The `check_read` function is a helper to filter reads based on SAM flags and mapping quality.

    .. note::
      `<https://broadinstitute.github.io/picard/explain-flags.html>`_ provides a useful tool for picking SAM flag values.

    .. note:
        Mapping quality scores are not consistent across different software for alignment, etc.

    """

    if include_flag is not None:
        if not read.flag & include_flag:
            return False

    if exclude_flag is not None:
        if read.flag & exclude_flag:
            return False

    if read.mapping_quality < min_mapq: 
        return False

    return True


def get_rlen_stats(chromosome: str, bam_file: str, sizes_file: str,
                blacklist_file: str=None,
                paired_end: bool=True,
                threads :int = -1,
                interval_length: int=5000,
                num_samples: int=1000,
                delete_random_features: bool=True,
                min_sample_size: int=50,
                random_seed: int=42) -> tuple:
    """The `get_rlen_stats` function estimates the read length statistics for a given chromosome and BAM file.
    
    :param chromosome: Chromosome name.
    :param bam_file: Path to BAM file.
    :param sizes_file: Path to sizes file.
    :param blacklist_file: Path to blacklist file.
    :param paired_end: Whether the alignment is paired-end.
    :param threads: Number of threads to use.
    :param interval_length: Length of random intervals.
    :param num_samples: Number of random intervals to sample.
    :param delete_random_features: Whether to delete temporary random features file.
    :param min_sample_size: Minimum number of reads to estimate read length.
    :param random_seed: Random seed for sampling.
    :return: Tuple of median read length, mean read lengths, and standard deviations of read lengths.

    """

    if threads < 0:
        threads = max(1,mp.cpu_count()//2)
    else:
        threads = threads
    
    chrom_end = get_chromsizes_dict(sizes_file=sizes_file)[chromosome]
    num_samples = num_samples if num_samples < chrom_end//interval_length else chrom_end//interval_length
    read_lengths = []
    mean_read_lengths = []
    std_read_lengths = []

    tmp_sizefile_content = f'{chromosome}\t{chrom_end}\n'
    tmp_sizefile_name = f'tmp_sizes_{chromosome}_{str(uuid.uuid4())[:5]}.bed'
    with open(tmp_sizefile_name, 'w') as f:
        f.write(tmp_sizefile_content)
    random_samples = pbt.BedTool().random(n=num_samples, l=interval_length, g=tmp_sizefile_name, seed=random_seed)
    try:
        os.remove(tmp_sizefile_name)
    except:
        logger.warning(f"Could not remove temporary file: {tmp_sizefile_name}")
    if blacklist_file is not None:
        random_samples = random_samples.subtract(b=blacklist_file)
    random_features_fname = f'chrom_{chromosome}_rand_{str(uuid.uuid4())[:5]}.bed'
    random_samples.saveas(random_features_fname)

    with pysam.AlignmentFile(bam_file,'rb', threads=threads) as bam:
        # iterate over each feature in `random_samples` and mapping reads
        for feature in random_samples:
            feature_rlen_list = []
            for read in bam.fetch(contig=chromosome, start=feature.start, stop=feature.end, multiple_iterators=False):
                # To estimate for PE: only consider primary, properly paired reads (one read per pair)
                if paired_end:
                    if read.is_proper_pair and read.is_read1 and not read.is_secondary and not read.is_unmapped and not read.is_duplicate:
                        feature_rlen_list.append(read.infer_query_length(always=False))
                # To estimate for SE: consider mapped primary reads
                else:
                    if not paired_end and not read.is_secondary and not read.is_unmapped and not read.is_duplicate:
                        feature_rlen_list.append(read.infer_query_length(always=False))

            if len(feature_rlen_list) > 0:
                read_lengths.append(np.median(feature_rlen_list))
                # track statistics for possible later use in uncertainty modeling
                if len(feature_rlen_list) >= min_sample_size:
                    mean_read_lengths.append(np.mean(feature_rlen_list))
                    std_read_lengths.append(max(1,np.nanstd(feature_rlen_list)))

    if len(read_lengths) == 0:
        read_lengths = None
    if len(mean_read_lengths) == 0:
        mean_read_lengths = None
    if len(std_read_lengths) == 0:
        std_read_lengths = None

    # this case warrants special attention
    if read_lengths is None:
        logger.warning(f"No reads found over {num_samples} random intervals...try increasing the number of samples or the interval length. Is your alignment single-end? Run with `paired_end=False`")

    if delete_random_features:
        try:
            os.remove(random_features_fname)
        except:
            logger.warning(f"Could not remove temporary file: {random_features_fname}")

    return np.median(read_lengths), np.array(mean_read_lengths), np.array(std_read_lengths)


def estimate_gwide_scale(bam_file: str, sizes_file: str,
                                effective_genome_size: float=None,
                                random_seed: int=42,
                                min_hq_reads: int=100,
                                threads: int=None) -> float:
    """The `estimate_gwide_scale` function estimates the scaling constant for alignment files such that read counts will be normalized to 1x coverage.
    """

    chroms_dict = get_chromsizes_dict(sizes_file=sizes_file)
    if effective_genome_size is None:
        effective_genome_size = np.sum([x for x in chroms_dict.values()])

    if threads is None or threads < 1:
        threads = max(1,(mp.cpu_count()//2)-1)

    random.seed(random_seed)
    scale_factor = 1.0
    reads = []
    with pysam.AlignmentFile(bam_file,'rb', threads=threads) as bam:
        total_mapped = bam.mapped
        for read in bam.fetch():
            if len(reads) >= min_hq_reads:
                break
            if read.mapping_quality > 10 and not read.is_unmapped and not read.is_duplicate and not read.is_secondary:
                reads.append(read.infer_query_length(always=False))
    scale_factor = effective_genome_size/(total_mapped*np.median(reads))
    return scale_factor


def estimate_scale(chromosome: str,
                bam_file: str,
                sizes_file: str,
                blacklist_file: str=None,
                paired_end: bool=True,
                threads :int = -1,
                rinterval_length: int=10000,
                num_rsamples: int=1000,
                delete_random_features: bool=True,
                min_rsample_size: int=25,
                random_seed: int=42,
                use_ivw: bool=True,
                use_medmean: bool=False) -> float:
    """The `estimate_scale` function estimates the scaling factor for a given chromosome and BAM file such that the read count is normalized to 1x coverage.
    
    :param chromosome: Chromosome name.
    :param bam_file: Path to BAM file.
    :param sizes_file: Path to sizes file.
    :param blacklist_file: Path to blacklist file.
    :param paired_end: Whether the alignment is paired-end.
    :param threads: Number of threads to use for opening BAM file.
    :param rinterval_length: Length of random intervals over which read lengths are estimated.
    :param num_rsamples: Number of random intervals to sample.
    :param delete_random_features: Whether to delete temporary random features file.
    :param min_rsample_size: Minimum number of reads observed in a given random interval for use in estimating read length and its uncertainty.
    :param random_seed: Random seed for sampling.
    :param use_ivw: Whether to use inverse-variance weighting for estimating read length.
    :param use_medmean: Whether to use median of mean read lengths over random features for estimating read length.
    :param effective_genome_size: Effective genome size to use for scaling (optional).
    :return: Scaling factor.

    """
    
    median_rlen_, mean_ests, std_ests = get_rlen_stats(chromosome, bam_file, sizes_file, blacklist_file, paired_end, threads, rinterval_length, num_rsamples, delete_random_features, min_rsample_size, random_seed)
    chrom_size_ = get_chromsizes_dict(sizes_file=sizes_file)[chromosome]

    # compute number of total mapped, primary reads in the chromosome
    total_mapped_reads = -1
    with pysam.AlignmentFile(bam_file,'rb', threads=threads) as bam:
        total_mapped_reads = bam.count(contig=chromosome, start=0, stop=chrom_size_, read_callback='all')

    rlen_estimate = None
    # compute IVW estimate of read length (wrt. random features in function: `get_rlen_stats()`)
    if use_ivw:
        try:
            rlen_estimate = np.sum(mean_ests/(std_ests**2))/np.sum(1/(std_ests**2))
        except Exception as zex:
            logger.warning(f"Could not compute IVW estimate of read length:\n{zex}\nUsing median read length instead.")
            rlen_estimate = None

    # compute read length estimate as median of mean read lengths
    elif use_medmean:
        try:
            rlen_estimate = np.median(mean_ests)
        except Exception as mex:
            logger.warning(f"Could not compute median of mean read lengths:\n{mex}\nUsing median read length instead.")
            rlen_estimate = None

    # use median read length
    if rlen_estimate is None:
        rlen_estimate = median_rlen_

    approx_span = total_mapped_reads*rlen_estimate
    scale_factor = chrom_size_/approx_span
    return scale_factor


def scale_treatment_control(treatment_bamfile: str, control_bamfile: str, threads: int=None) -> tuple:
    r"""Per convention, paired treatment/control samples are scaled *to each other* first.
    """
    if threads is None or threads < 1:
        threads = max(1,mp.cpu_count()//2)

    logger.info(f"Scaling treatment/control read counts for {treatment_bamfile} and {control_bamfile}...")

    treatment_readcounts = pysam.AlignmentFile(treatment_bamfile,'rb', threads=threads).mapped
    control_readcounts = pysam.AlignmentFile(control_bamfile, 'rb', threads=threads).mapped

    treatment_sf = 1.0
    control_sf = 1.0
    if treatment_readcounts < control_readcounts:
        control_sf = treatment_readcounts/control_readcounts
        treatment_sf = 1.0
    else:
        treatment_sf = control_readcounts/treatment_readcounts
        control_sf = 1.0
    return treatment_sf, control_sf


def get_readtrack(chromosome: str,
                bam_file: str,
                sizes_file: str,
                start: int=None,
                end: int=None,
                step: int=50,
                paired_end: bool=True,
                min_mapq: float=0.0,
                threads: int=-1,
                count_both: bool=True) -> tuple:
    r"""The `get_readtrack` function computes the 'read track' for a given chromosome and BAM file.
    
    Over intervals :math:`i=1,2,\ldots,n` the corresponding values in the length :math:`n` read track measures, loosely,
    the number of sequence alignments overlapping the interval. 

    :param chromosome: Chromosome name.
    :param bam_file: Path to BAM file.
    :param sizes_file: Path to sizes file.
    :param paired_end: Whether alignments are paired-end.
    :param min_mapq: Minimum mapping quality.
    :param blacklist_file: Path to blacklist file.
    :param threads: Number of threads to use for opening BAM file.
    :param count_both: Whether to count both reads in a pair with a +1. If False, each read in the pair is counted as 0.5.
    :return: Tuple of intervals (`np.ndarray`) and the read track (`np.ndarray`).

    """

    if threads < 0:
        threads = max(1,mp.cpu_count()//2)
    start = start if start is not None else 0
    start = start + (start % step)
    end = end if end is not None else get_chromsizes_dict(sizes_file=sizes_file)[chromosome]
    end = end - (end % step)

    if end <= start:
        raise ValueError(f"End must be greater than start: {end} <= {start}")

    intervals = np.arange(start, end + step, step)
    with pysam.AlignmentFile(bam_file,'rb', threads=threads) as bam:
        
        readtrack = np.zeros(len(intervals), dtype=float)

        for read in bam.fetch(chromosome, start, end, multiple_iterators=False):

            if paired_end and (not read.is_proper_pair or read.is_secondary or read.is_unmapped or read.is_duplicate or read.is_qcfail or read.is_supplementary or read.mapping_quality < min_mapq):
                continue
            if not paired_end and (read.is_secondary or read.is_unmapped or read.is_duplicate or read.is_qcfail or read.is_supplementary or read.mapping_quality < min_mapq):
                continue
            if read.reference_length < step:
                start_read = max(start, (read.reference_start - (read.reference_start % step)))
                idx = np.searchsorted(intervals, start_read, side='left')
                if paired_end and not count_both:
                    readtrack[max(0,idx-1)] += 0.5
                else:
                    readtrack[max(0,idx-1)] += 1.0
            else:
                start_read = max(start, (read.reference_start - (read.reference_start % step)))
                end_read = min(end, (read.reference_end - (read.reference_end % step)))
                idx_a = np.searchsorted(intervals, start_read, side='left')
                idx_b = np.searchsorted(intervals, end_read, side='left')
                if paired_end and not count_both:
                    readtrack[max(0,idx_a-1):min(len(intervals),idx_b)] += 0.5
                else:
                    readtrack[max(0,idx_a-1):min(len(intervals),idx_b)] += 1

    return intervals, readtrack


def get_readtrack_mp(
    bam_files: List[str],
    chromosome: str,
    sizes_file: str,
    start: int = None,
    end: int = None,
    step: int = 50,
    paired_end: bool = True,
    min_mapq: float = 0,
    threads: int = None,
    n_processes: int = None,
    count_both: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Wrapper for parallelizing read track computation across multiple BAM files.

    :param bam_files: List of BAM file paths.
    :param chromosome: Chromosome to fetch reads from.
    :param sizes_file: Path to sizes file
    :param norm_counts: Whether to normalize counts to 1x.
    :param paired_end: Whether alignments are paired-end.
    :param threads: Threads for reading BAM in each process.
    :param n_processes: Number of parallel processes.
    :return: List of tuples (intervals, readtrack) for each BAM file.

    """

    if n_processes is None or n_processes < 1:
        n_processes = max(1, (mp.cpu_count()//2) - 1)
    if threads is None or threads < 1:
        threads = max(1, (mp.cpu_count()//2) - 1)

    job_args = [
        (chromosome, bam, sizes_file, start, end, step, paired_end, min_mapq, threads, count_both)
        for bam in bam_files
    ]

    # Use multiprocessing Pool
    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(get_readtrack, job_args)

    return results


def get_munc_track_mp(chromosome: str,
                intervals: np.ndarray,
                chrom_matrix: np.ndarray,
                sparsemap: dict,
                munc_min: float=0.05,
                munc_max: float=500,
                munc_smooth: bool=True,
                munc_smooth_bp: int=500,
                munc_local_weight: float=0.333,
                munc_global_weight: float=0.667,
                conservative_munc: bool=False,
                n_processes: int=None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Wrapper for parallelizing observation noise track computation across multiple samples' read data tracks.
    """
    if n_processes is None or n_processes < 1:
        n_processes = max(1,(mp.cpu_count()//2) - 1)

    num_rows = get_shape(chrom_matrix)[0]
    job_args = [
        (chromosome, intervals, chrom_matrix[i], sparsemap, munc_min, munc_max, munc_smooth, munc_smooth_bp, munc_local_weight, munc_global_weight, conservative_munc) for i in range(num_rows)]

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(munc_track, job_args)
    return results


def find_proximal_features(chromosome: str,
                           intervals: np.ndarray,
                           sparsebed: str,
                           k: int=25) -> np.ndarray:
    r"""The function `find_proximal_features` identifies proximal genomic regions in `sparsebed` for each interval in a given chromosome.
    """
    
    step = get_step(intervals) 
    bed_starts_df = pd.read_csv(sparsebed, sep='\t', header=None, usecols=[0,1], names=['chrom','start'],
                                dtype={'chrom':str, 'start':int}, engine='c')
    bed_starts = np.array(bed_starts_df[bed_starts_df['chrom'] == chromosome].reset_index(drop=True)['start'], dtype=int)
    prox_map = {}
    prev_proxstart = -1
    for i, interval in enumerate(intervals):
        proximal_start = np.searchsorted(bed_starts, interval, side='right') - 1
        if proximal_start == prev_proxstart and i > 0:
            proximal = prox_map[intervals[i-1]]
            prox_map.update({interval: proximal})
            continue
        llim = max(0,proximal_start - k//2)
        ulim = min(proximal_start + k//2, len(bed_starts))
        prox_map.update({interval: bed_starts[llim:ulim]})
        prev_proxstart = proximal_start
    return prox_map


def munc_track(chromosome: str,
            intervals: np.ndarray,
            vals: np.ndarray,
            sparsemap: dict,
            munc_min: float=0.05,
            munc_max: float=500,
            munc_smooth: bool=True,
            munc_smooth_bp: int=1000,
            munc_local_weight: float=0.333,
            munc_global_weight: float=0.667,
            conservative_munc: bool=False
            ) -> tuple:
    """The function `munc_track` computes the observation noise track for a given chromosome and read track using nearby, 'sparse' genomic regions at each interval.

    :param chromosome: Chromosome name.
    :param intervals: Numpy array of intervals.
    :param vals: Numpy array of values (generated from `get_readtrack` in the default implementation).
    :param sparsemap: Dictionary of sparse segments.
    :param munc_min: Minimum observation noise value.
    :param munc_max: Maximum observation noise value.
    :param munc_smooth: Whether to smooth the observation noise track.
    :param munc_smooth_bp: Window size for smoothing.
    :param munc_local_weight: Local weight for observation noise.
    :param munc_global_weight: Global weight for observation noise.
    :return: Tuple of intervals (`np.ndarray`) and observation noise track (`np.ndarray`).
    """

    step = get_step(intervals)
    n = match_lengths(intervals, vals)
    if conservative_munc:
        break_len = max(2, n//100)
        if break_len % 2 == 0:
            break_len += 1
        vals = vals - signal.savgol_filter(vals, break_len, 2)
    vals_sq = np.square(vals)

    munc_track_ = np.zeros(n)
    prev = None
    prev_match_arr = np.zeros(n)
    for i, interval in enumerate(intervals):
        if i > 0 and hashlib.md5(sparsemap[interval].tobytes()).hexdigest() == prev:
            munc_track_[i] = munc_track_[i-1]
            prev_match_arr[i] = 1
            continue

        val_idx = np.searchsorted(intervals, sparsemap[interval], side='right') - 1
        if conservative_munc:
            munc_track_[i] = np.median(vals_sq[val_idx])
        else:
            munc_track_[i] = stats.trim_mean(vals_sq[val_idx], 0.005)
        # for faster comparisons later on
        prev = hashlib.md5(sparsemap[interval].tobytes()).hexdigest()

    munc_track_ = np.clip(munc_track_, munc_min, munc_max)
    if munc_smooth:
        window_steps = 2*(munc_smooth_bp // (2*step)) + 1
        munc_track_ = np.convolve(munc_track_, np.ones(window_steps)/window_steps, mode='same')

    munc_track_final = np.mean(munc_track_)*munc_global_weight + munc_track_*munc_local_weight
    return intervals, munc_track_final, prev_match_arr


def detrend_track(intervals: np.ndarray, vals: np.ndarray,
                  degree: int=None, percentile: int=None,
                  window_bp: int=None,
                  lbound: float=None,
                  ubound: float=None) -> tuple:
    r"""The function `detrend_track` models background dynamically with a bounded low-pass filter (Savitzky-Golay or general-percentile) and removes it.
    
    .. note:: 
        If both `degree` and `percentile` are None, a median filter is applied.

    :param intervals: Numpy array of intervals.
    :param vals: Numpy array of values.
    :param degree: Degree of polynomial for Savitzky-Golay filter.
    :param percentile: Percentile for general percentile filter.
    :param window_bp: Window size for filtering.
    :param lbound: Lower bound for values.
    :param ubound: Upper bound for values.
    :return: Tuple of intervals and detrended values.

    """

    step = get_step(intervals)
    n = match_lengths(intervals, vals)
    vals_ = vals.copy()

    if degree is not None and window_bp is not None:
        detrend_window_steps = 2*(window_bp // (2*step)) + 1
        filtered_vals = signal.savgol_filter(vals_, detrend_window_steps, degree)
    elif percentile is not None and window_bp is not None:
        detrend_window_steps = 2*(window_bp // (2*step)) + 1
        filtered_vals = ndimage.percentile_filter(vals_, percentile, detrend_window_steps)
    if degree is None and percentile is None and window_bp is not None:
        detrend_window_steps = 2*(window_bp // (2*step)) + 1
        filtered_vals = ndimage.median_filter(vals_, detrend_window_steps)

    detrended_vals = vals_ - filtered_vals

    if lbound is not None and ubound is not None:
        detrended_vals = np.clip(detrended_vals, lbound, ubound)

    return intervals, detrended_vals


def get_chromosome_matrix(chromosome: str,
                        bam_files: list,
                        sizes_file: str,
                        sparsebed: str,
                        step: int=50,
                        norm_counts: bool=True,
                        norm_gwide: bool=False,
                        gwide_scales: np.ndarray=None,
                        paired_end: bool=True,
                        exclude_flag: int=3840,
                        min_mapq: int=0,
                        blacklist_file: str=None,
                        threads: int=None,
                        count_both: bool=True,
                        backshift: int=None,
                        munc_min: float=0.05,
                        munc_max: float=500,
                        munc_smooth: bool=True,
                        munc_smooth_bp: int=500,
                        munc_local_weight: float=0.333,
                        munc_global_weight: float=0.667,
                        munc_k: int=25,
                        n_processes: int=None,
                        detrend_degree: int=None,
                        detrend_percentile: int=None,
                        detrend_window_bp: int=None,
                        detrend_lbound: float=None,
                        detrend_ubound: float=None,
                        save_matrix:  bool=False,
                        experiment_id: int=None,
                        control_files: list=None,
                        log_scale: bool=False,
                        log_pc: int=1,
                        no_sparsebed=False,
                        csparse_aggr_percentile=75,
                        csparse_wlen=51,
                        csparse_pdegree=3,
                        csparse_min_peak_len=10,
                        csparse_min_sparse_len=10,
                        csparse_min_dist=50,
                        csparse_max_features: int=5000,
                        csparse_min_prom_prop: float=0.05):

    r"""The function `get_chromosome_matrix` computes the matrix of read counts (:math:`m \times n` for :math:`m` samples) for a given chromosome and list of paths to samples' sequence alignment files (BAM format).

    :param chromosome: Chromosome name.
    :param bam_files: List of BAM file paths.
    :param sizes_file: Path to sizes file.
    :param sparsebed: Path to sparse BED file.
    :param step: Step size for intervals.
    :param norm_counts: Whether to normalize counts (per chromosome 1x).
    :param norm_gwide: Whether to normalize counts w.r.t. whole genome.
    :param gwide_scales: Numpy array of scaling factors for each BAM file.
    :param paired_end: Whether alignments are paired-end.
    :param exclude_flag: SAM flag to exclude reads.
    :param min_mapq: Minimum mapping quality.
    :param blacklist_file: Path to blacklist file.
    :param threads: Number of threads for reading BAM files.
    :param count_both: Whether to count both reads in a pair with a +1. If False, each read in the pair is counted as 0.5.
    :param backshift: Number of base pairs to backshift the last read.
    :param munc_min: Minimum observation noise value.
    :param munc_max: Maximum observation noise value.
    :param munc_smooth: Whether to smooth the observation noise track.
    :param munc_smooth_bp: Window size for smoothing.
    :param munc_local_weight: Local weight for observation noise.
    :param munc_global_weight: Global weight for observation noise.
    :param munc_k: Number of proximal features to consider.
    :param n_processes: Number of parallel processes.
    :param detrend_degree: Degree for polynomial detrending. Mutually exclusive with `detrend_percentile`.
    :param detrend_percentile: Percentile for percentile detrending. Mutually exclusive with `detrend_degree`.
    :param detrend_window_bp: Window size for detrending.
    :param detrend_lbound: Lower bound for detrended values.
    :param detrend_ubound: Upper bound for detrended values.
    :return: Tuple of intervals (`np.ndarray`), read matrix (`np.ndarray`), and observation noise matrix (`np.ndarray`).
    
    :seealso: `get_readtrack()`, `get_munc_track()`, `find_proximal_features()`, `detrend_track()`

    """

    if n_processes is None or n_processes < 1:
        n_processes = max(1,(mp.cpu_count()//2) - 1)
    if threads is None or threads < 1:
        threads = max(1,(mp.cpu_count()//2) - 1)

    first_reads = []
    last_reads = []
    
    for bam_file in bam_files:
        wrap_index(bam_file=bam_file)
        first_reads.append(get_first_read(chromosome, bam_file, sizes_file, exclude_flag=exclude_flag, min_mapq=min_mapq, step=step))
        last_reads.append(get_last_read(chromosome, bam_file, sizes_file, exclude_flag=exclude_flag, min_mapq=min_mapq, step=step, backshift=backshift))
    # maximally inclusive interval ranges
    start = min(first_reads)
    stop = max(last_reads)
    intervals = np.arange(start, stop + step, step)
    par_results_readtrack = get_readtrack_mp(
        bam_files, 
        chromosome, 
        sizes_file,
        start=intervals[0],
        end=intervals[-1],
        step=step,
        paired_end=paired_end, 
        min_mapq=min_mapq,
        threads=threads, 
        n_processes=n_processes,
        count_both=count_both)

    chrom_matrix = np.zeros((len(bam_files), len(intervals)))
    for i, result in enumerate(par_results_readtrack):
        chrom_matrix[i] = result[1]

    if control_files is not None and len(control_files) > 0:
        if len(control_files) != len(bam_files):
            raise ValueError("Number of control files must match number of treatment files.")
        logger.info(f'Computing treatment read tracks: {chromosome}...')
        par_controls_readtrack = get_readtrack_mp(
            control_files, 
            chromosome, 
            sizes_file,
            start=intervals[0],
            end=intervals[-1],
            step=step,
            paired_end=paired_end, 
            min_mapq=min_mapq,
            threads=threads, 
            n_processes=n_processes,
            count_both=count_both)
        for i, result in enumerate(par_controls_readtrack):
            treatment_scale_factor, control_scale_factor = scale_treatment_control(bam_files[i], control_files[i])
            rtrack = result[1]
            if log_scale:
                chrom_matrix[i] = np.log(chrom_matrix[i]*treatment_scale_factor + log_pc)
                - np.log(rtrack*control_scale_factor + log_pc)
            else:
                chrom_matrix[i] = (chrom_matrix[i]*treatment_scale_factor) - (rtrack*control_scale_factor)

    if norm_counts and not norm_gwide:
        for i in range(get_shape(chrom_matrix)[0]):
            track_sf = estimate_scale(chromosome, bam_files[i], sizes_file, blacklist_file=blacklist_file, paired_end=paired_end)
            chrom_matrix[i] = chrom_matrix[i]*track_sf

    if norm_gwide:
            for i in range(get_shape(chrom_matrix)[0]):
                if gwide_scales is None or len(gwide_scales) != len(bam_files):
                    track_sf = estimate_gwide_scale(bam_files[i], sizes_file, threads=threads)
                else:
                    # these need not be computed for every call to `get_chromosome_matrix`
                    # ...compute in main and pass as argument
                    track_sf = gwide_scales[i]
                
                chrom_matrix[i] = chrom_matrix[i]*track_sf
    munc_matrix = np.zeros((len(bam_files), len(intervals)))
    conservative_munc = False
    sparse_fname = None
    if no_sparsebed:
        sparse_fname = f'tmp_{chromosome}_sparse_{int(uuid.uuid4().hex[:5], base=16)}.bed'
        get_csparse(chromosome, intervals, chrom_matrix,
                                aggr_percentile=csparse_aggr_percentile,
                                wlen=csparse_wlen, pdegree=csparse_pdegree,
                                min_peak_len=csparse_min_peak_len,
                                min_sparse_len=csparse_min_sparse_len,
                                min_dist=csparse_min_dist, min_prom_prop=csparse_min_prom_prop,
                                bed=sparse_fname)
        conservative_munc = True
        if pbt.BedTool(sparse_fname).count() > csparse_max_features:
            sparsebed_selection = pbt.BedTool(sparse_fname).sort(chrThenScoreD=True)
            try:
                os.remove(sparse_fname)
            except:
                logger.warning(f"Could not remove temporary file: {sparse_fname}")
            sparse_fname = pbt.BedTool(sparsebed_selection.head(csparse_max_features, as_string=True), from_string=True).sort().saveas(sparse_fname + '.head.bed')
        sparsebed = sparse_fname 
    proximal_features = find_proximal_features(chromosome, intervals, sparsebed, k=munc_k)
    if no_sparsebed and sparse_fname is not None and os.path.exists(sparse_fname):
        try:
            os.remove(sparse_fname)
        except:
            logger.warning(f"Could not remove temporary file: {sparse_fname}")
    logger.info(f'Computing observation noise tracks: {chromosome}...')
    par_results_munctrack = get_munc_track_mp(
        chromosome, 
        intervals, 
        chrom_matrix, 
        proximal_features,
        munc_min=munc_min, 
        munc_max=munc_max, 
        munc_smooth=munc_smooth, 
        munc_smooth_bp=munc_smooth_bp, 
        munc_local_weight=munc_local_weight, 
        munc_global_weight=munc_global_weight,
        conservative_munc=conservative_munc)
    prev_match_arr = None
    for k, result in enumerate(par_results_munctrack):
        if k == 0:
            _, munc_matrix[k], prev_match_arr = result
        else:
            _, munc_matrix[k], _ = result
    if prev_match_arr is None:
        prev_match_arr = np.zeros(len(intervals))
    prev_match_arr[0] = 0

    if detrend_lbound is None:
        detrend_lbound = np.min(chrom_matrix) - 10e4
    if detrend_ubound is None:
        detrend_ubound = np.max(chrom_matrix) + 10e4
    if any ([detrend_degree is not None, detrend_percentile is not None, detrend_window_bp is not None]):
        for i in range(get_shape(chrom_matrix)[0]):
            logger.info(f'Modeling Dynamic Background with Low-Pass Filter: track {i+1}/{len(bam_files)}:\n{chromosome}, degree={detrend_degree}, percentile={detrend_percentile}, window_bp={detrend_window_bp}\n')
            intervals, chrom_matrix[i] = detrend_track(intervals, chrom_matrix[i], degree=detrend_degree, percentile=detrend_percentile, window_bp=detrend_window_bp, lbound=detrend_lbound, ubound=detrend_ubound)

    if save_matrix:
        if experiment_id is None:
            npz_fname = f'consenrich_{chromosome}_data_{datetime.now.strftime("%m%d%Y_%H_%M_%S")}.npz'
        else:
            npz_fname = f'consenrich_{chromosome}_data_{experiment_id}.npz'
        try:
            if os.path.exists(npz_fname):
                logger.warning(f"Overwriting existing data file: {npz_fname}")
                os.remove(npz_fname)
            np.savez_compressed(npz_fname, chromosome=chromosome, intervals=intervals, chrom_matrix=chrom_matrix, munc_matrix=munc_matrix)
            logger.info(f'Saved data to {npz_fname}. Dict keys: chromosome, intervals, chrom_matrix, munc_matrix')
        except:
            logger.warning(f'Could not save data to {npz_fname}')

    return intervals, chrom_matrix, munc_matrix, prev_match_arr


def mask_blacklisted(chromosome: str, intervals: np.ndarray, blacklist_file: str) -> np.ndarray:
    r"""Find intervals overlapping problematic genomic regions (e.g., blacklisted regions).
    """
    step = get_step(intervals)
    if blacklist_file is None:
        return np.array([])
    blacklist = pbt.BedTool(blacklist_file).filter(lambda x: x.chrom == chromosome)
    # create temp bed file of intervals
    temp_bed = pbt.BedTool('\n'.join([f'{chromosome}\t{intervals[i]}\t{intervals[i]+step}\tpeak_{i}\t{i}\t.' for i in range(len(intervals))]), from_string=True)
    # intersect temp bed with blacklist
    blacklisted = temp_bed.intersect(b=blacklist, wa=True, wb=False)
    mask_idx = np.zeros(len(intervals), dtype=int)
    for interval in blacklisted:
        mask_idx[int(interval.score)] = 1
    return mask_idx


def backward_pass(xvec_forward, Pmat_forward, Qmat_forward, Fmat):
    r"""The function `backward_pass` computes the backward pass of Consenrich

    :param xvec_forward: A NumPy array of the forward-pass state vectors
    :param Pmat_forward: A NumPy array of the forward-pass process noise covariance matrices
    :param Qmat_forward: A NumPy array of the forward-pass process noise covariance matrices
    :param Fmat: The process model (state-transition) matrix
    :return: Tuple of the backward-pass state vectors and process noise covariance matrices

    :seealso: `run_consenrich`

    """
    
    n = xvec_forward.shape[0]
    x_smooth = np.zeros_like(xvec_forward)
    P_smooth = np.zeros_like(Pmat_forward)
    x_smooth[-1] = xvec_forward[-1]
    P_smooth[-1] = Pmat_forward[-1]
    
    for k in range(n - 2, -1, -1):
        if k % ((n-2)//10) == 0:
            logger.info(f'Processing interval {k+1}/{n}')
        PFT = Pmat_forward[k] @ Fmat.T
        x_pred = Fmat @ xvec_forward[k]
        P_pred = Fmat @ PFT + Qmat_forward[k]
        G_k = PFT @ np.linalg.inv(P_pred)
        x_smooth[k] = xvec_forward[k] + G_k @ (x_smooth[k+1] - x_pred)
        P_smooth[k] = Pmat_forward[k] + G_k @ (P_smooth[k+1] - P_pred) @ G_k.T
    return x_smooth, P_smooth


def run_consenrich(chromosome, bam_files, sizes_file, blacklist_file, sparsebed,
                   step=50, norm_counts=True, norm_gwide=False, gwide_scales=None, paired_end=True,
                   exclude_flag=3840, min_mapq=0, threads=None, count_both=True,
                   backshift=None, munc_min=0.05, munc_max=500, munc_smooth=True,
                   munc_smooth_bp=500, munc_local_weight=0.333, munc_global_weight=0.667,
                   munc_k=25, munc_const=1.0, prunc_min=0.05, prunc_max=500.0,
                   n_processes=None, xvec=None, Pmat=None, Qmat=None, Fmat=None,
                   Hmat=None, Rmat=None, delta=1.0, Dstat_thresh=2.0, Dstat_scale=10.0,
                   Dstat_pc=2.0, state_lowerlim=None, state_upperlim=None, Qmat_offdiag=None,
                   joseph=True, detrend_degree=None, detrend_percentile=None, detrend_window_bp=None,
                   detrend_lbound=None, detrend_ubound=None,
                   save_matrix=True, experiment_id=None,
                   control_files=None, log_scale=False, log_pc=1.0,
                   no_sparsebed=False, csparse_aggr_percentile=75, csparse_wlen=51,
                   csparse_pdegree=3, csparse_min_peak_len=10, csparse_min_sparse_len=10,
                   csparse_min_dist=50, csparse_max_features=5000, csparse_min_prom_prop=0.05,
                   ignore_blacklist=True):
    r"""Run Consenrich on an individual chromosome.
    
    :param chromosome: Chromosome to run Consenrich on.
    :param bam_files: List of BAM files.
    :param sizes_file: Path to sizes file.
    :param blacklist_file: Path to blacklist file.
    :param sparsebed: Path to sparsebed file.
    :param step: Step size for intervals.
    :param norm_counts: `True` if normalizing counts to 1x.
    :param paired_end: `True` if using paired-end reads.
    :param exclude_flag: Exclude flag -- discard alignments with this flag. Default is 3840.
    :param min_mapq: Minimum mapping quality. Default is 0.
    :param threads: Number of threads to use. Default is half of CPU count.
    :param count_both: `True` if counting both reads as +1. Otherwise count each as 0.50.
    :param backshift: Backshift when searching for the last read.
    :param munc_min: Minimum observation noise allowed for a sample and region. Default is 0.10.
    :param munc_max: Maximum observation noise allowed for a sample and region. Default is 1000.
    :param munc_smooth: `True` if smoothing observation noise across intervals.
    :param munc_smooth_bp: Number of base pairs to smooth observation noise over.
    :param munc_local_weight: Weight of the 'local' component used to compute observation noise.
    :param munc_global_weight: Weight of the 'global' component used to compute observation noise.
    :param munc_k: Number of proximal features to consider when computing observation noise.
    :param munc_const: observation noise constant. Default is 1.0.
    :param prunc_min: Minimum process noise allowed. Default is 0.05.
    :param prunc_max: Maximum process noise allowed. Default is 500.0
    :param n_processes: Number of parallel processes to use when counting reads, computing observation noise, etc.
    :param xvec: State vector. Only modify if wanting to alter the model itself.
    :param Pmat: Process noise covariance matrix. Only modify if wanting to alter the model itself.
    :param Qmat: observation noise covariance matrix. Only modify if wanting to alter the model itself.
    :param Fmat: Process model matrix. Only modify if wanting to alter the model itself.
    :param Hmat: Observation model matrix. Only modify if wanting to alter the model itself.
    :param Rmat: observation noise covariance matrix.
    :param delta: Distance to propagate state and covariance. Default is unit 1.0.
    :param Dstat_thresh: D-statistic threshold. Default is 2.0.
    :param state_lowerlim: Lower limit of state vector. Default is minimum of 0 and minimum of data.
    :param state_upperlim: Upper limit of state vector. Default is maximum of data.
    :param Qmat_offdiag: Off-diagonal elements of Q matrix. Default is half of `prunc_min`.
    :param joseph: `True` if using Joseph form for covariance update.
    :param detrend_degree: Degree of polynomial for detrending. Mutually exclusive with `detrend_percentile` (Percentile filter)
    :param detrend_percentile: Percentile for detrending. Mutually exclusive with `detrend_degree` (Savitzky-Golay)
    :param detrend_window_bp: Window size for detrending.
    :param save_matrix: Save count and munc matrices to .npz file
    :param experiment_id: Experiment ID.

    """
    if n_processes is None or n_processes < 1:
        n_processes = max(1,(mp.cpu_count()//2) - 1)
    if threads is None or threads < 1:
        threads = max(1,(mp.cpu_count()//2) - 1)
        
    logger.info(f'Running Consenrich on chromosome {chromosome} with {len(bam_files)} BAM files and {n_processes} processes')

    # First compute read tracks and observation noise tracks
    intervals, chrom_matrix, munc_matrix, prev_match_arr = get_chromosome_matrix(chromosome, bam_files, sizes_file, sparsebed, step, norm_counts, norm_gwide, gwide_scales, paired_end, exclude_flag, min_mapq, blacklist_file, threads, count_both, backshift, munc_min, munc_max, munc_smooth, munc_smooth_bp, munc_local_weight, munc_global_weight, munc_k, n_processes, detrend_degree, detrend_percentile, detrend_window_bp, detrend_lbound, detrend_ubound, save_matrix, experiment_id, control_files, log_scale, log_pc, no_sparsebed, csparse_aggr_percentile, csparse_wlen, csparse_pdegree, csparse_min_peak_len, csparse_min_sparse_len, csparse_min_dist, csparse_max_features, csparse_min_prom_prop)

    if state_lowerlim is None:
        # if not set, use smallest value in data or 0
        state_lowerlim = min(0,np.min(chrom_matrix))
    if state_upperlim is None:
        # if not set, use largest value in data
        state_upperlim = np.max(chrom_matrix)

    if Qmat_offdiag is None and prunc_min is not None:
        # add small value to off-diagonal elements of the process noise cov. `Qmat`
        Qmat_offdiag = prunc_min/2.0

    # Estimates over blacklisted regions set to lower-limit by default
    if ignore_blacklist:
        blacklisted_idx = mask_blacklisted(chromosome, intervals, blacklist_file)
    else:
        blacklisted_idx = np.zeros(len(intervals), dtype=int)

    # Number of genomic intervals `i=1,2,...,n`
    n = len(intervals)
    # Number of samples `j=1,2,...,m`
    m = len(bam_files)

    # -- `xvec` is the state vector we are tracking.
    if xvec is None:
        # -- `xvec` is the state vector we are tracking.
        # -- The first state variable is of direct interest and initialized directly
        # from the median of samples' data
        # -- The second state variable is treated as the 'trend' of the first state variable
        # at each interval `i=1,2,...,n` and is initialized as zero. It is useful for recovering
        # true states over noisy data and can improve spatial resolution.
        
        xvec = np.array([np.median(chrom_matrix[:,0]), 0.0])

    if Pmat is None:
        # -- `Pmat` is the state estimate covariance matrix
        # This is used to represent the uncertainty in the
        # state estimate at each interval `i=1,2,...,n` that
        # is due to the process model -- NOT the data/observation model
        Pmat = np.eye(2)*100

    if Fmat is None:
        # `Fmat` defines the *process model*: simple recursive linear model that projects
        # forward the previous state along an estimated trajectory
        Fmat = np.array([[1.0, delta], [0.0, 1.0]])

    if Qmat is None:
        # -- `Qmat` is the process noise covariance matrix
        # This is used to represent the uncertainty in the process model itself
        # We can't expect to perfectly model the system, so this matrix is used
        # to ensure the process model does not dominate the algorithm and that
        # data is considered in the state estimate
        Qmat = np.eye(2)*prunc_min + np.ones((2,2))*Qmat_offdiag
    elif Qmat is not None and prunc_min is None:
        prunc_min = np.trace(Qmat)/2.0

    if Hmat is None:
        # `Hmat` defines the observation model: here, it just indicates that
        # we are observing data for the first state variable but NOT the
        # second state variable (which is the 'trend' of the first state variable)
        # -- Accordingly, the first column is all ones, second column is all zeros
        Hmat = np.zeros((m,2))
        Hmat[:,0] = 1.0
    Hmat_transpose = Hmat.T


    if Rmat is None and munc_matrix is None:
        # -- `Rmat` is the observation noise covariance matrix
        # where observation noise is modeled in Consenrich
        # as a sample-and-region-specific process (by default)
        # so that this covariance matrix is defined at each interval
        # i=1,2,...,n and is m x m.
        
        # But, if `munc_const` is set, we use a constant value for all samples
        # and regions. This is useful for testing/debugging or if Consenrich
        # is being used over smaller genomes or genomic subregions where the
        # sample-and-region-specific noise model is not as useful.
        Rmat = np.eye(m)*munc_const
        Rinv = np.eye(m)/munc_const
        Router_prod = np.outer(np.ones(m)/munc_const, np.ones(m)/munc_const)
        prev_match_arr = np.ones(n)

    Imat = np.eye(2)
    xvec_forward = np.zeros((n,2))
    Pmat_forward = np.zeros((n,2,2))
    Qmat_forward = np.zeros((n,2,2))
    #residuals_ivw = np.zeros(n)
    ZmatT = chrom_matrix.T
    RmatT = munc_matrix.T
    logger.info(f'Starting forward pass i=1, 2, ... --> n-1, n')
    for i in range(n):
        if i % (n//10) == 0:
            logger.info(f'Processing interval {i+1}/{n}')

        # `zvec` is the observed data vector at each interval i=1,2,...,n
        # it is m x 1, where m is the number of samples.
        zvec = ZmatT[i]

        if prev_match_arr[i] == 0:
            mmi_vec = RmatT[i]
            mminv_vec = 1/mmi_vec   
            Rmat = np.diag(mmi_vec)
            Rinv = np.diag(mminv_vec)
            Rinv_trace = np.sum(mminv_vec)
            Router_prod = np.outer(mminv_vec, mminv_vec)

        # -- Use `Fmat` recursively to compute a priori estimate of state at interval `i`
        # given the previous state estimate at interval `i-1`
        xvec = Fmat@xvec
        # -- Also use `Fmat` to also propagate forward the previous state's uncertainty
        # as an a priori estimate of the current state's uncertainty
        Pmat = Fmat@Pmat@Fmat.T + Qmat

        # This comes up a lot, so precompute once per iteration
        PHT = Pmat@Hmat_transpose

        # -- Compute the residual vector `yvec` at each interval `i`
        # This is the difference between the observed data and the
        # a priori state estimate
        # For future reference, if using a different observation model, use *this*:
        # yvec = zvec - Hmat@xvec
        yvec = zvec - xvec[0]

        # Compute in Sherman-Morrison form
        Emat_inv = Rinv - (Pmat[0,0]*Router_prod)/(1 + Pmat[0,0]*(Rinv_trace))

        # -- Compute -median- of the squared, uncertainty-weighted
        # residuals (D-statistic) $D_i$ at each interval `i=1,2,...,n`
        #   If the result indicates the system is mismatched to the
        #   data beyond what could be reasonably expected given the 
        #   current uncertainty estimate, we scale up the process noise `Qmat`.
        # -- Instead of the quadratic form `y.T@Emat_inv@y`, we take the
        # median of the squared/ivw residuals for robustness and invariance to `m`
        Dstat_ = np.median(np.square(yvec) * Emat_inv.diagonal())

        # -- `Kmat` gain is 2 x m (states x observations) and will determine
        # to what extent each sample's residual can affect the a priori estimate
        # -- Loosely, if the process model estimate (from `Fmat`) is more (less) reliable given
        #  Pmat` than the observation model (from `Hmat`) given `Rmat[j,j]` j=1,...,m,
        # then the jth residual will affect the state estimate less (more).
        Kmat = PHT@Emat_inv
        IKH = Imat - Kmat@Hmat # precompute

        # -- Now we compute the a posteriori estimates of the state and its uncertainty
        # -- The (default) covariance update is in Joseph form.
        #    This is a more flexible, numerically stable way to update the covariance matrix
        #    but is less intuitive and potentially less efficient than the more common KF form.

        xvec = xvec + Kmat@yvec
        if joseph:
            Pmat = (IKH)@Pmat@(IKH).T + Kmat@Rmat@Kmat.T
        else:
            Pmat = (Imat - Kmat@Hmat)@Pmat 

        # -- Adaptive process noise (APN) as discussed in manuscript
        current_process_noise = np.trace(Qmat)/2.0
        if Dstat_ > Dstat_thresh and current_process_noise < prunc_max:
            adaptive_increase = np.sqrt(Dstat_scale*(Dstat_ - Dstat_thresh) + Dstat_pc)
            Qmat = Qmat*adaptive_increase
        if Dstat_ <= Dstat_thresh and current_process_noise > prunc_min:
            adaptive_decrease = 1/(np.sqrt(Dstat_scale*(Dstat_thresh - Dstat_) + Dstat_pc))
            Qmat = Qmat*adaptive_decrease
        Qmat = np.clip(Qmat, prunc_min, prunc_max)

        xvec_forward[i] = xvec
        Pmat_forward[i] = Pmat
        Qmat_forward[i] = Qmat

    logger.info(f'Starting backward pass i=n, n-1, ... --> 2, 1')
    # -- Now we run the backward pass to retrospectively refine the
    # state estimates and further reduce their uncertainty, having knowledge
    # of the future states.
    xvec_smooth, Pmat_smooth = backward_pass(xvec_forward, Pmat_forward, Qmat_forward, Fmat)

    est_final = np.clip(xvec_smooth[:,0], state_lowerlim, state_upperlim)
    # set blacklisted regions' state estimates to lower limit
    est_final[blacklisted_idx == 1] = state_lowerlim
    Ptrace_final = np.round(np.array([np.trace(P) for P in Pmat_smooth]),3)
    Rtrace_final = np.round(np.sum(munc_matrix, axis=0), 3)
    residuals_ivw_final = np.zeros(n)
    for i in range(n):
        postsmooth_var = 1 / (munc_matrix[:,i] + Pmat_smooth[i,0,0]) # precompute
        postsmooth_res = np.dot(chrom_matrix[:,i] -  Hmat@xvec_smooth[i], postsmooth_var)
        residuals_ivw_final[i] = postsmooth_res / np.sum(postsmooth_var)
    logger.info('Done.')
    return intervals, est_final, Ptrace_final, Rtrace_final, residuals_ivw_final


def _parse_arguments(ID):
    r"""Parse CLI arguments for Consenrich.

    Arguments are parse either from command-line AND/OR a JSON config file.

    Example Config File (JSON):
    -----------------------------

    You can specify the arguments in a JSON file and pass it to the CLI using the `-f/--config_file` flag.

    .. code-block:: json
    
        {
            "bam_files": ["sample1.bam", "sample2.bam", "sample3.bam"],
            "control_files": ["control1.bam", "control2.bam", "control3.bam"],
            "genome": "hg38",
            "signal_bigwig": "output_signal.bw",
        }

    Each time Consenrich is run from `main()`, a unique ID/JSON-arg file is saved for future use.

    :param ID: Unique ID for the current run.
    :return: Namespace of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Consenrich CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=True, epilog="\n\nHomepage: https://github.com/nolan-h-hamilton/Consenrich\n\n")
    parser.add_argument(
        '-f', '--config_file',
        type=str,
        default=None,
        help='Path to a JSON config file containing arguments. '
             'Note: command-line arguments override config file settings.'
    )

    parser.add_argument('-t', '--bam_files', dest='bam_files', required=False, nargs='+',
                        help='Space-separated string of input BAM files.')
    parser.add_argument('-c', '--control_files', dest='control_files', nargs='+', default=[],
                        help='Space-separated string of control BAM files if applicable.')
    parser.add_argument('--sizes_file', default=None, help='Path to the chromosome sizes file.')
    parser.add_argument('--chroms', nargs='+', default=[], 
                        help='If not empty, only process chromosomes in this list.')
    parser.add_argument('--skip_chroms', nargs='+', default=[], help='List of chromosomes to skip.')
    parser.add_argument('--blacklist_file', default=None, help='Path to blacklist file.')
    parser.add_argument('--sparsebed', default=None, help='Path to sparsebed file.')
    parser.add_argument('--active_regions', default=None, help='Path to active regions file BED.')
    parser.add_argument('-g', '--genome', dest='genome', default=None, 
                        help='Convenience option. If supplied, use pre-packaged files for the given assembly [hg38, mm10, mm39, dm6].')
    parser.add_argument('--step', type=int, default=50, help='Step size for genomic intervals (default: 50bp).')
    parser.add_argument('--norm_gwide', '--use_1x_norm', action='store_true', dest='norm_gwide', 
                        help='If set, normalize counts to genome-wide read depth. Else this is done on a per-chromosome basis.')
    parser.add_argument('--no_norm_counts', action='store_true', help='If set, skip normalizing counts')
    parser.add_argument('--paired_end', action='store_true', default=True)
    parser.add_argument('--single_end', action='store_false', dest='paired_end',
                        help='Treat reads as single-end if invoked.')
    parser.add_argument('--retain_blacklist_estimates', '--retain', action='store_true', dest='retain_blacklist_estimates', default=False, help='If set, retain state estimates over blacklisted regions. By default, these estimates are manually set to the lower limit.')
    parser.add_argument('--exclude_flag', type=int, default=3840,
                        help='Exclude alignments with this SAM flag (default: 3840).')
    parser.add_argument('--min_mapq', type=float, default=0,
                        help='Minimum mapping quality (default: 0).')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads for reading BAM files via pysam.')
    parser.add_argument('-p', '--n_processes', type=int, default=None, dest='n_processes',
                        help="Number of parallel processes for computing read/noise tracks.")
    parser.add_argument('--count_both', action='store_true', default=True,
                        help='Count both reads in a proper pair as +1 (default: True).')
    parser.add_argument('--backshift', type=int, default=None,
                        help='Backshift when searching for the last read in a given chromosome.')
    parser.add_argument('--munc_min', type=float, default=0.05, help='Minimum observation noise.')
    parser.add_argument('--munc_max', type=float, default=100.0, help='Maximum observation noise.')
    parser.add_argument('--munc_smooth_bp', type=int, default=None,
                        help='Smoothing window for observation noise in base pairs.')
    parser.add_argument('--munc_local_weight', type=float, default=0.333,
                        help='Local weight for observation noise modeling (default: 1/3).')
    parser.add_argument('--munc_global_weight', type=float, default=0.667,
                        help='Global weight for observation noise modeling (default: 2/3).')
    parser.add_argument('--munc_k', type=int, default=25,
                        help='Number of proximal features for observation noise (default: 25).')
    parser.add_argument('--munc_const', type=float, default=1.0,
                        help='Use a constant observation noise--`munc_const*I` (default: 1.0).')
    parser.add_argument('--prunc_min', type=float, default=0.05, help='Minimum process noise.')
    parser.add_argument('--prunc_max', type=float, default=500.0, help='Maximum process noise.')
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Distance to propagate state (default: 1.0).')
    parser.add_argument('--Dstat_thresh', type=float, default=2.0,
                        help='D-stat threshold for process noise update (default: 2.0).')
    parser.add_argument('--Dstat_scale', type=float, default=10.0)
    parser.add_argument('--Dstat_pc', type=float, default=2.0)
    parser.add_argument('--log_scale', action='store_true', help='If set, log transform data.')
    parser.add_argument('--log_pc', type=float, default=1.0, 
                        help='Pseudocount for log transform (default: 1.0).')
    parser.add_argument('--llim', '--state_lowerlim', type=float, default=0, dest='state_lowerlim',
                        help='Lower limit of state variable (default: 0).')
    parser.add_argument('--ulim', '--state_upperlim', type=float, default=None, dest='state_upperlim',
                        help='Upper limit of state variable (default: max observed data).')
    parser.add_argument('--Qmat_offdiag', type=float, default=None)
    parser.add_argument('--no_joseph', action='store_true', default=False, help='If set, do not use the Joseph-form posterior covariance update.')
    parser.add_argument('--detrend_degree', type=int, default=None,
                        help='Degree for Savitzky-Golay-based detrend.')
    parser.add_argument('--detrend_percentile', type=int, default=None,
                        help='Percentile for sliding-percentile detrend.')
    parser.add_argument('--detrend_window_bp', type=int, default=5000,
                        help='Window size (bp) for detrending (default: 5000).')
    parser.add_argument('--detrend_lbound', type=float, default=None,
                        help='Lower bound for detrended values.')
    parser.add_argument('--detrend_ubound', type=float, default=None,
                        help='Upper bound for detrended values.')

    parser.add_argument('--signal_bigwig', type=str, default=None,
                        help='Write bigWig for state estimates.')
    parser.add_argument('--Ptrace_bigwig', type=str, default=None,
                        help='Write bigWig for trace of the state covariance.')
    parser.add_argument('--Rtrace_bigwig', type=str, default=None,
                        help='Write bigWig for trace of the observation noise covariance.')
    parser.add_argument('--residuals', '--residual_bigwig', dest='residual_bigwig',
                        type=str, default=None,
                        help='Write bigWig of inverse-variance-weighted residual mean.')
    parser.add_argument('-ares', '--abs_residuals', dest='abs_residuals', action='store_true', default=False, help='Record absolute value of ivw residuals. Only used if --residuals is set.')
    parser.add_argument('--ratio', '--ratio_bigwig', type=str, default=None,
                    help='Write bigWig of log(squared_signal/squared_ivw) ratio.', dest='ratio_bigwig')
    parser.add_argument('-o', '--output_file', dest='output_file', default=f'consenrich_output_{ID}.tsv',
                        help='Output file for Consenrich results.')
    parser.add_argument('--save_matrix', action='store_true',
                        help='Save count and noise covariance matrices to .npz for each chromosome.')
    parser.add_argument('--experiment_id', default=ID,
                        help='Experiment ID for saving data files.')

    parser.add_argument('--no_sparsebed', action='store_true',
                        help='If invoked, compute noise variances from inferred sparse regions specific to each sample based on post-detrend stationary or WSS regions.')
    parser.add_argument('--csparse_aggr_percentile', type=float, default=75,
                        help='Used to reduce data to 1d prior to the filter step in csparse if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_wlen', type=int, default=51,
                        help='Window length (in units of `--step`) for the filter step prior to computing sample-wise sparse regions if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_pdegree', type=int, default=3,
                        help='Polynomial degree for the filter step prior to computing sample-wise sparse regions if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_min_peak_len', type=int, default=10,
                        help='Minimum length of peaks (in units of `--step`) in the filter step prior to computing sample-wise sparse regions if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_min_sparse_len', type=int, default=10,
                        help='Minimum length of sparse regions (in units of `--step`) in the filter step prior to computing sample-wise sparse regions if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_min_dist', type=int, default=50,
                        help='Minimum distance (in units of `--step`) between first-pass enriched regions in the filter/fp-peak step prior to computing sample-wise sparse regions if `--no_sparsebed` is invoked.')
    parser.add_argument('--csparse_max_features', type=int, default=5000)
    parser.add_argument('--csparse_min_prom_prop', type=float, default=0.05, help='Minimum prominence threshold on first-pass peaks as a fraction of the dynamic range')
    parser.add_argument('--save_args', action='store_true',
                        help='Save arguments to a JSON file. These can be used to reproduce the experiment via `consenrich -f <json_file>`.')
    args = parser.parse_args()

    if args.config_file is not None:
        logger.info(f'Loading config file: {args.config_file}...')
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(args, key):
                current_val = getattr(args, key)
                default_val = parser.get_default(key)

                if current_val == default_val:
                    setattr(args, key, value)
            else:
                logger.warning(f'Config file key "{key}" not recognized by parser. skipping...')
                pass
    return args


def main():
    init_ID = str(int(uuid.uuid4().hex[:5], base=16))
    args = _parse_arguments(init_ID)
    ID = args.experiment_id
    logger.info(f'Consenrich Experiment {ID}')

    if len(sys.argv) == 1:
        logger.info('No arguments provided. Run `consenrich -h`.')
        sys.exit(0)

    if args.save_args:
        try:
            with open(f'consenrich_{ID}_args.json', 'w') as f:
                # dont include the config file in the saved args
                args_dict = vars(args)
                if 'config_file' in args_dict:
                    del args_dict['config_file']
                if 'save_args' in args_dict:
                    del args_dict['save_args']
                json.dump(args_dict, f, indent=4)
        except Exception as e:
            logger.warning(f'Could not save arguments to file:\n{str(e)}\n')

    if args.no_sparsebed:
        args.sparsebed = None

    # Use default resources for a supported genome
    if args.genome is not None:
        if args.sizes_file is None:
            args.sizes_file = get_genome_resource(f'{args.genome.lower()}.sizes')
        if args.blacklist_file is None:
            args.blacklist_file = get_genome_resource(f'{args.genome.lower()}_blacklist.bed')
        if args.sparsebed is None and not args.no_sparsebed:
            args.sparsebed = get_genome_resource(f'{args.genome.lower()}_sparse.bed')

    norm_counts = not args.no_norm_counts
    munc_smooth =  args.munc_smooth_bp is not None and args.munc_smooth_bp > (3*args.step)
    ignore_blacklist = not args.retain_blacklist_estimates
    joseph_ = not args.no_joseph

    # create BED of sparse segments from the complement of --active_regions
    if args.sparsebed is None and args.active_regions is not None and not args.no_sparsebed:
        sparsebed_fname=f'sparsebed_output_{args.experiment_id}.bed'
        try:
            create_sparsebed(args.active_regions, args.sizes_file, args.blacklist_file, outfile=sparsebed_fname)
            args.sparsebed = sparsebed_fname
        except Exception as ex:
            logger.info(f"Original exception:\n{ex}\n\n")
            raise Exception(
                f"Could not find/create sparsebed file {args.sparsebed}.\n"
                "You can either run with `--no_sparsebed`, in which case "
                "sparse regions are inferred as those satisfying both:\n"
                "\t(i) occur in gaps between highly enriched regions "
                "determined with a rough first pass, and\n"
                "\t(ii) The input data over the given region can be detrended such "
                "that they are approximately stationary. See "
                "`consenrich.misc_utils.check_acorr()` for more details.\n"
                "Alternatively, you can create a sparsebed file as the conservative "
                "complement of previously annotated \"active\" regions (e.g., "
                "`consenrich/refdata/hg38_sparse.bed`) or peak regions in "
                "heterochromatin/nucleosome-targeted ChIP-seq/CUT-N_RUN experiments.\n"
            )

    if os.path.exists(args.output_file):
        logger.warning(f'Output file {args.output_file} already exists. Overwriting...')
        os.remove(args.output_file)

    tmp_unsorted = f'consenrich_output_{ID}_tmp_unsorted.tsv'
    if os.path.exists(tmp_unsorted):
        os.remove(tmp_unsorted)

    bam_files = list(args.bam_files)
    chrom_list = []
    for chromosome in get_chromsizes_dict(args.sizes_file).keys():
        if args.chroms is not None and len(args.chroms) > 0 and chromosome not in args.chroms:
            continue
        if args.skip_chroms is not None and len(args.skip_chroms) > 0 and chromosome in args.skip_chroms:
            continue
        chrom_list.append(chromosome)
    chrom_list = chrom_lexsort(chrom_list, args.sizes_file)
    gwide_scales = None
    if args.norm_gwide:
        gwide_scales = [estimate_gwide_scale(bam_files[i], args.sizes_file) for i in range(len(bam_files))]

    for chromosome in chrom_list:
        logger.info(f'Processing chromosome {chromosome}...')
        intervals, est_final, Ptrace, Rtrace, residuals_ivw = run_consenrich(
            chromosome=chromosome,
            bam_files=bam_files,
            sizes_file=args.sizes_file,
            blacklist_file=args.blacklist_file,
            sparsebed=args.sparsebed,
            step=args.step,
            norm_counts=norm_counts,
            norm_gwide=args.norm_gwide,
            gwide_scales=gwide_scales,
            paired_end=args.paired_end,
            exclude_flag=args.exclude_flag,
            min_mapq=args.min_mapq,
            n_processes=args.n_processes,
            threads=args.threads,
            count_both=args.count_both,
            ignore_blacklist=ignore_blacklist,
            backshift=args.backshift,
            munc_min=args.munc_min,
            munc_max=args.munc_max,
            munc_smooth=munc_smooth,
            munc_smooth_bp=args.munc_smooth_bp,
            munc_local_weight=args.munc_local_weight,
            munc_global_weight=args.munc_global_weight,
            munc_k=args.munc_k,
            munc_const=args.munc_const,
            prunc_min=args.prunc_min,
            prunc_max=args.prunc_max,
            delta=args.delta,
            Dstat_thresh=args.Dstat_thresh,
            Dstat_scale=args.Dstat_scale,
            Dstat_pc=args.Dstat_pc,
            state_lowerlim=args.state_lowerlim,
            state_upperlim=args.state_upperlim,
            Qmat_offdiag=args.Qmat_offdiag,
            joseph=joseph_,
            detrend_degree=args.detrend_degree,
            detrend_percentile=args.detrend_percentile,
            detrend_window_bp=args.detrend_window_bp,
            detrend_lbound=args.detrend_lbound,
            detrend_ubound=args.detrend_ubound,
            save_matrix=args.save_matrix,
            experiment_id=args.experiment_id,
            log_scale=args.log_scale,
            log_pc=args.log_pc,
            control_files=args.control_files,
            no_sparsebed=args.no_sparsebed,
            csparse_aggr_percentile=args.csparse_aggr_percentile,
            csparse_wlen=args.csparse_wlen,
            csparse_pdegree=args.csparse_pdegree,
            csparse_min_peak_len=args.csparse_min_peak_len,
            csparse_min_sparse_len=args.csparse_min_sparse_len,
            csparse_min_dist=args.csparse_min_dist,
            csparse_max_features=args.csparse_max_features,
            csparse_min_prom_prop=args.csparse_min_prom_prop
        )
        with open(tmp_unsorted, 'a') as f:
            for i in range(len(intervals)):
                f.write(f'{chromosome}\t{intervals[i]}\t{intervals[i]+args.step}\t{round(est_final[i],3)}\t{Ptrace[i]}\t{round(Rtrace[i],3)}\t{round(residuals_ivw[i],3)}\n')
    logger.info(f'Calling `bedtools sort -i {tmp_unsorted}`...')
    failed_sort = False
    try:
        pbt.BedTool(tmp_unsorted).sort().saveas(args.output_file)
        if os.path.exists(args.output_file):
            logger.info(f'Successfully sorted output in standard lexicographical order: {args.output_file}')
    except Exception as e:
        logger.warning(f'Could not sort output in standard lexicographical...\ntry manually with `bedtools sort -i {tmp_unsorted}`\nException:\n{str(e)}\n')
        failed_sort = True
    if failed_sort:
        shutil.copy(tmp_unsorted, args.output_file)

    try:
        os.remove(tmp_unsorted)
    except:
        logger.warning(f'Could not remove temporary file {tmp_unsorted}')

    logger.info(f'Wrote human-readable, tab-separated Consenrich results to {args.output_file}')

    # write bigWig files if requested
    if args.signal_bigwig is not None:
        try:
            write_bigwig(args.output_file, args.sizes_file, chrom_list, args.signal_bigwig, stat='signal')
        except Exception as e:
            logger.warning(f'Could not write signal bigWig file {args.signal_bigwig}:\n{str(e)}\n')
    if args.Ptrace_bigwig is not None:
        try:
            write_bigwig(args.output_file, args.sizes_file, chrom_list, args.Ptrace_bigwig, stat='Ptrace')
        except Exception as e:
            logger.warning(f'Could not write Ptrace bigWig file {args.Ptrace_bigwig}:\n{str(e)}\n')
    if args.Rtrace_bigwig is not None:
        try:
            write_bigwig(args.output_file, args.sizes_file, chrom_list, args.Rtrace_bigwig, stat='Rtrace')
        except Exception as e:
            logger.warning(f'Could not write Rtrace bigWig file {args.Rtrace_bigwig}:\n{str(e)}\n')
    if args.residual_bigwig is not None:
        try:
            write_bigwig(args.output_file, args.sizes_file, chrom_list, args.residual_bigwig, stat='residuals_ivw', abs_residuals=args.abs_residuals)
        except Exception as e:
            logger.warning(f'Could not write residual ivw est. bigWig file {args.residual_bigwig}:\n{str(e)}\n')
    if args.ratio_bigwig is not None:
        try:
            write_bigwig(args.output_file, args.sizes_file, chrom_list, args.ratio_bigwig, stat='ratio')
        except Exception as e:
            logger.warning(f'Could not write ratio bigWig file {args.ratio_bigwig}:\n{str(e)}\n')

    return 0

if __name__ == '__main__':
    sys.exit(main())
