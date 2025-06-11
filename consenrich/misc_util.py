r"""
Consenrich Utilities `misc_util` Documentation
===============================================================

The `misc_util` module contains utility functions for Consenrich.

"""

from calendar import c
import logging
import math
from pprint import pprint
import os
import re
import uuid
from typing import Optional, Tuple, Union, Dict, Any, List, Callable

import numpy as np
from numpy._typing._array_like import NDArray
import pandas as pd
import pybedtools as pbt
import pysam
import pywt

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
            logger.info(f'Could not find index file for {bam_file}.calling pysam.index()')
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
    :return: Dictionary of chromosome sizes. Formatted as `{chromosome_name: size}`, e.g., `{'chr1': 248956422, 'chr2': 242193529, .}`

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


def write_bigwig(tsv_file, sizes_file, chrom_list, outfile_name,
                stat='signal', square_residuals=False,
                bigwig_precision=4) -> str:
    r"""Write a bigWig file from a TSV file and sizes file.
    :param tsv_file: The five-column BedGraph-like TSV files generated by `run_consenrich`.
    :param sizes_file: Path to sizes file.
    :param stat: Statistic to write (`signal`, `residual`, or `eratio`).
    """

    try:
        import pyBigWig as pbw
    except ImportError:
        logger.warning('pyBigWig is not installed. Skipping bigWig output.')
        return None

    if pbw.numpy != 1:
        logger.warning('To obtain bigWig output, pyBigWig must be compiled with NumPy support.\
                          Try reinstalling pyBigWig *after* installing NumPy \
                          See https://github.com/deeptools/pyBigWig #numpy for additional details.\
                          Skipping bigWig output.')
        return None
    if not os.path.exists(tsv_file):
        logger.warning(f'Could not find {tsv_file}. Skipping bigWig output.')
        return None

    if stat.lower() not in ['signal',  'residual', 'eratio']:
        logger.warning(f'Invalid statistic {stat} specified. Skipping bigWig output.')
        return None

    if os.path.exists(outfile_name):
        logger.warning(f'Overwriting existing bigWig file: {outfile_name}')
        os.remove(outfile_name)

    sizes_dict = get_chromsizes_dict(sizes_file)
    chrom_list = chrom_lexsort([str(x) for x in sizes_dict.keys() if x in chrom_list], sizes_file=sizes_file)
    tsv_df = pd.read_csv(tsv_file, sep='\t', names=['chrom', 'start', 'end', 'signal', 'residual'])
    tsv_df = tsv_df[tsv_df['chrom'].isin(chrom_list)]

    pbw_out = pbw.open(outfile_name, 'w')
    pbw_out.addHeader([(chrom, size) for chrom, size in sizes_dict.items() if chrom in chrom_list])

    for chrom in chrom_list:
        chrom_df = tsv_df[tsv_df['chrom'] == chrom]
        chroms = np.array(chrom_df['chrom'], dtype='str')
        starts = np.array(chrom_df['start'], dtype='int')
        ends = np.array(chrom_df['end'], dtype='int')

        if stat.lower() == 'signal':
            sig_values = np.array(chrom_df['signal'], dtype='float')
            pbw_out.addEntries(chroms, starts=starts, ends=ends, values=np.round(sig_values, decimals=bigwig_precision))

        elif stat.lower() == 'residual':
            res_values = np.array(chrom_df['residual'], dtype='float')
            if square_residuals:
                res_values = res_values**2
            pbw_out.addEntries(chroms, starts=starts, ends=ends, values=np.round(res_values, decimals=bigwig_precision))

        elif stat.lower() == 'eratio':
            # Not an SNR: values << 0 are not clipped to zero in the numerator
            # it's assumed signal of interest is increasing with counts
            sq_signal_values = np.maximum(np.array(chrom_df['signal'], dtype='float'),0)**2
            sq_res_values = np.array(chrom_df['residual'], dtype='float')**2
            ratio_vals = np.log1p(sq_signal_values) - np.log1p(sq_res_values)
            pbw_out.addEntries(chroms, starts=starts, ends=ends, values=np.round(ratio_vals, decimals=bigwig_precision))

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


def acorr_fft_bp(x, dtr_wlen_bp=250, step=50,):
    r"""Computes the autocorrelation function (ACF) via FFT convolution on a detrended `x`
    """
    n_x = len(x)
    if step == 0:
        raise ValueError('Step size must be greater than 0')
    dtr_wlen,degree = dtr_wlen_degree(step=step, n=len(x))
    x = x - signal.savgol_filter(x, dtr_wlen, degree)
    return signal.fftconvolve(x, x[::-1], mode='full')[len(x)-1:] / np.flip(np.arange(1,n_x+1))


def check_acorr(x, acorr_threshold = 0.667, dtr_wlen_bp=250, step=50):
    r"""Compares compares peaks in the autocorrelation function (ACF) to evaluate stationarity.

    Retains regions where the great value occurs at :math:`ACF(\tau = 0)` and the second greatest value is less than `acorr_threshold` times the greatest value. The autocorrelation is computed efficiently via the FFT convolution 
    method.

    :param x: array of values covering a candidate sparse ('csparse') region
    :param acorr_threshold: Thresholds the ratio of the autocorrelation function (ACF) peaks.
    :param dtr_wlen_bp: Window length in base pairs for detrending prior to computing ACF via fftconvolve in `acorr_fft_bp()`.
    :param step: Step size for intervals.
    :return: A tuple (`bool`, `float`). True if `acorr_threshold` is satisfied, and value :math:`\frac{ACF(0}{\left(\max_{\tau = 1,2,\ldots} ACF(\tau)\right)}`
    :rtype: tuple

    :seealso: `acorr_fft_bp()`, `scipy.signal.fftconvolve()`, `get_csparse()`
    """
    aacorr_vec = acorr_fft_bp(x, dtr_wlen_bp=dtr_wlen_bp, step=step)
    max_ = np.max(aacorr_vec)
    second_max_ = np.max(np.abs(aacorr_vec[1:]))
    return max_ == aacorr_vec[0] and second_max_*(1/acorr_threshold) < max_, max_ / (second_max_ + 1e-4)


def get_csparse(chromosome: str, intervals: np.ndarray, vals: np.ndarray,
               aggr_percentile: int=75, wlen=25, pdegree=3,
               min_peak_len=10, min_sparse_len=10, min_dist=50,
               min_prom_prop: float=.05, bed: str=None,
               acorr_threshold: float=0.667) -> np.ndarray:
    r"""Identify regions over which the local noise variance can be approximated quickly if Consenrich is run with `--no_sparsebed`.

    First calls clear peaks in a crudely aggregated+low-pass filtered version of the chromosome count matrix. Then checks for approximate stationarity in the 'sparse' regions between peaks. Writes a bed file of the sparse regions.

    :param chromosome: chromosome name.
    :param intervals:  array of genomic intervals 
    :param vals: np.ndarray of values monotonic with the number of sequence alignments at each interval.
    :param aggr_percentile: Percentile to use for aggregating values across rows.
    :param wlen: Window length for Savitzky-Golay filter.
    :param pdegree: Polynomial degree for Savitzky-Golay filter.
    :param min_peak_len: Minimum length of peaks.
    :param min_sparse_len: Minimum length of sparse regions (between peaks) for further consideration.
    :param min_dist: Minimum distance between peaks.
    :param bed: Output bed file name.
    :return: Path to BED file of qualifying sparse regions.

    :seealso: `scipy.signal.savgol_filter()`, `scipy.signal.find_peaks()`, `check_acorr()`    
    """

    if bed is None:
        bed = f'{chromosome}_csparse.bed'

    step = get_step(intervals)
    if wlen % 2 == 0:
        logger.info(f'Window length must be odd. Adding 1 to {wlen}')
        wlen += 1
    wlen_bp = wlen * step
    if wlen_bp < 100:
        logger.info(f'Window length for filtering the aggregated data is less than 100bp. Consider increasing wlen.')

    if len(vals) < wlen:
        wlen = len(vals)//2
        pdegree = max(min(pdegree, wlen//2),1)

    agg_vals = None
    if get_shape(vals)[0] > 1:
        agg_vals = np.percentile(vals, aggr_percentile, axis=0)
    else:
        agg_vals = vals

    # filter higher frequencies for peak calling using SG(wlen, pdegree)
    lowpass_filtered_vals = signal.savgol_filter(agg_vals, wlen, pdegree)

    iqr_ = stats.iqr(lowpass_filtered_vals, rng=(1,99))
    min_peak_prom = math.ceil(iqr_*min_prom_prop)
    logger.info(f'Using prominence threshold: {min_peak_prom}')

    # call peaks in the lowpass filtered data using a basic nonparametric approach
    peaks, peak_properties = signal.find_peaks(lowpass_filtered_vals, prominence=min_peak_prom, width=min_peak_len)
    prev_bound = 0

    if os.path.exists(bed):
        os.remove(bed)
    with open(bed,'w') as bed_out:
        for i in range(len(peaks)-1):
            sparse_bounds = (peak_properties['left_bases'][i], peak_properties['right_bases'][i])
            if sparse_bounds[1] - sparse_bounds[0] >= min_sparse_len and sparse_bounds[0] > prev_bound + min_dist:
                prev_bound = sparse_bounds[1]
                idx_range = np.arange(sparse_bounds[0],sparse_bounds[1])
                sufficient_, acorr_measure = check_acorr(agg_vals[idx_range], acorr_threshold)
                if sufficient_:
                    bed_out.write(f"{chromosome}\t{intervals[sparse_bounds[0]]}\t{intervals[sparse_bounds[1]]}\t{'_'.join([chromosome, str(intervals[sparse_bounds[0]]), str(intervals[sparse_bounds[1]])])}\t{acorr_measure}\n")

    return bed


def dtr_wlen_degree(step: int, n: int=None,
            n_bp: int=None,
            odd_window: bool=False):
    if n is None and n_bp is None:
        raise ValueError('Either n or n_bp must be specified')
    if step is None:
        raise ValueError('Step must be specified')

    n_bp = n * step if n is not None else n_bp
    n = n_bp // step if n_bp is not None else n

    # ideally, this function won't be called for n < 5 or n > 1000
    if n == 0:
        raise ValueError('n must be greater than 0')
    if n == 1:
        return 1,0
    if n < 3:
        if odd_window:
            return 1,0
        else:
            return n,0
    if n <= 5:
        return 3,0
    if n <= 10:
        return 5,0
    if n <= 25:
        return 7,1
    if n <= 50:
        return 11,2
    if n <= 100:
        return 21,2
    return 25,2


def match(
    intervals: np.ndarray,
    values: np.ndarray,
    wavelet: str = "sym4",
    level: Optional[int] = 1,
    min_len: Optional[int] = None,
    min_val: Optional[float] = None,
    min_val_data: Optional[float] = None,
    max_matches: Optional[int] = 5000,
    unit_template: Optional[bool] = True,
    square_response: Optional[bool] = False,
    logscale_data: Optional[bool] = False,
    verbose: bool = False) -> Dict[str, Any]:
    r"""Match discrete wavelet-based templates on over genomic segments

    :param intervals: Numpy array of genomic intervals.
    :param values: Numpy array of values (In the default use-case, this is the Consenrich signal track output)
    :param wavelet: Wavelet-based template to match in `values` . Default is "db2".
    :param level: See `pywt.Wavelet().wavefun()`. Default is 1.
    :param min_len: Used for the `order` parameter in `scipy.signal.argrelmax()`. Number of surrounding points to consider for relative maxima. Default is None, which sets it to half the length of the wavelet template.
    :param min_val: Minimum value to consider for relative maxima
    :param unit_template: Whether to scale the wavelet template to unit norm. Default is True.
    :param square_response: Whether to the template-`values` convolution before checking for relative maxima. Default is False.
    :param logscale_data: Whether to log-scale the `values` and `min_val_data`. Default is False.
    :param min_val_data: Minimum value in `values` to consider for relative maxima. Default is None, which sets it to the 95th percentile of `values`.
    :param max_matches: Maximum number of matches to return for a single segment represented by `intervals, values`, e.g., a chromosome. Default is 5000.
    :param verbose: Whether to print the output dictionary. Default is False.
    :return: A dictionary containing the following keys:
    """

    if len(intervals) != len(values):
        raise ValueError(f"Length of intervals and values must be the same: {len(intervals)}, {len(values)}")
    if len(np.unique(np.diff(intervals))) > 1:
        raise ValueError(f"Intervals are expected to be fixed in length: {np.unique(np.diff(intervals))}")
    step = intervals[1] - intervals[0]

    if logscale_data:
        logger.info(f"log-scaling data")
        values = np.log1p(values)
        if min_val is not None:
            logger.info(f"log-scaling `min_val`: {min_val} --> {np.log1p(min_val)}")
            min_val = np.log1p(min_val)
        if min_val_data is not None:
            logger.info(f"log-scaling `min_val_data`: {min_val_data} --> {np.log1p(min_val_data)}")
            min_val_data = np.log1p(min_val_data)

    skip_relmax = False
    wav = pywt.Wavelet(wavelet)
    scaling_func, wavelet_func, x = wav.wavefun(level=level)
    template = wavelet_func[::-1].copy()
    logger.info(f"Template length {len(template)}")

    if unit_template:
        template /= np.linalg.norm(template)

    conv_values: np.ndarray = signal.fftconvolve(values, template, mode='same')
    if square_response:
        logger.info(f"Squaring template-signal convolution ")
        conv_values = conv_values**2
    if min_val is None:
        min_val = max([np.percentile(conv_values, 95.0), 1.0, round(np.log1p(len(template)))]) # type: ignore
        logger.info(f"Minimum relmax value in convolution output: {min_val}")
    if min_val_data is None:
        min_val_data = max([np.percentile(values, 95.0), 1.0, round(np.log1p(len(template)))]) # type: ignore
        logger.info(f"Minimum relmax value in data: {min_val_data}")
    if min_len is None:
        min_len = max(int((len(template)/2)), 5)
        logger.info(f"Using match_min_len={min_len}")

    conv_indices = None
    ret_dict = None
    if not skip_relmax:
        relmax_indices = signal.argrelmax(conv_values, order=min_len)[0]
        conv_indices = [idx for idx in relmax_indices if values[idx] > min_val_data and conv_values[idx] > min_val]
        if max_matches is not None and len(conv_indices) > max_matches:
            logger.info(f"Matches limited by 'max_matches' {max_matches}")
            conv_indices = sorted(conv_indices, key=lambda idx: values[idx], reverse=True)[:max_matches]
        if len(conv_indices) == 0:
            logger.info(f"No relative maxima found with min_val={min_val} and min_val_data={min_val_data}")
            skip_relmax = True
        logger.info(f"relative maxima in convolution with template: {len(conv_indices)}")
        ret_dict = {"intervals": intervals,
            "values": values,
            "convolution": conv_values,
            "template": template,
            "maxima_idx": conv_indices,
            "maxima_intervals": intervals[conv_indices],
            "maxima_values": conv_values[conv_indices],
            "min_len": min_len,
            "min_val": min_val,
            "min_val_data": min_val_data,
            "step": step}
    else:
        ret_dict = {"intervals": intervals,
            "values": values,
            "convolution": conv_values,
            "template": template,
            "maxima_idx": None,
            "maxima_intervals": None,
            "maxima_values": None,
            "min_len": min_len,
            "min_val": min_val,
            "min_val_data": min_val_data,
            "step": step}
    if verbose:
        pprint(ret_dict)
    return ret_dict


def split_matches(bed_file: str,
                  wavelet: str,
                  level: int,
                  outfile: Optional[str] = None,
                  narrowPeak: Optional[bool] = True) -> str:
    pbt_matched = pbt.BedTool(bed_file).sort()
    if outfile is None:
        outfile = f'split_matches_{wavelet}_{level}.bed'
    if os.path.exists(outfile):
        logger.warning(f'Overwriting existing file {outfile}')
        os.remove(outfile)
    outfile_cpy = outfile
    with open(outfile, 'w') as out_f:
        for i, record in enumerate(pbt_matched):
            name_ = record.name
            if not name_.startswith(f'{wavelet}_{level}_'):
                continue
            out_f.write(f'{record.chrom}\t{record.start}\t{record.end}\t{name_}\t{record.score}\t.\n')
    if narrowPeak:
        try:
            logger.info(f'Converting {outfile} to narrowPeak format')
            outfile = to_narrowPeak(outfile, outfile.replace('.bed', '.narrowPeak'))
            if not os.path.exists(outfile) or not outfile.endswith('.narrowPeak'):
                raise FileNotFoundError(f'Could not convert {outfile} to narrowPeak format')
            else:
                if os.path.exists(outfile_cpy) and outfile_cpy != outfile:
                    os.remove(outfile_cpy)  # remove the original BED file
        except Exception as e:
            logger.warning(f'Skipping narrowPeak conversion...\n{e}\n')
            pass
    logger.info(f'Writing matches: wavelet/{wavelet} level/{level}')
    return outfile


def to_narrowPeak(input_path: str, output_path: str) -> str:
    """assumes BED6 format from main() in consenrich.py"""

    recs = []
    min_, max_ = float('inf'), float('-inf')
    line_fmt_check = True
    with open(input_path) as infile:
        for line in infile:
            try:
                chrom, start, end, name, score_str, strand = line.rstrip('\n').split('\t')[:6]
            except ValueError as ve:
                if line_fmt_check:
                    logger.warning(f'Expected BED6 format: {line.rstrip()}\n{ve}\n')
                    line_fmt_check = False
                continue

            score = float(score_str)
            min_, max_ = min(min_, score), max(max_, score)
            recs.append((chrom, int(start), int(end), name, score, strand))

    with open(output_path, 'w') as outfile:
        for chrom, start, end, name, score, strand in recs:
            if max_ > min_:
                frac_to_max = (score - min_) / (max_ - min_)
                normed_score = int(round(250 + frac_to_max * 750))
            else:
                normed_score = 250
            midpoint = np.round((start + end) / 2.0).astype(int)
            peak = midpoint - start
            strand = '.'
            outfile.write(f'{chrom}\t{start}\t{end}\t{name}\t{normed_score}\t{strand}\t{score}\t-1\t-1\t{peak}\n')
    return output_path
