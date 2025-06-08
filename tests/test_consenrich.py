import os
import numpy as np
import pybedtools as pbt
import pytest
import subprocess
import deeptools
import pywt


@pytest.mark.correctness
def test_import():
    import consenrich


@pytest.mark.correctness
def test_cli_noargs():
    # just display a help message if no arguments are passed -- handle gracefully
    help_cmd = ['consenrich']
    proc = subprocess.run(help_cmd, stdout=subprocess.PIPE)
    assert proc.returncode == 0, f'Error code {proc.returncode} returned'


@pytest.mark.consistency
def test_consistency_atac(refsig='test_ref_sig.bw', refres='test_ref_res.bw', thresh=0.95):
    oname_sig = 'test_sig_cmp.bw'
    oname_res = 'test_res_cmp.bw'
    consenrich_cmd = ['consenrich', '--bam_files', 'test_sample_one.bam', 'test_sample_two.bam', 'test_sample_three.bam', '-g', 'hg38', '--chroms', 'chr19', 'chr21', 'chr22', '--signal_bigwig', oname_sig, '--residuals', oname_res, '-p', '4', '--threads', '2', '--retain'] # run with --retain to keep track of similarities over all regions
    subprocess.run(consenrich_cmd, check=True)

    # Note: these will fail if the UCSC bigWigCorrelate tool isn't installed
    bigwigcorr_cmd = ['bigWigCorrelate',  refsig, oname_sig, '-ignoreMissing', '-restrict=test_region_file.bb']
    proc = subprocess.run(bigwigcorr_cmd, check=True, stdout=subprocess.PIPE)
    proc.stdout = str(proc.stdout.decode('utf-8')).strip()
    print(proc.stdout)
    assert float(proc.stdout) >= thresh, f'BigWigCorrelate correlation coefficient below {thresh}: {proc.stdout}'
    assert not np.isnan(float(proc.stdout)), f'BigWigCorrelate correlation coefficient is NaN: {proc.stdout}'

    bigwigcorr_cmd_res = ['bigWigCorrelate',  refres, oname_res, '-ignoreMissing', '-restrict=test_region_file.bb']
    proc_res = subprocess.run(bigwigcorr_cmd_res, check=True, stdout=subprocess.PIPE)
    proc_res.stdout = str(proc_res.stdout.decode('utf-8')).strip()
    print(proc_res.stdout)
    assert not np.isnan(float(proc_res.stdout)), f'BigWigCorrelate correlation coefficient is NaN: {proc_res.stdout}'
    assert float(proc.stdout) >= thresh, f'BigWigCorrelate correlation coefficient below {thresh}: {proc.stdout}'


@pytest.mark.consistency
def test_get_first_read():
    # Note, these will likely fail if the test BAMs are changed
    expected_step100 = 10522300
    expected_step5 = 5010175
    from consenrich.misc_util import get_first_read
    assert get_first_read('chr22', 'test_sample_one.bam', sizes_file='tests.sizes', step=100, exclude_flag=3840) == expected_step100, f'Expected {expected_step100} for step 100bp'
    assert get_first_read('chr21', 'test_sample_one.bam', sizes_file='tests.sizes', exclude_flag=3840, step=5) == expected_step5, f'Expected {expected_step5} for step 5bp'


@pytest.mark.consistency
def test_get_last_read():
    # Note, these will likely fail if the test BAMs are changed
    expected_step100 = 50808300
    expected_step5 = 46699335
    from consenrich.misc_util import get_last_read
    assert get_last_read('chr22', 'test_sample_one.bam', sizes_file='tests.sizes', exclude_flag=3840, step=100) == expected_step100, f'Expected {expected_step100} for step 100bp'
    assert get_last_read('chr21', 'test_sample_one.bam', sizes_file='tests.sizes', exclude_flag=3840, step=5) == expected_step5, f'Expected {expected_step5} for step 5bp'


@pytest.mark.correctness
def test_acorr_fft_bp():
    from consenrich.misc_util import acorr_fft_bp
    np.random.seed(42)
    x = np.random.normal(0,1,10000)
    acorr_vec = acorr_fft_bp(x, step=1)
    # note, the practically-oriented detrending procedure in acorr_fft_bp
    # will generally result in an ACF that is not exactly 1 at lag 0, but
    # we expect it to be close
    assert abs(acorr_vec[0] - 1) < 1.0e-01, f'Expected {acorr_vec[0]} ~ 1'


@pytest.mark.match
def test_consistency_match_dwt(refbed='test_ref_match.bed', thresh=0.95):
    output_file = 'test_cmp_match.bed'
    if os.path.exists(output_file):
        os.remove(output_file)
    consenrich_cmd = [
        "consenrich",
        "--bam_files", "test_sample_one.bam", "test_sample_two.bam", "test_sample_three.bam",
        "-g", "hg38",
        "--chroms", "chr21", "chr22",
        "-p","4",
        "--threads","4",
        "--match_wavelet","db2,sym3,dmey",
        "--match_level","1,2",
        "--match_output_file", output_file,
    ]
    subprocess.run(consenrich_cmd, check=True)
    assert os.path.exists(output_file), 'Output file does not exist or was named incorrectly'
    # compare jaccard similarity wrt `test_cmp_match.bed` (the output of this test)
    # and `test_ref_match.bed` (the reference)
    a = pbt.BedTool(output_file).sort()
    b = pbt.BedTool(refbed).sort()
    jsim = float(a.jaccard(b)['jaccard'])
    assert jsim >= thresh, f"Match filter results' Jaccard similarity insufficient {thresh}: {jsim}"
