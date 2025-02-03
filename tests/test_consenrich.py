import os
import numpy as np
import pybedtools as pbt
import pytest
import subprocess
import deeptools


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
def test_consistency_atac(refsig='test_ref_sig.bw', refres='test_ref_res.bw', thresh=0.99):
    oname_sig = 'test_sig_cmp.bw'
    oname_res = 'test_res_cmp.bw'
    consenrich_cmd = ['consenrich', '--bam_files', 'test_sample_one.bam', 'test_sample_two.bam', 'test_sample_three.bam', '-g', 'hg38', '--chroms', 'chr19', 'chr21', 'chr22', '--signal_bigwig', oname_sig, '--residuals', oname_res, '-p', '4', '--threads', '2', '--retain'] # run with --retain to keep track of similarities over all regions
    subprocess.run(consenrich_cmd, check=True)

    # Note: these will fail if the UCSC bigWigCorrelate tool isn't installed
    bigwigcorr_cmd = ['bigWigCorrelate',  refsig, oname_sig, '-ignoreMissing', '-restrict=test_region_file.bb']
    proc = subprocess.run(bigwigcorr_cmd, check=True, stdout=subprocess.PIPE)
    proc.stdout = str(proc.stdout.decode('utf-8')).strip()
    assert float(proc.stdout) >= thresh, f'BigWigCorrelate correlation coefficient below {thresh}: {proc.stdout}'

    bigwigcorr_cmd_res = ['bigWigCorrelate',  refres, oname_res, '-ignoreMissing', '-restrict=test_region_file.bb']
    proc_res = subprocess.run(bigwigcorr_cmd_res, check=True, stdout=subprocess.PIPE)
    proc_res.stdout = str(proc_res.stdout.decode('utf-8')).strip()
    assert float(proc.stdout) >= thresh, f'BigWigCorrelate correlation coefficient below {thresh}: {proc.stdout}'
    