# Consenrich

[![Tests](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml/badge.svg?event=workflow_dispatch)](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/consenrich?color=0%2C200%2C200)

*[Consenrich](https://github.com/nolan-h-hamilton/Consenrich) is a sequential genome-wide state estimator for extraction of reproducible, spatially-resolved, epigenomic signals hidden in noisy multisample HTS data.*

---

* **Input**:
  * $m \geq 1$ Sequence alignment files `-t/--bam_files` corresponding to each sample in a given HTS experiment
  * (*Optional*): $m_c = m$ control sample alignments, `-c/--control_files`, for each 'treatment' sample (e.g., ChIP-seq)

* **Output**: Real-valued 'consensus' epigenomic state estimates (BedGraph/BigWig) and uncertainty metrics.

---

* Robust, spatially informative consensus signal tracks representing multiple samples' epigenomic profiles $\implies$ Greater insight for a variety of conventional analyses aiming to construct encompassing regulatory characterizations of sample groups (e.g., [consensus peak calling](docs/consensus_peaks.md))
* Models trends and noise profiles for each sample with scale-invariance $\implies$ Enables estimation of target molecular states across data samples generated from multiple related assays, for example, ChIP-seq + CUT-N-RUN or [ATAC-seq + DNase-seq](docs/atac_dnase.png)
* Preservation of [spectral content](docs/filter_comparison.png) $\implies$ Permits targeted detection of structural/spatial patterns associated with specific regulatory properties/states and additional metrics for comparison of HTS data between trait-groups

Several technical features of Consenrich are discussed [below](#technical-features).

## Example Command-Line Use

* Run Consenrich on ten ATAC-seq samples in the current directory. Generate a BigWig signal track and inverse-variance-weighted residuals.

   ```bash
   consenrich --bam_files *.bam -g hg38 -o hg38_test_output.tsv --signal_bigwig demo_signal.bw --residual_bigwig demo_ivw_residuals.bw
   ```

![fig1](docs/figure_1aa.png)

---

* Use Consenrich for ChIP-seq enrichment analysis with treatment/control sample alignments (POL2RA, six donors' colon tissue samples). Generate separate BigWig output tracks for signal estimates and inverse-variance weighted residuals. Use fixed-width genomic intervals of 25bp:

   ```bash
  consenrich \
    --bam_files \
      ENCSR322JEO_POL2RA.bam \
      ENCSR472VBD_POL2RA.bam \
      ENCSR431EHE_POL2RA.bam \
      ENCSR724FCJ_POL2RA.bam \
      ENCSR974HQI_POL2RA.bam \
      ENCSR132XRW_POL2RA.bam \
    --control_files \
      ENCSR322JEO_CTRL.bam \
      ENCSR472VBD_CTRL.bam \
      ENCSR431EHE_CTRL.bam \
      ENCSR724FCJ_CTRL.bam \
      ENCSR974HQI_CTRL.bam \
      ENCSR132XRW_CTRL.bam \
    -g hg38 --step 25 \
    -o Consenrich_POL2RA.tsv \
    --signal_bigwig Consenrich_POL2RA_CTRL_Signal.bw \
    --residual_bigwig Consenrich_POL2RA_CTRL_IVW_Residuals.bw
   ```

**Output**
![ChIPDemo](docs/ChIP_POL2RA_Demo.png)

## Download/Install

Consenrich can be easily downloaded and installed from source:

1. `git clone https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python setup.py sdist bdist_wheel`
4. `python -m pip install .`
5. Check installation: `consenrich --help`

Consenrich is also available via [PyPI/pip](https://pypi.org/project/consenrich/):

* `pip install consenrich`

If managing multiple Python environments, use `python -m pip install consenrich`. If lacking administrative privileges, running with flag `--user` may be necessary.

## Technical Features

* Effectively models sample-and-region-varying noise to better integrate data across heterogeneous samples
* Balances biologically-informed *a priori* predictions with observed HTS data to determine final estimates
* Provides interpretable uncertainty quantification with respect to multiple model aspects
* Runs efficiently in linear time with respect to genome size.
