# Consenrich

[![Tests](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml/badge.svg?event=workflow_dispatch)](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/consenrich?logo=Python&logoColor=%23FFFFFF&color=%233776AB&link=https%3A%2F%2Fpypi.org%2Fproject%2Fconsenrich%2F)

[Consenrich](https://github.com/nolan-h-hamilton/Consenrich) is a sequential genome-wide state estimator for extraction of reproducible, spatially-resolved, epigenomic signals hidden in noisy multisample HTS data. The [corresponding manuscript preprint](https://www.biorxiv.org/content/10.1101/2025.02.05.636702v1) is available on $$\text{bio}\textcolor{#960018}{R}\chi \text{iv}$$.

---

* **Input**:
  * $m \geq 1$ Sequence alignment files `-t/--bam_files` corresponding to each sample in a given HTS experiment
  * (*Optional*): $m_c = m$ control sample alignments, `-c/--control_files`, for each 'control' sample (e.g., ChIP-seq)

* **Output**:

  * Genome-wide 'consensus' epigenomic state estimates and uncertainty metrics (BedGraph/BigWig)

---

**Features**

* Uncertainty-moderated signal tracks encompassing multiple samples' epigenomic profiles $\implies$ Insightful data representation for conventional analyses aiming to profile trait-specific regulatory landscapes (e.g., via [consensus peak calling](docs/consensus_peaks.md))


* Models trends and noise profiles for each sample with scale-invariance $\implies$ [Multi-sample, multi-assay estimation of target molecular states](docs/atac_dnase.png) from related functional genomics assays, e.g., ChIP-seq + CUT-N-RUN, ATAC-seq + DNase-seq.


* [Preservation of spectral content](docs/filter_comparison.png) $\implies$ Comparison and profiling of group-specific structural signatures discarded by traditional enrichment-focused measures for HTS data.

## Example Use

* Run Consenrich on ten heterogeneous ATAC-seq sample alignments in the current directory (`*.bam`).
  

  ```bash
  consenrich --bam_files *.bam -g hg38 --signal_bigwig demo_signal.bw --residuals_bigwig demo_residuals.bw
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

Consenrich is available via [PyPI/pip](https://pypi.org/project/consenrich/):

* `pip install consenrich`

If managing multiple Python environments, use `python -m pip install consenrich`. If lacking administrative privileges, running with flag `--user` may be necessary.

---

Consenrich can also be easily downloaded and installed from source:

1. `git clone https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python setup.py sdist bdist_wheel`
4. `python -m pip install .`
5. Check installation: `consenrich --help`

## Manuscript Preprint and Citation

*Genome-Wide Uncertainty-Moderated Extraction of Signal Annotations from Multi-Sample Functional Genomics Data*\
Nolan H Hamilton, Benjamin D McMichael, Michael I Love, Terrence S Furey; doi: `10.1101/2025.02.05.636702`

---

**BibTeX**

```bibtex
@article {Hamilton2025
	author = {Hamilton, Nolan H and McMichael, Benjamin D and Love, Michael I and Furey, Terrence S},
	title = {Genome-Wide Uncertainty-Moderated Extraction of Signal Annotations from Multi-Sample Functional Genomics Data},
	year = {2025},
	doi = {10.1101/2025.02.05.636702},
	url = {https://www.biorxiv.org/content/10.1101/2025.02.05.636702v1},
}
```
