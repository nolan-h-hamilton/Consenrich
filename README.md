# Consenrich

[![Tests](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml/badge.svg?event=workflow_dispatch)](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/consenrich?logo=Python&logoColor=%23FFFFFF&color=%233776AB&link=https%3A%2F%2Fpypi.org%2Fproject%2Fconsenrich%2F)

[Consenrich](https://github.com/nolan-h-hamilton/Consenrich) is a sequential state estimator for extraction of genome-wide epigenetic signals and uncertainty quantification inferred from multi-sample high-throughput functional genomics datasets.

<p align="center">
  <img src="docs/scheme.png" alt="Example output with --match_wavelet haar,db2,db4" width="800"/><br/>
  <em> Consenrich sequentially estimates epigenomic states from multisample DNA-enrichment assays--ATAC-seq, ChIP-seq, etc. By modeling  both (<b>i</b>) local and global spatial dependencies and (<b>ii</b>) noise due to regional artifacts and individual samples, Consenrich yields a genome-wide track of 'consensus' signal estimates with variance propagation and elucidated spatial features.</em>
</p>

## Usage

* **Input**:
  * $m \geq 1$ Sequence alignment files `-t/--bam_files` corresponding to each sample in a given HTS experiment
  * (*Optional*): $m_c = m$ control sample alignments, `-c/--control_files`, for each 'control' sample (e.g., ChIP-seq)
  * (*Optional*): approximated wavelet templates (cascade iterations) for pattern recognition (`--match_wavelet ` `db<2,3,...>`, `sym<2,3,...>`, `haar`, `coif<1,2,...>`, `dmey`)

* **Output**:
  * Genome-wide 'consensus' epigenomic state estimates and uncertainty metrics
  * (*Optional*): BED-like output(s) of localized enrichment patterns across multiple resolutions, obtained with a genomics-oriented matched filtering variant, e.g., `ConsenrichMatchedResult(Het10, <template_name>)`

<p align="center">
  <img src="docs/matched.png" alt="Example output with --match_wavelet haar,db2,db4" width="800"/><br/>
  <em>Example: Consenrich-estimated signal tracks and uncertainty metrics given an input dataset consisting of $m=10$ ATAC-seq alignments of varying data quality (lymphoblastoid)
  <code>consenrich --bam_files ENCFF*.bam -g hg38 --match_wavelet haar,db2,db4</code></em>
</p>

---

## Download/Install

Consenrich is available via [PyPI/pip](https://pypi.org/project/consenrich/):

* `python -m pip install consenrich`

---

Consenrich can also be cloned and built from source:

1. `git clone https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python setup.py sdist bdist_wheel`
4. `python -m pip install .`

Check installation: `consenrich -h`

## Manuscript Preprint and Citation

A manuscript preprint is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.05.636702v1). *Up-to-date version forthcoming.*

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
