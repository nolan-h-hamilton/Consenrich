# Consenrich

[![Tests](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml/badge.svg?event=workflow_dispatch)](https://github.com/nolan-h-hamilton/Consenrich/actions/workflows/Tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/consenrich?logo=Python&logoColor=%23FFFFFF&color=%233776AB&link=https%3A%2F%2Fpypi.org%2Fproject%2Fconsenrich%2F)

[Consenrich](https://github.com/nolan-h-hamilton/Consenrich) is a sequential genome-wide state estimator for extraction of epigenomic signals and pattern localization in noisy multisample HTS data.

---

* **Input**:
  * $m \geq 1$ Sequence alignment files `-t/--bam_files` corresponding to each sample in a given HTS experiment
  * (*Optional*): $m_c = m$ control sample alignments, `-c/--control_files`, for each 'control' sample (e.g., ChIP-seq)
  * (*Optional*): wavelet-based template(s) to match for genome-wide pattern matching (`--match_wavelet ` `db<2,3,...>`, `sym<2,3,...>`, `haar`, `coif<1,2,...>`, `dmey`)

* **Output**:
  * Genome-wide 'consensus' epigenomic state estimates and uncertainty metrics (BedGraph/BigWig)
  * (*Optional*) BED-like output of relative maxima in the *response sequence* (convolution) with wavelet-based template(s)

<p align="center">
  <img src="docs/matched.png" alt="Example output with --match_wavelet haar,db4" width="800"/><br/>
  <em>Example: Consenrich outputs given 10 input ATAC-seq samples (lymphoblastoid)
  <code>consenrich --bam_files ENCFF*.bam -g hg38 --match_wavelet haar,db4</code></em>
</p>

---

## Download/Install

Consenrich is available via [PyPI/pip](https://pypi.org/project/consenrich/):

* `python -m pip install consenrich`

If lacking administrative privileges, running with flag `--user` may be necessary.

---

Consenrich can also be easily downloaded and installed from source:

1. `git clone https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python setup.py sdist bdist_wheel`
4. `python -m pip install .`
5. Check installation: `consenrich --help`

## Manuscript Preprint and Citation

A [corresponding manuscript preprint is available on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.05.636702v1).

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
