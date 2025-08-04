# Consenrich

Consenrich is a sequential state estimator for extraction of genome-wide epigenetic signals in noisy, multi-sample high-throughput functional genomics datasets.

![Simplified Schematic of Consenrich.](docs/images/noise.png)

See the [Documentation](https://nolan-h-hamilton.github.io/Consenrich/) for more details and usage examples.

---

## Manuscript Preprint and Citation

A manuscript preprint is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2).

**BibTeX Citation**

```bibtex
@article {Hamilton2025,
	author = {Hamilton, Nolan H and Huang, Yu-Chen E and McMichael, Benjamin D and Love, Michael I and Furey, Terrence S},
	title = {Genome-Wide Uncertainty-Moderated Extraction of Signal Annotations from Multi-Sample Functional Genomics Data},
	year = {2025},
	doi = {10.1101/2025.02.05.636702},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

## Installation

The following steps should be most platform-independent and flexible, but you can also install from PyPI with `pip install consenrich`.

Note, if you don't have package management tools installed, you can first run

```bash
python -m pip install setuptools wheel Cython build
```

1. `git clone https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python -m build`
4. `python -m pip install .`
