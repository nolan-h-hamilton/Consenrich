# Consenrich (branch: `lean`)

---

The `lean` branch introduces a substantial internal refactor that positions Consenrich for a long-term, stable, API. Underlying methodology and functionality remain unchanged, but the following improvements are introduced:

* Core methodological aspects are now self-contained, allowing users greater flexibility to separate preprocessing and primary analysis for contexts that may require unique normalization techniques, transformations of data, or other preprocessing steps.

* Consistent, documented naming conventions for modules, functions, and arguments.

* Performance upgrades — Several previous bottlenecks are now rewritten in Cython, and alignment-level processing is buffered to restrict and configure memory use.

After `lean` is merged into `main`, some previous interfaces will become deprecated but remain accessible through older tagged versions of Consenrich.

---

Consenrich is a sequential state estimator for extraction of genome-wide epigenetic signals in noisy, multi-sample high-throughput functional genomics datasets.

![Simplified Schematic of Consenrich.](docs/images/noise.png)

See the [Documentation (branch:`lean`)](https://nolan-h-hamilton.github.io/Consenrich/) for more details and usage examples.

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

### From Source

Building and installing from source is recommended to ensure compatibility across platforms and Python versions.

1. `git clone --single-branch --branch lean https://github.com/nolan-h-hamilton/Consenrich.git`
2. `python -m pip install setuptools wheel Cython build`
3. `python -m build`
4. `python -m pip install .`

### From PyPI

Consenrich distributes multiple [wheels](https://peps.python.org/pep-0427/) on PyPI for different Python versions and platforms. To install the latest version, run:

```bash
python -m pip install consenrich
```

### Previous Versions

To install a specific version of Consenrich, you can specify the version number in the pip install command, for example:

```bash
python -m pip install consenrich==0.1.13b1
```
