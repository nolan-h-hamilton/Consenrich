# Consenrich (`lean` branch)

---
The `lean` branch introduces a substantial internal refactor that positions Consenrich for a long-term, stable, API.

* Core methodological aspects are now self-contained, allowing users greater flexibility to separate preprocessing and primary analysis for contexts that may require unique normalization techniques, transformations of data, or other preprocessing steps.

* Consistent, documented naming conventions for modules, functions, and arguments.

* Performance upgrades — Several previous bottlenecks are now rewritten in Cython, and alignment-level processing is buffered to restrict and configure memory use.

After `lean` is merged into `main`, some previous interfaces will become deprecated but remain accessible through older tagged versions of Consenrich. Note that `lean` does not introduce any fundamental methodological changes.

---

![Simplified Schematic of Consenrich.](docs/images/noise.png)

See the [Documentation](https://nolan-h-hamilton.github.io/Consenrich/) for more details and usage examples.

---

## Installation

1. `git clone --single-branch --branch lean https://github.com/nolan-h-hamilton/Consenrich.git`
2. `cd Consenrich`
3. `python -m pip install build .`


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
