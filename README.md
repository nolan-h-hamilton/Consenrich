# Consenrich

Consenrich is an adaptive linear state estimator that yields genome-wide, uncertainty-calibrated signal tracks from noisy multi-sample cohorts' epigenetic HTS data.

<p align="center">
  <img src="docs/images/noise.png" alt="Simplified Schematic of Consenrich." width="600">
</p>


Special emphasis is placed on computational efficiency, model interpretability, and practical utility for downstream tasks that require well-resolved genome-wide signal estimates and uncertainty quantification across samples, such as:

* Consensus detection of open chromatin regions, TF binding, histone modification, etc.
* Candidate prioritization for differential analyses, functional validation, integrative modeling, etc.

[**See the Documentation**](https://nolan-h-hamilton.github.io/Consenrich/) for usage examples, installation details, configuration options, and an API reference.


## Manuscript Preprint and Citation

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
