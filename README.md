# Consenrich

Consenrich estimates regulatory signals from multi-sample functional genomics datasets.

![Consenrich overview](docs/images/fig.png)

**Input:** Sequencing data (BAM files, fragments, etc.) from ATAC-seq, DNase-seq, ChIP-seq, CUT&RUN, and other functional genomics assays where multiple samples or replicates measure a shared regulatory signal but differ in local noise, artifacts, sequencing depth, assay quality, or biological heterogeneity.

**Output:** Consensus signal estimate tracks (bedGraph, bigWig), associated uncertainty tracks (bedGraph, bigWig), and optional consensus peak calls (narrowPeak, gappedPeak, BED).


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
