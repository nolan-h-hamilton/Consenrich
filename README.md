# Consenrich

Consenrich is a regularized estimator of genome-wide consensus signal in noisy multi-replicate HTS data.

The underlying method is a linear filter-smoother with explicit accounting for heteroskedasticity across replicates and loci.

The resulting estimates and uncertainty tracks can be analyzed directly or used downstream for consensus peak calling, model training, variant prioritization, differential analysis, and other tasks that require reliable high-resolution cohort-level signal estimates.

**Input:** Sequencing data (alignments, fragments, etc.) from ATAC-seq, DNase-seq, ChIP-seq, CUT&RUN, and other functional genomics assays where multiple samples or replicates measure a shared regulatory signal but differ in local noise, artifacts, sequencing depth, assay quality, or biological heterogeneity.

**Output:** Consensus signal estimate tracks (bedGraph, bigWig), associated uncertainty tracks (bedGraph, bigWig), and optional consensus peak calls (narrowPeak, BED).


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
