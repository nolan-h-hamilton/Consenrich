# Consenrich

Consenrich estimates shared latent regulatory signals from noisy, multi-sample epigenomic sequencing measurements 

![Consenrich overview](docs/images/fig.png)

**Input:** Sequencing data in BAM, 10x fragments, or other supported formats from bulk or single-cell ATAC-seq, DNase-seq, ChIP-seq, CUT&RUN, CUT&Tag, or related assays.

**Output:** Consensus signal estimate tracks (bedGraph, bigWig), associated uncertainty/background tracks (bedGraph, bigWig), and optional consensus peak calls (narrowPeak, gappedPeak, BED).


[**See the Documentation**](https://nolan-h-hamilton.github.io/Consenrich/).


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
