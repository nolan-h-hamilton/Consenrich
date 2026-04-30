Consenrich
===========================

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :name: Consenrich Homepage
   :hidden:

   installation
   examples
   modules

Consenrich is a regularized estimator of genome-wide consensus signal in noisy multi-replicate HTS data.

The underlying method is a linear filter-smoother with explicit accounting of heteroskedasticity across replicates and loci.

The resulting estimates and uncertainty tracks can be analyzed directly or used downstream for consensus peak calling, model training, variant prioritization, differential analysis, and other tasks that require reliable high-resolution cohort-level signal estimates.

**Input:** Sequencing data (BAM files, fragments, etc.) from ATAC-seq, DNase-seq, ChIP-seq, CUT&RUN, and other functional genomics assays where multiple samples or replicates measure a shared regulatory signal but differ in local noise, artifacts, sequencing depth, assay quality, or biological heterogeneity.

**Output:** Consensus signal estimate tracks (bedGraph, bigWig), associated uncertainty tracks (bedGraph, bigWig), and optional consensus peak calls (narrowPeak, BED).


.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Resource
     - Link
   * - Manuscript Preprint
     - `bioRxiv <https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2>`_
   * - Source Code
     - `GitHub <https://github.com/nolan-h-hamilton/Consenrich>`_
   * - Documentation, Examples, etc.
     - `(This site) <https://nolan-h-hamilton.github.io/Consenrich/>`_
   * - Contact
     - Nolan [dot] Hamilton <at> unc [dot] edu
