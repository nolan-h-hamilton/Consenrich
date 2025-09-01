Consenrich Documentation
===========================

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :name: Consenrich Homepage
   :hidden:

   installation
   examples
   modules

*Consenrich* estimates genome-wide regulatory signal hidden in noisy, multi-sample HTS datasets.

.. image:: ../images/noise.png
  :alt: Simplified schematic of Consenrich
  :width: 85%
  :align: center

Consenrich explicitly models critical but often-overlooked aspects in genomic signal quantification:

- *Sample-specific and region-specific noise* across the genome, addressing both technical and biological sources that corrupt sequencing data.
- *Dependencies between proximal loci* for spatially consistent, recursive propagation of signals and uncertainty genome-wide.

These refinements grant immediate practical appeal to a wide array of downstream tasks requiring quantitative, uncertainty-calibrated analysis of shared regulatory signals.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Resource
     - Link
   * - Manuscript Preprint
     - `bioRχiv <https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2>`_
   * - Source Code
     - `GitHub <https://github.com/nolan-h-hamilton/Consenrich>`_
   * - Documentation, Examples, etc.
     - `(This site) <https://nolan-h-hamilton.github.io/Consenrich/>`_


