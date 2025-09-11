Consenrich Homepage
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :name: Consenrich Homepage
   :hidden:

   installation
   examples
   modules

Consenrich is a fast, adaptive, and explainable algorithm for estimating quantitative genome-wide signals hidden in noisy multi-sample HTS data.

.. image:: ../images/noise.png
  :alt: Simplified schematic of Consenrich
  :width: 85%
  :align: center

Special consideration is given to critical but often-overlooked aspects in genome-wide signal and uncertainty quantification, including:

- *Sample-specific and region-specific noise* across the genome, addressing both technical and biological sources that corrupt sequencing data.
- *Dependencies between proximal loci* for spatially consistent, recursive propagation of signals and uncertainty genome-wide.

These refinements grant immediate practical appeal to a wide array of downstream tasks such as differential analyses, MPRA design, and so on.

This documentation includes installation instructions, a variety of usage examples, and an API reference for the software implementation of Consenrich.



.. list-table::
   :widths: 30 50
   :header-rows: 1

   * - Resource
     - Link
   * - Manuscript Preprint
     - `bioRχiv <https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2>`_
   * - Source Code
     - `GitHub <https://github.com/nolan-h-hamilton/Consenrich>`_
   * - Documentation, Examples, etc.
     - `(This site) <https://nolan-h-hamilton.github.io/Consenrich/>`_
   * - Contact
     - Nolan <dοt> Hamilton <aτ> unc [dοt] <eḏu>


