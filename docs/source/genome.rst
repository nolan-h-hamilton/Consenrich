Genome Configuration
--------------------

Use :confval:`genomeParams` to choose an assembly, chromosome sizes, fitting
chromosomes, normalization exclusions, and blacklist ranges.

.. role:: resourceUnavailable
   :class: resource-unavailable

.. list-table:: Built-in assembly resources
   :header-rows: 1
   :widths: auto
   :class: assembly-resources

   * - Assembly
     - Species
     - Chromosome sizes
     - Blacklist
     - Mappability
     - Assembly gaps
     - ``ccounts`` ref
   * - ``hg19``
     - *Homo sapiens*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["hg19"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["hg19"]

             # Mappability (50 bp)
             ccounts.Mappability("hg19", "chr1")

   * - ``hg38``
     - *Homo sapiens*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["hg38"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["hg38"]

             # Mappability (50 bp)
             ccounts.Mappability("hg38", "chr1")

   * - ``hs1``
     - *Homo sapiens*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["hs1"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["hs1"]

             # Mappability (50 bp)
             ccounts.Mappability("hs1", "chr1")

   * - ``mm10``
     - *Mus musculus*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["mm10"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["mm10"]

             # Mappability (50 bp)
             ccounts.Mappability("mm10", "chr1")

   * - ``mm39``
     - *Mus musculus*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["mm39"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["mm39"]

             # Mappability (50 bp)
             ccounts.Mappability("mm39", "chr1")

   * - ``rn6``
     - *Rattus norvegicus*
     - ✓
     - :resourceUnavailable:`✗`
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["rn6"]

             # Mappability (50 bp)
             ccounts.Mappability("rn6", "chr1")

   * - ``rn7``
     - *Rattus norvegicus*
     - ✓
     - :resourceUnavailable:`✗`
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["rn7"]

             # Mappability (50 bp)
             ccounts.Mappability("rn7", "chr1")

   * - ``rn8``
     - *Rattus norvegicus*
     - ✓
     - :resourceUnavailable:`✗`
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["rn8"]

             # Mappability (50 bp)
             ccounts.Mappability("rn8", "NC_086019.1")

   * - ``dm3``
     - *Drosophila melanogaster*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["dm3"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["dm3"]

             # Mappability (50 bp)
             ccounts.Mappability("dm3", "chr2L")

   * - ``dm6``
     - *Drosophila melanogaster*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["dm6"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["dm6"]

             # Mappability (50 bp)
             ccounts.Mappability("dm6", "chr2L")

   * - ``sacCer2``
     - *Saccharomyces cerevisiae*
     - ✓
     - :resourceUnavailable:`✗`
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["sacCer2"]

             # Mappability (50 bp)
             ccounts.Mappability("sacCer2", "chrI")

   * - ``sacCer3``
     - *Saccharomyces cerevisiae*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["sacCer3"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["sacCer3"]

             # Mappability (50 bp)
             ccounts.Mappability("sacCer3", "chrI")

   * - ``ce10``
     - *Caenorhabditis elegans*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["ce10"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["ce10"]

             # Mappability (50 bp)
             ccounts.Mappability("ce10", "chrI")

   * - ``ce11``
     - *Caenorhabditis elegans*
     - ✓
     - ✓
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["ce11"]

             # Blacklist
             ccounts.assemblyBlacklistRanges["ce11"]

             # Mappability (50 bp)
             ccounts.Mappability("ce11", "chrI")

   * - ``danRer11``
     - *Danio rerio*
     - ✓
     - :resourceUnavailable:`✗`
     - ✓
     - ✓
     - .. admonition:: References
          :class: resource-references
          :collapsible: closed

          .. code-block:: python

             import ccounts

             # Chromosome sizes
             ccounts.assemblyChromosomeSizes["danRer11"]

             # Mappability (50 bp)
             ccounts.Mappability("danRer11", "chr1")


The `ccounts <https://github.com/nolan-h-hamilton/ccounts>`_ package, which is a
dependency of Consenrich, provides these resources.

File formats: `chromosome-size files <https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/>`_
and `BED intervals <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`_.

Note, in the YAML config files, ``null`` results in use of the packaged resources in
`ccounts <https://github.com/nolan-h-hamilton/ccounts>`_:

.. code-block:: yaml
   :caption: Genome configuration example

   genomeParams:
     name: hg38  # use ccounts chromosome sizes and blacklist resources
     chromosomes: [chr1, chr2]
     excludeForNorm: [chrX, chrY]
