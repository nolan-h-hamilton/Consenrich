API Reference
=============

.. confval:: genomeParams

   Specify genome assembly resources.

   **Type:** a dictionary. Supply either ``name`` to use ``ccounts-packaged``
   resources or, alternatively, supply your own ``chromSizesFile`` and
   ``blacklistFile``.
   See :doc:`Genome Configuration <genome>` for packaged resource availability.

   .. list-table::
      :header-rows: 1
      :widths: 24 24 52

      * - Field
        - Type
        - Use / default
      * - ``name``
        - ``str | None``
        - Assembly identifier from ccounts, e.g. ``hg38``, ``mm39``, or ``hs1``. Default: ``null``.
      * - ``chromSizesFile``
        - ``str | None``
        - Two-column TSV of chromosome names and lengths in bp. Replaces the assembly chromosome sizes. Default: ``null`` uses ccounts sizes for assembly ``name``.
      * - ``blacklistFile``
        - ``str | None``
        - Blacklist BED override for fitting and peak filtering. Default: ``null`` uses the named assembly's blacklist. Supply a BED for assemblies without one.
      * - ``chromosomes``
        - ``list[str] | None``
        - Chromosomes to fit. Default: ``null`` uses the assembly's standard chromosomes, excluding sex-exclusive chromosomes such as ``chrY``, or all chromosomes in ``chromSizesFile``.
      * - ``excludeChroms``
        - ``list[str]``
        - Chromosomes removed from fitting. Default: ``[]``.
      * - ``excludeForNorm``
        - ``list[str]``
        - Chromosomes removed from the depth-normalization territory. Default: ``[]``. This selection is separate from the fitting chromosomes.
      * - ``effectiveGenomeSize``
        - ``int | None``
        - Positive effective genome size in bp for automatic RPGC normalization. Default: ``null`` uses assembly-specific estimates. Custom BAM chromosome geometry or normalization territory requires an explicit matching size.

.. confval:: observationParams

   Observation variance estimation and robust precision weighting.

   **Type:** a dictionary. Field names below are relative to
   ``observationParams``. See :doc:`Parameter Guidance <params>` for scale
   selection guidance.

   .. list-table::
      :header-rows: 1
      :widths: 28 20 52

      * - Field
        - Type
        - Use / default
      * - ``lengthScaleBP``
        - ``int | None``
        - Positive characteristic genomic length scale in bp. Default: ``null`` estimates a scale from the data. See :ref:`length-scale guidance <genomic-length-scale>`.
      * - ``lengthScaleOptions``
        - ``dict``
        - Options for automatic scale estimation in ccounts. Default: ``{}``. Subfields are listed below.
      * - ``precisionMultiplierMin``
        - ``float``
        - Lower bound for robust observation precision multipliers. Default: ``0.1``.
      * - ``precisionMultiplierMax``
        - ``float``
        - Upper bound for robust observation precision multipliers. Default: ``10.0``.
      * - ``muncTrendBlockSizeBP``
        - ``int | None``
        - Positive block width in bp for pooled mean-variance trend evidence. Default: ``null`` derives it from the genomic scale and ``muncTrendBlockLengthScaleMultiplier``. Capped at 1 Mb.
      * - ``muncLocalWindowSizeBP``
        - ``int | None``
        - Positive local variance window width in bp. Default: ``null`` derives it from the genomic scale and ``muncLocalWindowLengthScaleMultiplier``. Capped at 1 Mb.
      * - ``muncTrendBlockLengthScaleMultiplier``
        - ``float``
        - Positive multiplier for automatic trend-block sizing. Default: ``2.0``. An explicit ``muncTrendBlockSizeBP`` overrides it.
      * - ``muncLocalWindowLengthScaleMultiplier``
        - ``float``
        - Positive multiplier for automatic local variance windows. Default: ``4.0``. An explicit ``muncLocalWindowSizeBP`` overrides it.
      * - ``useCountNoiseFloor``
        - ``bool``
        - Add a regularized delta-method estimate of count-sampling variance to the estimated excess observation variance. Default: ``true``.
      * - ``useReplicateVarianceScale``
        - ``bool``
        - Apply fitted replicate-specific variance factors to the pooled variance trend. Default: ``true``.
      * - ``EB_setNu0``
        - ``float | None``
        - Prior degrees-of-freedom override when at least ``4``. Default: ``null`` estimates prior strength per sample from the dispersion of log ratios of local variance to trend variance over genomic blocks, adjusted for local estimation noise.
      * - ``EB_setNuL``
        - ``float | None``
        - Fixed local degrees of freedom when greater than ``3``. Default: ``null`` derives them from the local window and genomic scale.
      * - ``trendLambdaMin``
        - ``float``
        - Lower endpoint of the spline smoothing-parameter grid. Default: ``1.0e-6``.
      * - ``trendLambdaMax``
        - ``float``
        - Upper endpoint of the spline smoothing-parameter grid. Default: ``1.0e6``.
      * - ``trendLambdaGridSize``
        - ``int``
        - Number of logarithmically spaced smoothing parameters tested by guarded GCV. Default: ``101``.

   .. admonition:: Automatic scale estimation options
      :class: tip
      :collapsible: closed

      Subfields of ``observationParams.lengthScaleOptions``, supplied by
      `ccounts <https://github.com/nolan-h-hamilton/ccounts>`_.

      .. list-table::
         :header-rows: 1
         :widths: 28 20 52

         * - Field
           - Type
           - Use / default
         * - ``binSizeBP``
           - ``int | None``
           - Resolved from ``countingParams.intervalSizeBP``.
         * - ``blockCount``
           - ``int``
           - Maximum number of randomly sampled genomic blocks. Default: ``1024``.
         * - ``blockWidthBP``
           - ``int``
           - Positive sampled block width in bp, divisible by the bin size. Default: ``500000`` rounded up to the count grid.
         * - ``scaleStartBP``
           - ``int | None``
           - Smallest nominal filter width in bp. Default: ``null`` uses three bins. Explicit values also use at least three bins.
         * - ``scaleEndBP``
           - ``int | None``
           - Largest nominal filter width in bp, capped at the block width. Default: ``null`` uses ``blockWidthBP``.
         * - ``scaleFactor``
           - ``float``
           - Multiplier greater than one for nominal filter widths. Default: ``1.5``. Realized widths use odd bin counts.
         * - ``filterType``
           - ``str``
           - Smoothing filter: ``median`` or ``mean``. Default: ``median``.
         * - ``smoothingRatio``
           - ``float``
           - Maximum ratio of the unsmoothed peak in the surrounding window to the smoothed candidate peak. Must be at least one. Default: ``4.0``.
         * - ``significanceLevel``
           - ``float``
           - Tail probability for Poisson trimming and the binomial enrichment test. Default: ``0.05``.
         * - ``pValueWindowBP``
           - ``int``
           - Positive reference width in bp for count scaling in longer candidate regions. Default: ``1500``.
         * - ``backgroundWindowBP``
           - ``int | None``
           - Positive local null-count window width in bp. Default: ``null`` uses one tenth of the block width, bounded to 50000-5000000 bp and at least one bin.
         * - ``confidenceLevel``
           - ``float``
           - Confidence level for the length-scale interval. Default: ``0.95``.
         * - ``randomSeed``
           - ``int``
           - Nonnegative genomic block sampling seed. Default: ``42``.
         * - ``returnProfiles``
           - ``bool``
           - Include the largest accepted filter width for each sample, block, and bin in the scale artifact. Default: ``false``.

.. confval:: countingParams

   Genomic binning, normalization, and local centering.

   **Type:** a dictionary.

   .. list-table::
      :header-rows: 1
      :widths: 24 24 52

      * - Field
        - Type
        - Use / default
      * - ``intervalSizeBP``
        - ``int``
        - Positive genomic bin width in bp, shared by all sources. Default: ``50``.
      * - ``backgroundBlockSizeBP``
        - ``int``
        - Positive widths override automatic background smoothing and round up to an odd number of counting bins. Nonpositive values select automatic sizing. Default: ``-1``.
      * - ``scaleFactors``
        - ``list[float] | None``
        - Positive treatment-count multipliers in input order. Supply one value to reuse for every treatment, or one per treatment. Default: ``null`` derives factors automatically. See the :ref:`H5AD example <rpgc-h5ad-example>`.
      * - ``scaleFactorsControl``
        - ``list[float] | None``
        - Positive control-count multipliers in control input order. Supply one value to reuse for every control, or one per control. Default: ``null`` derives factors automatically.
      * - ``normMethod``
        - ``str``
        - Automatic normalization method: ``RPGC``, ``RPKM``, ``CPM``, ``SF``, or ``ReadsInRegions``. Default: ``CPM`` when any input is H5AD, otherwise ``RPGC``. Supplied factors replace automatic factor derivation for their inputs.
      * - ``normRegionsFile``
        - ``str | None``
        - BED file for ``ReadsInRegions``. Required when deriving factors. Default: ``null``.
      * - ``normScaleTo``
        - ``float``
        - Positive endpoint-count target for ``ReadsInRegions``. Default: ``10000.0``.
      * - ``fixControl``
        - ``bool``
        - Controls automatic paired scaling in assays that supply a *treatment* readout and a *control* readout, e.g., ChIP-seq. ``true`` leaves controls unscaled and only downscales the *treatment* data when needed. ``false`` downscales the deeper input to the shallower input. Default: ``false``.
      * - ``pseudocount``
        - ``float``
        - Positive pseudocount added before log2 transformation. Default: ``1.0``.

   ``ReadsInRegions`` merges overlapping BED intervals and scales filtered 5′
   endpoint counts to ``normScaleTo`` per input. It supports BAM and FRAGMENTS inputs,
   scaling treatment and control inputs independently.

.. _single-cell-api:

.. confval:: scParams.cellMetadata

   This mapping specifies which cells belong to each sample and cell group.

   **Type:** a dictionary specifying the file path and column names for a CSV,
   TSV, or AnnData ``.obs`` source.

   The ``path`` can be one of

   - CSV/TSV file (optionally gzipped): contains cell metadata with columns for barcodes (``barcodeColumn``), sample IDs (``sampleColumn``), and group labels (``groupColumn``)
   - `AnnData <https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html>`_
     (`.h5ad <https://anndata.readthedocs.io/en/stable/fileformat-prose.html>`_) file:
     contains cell metadata in the ``.obs`` table.

   .. list-table::
      :header-rows: 1
      :widths: 22 20 58

      * - Field
        - Type
        - Use
      * - ``path``
        - ``str``
        - Required. A ``.csv``, ``.tsv``, or ``.h5ad`` file. CSV/TSV may use ``.gz``.
      * - ``barcodeColumn``
        - ``str | None``
        - Cell identifier column. Required for CSV/TSV. **Omit** this for AnnData/H5AD so that ``.obs_names`` is used
      * - ``sampleColumn``
        - ``str | None``
        - Column used to match metadata rows to an input sample, e.g. ``library`` below. Required when the metadata file covers multiple input samples.
      * - ``groupColumn``
        - ``str | None``
        - Cell-group column, such as ``cellType``. Required for group selection.

   **Example**.

   .. list-table:: cells.csv
      :header-rows: 1

      * - barcode
        - library
        - cellType
      * - AAACGAAAGTACGCCC-1
        - PBMC_01
        - CD4_T
      * - AAACGAACAGCTACAT-1
        - PBMC_01
        - B_cell
      * - AAACGAAGTCGTTGGC-1
        - PBMC_01
        - NK_cell
      * - AAACGAAAGTACGCCC-1
        - PBMC_02
        - CD8_T
      * - AAACGAATCTTACGGA-1
        - PBMC_02
        - B_cell
      * - AAACGAAGTACAGCGA-1
        - PBMC_02
        - CD4_T

   For the ``cells.csv`` table above, specify the column names
   under ``scParams.cellMetadata`` in the Consenrich YAML:

   .. code-block:: yaml

      scParams:
        cellMetadata:
          path: cells.csv
          barcodeColumn: barcode
          sampleColumn: library
          groupColumn: cellType

   With ``sampleColumn: library``, an input named ``PBMC_01`` is matched to the
   three rows labelled ``PBMC_01``. If that input is named ``replicate1``, you would set
   ``metadataSample: PBMC_01`` on the input to keep the same match.

.. confval:: scParams.fits

   Use this to specify multiple Consenrich jobs for different cell groups within the same dataset, *in one YAML file*.
   See, for example, the :ref:`scATAC-seq example <scatac-demo>`.

   **Type:** ``list[dict[str, str | list[str]]] | None``.

   **Default:** ``None`` (one run without a fit list).

   Each entry requires ``name`` and ``selectGroups``:

   * ``name`` (``str``) names the output directory, e.g. ``TCells``. Must be unique
   * ``selectGroups`` (``list[str]``) holds one or more unique labels from
     ``groupColumn``, matched exactly

   .. code-block:: yaml

      # ...other consenrich config...


      scParams:
        fits:
          - {name: TCells, selectGroups: [CD4_T, CD8_T]} # Run a fit for T cells
          - {name: BCells, selectGroups: [B_cell]}       # and run a fit for B cells, too

.. confval:: inputParams.samples

   Input files and source-specific settings.

   **Type:** ``list[str | dict[str, Any]] | None``.
   **Default:** ``None``.

   Supply a nonempty list. H5AD counts require a mapping with ``matrixRows``.

   .. list-table::
      :header-rows: 1
      :widths: 24 28 48

      * - Field
        - Type
        - Use / default
      * - ``path``
        - ``str``
        - Required. E.g. ``library1.fragments.tsv.gz`` or ``library1.h5ad``.
      * - ``name``
        - ``str | None``
        - Output label and default metadata sample identifier. Default: ``None``.
      * - ``cellMetadata``
        - ``dict[str, str | None] | None``
        - Replaces :confval:`scParams.cellMetadata` for that source. Default: ``None`` uses the shared setting.
      * - ``metadataSample``
        - ``str | None``
        - Value to select from ``sampleColumn``, e.g. ``library1``. Default: ``None`` uses ``name``.
      * - ``matrixRows``
        - ``str | None``
        - Required for H5AD counts. ``cells`` sums selected rows. ``pseudobulks`` selects exactly one row per source and fit. Default: ``None``.
      * - ``countMode``
        - ``str | None``
        - Fragment counting override. Default: ``None`` uses ``scParams.defaultCountMode``. Rejected for H5AD counts.
      * - ``fragmentsUseReadSupport``
        - ``bool``
        - ``True`` weights fragment rows by column five. Inherits ``scParams.fragmentsUseReadSupport``, which defaults to ``False``.

   Fragment ``countMode`` choices:

   * ``coverage``: Adds overlap length divided by bin width.
   * ``fiveprime``: Counts the two endpoints of each fragment.
   * ``center``: Counts each fragment in the bin containing its midpoint.
   * ``conservedFractionalOverlap``: Adds overlap length divided by fragment length.

   For a fragment spanning ``[5, 25)`` with unit weight and no shifts:

   .. code-block:: text

      bp                          0         10        20        30
      bins                        |---------|---------|---------|
      fragment                         [===================)
      read1                            5'--->
      read2                                           <---5'

      countMode                    [0, 10)   [10, 20)  [20, 30)
      coverage                       0.5       1.0       0.5
      conservedFractionalOverlap     0.25      0.50      0.25
      center                          0         1         0
      fiveprime                       1         0         1

.. _h5ad-count-requirements:

H5AD Input
~~~~~~~~~~

* ``.X`` contains raw, nonnegative integer counts. Dense, CSR, and CSC storage
  are supported, including integer-valued floats.
* ``.var`` contains ``chrom``, ``start``, and ``end`` for complete, regular
  chromosome grids in 0-based, half-open coordinates. Terminal bins may be shorter.
* Set ``countingParams.intervalSizeBP`` (default ``50`` bp) to the bin width used
  to construct the H5AD matrix, e.g. ``1000`` for 1 kb bins. Existing H5AD bins
  are used directly and must match the chromosome grid described above. Reads
  and fragments are counted onto that same grid.

Normalization uses ``countingParams.normMethod``, which defaults to ``CPM``
when any input is H5AD. H5AD supports ``CPM``, ``RPKM``, or supplied scale
factors. To apply RPGC normalization on H5AD input data, scale factors must be
supplied explicitly.

Automatic library totals use normalization chromosomes shared by retained inputs
whose factors are being derived, after applying ``genomeParams.excludeForNorm``.
The resolved chromosomes are recorded in the normalization output. Retained H5AD
inputs must contain the requested fitting chromosomes.

.. _rpgc-h5ad-example:

In this example, ``0.8`` multiplies counts from ``sampleA.h5ad`` and ``1.2``
multiplies counts from ``sampleB.h5ad``. The factor order matches the samples
in :confval:`inputParams.samples`:

.. code-block:: yaml

   inputParams:
     samples:
       - name: sampleA
         path: sampleA.h5ad
         matrixRows: cells
       - name: sampleB
         path: sampleB.h5ad
         matrixRows: cells

   countingParams:
     normMethod: RPGC
     scaleFactors:
       - 0.8  # sampleA
       - 1.2  # sampleB

See the :ref:`single-cell examples <scatac-demo>` for complete configurations.
