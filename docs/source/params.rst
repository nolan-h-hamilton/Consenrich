Parameter Guidance
------------------

Many parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

Estimation Controls
~~~~~~~~~~~~~~~~~~~

``countingParams.intervalSizeBP``
    * Sets the genomic bin size in base pairs. Default is ``50`` bp.

      * Smaller ``intervalSizeBP`` values, such as 25 bp or 10 bp, can provide
        higher-resolution results for deeply sequenced data.
      * Larger values (e.g., 100–500 bp) may be used for domain-level
        enriched regions in broad marks like H3K27me3 and are especially
        encouraged for shallow broad mark sequencing data.

.. _genomic-length-scale:

``observationParams.lengthScaleBP``
    * Sets the `length scale <https://arxiv.org/abs/1602.06318>`_ used to control
      the sensitivity and frequency response of the background sequence
      :math:`\{g_{[i]}\}_{i=1}^{n}`. This quantity is similar to
      `csaw's window size <https://bioconductor.org/books/release/csawBook/counting-reads-into-windows.html#sec:windowsize>`_
      and `F-Seq's feature length <https://pmc.ncbi.nlm.nih.gov/articles/PMC2732284/>`_.

    * If no value is specified (``null``), Consenrich attempts to use a
      multiscale search inspired by `MUSIC <https://github.com/gersteinlab/MUSIC>`_.
      A fast `infinitesimal jackknife <https://arxiv.org/abs/1806.00550>`_ over many
      sampled genomic blocks quantifies uncertainty in the estimate due to
      block sampling noise.

    * For a manual value, consider ``10000``--``25000`` bp for
      `narrow marks <https://www.encodeproject.org/glossary/#target-categories>`_
      (TF binding, chromatin accessibility, H2AFZ, H3ac, H3K27ac, H3K4me2,
      H3K4me3, H3K9ac) and
      ``25000``--``100000`` bp for
      `broad marks <https://www.encodeproject.org/glossary/#target-categories>`_
      (H3F3A, H3K27me3, H3K36me3, H3K4me1, H3K79me2, H3K79me3,
      H3K9me1, H3K9me2, H4K20me1).

``observationParams.lengthScaleOptions``
    .. admonition:: Length-scale estimation options
       :class: tip
       :collapsible: closed

       Options are supplied by `ccounts <https://github.com/nolan-h-hamilton/ccounts>`_
       as ``ccounts.LengthScaleOptions``. Omitted fields use ccounts defaults.
       For context, see also `MUSIC <https://github.com/gersteinlab/MUSIC>`_,
       whose filter width calculations provide the intuition for length-scale
       estimation in Consenrich.

       * **Genomic block sampling for UQ**

         Contiguous genomic blocks are first sampled at random, excluding
         assembly gaps and blacklist ranges. The infinitesimal jackknife
         quantifies the estimator's sampling variance.

         * ``blockCount``: maximum number of sampled blocks (default ``1024``).
         * ``blockWidthBP``: block width in bp (default ``500000``).
           Use a multiple of the bin size. The default rounds up automatically.
         * ``randomSeed``: sampling seed (default ``42``).
       * **Multiscale search**

         Smoothed count tracks are compared at candidate filter widths
         to identify the scales of enriched regions.

         * ``scaleStartBP``: smallest filter width. The default ``null`` uses
           three counting bins.
         * ``scaleEndBP``: largest filter width. The default ``null`` uses
           ``blockWidthBP``. Candidate widths are capped at the block width.
         * ``scaleFactor``: multiplier for nominal filter widths (default ``1.5``),
           e.g. :math:`150 \to 225 \to 337.5 \to \cdots` bp before rounding to
           odd bin counts.
         * ``filterType``: ``median`` (default) or ``mean``.

       * **Enrichment screening**

         The screening criteria determine which candidate regions contribute
         to the length-scale estimate.

         * ``smoothingRatio``: maximum ratio of the unsmoothed peak in the
           surrounding window to the smoothed candidate peak (default ``4.0``).
         * ``significanceLevel``: tail probability for Poisson trimming and
           the binomial test (default ``0.05``).
         * ``pValueWindowBP``: reference width for count scaling in longer
           candidates (default ``1500`` bp).
         * ``backgroundWindowBP``: local null-count window width. The default
           is one tenth of ``blockWidthBP``, bounded to 50000--5000000 bp and
           at least one counting bin.

       * **Uncertainty and output**

         * ``confidenceLevel``: confidence level for the interval
           (default ``0.95``).
         * ``returnProfiles``: include accepted-width profiles for each sample,
           block, and bin (default ``false``).
           Set to ``true`` for debugging and diagnostics.
         * The estimate and its uncertainty are saved in the ``lengthScale``
           JSON artifact.

``fitParams.backgroundLengthScaleMultiplier``
    * *This value is ignored if observationParams.lengthScaleBP is set manually
      to a numeric value.* Otherwise, it multiplies the length scale estimated
      from the data (default ``5.0``).
      Increasing this value above ``1.0`` helps ensure that enriched regions
      are preserved in the signal estimates :math:`\{x_{[i]}\}_{i=1}^{n}`.

    * The default lower and upper bounds for computed length scales are
      ``5000`` bp and ``100000`` bp. Estimated length scales outside the bounds
      are clipped and adjusted to an odd number of counting bins without
      exceeding the upper bound.

``countingParams.backgroundBlockSizeBP``
    * Positive integer widths bypass the multiplier and automatic bounds and
      round up to odd counting bins with a warning when adjusted, whereas
      nonpositive integers select automatic sizing.

``observationParams.precisionMultiplierMin`` and ``observationParams.precisionMultiplierMax``
    * Bound observation precision multipliers.
      The defaults are ``0.1`` and ``10.0``.

``processParams.precisionMultiplierMin`` and ``processParams.precisionMultiplierMax``
    * Bound process precision multipliers.
      The defaults are ``1.0e-4`` and ``10.0``.

    * Because :math:`Q_i=Q_0/\kappa_i`, the multipliers control the
      **local adaptive response**: smaller :math:`\kappa_i` allows faster variation
      and capture of sharp, abrupt peaks that may contend with the linear
      prior prediction. Larger :math:`\kappa_i` increases confidence in the prior
      process model and can thus favor smoother variation.

For context, Student-:math:`t` observation and process error models are described in
`Aravkin et al., Section 4 <https://arxiv.org/html/1303.5588v1#S4>`_.

``outputParams.stateShrinkageEnabled``
    * Enables a conservative posterior state shrinkage procedure on the final
      signal estimates using a spike-and-Student-:math:`t` prior.
      The shrinkage is performed on the Consenrich-estimated signal and
      uncertainty tracks, output as ``state`` and ``uncertainty``.

    * Disable it in the YAML config w/ ``outputParams.stateShrinkageEnabled: false``.

Peak Calling Controls
~~~~~~~~~~~~~~~~~~~~~

``matchingParams.peakMode``
    Specifies calling of narrow and/or broad marks.

    * ``narrow``

      Writes `UCSC narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_ calls.

    * ``broad``

      Writes `UCSC gappedPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format13>`_ calls.

    * ``both``

      (*Recommended*) Writes narrow and broad calls.

``matchingParams.selectionPenaltyMultiplier``
    * Scales the data-derived per-bin selection penalty or budget.
      The default is ``1.0``:

      * Values below ``1.0`` can increase the number of peaks and/or their span.
      * Values above ``1.0`` increase stringency and can reduce the number/breadth
        of peaks.

``matchingParams.minMeanSignal``
    * Require that candidate peaks additionally satisfy the criterion:

      .. math::

         \overline{x}_{I}
         = \frac{\sum_{i\in I} w_i x_{[i]}}{\sum_{i\in I} w_i}
         \ge \tau.

      Here :math:`I` contains bins in the candidate peak's full span,
      :math:`w_i` is bin width in bp, and :math:`\tau` is
      ``matchingParams.minMeanSignal``. The signal :math:`x_{[i]}` is
      posterior-shrunk when ``outputParams.stateShrinkageEnabled`` and
      ``matchingParams.useShrunkStateScores`` are enabled, and unshrunk
      otherwise.

    * The default is ``0.01``. Disable with ``null``.

``matchingParams.gamma``
    * Sets the penalty for boundaries between selected and unselected bins
      (default ``0.25``):

      .. math::

         P_{\gamma}(\boldsymbol{\ell})
         = \gamma\sum_{i=1}^{n-1}
           \left|\ell_{[i+1]}-\ell_{[i]}\right|,
         \qquad \boldsymbol{\ell}\in\{0,1\}^{n}.

      Here :math:`n` is the number of chromosome bins, and
      :math:`\ell_{[i]}=1` selects bin :math:`i`.
      This is also called the `total variation <https://arxiv.org/abs/1203.1828>`_
      penalty or `fused lasso penalty <https://jmlr.org/papers/v18/16-532.html>`_.

    * **Larger** values of :math:`\gamma`:

      * Broader peaks, stronger regularization, and merging of nearby regions.
      * May miss small-scale genomic features and lower peak resolution.

    * **Smaller** values of :math:`\gamma`:

      * Narrow peaks and stronger separation of proximal signals.
      * May overfit shallow and/or broad-mark sequencing data.


(**Advanced**) ``matchingParams.thresholdZ``
    * This parameter controls which regions are used to estimate the selection/budget penalty. Higher values restrict the estimation to more strongly enriched genomic regions. Note that this
      quantity need *not* change monotonically with respect to the final selection/budget penalty.

    * Lower values allow less-enriched regions to influence budget/penalty calibration. The default is `2.0`

