Parameter Guidance
------------------

Most parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

Fragment Inputs
~~~~~~~~~~~~~~~

``scParams.fragmentsUseReadSupport``
    * Defaults to ``false``, assigning each fragment row weight one. Set to
      ``true`` to weight counts and CPM/RPKM depth by column five (read support).

    * Each ``inputParams.samples`` entry can override ``fragmentsUseReadSupport``.

    * The `10x fragments format
      <https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/outputs/fragments-file>`_
      records read support as the unique read pair plus duplicate read pairs.

Sorted BGZF fragment inputs receive a `tabix <https://www.htslib.org/doc/tabix.html>`_
index automatically when no index exists. Plain text or ordinary gzip inputs must be BGZF-compressed with
`bgzip <https://www.htslib.org/doc/bgzip.html>`_ first.

Peak Calling Controls
~~~~~~~~~~~~~~~~~~~~~

``matchingParams.peakMode``
    Specifies calling of narrow and/or broad marks.

    * ``narrow``

      Writes `UCSC narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_ calls.

    * ``broad``

      Writes `UCSC gappedPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format13>`_ calls.

    * ``both``

      Writes narrow and broad calls.

``matchingParams.selectionPenaltyMultiplier``
    * Scales the data-derived per-bin selection penalty. The default is ``1.0``.

    * Values below ``1.0`` yield more relaxed selection. Values above ``1.0`` increase stringency.

``matchingParams.gamma``
    * Sets the penalty for boundaries between selected and unselected bins (default ``0.25``). Sometimes
      called the `total variation <https://arxiv.org/abs/1203.1828>`_ penalty in signal and image processing literature.

    * Larger values discourage fragmentation of selected regions, while smaller values allow more granular calls.

    * For instance, if calling TF ChIP-seq peaks for a punctate mark like Pol II, a ``matchingParams.gamma: 0.05`` may be appropriate.

``matchingParams.minMeanSignal``
    * Keeps candidate peaks whose mean signal reaches this threshold.

    * The default is ``0.01``. Disable with ``null``.


(**Advanced**) ``matchingParams.thresholdZ``
    * This parameter controls which regions are used to estimate the selection/budget penalty. Higher values restrict the estimation to more strongly enriched genomic regions. Note that this
      quantity need *not* change monotonically with respect to the final selection/budget penalty.

    * Lower values allow less enriched regions to influence budget/penalty calibration. The default is `2.0`


Estimation Controls
~~~~~~~~~~~~~~~~~~~

``countingParams.intervalSizeBP``
    * Sets the genomic bin size in base pairs. The default is `50` and is appropriate for most cases.

    * Higher-resolution results may be obtained using `25`, `10`, etc.

    * For detecting domain-level enriched-regions in
      broad marks like H3K27me3, larger values (`100`, `250`, etc.) should suffice.

``observationParams.precisionMultiplierMin`` and ``observationParams.precisionMultiplierMax``
    * Bound observation precision multipliers. The defaults are ``0.1`` and
      ``10.0``.

``processParams.precisionMultiplierMin`` and ``processParams.precisionMultiplierMax``
    * Bound process precision multipliers. The defaults are ``1.0e-4`` and
      ``10.0``.

    * Because :math:`Q_i=Q_0/\kappa_i`, the multipliers let the local
      frequency response adapt: smaller :math:`\kappa_i` admits faster variation,
      whereas larger :math:`\kappa_i` favors smoother variation.

``outputParams.stateShrinkageEnabled``
    * Enables an *experimental* posterior state shrinkage using a spike-and-Student-:math:`t`
      prior, with consenrich-estimated positional signal and uncertainty plugged-in.

    * Disable it in the YAML config w/ ``outputParams.stateShrinkageEnabled: false``.

    * Depending on the downstream application, the shrunk state/uncertainty tracks
      should be interpreted as approximate since the inputs are (fixed) estimates themselves,
      i.e., uncertainty is not propagated in the fully Bayesian sense.

    * Specifically, cutoffs on the shrunkState tracks (e.g., `matchingParams.minMeanSignal` for peaks)
      may become conservative relative to the original `state`.
