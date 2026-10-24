Parameter Guidance
------------------

Most parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

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
    * Sets the genomic bin size in base pairs. Read and fragment inputs default to ``50``.

    * Higher-resolution results may be obtained for deeply sequenced data using
      smaller ``intervalSizeBP``: 25 bp, 10 bp, etc. For detecting domain-level
      enriched-regions in broad marks like H3K27me3, larger values (100, 250, etc.)
      can be used. Larger ``intervalSizeBP`` is encouraged for shallow sequencing
      data.

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
