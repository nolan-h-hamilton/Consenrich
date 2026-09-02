Parameter Guidance
------------------

Most parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

Peak Calling Controls
~~~~~~~~~~~~~~~~~~~~~

``matchingParams.peakMode``
    Specifies calling of narrow and/or broad marks.

    ``narrow``
        Writes `UCSC narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_ calls.

    ``broad``
        Writes `UCSC gappedPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format13>`_ calls.

    ``both``
        Writes narrow and broad calls.

    ``narrow`` is the default.

``matchingParams.thresholdZ``
    Sets a one-sided :math:`z`-score cutoff used to define the budget (max. proportion of genome called).
    Larger values will yield a smaller budget and fewer calls. Smaller values will yield a larger budget and more calls.
    The default value is `2.0`.


``matchingParams.minMeanSignal``
    Keeps regions whose covered-BP-weighted mean signal reaches the supplied
    descriptive threshold. Disable with ``null``.

``matchingParams.mergeToleranceBP``
    Sets the maximum gap eligible for broad-family merging. A positive value
    is required for broad and combined modes.

``matchingParams.maxRegionBP``
    Sets the maximum outer width for a broad family. A positive value is
    required for broad and combined modes.


Estimation Controls
~~~~~~~~~~~~~~~~~~~

``countingParams.intervalSizeBP``
    Sets the genomic bin size in base pairs. The default is `50` and is appropriate for most cases.
    Higher-resolution results may be obtained using `25`, `10`, etc. For detecting domain-level enriched-regions in
    broad marks like H3K27me3, larger values (`100`, `250`, etc.) should suffice.

``observationParams.precisionMultiplierMin`` and ``observationParams.precisionMultiplierMax``
    Bound observation precision multipliers. The defaults are ``0.1`` and
    ``10.0``.

``processParams.precisionMultiplierMin`` and ``processParams.precisionMultiplierMax``
    Bound process precision multipliers. The defaults are ``1.0e-4`` and
    ``10.0``. Because :math:`Q_i=Q_0/\kappa_i`, the multipliers let the local
    frequency response adapt: smaller :math:`\kappa_i` admits faster variation,
    whereas larger :math:`\kappa_i` favors smoother variation.

``outputParams.stateShrinkageEnabled``
    Enables experimental posterior state shrinkage and defaults to ``True``.
    Set it to ``False`` to disable it. Use the shrunk state as a visualization
    or ranking track, rather than as a replacement for the fitted state.
