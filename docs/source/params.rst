Parameter Guidance
------------------

Most parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

Peak Calling Controls
~~~~~~~~~~~~~~~~~~~~~

``matchingParams.peakMode``
    Selects ROCCO export shape.

    ``narrow``
        Writes `UCSC narrowPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format12>`_ calls.

    ``broad``
        Writes `UCSC gappedPeak <https://genome.ucsc.edu/FAQ/FAQformat.html#format13>`_ calls.

    ``both``
        Writes narrow and broad calls. This is the default.

``matchingParams.thresholdZ``
    Sets a one-sided :math:`z`-score cutoff used to define the budget (max. proportion of genome called).
    Larger values will yield a smaller budget and fewer calls. Smaller values will yield a larger budget and more calls.
    The default value is `2.0`.


``matchingParams.minPeakScore``
    Filter selected peaks to those with an average signal above this threshold.


Uncertainty Score
"""""""""""""""""

``matchingParams.uncertaintyScoreMode``
    ``state`` uses the fitted state track directly. ``lower_confidence`` uses
    ``state - matchingParams.uncertaintyScoreZ * uncertainty`` to penalize regions
    where estimates are uncertain.

``matchingParams.uncertaintyScoreZ``
    Sets the multiplier used by ``lower_confidence`` scoring. Larger values
    penalize uncertain regions more strongly.

Estimation Controls
~~~~~~~~~~~~~~~~~~~

``countingParams.intervalSizeBP``
    Sets the genomic bin size in base pairs. The default is `50` and is appropriate for most cases.
    Higher-resolution results may be obtained using `25`, `10`, etc. For detecting domain-level enriched-regions in
    broad marks like H3K27me3, larger values (`100`, `250`, etc.) should suffice.

``fitParams.ECM_backgroundLengthScaleMultiplier``
    Sets the multiplier that converts the inferred background dependence scale
    into the background fitting span. Larger values smooth the shared
    background estimate more strongly.
