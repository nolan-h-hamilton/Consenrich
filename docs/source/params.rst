Parameter Guidance
------------------

Most parameters can be left at their default values. The following sections provide guidance on key arguments that may be useful to adjust for specific applications.

Counting Presets
~~~~~~~~~~~~~~~~

``countingPreset``
    Optional, case-sensitive top-level selector for an assay counting recipe.
    The exact accepted values are ``atac``, ``chip-se``, ``chip-pe``,
    ``dnase``, ``cut-and-run``, and ``cut-and-tag``. For example:

    .. code-block:: yaml

      countingPreset: atac

    Omitting ``countingPreset`` keeps the package defaults. Invalid types and
    values are errors. Values are not lower-cased and no alternate spellings
    are accepted. ``configuration: atac`` is not an alias and is rejected with
    a hint to use ``countingPreset: atac``.

    Recipe values have the following priority, from highest to lowest:

    #. A source entry's ``countMode`` or ``bamInputMode``, where that source
       setting exists.
    #. An explicitly configured global parameter.
    #. The ``countingPreset`` recipe.
    #. The package or configuration-profile default.

    Thus a preset supplies omitted fields and does not replace explicit
    choices. The selector does not set ``countingParams.intervalSizeBP``,
    transforms, ``countingParams.centerMB``, control pairing,
    ``countingParams.fragmentsGroupNorm``, ``countingParams.fixControl``, or
    fitting parameters.

    If an active recipe field is written in dotted and nested YAML forms, the
    values must agree. An explicit unsupported ``countingParams.normMethod``
    is also an error when a recipe is active.

    .. list-table:: Exact preset recipes
      :header-rows: 1
      :widths: 11 10 12 12 11 12 13 15

      * - Preset
        - ``bamInputMode``
        - BAM ``defaultCountMode``
        - FRAGMENTS ``defaultCountMode``
        - ``shiftForward5p`` / ``shiftReverse5p``
        - ``extendFrom5pBP`` / ``inferFragmentLength``
        - ``samFlagExclude`` / ``minMappingQuality``
        - ``minTemplateLength`` / ``maxInsertSize``
      * - ``atac``
        - ``reads``
        - ``cutsite``
        - ``cutsite``
        - ``4`` / ``5``
        - ``null`` / ``0``
        - ``3844`` / ``30``
        - Not applied in read mode
      * - ``chip-se``
        - ``reads``
        - ``coverage``
        - ``coverage``
        - ``0`` / ``0``
        - ``null`` / ``1``
        - ``3844`` / ``30``
        - 1,000-bp inference cap, template filter not applied
      * - ``chip-pe``
        - ``fragments``
        - ``coverage``
        - ``coverage``
        - ``0`` / ``0``
        - ``null`` / ``0``
        - ``3844`` / ``30``
        - 1--1,000 bp
      * - ``dnase``
        - ``auto``
        - ``fiveprime``
        - ``fiveprime``
        - ``0`` / ``0``
        - ``null`` / ``0``
        - ``3844`` / ``30``
        - 1--1,000 bp in fragment mode
      * - ``cut-and-run``
        - ``fragments``
        - ``coverage``
        - ``coverage``
        - ``0`` / ``0``
        - ``null`` / ``0``
        - ``2820`` / ``20``
        - 10--1,000 bp
      * - ``cut-and-tag``
        - ``fragments``
        - ``coverage``
        - ``coverage``
        - ``0`` / ``0``
        - ``null`` / ``0``
        - ``2820`` / ``20``
        - 10--1,000 bp

    ``defaultCountMode`` in the table refers to
    ``samParams.defaultCountMode`` for BAM and ``scParams.defaultCountMode``
    for FRAGMENTS. Every recipe also sets
    ``countingParams.normMethod: CPM``, ``samParams.oneReadPerBin: 0``,
    ``samParams.extendFrom5pBP: null``, and
    ``observationParams.smoothToFraglen: false``. ``samFlagExclude: 3844``
    rejects unmapped, secondary, QC-failed, duplicate, and supplementary
    alignments. ``samFlagExclude: 2820`` leaves out the duplicate flag, so the
    two CUT recipes retain duplicate-marked alignments.

    ``atac`` leaves ``samParams.minTemplateLength`` and
    ``samParams.maxInsertSize`` unset by the recipe because template-length
    filtering does not act in ``reads`` mode. ``chip-se`` leaves
    ``minTemplateLength`` unset and uses ``maxInsertSize: 1000`` only as the
    fragment-length inference cap. In ``coverage`` mode, a retained read or
    fragment adds one count to every genomic bin it touches. The CPM divisor
    counts retained reads or fragments, so a longer interval can occupy more
    bins.

    For ``atac``, ``shiftForward5p: 4`` moves the forward 5-prime coordinate
    four bases right. ``shiftReverse5p: 5`` moves the reverse coordinate five
    bases left. The shifts follow the Tn5 offset convention used by the `ENCODE
    ATAC-seq pipeline
    <https://www.encodeproject.org/documents/c008d7bd-5d60-4a23-a833-67c5dfab006a/@@download/attachment/ATACSeqPipeline.pdf>`_.
    The BAM recipe counts reads independently, so paired ATAC BAM input should
    already contain only the proper-pair alignments intended for counting.
    A paired BAM produces an advisory warning about that requirement.

    The ``dnase`` recipe applies no strand shift. It counts the unshifted
    5-prime ends that mark nuclease cleavage sites, as in the DNase-I pipeline
    described by `Vierstra et al.
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC4772017/>`_. ``auto`` uses
    fragment endpoints for paired BAM and per-read 5-prime ends for
    single-end BAM.

    ``chip-se`` infers a characteristic fragment length and extends each
    retained single-end read from its shifted 5-prime end. ``chip-pe`` uses
    measured proper-pair fragment spans. The geometries match the BAM and BAMPE
    geometries described in the `MACS3 format documentation
    <https://macs3-project.github.io/MACS/docs/SAMBAMBAMPE.html>`_. A
    ``chip-se`` preset with paired BAM warns that the two mates are extended
    independently. A ``chip-pe`` preset with single-end BAM warns that
    fragment counting may yield no proper templates. Each warning keeps the
    resolved recipe.

    CUT presets use proper-pair fragment coverage and retain duplicate-marked
    alignments. This follows the paired-end fragment treatment in
    `CUT&RUNTools <https://link.springer.com/article/10.1186/s13059-019-1802-4>`_
    and the duplicate-retaining analysis in the `original CUT&Tag workflow
    <https://www.nature.com/articles/s41467-019-09982-5>`_. A CUT preset with
    single-end BAM warns that fragment counting may yield no proper templates
    and keeps the resolved recipe.

    FRAGMENTS coordinates are consumed as zero-origin, half-open ``[start, end)``
    spans. Endpoint recipes count ``start`` and ``end - 1``.
    BAM shifts, MAPQ, SAM flags, and template-length limits do not act on a
    FRAGMENTS source, and a preset emits an advisory warning to that effect.
    For ATAC, fragment bounds must already denote Tn5 insertion coordinates,
    as in the `10x
    Genomics fragments format
    <https://www.10xgenomics.com/support/software/cell-ranger-arc/latest/tutorials/outputs/fragments-file>`_.
    Consenrich does not apply the ``+4/-5`` BAM correction to such a file. For
    ChIP and CUT coverage, bounds must denote the intended physical fragment
    span. For DNase endpoint counting, bounds must denote the intended cut
    endpoints. ``fragmentPositionMode`` checks the coordinate declaration but
    does not shift or translate bounds. A fifth-column ``readSupport`` value
    weights the fragment, so a collapsed fragments file can retain its
    recorded multiplicity.

    CPM corrects library depth but does not estimate an exogenous-reference
    (spike-in) factor or recover a global occupancy change. Supply externally derived factors
    with ``countingParams.scaleFactors`` and, when controls are present,
    ``countingParams.scaleFactorsControl``. Supplying the arrays replaces
    automatic CPM factor estimation. Quantitative ChIP and CUT&RUN calibration
    designs are discussed by `Orlando et al.
    <https://genome.cshlp.org/content/24/7/1157>`_ and `Meers et al.
    <https://elifesciences.org/articles/46314>`_. Endpoint modes reject
    ``EGS`` or ``RPGC`` normalization and nonzero
    ``samParams.oneReadPerBin`` rather than silently changing event geometry.

Peak Calling Controls
~~~~~~~~~~~~~~~~~~~~~

``matchingParams.peakMode``
    Selects ROCCO export shape.

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
    descriptive threshold. ``null`` disables the gate.

``matchingParams.numRegionReplays``
    Sets the number of chromosome-local null candidate replays used for BED
    p/q columns. The default is `64`.

``outputParams.plotNullCalibrationDiagnostics``
    Writes one 2-by-2 PNG beside the ROCCO peak output. The default is
    ``True``. The metadata output field is ``nullCalibrationDiagnostics``, and
    the filename ends in ``.nullCalibration.png``.

    One histogram shows stationary-bootstrap null tail occupancy and the normal
    tail target set by ``matchingParams.thresholdZ``. The second shows
    chromosome signed tail excess and shrunk-budget distributions. The
    chromosome scatter relates local and shrunk budgets. Paired touching bars
    show bootstrap-null mean occupancy and signed track-tail excess, whose
    algebraic sum is track-tail occupancy. The views expose null fit, Monte
    Carlo variability, chromosome shrinkage, and the amount of tail mass
    available to ROCCO.

    Set the value to ``False`` to suppress the figure. The post-hoc CLI opt-out
    is ``--match-no-null-calibration-diagnostics``. The null draws use the
    stationary bootstrap introduced for weakly dependent stationary
    observations by `Politis and Romano (1994)
    <https://doi.org/10.1080/01621459.1994.10476870>`_.

``matchingParams.useLocalBootStrapRadius``
    Controls stationary-bootstrap restart-source locality. The default is
    ``True``. For an output bin in a contiguous coordinate segment of ``m``
    bins, each block origin is sampled from the same segment at most ``r`` bins
    from the output bin, where

    .. math::

       r_{\mathrm{square}} =
       \begin{cases}
       m-1, & b \ge m,\\
       \min\left\{m-1,\left\lceil\sqrt{bm}\right\rceil\right\}, & b < m,
       \end{cases}
       \qquad
       r = \min\left\{r_{\mathrm{square}},
       \left\lfloor\frac{1{,}000{,}000}{w}\right\rfloor\right\}.

    Here ``b`` is the residual span in bins and
    :math:`w=\max\{1,\operatorname{round}(\operatorname{median}(end-start))\}`
    is the chromosome's bin-width approximation in BP. The ``b >= m`` branch
    is checked before ``b * m`` is formed. A median bin wider than 1 Mb sets
    ``r`` to zero. Unequal bin widths can yield genomic separations above 1 Mb
    because the cap uses ``w``, not coordinate arrays.
    The radius limits block-origin selection, not geometric block length.
    Copying advances circularly at the source segment edge, so a coordinate gap
    is never crossed. Set the value to ``False`` for chromosome-wide origin
    sampling. The post-hoc CLI opt-out is
    ``--match-no-local-bootstrap-radius``.

    References: `Politis and Romano (1994)
    <https://doi.org/10.1080/01621459.1994.10476870>`_ and `Paparoditis and
    Politis (2002)
    <https://www.numdam.org/item/CRMATH_2002__335_11_959_0.pdf>`_.

``matchingParams.mergeToleranceBP``
    Sets the maximum gap eligible for broad-family merging. A positive value
    is required for broad and combined modes.

``matchingParams.maxRegionBP``
    Sets the maximum outer width for a broad family. A positive value is
    required for broad and combined modes.


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

``countingParams.centerMBMethod`` and ``countingParams.centerMBWindowBP``
    Set the centered low-frequency trend removed from each transformed count
    track. The defaults are degree-zero ``savgol`` and `1,000,000` bp.

``fitParams.t_innerIters``
    Sets the filter, smoother, and Student-t reweighting updates per ECM
    iteration. The default is `6`.

``fitParams.ECM_robustTNu`` and ``fitParams.ECM_processRobustTNu``
    Set the Student-t degrees of freedom for observation and process precision
    reweighting. The defaults are ``8.0`` and ``3.0``, respectively.

``fitParams.ECM_backgroundLengthScaleMultiplier``
    Sets the multiplier that converts the inferred correlation-length
    into the soft background fitting window. Larger values softly restrict the shared
    background estimate :math:`g_{[i=1,\ldots,i=n]}` to lower frequencies.

``fitParams.ECM_scaleObsPrecisionToMedian``
    If true, divides raw Student-t observation precision multipliers
    :math:`\lambda_i` by the midpoint sample median, then applies
    ``observationParams.precisionMultiplierMin`` and
    ``observationParams.precisionMultiplierMax``. The default is false.

``fitParams.ECM_scaleProcessPrecisionToMedian``
    If true, divides raw Student-t process precision multipliers
    :math:`\kappa_i` by the midpoint sample median, then applies
    ``processParams.precisionMultiplierMin`` and
    ``processParams.precisionMultiplierMax``. Index zero is fixed at one and
    omitted from the median. The default is true, with process precision
    bounds of ``1.0e-4`` and ``10.0``.

``outputParams.stateShrinkageSpikeOddsMultiplier``
    (Experimental) Ignored if posterior state shrinkage is disabled
    entirely (``outputParams.stateShrinkageEnabled``). Values above `1.0`
    multiply fitted point-null odds upward, which can reduce false positives
    in low-signal regions. Values below `1.0` multiply those odds downward.
