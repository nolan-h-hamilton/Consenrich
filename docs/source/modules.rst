API Reference
---------------


.. toctree::
   :maxdepth: 1
   :caption: API
   :name: API


``consenrich.core``
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``core``
    :name: core

The core module implements the main aspects of Consenrich and defines key parameter classes.

.. autoclass:: consenrich.core.processParams
.. autoclass:: consenrich.core.observationParams
.. autoclass:: consenrich.core.stateParams
.. autoclass:: consenrich.core.detrendParams
.. autoclass:: consenrich.core.inputParams
.. autoclass:: consenrich.core.genomeParams
.. autoclass:: consenrich.core.countingParams
.. autoclass:: consenrich.core.samParams
.. autoclass:: consenrich.core.matchingParams
.. autofunction:: consenrich.core.getChromRanges
.. autofunction:: consenrich.core.getChromRangesJoint
.. autofunction:: consenrich.core.getReadLength
.. autofunction:: consenrich.core.readBamSegments
.. autofunction:: consenrich.core.getAverageLocalVarianceTrack
.. autofunction:: consenrich.core.constructMatrixF
.. autofunction:: consenrich.core.constructMatrixQ
.. autofunction:: consenrich.core.constructMatrixH
.. autofunction:: consenrich.core.runConsenrich
.. autofunction:: consenrich.core.getPrimaryState
.. autofunction:: consenrich.core.getStateCovarTrace
.. autofunction:: consenrich.core.getPrecisionWeightedResidual
.. autofunction:: consenrich.core.getMuncTrack
.. autofunction:: consenrich.core.sparseIntersection
.. autofunction:: consenrich.core.adjustFeatureBounds
.. autofunction:: consenrich.core.getSparseMap
.. autofunction:: consenrich.core.getBedMask

``consenrich.detrorm``
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``detrorm``
    :name: detrorm

.. autofunction:: consenrich.detrorm.getScaleFactor1x
.. autofunction:: consenrich.detrorm.getScaleFactorPerMillion
.. autofunction:: consenrich.detrorm.getPairScaleFactors
.. autofunction:: consenrich.detrorm.detrendTrack


``consenrich.constants``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``constants``
    :name: constants

.. important::

    This module is provided for *convenience*. If a genome is not listed here, users can still specify resources (e.g., sizes file, blacklist) manually.

.. autofunction:: consenrich.constants.getEffectiveGenomeSize
.. autofunction:: consenrich.constants.getGenomeResourceFile
.. autofunction:: consenrich.constants.resolveGenomeName

.. _match:

``consenrich.matching``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: `matching`
    :name: matching



(Experimental) Detect genomic regions showing both **enrichment** and **non-random structure**


- Take a set of genomic intervals :math:`i=1,2,\ldots,n`, each spanning :math:`L` base pairs,

- and a 'consensus' signal track defined over the genomic intervals, estimated from multiple independent samples' high-throughput functional genomics sequencing data:

.. math::

  \widetilde{\mathbf{x}} = \{\widetilde{x}_{[i]}\}_{i=1}^{i=n}.

In this documentation, we assume :math:`\widetilde{\mathbf{x}}` is the Consenrich 'primary state estimate'.

**Aim**: Determine a set of 'structured' peak-like genomic regions where the consensus signal track :math:`\widetilde{\mathbf{x}}` exhibits both:

#. *Enrichment* (large relative amplitude)
#. *Non-random structure* (polynomial, oscillatory, etc.)

**Why**: Prioritizing genomic regions that are both enriched and show a prescribed level of structure is appealing for several reasons. Namely,

* **Improved confidence** that the identified genomic regions are not due to stochastic noise, which is characteristically unstructured.
* **Targeted detection** of biologically relevant signal patterns in a given assay (e.g., see related works analyzing peak-shape `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_, `Parodi et al., 2017 <https://doi.org/10.1093/bioinformatics/btx201>`_)
* **Speed**: Runs genome-wide in seconds/minutes using efficient Fast Fourier Transform (FFT)-based calculations.

In the case of Consenrich, that the primary signal estimates in :math:`\{\widetilde{x}_{[i]}\}^{n}` are reinforced by multiple samples and account for relevant sources of uncertainty is advantageous--it provides a more reliable basis for evaluating legitimate structure and identifying high-resolution features.

Algorithm Overview
""""""""""""""""""""""

To detect structured peaks, we run an approach akin to `matched filtering <https://en.wikipedia.org/wiki/Matched_filter>`_, with
*templates* derived from discrete wavelet/scaling functions.

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T}.


Denote the cross-correlation between the consensus signal :math:`\widetilde{\mathbf{x}}` and the matching template :math:`\boldsymbol{\xi}` as:

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \star \boldsymbol{\xi}.

Explicitly,

.. math::

  \mathcal{R}_{[i]} = \sum_{t=1}^{t=T} \widetilde{x}_{[i+t-1]} \cdot \xi_{[t]}.

We refer to :math:`\mathcal{R}` over :math:`i=1 \ldots n` as the *response sequence*.

The response is greatest where :math:`\widetilde{x}` is large and exhibits a structure similar to the template :math:`\boldsymbol{\xi}`.


**Detection Policy**

* To detect significant hits, we first construct an empirical null distribution of blockwise maxima in the response sequence :math:`\mathcal{R}`

  * That is, we randomly sample :math:`B` distinct blocks of genomic intervals and record each :math:`\max(\mathcal{R}_{[b_1]}, \ldots, \mathcal{R}_{[b_K]})`. Default :math:`B = 25000`.
  * The size of each sampled block (:math:`K`) is drawn from a (truncated) geometric distribution with a mean equal to the template length :math:`T` (or a user-specified minimum feature size).

.. note:: Alternating Sampling Scheme

  To avoid overlaps/leakage, note that the empirical null distributions are built from held-out genomic intervals (i.e., those *not* being tested for matches).

  Specifically, we first build empirical null distributions on the first :math:`M < n` genomic intervals in a given chromosome. We then detect peaks on the remaining :math:`n - M` intervals.
  A second empirical null is then built on the previously tested `n - M` intervals, and peaks are detected on the first :math:`M` intervals. This alternating procedure continues until all intervals have been tested.

* Relative maxima in the response sequence, where at interval :math:`i^*` we observe: :math:`\mathcal{R}_{[i^* - 1 : i^* - T/2]} < \mathcal{R}_{[i^*]} > \mathcal{R}_{[i^* + 1 : i^* + T/2]}` are identified as candidate matches.
* Each of the candidates is then assigned a :math:`p`-value based on the empirical null distribution of blockwise maxima. Those satisfying :math:`p_{\textsf{adj}} < \alpha` are deemed significant. Default :math:`\alpha = 0.05`.

  * Additional criteria can be applied: *signal value* :math:`\widetilde{x}_{[i^*]}` to exceed a given value, or requiring detected features to exceed a minimum length.
  * Overlapping/adjacent matches can be merged (By default, peaks within :math:`0.50 \times T` intervals of one another are merged, where :math:`T` is the template length).
  * Multiple templates and cascade levels can be used to capture features of varying shapes and sizes simultaneously.


**Thresholds**

* ``matchingParams.alpha``: Significance cutoff for detection (default ``0.05``).


* ``matchingParams.minSignalAtMaxima`` (Optional)
  Can be an absolute numeric value (`float`) or a `string`, ``"q:<quantileValue>"``, to require :math:`f(\widetilde{x}_{[i^*]})` to exceed the given quantile of :math:`f`-transformed values (default: :math:`f := \textsf{asinh}`, ``q:0.75``).

  - *To disable*: set to a negative numeric value.

* ``matchingParams.minMatchLengthBP``: (Optional)
  Minimum feature length in bp to qualify as a match (default ``250``).

  - *If set to a negative value, the minimum feature length is implicitly defined by one-half of the template length.*


**Generic Defaults**

The following defaults are not encompassing but should provide a strong starting point for many use cases. For broad marks, consider setting ``matchingParams.mergeGapBP`` to a large value.

.. code-block:: yaml

  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [2]
  matchingParams.minMatchLengthBP: 250
  matchingParams.alpha: 0.05
  matchingParams.minSignalAtMaxima: 'q:0.75'
  matchingParams.merge: true

If unspecified, `matchingParams.mergeGapBP` is set to half of `matchingParams.minMatchLengthBP`. If `matchingParams.minMatchLengthBP` is negative, `matchingParams.mergeGapBP` defaults to one-half of the *template* length in base pairs.

---

.. autofunction:: consenrich.matching.matchWavelet

.. autofunction:: consenrich.matching.mergeMatches


Cython functions: ``consenrich.cconsenrich``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``cconsenrich``
    :name: cconsenrich

Several computationally burdensome tasks are written in `Cython <https://cython.org/>`_ for improved efficiency.

.. autofunction:: consenrich.cconsenrich.creadBamSegment
.. autofunction:: consenrich.cconsenrich.cinvertMatrixE
.. autofunction:: consenrich.cconsenrich.updateProcessNoiseCovariance
.. autofunction:: consenrich.cconsenrich.csampleBlockStats
.. autofunction:: consenrich.cconsenrich.cSparseAvg
.. autofunction:: consenrich.cconsenrich.cgetFragmentLength
.. autofunction:: consenrich.cconsenrich.cbedMask
