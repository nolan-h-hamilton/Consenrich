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
.. autoclass:: consenrich.core.outputParams
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


Denote a 'consensus' signal track defined over fixed-length genomic intervals, estimated from multiple samples' functional genomics HTS data as

.. math::

  \widetilde{\mathbf{x}} = \{\widetilde{x}_{[i]}\}_{i=1}^{i=n}.

In this documentation, we assume :math:`\widetilde{\mathbf{x}}` is the Consenrich 'primary state estimate' track. (To match on an uncertainty-penalized version of the primary signal, invoke `matchingParams.penalizeBy`)

**Aim**: Determine a set of 'structured' peak-like genomic regions where the consensus signal track :math:`\widetilde{\mathbf{x}}` exhibits both:

#. *Enrichment* (large relative amplitude)
#. *Non-random structure* defined by a robust template (polynomial, oscillatory, etc.)

**Why**: Prioritizing genomic regions that are both enriched and show a prescribed level of structure is appealing for several reasons. Namely,

* **Targeted detection** of biologically relevant signal patterns in a given assay (e.g., see related works analyzing peak-shape `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_, `Parodi et al., 2017 <https://doi.org/10.1093/bioinformatics/btx201>`_)
* **Improved confidence** that the identified genomic regions are not due to stochastic noise, which is characteristically unstructured.
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

This response is greatest at genomic intervals where :math:`\widetilde{x}_{[i]}` is large and exhibits a structure/shape similar to the template :math:`\boldsymbol{\xi}`.

* To detect significant hits, we first construct an empirical null distribution of blockwise maxima in the response sequence, :math:`\mathcal{R}`

  * That is, we randomly sample :math:`B` distinct blocks of genomic intervals and record each :math:`\max(\mathcal{R}_{[b_1]}, \ldots, \mathcal{R}_{[b_K]})`. Default :math:`B = 25000`.
  * The size of each sampled block (:math:`K`) is drawn from a (truncated) geometric distribution with a mean equal to the desired feature size or template length, :math:`T`.

* Relative maxima in the response sequence, i.e., :math:`i^*` such that :math:`\mathcal{R}_{[i^* - 1 \,:\, i^* - T/2]}\, \leq \, \mathcal{R}_{[i^*]} \, \geq \, \mathcal{R}_{[i^* + 1 \,:\, i^* + T/2]}` are identified as candidate matches.

* Each of the candidates is then assigned a :math:`p`-value based on the empirical null distribution of blockwise maxima. Those satisfying :math:`p_{\textsf{adj}} < \alpha` are deemed significant. Default :math:`\alpha = 0.05`.

  * Additional criteria can be applied: e.g., require the *signal values* at candidate peaks, :math:`\widetilde{x}_{[i^*]}`, to exceed a cutoff (`matchingParams.minSignalAtMaxima`), and/or require the *length* of the matched feature to exceed a minimum size (`matchingParams.minMatchLengthBP`).
  * Overlapping/adjacent matches can be merged.

.. note:: **Alternating Sampling Scheme**

  To avoid overlaps/leakage when testing significance, note that the mentioned empirical null distributions are built from held-out genomic intervals (i.e., those *not* being tested for matches).

  Specifically, we first build empirical null distributions on the first :math:`M < n` genomic intervals in a given chromosome. We then detect peaks on the remaining :math:`n - M` intervals.
  A second empirical null is then built on the previously tested `n - M` intervals, and peaks are detected on the first :math:`M` intervals. This alternating procedure continues until all intervals have been tested.


**Thresholds**

* ``matchingParams.alpha``: Significance cutoff (default ``0.05``). Peaks with adjusted empirical :math:`p`-values below this threshold are considered significant.


* ``matchingParams.minSignalAtMaxima`` (Optional)
  Can be an absolute numeric value (`float`) or a `string`, ``"q:<quantileValue>"``, to require :math:`f(\widetilde{x}_{[i^*]})` to exceed the given quantile of :math:`f`-transformed values (default: :math:`f := \textsf{asinh}`, ``q:0.75``).

  - *To disable*: set to a negative numeric value.

* ``matchingParams.minMatchLengthBP``: (Optional)
  Minimum feature length in base pairs (default ``250``).


**Generic Defaults**

The following defaults should provide a strong starting point for many use cases. For broad marks, consider setting ``matchingParams.mergeGapBP`` and/or ``countingParams.stepSize`` to larger values to prioritize larger-scale trends.

.. code-block:: yaml

  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [2, 2]
  matchingParams.minMatchLengthBP: -1 # set as `-1` for 'auto', data-driven selection
  matchingParams.alpha: 0.05
  matchingParams.minSignalAtMaxima: 'q:0.75'
  matchingParams.merge: true

If unspecified, `matchingParams.mergeGapBP` is set to half of `matchingParams.minMatchLengthBP`.

**Note**, the matching algorithm can be run at the command-line on *existing* bedGraph files from previous Consenrich runs.
This avoids re-running Consenrich end-to-end when only matching/peak-calling is desired. For instance,

.. code-block:: console

  % consenrich \
    --match-bedGraph consenrichOutput_<experimentName>_state.bedGraph \
    --match-template haar \
    --match-level 3 \
    --match-alpha 0.01

This will return structured peaks detected using a Haar template/level 3 and significance threshold :math:`\alpha=0.01`. Run ``consenrich -h`` for additional options.

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
