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
.. autoclass:: consenrich.core.plotParams
.. autoclass:: consenrich.core.observationParams
.. autoclass:: consenrich.core.stateParams
.. autoclass:: consenrich.core.inputParams
.. autoclass:: consenrich.core.outputParams
.. autoclass:: consenrich.core.genomeParams
.. autoclass:: consenrich.core.countingParams
.. autoclass:: consenrich.core.samParams
.. autoclass:: consenrich.core.matchingParams
.. autofunction:: consenrich.core.readBamSegments
.. autofunction:: consenrich.core.getMuncTrack
.. autofunction:: consenrich.core.fitVarianceFunction
.. autofunction:: consenrich.core.EB_computePriorStrength
.. autofunction:: consenrich.core.runConsenrich
.. autofunction:: consenrich.core.constructMatrixF
.. autofunction:: consenrich.core.constructMatrixQ
.. autofunction:: consenrich.core.constructMatrixH

``consenrich.detrorm``
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``detrorm``
    :name: detrorm

.. autofunction:: consenrich.detrorm.getScaleFactor1x
.. autofunction:: consenrich.detrorm.getScaleFactorPerMillion
.. autofunction:: consenrich.detrorm.getPairScaleFactors


``consenrich.constants``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``constants``
    :name: constants

.. important::

    This module is provided for *convenience*. Users may directly specify effective genome sizes, resource file paths, etc.

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

**Why**: Prioritizing genomic regions that are both enriched and agree with a prescribed structure is appealing for several reasons. Namely,

* **Targeted detection** of biologically relevant signal patterns in a given assay (e.g., see related works analyzing peak-shape `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_, `Parodi et al., 2017 <https://doi.org/10.1093/bioinformatics/btx201>`_)
* **Improved confidence** that the identified genomic regions are not due to stochastic noise, which is characteristically unstructured.
* **Speed**: Runs in seconds/minutes using efficient numerical methods to compute large chromosome-scale convolutions (fast fourier transform (FFT)-based, overlap-add (OA), etc.)

In the case of Consenrich, that the primary signal estimates in :math:`\{\widetilde{x}_{[i]}\}^{n}` are reinforced by multiple samples and account for relevant sources of uncertainty is advantageous--it provides a more reliable basis for evaluating legitimate structure and identifying high-resolution features.

Algorithm Overview
""""""""""""""""""""""

To detect structured peaks, we run an approach akin to `matched filtering <https://en.wikipedia.org/wiki/Matched_filter>`_, with
*templates* derived from approximated discrete `wavelets <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_ or their scaling functions.

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T}.


Denote the cross-correlation between the consensus signal :math:`\widetilde{\mathbf{x}}` and the matching template :math:`\boldsymbol{\xi}` as:

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \star \boldsymbol{\xi},

.. math::

  \mathcal{R}_{[i]} = \sum_{t=1}^{t=T} \widetilde{x}_{[i+t-1]} \cdot \xi_{[t]}.

We refer to :math:`\mathcal{R}` over :math:`i=1 \ldots n` as the *response sequence*. The response is large in genomic regions where :math:`\widetilde{x}_{[i]}` is high in amplitude and correlated  with the (unit-normed) template :math:`\boldsymbol{\xi}`.

To detect significant hits,

* We first construct an empirical distribution of randomly-sampled blockwise maxima values in the response sequence, :math:`\mathcal{R}` That is, we sample :math:`B` distinct blocks of genomic intervals and record each :math:`\max(\mathcal{R}_{[b_1]}, \ldots, \mathcal{R}_{[b_K]})`. Default :math:`B = 25000`. Note, the size of each sampled block (:math:`K`) is drawn from a (truncated) geometric distribution with a mean equal to the desired feature size or template length, :math:`T`.

* Relative maxima in the response sequence, i.e., :math:`i^*` such that :math:`\mathcal{R}_{[i^* - 1 \,:\, i^* - T/2]}\, \leq \, \mathcal{R}_{[i^*]} \, \geq \, \mathcal{R}_{[i^* + 1 \,:\, i^* + T/2]}` are identified as candidate matches.

* Each of the candidates is then assigned an empirical :math:`p`-value based on its (interpolated) quantile in the empirical null distribution. Those satisfying :math:`p_{\textsf{adj}} < \alpha` are deemed significant. Default :math:`\alpha = 0.05`.

  * Additional criteria can be applied: e.g., require the *signal values* at candidate peaks, :math:`\widetilde{x}_{[i^*]}`, to exceed a cutoff (`matchingParams.minSignalAtMaxima`), and/or require the *length* of the matched feature to exceed a minimum size (`matchingParams.minMatchLengthBP`).
  * Overlapping/adjacent matches can be merged.

.. note:: **Alternating Sampling Scheme**

  The empirical distributions are built from held-out genomic intervals (i.e., those *not* being tested for matches).

  Specifically, we first sample blocks from the first :math:`M < n` genomic intervals in a given chromosome. We then detect peaks on the remaining :math:`n - M` intervals.
  A second empirical null is then built on the previously tested `n - M` intervals, and peaks are detected on the first :math:`M` intervals.

  Note that the block-sampling routine is intended to break long-range dependencies---the sampled blocks are drawn as geometric with mean :math:`T`.

**Thresholds**

* ``matchingParams.alpha``: Significance cutoff (default ``0.05``). Peaks with adjusted empirical :math:`p`-values below this threshold are considered significant within chromosomes.


* ``matchingParams.minSignalAtMaxima`` (Optional)
  Can be an absolute numeric value (`float`) or a `string`, ``"q:<quantileValue>"``, to require :math:`f(\widetilde{x}_{[i^*]})` to exceed the given quantile of :math:`f`-transformed values (default: :math:`f := \textsf{asinh}`, ``q:0.75``).

  - *To disable*: set to a negative numeric value.

* ``matchingParams.minMatchLengthBP``: (Optional)
  Minimum feature length in base pairs (Default ``250``. Use `-1` for 'auto', data-driven selection).


**Generic Defaults**

The following defaults should provide a strong starting point for many use cases. For broader marks, consider using higher-order wavelet-based templates or larger cascade levels, increasing ``matchingParams.mergeGapBP``, ``countingParams.intervalSizeBP``, etc. to prioritize larger-scale trends.

.. code-block:: yaml

  matchingParams.templateNames: [haar, haar, db2, db2]
  matchingParams.cascadeLevels: [2,3,2,3]
  matchingParams.minMatchLengthBP: -1 # auto-select based on data
  matchingParams.mergeGapBP: -1 # selects half of `minMatchLengthBP`
  matchingParams.alpha: 0.05


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
.. autofunction:: consenrich.cconsenrich.updateProcessNoiseCovariance
.. autofunction:: consenrich.cconsenrich.csampleBlockStats
.. autofunction:: consenrich.cconsenrich.cgetFragmentLength
.. autofunction:: consenrich.cconsenrich.cTransform
.. autofunction:: consenrich.cconsenrich.cDenseGlobalBaseline
.. autofunction:: consenrich.cconsenrich.cPAVA
.. autofunction:: consenrich.cconsenrich.cforwardPass
.. autofunction:: consenrich.cconsenrich.cbackwardPass
.. autofunction:: consenrich.cconsenrich.cEMA
.. autofunction:: consenrich.cconsenrich.cmeanVarPairs
.. autofunction:: consenrich.cconsenrich.projectToBox

