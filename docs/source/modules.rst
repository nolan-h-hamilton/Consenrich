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

.. autofunction:: consenrich.constants.getEffectiveGenomeSize
.. autofunction:: consenrich.constants.getGenomeResourceFile
.. autofunction:: consenrich.constants.resolveGenomeName



Cython functions: ``consenrich.cconsenrich``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``cconsenrich``
    :name: cconsenrich

Several computationally burdensome tasks are written in cython for efficiency.

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


.. _match:

``consenrich.matching``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: `matching`
    :name: matching



(Experimental) Detect genomic regions showing both **enrichment** and **non-random structure**


Denote a noisy signal over fixed-length genomic intervals, estimated from multiple samples' functional genomics HTS data as

.. math::

  \widetilde{\mathbf{x}} = \{\widetilde{x}_{[i]}\}_{i=1}^{i=n}.

**Aim**: Determine a set of 'structured' peak-like signal regions showing both:

#. *Enrichment* (large relative amplitude)
#. *Non-random structure* defined by a robust template (polynomial, oscillatory, etc.)

Prioritizing genomic regions that are both enriched and agree with a prescribed structure may be appealing for several reasons. Namely,

* **Targeted detection** of biologically relevant signal patterns in a given assay (e.g., see related works analyzing peak-shape `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_, `Parodi et al., 2017 <https://doi.org/10.1093/bioinformatics/btx201>`_)
* **Improved confidence** that the identified genomic regions are not due to stochastic noise, which is characteristically unstructured.
* **Speed**: Runs in seconds/minutes using efficient numerical methods to compute large chromosome-scale convolutions (fast fourier transform (FFT)-based, overlap-add (OA), etc.)

Algorithm Overview
""""""""""""""""""""""

To detect structured peaks, we run an approach akin to `matched filtering <https://en.wikipedia.org/wiki/Matched_filter>`_, with
*templates* derived from approximated discrete `wavelets <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_ or their scaling functions.

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T}.


Denote the cross-correlation between the signal track and a matching template :math:`\boldsymbol{\xi}` as:

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \star \boldsymbol{\xi},

.. math::

  \mathcal{R}_{[i]} = \sum_{t=1}^{t=T} \widetilde{x}_{[i+t-1]} \cdot \xi_{[t]}.

We refer to :math:`\mathcal{R}` over :math:`i=1 \ldots n` as the *response sequence*. The response will be greatest in genomic regions with a high read density and structural similarity with template :math:`\boldsymbol{\xi}`.

To detect significant hits,

* We first construct an observed empirical distribution from randomly-sampled genomic blocks. Specifically, we sample :math:`B` blocks and record each :math:`\max(\mathcal{R}_{[b_1]}, \ldots, \mathcal{R}_{[b_K]})`. Note, to mitigate artifacts, the size of each sampled block (:math:`K`) is drawn from a (truncated) geometric distribution with a mean equal to the desired feature size or template length, :math:`T`.

* Relative maxima in the response sequence, i.e., :math:`i^*` such that :math:`\mathcal{R}_{[i^* - 1 \,:\, i^* - T/2]}\, \leq \, \mathcal{R}_{[i^*]} \, \geq \, \mathcal{R}_{[i^* + 1 \,:\, i^* + T/2]}` are retained as candidate matches

* Each candidate is assigned an empirical :math:`p`-value based on its (interpolated) quantile in the sampled distribution. Those satisfying :math:`p_{\textsf{adj}} < \alpha` are deemed 'significant'.

  * Additional criteria for matching: require the *signal values* at candidate peaks/matches, :math:`\widetilde{x}_{[i^*]}`, to exceed a cutoff (`matchingParams.minSignalAtMaxima`), and/or require the *length* of the matched feature to exceed a minimum size (`matchingParams.minMatchLengthBP`).
  * Overlapping/adjacent matches can be merged.

.. note:: **Alternating Sampling Scheme**

  * Blocks are sampled from the first :math:`M < n` genomic intervals in each contig. The maximum value in each block is recorded to build an empirical distribution. Matches are detected on the remaining :math:`n - M` intervals using this empirical distribution.

  * A second empirical distribution is then built on intervals `n - M \ldots n`. Matches over intervals :math:`1 \ldots M` are then called using the empirical distribution over intervals :math:`n - M \ldots n`.

  The size of each block is random: Each is drawn from a truncated geometric distribution with :math:`p=\frac{1}{T}`.

**Generic Defaults**

.. code-block:: yaml

  matchingParams.templateNames: [haar, haar, db2, db2]
  matchingParams.cascadeLevels: [1,2,1,2]
  matchingParams.minMatchLengthBP: -1 # select via `consenrich.core.getContextSize`
  matchingParams.mergeGapBP: -1
  matchingParams.alpha: 0.05


---

.. autofunction:: consenrich.matching.matchWavelet



