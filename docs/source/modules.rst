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
*templates* derived from discrete wavelet or scaling functions.

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T}.


Define the **response sequence** as the cross-correlation between the consensus signal :math:`\widetilde{\mathbf{x}}` and the template :math:`\boldsymbol{\xi}`:

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \star \boldsymbol{\xi}

(Equivalently, the convolution of :math:`\widetilde{\mathbf{x}}` with the time-reversed template :math:`\boldsymbol{\xi}'`, where :math:`\xi'_{[t]} = \xi_{[T-t+1]}`.)

Where :math:`\mathcal{R}_{[i]}` is large, there is greater evidence that the signal :math:`\widetilde{\mathbf{x}}` is enriched and exhibits structure that is agreeable to the template :math:`\boldsymbol{\xi}`.

* To assess statistical significance, we compare the response sequence to a block-maxima null distribution that captures the behavior of relative maxima in :math:`\mathcal{R}` under an assumption that :math:`\widetilde{\mathbf{x}}` is generated by a random process with no structured features.

  * The null distribution is constructed by sampling blocks in :math:`\widetilde{\mathbf{x}}` and recording the maximum value within each block.
  * The size of each sampled block is drawn from a (truncated) geometric distribution with an expected value equal to the template length (or a user-specified minimum feature size).

* Response maxima exceeding a specified quantile threshold of the null distribution qualify as 'structured peaks'.

  * Additional criteria can be applied, such as requiring the value at response-maxima to exceed a given quantile of non-zero values in :math:`\widetilde{\mathbf{x}}`, and a minimum feature length.
  * Overlapping/adjacent matches can be merged (By default, peaks within :math:`0.50 \times T` intervals of one another are merged, where :math:`T` is the template length).
  * Multiple templates and cascade levels can be used to capture features of varying shapes and sizes.


**Detection Thresholds**

* ``matchingParams.alpha``
  defines the (:math:`1 - \alpha`)-quantile threshold of the block-maxima null on the response sequence, i.e., the cross-correlation between the Consenrich track and template (default ``0.05``).
  - Smaller values of ``alpha`` lead to more stringent detection thresholds and fewer detected matches.
  - By default, ``alpha = 0.05``.


* ``matchingParams.minSignalAtMaxima`` (Optional)
  Can be an absolute value (float) or string ``"q:<quantileValue>"`` to require the value at response-maxima to exceed the given quantile of non-zero values (default ``q:0.75``).

  - This threshold is applied after tempering the dynamic range (i.e., :math:`\sinh^{-1}(x)`).
  - By default, the 75th percentile of non-zero values is used: `q:0.75`
  - *To disable*: set to a negative numeric value.

* ``matchingParams.minMatchLengthBP``: (Optional)
  Minimum feature length in bp to qualify as a match (default ``250``).

  - *If set to a negative value, the minimum feature length is implicitly defined by the template length.*


**Generic Defaults**

Assuming ``matchingParams.templateNames`` is specified, the following defaults are used, unless overridden by the user:

.. code-block:: yaml

  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [2]
  matchingParams.minMatchLengthBP: 250
  matchingParams.alpha: 0.05
  matchingParams.minSignalAtMaxima: 'q:0.75'
  matchingParams.merge: true

These defaults are not encompassing but should provide a strong starting point for many use cases. See :ref:`additional-examples` for practical examples.

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
