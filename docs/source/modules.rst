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


- Take a set of successive genomic intervals :math:`i=1,2,\ldots,n`, each spanning :math:`L` base pairs.

- Define a 'consensus' signal track over the genomic intervals, estimated from multiple independent samples' high-throughput functional genomics sequencing data:

.. math::

  \widetilde{\mathbf{x}} = \{\widetilde{x}_{[i]}\}_{i=1}^{i=n}.

Note, we use the sequence of Consenrich signal estimates to define :math:`\widetilde{\mathbf{x}}`.

**Aim**: Determine a set of 'structured' peak-like genomic regions where the consensus signal track :math:`\widetilde{\mathbf{x}}` exhibits both:

#. *Enrichment* (large relative amplitude)
#. *Non-random structure* (polynomial or oscillatory trends)

**Why**: Prioritizing genomic regions that are both enriched and show a prescribed level of structure is appealing for several reasons. Namely,

* **Improved confidence** that the identified genomic regions are not due to stochastic noise, which is characteristically unstructured.
* **Targeted detection** of biologically relevant signal patterns in a given assay (`Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_, `Parodi et al., 2017 <https://doi.org/10.1093/bioinformatics/btx201>`_)
* **Speed**: runs in seconds on a genome-wide scale.

In the case of Consenrich, that :math:`\widetilde{\mathbf{x}}` is reinforced by multiple samples and accounts for multiple sources of uncertainty is particularly advantageous--it provides a more reliable basis for evaluating legitimate structure and identifying high-resolution features.

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

Where :math:`\mathcal{R}_{[i]}` is large, there is greater evidence that the signal :math:`\widetilde{\mathbf{x}}` is enriched and exhibits structure similar to the template :math:`\boldsymbol{\xi}`.

**Significance**

- ``matchingParams.alpha`` defines the (:math:`1 - \alpha`)-quantile threshold of the block-maxima null on the response sequence.
- ``matchingParams.minSignalAtMaxima`` can be an absolute value (float) or string ``"q:<quantileValue>"`` to require the asinh-transformed signal at response-maxima to exceed the given quantile of non-zero values (default ``q:0.75``).

.. seealso::

  Sections :ref:`minimal` and/or :ref:`additional-examples` which include browser shots demonstrating qualitative behavior of this feature.

In the following browser snapshot, we sweep several key matching parameters.

As opposed to the configs in :ref:`additional-examples`, here, we set ``matchingParams.merge: false`` to clearly illustrate contrasting results.

.. image:: ../images/structuredPeaks.png
  :alt: Structured Peaks
  :width: 85%
  :align: center
  :name: structuredPeaks

- ``matchingParams.templateNames``

  - Narrow, condensed features :math:`\rightarrow` shorter wavelet-based templates (e.g., ``haar``, ``db2``).
  - Broader features :math:`\rightarrow` longer, symmetric wavelet-based templates (e.g., ``sym4``).
  - Oscillatory features :math:`\rightarrow` longer, higher-order wavelets (e.g., ``db8``, ``dmey``).

- ``matchingParams.alpha`` (Significance Threshold)

  - Signifcance is measured relative to an approximated null distribution of response values.
  - Tunes precision vs. recall -- the stringency of match detection.
  - Smaller values :math:`\rightarrow` fewer but higher-confidence matches; larger values :math:`\rightarrow` more lower-confidence matches.

- ``matchingParams.minMatchLengthBP`` (Feature Width Threshold)

  - Enforces a minimum feature width (base pairs)
  - Increase this parameter to reduce matches with features that are narrower than the underlying pattern of interest.

- ``matchingParams.minSignalAtMaxima`` (Signal Threshold)

  - Enforces a minimum Consenrich *signal estimate* over the detected maxima.
  - If `None`, defaults to the median of nonzero signal values. This threshold is applied after tempering the dynamic range with an arsinh transform (i.e., :math:`\sinh^{-1}(x)`).


**Generic Defaults**

These defaults are not encompassing but provide a reasonable starting point for common use cases.

.. code-block:: yaml

  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [2]
  matchingParams.minMatchLengthBP: 250
  matchingParams.mergeGapBP: 50
  matchingParams.alpha: 0.05


The matching algorithm can be run directly at the command-line on existing bedGraph files generated by Consenrich. For instance, in the ChIP-seq experiment from :ref:`getting-started`,

  .. code-block:: console

    % consenrich \
      --match-bedGraph consenrichOutput_demoHistoneChIPSeq_state.bedGraph \
      --match-template db3 \
      --match-level 2 \
      --match-alpha 0.01

This is convenient as it avoids a full run of Consenrich and requires minimal runtime: See `consenrich -h` for more details.

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


