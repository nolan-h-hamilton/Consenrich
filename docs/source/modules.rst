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

The core module implements the main aspects of Consenrich.

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

``consenrich.detrorm``
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``detrorm``
    :name: detrorm

.. note::

    See :class:`consenrich.core.detrendParams` for relevant parameters.

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
    :caption: ``matching``
    :name: matching



(*Experimental*). Detect genomic regions showing both **enrichment** and **non-random structure**.


For a sequence of genomic intervals (fixed in size :math:`L\text{bp}`),

.. math::

  i \mapsto \{(i-1)\cdot L + 1, \ldots, i\cdot L\}.


for :math:`i=1,2,\ldots`, denote an estimated 'consensus' signal track derived from multi-sample HTS data:

.. math::

  \{\widetilde{x}_{[i]}\}_{i=1}^{i=n}

For instance, this could be the sequence of Consenrich signal estimates for a given dataset.


**Our aim is to determine a set of peak-like genomic regions over which** :math:`\widetilde{x}_{[:]}` **exhibits**:

#. *Enrichment* (large relative amplitude)
#. *Non-random structure* (polynomial or oscillatory trends)

Verifying genomic regions satisfy this dual criteria ('structured enrichment') provides several appealing features compared to traditional enrichment-based peak calling:

* Improved confidence that the identified genomic regions are not due to stochastic noise--which is characteristically unstructured.
* Targeted detection of biologically relevant signal patterns in a given assay, e.g., `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_


To detect regions of structured enrichment, we run an approach akin to `matched filtering <https://en.wikipedia.org/wiki/Matched_filter>`_ with
*templates* derived from discrete samplings of wavelet functions:

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T},


We define the *response sequence* as the cross-correlation

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \star \boldsymbol{\xi} \in \mathbb{R}^{n}

At genomic interval :math:`i \in \{1, \ldots, n\}`, a *match* is declared if the following are true:

- :math:`\mathcal{R}_{[i]}` is a relative maximum within a window defined by the template length :math:`T`.
- :math:`\mathcal{R}_{[i]}` exceeds a significance cutoff determined by the :math:`1 - \alpha` quantile of an approximated null distribution (See :func:`cconsenrich.csampleBlockStats`).
- *Optional*: The *signal* value at the response-maximum is above ``minSignalAtMaxima``.

.. seealso::

  Sections :ref:`minimal` and/or :ref:`additional-examples` which include browser shots demonstrating qualitative behavior of this feature.

  .. tip::

    Consider beginning with a Daubechies wavelet-based template with two vanishing moments `matchingParams.templateNames: [db2]` and `matchingParams.cascadeLevels: [2]`. For many settings, will provide a good balance between spatial/frequency resolution.


.. autofunction:: consenrich.matching.matchWavelet

.. .. admonition:: note
  :: Consensus Peak Calling

    Traditional enrichment-based peak calling on Consenrich signal track output can be performed using, e.g., `ROCCO <https://github.com/nolan-h-hamilton/ROCCO>`_,

     ``rocco -i <ConsenrichOutput.bw> -g <genomeName> [...]``

    Other peak callers accepting bedGraph or bigWig input may be plausible downstream companions to Consenrich, too (e.g., `MACS bdgpeakcall <https://macs3-project.github.io/MACS/docs/bdgpeakcall.html>`_)


Cython functions: ``consenrich.cconsenrich``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``cconsenrich``
    :name: cconsenrich

Several components are implemented with strong typing/manual memory management in Cython for efficiency.

.. autofunction:: consenrich.cconsenrich.creadBamSegment
.. autofunction:: consenrich.cconsenrich.cinvertMatrixE
.. autofunction:: consenrich.cconsenrich.updateProcessNoiseCovariance
.. autofunction:: consenrich.cconsenrich.csampleBlockStats
.. autofunction:: consenrich.cconsenrich.cSparseAvg



