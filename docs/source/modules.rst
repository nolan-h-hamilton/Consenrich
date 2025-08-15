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

    This module is provided for *convenience*. If a genome is not listed here, users can still specify resources manually.

.. autofunction:: consenrich.constants.getEffectiveGenomeSize
.. autofunction:: consenrich.constants.getGenomeResourceFile
.. autofunction:: consenrich.constants.resolveGenomeName

.. _match:

``consenrich.matching``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: ``matching``
    :name: matching

(*Experimental*). Detect genomic regions showing both **enrichment** and **non-random structure** in multiple samples.

We detect *structured enrichment* using the cross-covariance between the Consenrich signal and downsampled, coarse representations of discrete wavelet functions (Cascade algorithm iterations).

Local maxima in the cross-covariance ('response') are identified and then tested for significance using an empirical null distribution.

Verifying enrichment *and* a prescribed level of structure offers two interesting benefits:

#. Targeted detection of biologically relevant features which may exhibit distinct spatial patterns in a given assay, e.g., `Cremona et al., 2015 <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0787-6>`_
#. Improved confidence that the matched, peak-like regions are *not* due to stochastic noise which is characteristically *unstructured*.

- Denote the sequence of Consenrich signal estimates,

.. math::

   \widetilde{\mathbf{x}} = \{\widetilde{x}_{[i]}\}_{i=1}^{i=n},

- and denote a *template* for matching as,

.. math::

  \boldsymbol{\xi} = \{\xi_{[t]}\}_{t=1}^{t=T},

which we construct using downsampled discrete wavelet functions (`Cascade algorithm <https://en.wikipedia.org/wiki/Cascade_algorithm>`_). These provide a flexible, multi-resolution representation allowing for effective matching at different scales. Note that the template is unit-normalized in the default implementation.

- Define the *response sequence* as the convolution of the signal estimates with the (reversed) template:

.. math::

  \{\mathcal{R}_{[i]}\}_{i=1}^{i=n} = \widetilde{\mathbf{x}} \ast \boldsymbol{\xi}^{\textsf{rev}}

At genomic interval :math:`i \in \{1, \ldots, n\}`, a 'match' is declared if the following hold:

- :math:`\mathcal{R}_{[i]}` is a relative maximum within a window defined by the template length :math:`T`.
- :math:`\mathcal{R}_{[i]}` exceeds a significance cutoff determined by the :math:`1 - \alpha` quantile of an approximated null distribution (See :func:`cconsenrich.csampleBlockStats`).
- *Optional*: The *signal* value at the response-maximum is above ``minSignalAtMaxima``.

This structured enrichment detection can be introduced in the Minimal Usage example by adding the following to the YAML config file:

.. code-block:: yaml
  :name: demoMatchingParameters

  matchingParams.templateNames: [db2]
  matchingParams.cascadeLevels: [2]
  matchingParams.iters: 25_000
  matchingParams.alpha: 0.01

.. autofunction:: consenrich.matching.matchWavelet

  See :ref:`additional-examples` for example use in ATAC-seq.


.. note:: Consensus Peak Calling

    Traditional (consensus) peak calling on Consenrich signal track output can be performed using, e.g., `ROCCO <https://github.com/nolan-h-hamilton/ROCCO>`_,

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



