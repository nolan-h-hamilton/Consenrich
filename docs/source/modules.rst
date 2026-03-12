API Reference
---------------

.. _API:

``consenrich.core``
~~~~~~~~~~~~~~~~~~~~~~

Primary model and configuration interfaces.

.. note::

  Most parameters do not need tuning. If ``processParams.deltaF < 0``, Consenrich centers a narrow search around ``intervalSizeBP / medianFragmentLength``. If ``stateParams.conformalRescale=True``, the reported uncertainty track is a calibrated future-replicate predictive standard deviation.

.. autoclass:: consenrich.core.inputParams
.. autoclass:: consenrich.core.genomeParams
.. autoclass:: consenrich.core.countingParams
.. autoclass:: consenrich.core.processParams
.. autoclass:: consenrich.core.observationParams
.. autoclass:: consenrich.core.stateParams
.. autoclass:: consenrich.core.outputParams
.. autoclass:: consenrich.core.matchingParams

Primary functions
""""""""""""""""""

.. autofunction:: consenrich.core.runConsenrich
.. autofunction:: consenrich.core.readSegments
.. autofunction:: consenrich.core.getMuncTrack
.. autofunction:: consenrich.core.getContextSize

.. _matching:

``consenrich.matching``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Experimental) Detect genomic regions showing both **enrichment** and **non-random structure**

Algorithm Overview
"""""""""""""""""""""""

To detect structured peaks, we run an approach akin to `matched filtering <https://en.wikipedia.org/wiki/Matched_filter>`_, with
*templates* derived from approximated discrete `wavelets <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_ (or their corresponding scaling functions).

To call matches:

* relative maxima of the response are retained as candidates
* a pooled empirical null is built from sampled block maxima, after trimming the extreme upper tail
* deterministic blocked folds provide held-out null/test evaluations when available
* candidate :math:`p`-values are computed from the pooled null and any fold-specific nulls, then combined with a Cauchy rule
* signal and length filters are applied, and adjacent matches can be merged

**Generic Defaults**

.. code-block:: yaml

  matchingParams.templateNames: [haar, haar, db2, db2]
  matchingParams.cascadeLevels: [1,2,1,2]
  matchingParams.minMatchLengthBP: -1 # select via `consenrich.core.getContextSize`
  matchingParams.mergeGapBP: -1
  matchingParams.alpha: 0.05


---

.. autofunction:: consenrich.matching.matchWavelet
