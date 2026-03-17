API Reference
---------------

.. _API:

``consenrich.core``
~~~~~~~~~~~~~~~~~~~~~~

Primary model and configuration interfaces.

.. note::

  Many parameters do not require tuning in practice but are listed here for completeness.

Notation
""""""""

For interval :math:`i` and replicate :math:`j`:

* :math:`y_{[j,i]}` is the observed track value
* :math:`\mathbf{x}_{[i]} = (x_{[i,0]}, x_{[i,1]})^\top` is the latent level/slope state
* :math:`g_{[i]}` is the shared zero-centered background
* :math:`b_j` and :math:`a_j` are replicate bias and observation-scale terms
* :math:`v_{[j,i]}` is the plugin observation variance track
* :math:`b(i)` maps interval :math:`i` to block :math:`b`
* :math:`r_b` and :math:`q_b` are block observation/process scales
* :math:`\lambda_{[j,i]}` and :math:`\kappa_{[i]}` are Student-t precision weights

Model
"""""

**Observations**

.. math::

  y_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
  \qquad
  \mathrm{Var}(\epsilon_{[j,i]}) =
  \frac{a_j r_{b(i)} (v_{[j,i]} + \mathrm{pad})}{\lambda_{[j,i]}}.

**Prior**

The latent state vector :math:`\mathbf{x}_{[i]}` evolves according to intentionally simple dynamics to mitigate overfitting while capturing local structure.

.. math::

  \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
  \qquad
  \mathrm{Var}(\eta_{[i]}) = \frac{q_{b(i)} \mathbf{Q}_0}{\kappa_{[i]}}.

Here :math:`g_{[i]}` is a shared zero-centered smooth background. The outer loop updates
:math:`g_{[i]}` and, optionally, a shared interval-level plugin variance track.

.. autoclass:: consenrich.core.inputParams
.. autoclass:: consenrich.core.genomeParams
.. autoclass:: consenrich.core.countingParams
.. autoclass:: consenrich.core.scParams
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
* by default, a pooled empirical null is built from sampled block maxima, after trimming the extreme upper tail
* optional blocked folds can add held-out null/test evaluations
* candidate :math:`p`-values come from the pooled null alone by default, or from a pooled-plus-fold combination when split nulls are enabled
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
