API Reference
---------------

.. _API:

``consenrich.core``
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  Many parameters do not require tuning in practice but are listed here for completeness.

Notation
""""""""

For interval :math:`i` and replicate :math:`j`:

* :math:`y_{[j,i]}` is the observed track value
* :math:`\mathbf{x}_{[i]} = (x_{[i,0]}, x_{[i,1]})^\top` is the latent level/slope state
* :math:`g_{[i]}` is the shared zero-centered background
* :math:`b_j` and :math:`a_j` are replicate bias and observation-scale terms
* :math:`v_{[j,i]}` is the plugin observation variance track derived from the given data
* :math:`b(i)` maps interval :math:`i` to a block index
* :math:`q_b` is the corresponding process-noise scale
* :math:`\lambda_{[j,i]}` and :math:`\kappa_{[i]}` are precision weights

Model
"""""

**Observations**

.. math::

  y_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
  \qquad
  \mathrm{Var}(\epsilon_{[j,i]}) =
  \frac{a_j (v_{[j,i]} + \mathrm{pad})}{\lambda_{[j,i]}}.

**Prior**

The latent state vector :math:`\mathbf{x}_{[i]}` evolves according to a first-order process with fat-tailed innovations:

.. math::

  \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
  \qquad
  \mathrm{Var}(\eta_{[i]}) = \frac{q_{b(i)} \mathbf{Q}_0}{\kappa_{[i]}}.

Here :math:`g_{[i]}` is a shared zero-centered smooth background. The outer loop updates
:math:`g_{[i]}` while the plugin observation-variance track stays fixed within each inner solve.

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

``consenrich.peaks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: consenrich.peaks.getROCCOBudget
.. autofunction:: consenrich.peaks.solveRocco
