API Reference
---------------

.. _API:

``consenrich.core``
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  Most parameters do not require tuning in practice but are listed in this documentation for completeness.

Notation
""""""""

For interval :math:`i` and replicate :math:`j`:

* :math:`z_{[j,i]}` is the observed track value.
* :math:`\mathbf{x}_{[i]} = (x_{[i,0]}, x_{[i,1]})^\top` is the latent
  level/slope state.
* :math:`g_{[i]}` is a low-frequency background shared across replicates (penalized curvature)
* :math:`b_j` is the per-replicate bias term.
* :math:`v_{[j,i]}` is the plugin observation-variance track derived from the
  given data.
* :math:`\mathbf{Q}_0` is the fixed or warm-up-calibrated base process
  covariance. ``ECM_useAPN`` can be used as a substitute, but technically voids
  guarantees of monotonic descent.
* :math:`\lambda_{[i]}` and :math:`\kappa_{[i]}` are precision weights for the
  observation and process models, respectively. The observation weight rescales
  the full replicate covariance at interval ``i``. These are optimized in every
  outer iteration if precision reweighting is enabled.

Model
"""""

**Observation Model**

.. math::

  z_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
  \qquad
  \mathrm{Var}(\epsilon_{[j,i]}) =
  \frac{v_{[j,i]} + \mathrm{pad}}{\lambda_{[i]}}.

**Prior Process Model**

The latent state vector :math:`\mathbf{x}_{[i]}` evolves according to a
first-order process with fat-tailed innovations:

.. math::

  \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
  \qquad
  \mathrm{Var}(\eta_{[i]}) = \frac{\mathbf{Q}_0}{\kappa_{[i]}}.

Here :math:`g_{[i]}` is an optional shared per-interval background restricted
to low frequencies. Background refinement updates :math:`g_{[i]}` while the
data-derived or given observation-variance track stays fixed within each
fixed-background ECM solve.

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
.. autofunction:: consenrich.core.chooseDependenceLength
.. autofunction:: consenrich.core.chooseFeatureLength

``consenrich.peaks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: consenrich.peaks.getROCCOBudget
.. autofunction:: consenrich.peaks.solveRocco


Normalization by Input Type
""""""""""""""""""""""""""""""""""

``countingParams.normMethod`` is interpreted according to the input type:

* ``BAM`` inputs are raw alignments. Consenrich estimates read or fragment lengths,
  counts signal over genomic intervals, and applies the requested library-size
  normalization method such as ``EGS``, ``RPGC``, ``RPKM``, ``CPM``, or ``SF`` before
  transformation.

* ``FRAGMENTS`` inputs are 10x fragments files. Consenrich counts emitted
  insertions or fragment endpoints according to the fragments settings and uses
  ``CPM``/``RPKM`` scaling. ``EGS`` and ``RPGC`` are not used for fragments;
  ``countingParams.fragmentsGroupNorm: CELLS`` can additionally divide by the
  number of selected cells. Fragments file support is currently experimental.

* ``BEDGRAPH`` inputs are treated as **pre-normalized** continuous tracks. Consenrich
  bins them by *overlap-weighted* mean signal and uses unit-scaling by default.
  Provide ``countingParams.scaleFactors`` (and ``scaleFactorsControl`` when using
  controls) to apply explicit multiplicative scaling. Bedgraph file inputs are currently experimental.
