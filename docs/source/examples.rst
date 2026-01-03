Quickstart + Usage
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Quickstart + Usage
   :name: Usage

After installing Consenrich, you can run it via the command line (``consenrich -h``) or programmatically using the Python/Cython :ref:`API`.

See also :ref:`files` and :ref:`tips` for more information.


.. _getting-started:

Getting Started: Minimal Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: minimal

A brief analysis using H3K27ac (narrow) ChIP-seq data is carried out for demonstration.

Input Data
"""""""""""""""""""""

The input data in this example consists of four donors' treatment and control samples (epidermal tissue) from ENCODE.

.. list-table:: Input Data
  :header-rows: 1
  :widths: 20 20 30 30

  * - Experiment
    - Biosample
    - H3K27ac Alignment
    - Control Alignment
  * - `ENCSR214UZE <https://www.encodeproject.org/experiments/ENCSR214UZE/>`_
    - Epidermis/Female/71
    - `ENCFF793ZHL.bam <https://www.encodeproject.org/files/ENCFF793ZHL/@@download/ENCFF793ZHL.bam>`_
    - `ENCFF444WVG.bam <https://www.encodeproject.org/files/ENCFF444WVG/@@download/ENCFF444WVG.bam>`_
  * - `ENCSR334DRN <https://www.encodeproject.org/experiments/ENCSR334DRN/>`_
    - Epidermis/Male/67
    - `ENCFF647VPO.bam <https://www.encodeproject.org/files/ENCFF647VPO/@@download/ENCFF647VPO.bam>`_
    - `ENCFF619NYP.bam <https://www.encodeproject.org/files/ENCFF619NYP/@@download/ENCFF619NYP.bam>`_
  * - `ENCSR340ZTB <https://www.encodeproject.org/experiments/ENCSR340ZTB/>`_
    - Epidermis/Female/80
    - `ENCFF809VKT.bam <https://www.encodeproject.org/files/ENCFF809VKT/@@download/ENCFF809VKT.bam>`_
    - `ENCFF898LKJ.bam <https://www.encodeproject.org/files/ENCFF898LKJ/@@download/ENCFF898LKJ.bam>`_
  * - `ENCSR386CKJ <https://www.encodeproject.org/experiments/ENCSR386CKJ/>`_
    - Epidermis/Male/75
    - `ENCFF295EFL.bam <https://www.encodeproject.org/files/ENCFF295EFL/@@download/ENCFF295EFL.bam>`_
    - `ENCFF490MWV.bam <https://www.encodeproject.org/files/ENCFF490MWV/@@download/ENCFF490MWV.bam>`_


Download Alignment Files from ENCODE
"""""""""""""""""""""""""""""""""""""""

Copy+paste the following to your terminal to download and index the BAM files for this demo.

You can also use ``curl -O <URL>`` in place of ``wget <URL>`` if the latter is not available on your system.

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files
  for file in ENCFF793ZHL ENCFF647VPO ENCFF809VKT ENCFF295EFL; do
      wget "$encodeFiles/$file/@@download/$file.bam"
  done
  for ctrl in ENCFF444WVG ENCFF619NYP ENCFF898LKJ ENCFF490MWV; do
      wget "$encodeFiles/$ctrl/@@download/$ctrl.bam"
  done
  samtools index -M *.bam


Using a YAML Configuration file
"""""""""""""""""""""""""""""""""""""

.. tip::

   Refer to the ``<process,observation,etc.>Params`` classes in module in the :ref:`API` for complete documentation of configuration options.


Copy and paste the following YAML into a file named ``demoHistoneChIPSeq.yaml``. For a quick trial run (:math:`\approx` 1 minute), you can restrict analysis to a subset of chromosomes: To reproduce the results shown in the browser snapshot, add ``genomeParams.chromosomes: [chr21, chr22]`` to the configuration file.

.. code-block:: yaml
  :name: demoHistoneChIPSeq.yaml

  # v0.8.0rc1
  experimentName: demoHistoneChIPSeq
  genomeParams.name: hg38
  genomeParams.chromosomes: [chr21, chr22] # remove this line to run genome-wide
  genomeParams.excludeForNorm: [chrX, chrY]

  inputParams.bamFiles: [ENCFF793ZHL.bam,
  ENCFF647VPO.bam,
  ENCFF809VKT.bam,
  ENCFF295EFL.bam]

  inputParams.bamFilesControl: [ENCFF444WVG.bam,
  ENCFF619NYP.bam,
  ENCFF898LKJ.bam,
  ENCFF490MWV.bam]

  # Optional: call 'structured peaks'
  matchingParams.templateNames: [haar,haar,db2,db2]
  matchingParams.cascadeLevels: [2,3,2,3]



.. admonition:: Control Inputs
  :class: tip

  Omit ``inputParams.bamFilesControl`` for ATAC-seq, DNase-seq, Cut&Run, and other assays where no control is available or applicable.


Run Consenrich
"""""""""""""""""""""

.. admonition:: Guidance: Command-line vs. Programmatic Usage
  :class: tip
  :collapsible: closed

  The command-line interface is a convenience wrapper that may not expose all available objects or more niche features.
  Some users may find it beneficial to run Consenrich programmatically (e.g., in a Jupyter notebook, Python script), as the :ref:`API` enables
  greater flexibility to apply custom preprocessing steps and various context-specific protocols within existing workflows.


.. code-block:: console
  :name: Run Consenrich

  % consenrich --config demoHistoneChIPSeq.yaml --verbose

Results
""""""""""""""""""""""""""

* We display Consenrich results (blue) at ``APOL2 <--| |--> APOL1``


* For reference, ENCODE peaks (label: `rep1 pseudoreplicated peaks`) for the same `Experiments` and donor samples are included (black):

  * `ENCODE Histone ChIP-seq pipeline (unreplicated) <https://www.encodeproject.org/pipelines/ENCPL841HGV/>`_ (MACS peak calls, partition concordance)

.. image:: ../images/ConsenrichIGVdemoHistoneChIPSeq.png
  :alt: Output Consenrich Signal Estimates
    :width: 600px
    :align: left


.. _files:

File Formats
~~~~~~~~~~~~~~~~~~~~~~

* Input

  * Per-sample sequence alignment files (BAM format)

    * *Optional*: Control/input alignment files (e.g., ChIP-seq)

  * Note, if using Consenrich programmatically, users can provide preprocessed sample-by-interval count matrices directly instead of BAM files (see :func:`consenrich.core.runConsenrich`)


* Output

  * *Posterior Signal estimate track*: ``<experimentName>_consenrich_state.bw``

    * This track records genome-wide Consenrich estimates for the targeted signal of interest
    * A human-readable bedGraph file is also generated: ``consenrichOutput_<experimentName>_consenrich_state.bedGraph``

  * *Posterior state uncertainty track*: ``<experimentName>_consenrich_stateStdDev.bw``

    * Pointwise conditional uncertainty in the primary state estimate, :math:`\sqrt{\widetilde{P}_{i,(11)}}` under the assumed model.
    * Invoke ``consenrich.core.outputParams.writeStateStdDev`` (YAML: ``outputParams.writeWRMS: true``)
    * A human-readable bedGraph file is also generated: ``consenrichOutput_<experimentName>_consenrich_stateStd.bedGraph``

  * *Weighted RMS of Post-Fit Residuals (Optional)*: ``<experimentName>_consenrich_WRMS.bw``

    * Pointwise root-mean-square wrt post-fit residuals (WRMS) to bedGraph, whitened with respect to
      observation uncertainty (variance). Values are computed as the RMS across input samples: :math:`(y_{[j,i]}-\hat{x}_{[0,i]})/\sqrt{R_{j,i}}`,
      where :math:`R_{j,i}` is the observation variance from :math:`\mathbf{R}` (``matrixMunc``).
    * Invoke ``consenrich.core.outputParams.writeWRMS`` (YAML: ``outputParams.writeWRMS: true``)
    * A human-readable bedGraph file is also generated: ``consenrichOutput_<experimentName>_consenrich_WRMS.bedGraph``

  * *Structured peak calls* (Optional): ``<experimentName>_matches.mergedMatches.narrowPeak``

    * BED-like annotation of enriched signal regions showing a regular structure. Only generated if the matching algorithm is invoked.
    * See :ref:`matching` and :func:`consenrich.matching.matchWavelet`


See :class:`outputParams` in the :ref:`API` for full documentation of output options.


.. _tips:

Miscellaneous Guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Consensus Peak Calling + Downstream Differential Analyses
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Consenrich can improve between-group differential analyses that depend on a good set of initial 'candidate' consensus peaks (see `Enhanced Consensus Peak Calling and Differential Analyses in Complex Human Disease <https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2>`_ in the manuscript preprint.)

`ROCCO <https://github.com/nolan-h-hamilton/ROCCO>`_ can accept Consenrich bigWig files as input and is well-suited to leverage high-resolution open chromatin signal estimates while balancing regularity for simultaneous broad/narrow peak calling.

For example, to run the `Consenrich+ROCCO` protocol as it is used in the manuscript,

.. code-block:: console

 % python -m pip install rocco --upgrade
 % rocco -i <experimentName>_consenrich_state.bw \
    -g hg38 -o consenrichRocco_<experimentName>.bed \
    # <...>

The budgeted/total-variation-regularized optimization procedure performed by ROCCO to select consensus peak regions prevents excessive multiple comparisons downstream and enforces biological plausibility. Other peak calling methods can be applied downstream, too,---including the :ref:`matching` algorithm packaged with Consenrich---that accept bedGraph or bigWig input (e.g., `MACS' bdgpeakcall <https://macs3-project.github.io/MACS/docs/bdgpeakcall.html>`_). Only Consenrich+ROCCO has been extensively benchmarked for differential accessibility analyses to date.

In general, for workflows of the form ``Consenrich Signal Track --> Peak Caller --> Sample-by-CalledPeaks Count Matrix --> Differential Analysis between Conditions``, it is recommended to use *all samples from all experimental conditions* as input to Consenrich for better control of downstream false discovery rates. See, for example, `Lun and Smyth, 2014`.


Broad Features / Shallow Sequencing Depth
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Default value is 25 bp (``countingParams.intervalSizeBP``), which may be too fine for very broad marks and/or samples with less than :math:`\approx 5\textsf{M}` tags. Consider using larger interval sizes, e.g., 50-200 bp in these cases.

