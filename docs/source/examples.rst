Quickstart + Usage
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Quickstart + Usage
   :name: Usage

After installing Consenrich, you can run it via the command line (``consenrich -h``) or programmatically using the Python/Cython :ref:`API`.


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
  matchingParams.templateNames: [haar, db2, db3]
  matchingParams.cascadeLevels: [2,2,2]


.. admonition:: Control Inputs
  :class: tip

  Omit ``inputParams.bamFilesControl`` for ATAC-seq, DNase-seq, Cut&Run, and other assays where no control is available or applicable.


Run Consenrich
"""""""""""""""""""""

.. code-block:: console
  :name: Run Consenrich

  % consenrich --config demoHistoneChIPSeq.yaml --verbose

Results
""""""""""""""""""""""""""

* We display Consenrich results (blue) over a 100 kb locus in human chromosome 22


.. image:: ../images/ConsenrichIGVdemoHistoneChIPSeq.png
  :alt: Output Consenrich Signal Estimates
    :width: 600px
    :align: left


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

Other peak calling methods can be applied downstream, too, including the :ref:`matching` algorithm packaged with Consenrich, that accept a bedGraph or bigWig input (e.g., `MACS' bdgpeakcall <https://macs3-project.github.io/MACS/docs/bdgpeakcall.html>`_). Only Consenrich+ROCCO has been benchmarked for differential accessibility analyses to date.

In general, for workflows of the form ``Consenrich Signal Track --> Peak Caller --> Sample-by-CalledPeaks Count Matrix --> Differential Analysis between Conditions``, it is recommended to use *all samples from all experimental conditions* as input to Consenrich for better control of downstream false discovery rates. See, for example, `Lun and Smyth, 2014`.


Broad Features / Shallow Sequencing Depth
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The default for (``countingParams.intervalSizeBP``) is 25 base pairs, which may be too fine for very broad marks and/or samples with less than :math:`\approx 5\textsf{M}` tags. Consider using larger interval sizes, e.g., 50-200 bp in these cases.

