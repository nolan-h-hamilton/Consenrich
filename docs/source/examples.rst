Quickstart + Usage
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Quickstart + Usage
   :name: Usage

After installing Consenrich, you can run it via the command line (``consenrich -h``) or programmatically using the Python/Cython :ref:`API`.

Getting Started: Minimal Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: minimal

Here, a brief analysis using H3K27ac ChIP-seq data is carried out for demonstration.

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


Copy and paste the following YAML into a file named ``demoHistoneChIPSeq.yaml``. For a quick trial run (:math:`\approx` 1 minute), you can restrict analysis to a subset of chromosomes: To reproduce the results shown in the IGV browser snapshot below add ``genomeParams.chromosomes: [chr21, chr22]`` to the configuration file.

.. code-block:: yaml
  :name: demoHistoneChIPSeq.yaml

  experimentName: demoHistoneChIPSeq
  genomeParams.name: hg38
  genomeParams.excludeForNorm: [chrX, chrY]
  inputParams.bamFiles: [ENCFF793ZHL.bam,
  ENCFF647VPO.bam,
  ENCFF809VKT.bam,
  ENCFF295EFL.bam]
  inputParams.bamFilesControl: [ENCFF444WVG.bam,
  ENCFF619NYP.bam,
  ENCFF898LKJ.bam,
  ENCFF490MWV.bam]
  matchingParams.templateNames: [db2]
  matchingParams.cascadeLevels: [2]
  matchingParams.iters: 25_000
  matchingParams.alpha: 0.01

.. admonition:: Control Inputs
  :class: tip

  Omit ``inputParams.bamFilesControl`` for ATAC-seq, DNase-seq, Cut&Run, and other assays where no control is available or applicable.


Run Consenrich
"""""""""""""""""""""

Invoke the command-line interface to run Consenrich:

.. code-block:: bash
  :name: Run Consenrich

  consenrich --config demoHistoneChIPSeq.yaml --verbose

.. note::
  The command-line interface is a convenience wrapper that may not expose all available objects or more niche features.
  Some users may find it beneficial to run Consenrich programmatically (via Jupyter notebooks, Python scripts), as the :ref:`API` enables
  greater flexibility to apply custom preprocessing steps and various context-specific protocols within existing workflows.


Visualizing Results
""""""""""""""""""""""""""

We display results at a **50kb** enhancer-rich region overlapping `MYH9`.

.. image:: ../images/ConsenrichIGVdemoHistoneChIPSeq.png
  :alt: Output Consenrich Signal Estimates
    :width: 800px
    :align: left


Input alignments (Black) and ENCODE ``fold change over control`` bigWig files for each sample (Red) are displayed for reference.

* Consenrich signal estimate track: ``demoHistoneChIPSeq_consenrich_state.bw``

* Consenrich precision-weighted residual track: ``demoHistoneChIPSeq_consenrich_residuals.bw``

* Consenrich 'Matched' regions showing a structured enrichment pattern: ``consenrichOutput_demoHistoneChIPSeq_matches.narrowPeak``


.. admonition:: `Consenrich+ROCCO`: Consensus Peak Calling
  :class: tip

  Consenrich can markedly improve conventional consensus peak calling (See 'Results' in the `manuscript preprint <https://www.biorxiv.org/content/10.1101/2025.02.05.636702v2>`_).

  `ROCCO <https://github.com/nolan-h-hamilton/ROCCO>`_ accepts Consenrich bigWig files as input and is particularly well-suited to leverage the sharpened signal tracks. In the example above, to call peaks using the `Consenrich+ROCCO` protocol,

  .. code-block:: console

	  python -m pip install rocco --upgrade
	  rocco -i demoHistoneChIPSeq_consenrich_state.bw -g hg38


  Alternative peak calling methods that accept bedGraph or bigWig input (e.g., `MACS' bdgpeakcall <https://macs3-project.github.io/MACS/docs/bdgpeakcall.html>`_) should be capable of utilizing Consenrich signal tracks. Only ROCCO has been evaluated for this task to date.

  Depending on the signal target and goals of analysis, the :ref:`matching` algorithm available with Consenrich can also be used to complement existing peak calling methods---e.g., detecting 'structured' enrichment patterns across multiple samples or identifying subpeaks within broad regions of interest.

Further analyses are available in :ref:`additional-examples`.

.. _additional-examples:

Additional Examples and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Additional Examples and Computational Benchmarking


This section of the documentation will be regularly updated to include a breadth of assays, downstream analyses, and runtime benchmarks.

ATAC-seq
""""""""""""""""

- Input data: :math:`m=20` ATAC-seq BAM files derived from lymphoblastoid cell lines (ENCODE)

Environment
''''''''''''''

- MacBook MX313LL/A (arm64)
- Python 3.12.9
- `HTSlib (Samtools) <https://www.htslib.org/>`_ 1.21
- `Bedtools <https://bedtools.readthedocs.io/en/latest/>`_ 2.31.1

Names and versions of packages that are relevant to computational performance. These specific versions are *not required* but are included for reproducibility.

.. list-table::
     :header-rows: 1
     :widths: 40 60

     * - Package
       - Version
     * - ``cython``
       - 3.1.2
     * - ``numpy``
       - 2.3.2
     * - ``scipy``
       - 1.16.1
     * - ``consenrich``
       - 0.4.2b0


Run with the following YAML config file `atac20Benchmark.yaml`. Note that globs, e.g., `*.bam`, are allowed, but each BAM file is listed here explicitly for reproducibility.

.. code-block:: yaml

  experimentName: atac20Benchmark
  genomeParams.name: hg38
  genomeParams.excludeForNorm: ['chrX', 'chrY']
  genomeParams.excludeChroms: ['chrX','chrY']
  inputParams.bamFiles: [ENCFF326QXM.bam,
    ENCFF497QOS.bam,
    ENCFF919PWF.bam,
    ENCFF447ZRG.bam,
    ENCFF632MBC.bam,
    ENCFF949CVL.bam,
    ENCFF462RHM.bam,
    ENCFF687QML.bam,
    ENCFF495DQP.bam,
    ENCFF767FGV.bam,
    ENCFF009NCL.bam,
    ENCFF110EWQ.bam,
    ENCFF797EAL.bam,
    ENCFF801THG.bam,
    ENCFF216MFD.bam,
    ENCFF588QWF.bam,
    ENCFF795UPB.bam,
    ENCFF395ZMS.bam,
    ENCFF130DND.bam,
    ENCFF948HNW.bam
  ]
  processParams.minQ: 0.05 # lower bounds minimum process noise variance
  observationParams.minR: 0.05 # lower bounds minimum observation noise variance
  countingParams.stepSize: 25
  matchingParams.templateNames: [db2] # structured enrichment patterns w/ db2
  matchingParams.cascadeLevels: [2]
  matchingParams.alpha: 0.01
  samParams.samThreads: 1
  samParams.chunkSize: 1000000


Run Consenrich
''''''''''''''''''''

.. code-block:: console

  consenrich --config atac20Benchmark.yaml --verbose


After running, the following files will be generated in the current working directory:

* Consenrich signal estimate track: ``atac20Benchmark_consenrich_state.bw``

* Consenrich precision-weighted residual track: ``atac20Benchmark_consenrich_residuals.bw``

* Detected peak-like regions (:ref:`matching`): ``consenrichOutput_atac20Benchmark_matches.narrowPeak``


Visualizing Results
''''''''''''''''''''''''''''

- Output tracks and features are visualized above in a **100kb** region around the transcription start site of `NOTCH1`.

.. image:: ../benchmarks/atac20/images/atac20BenchmarkIGVSpib.png
    :alt: IGV Browser Snapshot
    :width: 800px
    :align: left



The bigWig files `atac20Benchmark_consenrich_state.bw` and `atac20Benchmark_consenrich_residuals.bw` are overlaid (blue/red) for comparison.

Regions showing a structured enrichment pattern (`db2, level=2`) are positioned above the Consenrich signal as BED features in narrowPeak format.

- Focused view over a **25kb** subregion:

.. image:: ../benchmarks/atac20/images/atac20BenchmarkIGVSpib25KB.png
    :alt: IGV Browser Snapshot (25kb)
    :width: 800px
    :align: left

Runtime and Memory Profiling
''''''''''''''''''''''''''''''''''

Memory was profiled using the package `memory-profiler <https://pypi.org/project/memory-profiler/>`_. See the plot below for memory usage over time. Function calls are marked as notches.

Note that the repeated sampling of memory every 0.1 seconds during profiling introduces some overhead that affects runtime.

.. image:: ../benchmarks/atac20/images/atac20BenchmarkMemoryPlot.png
    :alt: Time vs. Memory Usage (`memory-profiler`)
    :width: 800px
    :align: center


.. tip::

  Memory cost can be reduced by decreasing `samParams.chunkSize` in the configuration file. Smaller chunk sizes may affect runtime due to overhead from more frequent file I/O, however.

