Quickstart + Usage
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Quickstart + Usage
   :name: Usage

After installing Consenrich, you can run it via the command line (``consenrich -h``) or programmatically using the Python/Cython :ref:`API`.

We provide several usage examples below. See also :ref:`files` and :ref:`tips` for more information.


.. _getting-started:

Getting Started: Minimal Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: minimal

A brief analysis using H3K27ac (narrow mark) ChIP-seq data is carried out for demonstration.

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
  genomeParams.chromosomes: [chr21, chr22] # remove to run genome-wide
  genomeParams.excludeForNorm: [chrX, chrY]

  inputParams.bamFiles: [ENCFF793ZHL.bam,
  ENCFF647VPO.bam,
  ENCFF809VKT.bam,
  ENCFF295EFL.bam]

  inputParams.bamFilesControl: [ENCFF444WVG.bam,
  ENCFF619NYP.bam,
  ENCFF898LKJ.bam,
  ENCFF490MWV.bam]

  # Optional: call 'structured peaks' via `consenrich.matching`
  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [3, 3]
  matchingParams.minMatchLengthBP: -1
  matchingParams.mergeGapBP: -1


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


* For reference, ENCODE peaks for the same `Experiments` and donor samples are included (black):
  * `Histone ChIP-seq (unreplicated) <https://www.encodeproject.org/pipelines/ENCPL841HGV/>`_ (MACS2 calls,  partition concordance)

.. image:: ../images/ConsenrichIGVdemoHistoneChIPSeq.png
  :alt: Output Consenrich Signal Estimates
    :width: 700px
    :align: left


.. _additional-examples:

Additional Examples and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Additional Examples and Computational Benchmarking


This section of the documentation will be regularly updated to include a breadth of assays, downstream analyses, and runtime benchmarks.

ATAC-seq
""""""""""""""""

- Input data (`atac20`): :math:`m=20` ATAC-seq BAM files derived from lymphoblastoid cell lines (ENCODE)
- Varying data quality (e.g., `Extremely low read depth <https://www.encodeproject.org/data-standards/audits/#extremely_low_read_depth>`_)

Environment
''''''''''''''

- MacBook MX313LL/A (arm64)
- Python `3.12.9`
- Consenrich `v0.7.4b3`
- `HTSlib (Samtools) <https://www.htslib.org/>`_ 1.22.1
- `Bedtools <https://bedtools.readthedocs.io/en/latest/>`_ 2.31.1

Names and versions of packages that are relevant to computational performance. These specific versions are *not required* but are included for reproducibility.

.. list-table::
     :header-rows: 1
     :widths: 40 60

     * - Package
       - Version
     * - ``cython``
       - 3.1.4
     * - ``numpy``
       - 2.3.3
     * - ``scipy``
       - 1.16.2


Configuration
''''''''''''''''''''''''''''

Run with the following YAML config file `atac20Benchmark.yaml`. Note that several parameters are listed and/or adjusted for demonstration purposes.

Note that globs, e.g., `*.bam`, are allowed, but the BAM file names are listed explicitly in the config to show their ENCODE accessions for reference.


.. code-block:: yaml

  experimentName: atac20Benchmark
  genomeParams.name: hg38
  genomeParams.excludeChroms: ['chrY']
  genomeParams.excludeForNorm: ['chrX', 'chrY']
  inputParams.bamFiles: [
    ENCFF326QXM.bam,
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

  matchingParams.templateNames: [haar, db2]
  matchingParams.cascadeLevels: [3, 3]
  matchingParams.minMatchLengthBP: -1
  matchingParams.mergeGapBP: -1


Run Consenrich
''''''''''''''''''''

.. code-block:: console

  % consenrich --config atac20Benchmark.yaml --verbose


Results
''''''''''''''''''''''''''''

Consenrich outputs are visualized over a 25kb genomic region centered around `LYL1`, which is highly expressed in LCLs.


.. image:: ../benchmarks/atac20/images/atac20BenchmarkIGVSpib25KB.png
    :alt: IGV Browser Snapshot (25kb)
    :width: 800px
    :align: left

**Evaluating Structured Peak Results: cCRE Overlaps**

Here, we count *genome-wide* overlaps between Consenrich-detected matches and previously-identified candidate regulatory elements (`ENCODE4 GRCh38 cCREs <https://screen.wenglab.org/downloads>`_).


Note that the ENCODE cCREs are not specific to our lymphoblastoid input dataset (`atac20`) and strict concordance is not expected. Nonetheless, given the breadth of cell types and tissues surveyed in ENCODE, a substantial overlap between Consenrich-detected structured peaks and cCREs is desirable.

* We first count:

  - The total number of Consenrich-detected structured peaks (165,090)
  - The number of *unique* Consenrich-detected structured peaks sharing at least a :math:`25\%` *reciprocal* overlap with an ENCODE4 cCRE (148,767)

  .. code-block:: console

    % bedtools intersect \
      -a consenrichOutput_atac20Benchmark_matches.mergedMatches.narrowPeak \
      -b GRCh38-cCREs.bed \
      -f 0.25 -r -u \
      | wc -l


* We also evaluate overlaps compared to a null baseline addressing random chance,

  |    *Controlling for peak size (avg. 534 bp) and chromosome placement, how many cCRE overlaps would we expect by randomly selecting 165,090 regions?*

  We invoke `bedtools shuffle <https://bedtools.readthedocs.io/en/latest/content/tools/shuffle.html>`_,

  .. code-block:: console

    % bedtools shuffle \
      -i consenrichOutput_atac20Benchmark_matches.mergedMatches.narrowPeak \
      -g hg38.sizes \
      -chrom \
      | bedtools intersect -a stdin -b GRCh38-cCREs.bed -f 0.25 -r -u \
      | wc -l

  and aggregate results for `N=250` independent trials to build an empirical distribution for cCRE-hits under our null model.


We find a substantial overlap between Consenrich-detected regions and cCREs, with a significant enrichment versus null hits (:math:`\hat{p} \approx 0.0039`):

+------------------------------------------------------------------------------------------+----------------------------------------------+
| Feature                                                                                  | Value                                        |
+==========================================================================================+==============================================+
| Consenrich: Total structured peaks (α=0.05)                                              | 165,090                                      |
+------------------------------------------------------------------------------------------+----------------------------------------------+
| Consenrich: Distinct cCRE overlaps*                                                      | 148,767                                      |
+------------------------------------------------------------------------------------------+----------------------------------------------+
| Consenrich: Percent overlapping                                                          | **90.1%**                                    |
+------------------------------------------------------------------------------------------+----------------------------------------------+
| Random (``shuffle``): Distinct cCRE overlaps*                                            | μ ≈ 56,652.8,  σ ≈ 196.9                     |
+------------------------------------------------------------------------------------------+----------------------------------------------+
| Random (``shuffle``): Percent overlapping                                                | ≈ **34.2%**                                  |
+------------------------------------------------------------------------------------------+----------------------------------------------+

:math:`\ast`: ``bedtools intersect -f 0.25 -r -u``


.. _runtimeAndMemoryProfilingAtac20:

Runtime and Memory Profiling
''''''''''''''''''''''''''''''''''

Memory was profiled using the package `memory-profiler <https://pypi.org/project/memory-profiler/>`_. See the plot below for memory usage over time. Function calls are marked as notches.

Note that the repeated sampling of memory every 0.1 seconds during profiling introduces some overhead that affects runtime.

.. image:: ../benchmarks/atac20/images/atac20BenchmarkMemoryPlot.png
    :alt: Time vs. Memory Usage (`memory-profiler`)
    :width: 800px
    :align: center

----

ChIP-seq: Broad Histone Marks
"""""""""""""""""""""""""""""""""""""""""""""

In this demo, we use :math:`m=6` H3K36me3 ChIP-seq samples from separate donors' lung tissues.

Twelve total alignment files (single-end, treatment/control input pairs) are used. See the YAML below for `ENCFF<fileID>` accessions.


Configuration
''''''''''''''''''''''''''''

* ``experimentH3K36me3.yaml``.

  .. code-block:: yaml

    experimentName: experimentH3K36me3
    genomeParams.name: hg38
    genomeParams.excludeChroms: [chrY]
    genomeParams.excludeForNorm: [chrX, chrY]

    inputParams.bamFiles: [ENCFF441SHP.bam,
     ENCFF450ORQ.bam,
     ENCFF903UTS.bam,
     ENCFF790HIV.bam,
     ENCFF591YNK.bam,
     ENCFF870AMP.bam
     ]

    inputParams.bamFilesControl: [ENCFF794QJK.bam,
     ENCFF831MFQ.bam,
     ENCFF660HBS.bam,
     ENCFF430OFG.bam,
     ENCFF347ENG.bam,
     ENCFF648HNK.bam
    ]

    # Increased from default (25 bp) given H3K36me3's classification
    # as a broad histone mark associated with gene bodies
    countingParams.stepSize: 100

    matchingParams.templateNames: [haar, db2]
    matchingParams.cascadeLevels: [3, 3]
    matchingParams.minMatchLengthBP: -1
    matchingParams.mergeGapBP: -1


Run Consenrich
''''''''''''''''''''''''''''

.. code-block:: console

  % consenrich --config experimentH3K36me3.yaml --verbose


Results
''''''''''''''''''''''''''''

Signal estimates, weighted residuals, and structured peaks (via :ref:`matching`) over a **150kb region** spanning `LINC01176`, `NOD1`, `GGCT`:

.. image:: ../benchmarks/H3K36me3/images/Consenrich_ENTexFour_DualMark.png
    :alt: H3K36me3 Intron-Exon
    :width: 800px
    :align: left


**Genome-Wide Exonic Enrichment**

We evaluate signal intensities and peak density at exonic regions given H3K36me3's association with actively transcribed gene bodies.

Using the set of Consenrich peaks, we apply `bedtools shuffle` to permute their genomic locations while preserving chromosome assignment and feature lengths to build a null distribution. (This is effectively the same procedure as in the previous `ATAC-seq` example.)


- Peaks: Exon Overlaps (bp)

  - *Shuffled (Null)*: mean exonic overlap (:math:`N=250` iterations): μ ≈ 1,134,214, σ ≈ 25,008.34
  - *Consenrich*: observed exonic overlap: **7,510,079**

    - *Fold-enrichment*: :math:`7.14`, :math:`\hat{p} \approx 0.0039`

- Relative Signals: All Consenrich Peaks vs. Consenrich Peaks :math:`\cap` Exons

  - Median signal at peaks that overlap exons: **9.554**
  - Median signal at all peaks: **5.950**



.. _files:

File Formats
~~~~~~~~~~~~~~~~~~~~~~

* Input

  * Per-sample sequence alignment files (BAM format)

    * *Optional*: Control/input alignment files (e.g., ChIP-seq)

  * Note, if using Consenrich programmatically, users can provide preprocessed sample-by-interval count matrices directly instead of BAM files (see :func:`consenrich.core.runConsenrich`)

* Output

  * *Signal estimate track*: ``<experimentName>_consenrich_state.bw``

    * This track records genome-wide Consenrich estimates for the targeted signal of interest
    * A human-readable bedGraph file is also generated: ``consenrichOutput_<experimentName>_consenrich_state.bedGraph``

  * *Precision-weighted residual track*: ``<experimentName>_consenrich_residuals.bw``

    * This track records genome-wide differences between (*a*) Consenrich estimates and (*b*) observed sample data -- after accounting for regional + sample-specific uncertainty.
    * These values can reflect model mismatch: Where they are large (magnitude), the model's estimated uncertainty may fail to explain discrepancies with the observed data.
    * A human-readable bedGraph file is also generated: ``consenrichOutput_<experimentName>_consenrich_residuals.bedGraph``

  * *Structured peak calls* (Optional): ``<experimentName>_matches.mergedMatches.narrowPeak``

    * BED-like annotation of enriched signal regions showing a regular structure. Only generated if the matching algorithm is invoked.
    * See :ref:`matching` and :func:`consenrich.matching.matchWavelet`

See also :class:`outputParams` in the :ref:`API` for additional output options.

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

The budgeted/total-variation-regularized optimization procedure performed by ROCCO to select consensus peak regions prevents excessive multiple comparisons downstream and enforces biological plausibility. Other peak calling methods---including the :ref:`matching` algorithm packaged with Consenrich---that accept bedGraph or bigWig input (e.g., `MACS' bdgpeakcall <https://macs3-project.github.io/MACS/docs/bdgpeakcall.html>`_) may also prove viable, but only Consenrich+ROCCO has been extensively benchmarked for differential accessibility analyses to date.


Matching Algorithm: Command-line Usage
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To avoid a full run/rerun of Consenrich when calling structured peaks, the matching algorithm can be run directly at the command-line on existing Consenrich-generated bedGraph files. For example:

.. code-block:: console

  % consenrich \
    --match-bedGraph consenrichOutput_<experimentName>_state.bedGraph \
    --match-template haar \
    --match-level 3 \
    --match-alpha 0.01

This calls structured peaks with a Haar template/level 3 and significance threshold :math:`\alpha=0.01`. Run ``consenrich -h`` for additional options.

For more details on the matching algorithm in general, see :ref:`matching` and :func:`consenrich.matching.matchWavelet` for more details.


Broad, Heterochromatic and/or Repressive targets
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
* When targeting large domain-level features, consider increasing `countingParams.stepSize` and/or `detrendParams.detrendWindowLengthBP` from their defaults (25 bp, 10000 bp respectively) to prioritize larger-scale trends.

  * For instance, polycomb-repressed domains (H3K27me3) and constitutive heterochromatin (H3K9me3): 

    - `countingParams.stepSize: 100`
    - `detrendParams.detrendWindowLengthBP: 25000`

* When targeting signals associated with *heterochromatin/repression*, consider setting ``observationParams.useALV: true`` in the YAML configuration file to avoid conflating signal with noise.


Preprocessing and Calibration of Uncertainty Metrics
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To promote homoskedastic, symmetric, and uncorrelated residuals that are amenable to analyses requiring well-calibrated/absolute uncertainty quantification, consider:

.. code-block:: yaml

  countingParams.applyLog: true # or `applyAsinh`` to maintain linearity near zero
  stateParams.boundState: false

Otherwise---particularly in the absence of control input samples---Consenrich outputs such as :math:`\sqrt{\widetilde{P}_{[i,11]}}~` (`outputParams.writeStateStd`) are better interpreted as *relative, pointwise* measures of uncertainty (i.e., higher vs. lower uncertainty regions).

