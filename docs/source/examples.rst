Quickstart + Usage
------------------

After installing Consenrich, check the build and try running from the command line with
``consenrich -h``

.. _getting-started:

Getting Started: H3K27ac ChIP-seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This minimal example estimates a consensus from *four* ENCODE
epidermis ChIP-seq experiments with matched input controls.

Input Data
""""""""""

.. list-table:: H3K27ac demo inputs
  :header-rows: 1
  :widths: 18 22 30 30

  * - Experiment
    - Biosample
    - H3K27ac alignment
    - Control alignment
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


Download Alignments
"""""""""""""""""""

Copy and paste the following into your terminal to download and index the BAM
files. You can use ``curl -L -O <URL>`` in place of ``wget <URL>`` if ``wget``
is not available.

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files
  for file in ENCFF793ZHL ENCFF647VPO ENCFF809VKT ENCFF295EFL; do
      wget "$encodeFiles/$file/@@download/$file.bam"
  done
  for ctrl in ENCFF444WVG ENCFF619NYP ENCFF898LKJ ENCFF490MWV; do
      wget "$encodeFiles/$ctrl/@@download/$ctrl.bam"
  done
  samtools index -M *.bam


YAML Configuration
""""""""""""""""""

Save the following as ``demoHistoneChIPSeq.yaml`` in the directory containing
the BAM files:

.. code-block:: yaml
  :name: demoHistoneChIPSeq.yaml

  experimentName: demoHistoneChIPSeq

  genomeParams:
    name: hg38
    chromosomes: [chr21, chr22]
    excludeForNorm: [chrX, chrY]

  inputParams:
    samples:
      - name: H3K27ac_1
        path: ENCFF793ZHL.bam
      - name: H3K27ac_2
        path: ENCFF647VPO.bam
      - name: H3K27ac_3
        path: ENCFF809VKT.bam
      - name: H3K27ac_4
        path: ENCFF295EFL.bam
      - name: input_1
        path: ENCFF444WVG.bam
        role: control
      - name: input_2
        path: ENCFF619NYP.bam
        role: control
      - name: input_3
        path: ENCFF898LKJ.bam
        role: control
      - name: input_4
        path: ENCFF490MWV.bam
        role: control


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run H3K27ac demo

  % consenrich --config demoHistoneChIPSeq.yaml --verbose

The run will output a state bigWig,
uncertainty bigWig, ROCCO narrowPeak file, and ROCCO gappedPeak file. With the
current package version, the principal output files are:

.. code-block:: text

  demoHistoneChIPSeq_consenrich_state.VERSION.bw
  demoHistoneChIPSeq_consenrich_uncertainty.VERSION.bw
  consenrichOutput_demoHistoneChIPSeq_state.VERSION_rocco.narrowPeak
  consenrichOutput_demoHistoneChIPSeq_state.VERSION_rocco.gappedPeak

The tags switch from `state` to `stateShrunk` and from `uncertainty` to `uncertaintyShrunk` for ``outputParams.stateShrinkageEnabled``.
To retain human-readable bedGraph files, set ``outputParams.bedGraphEnabled: true`` in the configuration.

.. _atac-demo:

ATAC-seq Demo
~~~~~~~~~~~~~

This demo estimates a consensus chromatin accessibility signal, uncertainty, and peaks from *ten* ENCODE
ATAC-seq alignments.

Download Alignments
"""""""""""""""""""

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files
  for file in ENCFF326QXM ENCFF447ZRG ENCFF462RHM ENCFF495DQP ENCFF497QOS ENCFF632MBC ENCFF687QML ENCFF767FGV ENCFF919PWF ENCFF949CVL; do
      wget "$encodeFiles/$file/@@download/$file.bam"
  done
  samtools index -M *.bam


YAML Configuration
""""""""""""""""""

Save the following as ``atacDemo.yaml``:

.. code-block:: yaml
  :name: atacDemo.yaml

  experimentName: atacDemo

  genomeParams:
    name: hg38
    # chromosomes: [chr11, chr22] # faster: Uncomment and specify chromosomes
    excludeForNorm: [chrX, chrY]

  inputParams:
    bamFiles:
      - ENCFF326QXM.bam
      - ENCFF447ZRG.bam
      - ENCFF462RHM.bam
      - ENCFF495DQP.bam
      - ENCFF497QOS.bam
      - ENCFF632MBC.bam
      - ENCFF687QML.bam
      - ENCFF767FGV.bam
      - ENCFF919PWF.bam
      - ENCFF949CVL.bam


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run ATAC demo

  % consenrich --config atacDemo.yaml --verbose

Results
"""""""

.. image:: ../images/atacDemo.png
   :alt: ATAC-seq demo IGV snapshot
   :align: center
   :width: 100%


Broad Mark ChIP-seq Demo
~~~~~~~~~~~~~~~~~~~~~~~~

This demo estimates a consensus H3K4me1 signal from *six* ENCODE ChIP-seq experiments with matched input controls.

Download Alignments
"""""""""""""""""""

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files

  while read -r accession output; do
      wget -O "$output" "$encodeFiles/$accession/@@download/$accession.bam"
  done <<'EOF'
  ENCFF451FGF ENCFF451FGF_ENCSR724MJX_treatment_GRCh38.bam
  ENCFF660DDB ENCFF660DDB_ENCSR438QZN_treatment_GRCh38.bam
  ENCFF828GWI ENCFF828GWI_ENCSR564QBS_treatment_GRCh38.bam
  ENCFF671AAF ENCFF671AAF_ENCSR299NYB_treatment_GRCh38.bam
  ENCFF366HMH ENCFF366HMH_ENCSR817JNE_treatment_GRCh38.bam
  ENCFF392MXC ENCFF392MXC_ENCSR485LPA_treatment_GRCh38.bam
  ENCFF013ION ENCFF013ION_ENCSR109IWL_control_GRCh38.bam
  ENCFF730PAF ENCFF730PAF_ENCSR526EXI_control_GRCh38.bam
  ENCFF971XWL ENCFF971XWL_ENCSR945LPX_control_GRCh38.bam
  ENCFF271VZL ENCFF271VZL_ENCSR261PLD_control_GRCh38.bam
  ENCFF944AYC ENCFF944AYC_ENCSR061EXX_control_GRCh38.bam
  ENCFF273EYI ENCFF273EYI_ENCSR120ITZ_control_GRCh38.bam
  EOF

  samtools index -M *.bam


YAML Configuration
""""""""""""""""""

Save the following as ``bigH3K4me1Demo.yaml``:

.. code-block:: yaml
  :name: bigH3K4me1Demo.yaml

  experimentName: bigH3K4me1Demo

  genomeParams:
    name: hg38
    chromosomes: [chr21, chr22]
    excludeForNorm: [chrX, chrY]

  inputParams:
    samples:
      - name: ENCSR724MJX_H3K4me1
        path: ENCFF451FGF_ENCSR724MJX_treatment_GRCh38.bam
      - name: ENCSR438QZN_H3K4me1
        path: ENCFF660DDB_ENCSR438QZN_treatment_GRCh38.bam
      - name: ENCSR564QBS_H3K4me1
        path: ENCFF828GWI_ENCSR564QBS_treatment_GRCh38.bam
      - name: ENCSR299NYB_H3K4me1
        path: ENCFF671AAF_ENCSR299NYB_treatment_GRCh38.bam
      - name: ENCSR817JNE_H3K4me1
        path: ENCFF366HMH_ENCSR817JNE_treatment_GRCh38.bam
      - name: ENCSR485LPA_H3K4me1
        path: ENCFF392MXC_ENCSR485LPA_treatment_GRCh38.bam

      - name: ENCSR109IWL_input_for_ENCSR724MJX
        path: ENCFF013ION_ENCSR109IWL_control_GRCh38.bam
        role: control
      - name: ENCSR526EXI_input_for_ENCSR438QZN
        path: ENCFF730PAF_ENCSR526EXI_control_GRCh38.bam
        role: control
      - name: ENCSR945LPX_input_for_ENCSR564QBS
        path: ENCFF971XWL_ENCSR945LPX_control_GRCh38.bam
        role: control
      - name: ENCSR261PLD_input_for_ENCSR299NYB
        path: ENCFF271VZL_ENCSR261PLD_control_GRCh38.bam
        role: control
      - name: ENCSR061EXX_input_for_ENCSR817JNE
        path: ENCFF944AYC_ENCSR061EXX_control_GRCh38.bam
        role: control
      - name: ENCSR120ITZ_input_for_ENCSR485LPA
        path: ENCFF273EYI_ENCSR120ITZ_control_GRCh38.bam
        role: control


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run H3K4me1 demo

  % consenrich --config bigH3K4me1Demo.yaml --verbose

Results
"""""""

.. image:: ../images/bigH3K4me1Demo.png
   :alt: H3K4me1 demo IGV snapshot
   :align: center
   :width: 100%


.. _scatac-demo:

Consenrich for Single-Cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this demo, we use publicly available scATAC-seq data to infer signal and peaks for Goblet, Early Colonocyte, Colonocyte, and BEST4 cells in the ascending colon. Note that Consenrich can also accept data from other single-cell epigenomic assays, e.g., scCut&Tag.

* **Paper:** `Single-cell analysis of chromatin accessibility in the human intestine identifies regulatory programs and clarifies genetic associations in Crohn’s disease <https://www.nature.com/articles/s41588-026-02755-z>`_
* **GEO accession:** `GSE273194 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273194>`_
* **Cell annotations:** `Zenodo <https://zenodo.org/records/21508422>`_

.. list-table::
  :header-rows: 1
  :widths: 10 26 14 22 16 12

  * - Donor
    - GEO sample
    - Goblet
    - Early Colonocyte
    - Colonocyte
    - BEST4
  * - P3
    - `GSM8423799 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8423799>`_
    - 412
    - 690
    - 1277
    - 46
  * - P4
    - `GSM8423800 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8423800>`_
    - 647
    - 983
    - 1125
    - 56
  * - P6
    - `GSM8423801 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8423801>`_
    - 893
    - 1355
    - 1183
    - 54
  * - P12
    - `GSM8423805 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8423805>`_
    - 336
    - 591
    - 811
    - 15
  * - P13
    - `GSM8423806 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8423806>`_
    - 403
    - 827
    - 1034
    - 74

Download Fragments and Cell Labels
""""""""""""""""""""""""""""""""""

This will be about 6 GB of data. You can remove donors and adapt the YAML config below as needed to reduce size.

In your terminal (assuming `bash`):

.. code-block:: bash

  geoFiles=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8423nnn

  # you can use wget or curl, depending
  while read -r accession donor; do
      fragmentFile="${accession}_${donor}-AC-CDnonInf_fragments.tsv.gz"
      curl -fLO "$geoFiles/$accession/suppl/$fragmentFile"
      curl -fLO "$geoFiles/$accession/suppl/$fragmentFile.tbi.gz"
      gzip -dc "$fragmentFile.tbi.gz" > "$fragmentFile.tbi"
  done <<'EOF'
  GSM8423799 P3
  GSM8423800 P4
  GSM8423801 P6
  GSM8423805 P12
  GSM8423806 P13
  EOF

  curl -fL -o scATAC_cell_metadata.csv \
      'https://zenodo.org/records/21508422/files/scATAC_cell_metadata.csv?download=1'


YAML Configuration
""""""""""""""""""""

Select an input format to view its configuration. Each option uses
five donors, four cell groups, and CPM normalization. The Pott downloads above match the first option.

See the :ref:`single-cell API reference <single-cell-api>` for field types and defaults.

.. tab-set::

  .. tab-item:: Fragments + table

    Use fragment files and the downloaded CSV cell labels.

    .. literalinclude:: ../examples/scATACFragments.yaml
      :language: yaml
      :caption: scATACFragments.yaml

    .. code-block:: console

      % consenrich --config scATACFragments.yaml --verbose2

  .. tab-item:: Fragments + H5AD

    Use fragment files and labels stored in ``scATAC_cell_metadata.h5ad``.
    In this case, the label file supplies ``.obs`` only.

    For an H5AD option, install the optional dependency:

    .. code-block:: bash

      pip install "consenrich[singlecell]"

    See the :ref:`single-cell API reference <single-cell-api>` for H5AD input
    requirements and metadata fields.

    .. literalinclude:: ../examples/scATACFragmentsH5ADMetadata.yaml
      :language: yaml
      :caption: scATACFragmentsH5ADMetadata.yaml

    .. code-block:: console

      % consenrich --config scATACFragmentsH5ADMetadata.yaml --verbose2

  .. tab-item:: H5AD cells

    This case uses one raw cell-by-bin count matrix per donor, such as
    ``P3.cells.h5ad``. ``matrixRows: cells`` sums selected cells per donor and
    cell group into internal pseudobulks.

    For an H5AD option, install the optional dependency:

    .. code-block:: bash

      pip install "consenrich[singlecell]"

    See the :ref:`single-cell API reference <single-cell-api>` for H5AD input
    requirements and metadata fields.

    .. literalinclude:: ../examples/scATACCells.yaml
      :language: yaml
      :caption: scATACCells.yaml

    .. code-block:: console

      % consenrich --config scATACCells.yaml --verbose2

  .. tab-item:: H5AD pseudobulks

    This case uses one raw pseudobulk-by-bin count matrix per donor, such as
    ``P3.pseudobulks.h5ad``. ``matrixRows: pseudobulks`` selects one prepared
    pseudobulk per donor and cell group.

    For an H5AD option, install the optional dependency:

    .. code-block:: bash

      pip install "consenrich[singlecell]"

    See the :ref:`single-cell API reference <single-cell-api>` for H5AD input
    requirements and metadata fields.

    .. literalinclude:: ../examples/scATACPseudobulks.yaml
      :language: yaml
      :caption: scATACPseudobulks.yaml

    .. code-block:: console

      % consenrich --config scATACPseudobulks.yaml --verbose2

Run the command shown in the selected tab from the directory containing its
YAML file and inputs. Outputs are written under ``scATACDemo/<cell group>/``.

.. _scatac-results:

Results
"""""""

Browser snapshots for each cell type (uncertainty tracks omitted for spacing).
Signac and ArchR are included here as familiar references, both run with generic defaults.

Goblet
^^^^^^

.. image:: ../images/scATACGobletIGV.png
   :alt: Goblet donor tracks, Consenrich signal and peaks, and Signac and ArchR peaks in IGV
   :align: center
   :width: 100%
   :class: no-scaled-link

Early Colonocyte
^^^^^^^^^^^^^^^^

.. image:: ../images/scATACEarlyColonocyteIGV.png
   :alt: Early Colonocyte donor tracks, Consenrich signal and peaks, and Signac and ArchR peaks in IGV
   :align: center
   :width: 100%
   :class: no-scaled-link

Colonocyte
^^^^^^^^^^

.. image:: ../images/scATACColonocyteIGV.png
   :alt: Colonocyte donor tracks, Consenrich signal and peaks, and Signac and ArchR peaks in IGV
   :align: center
   :width: 100%
   :class: no-scaled-link

BEST4
^^^^^

.. image:: ../images/scATACBEST4IGV.png
   :alt: BEST4 donor tracks, Consenrich signal and peaks, and Signac and ArchR peaks in IGV
   :align: center
   :width: 100%
   :class: no-scaled-link
