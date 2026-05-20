Quickstart + Usage
------------------

After installing Consenrich, you can run it from the command line
(``consenrich -h``) or programmatically using the Python/Cython :ref:`API`.
The examples below are intentionally short demos on a couple chromosomes.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart + Usage
   :name: Usage

.. tip::

   Refer to the ``<process,observation,etc.>Params`` classes in the
   :ref:`API` for complete documentation of configuration options.


.. _getting-started:

Getting Started: H3K27ac ChIP-seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This minimal example estimates a consensus H3K27ac signal from four ENCODE
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
        format: bam
        role: treatment
      - name: H3K27ac_2
        path: ENCFF647VPO.bam
        format: bam
        role: treatment
      - name: H3K27ac_3
        path: ENCFF809VKT.bam
        format: bam
        role: treatment
      - name: H3K27ac_4
        path: ENCFF295EFL.bam
        format: bam
        role: treatment
      - name: input_1
        path: ENCFF444WVG.bam
        format: bam
        role: control
      - name: input_2
        path: ENCFF619NYP.bam
        format: bam
        role: control
      - name: input_3
        path: ENCFF898LKJ.bam
        format: bam
        role: control
      - name: input_4
        path: ENCFF490MWV.bam
        format: bam
        role: control


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run H3K27ac demo

  % consenrich --config demoHistoneChIPSeq.yaml --verbose

The run writes a state bedGraph, uncertainty bedGraph, state bigWig,
uncertainty bigWig, and ROCCO narrowPeak file. With the current package
version, the principal output files are:

.. code-block:: text

  demoHistoneChIPSeq_consenrich_state.v0.10.5a0.bw
  demoHistoneChIPSeq_consenrich_uncertainty.v0.10.5a0.bw
  consenrichOutput_demoHistoneChIPSeq_state.v0.10.5a0_rocco.narrowPeak

Results
"""""""

.. admonition:: Image placeholder

   IGV/browser snapshot of the H3K27ac state estimate, local uncertainty, and
   ROCCO peaks over a representative locus.


.. _atac-demo:

ATAC-seq Demo
~~~~~~~~~~~~~

This demo estimates a consensus chromatin-accessibility signal from ten ENCODE
ATAC-seq alignments. ATAC-seq does not require matched input controls, so only
treatment samples are listed.

Download Alignments
"""""""""""""""""""

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files
  for file in ENCFF009NCL ENCFF110EWQ ENCFF231YYD ENCFF239RGZ ENCFF724QHH ENCFF767FGV ENCFF801THG ENCFF822BKT ENCFF925NDY ENCFF978QLZ; do
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
    chromosomes: [chr21, chr22]
    excludeForNorm: [chrX, chrY]

  inputParams:
    bamFiles:
      - ENCFF009NCL.bam
      - ENCFF110EWQ.bam
      - ENCFF231YYD.bam
      - ENCFF239RGZ.bam
      - ENCFF724QHH.bam
      - ENCFF767FGV.bam
      - ENCFF801THG.bam
      - ENCFF822BKT.bam
      - ENCFF925NDY.bam
      - ENCFF978QLZ.bam


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run ATAC demo

  % consenrich --config atacDemo.yaml --verbose

Principal output files:

.. code-block:: text

  atacDemo_consenrich_state.v0.10.5a0.bw
  atacDemo_consenrich_uncertainty.v0.10.5a0.bw
  consenrichOutput_atacDemo_state.v0.10.5a0_rocco.narrowPeak

Results
"""""""

.. admonition:: Image placeholder

   IGV/browser snapshot of the ATAC-seq state estimate, local uncertainty, and
   ROCCO peaks over a representative locus.


Broad Mark ChIP-seq Demo
~~~~~~~~~~~~~~~~~~~~~~~~

This demo estimates a consensus H3K4me1 signal from ten ENCODE heart left
ventricle ChIP-seq experiments with matched input controls. The current demo
uses 100 bp intervals and a larger background length-scale multiplier, which is
a useful starting point for broad histone marks.

Download Alignments
"""""""""""""""""""

The H3K4me1 YAML uses descriptive filenames that include both the file
accession and experiment accession, so the download commands use ``wget -O`` to
write each BAM to the expected name.

.. code-block:: bash

  encodeFiles=https://www.encodeproject.org/files
  download_bam () {
      accession="$1"
      output="$2"
      wget -O "$output" "$encodeFiles/$accession/@@download/$accession.bam"
  }

  download_bam ENCFF365LTV ENCFF365LTV_ENCSR449FRQ_treatment_GRCh38.bam
  download_bam ENCFF630KZV ENCFF630KZV_ENCSR678RFS_treatment_GRCh38.bam
  download_bam ENCFF851FLQ ENCFF851FLQ_ENCSR412THE_treatment_GRCh38.bam
  download_bam ENCFF581VRR ENCFF581VRR_ENCSR208XKJ_treatment_GRCh38.bam
  download_bam ENCFF451FGF ENCFF451FGF_ENCSR724MJX_treatment_GRCh38.bam
  download_bam ENCFF660DDB ENCFF660DDB_ENCSR438QZN_treatment_GRCh38.bam
  download_bam ENCFF828GWI ENCFF828GWI_ENCSR564QBS_treatment_GRCh38.bam
  download_bam ENCFF392MXC ENCFF392MXC_ENCSR485LPA_treatment_GRCh38.bam
  download_bam ENCFF366HMH ENCFF366HMH_ENCSR817JNE_treatment_GRCh38.bam
  download_bam ENCFF671AAF ENCFF671AAF_ENCSR299NYB_treatment_GRCh38.bam

  download_bam ENCFF536MLZ ENCFF536MLZ_ENCSR178FVP_control_GRCh38.bam
  download_bam ENCFF007LNN ENCFF007LNN_ENCSR632MPN_control_GRCh38.bam
  download_bam ENCFF422MKH ENCFF422MKH_ENCSR040TRJ_control_GRCh38.bam
  download_bam ENCFF525NLT ENCFF525NLT_ENCSR979YKY_control_GRCh38.bam
  download_bam ENCFF013ION ENCFF013ION_ENCSR109IWL_control_GRCh38.bam
  download_bam ENCFF730PAF ENCFF730PAF_ENCSR526EXI_control_GRCh38.bam
  download_bam ENCFF971XWL ENCFF971XWL_ENCSR945LPX_control_GRCh38.bam
  download_bam ENCFF273EYI ENCFF273EYI_ENCSR120ITZ_control_GRCh38.bam
  download_bam ENCFF944AYC ENCFF944AYC_ENCSR061EXX_control_GRCh38.bam
  download_bam ENCFF271VZL ENCFF271VZL_ENCSR261PLD_control_GRCh38.bam

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
      - name: ENCSR449FRQ_H3K4me1
        path: ENCFF365LTV_ENCSR449FRQ_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR678RFS_H3K4me1
        path: ENCFF630KZV_ENCSR678RFS_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR412THE_H3K4me1
        path: ENCFF851FLQ_ENCSR412THE_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR208XKJ_H3K4me1
        path: ENCFF581VRR_ENCSR208XKJ_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR724MJX_H3K4me1
        path: ENCFF451FGF_ENCSR724MJX_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR438QZN_H3K4me1
        path: ENCFF660DDB_ENCSR438QZN_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR564QBS_H3K4me1
        path: ENCFF828GWI_ENCSR564QBS_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR485LPA_H3K4me1
        path: ENCFF392MXC_ENCSR485LPA_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR817JNE_H3K4me1
        path: ENCFF366HMH_ENCSR817JNE_treatment_GRCh38.bam
        format: bam
        role: treatment
      - name: ENCSR299NYB_H3K4me1
        path: ENCFF671AAF_ENCSR299NYB_treatment_GRCh38.bam
        format: bam
        role: treatment

      - name: ENCSR178FVP_input_for_ENCSR449FRQ
        path: ENCFF536MLZ_ENCSR178FVP_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR632MPN_input_for_ENCSR678RFS
        path: ENCFF007LNN_ENCSR632MPN_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR040TRJ_input_for_ENCSR412THE
        path: ENCFF422MKH_ENCSR040TRJ_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR979YKY_input_for_ENCSR208XKJ
        path: ENCFF525NLT_ENCSR979YKY_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR109IWL_input_for_ENCSR724MJX
        path: ENCFF013ION_ENCSR109IWL_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR526EXI_input_for_ENCSR438QZN
        path: ENCFF730PAF_ENCSR526EXI_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR945LPX_input_for_ENCSR564QBS
        path: ENCFF971XWL_ENCSR945LPX_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR120ITZ_input_for_ENCSR485LPA
        path: ENCFF273EYI_ENCSR120ITZ_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR061EXX_input_for_ENCSR817JNE
        path: ENCFF944AYC_ENCSR061EXX_control_GRCh38.bam
        format: bam
        role: control
      - name: ENCSR261PLD_input_for_ENCSR299NYB
        path: ENCFF271VZL_ENCSR261PLD_control_GRCh38.bam
        format: bam
        role: control

  countingParams:
    intervalSizeBP: 100

  fitParams.ECM_backgroundLengthScaleMultiplier: 32


Run Consenrich
""""""""""""""

.. code-block:: console
  :name: Run H3K4me1 demo

  % consenrich --config bigH3K4me1Demo.yaml --verbose

Principal output files:

.. code-block:: text

  bigH3K4me1Demo_consenrich_state.v0.10.5a0.bw
  bigH3K4me1Demo_consenrich_uncertainty.v0.10.5a0.bw
  consenrichOutput_bigH3K4me1Demo_state.v0.10.5a0_rocco.narrowPeak

Results
"""""""

.. admonition:: Image placeholder

   IGV/browser snapshot of the H3K4me1 state estimate, local uncertainty, and
   ROCCO peaks over a representative locus.


Configuration Suggestions
~~~~~~~~~~~~~~~~~~~~~~~~~

Consenrich offers data-driven defaults so that many analyses should not require
exhaustive tuning. If a specific experiment requires adjustments, here are some suggestions:

Want stronger (weaker) shrinkage to the smooth prior process model?

* ``processParams.stateModel``: Default is ``levelTrend``. Use ``level`` for a
  stronger smoothness assumption when a separate slope state is not supported by
  the data.
* ``processParams.minQ``: Decrease the level-noise floor to permit smoother
  state estimates. Increase it only when the fitted state is too stiff.

Want broader (narrower) signal resolution?

* ``countingParams.intervalSizeBP``: Default is 25 bp. Consider larger bins for broader signal targets or shallow sequencing depth (for example: 100-250bp for H3K4me1, H3K27me3, H3K36me3, etc.).
  Consider smaller bins (for example: 5-10 bp) to capture narrow signal targets in deeply-sequenced data in assays like ATAC-seq or TF ChIP-seq.

* ``fitParams.ECM_backgroundLengthScaleMultiplier``: Default is 16.0 which mitigates risk of signal-background conflation. Smaller values allow a more flexible background that can adapt to sharper local changes in signal.
  Note, values below 4.0 may dilute signal estimates and should be used with caution.

Want reduced computational expense?

* ``processParams.processNoiseWarmupECMIters``: Reduce this if process-noise
  diagnostics are stable across runs.
* ``fitParams.ECM_fixedBackgroundIters``: Lower this to reduce per-fit ECM work
  when convergence diagnostics remain acceptable.
* ``uncertaintyCalibrationParams.enabled``: Disable cross-fit uncertainty
  calibration when calibrated uncertainty tracks are not needed.
