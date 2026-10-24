API Reference
=============

.. _single-cell-api:

.. confval:: scParams.cellMetadata

   This mapping specifies which cells belong to each sample and cell group.

   **Type:** a dictionary specifying the file path and column names for a CSV,
   TSV, or AnnData ``.obs`` source.

   The ``path`` can be one of

   - CSV/TSV file (optionally gzipped): contains cell metadata with columns for barcodes (``barcodeColumn``), sample IDs (``sampleColumn``), and group labels (``groupColumn``)
   - `AnnData <https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html>`_
     (`.h5ad <https://anndata.readthedocs.io/en/stable/fileformat-prose.html>`_) file:
     contains cell metadata in the ``.obs`` table.

   .. list-table::
      :header-rows: 1
      :widths: 22 20 58

      * - Field
        - Type
        - Use
      * - ``path``
        - ``str``
        - Required. A ``.csv``, ``.tsv``, or ``.h5ad`` file. CSV/TSV may use ``.gz``.
      * - ``barcodeColumn``
        - ``str | None``
        - Cell identifier column. Required for CSV/TSV. **Omit** this for AnnData/H5AD so that ``.obs_names`` is used
      * - ``sampleColumn``
        - ``str | None``
        - Column used to match metadata rows to an input sample, e.g. ``library`` below. Required when the metadata file covers multiple input samples.
      * - ``groupColumn``
        - ``str | None``
        - Cell-group column, such as ``cellType``. Required for group selection.

   **Example**.

   .. list-table:: cells.csv
      :header-rows: 1

      * - barcode
        - library
        - cellType
      * - AAACGAAAGTACGCCC-1
        - PBMC_01
        - CD4_T
      * - AAACGAACAGCTACAT-1
        - PBMC_01
        - B_cell
      * - AAACGAAGTCGTTGGC-1
        - PBMC_01
        - NK_cell
      * - AAACGAAAGTACGCCC-1
        - PBMC_02
        - CD8_T
      * - AAACGAATCTTACGGA-1
        - PBMC_02
        - B_cell
      * - AAACGAAGTACAGCGA-1
        - PBMC_02
        - CD4_T

   For the ``cells.csv`` table above, specify the column names
   under ``scParams.cellMetadata`` in the Consenrich YAML:

   .. code-block:: yaml

      scParams:
        cellMetadata:
          path: cells.csv
          barcodeColumn: barcode
          sampleColumn: library
          groupColumn: cellType

   With ``sampleColumn: library``, an input named ``PBMC_01`` is matched to the
   three rows labelled ``PBMC_01``. If that input is named ``replicate1``, you would set
   ``metadataSample: PBMC_01`` on the input to keep the same match.

.. confval:: scParams.fits

   Use this to specify multiple Consenrich jobs for different cell groups within the same dataset, *in one YAML file*.
   See, for example, the :ref:`scATAC-seq example <scatac-demo>`.

   **Type:** ``list[dict[str, str | list[str]]] | None``.

   **Default:** ``None`` (one run without a fit list).

   Each entry requires ``name`` and ``selectGroups``:

   * ``name`` (``str``) names the output directory, e.g. ``TCells``. Must be unique
   * ``selectGroups`` (``list[str]``) holds one or more unique labels from
     ``groupColumn``, matched exactly

   .. code-block:: yaml

      # ...other consenrich config...


      scParams:
        fits:
          - {name: TCells, selectGroups: [CD4_T, CD8_T]} # Run a fit for T cells
          - {name: BCells, selectGroups: [B_cell]}       # and run a fit for B cells, too

.. confval:: inputParams.samples

   Input files and source-specific settings.

   **Type:** ``list[str | dict[str, Any]] | None``.
   **Default:** ``None``.

   Supply a nonempty list. H5AD counts require a mapping with ``matrixRows``.

   .. list-table::
      :header-rows: 1
      :widths: 24 28 48

      * - Field
        - Type
        - Use / default
      * - ``path``
        - ``str``
        - Required. E.g. ``library1.fragments.tsv.gz`` or ``library1.h5ad``.
      * - ``name``
        - ``str | None``
        - Output label and default metadata sample identifier. Default: ``None``.
      * - ``cellMetadata``
        - ``dict[str, str | None] | None``
        - Replaces :confval:`scParams.cellMetadata` for that source. Default: ``None`` uses the shared setting.
      * - ``metadataSample``
        - ``str | None``
        - Value to select from ``sampleColumn``, e.g. ``library1``. Default: ``None`` uses ``name``.
      * - ``matrixRows``
        - ``str | None``
        - Required for H5AD counts. ``cells`` sums selected rows. ``pseudobulks`` selects exactly one row per source and fit. Default: ``None``.
      * - ``countMode``
        - ``str | None``
        - Fragment counting override. Default: ``None`` uses ``scParams.defaultCountMode``. Rejected for H5AD counts.
      * - ``fragmentsUseReadSupport``
        - ``bool``
        - ``True`` weights fragment rows by column five. Inherits ``scParams.fragmentsUseReadSupport``, which defaults to ``False``.

   Fragment ``countMode`` choices: ``coverage``, ``cutsite``, ``fiveprime``,
   ``center``, ``midpoint``, or ``conservedFractionalOverlap``.
   Use ``center`` to count fragment centers.

H5AD count requirements
~~~~~~~~~~~~~~~~~~~~~~~

* ``.X`` contains raw, nonnegative integer counts. Dense, CSR, and CSC storage
  are supported, including integer-valued floats.
* ``.var`` contains ``chrom``, ``start``, and ``end`` for complete, regular
  chromosome grids in 0-based, half-open coordinates. Terminal bins may be shorter.
* ``countingParams.intervalSizeBP`` is inferred when omitted. All H5AD inputs
  must share that width, and an explicit value must match. Fragment inputs in
  the same run use that width.

Normalization uses ``countingParams.normMethod``. H5AD supports ``CPM``, ``RPKM``,
``SF``, or supplied scale factors. ``EGS``/``RPGC`` require supplied factors.
Fragment barcode-file options on H5AD inputs are ignored with a warning.

See the :ref:`single-cell examples <scatac-demo>` for complete configurations.
