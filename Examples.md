Functional Genomic Analysis of Multi-Sample HTS Datasets with Consenrich
================
Nolan H. Hamilton
([`@nolan-h-hamilton`](https://github.com/nolan-h-hamilton))

March 2025

------------------------------------------------------------------------

[Consenrich](https://github.com/nolan-h-hamilton/Consenrich) is a state
estimation algorithm for integrating multiple high-throughput sequencing
(HTS) datasets to extract a ‘consensus signal tracks’ that represent
consistent, underlying genomic signal while attenuating
sample-*and*-region-specific noise and preserving informative
spectral/spatial content with high resolution (Hamilton et al. 2025).

We demonstrate its applications to multi-sample ATAC-seq and ChIP-seq
datasets, joint analysis of data from related assays, and differential
analysis between disease conditions.

------------------------------------------------------------------------

# Table of Contents

- [**Genome-Wide Signal Estimation in Heterogeneous HTS Datasets**:
  *Robust, Scalable Extraction of True States in Noisy ATAC-seq
  Data*](#genome-wide-chromatin-state-estimation-in-heterogeneous-atac-seq-datasets)
- [**Multi-Sample ChIP-Seq Analysis**: *Building Encompassing Profiles
  of TF Binding Activity*](#multi-sample-chip-seq-tf-analysis)
- [**Joint Analysis of HTS Signals from Related Assays**: *Fusion of
  DNase-seq and ATAC-seq Data
  samples*](#mixed-assay--multi-sample-analysis-dnase-seq--atac-seq)
- [**Applications to Differential Analyses**: *Distinct Epigenomic
  Features of Alzheimer’s Disease Recovered in
  DNase-Seq*](#differential-analysis-between-disease-conditions)

# Environment Details

The following is a summary of the computing environment used to generate
the results in this document.

Note, consumer-grade hardware is employed for all analyses in this
presentation.

### Hardware

- Excerpt from `system_profiler SPHardwareDataType`:

| Metric                | Value        |
|-----------------------|--------------|
| Model Name            | MacBook Pro  |
| Model Identifier      | Mac16,5      |
| Model Number          | MX313LL/A    |
| Chip                  | Apple M4 Max |
| Total Number of Cores | 16           |
| Memory                | 48 GB        |

- Note, systems with limited memory ($<16\,\text{Gb}$) may experience
  longer runtimes due to increased disk IO/swapping.

### Software

*Note*, recent versions of Python (3.9+) and the other listed software
will likely work, too–The specific versions given in this section are
exact for the sake of reproducibility.

#### System Dependencies

| Package                                                | Version |
|--------------------------------------------------------|---------|
| [samtools](http://www.htslib.org/download/)            | 1.21    |
| [bedtools](https://bedtools.readthedocs.io/en/latest/) | 2.31.1  |

These popular software can be installed directly using common package
managers, e.g., `<brew, apt, etc.> install samtools bedtools` or
[Conda/BioConda](https://bioconda.github.io/). They are also easily
built from source provided by the authors at the links given above.

#### Python Environment

- Python version: `3.12.9`

- Running the following two lines should install all necessary Python
  dependencies for the experiments in this document:

  - `python -m pip install consenrich`
  - `python -m pip install rocco`

Version-specific packages can be installed using
`python -m pip install <package>==<version>`. ROCCO is not required to
run Consenrich but is used for peak calling in the presented usage
examples.

For debugging/reproducibility, exact versions of the Python packages
used to produce results are given below:

| Package                                                      | Version   |
|--------------------------------------------------------------|-----------|
| numpy                                                        | 2.2.3     |
| scipy                                                        | 1.15.2    |
| pandas                                                       | 2.2.3     |
| ortools                                                      | 9.12.4544 |
| deeptools                                                    | 3.5.6     |
| pybigwig                                                     | 0.3.24    |
| pybedtools                                                   | 0.11.0    |
| pysam                                                        | 0.23.0    |
| [rocco](https://github.com/nolan-h-hamilton/ROCCO)           | 1.6.1     |
| [consenrich](https://github.com/nolan-h-hamilton/Consenrich) | 0.1.1b0   |

## Data Availability

Every dataset used in this document is publicly available and can be
accessed from the [ENCODE](https://www.encodeproject.org/) project.
Files are named according to their ENCODE accession numbers.

# Genome-Wide Signal Estimation in Heterogeneous HTS Datasets

- The aim of these experiments is to demonstrate both robust and
  encompassing signal identification and quantification from
  multi-sample, noisy ATAC-seq data.

- A set of heterogeneous, publicly available ATAC-seq alignments from
  human lymphoblastoid cell lines is used for evaluation, combined with
  sets of intentionally noisy samples generated with the procedure
  discussed in the [Appendix](#appendix-noisy-data-generation).

  - These noisy alignments mimic low-quality samples and are used to
    validate Consenrich’s ability to extract common, ‘true’ signals
    despite heterogeneous, noisy data samples.
  - Ideally, Consenrich’s output should be unaffected by the presence of
    these noisy samples – a property that is evaluated in the results
    below.

## Het10 Lymphoblastoid Dataset

- $m=10$ independent human lymphoblast cell lines from ENCODE.
  - These ‘Het10’ samples were selected as the ‘best’ and ‘worst’ five
    alignments from a [complete set of
    fifty-six](het10/ENCODE_ATAC_LymphoblastCellLines.tsv) according to
    the TSS enrichment score.

| ID | Datasets | TSS | Reads (flag -f 64) | Frac_Prom_Enh |
|----|----|----|----|----|
| [ENCFF632MBC](https://www.encodeproject.org/files/ENCFF632MBC/) | Worst5,Het10 | 4.954 | 12592755 | 0.369458 |
| [ENCFF919PWF](https://www.encodeproject.org/files/ENCFF919PWF/) | Worst5,Het10 | 5.196 | 38896396 | 0.395363 |
| [ENCFF497QOS](https://www.encodeproject.org/files/ENCFF497QOS/) | Worst5,Het10 | 5.203 | 54287803 | 0.371295 |
| [ENCFF767FGV](https://www.encodeproject.org/files/ENCFF767FGV/) | Worst5,Het10 | 5.311 | 43453140 | 0.415719 |
| [ENCFF326QXM](https://www.encodeproject.org/files/ENCFF326QXM/) | Worst5,Het10 | 5.532 | 41055720 | 0.395096 |
| [ENCFF495DQP](https://www.encodeproject.org/files/ENCFF495DQP/) | Best5,Het10 | 7.744 | 29734258 | 0.404184 |
| [ENCFF949CVL](https://www.encodeproject.org/files/ENCFF949CVL/) | Best5,Het10 | 7.766 | 30636528 | 0.442786 |
| [ENCFF687QML](https://www.encodeproject.org/files/ENCFF687QML/) | Best5,Het10 | 7.814 | 51948134 | 0.447943 |
| [ENCFF447ZRG](https://www.encodeproject.org/files/ENCFF447ZRG/) | Best5,Het10 | 7.944 | 52920682 | 0.421713 |
| [ENCFF462RHM](https://www.encodeproject.org/files/ENCFF462RHM/) | Best5,Het10 | 8.919 | 24810004 | 0.463227 |

### Het10 $\cup$ Noisy Datasets

Details regarding the generation of ‘noisy’ samples is discussed in the
[Appendix](#appendix-noisy-data-generation). The ‘Het10 $\cup$ Noisy5’
and ‘Het10 $\cup$ Noisy10’ datasets are comprised of the above
legitimate ATAC-seq BAM alignments and $m=5$ and $m=10$ noisy low-SNR
alignments, respectively.

## Running Consenrich

- Without noisy samples

  ``` bash
  consenrich \
  --bam_files \
    ENCFF632MBC.bam \
    ENCFF919PWF.bam \
    ENCFF497QOS.bam \
    ENCFF767FGV.bam \
    ENCFF326QXM.bam \
    ENCFF495DQP.bam \
    ENCFF949CVL.bam \
    ENCFF687QML.bam \
    ENCFF447ZRG.bam \
    ENCFF462RHM.bam \
  -g hg38 \
  -o het10.tsv \
  --signal_bigwig consenrich_het10_signal_v0.1.1.b0.bw \
  --eratio consenrich_het10_eratio_v0.1.1.b0.bw \
  --residuals consenrich_het10_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
  ```

- With 5 noisy samples (Het10 $\cup$ Noisy5)

  ``` bash
  consenrich \
  --bam_files \
    ENCFF632MBC.bam \
    ENCFF919PWF.bam \
    ENCFF497QOS.bam \
    ENCFF767FGV.bam \
    ENCFF326QXM.bam \
    ENCFF495DQP.bam \
    ENCFF949CVL.bam \
    ENCFF687QML.bam \
    ENCFF447ZRG.bam \
    ENCFF462RHM.bam \
    noisy/noisy1.sorted.bam \
    noisy/noisy2.sorted.bam \
    noisy/noisy3.sorted.bam \
    noisy/noisy4.sorted.bam \
    noisy/noisy5.sorted.bam \
  -g hg38 \
  -o het10_noisy.tsv \
  --signal_bigwig consenrich_het10_noisy_signal_v0.1.1.b0.bw \
  --eratio consenrich_het10_noisy_eratio_v0.1.1.b0.bw \
  --residuals consenrich_het10_noisy_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
  ```

- With 10 noisy samples (Het10 $\cup$ Noisy10)

  ``` bash
  consenrich \
  --bam_files \
    ENCFF632MBC.bam \
    ENCFF919PWF.bam \
    ENCFF497QOS.bam \
    ENCFF767FGV.bam \
    ENCFF326QXM.bam \
    ENCFF495DQP.bam \
    ENCFF949CVL.bam \
    ENCFF687QML.bam \
    ENCFF447ZRG.bam \
    ENCFF462RHM.bam \
    noisy/noisy1.sorted.bam \
    noisy/noisy2.sorted.bam \
    noisy/noisy3.sorted.bam \
    noisy/noisy4.sorted.bam \
    noisy/noisy5.sorted.bam \
    noisy/noisy6.sorted.bam \
    noisy/noisy7.sorted.bam \
    noisy/noisy8.sorted.bam \
    noisy/noisy9.sorted.bam \
    noisy/noisy10.sorted.bam \
  -g hg38 \
  -o het10_noisy10.tsv \
  --signal_bigwig consenrich_het10_noisy10_signal_v0.1.1.b0.bw \
  --eratio consenrich_het10_noisy10_eratio_v0.1.1.b0.bw \
  --residuals consenrich_het10_noisy10_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
  ```

  - Note, run with `gtime -v consenrich --bam_files ...` to obtain
    memory usage and runtime stats.
  - Wildcards are supported, e.g., replace above with
    `consenrich --bam_files *.bam ...`

### Runtime/Memory Stats

| Metric                                      | Value    |
|---------------------------------------------|----------|
| Elapsed (wall clock) time (h:mm:ss or m:ss) | 1:02:14  |
| Maximum resident set size (kbytes)          | 22420352 |
| Swaps                                       | 0        |

## Results

We compute per-chromosome estimates of the root mean square deviation
(RMSD) and Spearman correlations between Consenrich signal estimates
given input datasets with multiple levels of heterogeneity and noise.

1.  To establish a reference point, we first run Consenrich on the
    `Het10` dataset containing 10 legitimate ATAC-seq alignments from
    lymphoblastoid cell lines (see [Het10 Lymphoblastoid
    Dataset](#het10-lymphoblastoid-dataset)). We refer to the
    corresponding signal track output as ‘Consenrich(Het10)’.

2.  The ‘Consenrich(Het10 $\cup$ Noisy5)’ and ‘Consenrich(Het10 $\cup$
    Noisy10)’ signal tracks are then obtained by running Consenrich with
    $m=5$ and $m=10$ intentionally noisy samples augmented to `Het10`
    (see [Appendix](#appendix-noisy-data-generation)).

For benchmarking, we conduct the same analyses for **Pointwise Trimmed
Mean (25%)**, which ‘trims’ samples’ alignment counts at each genomic
interval below/above the lower and upper quartiles, respectively, before
calculating the mean of remaining values in the IQR. **Pointwise
Median** is likewise included as an additional robust, positional
aggregate quantification approach for comparison.

### Influence of Noisy Samples

<figure>
<img src="docs/het10_with_without_noisy.png"
alt="Chromosome-specific RMSD between signal estimates using the ‘Het10’ dataset and ‘Het10 with Noisy5’ and ‘Het10 with Noisy10’ datasets. Results are included for Consenrich (first row) and two benchmark methods (second, third rows)." />
<figcaption aria-hidden="true">Chromosome-specific RMSD between signal
estimates using the ‘Het10’ dataset and ‘Het10 with Noisy5’ and ‘Het10
with Noisy10’ datasets. Results are included for Consenrich (first row)
and two benchmark methods (second, third rows).</figcaption>
</figure>

### Spearman Correlation Coefficients

<figure>
<img src="docs/het10_spearman.png"
alt="Chromosome-specific Spearman correlation coefficients between Consenrich(Het10) and Consenrich(Het10 \cup Noisy&lt;5,10&gt;) signal tracks. High correlations (near 1.0 for most chromosomes) suggest that noisy samples have minimal effect on the consensus signal profile. Several pointwise data aggregration strategies are included for reference." />
<figcaption aria-hidden="true"><em>Chromosome-specific Spearman
correlation coefficients between Consenrich(Het10) and Consenrich(Het10
<span class="math inline">∪</span> Noisy&lt;5,10&gt;) signal
tracks</em>. High correlations (near 1.0 for most chromosomes) suggest
that noisy samples have minimal effect on the consensus signal profile.
Several pointwise data aggregration strategies are included for
reference.</figcaption>
</figure>

## IGV Snapshot

<figure>
<img src="docs/het10_03032025.png"
alt="Consenrich signal track outputs (middle panel, blue) given the ‘Het10’, ‘Het10 \cup Noisy5’, and ‘Het10 \cup Noisy10’ datasets as input. The alignment coverage tracks computed by IGV are included for reference. LYL1 was chosen as it is highly expressed in lymphoblasts. This visual comparison shows that Consenrich produces consistent signal estimates even as increasing numbers of noisy inputs are added" />
<figcaption aria-hidden="true">Consenrich signal track outputs (middle
panel, blue) given the ‘Het10’, ‘Het10 <span
class="math inline">∪</span> Noisy5’, and ‘Het10 <span
class="math inline">∪</span> Noisy10’ datasets as input. The alignment
coverage tracks computed by IGV are included for reference. LYL1 was
chosen as it is highly expressed in lymphoblasts. This visual comparison
shows that Consenrich produces consistent signal estimates even as
increasing numbers of noisy inputs are added</figcaption>
</figure>

# Multi-Sample ChIP-Seq Analysis

The aim of these experiments is to evaluate Consenrich’s ability to
extract common, relevant signals from multiple ChIP-seq experiments with
paired controls and to demonstrate its utility in multi-sample peak
calling workflows.

- We show utility of Consenrich signal tracks for multi-sample/consensus
  peak calling using [ROCCO](https://github.com/nolan-h-hamilton/ROCCO)
  (Hamilton and Furey 2023). But other peak calling methods can also
  benefit from the improved capture of multi-sample signal conveyed in
  Consenrich’s output.

- A functional enrichment analysis on the returned peaks is performed to
  evaluate relevant biological patterns. The BioConductor package,
  [ChIPseeker](https://bioconductor.org/packages/release/bioc/html/ChIPseeker.html)
  (Yu, Wang, and He 2015), is used for this task.

## Colon Tissue-POL2RA Transcription Factor ChIP-Seq Dataset

- $N=6$ ChIP-Seq experiments targeting TF *POL2RA* (with
  paired-controls) in independent donors’ colon tissue samples.

| Experiment  | Tissue           | Age | Gender |
|-------------|------------------|-----|--------|
| ENCSR322JEO | sigmoid colon    | 53  | female |
| ENCSR472VBD | sigmoid colon    | 54  | male   |
| ENCSR431EHE | sigmoid colon    | 37  | male   |
| ENCSR724FCJ | sigmoid colon    | 51  | female |
| ENCSR974HQI | transverse colon | 51  | female |
| ENCSR132XRW | transverse colon | 54  | male   |

## Running Consenrich with Control Inputs

By default, Consenrich integrate control inputs by first scaling
treatment/control data in each pair to each other
($\text{larger} \rightarrow \text{smaller}$), removing background
modeled from the control input, and finally scaling all such
background-corrected treatment signals so that across-sample comparisons
are minimally affected by differences in sequencing depth.

- To incorporate controls, run Consenrich using both `--bam_files` and
  `--control_files` arguments (ChIP and control BAMs, respectively).

``` bash
consenrich --bam_files ENCSR322JEO_POL2RA.bam \
ENCSR472VBD_POL2RA.bam \
ENCSR431EHE_POL2RA.bam \
ENCSR724FCJ_POL2RA.bam \
ENCSR974HQI_POL2RA.bam \
ENCSR132XRW_POL2RA.bam \
--control_files ENCSR322JEO_CTRL.bam \
ENCSR472VBD_CTRL.bam \
ENCSR431EHE_CTRL.bam \
ENCSR724FCJ_CTRL.bam \
ENCSR974HQI_CTRL.bam \
ENCSR132XRW_CTRL.bam \
-g hg38 \
-o Consenrich_POL2RA.tsv \
--signal_bigwig consenrich_pol2_chip_signal_v0.1.1.b0.bw \
--eratio consenrich_pol2_chip_eratio_v0.1.1.b0.bw \
--residuals consenrich_pol2_chip_residuals_v0.1.1.b0.bw \
--save_args \
--skip_chroms chrY \
--threads 8 \
-p 8
```

### Runtime/Memory Stats

| Metric                                      | Value    |
|---------------------------------------------|----------|
| Elapsed (wall clock) time (h:mm:ss or m:ss) | 58:43.43 |
| Maximum resident set size (kbytes)          | 22427344 |
| Swaps                                       | 0        |

## Running ROCCO

We feed the results of Consenrich into ROCCO to obtain
narrowPeak-formatted output.

``` bash
rocco -i consenrich_pol2_chip_signal_v0.1.1.b0.bw \
-g hg38 \
--skip_chroms chrY \
--narrowPeak \
--bamlist_txt pol2_chip_bamlist.txt \
--verbose \
-o rocco_consenrich_pol2_chip_signal_v1.6.1.bed \
--ecdf_samples 1000 \
--ecdf_proc 8 \
--threads 8
```

- Note, since we’ve supplied the Consenrich signal BigWig track as input
  to ROCCO, to obtain full narrowPeak-formatted output, we must supply a
  list of BAM files used to generate the
  `--bamlist_txt pol2_chip_bamlist.txt` file:

      ENCSR132XRW_CTRL.bam
      ENCSR132XRW_POL2RA.bam
      ENCSR322JEO_CTRL.bam
      ENCSR322JEO_POL2RA.bam
      ENCSR431EHE_CTRL.bam
      ENCSR431EHE_POL2RA.bam
      ENCSR472VBD_CTRL.bam
      ENCSR472VBD_POL2RA.bam
      ENCSR724FCJ_CTRL.bam
      ENCSR724FCJ_POL2RA.bam
      ENCSR974HQI_CTRL.bam
      ENCSR974HQI_POL2RA.bam

If we’d used BAM files as input to ROCCO, we could have simply invoked
`--narrowPeak` without the need for `--bamlist_txt`.

## Results

### Functional Annotation of Identified Peaks

Many tools for functional annotation of peak results are available.
These can be used to evaluate the biological relevance of the peaks
regions returned by ROCCO.

- A filter on the `pValue` (Column 8) or `qValue` (Column 9) columns of
  ROCCO’s narrowPeak output can easily be applied for further validation
  of enrichment beyond satisfying ROCCO’s optimality criterion.

  ``` bash
  cat rocco_consenrich_pol2_chip_signal_v1.6.1.narrowPeak | \
  awk '$9 > 2' > \
  rocco_consenrich_pol2_chip_signal_v1.6.1.q01.narrowPeak
  ```

  We can then run the following R code and obtain genomic feature
  frequencies in the table below.

``` r
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(clusterProfiler)
library(org.Hs.eg.db)

txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
peaks <- readPeakFile("rocco_consenrich_pol2_chip_signal_v1.6.1.q01.narrowPeak")
peakAnno <- annotatePeak(peaks, tssRegion=c(-1000, 1000),
                         TxDb=txdb, annoDb="org.Hs.eg.db")
peakAnno@annoStat
```

| Feature    | Frequency |
|------------|-----------|
| *Promoter* | **92.53** |
| 5’ UTR     | 1.49      |
| 1st Exon   | 4.47      |
| 1st Intron | 1.49      |

*Percent of Consenrich –\> ROCCO peaks classified as the elements given
in the left column. The vast majority of peaks are annotated as
promoters which is expected given the target of this ChIP-Seq analysis
(POL2RA).*

# Joint Analysis of HTS Signals from Related Assays

We next evaluate Consenrich’s performance on a multi-sample, multi-assay
dataset, combining ATAC-seq and DNase-seq experiments. We aim to extract
true open-chromatin signals from heart tissue samples where data come
from two assay types. Adding further heterogeneity, the ATAC-seq and
DNase-seq data differ in read length and pairing: the ATAC-seq libraries
are 101 bp paired-end reads, whereas the DNase-seq libraries are 36 bp
single-end reads.

## Heart Tissue DNase-seq and ATAC-seq Dataset

$m=4$ *unique* donors in total from
[ENCODE](https://www.encodeproject.org):

- Two DNase-seq heart tissue donors
- Two ATAC-seq heart tissue donors

| ID | Gender | Tissue | Age | Assay | Read-length | PE/SE |
|:--:|:---|:---|:---|:---|:---|:---|
| [ENCFF709NIR](https://www.encodeproject.org/files/ENCFF709NIR/) | male | heart | 43 | ATAC-seq | 101 | Paired |
| [ENCFF804KIW](https://www.encodeproject.org/files/ENCFF804KIW/) | male | heart | 43 | ATAC-seq | 101 | Paired |
| [ENCFF933GFX](https://www.encodeproject.org/files/ENCFF933GFX/) | male | heart | 27 | DNase-seq | 36 | Single |
| [ENCFF916MAS](https://www.encodeproject.org/files/ENCFF916MAS/) | male | heart | 35 | DNase-seq | 36 | Single |

- Paired-end data from the ATAC-seq experiments was filtered to retain
  only one read-per-pair, e.g.,

  ``` bash
  samtools view -b -f 64 ATAC_ENCFF804NIR.bam > ATAC_ENCFF804NIR_f64.bam
  samtools view -b -f 64 ATAC_ENCFF804KIW.bam > ATAC_ENCFF804KIW_f64.bam
  ```

## Running Consenrich in `--single_end` mode

``` bash
consenrich \
--bam_files \
ATAC_ENCFF709NIR_f64.bam \
ATAC_ENCFF804KIW_f64.bam \
DNASE_ENCFF933GFX.bam \
DNASE_ENCFF916MAS.bam \
--single_end \
-g hg38 \
-o dnase_atac.tsv \
--signal_bigwig consenrich_dnase_atac_signal_v0.1.1.b0.bw \
--eratio consenrich_dnase_atac_eratio_v0.1.1.b0.bw \
--residuals consenrich_dnase_atac_residuals_v0.1.1.b0.bw \
--save_args --skip_chroms chrY --threads 8 -p 8
```

- Note Consenrich was run in `--single_end` mode to accomodate the
  single-end DNase-seq data.

### Runtime/Memory Stats

| Metric                                      | Value    |
|---------------------------------------------|----------|
| Elapsed (wall clock) time (h:mm:ss or m:ss) | 49:33.20 |
| Maximum resident set size (kbytes)          | 22451344 |
| Swaps                                       | 0        |

## Running ROCCO

``` bash
rocco \
-i consenrich_dnase_atac_signal_v0.1.1.b0.bw \
-g hg38 \
--skip_chroms chrY \
--narrowPeak \
--bamlist_txt dnase_atac_bamfiles.txt \
--verbose \
-o rocco_consenrich_dnase_atac_signal_v1.6.1.bed \
--ecdf_samples 1000 \
--ecdf_proc 8 --threads 8
```

- `--bamlist_txt dnase_atac_bamfiles.txt` is necessary to obtain
  narrowPeak-formatted output from ROCCO given BigWig input
  - Its contents are the BAM files used as input to Consenrich:

  ``` bash
    ATAC_ENCFF709NIR_f64.bam
    ATAC_ENCFF804KIW_f64.bam
    DNASE_ENCFF916MAS.bam
    DNASE_ENCFF933GFX.bam
  ```

## IGV Snapshot

<figure>
<img src="docs/dnase_atac_03032025.png"
alt="Joint Signal Extraction from DNase-seq and ATAC-seq data samples with Consenrich." />
<figcaption aria-hidden="true"><em>Joint Signal Extraction from
DNase-seq and ATAC-seq data samples with Consenrich</em>.</figcaption>
</figure>

# Applications to Differential Analyses

The following analysis uses ENCODE DNase-seq alignment input from
Alzheimer’s Disease (AD) and non-AD patients to demonstrate Consenrich’s
utility in differential analysis workflows.

Note that the protocol applied here for DNase-seq is also applicable to
ATAC-seq data.

## Alzheimer’s Disease DNase-seq Dataset

| File | Diagnosis | Age | Mapped rlen | Additional Info |
|----|----|----|----|----|
| ENCFF502IOV.bam | No_AD | 90+ years | 76 | prefrontal cortex, female adult |
| ENCFF306XXE.bam | No_AD | 84 years | 76 | prefrontal cortex, female adult |
| ENCFF990EJE.bam | No_AD | 82 years | 36 | prefrontal cortex, female adult |
| ENCFF195KCN.bam | No_AD | 83 years | 76 | prefrontal cortex, female adult |
| ENCFF096RBN.bam | No_AD | 90+ years | 101 | prefrontal cortex, female adult |
| ENCFF375NXF.bam | AD | 87 years | 76 | prefrontal cortex, female adult |
| ENCFF446JDN.bam | AD | 90+ years | 76 | prefrontal cortex, female adult |
| ENCFF617VLY.bam | AD | 89 years | 36 | prefrontal cortex, female adult |
| ENCFF669HSH.bam | AD | 90+ years | 36 | prefrontal cortex, female adult |
| ENCFF647TPO.bam | AD | 89 years | 76 | prefrontal cortex, female adult |

## Using Consenrich in Differential Analysis Workflows

Several upstream strategies can be employed to extract
consensus-enriched signal regions in multi-sample HTS data for
subsequent downstream differential analyses.

Some discretion is involved in choosing the appropriate strategy, as the
number of samples, their balance between condition groups, etc. can
affect the optimal approach. We briefly discuss two options below.

### Extract signal from all samples (AD and No_AD)

- In this demonstrative analysis, we opt to evaluate enrichment across
  all sample regardless of condition. This is considered a conservative
  approach to avoid ‘data snooping’ prior to testing for differential
  accessibility.

``` bash
consenrich \
  --bam_files \
    ENCFF502IOV.bam ENCFF306XXE.bam \
    ENCFF990EJE.bam ENCFF195KCN.bam \
    ENCFF096RBN.bam ENCFF375NXF.bam \
    ENCFF446JDN.bam ENCFF617VLY.bam \
    ENCFF669HSH.bam ENCFF647TPO.bam \
  -g hg38 \
  -o dnase_ad.tsv \
  --signal_bigwig consenrich_dnase_combined_signal_v0.1.1.b0.bw \
  --eratio consenrich_dnase_combined_eratio_v0.1.1.b0.bw \
  --residuals consenrich_dnase_combined_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
```

### Runtime/Memory Stats

| Metric                                      | Value    |
|---------------------------------------------|----------|
| Elapsed (wall clock) time (h:mm:ss or m:ss) | 1:10:10  |
| Maximum resident set size (kbytes)          | 21385152 |
| Swaps                                       | 0        |

### (Optional) Extract condition-specific signals/peaks (One for each: AD, No_AD) $\rightarrow$ Merge

- This approach may be useful if the number of samples in each condition
  is imbalanced, or if the user wishes to apply different peak calling
  thresholds to each condition. Be careful to consider potentially
  inflated false discovery rates if applying this approach.

  - Particularly for smaller datasets where condition-specific peaks may
    arise from irrelevant technical or biological confounders in the
    data, rather than a reproducible population-level phenomenon.

- Alzheimer’s Disease (AD)

``` bash
consenrich \
  --bam_files \
    ENCFF375NXF.bam \
    ENCFF446JDN.bam \
    ENCFF617VLY.bam \
    ENCFF669HSH.bam \
    ENCFF647TPO.bam \
  -g hg38 \
  -o dnase_ad.tsv \
  --signal_bigwig consenrich_dnase_ad_signal_v0.1.1.b0.bw \
  --eratio consenrich_dnase_ad_eratio_v0.1.1.b0.bw \
  --residuals consenrich_dnase_ad_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
```

- No Alzheimer’s Disease (No_AD)

``` bash
consenrich \
  --bam_files \
    ENCFF502IOV.bam \ 
    ENCFF306XXE.bam \
    ENCFF990EJE.bam \    
    ENCFF195KCN.bam \    
    ENCFF096RBN.bam \
  -g hg38 \
  -o dnase_no_ad.tsv \
  --signal_bigwig consenrich_dnase_no_ad_signal_v0.1.1.b0.bw \
  --eratio consenrich_dnase_no_ad_eratio_v0.1.1.b0.bw \
  --residuals consenrich_dnase_no_ad_residuals_v0.1.1.b0.bw \
  --save_args --skip_chroms chrY --threads 8 -p 8
```

## Identifying Candidate Differential Regions

- We use ROCCO to determine genomic regions that are enriched in the
  Consenrich signal track output to identify candidate regions for
  [differential
  analysis](#evaluating-candidate-peak-regions-for-differential-accessibility).

- The R code used to perform the differential anlysis is available with
  commentary in [docs/dnase_ad.pdf](dnase_ad/dnase_ad.pdf).

``` bash
rocco \
-i consenrich_dnase_combined_signal_v0.1.1.b0.bw \
-g hg38 \
--skip_chroms chrY \
--narrowPeak \
--bamlist_txt dnase_ad_bamlist.txt \
--verbose \
-o rocco_consenrich_dnase_ad_signal_v1.6.1.bed \
--ecdf_samples 1000 \
--ecdf_proc 8 --threads 8
```

The contents of `dnase_ad_bamlist.txt` are the BAM files used as input
to Consenrich:

    ENCFF502IOV.bam
    ENCFF306XXE.bam
    ENCFF990EJE.bam
    ENCFF195KCN.bam
    ENCFF096RBN.bam
    ENCFF375NXF.bam
    ENCFF446JDN.bam
    ENCFF617VLY.bam
    ENCFF669HSH.bam
    ENCFF647TPO.bam

- If further validation of consensus peaks (in addition to ROCCO’s
  optimality criterion) is desired, an explicit filter on the `pValue`
  (Column 8) or `qValue` (Column 9) columns in the output narrowPeak
  file can be applied:

  ``` bash
  cat rocco_consenrich_dnase_ad_signal_v1.6.1.narrowPeak | \
  awk '$8 > 2' > \
  rocco_consenrich_dnase_ad_signal_v1.6.1.p01.narrowPeak
  ```

  - Per convention, the $p,q$ values are in $-\log_{10}$ scale in the
    narrowPeak file, e.g., a $p$-value of 0.01 would be scored as 2 in
    the $8^{\text{th}}$ column.

## Evaluating Candidate Peak Regions for Differential Accessibility

After obtaining the candidate differential peak regions with Consenrich
and ROCCO, we use
[RUVseq](https://bioconductor.org/packages/release/bioc/html/RUVSeq.html)
(Risso et al. 2014) and
[DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html)
(Love, Huber, and Anders 2014) to test for differential accessibility
between the `AD` and `No_AD` conditions while accounting for known and
unknown confounders.

Negative/empirical control regions for RUVseq’s `RUVg()` were selected
per guidance in [*RNA-seq workflow: gene-level exploratory analysis and
differential expression - Section
8.2*](https://bioconductor.org/packages/release/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html)

We include a snippet of the R code used to perform this analysis below,
but please refer to [docs/dnase_ad.pdf](docs/dnase_ad.pdf) for the full
workflow and results.

``` r
[...] 
# See docs/dnase_ad.pdf for full code

# Design formula includes known covariates (age) and two RUV factors (LSVs)
# use to model unknown, impertinent sources of variation.

design_formula <- as.formula(
  paste0("~ age +", paste0(w_terms, collapse = "+"),
         "+ status")
)

dds_final <- DESeqDataSetFromMatrix(
  countData = cts,
  colData = coldata,
  design = design_formula
)
```

See track
`Consenrich --> ROCCO --> DESeq2 (padj < .05, |LFC| > 0.25) : ~ age + RUV_W1 + RUV_W2 + AD_status`
in [the IGV snapshot below](#dacs-near-arc) for a visual representation
of genomic features in proximity to differentially accessible regions.

## Results

### Volcano Plot of Differentially Accessible Regions

We use the R/BioConductor package
[EnhancedVolcano](https://bioconductor.org/packages/release/bioc/html/EnhancedVolcano.html)
to visualize significant differential regions with respect to both
enrichment fold-change ($\log_2\text{fc}$) and statistical significance
(FDR-corrected $p$-values).

<figure>
<img src="docs/dnase_ad_DESeq_results_k2_volcano.png"
alt="Regions with DESeq2 FDR-corrected p-values and \log_2\text{fc} are depicted in red. A total of \mathbf{132{,}860} consensus accessible peaks were identified using Consenrich \rightarrow ROCCO. Of these, \mathbf{1183} were found to be differentially accessible using RUVseq and DESeq2." />
<figcaption aria-hidden="true"><em>Regions with DESeq2 FDR-corrected
<span class="math inline"><em>p</em></span>-values and <span
class="math inline">log<sub>2</sub>fc</span> are depicted in red. A
total of <span
class="math inline"><strong>132</strong><strong>,</strong> <strong>860</strong></span>
consensus accessible peaks were identified using Consenrich <span
class="math inline">→</span> ROCCO. Of these, <span
class="math inline"><strong>1183</strong></span> were found to be
differentially accessible using RUVseq and DESeq2.</em></figcaption>
</figure>

### GREAT: Enriched Gene Associations

- Supplying the differentially accessible regions (DARs) in BED format,
  [GREAT](http://great.stanford.edu/public/html/) (McLean et al. 2010)
  (Tanigawa, Dyer, and Bejerano 2022,) returns ARC as the most
  significantly enriched gene-region association, with a binomial
  FDR-corrected $q$-value of $7.619 \times 10^{-4}$.

- ARC is particularly well-established as a driver of synaptic function
  (Epstein and Finkbeiner 2018).

- Recent studies suggest significant potential of `ARC` as a therapeutic
  target for Alzheimer’s Disease (VanDongen 2025).

- Several additional Alzheimer’s-relevant genes were found to be
  associated with the obtained set of differential regions, including
  `ADAMTS4`, `APOE`, `TOMM40`, `BIN1`, `LRP4`, etc.

  - Notably, several identified DARs directly overlap the transcription
    start sites of these genes (e.g., `ADAMTS4`, `BIN1`)
  - See the IGV snapshots below for a visual representation of some of
    these regions.

<figure>
<img src="docs/BIN1.png"
alt="Differential accessibility at the transcription start site of BIN1, a primary genetic risk factor for Alzheimer’s disease." />
<figcaption aria-hidden="true"><em>Differential accessibility at the
transcription start site of BIN1, a primary genetic risk factor for
Alzheimer’s disease</em>.</figcaption>
</figure>

<figure>
<img src="docs/ADAMTS4.png"
alt="Differential accessibility at the transcription start site of ADAMTS4, a primary genetic risk factor for Alzheimer’s disease." />
<figcaption aria-hidden="true"><em>Differential accessibility at the
transcription start site of ADAMTS4, a primary genetic risk factor for
Alzheimer’s disease</em>.</figcaption>
</figure>

### GREAT: Enriched Biological Processes

The top enriched biological process returned by GREAT was
[**`GO:2000311`**](https://www.ebi.ac.uk/QuickGO/term/GO:2000311):
*regulation of AMPA receptor activity*. This process was determined to
be significant by both the binomial and hypergeometric models employed
by GREAT. From (Zhang et al. 2018),

> “As the primary mediator for synaptic transmission, AMPA receptors
> (AMPARs) are crucial for synaptic plasticity and higher brain
> functions. A downregulation of AMPAR expression has been indicated as
> one of the early pathological molecular alterations in Alzheimer’s
> disease (AD)…”

Other significant AD-relevant processes enriched in the determined
differentially accessible regions include

- [**`GO:0007611`**](https://www.ebi.ac.uk/QuickGO/term/GO:0007611):
  *learning or memory*

- [**`GO:1901629`**](https://www.ebi.ac.uk/QuickGO/term/GO:1901629)
  *regulation of pre-synaptic membrane organization*.

### Sample clustering in PCA space

We note that the samples are linearly separable with respect to disease
status in the first two principal components of DARs’ count data,
suggesting collective discriminative power.

<figure>
<img src="docs/dnase_ad_DESeq_results2_pca_dar_nTop1000.png"
alt="PCA representation of nTop=1000 most variable differentially accesible peak regions colored according to disease status" />
<figcaption aria-hidden="true">PCA representation of
<em><code>nTop=1000</code> most variable differentially accesible peak
regions</em> colored according to disease status</figcaption>
</figure>

# Appendix: ‘Noisy’ Data Generation

1.  Merge the following into one alignment (`noisy.bam`) to incorporate
    ‘real’ biological data and better capture plausible noise processes
    while forcing low SNR. Only a small fraction of the aligned
    sequences in ‘noisy.bam’ correspond to directly relevant ATAC-seq
    data.

- ENCFF320DVY.sub25.sorted.bam
  - Subsampled ATAC-seq alignment file from ENCODE in a lymphoblastoid
    cell line
- ENCFF295KVO.bam
  - MCF-7 ChIP-seq control input from ENCODE
- ENCFF415JON.bam
  - GM12878 ChIP-seq control input from ENCODE
- ENCFF492KJP.bam
  - K562 ChIP-seq control input from ENCODE

``` bash
samtools merge -@ 12 -o noisy.bam ENCFF320DVY.sub25.sorted.bam \
ENCFF295KVO.bam ENCFF415JON.bam ENCFF492KJP.bam
```

``` bash
> du -h *.bam
5.6G    ENCFF295KVO.bam
1.0G    ENCFF320DVY.sub25.sorted.bam
3.9G    ENCFF415JON.bam
3.2G    ENCFF492KJP.bam
 12G    noisy.bam
```

2.  Randomly subsample from `noisy.bam` to generate distinct noisy
    samples.

- **Seeds 1-5**: 100, 2000, 30000, 400000, 5000000
- **Seeds 6-10**: 60000, 5000, 400, 30, 1

``` bash
# CWD: ./noisy
samtools view -@ 4 -b -s <seed>.25 noisy_source/noisy.bam | samtools sort -@ 4 > noisy<1...5>.bam
samtools view -@ 4 -b -s <seed>.50 noisy_source/noisy.bam | samtools sort -@ 4 > noisy<6...10>.bam
```

``` bash
3.7G    noisy1.bam
3.7G    noisy2.bam
3.7G    noisy3.bam
3.7G    noisy4.bam
3.7G    noisy5.bam
6.7G    noisy6.bam
6.7G    noisy7.bam
6.7G    noisy8.bam
6.7G    noisy9.bam
6.7G    noisy10.bam
```

# References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-epstein2018" class="csl-entry">

Epstein, Irina, and Steven Finkbeiner. 2018. “The Arc of Cognition:
Signaling Cascades Regulating Arc and Implications for Cognitive
Function and Disease.” *Seminars in Cell &Amp; Developmental Biology* 77
(May): 63–72. <https://doi.org/10.1016/j.semcdb.2017.09.023>.

</div>

<div id="ref-hamilton2023" class="csl-entry">

Hamilton, Nolan H, and Terrence S Furey. 2023. “ROCCO: A Robust Method
for Detection of Open Chromatin via Convex Optimization.”
*Bioinformatics* 39 (12): btad725.
<https://doi.org/10.1093/bioinformatics/btad725>.

</div>

<div id="ref-hamilton2025" class="csl-entry">

Hamilton, Nolan H, Benjamin D McMichael, Michael I Love, and Terrence S
Furey. 2025. “Genome-Wide Uncertainty-Moderated Extraction of Signal
Annotations from Multi-Sample Functional Genomics Data.” *bioRxiv*.
<https://doi.org/10.1101/2025.02.05.636702>.

</div>

<div id="ref-love2014" class="csl-entry">

Love, Michael I, Wolfgang Huber, and Simon Anders. 2014. “Moderated
Estimation of Fold Change and Dispersion for RNA-Seq Data with DESeq2.”
*Genome Biology* 15 (12). <https://doi.org/10.1186/s13059-014-0550-8>.

</div>

<div id="ref-mclean2010" class="csl-entry">

McLean, Cory Y, Dave Bristor, Michael Hiller, Shoa L Clarke, Bruce T
Schaar, Craig B Lowe, Aaron M Wenger, and Gill Bejerano. 2010. “GREAT
Improves Functional Interpretation of Cis-Regulatory Regions.” *Nature
Biotechnology* 28 (5): 495–501. <https://doi.org/10.1038/nbt.1630>.

</div>

<div id="ref-risso2014" class="csl-entry">

Risso, Davide, John Ngai, Terence P Speed, and Sandrine Dudoit. 2014.
“Normalization of RNA-Seq Data Using Factor Analysis of Control Genes or
Samples.” *Nature Biotechnology* 32 (9): 896–902.
<https://doi.org/10.1038/nbt.2931>.

</div>

<div id="ref-tanigawa2022" class="csl-entry">

Tanigawa, Yosuke, Ethan S. Dyer, and Gill Bejerano. 2022. “WhichTF Is
Functionally Important in Your Open Chromatin Data?” Edited by Ferhat
Ay. *PLOS Computational Biology* 18 (8): e1010378.
<https://doi.org/10.1371/journal.pcbi.1010378>.

</div>

<div id="ref-vandongen2025" class="csl-entry">

VanDongen, Antonius M. 2025. “Arc: A Therapeutic Hub for Alzheimer’s
Disease.” *bioRxiv*. <https://doi.org/10.1101/2025.01.16.632170>.

</div>

<div id="ref-yu2015" class="csl-entry">

Yu, Guangchuang, Li-Gen Wang, and Qing-Yu He. 2015. “ChIPseeker: An
r/Bioconductor Package for ChIP Peak Annotation, Comparison and
Visualization.” *Bioinformatics* 31 (14): 2382–83.
<https://doi.org/10.1093/bioinformatics/btv145>.

</div>

<div id="ref-zhang2018" class="csl-entry">

Zhang, Yanmin, Ouyang Guo, Yuda Huo, Guan Wang, and Heng-Ye Man. 2018.
“Amyloid-β Induces AMPA Receptor Ubiquitination and Degradation in
Primary Neurons and Human Brains of Alzheimer’s Disease.” Edited by Tao
Ma. *Journal of Alzheimer’s Disease* 62 (4): 1789–1801.
<https://doi.org/10.3233/jad-170879>.

</div>

</div>
