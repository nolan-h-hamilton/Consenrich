# Consenrich → Consensus Peak Calling

* Consenrich can be paired with peak callers such as [ROCCO](https://github.com/nolan-h-hamilton/ROCCO) to call consensus peaks given multiple HTS data.
* Methods based on the [continuous wavelet transform (CWT)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html) may also be effective given Consenrich's preservation of spatial structure present in HTS data.

We provide some casual examples below for ATAC-seq and ChIP-seq.

## ATAC-seq Het10

![ATAC Het10 Consensus Peaks](peaks_demo.png)

## ChIP-seq

![ChIP-seq POL2RA Consensus Peaks](consensus_peaks_chip.png)

When applying ROCCO--originally designed for ATAC-seq--to Consenrich-extracted ChIP-seq signals, users may find it useful to experiment with the `--disable_locratio`, `--disable_parsig`, and `--budget` arguments of ROCCO to balance precision and recall. A specific ChIP-seq 'mode' for ROCCO is upcoming.
