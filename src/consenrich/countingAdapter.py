from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path

import ccounts as counts


def extensionValues(sources, configured):
    if configured is None or isinstance(configured, int):
        return [configured] * len(sources)
    if len(configured) != len(sources):
        raise ValueError("extendFrom5pBP must match the treatment source count")
    return list(configured)


def normalizationTerritory(genomeArgs, reader, sampleIDs):
    territory = reader.commonFeatureTerritory(
        chromosomes=genomeArgs.normalizationChromosomes, sampleIDs=sampleIDs,
    )
    if not territory["usedChromosomes"]:
        raise ValueError(
            "No normalization chromosomes are shared by the requested territory "
            "and selected source features"
        )
    return territory


def selectionSpec(source, scArgs):
    metadata = source.cellMetadata or scArgs.cellMetadata
    selection = dict(metadata or {})
    sampleValue = source.metadataSample if source.metadataSample is not None else source.sampleName
    if metadata is not None and metadata.get("sampleColumn") is not None:
        if not isinstance(sampleValue, str) or not sampleValue:
            raise ValueError("A metadata sample column requires source name or metadataSample")
        selection["sampleValue"] = sampleValue
    selection["groups"] = source.selectGroups
    selection["barcodes"] = source.selectedBarcodes
    for field in ("barcodeAllowListFile", "barcodeGroupMapFile"):
        if getattr(source, field) is not None:
            selection[field] = getattr(source, field)
    return {key: value for key, value in selection.items() if value is not None} or None


def sourceSpecs(config):
    samArgs = config["samArgs"]
    scArgs = config["scArgs"]
    inputs = config["inputArgs"]
    treatments = list(inputs.treatmentSources or ())
    references = list(inputs.controlSources or ())
    extensions = extensionValues(treatments, samArgs.extendFrom5pBP)
    minimumTemplateLength = samArgs.minTemplateLength
    result = []
    for role, sources in (("treatment", treatments), ("reference", references)):
        for index, source in enumerate(sources):
            selection = selectionSpec(source, scArgs)
            options = None
            if source.sourceKind == "BAM":
                extension = extensions[index] if extensions else None
                bamMode = source.bamInputMode or samArgs.bamInputMode
                options = counts.CountOptions(
                    countMode=source.countMode or samArgs.defaultCountMode,
                    bamInputMode=bamMode,
                    flagExclude=samArgs.samFlagExclude,
                    minMappingQuality=samArgs.minMappingQuality,
                    minTemplateLengthBP=(-1 if minimumTemplateLength is None
                                         or minimumTemplateLength < 0 else minimumTemplateLength),
                    maxTemplateLengthBP=samArgs.maxInsertSize,
                    shiftForward5p=samArgs.shiftForward5p,
                    shiftReverse5p=samArgs.shiftReverse5p,
                    extendFrom5pBP=extension,
                    inferFragmentLength=role == "treatment" and bamMode != "fragments" and (
                        (samArgs.inferFragmentLength is None and extension is None
                         and bamMode == "auto")
                        or bool(samArgs.inferFragmentLength)
                    ),
                )
            elif source.sourceKind == "FRAGMENTS":
                options = counts.CountOptions(
                    countMode=source.countMode or scArgs.defaultCountMode,
                    fragmentsUseReadSupport=source.fragmentsUseReadSupport,
                )
            result.append(counts.SourceSpec(
                path=source.path, sourceKind=source.sourceKind,
                sampleID=f"{role}{index}", countOptions=options,
                barcodeTag=source.barcodeTag or scArgs.barcodeTag,
                matrixRows=source.matrixRows or "cells", selection=selection or None,
            ))
    return result


def _automaticThreadCount(sources):
    handleCount = sum(source.sourceKind != "H5AD" for source in sources)
    if not handleCount:
        return 1
    # Paired-end fragment length estimation can open one additional BAM handle.
    handleCount += any(source.sourceKind == "BAM" for source in sources)

    try:
        if hasattr(os, "process_cpu_count"):
            cpuBudget = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):
            cpuBudget = len(os.sched_getaffinity(0))
        else:
            cpuBudget = os.cpu_count()
    except OSError:
        cpuBudget = None

    slurmAllocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurmAllocation is not None:
        slurmCPUs = int(slurmAllocation)
        if slurmCPUs < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
        cpuBudget = min(cpuBudget, slurmCPUs) if cpuBudget else slurmCPUs

    # Reserve the calling thread and one HTSlib I/O thread per handle.
    return max(1, min(4, ((cpuBudget or 1) - 1) // handleCount - 1))


@contextmanager
def openReader(config):
    samArgs = config["samArgs"]
    grid = counts.GridSpec(
        config["genomeArgs"].chromosomeSizes,
        binSizeBP=config["countingArgs"].intervalSizeBP,
    )
    specs = sourceSpecs(config)
    threads = _automaticThreadCount(specs) if samArgs.samThreads == "auto" else samArgs.samThreads
    prepared = []
    for source in specs:
        if source.sourceKind in ("BAM", "FRAGMENTS"):
            if source.sourceKind == "BAM":
                indexPaths = (
                    Path(source.path + ".bai"), Path(source.path).with_suffix(".bai"),
                    Path(source.path + ".csi"), Path(source.path).with_suffix(".csi"),
                )
            else:
                indexPaths = (Path(source.path + ".tbi"), Path(source.path + ".csi"))
            if not any(path.is_file() for path in indexPaths):
                counts.prepareIndex(source.path, source.sourceKind, threads=threads)
        elif source.sourceKind == "BEDGRAPH":
            indexed = Path(source.path + ".tbi").exists() or Path(source.path + ".csi").exists()
            if not indexed:
                target = source.path + ".bgz"
                if not Path(target).exists() or not (Path(target + ".tbi").exists() or Path(target + ".csi").exists()):
                    counts.prepareIndex(source.path, "BEDGRAPH", targetPath=target,
                                        threads=threads)
                source = replace(source, path=target)
        prepared.append(source)
    with counts.openSources(
        prepared, grid, threads=threads,
        fragmentLengthOptions=samArgs.fragmentLengthOptions,
    ) as reader:
        for index, source in enumerate(config["inputArgs"].controlSources or ()):
            referenceID = f"reference{index}"
            reference = reader.getSourceInfo(referenceID)
            options = reference["countOptions"]
            if (source.sourceKind != "BAM" or options.bamInputMode == "fragments"
                    or options.extendFrom5pBP is not None):
                continue
            treatmentID = f"treatment{index}"
            treatment = reader.getSourceInfo(treatmentID)
            treatmentOptions = treatment["countOptions"]
            length = (treatmentOptions.extendFrom5pBP
                      if treatmentOptions is not None else None)
            infer = bool(samArgs.inferFragmentLength) or (
                samArgs.inferFragmentLength is None
                and ((source.bamInputMode or samArgs.bamInputMode) == "auto"
                     or length is not None)
            )
            if infer:
                if length is None:
                    length = reader.estimateFragmentLength(treatmentID)
                reader.setReadExtension(referenceID, length)
        yield reader
