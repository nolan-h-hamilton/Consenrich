import copy
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from . import core, io, misc_util
from . import _normalization as normalization
from .cellmetadata import MetadataSelections


logger = logging.getLogger("consenrich.sc_runs")


def executeFits(config, args, runConfigured, configureLogging):
    invocationDir = Path.cwd()
    config = copy.deepcopy(config)
    inputArgs = config["inputArgs"]
    sources = list(inputArgs.treatmentSources) + list(inputArgs.controlSources or [])
    resolvedSources = []
    for source in sources:
        fields = {"path": str(Path(source.path).resolve())}
        for field in ("barcodeAllowListFile", "barcodeGroupMapFile"):
            if getattr(source, field) is not None:
                fields[field] = str(Path(getattr(source, field)).resolve())
        resolvedSources.append(source._replace(**fields))
    treatmentCount = len(inputArgs.treatmentSources)
    treatmentSources = resolvedSources[:treatmentCount]
    controlSources = resolvedSources[treatmentCount:]
    genomeArgs = config["genomeArgs"]
    genomePaths = {}
    for field in ("chromSizesFile", "blacklistFile", "sparseBedFile"):
        path = getattr(genomeArgs, field)
        if path is not None:
            genomePaths[field] = str(Path(path).resolve())
    config["genomeArgs"] = genomeArgs._replace(**genomePaths)
    scArgs = config["scArgs"]
    fits = scArgs.fits
    selections = MetadataSelections()
    chromSizes = misc_util.getChromSizesDict(
        config["genomeArgs"].chromSizesFile,
        excludeChroms=config["genomeArgs"].excludeChroms,
    )
    physicalChroms = misc_util.getChromSizesDict(config["genomeArgs"].chromSizesFile)
    missingChroms = set(config["genomeArgs"].chromosomes) - physicalChroms.keys()
    if missingChroms:
        raise ValueError(f"Requested chromosomes absent from chromosome sizes: {sorted(missingChroms)}")
    referenceGrid = {
        chromosome: (0, int(chromSizes[chromosome]))
        for chromosome in config["genomeArgs"].chromosomes
        if chromosome in chromSizes
    }
    fitSpecs = fits or ({"name": None, "selectGroups": None},)
    experimentDir = (invocationDir / str(config["experimentName"])).resolve()
    summaryPath = (
        experimentDir / "singleCellFits.json" if fits is not None
        else invocationDir / f"consenrichOutput_{config['experimentName']}_inputs.json"
    )
    summaries = []
    with TemporaryDirectory(prefix="consenrich_cells_") as selectionDir:
        for fitIndex, fit in enumerate(fitSpecs):
            configureLogging(
                None, verbose=bool(args.verbose), verbose2=bool(args.verbose2),
                verbosity=args.verbosity, progress=args.progress,
            )
            fitConfig = copy.deepcopy(config)
            prepared = selections.prepareSources(
                treatmentSources + controlSources,
                sharedMetadata=scArgs.cellMetadata,
                selectGroups=fit["selectGroups"],
                grouped=fits is not None,
            )
            selectionPaths = {}
            for sourceIndex, source in enumerate(prepared):
                if (
                    fits is not None and sourceIndex < treatmentCount
                    and source.cellMetadata is None and scArgs.cellMetadata is None
                    and source.barcodeGroupMapFile is None
                ):
                    raise ValueError("Grouped treatment inputs require cell metadata or a barcode group map")
                if fits is not None and source.barcodeGroupMapFile is not None:
                    source = source._replace(selectGroups=list(fit["selectGroups"]))
                barcodes = core._resolveFragmentsBarcodeAllowSet(source)
                if barcodes is not None:
                    selectionKey = (source.path, source.barcodeTag, frozenset(barcodes))
                    if selectionKey not in selectionPaths:
                        allowPath = Path(selectionDir) / f"fit{fitIndex}_source{sourceIndex}.txt"
                        allowPath.write_text("".join(f"{barcode}\n" for barcode in sorted(barcodes)))
                        selectionPaths[selectionKey] = str(allowPath)
                    source = source._replace(
                        barcodeAllowListFile=selectionPaths[selectionKey],
                        barcodeGroupMapFile=None,
                        selectedBarcodes=frozenset(barcodes),
                        selectedCellCount=len(barcodes),
                        barcodeSelectionPrepared=True,
                    )
                prepared[sourceIndex] = source
            treatments = prepared[:treatmentCount]
            controls = prepared[treatmentCount:]
            retainedIndices = []
            omitted = []
            cells = {}
            for sourceIndex, treatment in enumerate(treatments):
                donor = treatment.sampleName or treatment.path
                cells[donor] = treatment.selectedCellCount
                pairedSources = [treatment] + ([controls[sourceIndex]] if controls else [])
                reason = None
                for source in pairedSources:
                    if source.selectedCellCount == 0:
                        reason = f"empty cell selection: {source.sampleName or source.path}"
                        break
                    if source.sourceKind == core.BEDGRAPH_SOURCE_KIND:
                        continue
                    mode = io._getSourceCountMode(
                        source, config["samArgs"].defaultCountMode, scArgs.defaultCountMode
                    )
                    bamMode = core._resolveSourceBamInputModeForCountMode(
                        source, config["samArgs"].bamInputMode, mode
                    )
                    excludedChroms = list(config["genomeArgs"].excludeForNorm or ())
                    if source.sourceKind == "FRAGMENTS":
                        excludedChroms.extend(
                            chromosome
                            for chromosome in core.ccounts.ccounts_getChromosomeNames(
                                source.path, sourceKind=source.sourceKind
                            )
                            if chromosome not in physicalChroms and chromosome not in excludedChroms
                        )
                    eventCount, _ = core.ccounts.ccounts_getAlignmentMappedReadCount(
                        source.path,
                        excludeChromosomes=excludedChroms,
                        threadCount=config["samArgs"].samThreads,
                        sourceKind=source.sourceKind,
                        barcodeAllowListFile=source.barcodeAllowListFile or "",
                        barcodeTag=source.barcodeTag or "CB",
                        countMode=normalization.nativeCountModeForPreset(mode),
                        flagExclude=core._resolveSourceFlagExclude(
                            config["samArgs"].samFlagExclude, bamMode
                        ),
                        minMappingQuality=config["samArgs"].minMappingQuality or 0,
                        minTemplateLength=config["samArgs"].minTemplateLength,
                        maxInsertSize=config["samArgs"].maxInsertSize,
                        pairedEndMode=int(bamMode == "fragments"),
                        fragmentsUseReadSupport=source.fragmentsUseReadSupport,
                    )
                    if eventCount <= 0:
                        reason = f"zero usable events: {source.sampleName or source.path}"
                        break
                if reason is None:
                    retainedIndices.append(sourceIndex)
                else:
                    logger.warning("fit=%s donor=%s omitted: %s", fit["name"], donor, reason)
                    omitted.append({"donor": donor, "reason": reason})
            outputDir = experimentDir / fit["name"] if fits is not None else invocationDir
            summary = {
                "fit": fit["name"],
                "selectGroups": list(fit["selectGroups"]) if fit["selectGroups"] is not None else None,
                "retainedDonors": [treatments[index].sampleName or treatments[index].path for index in retainedIndices],
                "omittedDonors": omitted,
                "cellCounts": cells,
                "controlCellCounts": {
                    source.sampleName or source.path: source.selectedCellCount for source in controls
                },
                "outputPath": str(outputDir),
                "status": "skipped",
            }
            summaries.append(summary)
            if not retainedIndices:
                summary["reason"] = "zero donors"
                logger.warning("fit=%s skipped: zero donors", fit["name"])
                continue
            if config["countingArgs"].normMethod == "SF" and (len(retainedIndices) < 3 or controls):
                summary["reason"] = "SF requires at least three donors without controls"
                logger.warning("fit=%s skipped: %s", fit["name"], summary["reason"])
                continue
            retainedTreatments = [treatments[index] for index in retainedIndices]
            retainedControls = [controls[index] for index in retainedIndices] if controls else []
            fitConfig["inputArgs"] = core.inputParams(
                core.getSourcePaths(retainedTreatments), core.getSourcePaths(retainedControls),
                retainedTreatments, retainedControls,
            )
            factors = {}
            for field in ("scaleFactors", "scaleFactorsControl"):
                values = getattr(config["countingArgs"], field)
                if values is not None and (field == "scaleFactors" or controls):
                    factors[field] = [values[index] for index in retainedIndices]
            fitConfig["countingArgs"] = config["countingArgs"]._replace(**factors)
            extensions = core._resolveExtendFrom5pBP(config["samArgs"].extendFrom5pBP, treatments)
            fitConfig["samArgs"] = config["samArgs"]._replace(
                extendFrom5pBP=[extensions[index] for index in retainedIndices]
            )
            fitArgs = copy.copy(args)
            if fits is not None:
                outputDir.mkdir(parents=True, exist_ok=True)
                fitArgs.logFile = str(outputDir / "consenrich.log.jsonl")
            configureLogging(
                fitArgs.logFile, verbose=bool(args.verbose), verbose2=bool(args.verbose2),
                verbosity=args.verbosity, progress=args.progress,
            )
            try:
                os.chdir(outputDir)
                result = runConfigured(
                    fitConfig, fitArgs,
                    referenceGrid if fits is not None else None,
                )
            except Exception as error:
                summary["status"] = "failed"
                summary["error"] = str(error)
                summaryPath.write_text(json.dumps(summaries, indent=2) + "\n")
                raise
            finally:
                os.chdir(invocationDir)
            summary["status"] = "completed"
            summary["normalization"] = result
    if fits is not None:
        experimentDir.mkdir(parents=True, exist_ok=True)
    summaryPath.write_text(json.dumps(summaries, indent=2) + "\n")
    logger.info("fit summary: %s", json.dumps(summaries))
    if all(summary["status"] == "skipped" for summary in summaries):
        logger.warning("All fits skipped successfully: %d", len(summaries))
    return summaries
