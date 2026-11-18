import copy
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from . import core, countingAdapter


logger = logging.getLogger(__name__)


def executeFits(config, args, runConfigured, configureLogging):
    invocationDir = Path.cwd()
    config = copy.deepcopy(config)
    inputArgs = config["inputArgs"]
    sources = list(inputArgs.treatmentSources) + list(inputArgs.controlSources or ())
    resolved = []
    for source in sources:
        fields = {"path": str(Path(source.path).resolve())}
        for field in ("barcodeAllowListFile", "barcodeGroupMapFile"):
            if getattr(source, field) is not None:
                fields[field] = str(Path(getattr(source, field)).resolve())
        resolved.append(source._replace(**fields))
    treatmentCount = len(inputArgs.treatmentSources)
    treatments = resolved[:treatmentCount]
    references = resolved[treatmentCount:]
    genomeArgs = config["genomeArgs"]
    config["genomeArgs"] = genomeArgs._replace(**{
        field: str(Path(getattr(genomeArgs, field)).resolve())
        for field in ("chromSizesFile", "blacklistFile", "sparseBedFile")
        if getattr(genomeArgs, field) is not None
    })
    sizes = config["genomeArgs"].chromosomeSizes
    requested = tuple(config["genomeArgs"].chromosomes)
    missing = set(requested) - sizes.keys()
    if missing:
        raise ValueError(f"Requested chromosomes absent from sizes: {sorted(missing)}")
    referenceGrid = {chromosome: (0, sizes[chromosome]) for chromosome in requested}
    fits = config["scArgs"].fits
    fitSpecs = fits or ({"name": None, "selectGroups": None},)
    experimentDir = (invocationDir / str(config["experimentName"])).resolve()
    summaryPath = (experimentDir / "singleCellFits.json" if fits is not None
                   else invocationDir / f"consenrichOutput_{config['experimentName']}_inputs.json")
    summaryPath.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fit in fitSpecs:
        fitConfig = copy.deepcopy(config)
        selected = []
        for source in treatments + references:
            if fits is not None:
                if source.cellMetadata is None and config["scArgs"].cellMetadata is None and source.barcodeGroupMapFile is None:
                    raise ValueError("Grouped inputs require cell metadata or a group map")
                source = source._replace(selectGroups=list(fit["selectGroups"]))
            selected.append(source)
        fitConfig["inputArgs"] = core.inputParams(
            core.getSourcePaths(selected[:treatmentCount]), core.getSourcePaths(selected[treatmentCount:]),
            selected[:treatmentCount], selected[treatmentCount:],
        )
        outputDir = experimentDir / fit["name"] if fits is not None else invocationDir
        outputDir.mkdir(parents=True, exist_ok=True)
        fitArgs = copy.copy(args)
        if fits is not None:
            fitArgs.logFile = str(outputDir / "consenrich.log.jsonl")
        configureLogging(fitArgs.logFile, verbose=bool(args.verbose), verbose2=bool(args.verbose2),
                         verbosity=args.verbosity, progress=args.progress)
        summary = {
            "fit": fit["name"], "selectGroups": fit["selectGroups"],
            "retainedDonors": [], "omittedDonors": [], "cellCounts": {},
            "controlCellCounts": {}, "outputPath": str(outputDir), "status": "skipped",
        }
        summaries.append(summary)
        try:
            with countingAdapter.openReader(fitConfig) as reader:
                infos = reader.sourceInfo
                candidates = []
                omissionReasons = {}
                for index, source in enumerate(selected[:treatmentCount]):
                    sampleID = f"treatment{index}"
                    donor = source.sampleName or source.path
                    summary["cellCounts"][donor] = infos[sampleID]["selectedCellCount"]
                    pair = (sampleID, f"reference{index}") if references else (sampleID,)
                    for key in pair:
                        info = infos[key]
                        if info["selectedCellCount"] == 0 or (
                            info["sourceKind"] == "H5AD"
                            and info["selectedBarcodes"] is not None
                            and not info["selectedBarcodes"]
                        ):
                            omissionReasons[index] = "empty cell selection"
                            break
                    if index not in omissionReasons:
                        candidates.append(index)
                for index, source in enumerate(selected[treatmentCount:]):
                    summary["controlCellCounts"][source.sampleName or source.path] = infos[f"reference{index}"]["selectedCellCount"]
                retained = []
                if candidates:
                    candidateIDs = tuple(f"treatment{index}" for index in candidates) + (
                        tuple(f"reference{index}" for index in candidates) if references else ()
                    )
                    countingArgs = config["countingArgs"]
                    automaticFactors = countingArgs.scaleFactors is None or (
                        bool(references) and countingArgs.scaleFactorsControl is None
                    )
                    territory = (
                        countingAdapter.normalizationTerritory(
                            config["genomeArgs"], reader, candidateIDs,
                        )["usedChromosomes"]
                        if automaticFactors and countingArgs.normMethod not in ("SF", "ReadsInRegions")
                        else requested
                    )
                    totals = reader.libraryTotals(chromosomes=territory, sampleIDs=candidateIDs)
                    for index in candidates:
                        pair = (f"treatment{index}", f"reference{index}") if references else (f"treatment{index}",)
                        if any(infos[key]["sourceKind"] != "BEDGRAPH"
                               and totals.records[key]["eventMass"] <= 0 for key in pair):
                            omissionReasons[index] = "zero usable events"
                        else:
                            retained.append(index)
                summary["retainedDonors"] = [
                    selected[index].sampleName or selected[index].path for index in retained
                ]
                summary["omittedDonors"] = [
                    {"donor": source.sampleName or source.path, "reason": omissionReasons[index]}
                    for index, source in enumerate(selected[:treatmentCount]) if index in omissionReasons
                ]
                if not retained:
                    summary["reason"] = "zero donors"
                    continue
                keptTreatments = [selected[index] for index in retained]
                keptReferences = [selected[treatmentCount + index] for index in retained] if references else []
                fitConfig["inputArgs"] = core.inputParams(
                    core.getSourcePaths(keptTreatments), core.getSourcePaths(keptReferences), keptTreatments, keptReferences)
                fitConfig["countTreatmentIDs"] = tuple(f"treatment{index}" for index in retained)
                fitConfig["countReferenceIDs"] = tuple(f"reference{index}" for index in retained) if references else ()
                factorFields = {}
                for field in ("scaleFactors", "scaleFactorsControl"):
                    values = getattr(config["countingArgs"], field)
                    if values is not None:
                        factorFields[field] = [values[index] for index in retained]
                fitConfig["countingArgs"] = config["countingArgs"]._replace(**factorFields)
                os.chdir(outputDir)
                result = runConfigured(
                    fitConfig, fitArgs, referenceGrid if fits is not None else None,
                    countsReader=reader,
                )
                summary["normalization"] = result
                summary["status"] = "completed"
        except Exception as error:
            summary["status"] = "failed"
            summary["error"] = str(error)
            summaryPath.write_text(json.dumps(summaries, indent=2, default=asdict) + "\n")
            raise
        finally:
            os.chdir(invocationDir)
    summaryPath.write_text(json.dumps(summaries, indent=2, default=asdict) + "\n")
    return summaries
