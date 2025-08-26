
# Part of the usage documentation (atac20)
library(ChIPseeker)
library(clusterProfiler)
library(ggplot2)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(org.Hs.eg.db)
library(enrichplot)
library(Rgraphviz)
library(GOSemSim)
library(topGO)
library(knitr)

peakfile="excluded.bed"
peaks=readPeakFile(peakfile)
txdb= TxDb.Hsapiens.UCSC.hg38.knownGene
peakAnno <- annotatePeak(
  peaks,
  TxDb = txdb,
  annoDb= "org.Hs.eg.db")

gene_ids=as.data.frame(peakAnno)$geneId

ego = enrichGO(
  gene = gene_ids,
  OrgDb  = org.Hs.eg.db,
  keyType  = "ENTREZID",
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.01,
  qvalueCutoff  = 0.05,
  readable = TRUE)

# merges redundant terms at semantic similarity 0.70
ego_simp <- simplify(
  ego,
  cutoff= 0.70,
  by = "FoldEnrichment",
  select_fun = min,
  measure="Wang")

kable(head(as.data.frame(ego_simp), 50), caption = "GO Enrichment Results")