suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
  library(fgsea)
  library(AUCell)
})

root <- normalizePath(file.path(getwd()), mustWork = TRUE)
out_dir <- file.path(root, "bulk_pre_sc", "biology_interpretation", "figure6_program689_biology")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

programs <- c("Program_6", "Program_8", "Program_9")
state_levels <- c("Other", programs)
high_z_threshold <- 1.60
high_margin_threshold <- 0.80
set.seed(20260605)

object_files <- c(
  GSE161195 = file.path(root, "raw scrna data", "GSE161195_PRMM_program689_harmony_seurat.rds"),
  GSE161801 = file.path(root, "raw scrna data2", "GSE161801_paired_myeloma_program689_harmony_seurat.rds")
)

program_gene_file <- file.path(
  root,
  "bulk_pre_sc",
  "model_upgrade_v14_multimodal_dictionary",
  "balanced",
  "our_multimodal_dictionary_v14_core_program_genes.csv"
)

custom_gene_sets <- list(
  MYELOMA_PLASMA_CELL_IDENTITY = c("SDC1", "XBP1", "PRDM1", "IRF4", "MZB1", "JCHAIN", "TNFRSF17", "SLAMF7", "CD38", "CD79A", "DERL3"),
  MYELOMA_PROTEIN_HOMEOSTASIS = c("XBP1", "HSPA5", "HSP90B1", "DNAJB9", "HERPUD1", "PDIA4", "PDIA6", "SSR4", "SEC61A1", "DERL3", "DNAJC3"),
  MYELOMA_PROTEASOME_STRESS = c("PSMB5", "PSMB8", "PSMB9", "PSMC1", "PSMC2", "PSMD1", "PSMD2", "PSMD11", "PSME1", "PSME2", "NFE2L2"),
  MYELOMA_NFKB_SURVIVAL = c("NFKB1", "NFKB2", "RELA", "REL", "NFKBIA", "TNFAIP3", "BIRC3", "TRAF1", "CD40", "LTBR", "MAP3K8"),
  MYELOMA_ANTIGEN_PRESENTATION_COSTIM = c("CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CIITA", "CD80", "CD86", "TAP1", "TAP2"),
  TUMOR_CELL_CYCLE_PROLIFERATION = c("MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "CCNB1", "CDK1", "BUB1", "AURKA"),
  TUMOR_MYC_TRANSLATION_GROWTH = c("MYC", "NPM1", "NCL", "HSPD1", "HSPA9", "LDHA", "ODC1", "EIF4A1", "EIF4E", "RPLP0", "RPS3", "RAN"),
  TUMOR_DNA_REPAIR_REPLICATION_STRESS = c("BRCA1", "RAD51", "CHEK1", "MSH2", "MSH6", "PCNA", "PARP1", "XRCC5", "XRCC6", "TOPBP1", "RPA1", "FEN1"),
  TUMOR_OXPHOS_MITOCHONDRIAL = c("NDUFA1", "NDUFA9", "NDUFB8", "SDHB", "UQCRC1", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "ATP5MC1", "VDAC1"),
  TUMOR_GLYCOLYSIS_HYPOXIA = c("SLC2A1", "HK2", "PFKP", "ALDOA", "GAPDH", "PGK1", "ENO1", "PKM", "LDHA", "CA9", "VEGFA", "BNIP3"),
  TUMOR_UPR_ER_STRESS = c("XBP1", "HSPA5", "ATF4", "ATF6", "ERN1", "DDIT3", "DNAJB9", "HERPUD1", "SEL1L", "SYVN1", "PDIA4", "PDIA6"),
  TUMOR_INTERFERON_RESPONSE = c("STAT1", "IRF1", "IRF7", "ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "OAS2", "CXCL10", "PSMB8", "TAP1"),
  TUMOR_ADHESION_MIGRATION = c("CD44", "ITGA4", "ITGB1", "ITGA5", "CXCR4", "VCAM1", "LGALS3", "FN1", "ZEB2", "CD36"),
  TUMOR_STRESS_APOPTOSIS = c("DDIT3", "ATF4", "ATF3", "PMAIP1", "BBC3", "BAX", "BCL2L11", "GADD45A", "HMOX1", "DUSP1"),
  TUMOR_DORMANCY_STEMLIKE = c("CDKN1C", "DUSP1", "KLF4", "ZEB2", "ALDH1A1", "ALDH1L1", "GAS6", "YAP1", "NTRK2", "AREG")
)

assign_high_stringency <- function(obj) {
  z <- as.matrix(obj@meta.data[, paste0(programs, "_z")])
  colnames(z) <- programs
  best_idx <- max.col(z, ties.method = "first")
  sorted <- t(apply(z, 1, sort))
  best_score <- z[cbind(seq_len(nrow(z)), best_idx)]
  margin <- sorted[, ncol(sorted)] - sorted[, ncol(sorted) - 1]
  labels <- programs[best_idx]
  labels[best_score < high_z_threshold | margin < high_margin_threshold] <- "Other"
  obj$program_class_high <- factor(labels, levels = state_levels)
  obj$program_high_score <- as.numeric(best_score)
  obj$program_high_margin <- as.numeric(margin)
  obj
}

detected_features <- function(obj, min_fraction = 0.01, include_genes = character()) {
  data_mat <- GetAssayData(obj, assay = "RNA", layer = "data")
  detected <- Matrix::rowSums(data_mat > 0) >= ceiling(min_fraction * ncol(obj))
  unique(intersect(rownames(obj), c(rownames(obj)[detected], include_genes, VariableFeatures(obj))))
}

load_gene_sets <- function() {
  custom_gene_sets
}

run_markers <- function(obj, dataset, features) {
  Idents(obj) <- "program_class_high"
  message(dataset, ": FindAllMarkers")
  markers <- FindAllMarkers(
    obj,
    features = features,
    assay = "RNA",
    only.pos = TRUE,
    logfc.threshold = 0.15,
    test.use = "wilcox",
    slot = "data",
    min.pct = 0.05,
    return.thresh = 0.05,
    max.cells.per.ident = 4000,
    random.seed = 20260605,
    verbose = FALSE
  )
  markers <- markers[markers$cluster %in% programs, , drop = FALSE]
  markers$dataset <- dataset
  markers
}

run_ranked_de <- function(obj, dataset, features) {
  Idents(obj) <- "program_class_high"
  rows <- list()
  for (program in programs) {
    message(dataset, ": ranked DE ", program)
    de <- FindMarkers(
      obj,
      ident.1 = program,
      ident.2 = setdiff(state_levels, program),
      features = features,
      assay = "RNA",
      test.use = "wilcox",
      slot = "data",
      min.pct = 0.01,
      logfc.threshold = 0,
      max.cells.per.ident = 4000,
      random.seed = 20260605,
      verbose = FALSE
    )
    de$gene <- rownames(de)
    de$program <- program
    de$dataset <- dataset
    rows[[program]] <- de
  }
  rbindlist(rows, fill = TRUE)
}

run_fgsea_for_de <- function(de, gene_sets, dataset_label) {
  rows <- list()
  for (program in programs) {
    sub <- de[de$program == program, , drop = FALSE]
    sub <- as.data.table(sub)
    sub <- sub[, .(avg_log2FC = mean(avg_log2FC, na.rm = TRUE)), by = gene]
    stats <- sub$avg_log2FC
    names(stats) <- sub$gene
    stats <- stats[is.finite(stats)]
    stats <- sort(stats, decreasing = TRUE)
    fg <- fgsea(gene_sets, stats, minSize = 5, maxSize = 500, eps = 0)
    fg$program <- program
    fg$dataset <- dataset_label
    rows[[program]] <- fg
  }
  rbindlist(rows, fill = TRUE)
}

run_consensus_fgsea <- function(all_de, gene_sets) {
  rows <- list()
  for (program in programs) {
    sub <- all_de[all_de$program == program, , drop = FALSE]
    sub <- as.data.table(sub)
    sub <- sub[, .(avg_log2FC = mean(avg_log2FC, na.rm = TRUE)), by = c("gene", "dataset")]
    wide <- reshape(
      as.data.frame(sub[, c("gene", "dataset", "avg_log2FC")]),
      idvar = "gene",
      timevar = "dataset",
      direction = "wide"
    )
    lfc_cols <- grep("^avg_log2FC\\.", colnames(wide), value = TRUE)
    wide <- wide[complete.cases(wide[, lfc_cols]), , drop = FALSE]
    score <- rowMeans(scale(wide[, lfc_cols]), na.rm = TRUE)
    names(score) <- wide$gene
    score <- sort(score[is.finite(score)], decreasing = TRUE)
    fg <- fgsea(gene_sets, score, minSize = 5, maxSize = 500, eps = 0)
    fg$program <- program
    fg$dataset <- "consensus"
    rows[[program]] <- fg
  }
  rbindlist(rows, fill = TRUE)
}

score_aucell <- function(obj, dataset, gene_sets) {
  message(dataset, ": AUCell pathway scoring")
  gene_universe <- unique(c(VariableFeatures(obj), unlist(gene_sets, use.names = FALSE)))
  gene_universe <- intersect(gene_universe, rownames(obj))
  expr <- GetAssayData(obj, assay = "RNA", layer = "data")[gene_universe, , drop = FALSE]
  filtered_sets <- lapply(gene_sets, function(x) intersect(unique(x), rownames(expr)))
  filtered_sets <- filtered_sets[vapply(filtered_sets, length, integer(1)) >= 5]
  rankings <- AUCell_buildRankings(expr, nCores = 1, plotStats = FALSE, verbose = FALSE)
  auc <- AUCell_calcAUC(filtered_sets, rankings, aucMaxRank = max(25, ceiling(0.05 * nrow(expr))), verbose = FALSE)
  auc_df <- as.data.frame(t(getAUC(auc)), check.names = FALSE)
  auc_df$cell_id <- rownames(auc_df)
  auc_df$dataset <- dataset
  auc_df$program_class_high <- obj$program_class_high
  auc_df$phase_display <- obj$phase_display
  auc_df
}

summarize_auc <- function(auc_df) {
  pathway_cols <- setdiff(colnames(auc_df), c("cell_id", "dataset", "program_class_high", "phase_display"))
  rows <- list()
  for (program in programs) {
    in_class <- auc_df$program_class_high == program
    for (pathway in pathway_cols) {
      x <- auc_df[in_class, pathway]
      y <- auc_df[auc_df$program_class_high == "Other", pathway]
      if (length(x) < 5 || length(y) < 5) next
      wt <- wilcox.test(x, y)
      rows[[length(rows) + 1]] <- data.frame(
        dataset = unique(auc_df$dataset),
        program = program,
        pathway = pathway,
        mean_program = mean(x, na.rm = TRUE),
        mean_other = mean(y, na.rm = TRUE),
        delta = mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE),
        p_value = wt$p.value,
        stringsAsFactors = FALSE
      )
    }
  }
  out <- rbindlist(rows, fill = TRUE)
  out$p_adj <- p.adjust(out$p_value, method = "BH")
  out
}

consensus_markers <- function(all_markers) {
  m <- all_markers[all_markers$cluster %in% programs & all_markers$avg_log2FC > 0, , drop = FALSE]
  m$sig <- m$p_val_adj < 0.05
  rows <- list()
  for (program in programs) {
    sub <- m[m$cluster == program, , drop = FALSE]
    genes <- unique(sub$gene)
    for (gene in genes) {
      g <- sub[sub$gene == gene, , drop = FALSE]
      if (!all(c("GSE161195", "GSE161801") %in% g$dataset)) next
      a <- g[g$dataset == "GSE161195", ][1, ]
      b <- g[g$dataset == "GSE161801", ][1, ]
      rows[[length(rows) + 1]] <- data.frame(
        program = program,
        gene = gene,
        logfc_GSE161195 = a$avg_log2FC,
        logfc_GSE161801 = b$avg_log2FC,
        padj_GSE161195 = a$p_val_adj,
        padj_GSE161801 = b$p_val_adj,
        pct1_GSE161195 = a$pct.1,
        pct1_GSE161801 = b$pct.1,
        consensus_score = mean(c(a$avg_log2FC, b$avg_log2FC)) *
          (as.numeric(a$p_val_adj < 0.05) + as.numeric(b$p_val_adj < 0.05) + 0.5),
        stringsAsFactors = FALSE
      )
    }
  }
  out <- rbindlist(rows, fill = TRUE)
  out <- out[order(out$program, -out$consensus_score, -pmin(out$logfc_GSE161195, out$logfc_GSE161801)), ]
  out
}

annotate_marker_genes <- function(consensus, gene_sets) {
  rows <- list()
  for (i in seq_len(nrow(consensus))) {
    gene <- consensus$gene[i]
    hits <- names(gene_sets)[vapply(gene_sets, function(gs) gene %in% gs, logical(1))]
    rows[[i]] <- paste(hits, collapse = ";")
  }
  consensus$pathway_membership <- unlist(rows)
  consensus
}

message("Loading gene sets")
gene_sets <- load_gene_sets()
program_genes <- read.csv(program_gene_file, stringsAsFactors = FALSE)
program_gene_union <- unique(program_genes$gene[program_genes$program %in% programs])
include_genes <- unique(c(unlist(gene_sets, use.names = FALSE), program_gene_union))

objects <- list()
all_markers <- list()
all_de <- list()
all_auc <- list()
all_auc_summary <- list()

for (dataset in names(object_files)) {
  message("Loading object ", dataset)
  obj <- readRDS(object_files[[dataset]])
  DefaultAssay(obj) <- "RNA"
  obj <- assign_high_stringency(obj)
  features <- detected_features(obj, min_fraction = 0.01, include_genes = include_genes)
  message(dataset, ": marker feature universe = ", length(features))
  write.csv(
    data.frame(dataset = dataset, feature = features),
    file.path(out_dir, paste0("figure6_", dataset, "_marker_feature_universe.csv")),
    row.names = FALSE
  )

  markers <- run_markers(obj, dataset, features)
  ranked_de <- run_ranked_de(obj, dataset, features)
  auc_df <- score_aucell(obj, dataset, gene_sets)
  auc_summary <- summarize_auc(auc_df)

  all_markers[[dataset]] <- markers
  all_de[[dataset]] <- ranked_de
  all_auc[[dataset]] <- auc_df
  all_auc_summary[[dataset]] <- auc_summary
  objects[[dataset]] <- NULL

  fwrite(markers, file.path(out_dir, paste0("figure6_", dataset, "_program689_FindAllMarkers.csv")))
  fwrite(ranked_de, file.path(out_dir, paste0("figure6_", dataset, "_program689_ranked_DE.csv")))
  fwrite(auc_summary, file.path(out_dir, paste0("figure6_", dataset, "_program689_AUCell_pathway_summary.csv")))
}

all_markers_df <- rbindlist(all_markers, fill = TRUE)
all_de_df <- rbindlist(all_de, fill = TRUE)
all_auc_summary_df <- rbindlist(all_auc_summary, fill = TRUE)
fwrite(all_markers_df, file.path(out_dir, "figure6_program689_FindAllMarkers_all.csv"))
fwrite(all_de_df, file.path(out_dir, "figure6_program689_ranked_DE_all.csv"))
fwrite(all_auc_summary_df, file.path(out_dir, "figure6_program689_AUCell_pathway_summary_all.csv"))

marker_consensus <- consensus_markers(all_markers_df)
marker_consensus <- annotate_marker_genes(marker_consensus, gene_sets)
fwrite(marker_consensus, file.path(out_dir, "figure6_program689_consensus_marker_genes.csv"))

fgsea_dataset <- run_fgsea_for_de(all_de_df, gene_sets, "per_dataset")
fgsea_consensus <- run_consensus_fgsea(all_de_df, gene_sets)
fwrite(fgsea_dataset, file.path(out_dir, "figure6_program689_marker_rank_GSEA_by_dataset.csv"))
fwrite(fgsea_consensus, file.path(out_dir, "figure6_program689_marker_rank_GSEA_consensus.csv"))

auc_wide <- reshape(
  as.data.frame(all_auc_summary_df[, c("dataset", "program", "pathway", "delta", "p_adj")]),
  idvar = c("program", "pathway"),
  timevar = "dataset",
  direction = "wide"
)
auc_wide$direction_consistent <- sign(auc_wide$delta.GSE161195) == sign(auc_wide$delta.GSE161801)
auc_wide$mean_delta <- rowMeans(auc_wide[, grep("^delta\\.", colnames(auc_wide))], na.rm = TRUE)
auc_wide$min_abs_delta <- apply(abs(auc_wide[, grep("^delta\\.", colnames(auc_wide))]), 1, min, na.rm = TRUE)
auc_wide$both_significant <- auc_wide$p_adj.GSE161195 < 0.05 & auc_wide$p_adj.GSE161801 < 0.05
auc_wide <- auc_wide[order(auc_wide$program, -auc_wide$direction_consistent, -auc_wide$both_significant, -auc_wide$min_abs_delta), ]
write.csv(auc_wide, file.path(out_dir, "figure6_program689_AUCell_consensus_pathways.csv"), row.names = FALSE)

gsea_wide <- reshape(
  as.data.frame(fgsea_consensus[, c("program", "pathway", "NES", "padj")]),
  idvar = c("program", "pathway"),
  timevar = "program",
  direction = "wide"
)

top_marker_candidates <- marker_consensus[marker_consensus$padj_GSE161195 < 0.05 | marker_consensus$padj_GSE161801 < 0.05, ]
top_marker_candidates <- top_marker_candidates[order(top_marker_candidates$program, -top_marker_candidates$consensus_score), ]
top_marker_candidates <- rbindlist(lapply(programs, function(p) head(top_marker_candidates[top_marker_candidates$program == p, ], 20)), fill = TRUE)
fwrite(top_marker_candidates, file.path(out_dir, "figure6_program689_top_consensus_marker_candidates.csv"))

top_gsea <- fgsea_consensus[padj < 0.10, ]
top_gsea <- top_gsea[order(program, -abs(NES)), ]
top_gsea <- rbindlist(lapply(programs, function(p) head(top_gsea[top_gsea$program == p, ], 15)), fill = TRUE)
fwrite(top_gsea, file.path(out_dir, "figure6_program689_top_consensus_GSEA.csv"))

top_auc <- auc_wide[auc_wide$direction_consistent == TRUE, ]
top_auc <- rbindlist(lapply(programs, function(p) head(top_auc[top_auc$program == p, ], 15)), fill = TRUE)
write.csv(top_auc, file.path(out_dir, "figure6_program689_top_consensus_AUCell_pathways.csv"), row.names = FALSE)

message("Done. Outputs written to: ", out_dir)
message("Top marker candidates:")
print(top_marker_candidates[, c("program", "gene", "logfc_GSE161195", "logfc_GSE161801", "padj_GSE161195", "padj_GSE161801", "pathway_membership")])
message("Top AUCell consensus pathways:")
print(top_auc[, c("program", "pathway", "delta.GSE161195", "delta.GSE161801", "p_adj.GSE161195", "p_adj.GSE161801", "direction_consistent", "both_significant")])
