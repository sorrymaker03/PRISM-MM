#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(decoupleR)
  library(OmnipathR)
  library(fgsea)
})

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- "/tmp/prism_review1_test/scrna_kmeans_baseline"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

kmeans_file <- file.path(out_dir, "figureS8_scrna_kmeans_cells.csv")
if (!file.exists(kmeans_file)) {
  stop("Run extract_figureS8_scrna_kmeans_source.R before this script.")
}
kmeans_cells <- read.csv(kmeans_file, stringsAsFactors = FALSE)

cluster_levels <- paste0("Cluster ", 1:10)
phase_levels <- c("Ctrl", "Drug")

marker_sets <- list(
  "Program 6 markers" = c("TMEM156", "CTSF", "IDUA", "UGT8", "LAMP3", "PRDM1", "CD38"),
  "Program 8 markers" = c("CD44", "RGS2", "TYMP", "CD74", "HLA-DQB1", "STAT1", "BIRC3", "NFKBIA"),
  "Program 9 markers" = c("DUSP1", "CDKN1C", "KLRD1", "FOS", "JUNB", "ZFP36", "IL32")
)

custom_gene_sets <- list(
  MYELOMA_PLASMA_CELL_IDENTITY = c("SDC1", "XBP1", "PRDM1", "IRF4", "MZB1", "JCHAIN", "TNFRSF17", "SLAMF7", "CD38", "CD79A", "DERL3"),
  MYELOMA_PROTEIN_HOMEOSTASIS = c("XBP1", "HSPA5", "HSP90B1", "DNAJB9", "HERPUD1", "PDIA4", "PDIA6", "SSR4", "SEC61A1", "DERL3", "DNAJC3"),
  MYELOMA_NFKB_SURVIVAL = c("NFKB1", "NFKB2", "RELA", "REL", "NFKBIA", "TNFAIP3", "BIRC3", "TRAF1", "CD40", "LTBR", "MAP3K8"),
  MYELOMA_ANTIGEN_PRESENTATION_COSTIM = c("CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CIITA", "CD80", "CD86", "TAP1", "TAP2"),
  TUMOR_INTERFERON_RESPONSE = c("STAT1", "IRF1", "IRF7", "ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "OAS2", "CXCL10", "PSMB8", "TAP1"),
  TUMOR_ADHESION_MIGRATION = c("CD44", "ITGA4", "ITGB1", "ITGA5", "CXCR4", "VCAM1", "LGALS3", "FN1", "ZEB2", "CD36"),
  TUMOR_STRESS_APOPTOSIS = c("DDIT3", "ATF4", "ATF3", "PMAIP1", "BBC3", "BAX", "BCL2L11", "GADD45A", "HMOX1", "DUSP1"),
  TUMOR_DORMANCY_STEMLIKE = c("CDKN1C", "DUSP1", "KLF4", "ZEB2", "ALDH1A1", "ALDH1L1", "GAS6", "YAP1", "NTRK2", "AREG"),
  TUMOR_GLYCOLYSIS_HYPOXIA = c("SLC2A1", "HK2", "PFKP", "ALDOA", "GAPDH", "PGK1", "ENO1", "PKM", "LDHA", "CA9", "VEGFA", "BNIP3")
)

pathway_labels <- c(
  MYELOMA_PLASMA_CELL_IDENTITY = "Plasma-cell identity",
  MYELOMA_PROTEIN_HOMEOSTASIS = "Protein homeostasis",
  MYELOMA_NFKB_SURVIVAL = "NF-kB survival",
  MYELOMA_ANTIGEN_PRESENTATION_COSTIM = "Antigen presentation",
  TUMOR_INTERFERON_RESPONSE = "Interferon response",
  TUMOR_ADHESION_MIGRATION = "Adhesion/migration",
  TUMOR_STRESS_APOPTOSIS = "Stress/apoptosis",
  TUMOR_DORMANCY_STEMLIKE = "Dormancy-like state",
  TUMOR_GLYCOLYSIS_HYPOXIA = "Hypoxia/glycolysis"
)

selected_tfs <- c("TFAM", "BCL11A", "HIVEP2", "CIITA", "NFKB2", "REL", "HBP1", "NFYB", "ATF1", "ELK4", "DMTF1", "ZBTB14")
selected_progeny <- c("TNFa", "NFkB", "p53", "JAK-STAT", "Hypoxia", "TGFb")
progeny_labels <- c("TNFa" = "TNFa", "NFkB" = "NF-kB", "p53" = "p53 stress", "JAK-STAT" = "JAK-STAT", "Hypoxia" = "Hypoxia", "TGFb" = "TGFb")
selected_lr <- data.frame(
  pair = c("MIF->CD74", "B2M->HLA-F", "TIMP1->CD63", "MIF->CXCR4", "VIM->CD44", "B2M->LILRB1", "B2M->KLRD1", "HLA-B->KLRD1"),
  ligand = c("MIF", "B2M", "TIMP1", "MIF", "VIM", "B2M", "B2M", "HLA-B"),
  receptor = c("CD74", "HLA-F", "CD63", "CXCR4", "CD44", "LILRB1", "KLRD1", "KLRD1"),
  label = c("MIF-CD74", "B2M-HLA-F", "TIMP1-CD63", "MIF-CXCR4", "VIM-CD44", "B2M-LILRB1", "B2M-KLRD1", "HLA-B-KLRD1"),
  stringsAsFactors = FALSE
)

object_paths <- list(
  GSE161195 = file.path(root, "raw scrna data", "GSE161195_PRMM_program689_harmony_seurat.rds"),
  GSE161801 = file.path(root, "raw scrna data2", "GSE161801_paired_myeloma_program689_harmony_seurat.rds")
)

load_object_with_kmeans <- function(dataset) {
  obj <- readRDS(object_paths[[dataset]])
  if (dataset == "GSE161801" && "main_pre_post_pair" %in% colnames(obj@meta.data)) {
    obj <- subset(obj, subset = main_pre_post_pair == TRUE)
  }
  cells <- kmeans_cells[kmeans_cells$dataset == dataset, ]
  keep <- intersect(colnames(obj), cells$cell_id)
  obj <- subset(obj, cells = keep)
  rownames(cells) <- cells$cell_id
  cells <- cells[colnames(obj), ]
  obj$phase_display <- factor(cells$phase_display, levels = phase_levels)
  obj$kmeans_cluster <- factor(cells$kmeans_cluster, levels = cluster_levels)
  obj
}

average_features <- function(obj, features) {
  features <- intersect(unique(features), rownames(obj))
  mat <- GetAssayData(obj, assay = "RNA", layer = "data")
  groups <- paste(obj$phase_display, obj$kmeans_cluster, sep = "__")
  group_levels <- as.vector(outer(phase_levels, cluster_levels, paste, sep = "__"))
  avg <- matrix(NA_real_, nrow = length(features), ncol = length(group_levels), dimnames = list(features, group_levels))
  for (group in group_levels) {
    cells <- colnames(obj)[groups == group]
    if (length(cells) == 0) next
    avg[, group] <- Matrix::rowMeans(mat[features, cells, drop = FALSE])
  }
  avg
}

delta_from_avg <- function(avg) {
  rows <- list()
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(avg) || !drug %in% colnames(avg)) next
    rows[[cluster]] <- data.frame(
      cluster = cluster,
      gene = rownames(avg),
      delta = avg[, drug] - avg[, ctrl],
      stringsAsFactors = FALSE
    )
  }
  bind_rows(rows)
}

objs <- lapply(names(object_paths), load_object_with_kmeans)
names(objs) <- names(object_paths)

marker_gene_order <- unique(unlist(marker_sets, use.names = FALSE))
gene_program <- stack(marker_sets)
colnames(gene_program) <- c("gene", "marker_group")
marker_rows <- list()
for (dataset in names(objs)) {
  avg <- average_features(objs[[dataset]], marker_gene_order)
  marker_rows[[dataset]] <- delta_from_avg(avg) |>
    mutate(dataset = dataset)
}
marker_delta <- bind_rows(marker_rows) |>
  left_join(gene_program, by = "gene") |>
  group_by(cluster, gene, marker_group) |>
  summarise(delta = mean(delta, na.rm = TRUE), .groups = "drop")
write.csv(marker_delta, file.path(out_dir, "figureS8_D_marker_gene_delta.csv"), row.names = FALSE)

collectri <- get_collectri(organism = "human", split_complexes = FALSE)
tf_rows <- list()
for (dataset in names(objs)) {
  avg <- average_features(objs[[dataset]], unique(collectri$target))
  avg[is.na(avg)] <- 0
  res <- run_ulm(mat = as.matrix(avg), net = collectri, .source = "source", .target = "target", .mor = "mor", minsize = 5)
  wide <- reshape(as.data.frame(res[, c("source", "condition", "score")]), idvar = "source", timevar = "condition", direction = "wide")
  names(wide) <- sub("score.", "", names(wide), fixed = TRUE)
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(wide) || !drug %in% colnames(wide)) next
    tf_rows[[length(tf_rows) + 1]] <- data.frame(
      dataset = dataset,
      cluster = cluster,
      TF = wide$source,
      activity_delta = wide[[drug]] - wide[[ctrl]],
      stringsAsFactors = FALSE
    )
  }
}
tf_delta <- bind_rows(tf_rows) |>
  filter(TF %in% selected_tfs) |>
  group_by(cluster, TF) |>
  summarise(activity_delta = mean(activity_delta, na.rm = TRUE), .groups = "drop")
write.csv(tf_delta, file.path(out_dir, "figureS8_E_tf_activity_delta.csv"), row.names = FALSE)

gsea_rows <- list()
for (dataset in names(objs)) {
  avg <- average_features(objs[[dataset]], rownames(objs[[dataset]]))
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(avg) || !drug %in% colnames(avg)) next
    stats <- avg[, drug] - avg[, ctrl]
    names(stats) <- rownames(avg)
    stats <- stats[is.finite(stats)]
    stats <- sort(stats, decreasing = TRUE)
    fg <- suppressWarnings(fgsea(custom_gene_sets, stats, minSize = 5, maxSize = 500, eps = 0))
    fg$dataset <- dataset
    fg$cluster <- cluster
    gsea_rows[[length(gsea_rows) + 1]] <- as.data.frame(fg)
  }
}
gsea_delta <- bind_rows(gsea_rows)
if ("leadingEdge" %in% colnames(gsea_delta)) {
  gsea_delta$leadingEdge <- vapply(gsea_delta$leadingEdge, function(x) paste(x, collapse = "|"), character(1))
}
gsea_delta$pathway_label <- unname(pathway_labels[gsea_delta$pathway])
gsea_pooled <- gsea_delta |>
  group_by(cluster, pathway, pathway_label) |>
  summarise(NES = mean(NES, na.rm = TRUE), padj = median(padj, na.rm = TRUE), .groups = "drop") |>
  mutate(neg_log10_fdr = pmin(-log10(pmax(padj, 1e-12)), 12))
write.csv(gsea_delta, file.path(out_dir, "figureS8_F_gsea_by_dataset.csv"), row.names = FALSE)
write.csv(gsea_pooled, file.path(out_dir, "figureS8_F_gsea_pooled.csv"), row.names = FALSE)

progeny <- get_progeny(organism = "human", top = 500)
progeny_rows <- list()
for (dataset in names(objs)) {
  avg <- average_features(objs[[dataset]], unique(progeny$target))
  avg[is.na(avg)] <- 0
  res <- run_ulm(mat = as.matrix(avg), net = progeny, .source = "source", .target = "target", .mor = "weight", minsize = 5)
  wide <- reshape(as.data.frame(res[, c("source", "condition", "score")]), idvar = "source", timevar = "condition", direction = "wide")
  names(wide) <- sub("score.", "", names(wide), fixed = TRUE)
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(wide) || !drug %in% colnames(wide)) next
    progeny_rows[[length(progeny_rows) + 1]] <- data.frame(
      dataset = dataset,
      cluster = cluster,
      pathway = wide$source,
      activity_delta = wide[[drug]] - wide[[ctrl]],
      stringsAsFactors = FALSE
    )
  }
}
progeny_delta <- bind_rows(progeny_rows) |>
  filter(pathway %in% selected_progeny) |>
  group_by(cluster, pathway) |>
  summarise(activity_delta = mean(activity_delta, na.rm = TRUE), .groups = "drop") |>
  mutate(pathway_label = unname(progeny_labels[pathway]))
write.csv(progeny_delta, file.path(out_dir, "figureS8_G_progeny_activity_delta.csv"), row.names = FALSE)

lr_features <- unique(c(selected_lr$ligand, selected_lr$receptor))
lr_rows <- list()
for (dataset in names(objs)) {
  avg <- average_features(objs[[dataset]], lr_features)
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(avg) || !drug %in% colnames(avg)) next
    for (i in seq_len(nrow(selected_lr))) {
      lig <- selected_lr$ligand[i]
      rec <- selected_lr$receptor[i]
      if (!lig %in% rownames(avg) || !rec %in% rownames(avg)) next
      ctrl_score <- avg[lig, ctrl] * avg[rec, ctrl]
      drug_score <- avg[lig, drug] * avg[rec, drug]
      lr_rows[[length(lr_rows) + 1]] <- data.frame(
        dataset = dataset,
        cluster = cluster,
        pair = selected_lr$pair[i],
        pair_label = selected_lr$label[i],
        delta = drug_score - ctrl_score,
        stringsAsFactors = FALSE
      )
    }
  }
}
lr_delta <- bind_rows(lr_rows) |>
  group_by(cluster, pair, pair_label) |>
  summarise(delta = mean(delta, na.rm = TRUE), .groups = "drop") |>
  mutate(plot_delta = sign(delta) * log1p(abs(delta)))
write.csv(lr_delta, file.path(out_dir, "figureS8_H_ligand_receptor_delta.csv"), row.names = FALSE)

message("Wrote K-means downstream source tables to ", out_dir)
