#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(decoupleR)
  library(OmnipathR)
  library(fgsea)
})

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- "/tmp/prism_review1_test/scrna_kmeans_baseline"
panel_dir <- file.path(root, "supplementary_panels")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(panel_dir, recursive = TRUE, showWarnings = FALSE)

kmeans_file <- file.path(out_dir, "figureS8_scrna_kmeans_cells.csv")
if (!file.exists(kmeans_file)) {
  stop("Run codes/revised/extract_figureS8_scrna_kmeans_source.R before this script.")
}

cluster_levels <- paste0("Cluster ", 1:10)
cluster_short <- paste0("C", 1:10)
phase_levels <- c("Ctrl", "Drug")

cluster_colors <- c(
  "Cluster 1" = "#2F6690",
  "Cluster 2" = "#B93A32",
  "Cluster 3" = "#0F766E",
  "Cluster 4" = "#7A5AF8",
  "Cluster 5" = "#6D7986",
  "Cluster 6" = "#C47A2C",
  "Cluster 7" = "#4D7C3A",
  "Cluster 8" = "#C11574",
  "Cluster 9" = "#7C6D5F",
  "Cluster 10" = "#D6A21F"
)
phase_colors <- c("Ctrl" = "#2F6690", "Drug" = "#B93A32")

marker_sets <- list(
  "Program 6" = c("TMEM156", "CTSF", "IDUA", "UGT8", "LAMP3", "PRDM1", "CD38"),
  "Program 8" = c("CD44", "RGS2", "TYMP", "CD74", "HLA-DQB1", "STAT1", "BIRC3", "NFKBIA"),
  "Program 9" = c("DUSP1", "CDKN1C", "KLRD1", "FOS", "JUNB", "ZFP36", "IL32")
)
marker_gene_order <- unique(unlist(marker_sets, use.names = FALSE))
gene_program <- stack(marker_sets)
colnames(gene_program) <- c("gene", "marker_program")

custom_gene_sets <- list(
  MYELOMA_PLASMA_CELL_IDENTITY = c("SDC1", "XBP1", "PRDM1", "IRF4", "MZB1", "JCHAIN", "TNFRSF17", "SLAMF7", "CD38", "CD79A", "DERL3"),
  MYELOMA_NFKB_SURVIVAL = c("NFKB1", "NFKB2", "RELA", "REL", "NFKBIA", "TNFAIP3", "BIRC3", "TRAF1", "CD40", "LTBR", "MAP3K8"),
  TUMOR_CELL_CYCLE_PROLIFERATION = c("MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "CCNB1", "CDK1", "BUB1", "AURKA"),
  TUMOR_INTERFERON_RESPONSE = c("STAT1", "IRF1", "IRF7", "ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "OAS2", "CXCL10", "PSMB8", "TAP1"),
  TUMOR_ADHESION_MIGRATION = c("CD44", "ITGA4", "ITGB1", "ITGA5", "CXCR4", "VCAM1", "LGALS3", "FN1", "ZEB2", "CD36"),
  TUMOR_STRESS_APOPTOSIS = c("DDIT3", "ATF4", "ATF3", "PMAIP1", "BBC3", "BAX", "BCL2L11", "GADD45A", "HMOX1", "DUSP1"),
  TUMOR_DORMANCY_STEMLIKE = c("CDKN1C", "DUSP1", "KLF4", "ZEB2", "ALDH1A1", "ALDH1L1", "GAS6", "YAP1", "NTRK2", "AREG")
)
pathway_labels <- c(
  "TUMOR_CELL_CYCLE_PROLIFERATION" = "Cell-cycle arrest",
  "TUMOR_DORMANCY_STEMLIKE" = "Dormancy-like state",
  "MYELOMA_NFKB_SURVIVAL" = "NF-kB survival",
  "TUMOR_STRESS_APOPTOSIS" = "Stress response",
  "MYELOMA_PLASMA_CELL_IDENTITY" = "Plasma-cell identity",
  "TUMOR_ADHESION_MIGRATION" = "Adhesion/migration",
  "TUMOR_INTERFERON_RESPONSE" = "Interferon response"
)
selected_pathways <- names(pathway_labels)

selected_tfs <- c(
  "TFAM", "BCL11A", "HIVEP2", "CIITA", "NFKB2",
  "REL", "HBP1", "NFYB", "ATF1", "ELK4", "DMTF1", "ZBTB14"
)
selected_progeny <- c("TNFa", "NFkB", "p53", "JAK-STAT", "Hypoxia", "TGFb")
progeny_labels <- c(
  "TNFa" = "TNFa",
  "NFkB" = "NF-kB",
  "p53" = "p53 stress",
  "JAK-STAT" = "JAK-STAT",
  "Hypoxia" = "Hypoxia",
  "TGFb" = "TGFb"
)
selected_lr <- data.frame(
  pair = c("MIF->CD74", "B2M->HLA-F", "TIMP1->CD63", "MIF->CXCR4", "VIM->CD44", "B2M->LILRB1", "B2M->KLRD1", "HLA-B->KLRD1"),
  ligand = c("MIF", "B2M", "TIMP1", "MIF", "VIM", "B2M", "B2M", "HLA-B"),
  receptor = c("CD74", "HLA-F", "CD63", "CXCR4", "CD44", "LILRB1", "KLRD1", "KLRD1"),
  label = c("MIF-CD74", "B2M-HLA-F", "TIMP1-CD63", "MIF-CXCR4", "VIM-CD44", "B2M-LILRB1", "B2M-KLRD1", "HLA-B-KLRD1"),
  stringsAsFactors = FALSE
)

theme_fig6 <- function(base_size = 8) {
  theme_classic(base_family = "Helvetica", base_size = base_size) +
    theme(
      text = element_text(color = "black"),
      axis.text = element_text(size = base_size, color = "black"),
      axis.title = element_text(size = base_size + 1, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = 0.25),
      axis.line = element_line(color = "black", linewidth = 0.3),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size, color = "black"),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size + 1, color = "black"),
      plot.margin = margin(5, 5, 5, 5),
      plot.tag = element_text(size = 14, face = "bold", color = "black")
    )
}

theme_umap <- function() {
  theme_classic(base_family = "Helvetica", base_size = 8) +
    theme(
      text = element_text(color = "black"),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      axis.title = element_text(size = 9, color = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 7.5, color = "black"),
      legend.key.height = unit(0.14, "in"),
      legend.key.width = unit(0.14, "in"),
      strip.background = element_blank(),
      strip.text = element_text(size = 9, color = "black"),
      plot.margin = margin(4, 4, 4, 4),
      plot.tag = element_text(size = 14, face = "bold", color = "black")
    )
}

scale_by_group <- function(x, limit = 2) {
  if (length(unique(x[is.finite(x)])) < 2) return(rep(0, length(x)))
  pmax(pmin(as.numeric(scale(x)), limit), -limit)
}

kmeans_cells <- read.csv(kmeans_file, stringsAsFactors = FALSE) |>
  mutate(
    kmeans_cluster = factor(kmeans_cluster, levels = cluster_levels),
    cluster_short = factor(paste0("C", sub("^Cluster ", "", kmeans_cluster)), levels = cluster_short),
    phase_display = factor(phase_display, levels = phase_levels)
  )

object_paths <- list(
  GSE161195 = file.path(root, "raw scrna data", "GSE161195_PRMM_program689_harmony_seurat.rds"),
  GSE161801 = file.path(root, "raw scrna data2", "GSE161801_paired_myeloma_program689_harmony_seurat.rds")
)

load_object <- function(dataset) {
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
  obj$cluster_short <- factor(cells$cluster_short, levels = cluster_short)
  obj
}

objects <- lapply(names(object_paths), load_object)
names(objects) <- names(object_paths)

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

marker_summary_one <- function(obj, dataset) {
  genes <- intersect(marker_gene_order, rownames(obj))
  expr <- FetchData(obj, vars = genes, layer = "data")
  expr$phase_display <- obj$phase_display
  expr$kmeans_cluster <- obj$kmeans_cluster
  expr |>
    as_tibble() |>
    pivot_longer(cols = all_of(genes), names_to = "gene", values_to = "expr") |>
    group_by(phase_display, kmeans_cluster, gene) |>
    summarise(avg_expr = mean(expr, na.rm = TRUE), pct_expr = mean(expr > 0, na.rm = TRUE), .groups = "drop") |>
    mutate(dataset = dataset)
}

marker_phase <- bind_rows(lapply(names(objects), function(dataset) marker_summary_one(objects[[dataset]], dataset)))
marker_wide <- marker_phase |>
  pivot_wider(
    id_cols = c(dataset, kmeans_cluster, gene),
    names_from = phase_display,
    values_from = c(avg_expr, pct_expr)
  ) |>
  mutate(
    avg_delta = avg_expr_Drug - avg_expr_Ctrl,
    pct_expr = (pct_expr_Drug + pct_expr_Ctrl) / 2
  )
marker_plot <- marker_wide |>
  left_join(gene_program, by = "gene") |>
  group_by(kmeans_cluster, gene, marker_program) |>
  summarise(
    avg_delta = mean(avg_delta, na.rm = TRUE),
    pct_expr = mean(pct_expr, na.rm = TRUE),
    .groups = "drop"
  ) |>
  group_by(gene) |>
  mutate(delta_scaled = scale_by_group(avg_delta, 2.4)) |>
  ungroup()
marker_plot$gene <- factor(marker_plot$gene, levels = marker_gene_order)
marker_plot$marker_program <- factor(marker_plot$marker_program, levels = names(marker_sets))
marker_plot$cluster_short <- factor(paste0("C", sub("^Cluster ", "", marker_plot$kmeans_cluster)), levels = cluster_short)
write.csv(marker_plot, file.path(out_dir, "figureS8_D_marker_gene_dotplot_figure6style.csv"), row.names = FALSE)

collectri <- get_collectri(organism = "human", split_complexes = FALSE)
tf_rows <- list()
for (dataset in names(objects)) {
  avg <- average_features(objects[[dataset]], unique(collectri$target))
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
      kmeans_cluster = cluster,
      TF = wide$source,
      activity_delta = wide[[drug]] - wide[[ctrl]],
      stringsAsFactors = FALSE
    )
  }
}
tf_plot <- bind_rows(tf_rows) |>
  filter(TF %in% selected_tfs) |>
  group_by(kmeans_cluster, TF) |>
  summarise(activity_delta = mean(activity_delta, na.rm = TRUE), .groups = "drop")
tf_plot$TF <- factor(tf_plot$TF, levels = rev(selected_tfs))
tf_plot$cluster_short <- factor(paste0("C", sub("^Cluster ", "", tf_plot$kmeans_cluster)), levels = cluster_short)
write.csv(tf_plot, file.path(out_dir, "figureS8_E_tf_activity_figure6style.csv"), row.names = FALSE)

gsea_rows <- list()
for (dataset in names(objects)) {
  avg <- average_features(objects[[dataset]], rownames(objects[[dataset]]))
  for (cluster in cluster_levels) {
    ctrl <- paste("Ctrl", cluster, sep = "__")
    drug <- paste("Drug", cluster, sep = "__")
    if (!ctrl %in% colnames(avg) || !drug %in% colnames(avg)) next
    stats <- avg[, drug] - avg[, ctrl]
    names(stats) <- rownames(avg)
    stats <- stats[is.finite(stats)]
    stats <- sort(stats, decreasing = TRUE)
    fg <- suppressWarnings(fgsea(custom_gene_sets[selected_pathways], stats, minSize = 5, maxSize = 500, eps = 0))
    fg$dataset <- dataset
    fg$kmeans_cluster <- cluster
    gsea_rows[[length(gsea_rows) + 1]] <- as.data.frame(fg)
  }
}
gsea_plot <- bind_rows(gsea_rows)
if ("leadingEdge" %in% colnames(gsea_plot)) {
  gsea_plot$leadingEdge <- vapply(gsea_plot$leadingEdge, function(x) paste(x, collapse = "|"), character(1))
}
gsea_plot <- gsea_plot |>
  mutate(pathway_label = unname(pathway_labels[pathway])) |>
  group_by(kmeans_cluster, pathway, pathway_label) |>
  summarise(NES = mean(NES, na.rm = TRUE), padj = median(padj, na.rm = TRUE), .groups = "drop") |>
  mutate(neg_log10_fdr = pmin(-log10(pmax(padj, 1e-12)), 12))
gsea_plot$pathway_label <- factor(gsea_plot$pathway_label, levels = rev(unname(pathway_labels[selected_pathways])))
gsea_plot$cluster_short <- factor(paste0("C", sub("^Cluster ", "", gsea_plot$kmeans_cluster)), levels = cluster_short)
write.csv(gsea_plot, file.path(out_dir, "figureS8_F_gsea_figure6style.csv"), row.names = FALSE)

progeny <- get_progeny(organism = "human", top = 500)
progeny_rows <- list()
for (dataset in names(objects)) {
  avg <- average_features(objects[[dataset]], unique(progeny$target))
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
      kmeans_cluster = cluster,
      pathway = wide$source,
      activity_delta = wide[[drug]] - wide[[ctrl]],
      stringsAsFactors = FALSE
    )
  }
}
progeny_plot <- bind_rows(progeny_rows) |>
  filter(pathway %in% selected_progeny) |>
  group_by(kmeans_cluster, pathway) |>
  summarise(activity_delta = mean(activity_delta, na.rm = TRUE), .groups = "drop") |>
  mutate(pathway_label = unname(progeny_labels[pathway])) |>
  group_by(pathway) |>
  mutate(activity_z = scale_by_group(activity_delta, 2)) |>
  ungroup()
progeny_plot$pathway_label <- factor(progeny_plot$pathway_label, levels = rev(unname(progeny_labels[selected_progeny])))
progeny_plot$cluster_short <- factor(paste0("C", sub("^Cluster ", "", progeny_plot$kmeans_cluster)), levels = cluster_short)
write.csv(progeny_plot, file.path(out_dir, "figureS8_G_progeny_activity_figure6style.csv"), row.names = FALSE)

lr_features <- unique(c(selected_lr$ligand, selected_lr$receptor))
lr_rows <- list()
for (dataset in names(objects)) {
  avg <- average_features(objects[[dataset]], lr_features)
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
        kmeans_cluster = cluster,
        pair = selected_lr$pair[i],
        pair_label = selected_lr$label[i],
        delta_vs_ctrl = drug_score - ctrl_score,
        stringsAsFactors = FALSE
      )
    }
  }
}
lr_plot <- bind_rows(lr_rows) |>
  group_by(kmeans_cluster, pair, pair_label) |>
  summarise(delta_vs_ctrl = mean(delta_vs_ctrl, na.rm = TRUE), .groups = "drop") |>
  mutate(plot_delta = sign(delta_vs_ctrl) * log1p(abs(delta_vs_ctrl)))
lr_plot$pair_label <- factor(lr_plot$pair_label, levels = rev(selected_lr$label))
lr_plot$cluster_short <- factor(paste0("C", sub("^Cluster ", "", lr_plot$kmeans_cluster)), levels = cluster_short)
write.csv(lr_plot, file.path(out_dir, "figureS8_H_ligand_receptor_figure6style.csv"), row.names = FALSE)

comp_dataset <- kmeans_cells |>
  count(dataset, phase_display, kmeans_cluster, cluster_short, name = "n_cells") |>
  group_by(dataset, phase_display) |>
  mutate(phase_cells = sum(n_cells), fraction = n_cells / phase_cells) |>
  ungroup()
comp_plot <- comp_dataset |>
  group_by(phase_display, kmeans_cluster, cluster_short) |>
  summarise(fraction = mean(fraction, na.rm = TRUE), .groups = "drop") |>
  mutate(percent = 100 * fraction)
write.csv(comp_plot, file.path(out_dir, "figureS8_C_kmeans_phase_composition_figure6style.csv"), row.names = FALSE)

pA <- ggplot(kmeans_cells[kmeans_cells$dataset == "GSE161195", ], aes(UMAP1, UMAP2, color = kmeans_cluster)) +
  geom_point(size = 0.22, alpha = 0.58, stroke = 0) +
  scale_color_manual(values = cluster_colors, breaks = cluster_levels, labels = cluster_short, drop = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1), nrow = 2)) +
  labs(x = "UMAP1", y = "UMAP2", title = "GSE161195") +
  theme_umap() +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0))

pB <- ggplot(kmeans_cells[kmeans_cells$dataset == "GSE161801", ], aes(UMAP1, UMAP2, color = kmeans_cluster)) +
  geom_point(size = 0.22, alpha = 0.58, stroke = 0) +
  scale_color_manual(values = cluster_colors, breaks = cluster_levels, labels = cluster_short, drop = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1), nrow = 2)) +
  labs(x = "UMAP1", y = "UMAP2", title = "GSE161801") +
  theme_umap() +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0))

pC <- ggplot(comp_plot, aes(x = cluster_short, y = percent, fill = phase_display)) +
  geom_col(position = position_dodge(width = 0.70), width = 0.62, color = NA) +
  scale_fill_manual(values = phase_colors, breaks = phase_levels) +
  labs(x = "K-means cluster", y = "Cell fraction (%)") +
  theme_fig6(8) +
  theme(legend.position = c(0.94, 0.82), legend.background = element_blank())

pD <- ggplot(marker_plot, aes(x = gene, y = cluster_short)) +
  geom_point(aes(size = pct_expr, color = delta_scaled), alpha = 0.92) +
  facet_grid(. ~ marker_program, scales = "free_x", space = "free_x") +
  scale_size_continuous(range = c(0.25, 3.6), limits = c(0, 1), breaks = c(0.25, 0.5, 0.75, 1)) +
  scale_color_gradient2(low = "#3B5BA7", mid = "#F7F7F7", high = "#B72C2C", midpoint = 0, limits = c(-2.4, 2.4)) +
  labs(x = NULL, y = NULL) +
  theme_fig6(7) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6.7),
    panel.spacing.x = unit(0.08, "in")
  )

pE <- ggplot(tf_plot, aes(x = cluster_short, y = TF, fill = activity_delta)) +
  geom_tile(color = "white", linewidth = 0.55, width = 0.92, height = 0.88) +
  scale_fill_gradient2(low = "#3B5BA7", mid = "#F7F7F7", high = "#B72C2C", midpoint = 0, limits = c(-3.5, 3.5), oob = scales::squish) +
  labs(x = "K-means cluster", y = NULL) +
  theme_fig6(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

pF <- ggplot(gsea_plot, aes(x = cluster_short, y = pathway_label)) +
  geom_point(aes(size = neg_log10_fdr, fill = NES), shape = 21, color = "black", stroke = 0.18, alpha = 0.94) +
  scale_fill_gradient2(low = "#3B5BA7", mid = "#F7F7F7", high = "#B72C2C", midpoint = 0, limits = c(-2.5, 2.5), oob = scales::squish) +
  scale_size_continuous(range = c(0.5, 4.3), breaks = c(2, 4, 8, 12)) +
  labs(x = "K-means cluster", y = NULL) +
  theme_fig6(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

pG <- ggplot(progeny_plot, aes(x = cluster_short, y = pathway_label, fill = activity_z)) +
  geom_tile(color = "white", linewidth = 0.55, width = 0.92, height = 0.88) +
  scale_fill_gradient2(low = "#3B5BA7", mid = "#F7F7F7", high = "#B72C2C", midpoint = 0, limits = c(-2, 2), oob = scales::squish) +
  labs(x = "K-means cluster", y = NULL) +
  theme_fig6(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

pH <- ggplot(lr_plot, aes(x = cluster_short, y = pair_label)) +
  geom_point(aes(size = pmax(delta_vs_ctrl, 0), fill = plot_delta), shape = 21, color = "black", stroke = 0.18, alpha = 0.92) +
  scale_size_continuous(range = c(0.25, 4.5)) +
  scale_fill_gradient2(low = "#3B5BA7", mid = "#F7F7F7", high = "#B72C2C", midpoint = 0) +
  labs(x = "K-means cluster", y = NULL) +
  theme_fig6(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

layout <- "
AABB
CCCC
DDDD
EEFF
GGHH
"
fig <- pA + pB + pC + pD + pE + pF + pG + pH +
  plot_layout(design = layout, heights = c(1.05, 0.58, 1.05, 0.95, 0.92), guides = "keep") +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(size = 14, face = "bold", color = "black"))

out_pdf <- file.path(panel_dir, "figureS8_scrna_kmeans_cluster_baseline.pdf")
ggsave(out_pdf, fig, width = 13.2, height = 15.2, units = "in", device = "pdf")
message(out_pdf)
