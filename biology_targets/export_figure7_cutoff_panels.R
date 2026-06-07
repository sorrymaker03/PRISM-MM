suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(ggplot2)
  library(msigdbr)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

root <- "bulk_pre_sc"
trial_dir <- file.path(root, "biology_interpretation", "gene_target_trial")
panel_dirs <- c(
  file.path(root, "main_figure_panels"),
  file.path(dirname(normalizePath(root)), "PRISM-MM", "figures", "main_figure_panels")
)
source_dir <- file.path(root, "main_figure_source_data")
dir.create(trial_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(source_dir, showWarnings = FALSE, recursive = TRUE)
for (d in panel_dirs) dir.create(d, showWarnings = FALSE, recursive = TRUE)

program_colors <- c(Program_6 = "#B93A32", Program_8 = "#0F766E", Program_9 = "#7A5AF8")
program_labels <- c(Program_6 = "Program 6", Program_8 = "Program 8", Program_9 = "Program 9")
dataset_labels <- c(GSE161195 = "GSE161195", GSE161801 = "GSE161801")

selected <- list(
  Program_6 = c("TMEM156", "CTSF", "IDUA", "UGT8", "LAMP3"),
  Program_8 = c("RGS2", "TYMP", "CD44"),
  Program_9 = c("KLRD1", "CDKN1C", "DUSP1")
)

hallmark_table <- msigdbr(species = "Homo sapiens", collection = "H")
hallmark_sets <- split(hallmark_table$gene_symbol, hallmark_table$gs_name)
hallmark_sets <- lapply(hallmark_sets, unique)

pathway_panels <- data.frame(
  panel = c("F", "G", "H", "I", "J", "K", "L", "M"),
  program = c("Program_6", "Program_6", "Program_6", "Program_8", "Program_8", "Program_8", "Program_9", "Program_9"),
  pathway = c(
    "HALLMARK_PROTEIN_SECRETION",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_APOPTOSIS"
  ),
  label = c(
    "Hallmark protein secretion",
    "Hallmark unfolded protein response",
    "Hallmark oxidative phosphorylation",
    "Hallmark TNF-alpha signaling via NF-kB",
    "Hallmark interferon gamma response",
    "Hallmark IL6-JAK-STAT3 signaling",
    "Hallmark TGF-beta signaling",
    "Hallmark apoptosis"
  ),
  stringsAsFactors = FALSE
)

files <- c(
  GSE161195 = "raw scrna data/GSE161195_PRMM_program689_harmony_seurat.rds",
  GSE161801 = "raw scrna data2/GSE161801_paired_myeloma_program689_harmony_seurat.rds"
)

top_high <- function(x, frac = 0.10) {
  if (!any(x > 0, na.rm = TRUE)) return(rep(FALSE, length(x)))
  if (mean(x > 0, na.rm = TRUE) <= frac) return(x > 0)
  rank(-x, ties.method = "first") <= ceiling(length(x) * frac)
}

log2_or <- function(a, b, c, d) {
  log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
}

wilcox_p <- function(x, y) {
  if (length(x) < 3 || length(y) < 3 || length(unique(c(x, y))) < 2) return(NA_real_)
  suppressWarnings(wilcox.test(x, y)$p.value)
}

score_signature <- function(expr, genes, exclude = character()) {
  use <- intersect(setdiff(genes, exclude), rownames(expr))
  if (length(use) < 3) return(rep(NA_real_, ncol(expr)))
  mat <- as.matrix(expr[use, , drop = FALSE])
  z <- t(scale(t(mat)))
  z[is.na(z)] <- 0
  colMeans(z)
}

rows <- list()
cell_rows <- list()
pathway_cell_rows <- list()
for (dataset in names(files)) {
  message("Loading ", dataset)
  obj <- readRDS(files[[dataset]])
  assay <- if ("RNA" %in% Assays(obj)) "RNA" else DefaultAssay(obj)
  expr <- GetAssayData(obj, assay = assay, layer = "data")
  meta <- obj@meta.data
  if (dataset == "GSE161195") {
    drug <- tolower(as.character(meta$treatment_phase)) %in% c("drug", "treated", "treatment")
  } else {
    drug <- tolower(as.character(meta$analysis_phase)) %in% c("relapse", "post", "drug", "treated") |
      tolower(as.character(meta$timepoint)) %in% c("post", "relapse")
  }
  pathway_cache <- lapply(unique(pathway_panels$pathway), function(pathway) score_signature(expr, hallmark_sets[[pathway]]))
  names(pathway_cache) <- unique(pathway_panels$pathway)

  for (program in names(selected)) {
    score_col <- paste0(program, "_z")
    if (!score_col %in% colnames(meta)) score_col <- paste0(program, "_signed_score")
    score <- meta[[score_col]]
    pnum <- sub("Program_", "", program)
    class_hit <- as.character(meta$program_class) %in% c(paste0("P", pnum), program, paste("Program", pnum))
    target_pathways <- pathway_panels[pathway_panels$program == program, , drop = FALSE]

    for (gene in selected[[program]]) {
      if (!gene %in% rownames(expr)) next
      x <- as.numeric(expr[gene, ])
      high <- top_high(x, 0.10)
      state <- ifelse(high, "high", "low")
      pathway_deltas <- lapply(seq_len(nrow(target_pathways)), function(i) {
        pathway <- target_pathways$pathway[i]
        pathway_score <- pathway_cache[[pathway]]
        data.frame(
          pathway = pathway,
          pathway_label = target_pathways$label[i],
          pathway_score_delta = mean(pathway_score[high], na.rm = TRUE) - mean(pathway_score[!high], na.rm = TRUE),
          pathway_score_p = wilcox_p(pathway_score[high], pathway_score[!high]),
          stringsAsFactors = FALSE
        )
      })
      pathway_delta_df <- do.call(rbind, pathway_deltas)
      rows[[length(rows) + 1]] <- data.frame(
        dataset = dataset,
        program = program,
        gene = gene,
        high_fraction = mean(high),
        expressed_fraction = mean(x > 0),
        program_score_delta = mean(score[high], na.rm = TRUE) - mean(score[!high], na.rm = TRUE),
        program_score_p = wilcox_p(score[high], score[!high]),
        log2or_program_class = log2_or(sum(high & class_hit), sum(high & !class_hit), sum(!high & class_hit), sum(!high & !class_hit)),
        log2or_drug = log2_or(sum(high & drug), sum(high & !drug), sum(!high & drug), sum(!high & !drug)),
        pathway_delta_summary = paste(pathway_delta_df$pathway, sprintf("%.3f", pathway_delta_df$pathway_score_delta), sep = ":", collapse = ";"),
        stringsAsFactors = FALSE
      )
      keep <- rep(TRUE, length(score))
      # Downsample the larger low group for plotting only; statistics above use all cells.
      low_idx <- which(!high)
      high_idx <- which(high)
      if (length(low_idx) > min(6000, length(high_idx) * 3)) {
        set.seed(7 + nchar(dataset) + nchar(gene))
        low_idx <- sample(low_idx, min(6000, max(length(high_idx) * 3, 1000)))
      }
      plot_idx <- c(high_idx, low_idx)
      cell_rows[[length(cell_rows) + 1]] <- data.frame(
        dataset = dataset,
        program = program,
        program_label = program_labels[[program]],
        gene = gene,
        state = factor(state[plot_idx], levels = c("low", "high")),
        program_score = score[plot_idx],
        stringsAsFactors = FALSE
      )
      for (i in seq_len(nrow(target_pathways))) {
        pathway <- target_pathways$pathway[i]
        pathway_score <- pathway_cache[[pathway]]
        pathway_cell_rows[[length(pathway_cell_rows) + 1]] <- data.frame(
          dataset = dataset,
          program = program,
          program_label = program_labels[[program]],
          gene = gene,
          state = factor(state[plot_idx], levels = c("low", "high")),
          pathway = pathway,
          pathway_label = target_pathways$label[i],
          pathway_score = pathway_score[plot_idx],
          stringsAsFactors = FALSE
        )
      }
    }
  }
  rm(obj, expr)
  gc()
}

df <- do.call(rbind, rows)
cell_df <- do.call(rbind, cell_rows)
pathway_cell_df <- do.call(rbind, pathway_cell_rows)
df$program_label <- program_labels[df$program]
df$gene <- factor(df$gene, levels = unlist(selected))
cell_df$gene <- factor(cell_df$gene, levels = unlist(selected))
pathway_cell_df$gene <- factor(pathway_cell_df$gene, levels = unlist(selected))
write.csv(df, file.path(trial_dir, "trial_single_gene_cutoff_state_scores.csv"), row.names = FALSE)
write.csv(df, file.path(source_dir, "figure7_EFG_single_gene_cutoff_state_scores.csv"), row.names = FALSE)
write.csv(cell_df, file.path(source_dir, "figure7_single_gene_cutoff_cell_level_plot_values.csv"), row.names = FALSE)
write.csv(pathway_cell_df, file.path(source_dir, "figure7_hallmark_single_gene_cutoff_cell_level_plot_values.csv"), row.names = FALSE)

theme_prism <- function(base_size = 8.5) {
  theme_classic(base_size = base_size) +
    theme(
      axis.text = element_text(color = "#111111"),
      axis.title = element_text(color = "#111111"),
      legend.title = element_blank(),
      legend.position = "top",
      strip.background = element_blank(),
      strip.text = element_text(color = "#111111", face = "bold"),
      panel.grid.major.y = element_line(color = "#DDE3EA", linewidth = 0.35),
      panel.grid.major.x = element_line(color = "#E8EEF5", linewidth = 0.35)
    )
}

save_pdf <- function(plot, filename, width, height) {
  for (d in panel_dirs) {
    ggsave(file.path(d, filename), plot, width = width, height = height, units = "in", device = "pdf")
  }
}

state_colors <- c(low = "#A3ADBB", high = "#2F6690")

panel_e <- ggplot(cell_df, aes(x = gene, y = program_score, fill = state)) +
  geom_boxplot(width = 0.62, outlier.shape = NA, linewidth = 0.28, position = position_dodge(width = 0.72)) +
  stat_summary(fun = median, geom = "point", aes(group = state), position = position_dodge(width = 0.72), size = 0.65, color = "#111111") +
  facet_grid(program_label ~ ., scales = "free_x", space = "free_x") +
  scale_fill_manual(values = state_colors, labels = c("low", "high")) +
  coord_cartesian(ylim = c(-1.5, 3.5)) +
  labs(x = "", y = "Corresponding program score") +
  theme_prism() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), panel.grid.major.x = element_blank())
save_pdf(panel_e, "figure7_E_single_gene_cutoff_program_score.pdf", 6.2, 4.6)

plot_pathway <- function(pathway, label, filename, width = 3.4, height = 3.2, ylim = NULL) {
  sub <- pathway_cell_df[pathway_cell_df$pathway == pathway, , drop = FALSE]
  sub$gene <- factor(as.character(sub$gene), levels = selected[[unique(sub$program)]])
  if (is.null(ylim)) {
    q <- quantile(sub$pathway_score, probs = c(0.03, 0.97), na.rm = TRUE)
    pad <- max(0.08, diff(q) * 0.18)
    ylim <- c(q[1] - pad, q[2] + pad)
  }
  p <- ggplot(sub, aes(x = gene, y = pathway_score, fill = state)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, linewidth = 0.28, position = position_dodge(width = 0.72)) +
    stat_summary(fun = median, geom = "point", aes(group = state), position = position_dodge(width = 0.72), size = 0.65, color = "#111111") +
    scale_fill_manual(values = state_colors, labels = c("low", "high")) +
    coord_cartesian(ylim = ylim) +
    labs(x = "", y = label) +
    theme_prism() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), panel.grid.major.x = element_blank())
  save_pdf(p, filename, width, height)
}

for (i in seq_len(nrow(pathway_panels))) {
  row <- pathway_panels[i, ]
  safe_name <- tolower(gsub("^HALLMARK_", "", row$pathway))
  safe_name <- gsub("[^a-z0-9]+", "_", safe_name)
  plot_pathway(
    row$pathway,
    row$label,
    sprintf("figure7_%s_single_gene_cutoff_%s.pdf", row$panel, safe_name),
    width = ifelse(row$program == "Program_6", 4.0, 3.4),
    height = 3.2
  )
}

message("Saved single-gene cutoff panels and source data.")
