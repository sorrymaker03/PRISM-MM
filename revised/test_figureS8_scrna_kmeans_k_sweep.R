#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
})

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- "/tmp/prism_review1_test/scrna_kmeans_baseline"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20260806)

objects <- list(
  GSE161195 = file.path(root, "raw scrna data", "GSE161195_PRMM_program689_harmony_seurat.rds"),
  GSE161801 = file.path(root, "raw scrna data2", "GSE161801_paired_myeloma_program689_harmony_seurat.rds")
)

load_harmony <- function(dataset, path) {
  obj <- readRDS(path)
  if (dataset == "GSE161801") {
    if ("main_pre_post_pair" %in% colnames(obj@meta.data)) {
      obj <- subset(obj, subset = main_pre_post_pair == TRUE)
    }
    phase <- ifelse(obj$analysis_phase == "baseline", "Ctrl", "Drug")
  } else {
    phase <- ifelse(obj$treatment_phase == "ctrl", "Ctrl", "Drug")
  }
  harmony <- Embeddings(obj, "harmony")
  dims <- seq_len(min(30, ncol(harmony)))
  list(x = harmony[, dims, drop = FALSE], phase = factor(phase, levels = c("Ctrl", "Drug")))
}

datasets <- lapply(names(objects), function(dataset) load_harmony(dataset, objects[[dataset]]))
names(datasets) <- names(objects)

rows <- list()
for (dataset in names(datasets)) {
  x <- datasets[[dataset]]$x
  phase <- datasets[[dataset]]$phase
  for (k in 3:10) {
    km <- kmeans(x, centers = k, nstart = 30, iter.max = 100)
    raw <- paste0("Cluster ", km$cluster)
    cluster_size <- sort(table(raw), decreasing = TRUE)
    rename <- setNames(paste0("Cluster ", seq_along(cluster_size)), names(cluster_size))
    cluster <- factor(rename[raw], levels = paste0("Cluster ", seq_len(k)))
    tab <- table(phase, cluster)
    prop <- prop.table(tab, margin = 1)
    p_value <- suppressWarnings(chisq.test(tab)$p.value)
    delta <- prop["Drug", ] - prop["Ctrl", ]
    rows[[length(rows) + 1]] <- data.frame(
      dataset = dataset,
      k = k,
      chi_square_p = p_value,
      max_abs_delta = max(abs(delta)),
      n_changed_2pct = sum(abs(delta) >= 0.02),
      n_changed_5pct = sum(abs(delta) >= 0.05),
      stringsAsFactors = FALSE
    )
  }
}

summary <- do.call(rbind, rows)
write.csv(summary, file.path(out_dir, "figureS8_kmeans_k_sweep_summary.csv"), row.names = FALSE)
print(summary)
