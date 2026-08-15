#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Seurat)
})

root <- normalizePath(getwd(), mustWork = TRUE)
out_dir <- "/tmp/prism_review1_test/scrna_kmeans_baseline"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

k_clusters <- 10
set.seed(20260806)

objects <- list(
  GSE161195 = file.path(root, "raw scrna data", "GSE161195_PRMM_program689_harmony_seurat.rds"),
  GSE161801 = file.path(root, "raw scrna data2", "GSE161801_paired_myeloma_program689_harmony_seurat.rds")
)

extract_dataset <- function(dataset, path) {
  message("Loading ", dataset)
  obj <- readRDS(path)
  if (dataset == "GSE161801") {
    if ("main_pre_post_pair" %in% colnames(obj@meta.data)) {
      obj <- subset(obj, subset = main_pre_post_pair == TRUE)
    }
    phase <- ifelse(obj$analysis_phase == "baseline", "Ctrl", "Drug")
    patient <- obj$patient
  } else {
    phase <- ifelse(obj$treatment_phase == "ctrl", "Ctrl", "Drug")
    patient <- obj$patient_id
  }

  harmony <- Embeddings(obj, "harmony")
  dims <- seq_len(min(30, ncol(harmony)))
  km <- kmeans(harmony[, dims, drop = FALSE], centers = k_clusters, nstart = 30, iter.max = 100)

  umap <- as.data.frame(Embeddings(obj, "umap"))
  colnames(umap) <- c("UMAP1", "UMAP2")
  umap$cell_id <- rownames(umap)
  umap$dataset <- dataset
  umap$patient_id <- as.character(patient)
  umap$phase_display <- factor(phase, levels = c("Ctrl", "Drug"))
  umap$kmeans_cluster_raw <- paste0("Cluster ", km$cluster)

  # Keep labels stable and readable by ordering clusters by decreasing size.
  cluster_size <- sort(table(umap$kmeans_cluster_raw), decreasing = TRUE)
  rename <- setNames(paste0("Cluster ", seq_along(cluster_size)), names(cluster_size))
  umap$kmeans_cluster <- factor(rename[umap$kmeans_cluster_raw], levels = paste0("Cluster ", seq_len(k_clusters)))
  umap
}

all_cells <- do.call(
  rbind,
  lapply(names(objects), function(dataset) extract_dataset(dataset, objects[[dataset]]))
)

out_file <- file.path(out_dir, "figureS8_scrna_kmeans_cells.csv")
write.csv(all_cells, out_file, row.names = FALSE)
message(out_file)
