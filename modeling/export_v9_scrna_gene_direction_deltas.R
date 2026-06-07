#!/usr/bin/env Rscript
# Export paired scRNA drug-minus-control deltas for every v9 signature gene.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(script_arg) > 0) {
  normalizePath(sub("^--file=", "", script_arg[1]))
} else {
  normalizePath("bulk_pre_sc/export_v9_scrna_gene_direction_deltas.R")
}
root <- normalizePath(file.path(dirname(script_path), ".."))
model_dir <- file.path(root, "bulk_pre_sc", "model_upgrade_v9_final")
outdir <- file.path(model_dir, "gene_direction_validation")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

prog <- read.csv(
  file.path(model_dir, "our_anchored_sparse_attention_v9_core_program_genes.csv"),
  stringsAsFactors = FALSE
)

obj <- readRDS(file.path(root, "raw scrna data", "processed", "GSE161195_PRMM_seurat.rds"))
DefaultAssay(obj) <- "RNA"
layers <- tryCatch(Layers(obj[["RNA"]]), error = function(e) character())
expr <- NULL
if ("data" %in% layers) expr <- GetAssayData(obj, assay = "RNA", layer = "data")
if (is.null(expr) || length(expr@x) == 0) {
  obj <- NormalizeData(obj, verbose = FALSE)
  expr <- GetAssayData(obj, assay = "RNA", layer = "data")
}

meta <- obj@meta.data[colnames(expr), , drop = FALSE]
keep <- meta$nFeature_RNA >= 200 &
  meta$nCount_RNA >= 500 &
  meta$treatment_phase %in% c("ctrl", "drug_treatment") &
  !is.na(meta$matched_pair_index)
meta <- meta[keep, , drop = FALSE]

export_scope <- function(scope, top_n = NULL) {
  signature <- prog
  if (!is.null(top_n)) signature <- signature[signature$rank <= top_n, ]
  genes <- intersect(unique(signature$gene), rownames(expr))
  pair_ids <- unique(meta$matched_pair_index)
  deltas <- list()
  for (pair in pair_ids) {
    ctrl_cells <- rownames(meta)[meta$matched_pair_index == pair & meta$treatment_phase == "ctrl"]
    drug_cells <- rownames(meta)[meta$matched_pair_index == pair & meta$treatment_phase == "drug_treatment"]
    if (length(ctrl_cells) >= 50 && length(drug_cells) >= 50) {
      deltas[[as.character(pair)]] <- Matrix::rowMeans(expr[genes, drug_cells, drop = FALSE]) -
        Matrix::rowMeans(expr[genes, ctrl_cells, drop = FALSE])
    }
  }
  delta_mat <- do.call(cbind, deltas)
  colnames(delta_mat) <- names(deltas)
  long <- data.frame(
    gene = rep(rownames(delta_mat), times = ncol(delta_mat)),
    matched_pair_index = rep(colnames(delta_mat), each = nrow(delta_mat)),
    delta = as.numeric(delta_mat),
    stringsAsFactors = FALSE
  )
  summary <- data.frame(
    gene = rownames(delta_mat),
    mean_delta = as.numeric(rowMeans(delta_mat, na.rm = TRUE)),
    median_delta = as.numeric(apply(delta_mat, 1, median, na.rm = TRUE)),
    frac_increase = as.numeric(rowMeans(delta_mat > 0, na.rm = TRUE)),
    n_pairs = ncol(delta_mat),
    stringsAsFactors = FALSE
  )
  write.csv(long, file.path(outdir, paste0("scrna_", scope, "_gene_delta_by_pair.csv")), row.names = FALSE)
  write.csv(summary, file.path(outdir, paste0("scrna_", scope, "_gene_delta_summary.csv")), row.names = FALSE)
  cat(sprintf("%s: genes=%d pairs=%d\n", scope, nrow(delta_mat), ncol(delta_mat)))
}

export_scope("top50", 50)
export_scope("all_adaptive")
