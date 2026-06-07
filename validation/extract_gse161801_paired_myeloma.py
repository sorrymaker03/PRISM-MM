#!/usr/bin/env python3
"""Extract validation-ready paired malignant scRNA data from GSE161801.

The GEO supplementary matrices are dense CSV files split by orig.ident. This
script scans them in gene-row chunks to avoid loading a large sample into
memory. It exports:
- malignant-cell metadata for patients with paired pre/post observations;
- patient-level pre/post and optional post_2 pair manifests;
- all-gene patient-timepoint pseudobulk counts and log1p CPM;
- a cell-level sparse count matrix restricted to v14 program genes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw scrna data2/GSE161801_RAW")
    parser.add_argument("--metadata", default="raw scrna data2/GSE161801_K43R_metadata_table.csv")
    parser.add_argument("--outdir", default="raw scrna data2/processed")
    parser.add_argument(
        "--program-genes",
        default="bulk_pre_sc/model_upgrade_v14_multimodal_dictionary/balanced/our_multimodal_dictionary_v14_core_program_genes.csv",
    )
    parser.add_argument("--chunksize", type=int, default=512)
    return parser.parse_args()


def file_orig_ident(path: Path) -> str:
    return path.name.removesuffix(".csv.gz").split("_K43R_", 1)[1]


def group_name(patient: str, timepoint: str) -> str:
    return f"{patient}__{timepoint}"


def paired_patients(metadata: pd.DataFrame) -> list[str]:
    counts = metadata.groupby(["patient", "timepoint"]).size().unstack(fill_value=0)
    return sorted(counts.index[(counts.get("pre", 0) > 0) & (counts.get("post", 0) > 0)].astype(str))


def cell_metadata(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(args.metadata, dtype=str)
    required = {
        "Cell_barcode",
        "orig.ident",
        "sample_id",
        "patient",
        "PID_new",
        "PID_sample_new",
        "timepoint",
        "sorting",
        "major_celltype",
        "celltype_1",
        "major_population",
        "tumor_clone",
        "sc_gain_1q",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    malignant = metadata[metadata["celltype_1"].eq("Myeloma")].copy()
    patients = paired_patients(malignant)
    malignant = malignant[malignant["patient"].isin(patients)].copy()
    malignant["patient_timepoint"] = [
        group_name(patient, timepoint) for patient, timepoint in malignant[["patient", "timepoint"]].itertuples(index=False)
    ]
    malignant["analysis_phase"] = malignant["timepoint"].map(
        {"pre": "baseline", "post": "relapse", "post_2": "later_relapse"}
    )
    malignant["main_pre_post_pair"] = malignant["timepoint"].isin(["pre", "post"])
    malignant["longitudinal_extension"] = malignant["timepoint"].isin(["post_2"])
    return malignant, patients


def pair_manifests(metadata: pd.DataFrame, patients: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = metadata.groupby(["patient", "timepoint"]).size().unstack(fill_value=0)
    main_rows = []
    extension_rows = []
    for index, patient in enumerate(patients, start=1):
        main_rows.append(
            {
                "pair_index": index,
                "patient": patient,
                "baseline_group": group_name(patient, "pre"),
                "relapse_group": group_name(patient, "post"),
                "n_baseline_cells": int(counts.loc[patient].get("pre", 0)),
                "n_relapse_cells": int(counts.loc[patient].get("post", 0)),
            }
        )
        if int(counts.loc[patient].get("post_2", 0)) > 0:
            extension_rows.append(
                {
                    "pair_index": index,
                    "patient": patient,
                    "baseline_group": group_name(patient, "pre"),
                    "later_relapse_group": group_name(patient, "post_2"),
                    "n_baseline_cells": int(counts.loc[patient].get("pre", 0)),
                    "n_later_relapse_cells": int(counts.loc[patient].get("post_2", 0)),
                }
            )
    return pd.DataFrame(main_rows), pd.DataFrame(extension_rows)


def scan_matrices(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    program_genes: list[str],
) -> tuple[list[str], list[str], np.ndarray, sparse.csr_matrix, list[str], pd.DataFrame]:
    raw_files = sorted(Path(args.raw_dir).glob("*.csv.gz"))
    file_mapping = {file_orig_ident(path): path for path in raw_files}
    missing_files = sorted(set(metadata["orig.ident"]).difference(file_mapping))
    if missing_files:
        raise FileNotFoundError(f"Missing raw matrices for orig.ident: {missing_files}")

    reference_file = raw_files[0]
    genes = pd.read_csv(reference_file, usecols=["gene"], dtype={"gene": str})["gene"].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("Raw matrix contains duplicated gene symbols; aggregate before extraction.")
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    retained_program_genes = [gene for gene in program_genes if gene in gene_to_index]
    retained_program_index = {gene: index for index, gene in enumerate(retained_program_genes)}
    patient_timepoints = sorted(metadata["patient_timepoint"].unique())
    group_to_index = {group: index for index, group in enumerate(patient_timepoints)}
    pseudobulk = np.zeros((len(genes), len(patient_timepoints)), dtype=np.int64)
    sparse_blocks: list[sparse.csr_matrix] = []
    cell_order: list[str] = []
    scan_rows = []

    for file_index, (orig_ident, subset) in enumerate(metadata.groupby("orig.ident", sort=True), start=1):
        path = file_mapping[str(orig_ident)]
        raw_columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        raw_cells = raw_columns[1:]
        selected_set = set(subset["Cell_barcode"].astype(str))
        selected_cells = [cell for cell in raw_cells if cell in selected_set]
        if len(selected_cells) != len(subset):
            missing_cells = sorted(selected_set.difference(selected_cells))
            raise ValueError(f"{orig_ident}: {len(missing_cells)} selected cells are absent from raw matrix")
        groups = subset.set_index("Cell_barcode").loc[selected_cells, "patient_timepoint"]
        unique_groups = groups.unique().tolist()
        if len(unique_groups) != 1:
            raise ValueError(f"{orig_ident}: expected one patient-timepoint group, found {unique_groups}")
        patient_timepoint = str(unique_groups[0])
        group_index = group_to_index[patient_timepoint]
        block = np.zeros((len(retained_program_genes), len(selected_cells)), dtype=np.int32)
        cursor = 0
        usecols = ["gene", *selected_cells]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize):
            chunk_genes = chunk["gene"].astype(str).tolist()
            if chunk_genes != genes[cursor : cursor + len(chunk)]:
                raise ValueError(f"{orig_ident}: gene order differs from the reference matrix near row {cursor}")
            counts = chunk[selected_cells].to_numpy(dtype=np.int64, copy=False)
            pseudobulk[cursor : cursor + len(chunk), group_index] += counts.sum(axis=1)
            program_rows = [row for row, gene in enumerate(chunk_genes) if gene in retained_program_index]
            for row in program_rows:
                block[retained_program_index[chunk_genes[row]], :] = counts[row]
            cursor += len(chunk)
        if cursor != len(genes):
            raise ValueError(f"{orig_ident}: expected {len(genes)} gene rows, observed {cursor}")
        sparse_blocks.append(sparse.csr_matrix(block))
        cell_order.extend(selected_cells)
        scan_rows.append(
            {
                "orig.ident": orig_ident,
                "raw_file": path.name,
                "patient_timepoint": patient_timepoint,
                "n_selected_cells": len(selected_cells),
                "n_raw_cells": len(raw_cells),
            }
        )
        print(
            f"[{file_index:02d}/{metadata['orig.ident'].nunique():02d}] {orig_ident}: "
            f"{len(selected_cells)} malignant cells -> {patient_timepoint}",
            flush=True,
        )

    program_counts = sparse.hstack(sparse_blocks, format="csr") if sparse_blocks else sparse.csr_matrix((0, 0))
    return genes, patient_timepoints, pseudobulk, program_counts, cell_order, pd.DataFrame(scan_rows)


def write_outputs(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    patients: list[str],
    main_pairs: pd.DataFrame,
    extension_pairs: pd.DataFrame,
    genes: list[str],
    patient_timepoints: list[str],
    pseudobulk: np.ndarray,
    program_counts: sparse.csr_matrix,
    cell_order: list[str],
    program_genes: list[str],
    scan_summary: pd.DataFrame,
) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = metadata.set_index("Cell_barcode").loc[cell_order].reset_index()
    metadata.to_csv(outdir / "GSE161801_paired_myeloma_cell_metadata.csv.gz", index=False)
    main_pairs.to_csv(outdir / "GSE161801_paired_myeloma_main_pairs.csv", index=False)
    extension_pairs.to_csv(outdir / "GSE161801_paired_myeloma_post2_pairs.csv", index=False)
    scan_summary.to_csv(outdir / "GSE161801_paired_myeloma_orig_ident_summary.csv", index=False)
    sample_summary = (
        metadata.groupby(["patient", "PID_new", "timepoint", "analysis_phase", "patient_timepoint"], as_index=False)
        .agg(n_cells=("Cell_barcode", "size"), n_orig_ident=("orig.ident", "nunique"))
        .sort_values(["patient", "timepoint"])
    )
    sample_summary.to_csv(outdir / "GSE161801_paired_myeloma_patient_timepoint_summary.csv", index=False)

    pseudobulk_frame = pd.DataFrame(pseudobulk, index=genes, columns=patient_timepoints)
    pseudobulk_frame.index.name = "gene"
    pseudobulk_frame.to_csv(outdir / "GSE161801_paired_myeloma_patient_timepoint_pseudobulk_counts.csv.gz")
    library_size = pseudobulk_frame.sum(axis=0).replace(0, np.nan)
    log1p_cpm = np.log1p(pseudobulk_frame.div(library_size, axis=1) * 1_000_000)
    log1p_cpm.to_csv(outdir / "GSE161801_paired_myeloma_patient_timepoint_log1p_cpm.csv.gz")

    sparse.save_npz(outdir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts.npz", program_counts)
    pd.Series(program_genes, name="gene").to_csv(
        outdir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts_genes.csv", index=False
    )
    pd.Series(cell_order, name="Cell_barcode").to_csv(
        outdir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts_cells.csv.gz", index=False
    )
    report = {
        "dataset": "GSE161801",
        "selection": "celltype_1 == Myeloma; patients with both pre and post malignant cells",
        "paired_patients": patients,
        "n_paired_patients": len(patients),
        "n_main_pre_post_pairs": len(main_pairs),
        "n_post2_extension_pairs": len(extension_pairs),
        "n_retained_cells_all_timepoints": len(metadata),
        "n_main_pre_post_cells": int(metadata["timepoint"].isin(["pre", "post"]).sum()),
        "n_post2_cells": int(metadata["timepoint"].eq("post_2").sum()),
        "n_genes_pseudobulk": len(genes),
        "n_v14_program_genes": len(program_genes),
        "program_gene_cell_matrix_shape": list(program_counts.shape),
        "program_gene_cell_matrix_nonzero": int(program_counts.nnz),
    }
    (outdir / "GSE161801_paired_myeloma_extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    metadata, patients = cell_metadata(args)
    main_pairs, extension_pairs = pair_manifests(metadata, patients)
    program_table = pd.read_csv(args.program_genes)
    program_genes = sorted(program_table["gene"].dropna().astype(str).unique())
    genes, patient_timepoints, pseudobulk, program_counts, cells, scan_summary = scan_matrices(
        args, metadata, program_genes
    )
    retained_program_genes = [gene for gene in program_genes if gene in set(genes)]
    write_outputs(
        args,
        metadata,
        patients,
        main_pairs,
        extension_pairs,
        genes,
        patient_timepoints,
        pseudobulk,
        program_counts,
        cells,
        retained_program_genes,
        scan_summary,
    )
    print(f"\nDone: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
