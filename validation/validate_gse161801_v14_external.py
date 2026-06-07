#!/usr/bin/env python3
"""Validate frozen v14 programs in paired GSE161801 malignant plasma cells.

The primary analysis mirrors Seurat NormalizeData followed by patient-timepoint
averaging: log1p(count / cell_library_size * 10000). A pseudobulk log1p-CPM
analysis is exported as a sensitivity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest, wilcoxon


PROGRAM_ORDER = [f"Program_{index}" for index in range(1, 11)]
STABLE_PROGRAMS = {"Program_1", "Program_4", "Program_6", "Program_8", "Program_9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="raw scrna data2/processed")
    parser.add_argument("--raw-dir", default="raw scrna data2/GSE161801_RAW")
    parser.add_argument(
        "--program-genes",
        default="bulk_pre_sc/model_upgrade_v14_multimodal_dictionary/balanced/"
        "our_multimodal_dictionary_v14_core_program_genes.csv",
    )
    parser.add_argument(
        "--discovery-summary",
        default="bulk_pre_sc/model_upgrade_v14_multimodal_dictionary/balanced/validation/"
        "discovery_core_program_shift_summary.csv",
    )
    parser.add_argument("--outdir", default="raw scrna data2/processed/GSE161801_v14_external_validation")
    parser.add_argument("--chunksize", type=int, default=512)
    return parser.parse_args()


def program_key(program: str) -> int:
    return int(str(program).split("_")[-1])


def file_orig_ident(path: Path) -> str:
    return path.name.removesuffix(".csv.gz").split("_K43R_", 1)[1]


def bh_fdr(values: pd.Series) -> pd.Series:
    raw = values.astype(float).to_numpy()
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def bootstrap_ci(values: np.ndarray, seed: int = 17, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def expected_directions(discovery_summary: Path) -> dict[str, str]:
    summary = pd.read_csv(discovery_summary)
    return {
        str(row.program): "up" if float(row.mean_delta) > 0 else "down"
        for row in summary.itertuples(index=False)
    }


def cell_library_sizes(
    raw_dir: Path,
    metadata: pd.DataFrame,
    cells: list[str],
    cache_path: Path,
    chunksize: int,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["Cell_barcode"].astype(str).tolist() == cells:
            print(f"Using cached cell library sizes: {cache_path}", flush=True)
            return cached

    file_mapping = {file_orig_ident(path): path for path in sorted(raw_dir.glob("*.csv.gz"))}
    cell_to_index = {cell: index for index, cell in enumerate(cells)}
    totals = np.zeros(len(cells), dtype=np.int64)
    relevant = metadata[metadata["Cell_barcode"].isin(cells)].copy()
    for file_index, (orig_ident, subset) in enumerate(relevant.groupby("orig.ident", sort=True), start=1):
        path = file_mapping[str(orig_ident)]
        selected_set = set(subset["Cell_barcode"].astype(str))
        selected = [cell for cell in cells if cell in selected_set]
        indices = np.array([cell_to_index[cell] for cell in selected], dtype=int)
        for chunk in pd.read_csv(path, usecols=["gene", *selected], chunksize=chunksize):
            totals[indices] += chunk[selected].to_numpy(dtype=np.int64, copy=False).sum(axis=0)
        print(
            f"[{file_index:02d}/{relevant['orig.ident'].nunique():02d}] "
            f"{orig_ident}: library sizes for {len(selected)} cells",
            flush=True,
        )
    if np.any(totals <= 0):
        raise ValueError(f"{int(np.sum(totals <= 0))} retained cells have zero total UMI counts")
    out = pd.DataFrame({"Cell_barcode": cells, "total_umi": totals})
    out.to_csv(cache_path, index=False)
    return out


def signature_scores(
    expression: pd.DataFrame,
    program_genes: pd.DataFrame,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=expression.columns)
    for program in PROGRAM_ORDER:
        sub = program_genes[
            program_genes["program"].eq(program) & program_genes["gene"].isin(expression.index)
        ].copy()
        sub["abs_weight"] = sub["weight"].abs()
        sub = sub.sort_values("abs_weight", ascending=False).drop_duplicates("gene")
        weights = sub.set_index("gene")["weight"]
        scores[program] = expression.loc[weights.index].T.dot(weights) / weights.abs().sum()
    scores.index.name = "patient_timepoint"
    return scores.reset_index()


def paired_deltas(scores: pd.DataFrame, pairs: pd.DataFrame, method: str) -> pd.DataFrame:
    indexed = scores.set_index("patient_timepoint")
    rows = []
    for pair in pairs.itertuples(index=False):
        delta = indexed.loc[pair.relapse_group, PROGRAM_ORDER] - indexed.loc[pair.baseline_group, PROGRAM_ORDER]
        for program in PROGRAM_ORDER:
            rows.append(
                {
                    "method": method,
                    "pair_index": int(pair.pair_index),
                    "patient": str(pair.patient),
                    "program": program,
                    "delta_score": float(delta[program]),
                }
            )
    return pd.DataFrame(rows)


def summarize_deltas(long: pd.DataFrame, expected: dict[str, str]) -> pd.DataFrame:
    rows = []
    for program in PROGRAM_ORDER:
        values = long.loc[long["program"].eq(program), "delta_score"].to_numpy(dtype=float)
        direction = expected[program]
        expected_hits = values > 0 if direction == "up" else values < 0
        ci_low, ci_high = bootstrap_ci(values)
        p_value = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        directional_p_value = float(
            wilcoxon(
                values,
                zero_method="wilcox",
                alternative="greater" if direction == "up" else "less",
            ).pvalue
        )
        rows.append(
            {
                "method": str(long["method"].iloc[0]),
                "program": program,
                "n_pairs": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "frac_increase": float(np.mean(values > 0)),
                "expected_direction": direction,
                "mean_direction_concordant": bool((values.mean() > 0) == (direction == "up")),
                "n_patients_expected_direction": int(expected_hits.sum()),
                "frac_patients_expected_direction": float(expected_hits.mean()),
                "wilcoxon_p": p_value,
                "directional_wilcoxon_p": directional_p_value,
                "direction_binomial_p": float(
                    binomtest(int(expected_hits.sum()), len(values), p=0.5, alternative="greater").pvalue
                ),
                "previously_stable_program": program in STABLE_PROGRAMS,
            }
        )
    out = pd.DataFrame(rows).sort_values("program", key=lambda values: values.map(program_key))
    out["wilcoxon_fdr"] = bh_fdr(out["wilcoxon_p"])
    out["directional_wilcoxon_fdr"] = bh_fdr(out["directional_wilcoxon_p"])
    out["direction_binomial_fdr"] = bh_fdr(out["direction_binomial_p"])
    stable = out["previously_stable_program"]
    out["prespecified_stable_directional_wilcoxon_fdr"] = np.nan
    out["prespecified_stable_direction_binomial_fdr"] = np.nan
    out.loc[stable, "prespecified_stable_directional_wilcoxon_fdr"] = bh_fdr(
        out.loc[stable, "directional_wilcoxon_p"]
    )
    out.loc[stable, "prespecified_stable_direction_binomial_fdr"] = bh_fdr(
        out.loc[stable, "direction_binomial_p"]
    )
    return out


def cell_normalized_expression(
    processed_dir: Path,
    metadata: pd.DataFrame,
    libraries: pd.DataFrame,
) -> pd.DataFrame:
    matrix = sparse.load_npz(processed_dir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts.npz")
    genes = pd.read_csv(
        processed_dir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts_genes.csv"
    )["gene"].astype(str).tolist()
    cells = pd.read_csv(
        processed_dir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts_cells.csv.gz"
    )["Cell_barcode"].astype(str).tolist()
    if libraries["Cell_barcode"].astype(str).tolist() != cells:
        raise ValueError("Cell library-size order does not match the exported sparse matrix")
    normalized = matrix @ sparse.diags(10_000.0 / libraries["total_umi"].to_numpy(dtype=float))
    normalized.data = np.log1p(normalized.data)
    group_by_cell = metadata.set_index("Cell_barcode").loc[cells, "patient_timepoint"]
    groups = sorted(group_by_cell.unique())
    means = np.column_stack(
        [np.asarray(normalized[:, group_by_cell.to_numpy() == group].mean(axis=1)).ravel() for group in groups]
    )
    return pd.DataFrame(means, index=genes, columns=groups)


def pseudobulk_expression(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        processed_dir / "GSE161801_paired_myeloma_patient_timepoint_log1p_cpm.csv.gz",
        index_col=0,
    )


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(processed_dir / "GSE161801_paired_myeloma_cell_metadata.csv.gz")
    pairs = pd.read_csv(processed_dir / "GSE161801_paired_myeloma_main_pairs.csv")
    program_genes = pd.read_csv(args.program_genes)
    cells = pd.read_csv(
        processed_dir / "GSE161801_paired_myeloma_v14_program_gene_cell_counts_cells.csv.gz"
    )["Cell_barcode"].astype(str).tolist()
    expected = expected_directions(Path(args.discovery_summary))

    libraries = cell_library_sizes(
        Path(args.raw_dir),
        metadata,
        cells,
        outdir / "GSE161801_paired_myeloma_cell_library_sizes.csv.gz",
        args.chunksize,
    )
    cell_expression = cell_normalized_expression(processed_dir, metadata, libraries)
    pseudobulk = pseudobulk_expression(processed_dir)

    cell_scores = signature_scores(cell_expression, program_genes)
    pb_scores = signature_scores(pseudobulk, program_genes)
    cell_long = paired_deltas(cell_scores, pairs, "cell_normalized_mean")
    pb_long = paired_deltas(pb_scores, pairs, "pseudobulk_log1p_cpm")
    long = pd.concat([cell_long, pb_long], ignore_index=True)
    summary = pd.concat(
        [
            summarize_deltas(cell_long, expected),
            summarize_deltas(pb_long, expected),
        ],
        ignore_index=True,
    )
    stable = summary[summary["previously_stable_program"]].copy()

    cell_expression.to_csv(outdir / "GSE161801_cell_normalized_patient_timepoint_expression.csv.gz")
    cell_scores.to_csv(outdir / "GSE161801_cell_normalized_program_scores.csv", index=False)
    pb_scores.to_csv(outdir / "GSE161801_pseudobulk_program_scores.csv", index=False)
    long.to_csv(outdir / "GSE161801_v14_program_pair_deltas.csv", index=False)
    summary.to_csv(outdir / "GSE161801_v14_program_external_validation_summary.csv", index=False)
    stable.to_csv(outdir / "GSE161801_v14_stable_program_external_validation_summary.csv", index=False)
    print("\nPreviously stable programs:", flush=True)
    print(
        stable[
            [
                "method",
                "program",
                "mean_delta",
                "expected_direction",
                "mean_direction_concordant",
                "n_patients_expected_direction",
                "wilcoxon_p",
                "wilcoxon_fdr",
                "directional_wilcoxon_p",
                "prespecified_stable_directional_wilcoxon_fdr",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
