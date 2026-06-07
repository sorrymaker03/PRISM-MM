#!/usr/bin/env python3
"""Compute symmetric held-out bulk and external scRNA program validations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT = ROOT / "bulk_pre_sc" / "supplement"
SOURCE = SUPPLEMENT / "source_data"
TABLES = SUPPLEMENT / "tables"
MODEL = ROOT / "bulk_pre_sc" / "model_upgrade_v14_multimodal_dictionary" / "balanced"
PROCESSED = ROOT / "raw scrna data2" / "processed"
EXTERNAL = PROCESSED / "GSE161801_v14_external_validation"
PAIRS = PROCESSED / "GSE161801_paired_myeloma_main_pairs.csv"

sys.path.append(str(ROOT / "bulk_pre_sc" / "main_figure_code"))
sys.path.append(str(ROOT / "bulk_pre_sc"))
import figure4_bulk_validation as f4
import export_figure5_external_validation_panels as f5
import bulk_pre_sc_pipeline as base


CORE = ["Program_6", "Program_8", "Program_9"]
ALL_PROGRAMS = [f"Program_{index}" for index in range(1, 11)]
ORIENTATION = {"Program_6": -1.0, "Program_8": 1.0, "Program_9": 1.0}


def safe_wilcoxon(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative=alternative).pvalue)


def bootstrap_ci(values: np.ndarray, seed: int = 17, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def summarize_oriented_units(frame: pd.DataFrame, dataset: str, unit_col: str) -> pd.DataFrame:
    wide = frame.pivot_table(index=unit_col, columns="program", values="delta_score", aggfunc="mean")
    wide = wide.loc[:, CORE].dropna()
    oriented = wide.copy()
    for program in CORE:
        oriented[program] *= ORIENTATION[program]
    out = oriented.reset_index().melt(id_vars=unit_col, var_name="program", value_name="oriented_delta")
    composite = oriented.mean(axis=1).rename("directional_state").reset_index()
    out["dataset"] = dataset
    composite["dataset"] = dataset
    return out, composite


def bulk_program_units() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta, _, genes = f4.load_inputs()
    scores = f4.score_samples(meta, genes)
    eval_meta = f4.heldout_eval_meta(meta)
    eval_scores = scores[scores["sample"].isin(eval_meta["sample"])].copy()
    delta = f4.program_deltas(scores, eval_meta, ALL_PROGRAMS)
    delta["unit_id"] = delta["block_id"].astype(str) + ":" + delta["drug_name"].astype(str)
    context = delta.groupby(["unit_id", "block_id", "drug_name", "program"], as_index=False)["delta_score"].mean()
    oriented, composite = summarize_oriented_units(context[context["program"].isin(CORE)], "heldout_bulk", "unit_id")
    return meta, eval_scores, context, composite


def external_program_units() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long = pd.read_csv(EXTERNAL / "GSE161801_v14_program_pair_deltas.csv")
    long = long[long["method"].eq("cell_normalized_mean") & long["program"].isin(ALL_PROGRAMS)].copy()
    long["unit_id"] = long["pair_index"].astype(str)
    oriented, composite = summarize_oriented_units(long[long["program"].isin(CORE)], "external_scRNA", "unit_id")
    return long, oriented, composite


def high_fraction_bulk(eval_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for block_id, block in eval_scores.groupby("block_id"):
        ctrl = block[block["is_control"]]
        for drug_name, treated in block[block["is_treated"]].groupby("drug_name"):
            unit_id = f"{block_id}:{drug_name}"
            for program in CORE:
                threshold = float(ctrl[program].quantile(0.75))
                delta = float((treated[program] > threshold).mean() - (ctrl[program] > threshold).mean())
                rows.append(
                    {
                        "dataset": "heldout_bulk",
                        "unit_id": unit_id,
                        "program": program,
                        "fraction_delta": delta,
                        "oriented_fraction_delta": delta * ORIENTATION[program],
                    }
                )
    long = pd.DataFrame(rows)
    composite = long.groupby(["dataset", "unit_id"], as_index=False)["oriented_fraction_delta"].mean()
    composite = composite.rename(columns={"oriented_fraction_delta": "directional_high_state_shift"})
    return long, composite


def high_fraction_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = f5.cell_level_program_scores()
    rows = []
    for pair_index, pair in cells.groupby("pair_index"):
        baseline = pair[pair["phase"].eq("Baseline")]
        relapse = pair[pair["phase"].eq("Relapse")]
        for program in CORE:
            threshold = float(baseline[program].quantile(0.75))
            delta = float((relapse[program] > threshold).mean() - (baseline[program] > threshold).mean())
            rows.append(
                {
                    "dataset": "external_scRNA",
                    "unit_id": str(pair_index),
                    "program": program,
                    "fraction_delta": delta,
                    "oriented_fraction_delta": delta * ORIENTATION[program],
                }
            )
    long = pd.DataFrame(rows)
    composite = long.groupby(["dataset", "unit_id"], as_index=False)["oriented_fraction_delta"].mean()
    composite = composite.rename(columns={"oriented_fraction_delta": "directional_high_state_shift"})
    return long, composite


def heldout_gene_deltas(meta: pd.DataFrame, program_genes: pd.DataFrame) -> pd.DataFrame:
    expr = base.read_expression(ROOT / "raw data" / "RNAfinal.csv", human_symbol_like=True, exclude_technical_gene_families=True)
    expr = np.log2(expr + 1.0).astype("float32")
    selected_genes = sorted(set(program_genes["gene"]).intersection(expr.index))
    heldout = f4.heldout_eval_meta(meta)
    rows = []
    for block_id, block in heldout.groupby("block_id"):
        ctrl_samples = block.loc[block["is_control"], "sample"].tolist()
        ctrl_mean = expr.loc[selected_genes, ctrl_samples].mean(axis=1)
        for drug_name, treated in block[block["is_treated"]].groupby("drug_name"):
            unit_id = f"{block_id}:{drug_name}"
            treated_mean = expr.loc[selected_genes, treated["sample"].tolist()].mean(axis=1)
            delta = treated_mean - ctrl_mean
            for gene, value in delta.items():
                rows.append({"dataset": "heldout_bulk", "unit_id": unit_id, "gene": gene, "gene_delta": float(value)})
    return pd.DataFrame(rows)


def external_gene_deltas(program_genes: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.read_csv(PROCESSED / "GSE161801_paired_myeloma_patient_timepoint_log1p_cpm.csv.gz", index_col=0)
    pairs = pd.read_csv(PAIRS)
    selected_genes = sorted(set(program_genes["gene"]).intersection(matrix.index))
    rows = []
    for pair in pairs.itertuples(index=False):
        delta = matrix.loc[selected_genes, pair.relapse_group] - matrix.loc[selected_genes, pair.baseline_group]
        for gene, value in delta.items():
            rows.append({"dataset": "external_scRNA", "unit_id": str(pair.pair_index), "gene": gene, "gene_delta": float(value)})
    return pd.DataFrame(rows)


def annotate_oriented_genes(gene_delta: pd.DataFrame, program_genes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = program_genes[program_genes["program"].isin(CORE)][["program", "gene", "weight"]].copy()
    membership["expected_sign"] = membership["program"].map(ORIENTATION) * np.sign(membership["weight"])
    detail = membership.merge(gene_delta, on="gene", how="inner")
    detail["oriented_gene_delta"] = detail["gene_delta"] * detail["expected_sign"]
    gene_means = detail.groupby(["dataset", "program", "gene", "weight", "expected_sign"], as_index=False).agg(
        oriented_gene_delta=("oriented_gene_delta", "mean"),
        raw_gene_delta=("gene_delta", "mean"),
        n_units=("unit_id", "nunique"),
    )
    rows = []
    for (dataset, program), sub in gene_means.groupby(["dataset", "program"]):
        values = sub["oriented_gene_delta"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        rows.append(
            {
                "dataset": dataset,
                "program": program,
                "n_genes": len(values),
                "mean_oriented_gene_delta": values.mean(),
                "median_oriented_gene_delta": np.median(values),
                "ci_low": lo,
                "ci_high": hi,
                "fraction_genes_expected": float(np.mean(values > 0)),
                "wilcoxon_greater_p": safe_wilcoxon(values, "greater"),
                "binomial_greater_p": float(binomtest(int(np.sum(values > 0)), len(values), 0.5, alternative="greater").pvalue),
            }
        )
    return gene_means, pd.DataFrame(rows)


def gene_set_size_sensitivity(
    gene_delta: pd.DataFrame, program_genes: pd.DataFrame, dataset: str, sizes: tuple[int, ...] = (5, 8, 10, 999)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = program_genes[program_genes["program"].isin(CORE)].copy()
    membership["abs_weight"] = membership["weight"].abs()
    merged = gene_delta.merge(membership, on="gene", how="inner")
    rows = []
    for (unit_id, program), sub in merged.groupby(["unit_id", "program"]):
        sub = sub.sort_values("abs_weight", ascending=False)
        for size in sizes:
            chosen = sub if size == 999 else sub.head(size)
            score = float((chosen["gene_delta"] * chosen["weight"]).sum() / chosen["weight"].abs().sum())
            rows.append(
                {
                    "dataset": dataset,
                    "unit_id": unit_id,
                    "program": program,
                    "gene_set_size": "All" if size == 999 else str(size),
                    "oriented_program_score": score * ORIENTATION[program],
                }
            )
    long = pd.DataFrame(rows)
    composite = long.groupby(["dataset", "unit_id", "gene_set_size"], as_index=False)["oriented_program_score"].mean()
    composite = composite.rename(columns={"oriented_program_score": "directional_state"})
    summary_rows = []
    for (dataset_name, size), sub in composite.groupby(["dataset", "gene_set_size"], sort=False):
        values = sub["directional_state"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "gene_set_size": size,
                "n_units": len(values),
                "mean_directional_state": values.mean(),
                "median_directional_state": np.median(values),
                "ci_low": lo,
                "ci_high": hi,
                "fraction_positive": float(np.mean(values > 0)),
                "wilcoxon_greater_p": safe_wilcoxon(values, "greater"),
            }
        )
    return composite, pd.DataFrame(summary_rows)


def summarize_unit_metric(frame: pd.DataFrame, metric: str, label: str) -> dict[str, object]:
    values = frame[metric].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(values)
    return {
        "validation": label,
        "n_units": len(values),
        "mean": values.mean(),
        "median": np.median(values),
        "ci_low": lo,
        "ci_high": hi,
        "fraction_positive": float(np.mean(values > 0)),
        "wilcoxon_greater_p": safe_wilcoxon(values, "greater"),
        "binomial_greater_p": float(binomtest(int(np.sum(values > 0)), len(values), 0.5, alternative="greater").pvalue),
    }


def summarize_program_units(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for program, sub in frame.groupby("program"):
        values = sub["delta_score"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        rows.append(
            {
                "dataset": dataset,
                "program": program,
                "n_units": len(values),
                "mean_delta": values.mean(),
                "median_delta": np.median(values),
                "ci_low": lo,
                "ci_high": hi,
                "fraction_increase": float(np.mean(values > 0)),
                "wilcoxon_two_sided_p": safe_wilcoxon(values, "two-sided"),
            }
        )
    return pd.DataFrame(rows).sort_values("program", key=lambda x: x.str.extract(r"_(\d+)")[0].astype(int))


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    program_genes = pd.read_csv(MODEL / "our_multimodal_dictionary_v14_core_program_genes.csv")

    meta, eval_scores, bulk_context, bulk_composite = bulk_program_units()
    external_pair, external_oriented, external_composite = external_program_units()
    bulk_context.to_csv(SOURCE / "symmetric_validation_bulk_context_program_deltas.csv", index=False)
    external_pair.to_csv(SOURCE / "symmetric_validation_external_pair_program_deltas.csv", index=False)
    pd.concat([bulk_composite, external_composite], ignore_index=True).to_csv(
        SOURCE / "symmetric_validation_directional_state_by_unit.csv", index=False
    )
    pd.concat(
        [
            summarize_program_units(bulk_context, "heldout_bulk"),
            summarize_program_units(external_pair, "external_scRNA"),
        ],
        ignore_index=True,
    ).to_csv(TABLES / "supplementary_table_10_independent_unit_program_shifts.csv", index=False)

    bulk_fraction, bulk_fraction_composite = high_fraction_bulk(eval_scores)
    external_fraction, external_fraction_composite = high_fraction_external()
    pd.concat([bulk_fraction, external_fraction], ignore_index=True).to_csv(
        SOURCE / "symmetric_validation_high_state_fraction_by_program_unit.csv", index=False
    )
    pd.concat([bulk_fraction_composite, external_fraction_composite], ignore_index=True).to_csv(
        SOURCE / "symmetric_validation_directional_high_state_by_unit.csv", index=False
    )

    bulk_gene = heldout_gene_deltas(meta, program_genes)
    external_gene = external_gene_deltas(program_genes)
    gene_means, gene_summary = annotate_oriented_genes(pd.concat([bulk_gene, external_gene], ignore_index=True), program_genes)
    gene_means.to_csv(SOURCE / "candidate_symmetric_validation_oriented_gene_deltas.csv", index=False)
    gene_summary.to_csv(SOURCE / "candidate_symmetric_validation_gene_direction_summary.csv", index=False)

    bulk_size, bulk_size_summary = gene_set_size_sensitivity(bulk_gene, program_genes, "heldout_bulk")
    external_size, external_size_summary = gene_set_size_sensitivity(external_gene, program_genes, "external_scRNA")
    pd.concat([bulk_size, external_size], ignore_index=True).to_csv(
        SOURCE / "symmetric_validation_gene_set_size_by_unit.csv", index=False
    )
    pd.concat([bulk_size_summary, external_size_summary], ignore_index=True).to_csv(
        TABLES / "supplementary_table_11_core_program_gene_set_size_sensitivity.csv", index=False
    )

    summary = pd.DataFrame(
        [
            summarize_unit_metric(bulk_composite, "directional_state", "heldout_bulk_directional_state"),
            summarize_unit_metric(external_composite, "directional_state", "external_scRNA_directional_state"),
            summarize_unit_metric(bulk_fraction_composite, "directional_high_state_shift", "heldout_bulk_high_state_shift"),
            summarize_unit_metric(external_fraction_composite, "directional_high_state_shift", "external_scRNA_high_state_shift"),
        ]
    )
    summary.to_csv(TABLES / "supplementary_table_7_symmetric_validation_summary.csv", index=False)
    print("\nSymmetric validation summary")
    print(summary.round(4).to_string(index=False))
    print("\nGene direction summary")
    print(gene_summary.round(4).to_string(index=False))
    print("\nGene-set size sensitivity")
    print(pd.concat([bulk_size_summary, external_size_summary], ignore_index=True).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
