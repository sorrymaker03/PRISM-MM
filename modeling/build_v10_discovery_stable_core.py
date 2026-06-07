#!/usr/bin/env python3
"""Freeze variable-size, knowledge-guided program cores from discovery bulk only.

The v9 neural network learns the candidate program dictionary. This script
adds a publication-facing freezing layer that retains genes supported across
discovery studies and drugs. External bulk and scRNA data are intentionally
not read here.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import bulk_pre_sc_pipeline as base


VARIANTS = {
    "compact_pathway": {"threshold": 0.60, "min_per_direction": 5, "max_per_direction": 18, "pathway_weight": 0.10},
    "balanced_pathway": {"threshold": 0.56, "min_per_direction": 8, "max_per_direction": 30, "pathway_weight": 0.10},
    "balanced_strong_pathway": {"threshold": 0.56, "min_per_direction": 8, "max_per_direction": 30, "pathway_weight": 0.20},
    "broad_pathway": {"threshold": 0.52, "min_per_direction": 10, "max_per_direction": 40, "pathway_weight": 0.10},
    "broad_strong_pathway": {"threshold": 0.52, "min_per_direction": 10, "max_per_direction": 40, "pathway_weight": 0.20},
    "balanced_no_pathway": {"threshold": 0.56, "min_per_direction": 8, "max_per_direction": 30, "pathway_weight": 0.00},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--source-model-dir", default="bulk_pre_sc/model_upgrade_v9_final")
    parser.add_argument("--source-prefix", default="our_anchored_sparse_attention_v9")
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade_v10_discovery_core")
    parser.add_argument("--prefix", default="our_knowledge_guided_v10")
    parser.add_argument("--llm-prior-file", default="")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def program_key(value: str) -> int:
    match = re.search(r"_(\d+)$", str(value))
    return int(match.group(1)) if match else 10**6


def split_samples(value: object) -> list[str]:
    return [item for item in str(value).split(";") if item]


def percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(1.0, index=values.index)
    return values.rank(method="average", pct=True)


def load_pathway_prior(path: Path) -> pd.DataFrame:
    enrich = pd.read_csv(path)
    enrich = enrich[enrich["adjusted_p_value"].le(0.05)].copy()
    if enrich.empty:
        return pd.DataFrame(columns=["program", "direction", "gene", "pathway_prior"])
    enrich["evidence"] = -np.log10(enrich["adjusted_p_value"].clip(lower=1e-12))
    enrich["evidence"] = enrich["evidence"].clip(upper=8.0)
    rows: list[dict[str, object]] = []
    for item in enrich.itertuples(index=False):
        for gene in str(item.overlap_genes).split(";"):
            if gene:
                rows.append(
                    {
                        "program": item.program,
                        "direction": item.direction,
                        "gene": gene,
                        "evidence": item.evidence,
                    }
                )
    prior = pd.DataFrame(rows)
    if prior.empty:
        return pd.DataFrame(columns=["program", "direction", "gene", "pathway_prior"])
    prior = prior.groupby(["program", "direction", "gene"], as_index=False)["evidence"].sum()
    prior["pathway_prior"] = prior.groupby(["program", "direction"])["evidence"].transform(percentile)
    return prior[["program", "direction", "gene", "pathway_prior"]]


def load_llm_prior(path_string: str) -> pd.DataFrame:
    if not path_string:
        return pd.DataFrame(columns=["program", "gene", "llm_prior"])
    prior = pd.read_csv(path_string)
    required = {"program", "gene", "llm_prior"}
    missing = sorted(required.difference(prior.columns))
    if missing:
        raise ValueError(f"LLM prior is missing columns: {missing}")
    prior = prior[["program", "gene", "llm_prior"]].copy()
    prior["llm_prior"] = prior["llm_prior"].astype(float).clip(0.0, 1.0)
    return prior


def discovery_gene_deltas(
    expression: pd.DataFrame,
    contrasts: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_genes = sorted(set(genes).intersection(expression.index))
    expression = np.log2(expression.loc[use_genes] + 1.0).astype("float32")
    rows = []
    values = []
    for contrast in contrasts.itertuples(index=False):
        treated = split_samples(contrast.treated_samples)
        controls = split_samples(contrast.control_samples)
        delta = expression[treated].mean(axis=1) - expression[controls].mean(axis=1)
        rows.append(
            {
                "context_id": contrast.context_id,
                "block_id": contrast.block_id,
                "drug_name": contrast.drug_name,
            }
        )
        values.append(delta.to_numpy())
    return pd.DataFrame(values, index=pd.DataFrame(rows)["context_id"], columns=use_genes), pd.DataFrame(rows)


def bootstrap_positive_fraction(values: pd.DataFrame, repeats: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    matrix = values.to_numpy(dtype=float)
    positive = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(repeats):
        indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        positive += matrix[indices].mean(axis=0) > 0
    return pd.Series(positive / repeats, index=values.columns)


def candidate_evidence(
    genes: pd.DataFrame,
    context_deltas: pd.DataFrame,
    context_meta: pd.DataFrame,
    discovery_sign: pd.Series,
    pathway_prior: pd.DataFrame,
    llm_prior: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    block_labels = context_meta.set_index("context_id")["block_id"]
    drug_labels = context_meta.set_index("context_id")["drug_name"]
    rows = []
    for index, program in enumerate(sorted(genes["program"].unique(), key=program_key)):
        membership = genes[genes["program"].eq(program)].copy()
        use_genes = [gene for gene in membership["gene"] if gene in context_deltas.columns]
        membership = membership[membership["gene"].isin(use_genes)].copy()
        expected_sign = np.sign(membership.set_index("gene")["weight"]) * discovery_sign.loc[program]
        oriented = context_deltas[use_genes].mul(expected_sign, axis=1)
        by_block = oriented.groupby(block_labels).mean()
        by_drug = oriented.groupby(drug_labels).mean()
        membership["expected_sign"] = membership["gene"].map(expected_sign)
        membership["context_positive_fraction"] = membership["gene"].map((oriented > 0).mean())
        membership["block_positive_fraction"] = membership["gene"].map((by_block > 0).mean())
        membership["drug_positive_fraction"] = membership["gene"].map((by_drug > 0).mean())
        membership["mean_oriented_delta"] = membership["gene"].map(by_block.mean())
        membership["bootstrap_sign_probability"] = membership["gene"].map(
            bootstrap_positive_fraction(by_block, repeats, seed + index)
        )
        membership["effect_percentile"] = percentile(membership["mean_oriented_delta"])
        membership["gate_percentile"] = percentile(membership["selection_probability"])
        membership["rank_prior"] = 1.0 - (membership["rank"] - 1.0) / max(float(membership["rank"].max() - 1.0), 1.0)
        rows.append(membership)
    evidence = pd.concat(rows, ignore_index=True)
    evidence = evidence.merge(pathway_prior, on=["program", "direction", "gene"], how="left")
    evidence = evidence.merge(llm_prior, on=["program", "gene"], how="left")
    evidence["pathway_prior"] = evidence["pathway_prior"].fillna(0.0)
    evidence["llm_prior"] = evidence["llm_prior"].fillna(0.0)
    return evidence


def select_variant(evidence: pd.DataFrame, config: dict[str, float]) -> pd.DataFrame:
    pathway_weight = float(config["pathway_weight"])
    llm_weight = 0.05 if evidence["llm_prior"].gt(0).any() else 0.0
    evidence = evidence.copy()
    evidence["selection_score"] = (
        0.23 * evidence["block_positive_fraction"]
        + 0.12 * evidence["drug_positive_fraction"]
        + 0.10 * evidence["context_positive_fraction"]
        + 0.18 * evidence["bootstrap_sign_probability"]
        + 0.12 * evidence["effect_percentile"]
        + 0.10 * evidence["gate_percentile"]
        + 0.05 * evidence["rank_prior"]
        + pathway_weight * evidence["pathway_prior"]
        + llm_weight * evidence["llm_prior"]
    )
    normalizer = 0.90 + pathway_weight + llm_weight
    evidence["selection_score"] = evidence["selection_score"] / normalizer
    frames = []
    for (_, _), group in evidence.groupby(["program", "direction"], sort=False):
        group = group.sort_values(
            ["selection_score", "selection_probability", "rank"],
            ascending=[False, False, True],
        )
        chosen = group[group["selection_score"].ge(float(config["threshold"]))].copy()
        minimum = min(int(config["min_per_direction"]), len(group))
        maximum = min(int(config["max_per_direction"]), len(group))
        if len(chosen) < minimum:
            chosen = group.head(minimum).copy()
        chosen = chosen.head(maximum).copy()
        frames.append(chosen)
    selected = pd.concat(frames, ignore_index=True)
    selected["original_weight"] = selected["weight"]
    selected["weight"] = selected["weight"] * (0.75 + 0.50 * selected["selection_score"])
    selected["rank"] = selected.groupby(["program", "direction"])["selection_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    selected = selected.sort_values(["program", "direction", "rank"], key=lambda col: col.map(program_key) if col.name == "program" else col)
    return selected


def internal_summary(
    selected: pd.DataFrame,
    context_deltas: pd.DataFrame,
    context_meta: pd.DataFrame,
) -> pd.DataFrame:
    block_labels = context_meta.set_index("context_id")["block_id"]
    drug_labels = context_meta.set_index("context_id")["drug_name"]
    rows = []
    for program, membership in selected.groupby("program"):
        use = [gene for gene in membership["gene"] if gene in context_deltas.columns]
        indexed = membership.set_index("gene").loc[use]
        weights = indexed["weight"].abs()
        scores = context_deltas[use].mul(indexed["expected_sign"], axis=1).mul(weights, axis=1).sum(axis=1) / weights.sum()
        block_scores = scores.groupby(block_labels).mean()
        drug_scores = scores.groupby(drug_labels).mean()
        rows.append(
            {
                "program": program,
                "n_genes": len(indexed),
                "n_up": int(indexed["direction"].eq("up").sum()),
                "n_down": int(indexed["direction"].eq("down").sum()),
                "context_mean_oriented_score": float(scores.mean()),
                "context_positive_fraction": float((scores > 0).mean()),
                "block_positive_fraction": float((block_scores > 0).mean()),
                "drug_positive_fraction": float((drug_scores > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("program", key=lambda values: values.map(program_key))


def write_variant(
    outdir: Path,
    prefix: str,
    name: str,
    config: dict[str, float],
    evidence: pd.DataFrame,
    context_deltas: pd.DataFrame,
    context_meta: pd.DataFrame,
) -> dict[str, object]:
    variant_dir = outdir / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    selected = select_variant(evidence, config)
    summary = internal_summary(selected, context_deltas, context_meta)
    selected.to_csv(variant_dir / f"{prefix}_core_program_genes.csv", index=False)
    summary.to_csv(variant_dir / f"{prefix}_discovery_internal_summary.csv", index=False)
    with open(variant_dir / f"{prefix}_selector_config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    return {
        "variant": name,
        "mean_genes": float(summary["n_genes"].mean()),
        "min_genes": int(summary["n_genes"].min()),
        "max_genes": int(summary["n_genes"].max()),
        "mean_block_positive_fraction": float(summary["block_positive_fraction"].mean()),
        "mean_drug_positive_fraction": float(summary["drug_positive_fraction"].mean()),
        "programs_block_support_ge_0_60": int(summary["block_positive_fraction"].ge(0.60).sum()),
        "discovery_objective": float(
            0.55 * summary["block_positive_fraction"].mean()
            + 0.35 * summary["drug_positive_fraction"].mean()
            + 0.10 * summary["context_positive_fraction"].mean()
        ),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_model_dir)
    contrasts = pd.read_csv(Path(args.article_dir) / "ordered_contrasts_core.csv")
    genes = pd.read_csv(source / f"{args.source_prefix}_core_program_genes.csv")
    discovery = pd.read_csv(source / "validation_all" / "discovery_core_program_shift_summary.csv")
    discovery_sign = discovery.set_index("program")["mean_delta"].map(np.sign).replace(0, 1)
    pathway_prior = load_pathway_prior(source / f"{args.source_prefix}_core_program_enrichr_pathways.csv")
    llm_prior = load_llm_prior(args.llm_prior_file)
    expression = base.read_expression(Path(args.expr), human_symbol_like=True, exclude_technical_gene_families=True)
    context_deltas, context_meta = discovery_gene_deltas(expression, contrasts, genes["gene"].tolist())
    evidence = candidate_evidence(
        genes,
        context_deltas,
        context_meta,
        discovery_sign,
        pathway_prior,
        llm_prior,
        args.bootstrap_repeats,
        args.seed,
    )
    evidence.to_csv(outdir / f"{args.prefix}_candidate_gene_evidence.csv", index=False)
    context_meta.to_csv(outdir / f"{args.prefix}_discovery_context_metadata.csv", index=False)
    pd.DataFrame({"program": discovery_sign.index, "discovery_sign": discovery_sign.values}).to_csv(
        outdir / f"{args.prefix}_discovery_program_orientation.csv", index=False
    )

    variant_rows = [
        write_variant(outdir, args.prefix, name, config, evidence, context_deltas, context_meta)
        for name, config in VARIANTS.items()
    ]
    variants = pd.DataFrame(variant_rows).sort_values("discovery_objective", ascending=False)
    variants.to_csv(outdir / f"{args.prefix}_variant_discovery_summary.csv", index=False)
    pathway_variants = variants[~variants["variant"].eq("balanced_no_pathway")]
    selected_name = str(pathway_variants.iloc[0]["variant"])
    selected_dir = outdir / "variants" / selected_name
    shutil.copyfile(
        selected_dir / f"{args.prefix}_core_program_genes.csv",
        outdir / f"{args.prefix}_core_program_genes.csv",
    )
    shutil.copyfile(
        selected_dir / f"{args.prefix}_discovery_internal_summary.csv",
        outdir / f"{args.prefix}_discovery_internal_summary.csv",
    )
    with open(outdir / f"{args.prefix}_selected_variant.json", "w") as handle:
        json.dump(
            {
                "selected_variant": selected_name,
                "selection_rule": "highest discovery-only objective among pathway-guided variants",
                "external_validation_used_for_selection": False,
                "llm_prior_used": bool(args.llm_prior_file),
            },
            handle,
            indent=2,
        )
    print(variants.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected discovery-only variant: {selected_name}")


if __name__ == "__main__":
    main()
