#!/usr/bin/env python3
"""Freeze v11 program cores with discovery-only leave-block-out selection.

The v9 neural dictionary remains the candidate bank. For every program, this
script selects the core size and evidence recipe by cross-validation across
discovery blocks. External bulk samples and scRNA data are intentionally not
read here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import bulk_pre_sc_pipeline as base
from build_v10_discovery_stable_core import (
    discovery_gene_deltas,
    load_llm_prior,
    load_pathway_prior,
    program_key,
)


RECIPES = {
    "study_stability": {
        "block_positive_fraction": 0.30,
        "drug_positive_fraction": 0.15,
        "context_positive_fraction": 0.05,
        "bootstrap_sign_probability": 0.20,
        "effect_percentile": 0.10,
        "gate_percentile": 0.08,
        "rank_prior": 0.02,
        "pathway_prior": 0.10,
    },
    "balanced": {
        "block_positive_fraction": 0.23,
        "drug_positive_fraction": 0.12,
        "context_positive_fraction": 0.10,
        "bootstrap_sign_probability": 0.18,
        "effect_percentile": 0.12,
        "gate_percentile": 0.10,
        "rank_prior": 0.05,
        "pathway_prior": 0.10,
    },
    "effect_aware": {
        "block_positive_fraction": 0.20,
        "drug_positive_fraction": 0.10,
        "context_positive_fraction": 0.05,
        "bootstrap_sign_probability": 0.15,
        "effect_percentile": 0.25,
        "gate_percentile": 0.08,
        "rank_prior": 0.02,
        "pathway_prior": 0.15,
    },
}
K_PER_DIRECTION = [5, 8, 10, 15, 20, 25, 30, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--source-model-dir", default="bulk_pre_sc/model_upgrade_v9_final")
    parser.add_argument("--source-prefix", default="our_anchored_sparse_attention_v9")
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade_v11_cross_study_core")
    parser.add_argument("--prefix", default="our_cross_study_v11")
    parser.add_argument("--llm-prior-file", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(1.0, index=values.index)
    return values.rank(method="average", pct=True)


def bootstrap_positive_fraction(values: pd.DataFrame, repeats: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    matrix = values.to_numpy(dtype=float)
    positive = np.zeros(matrix.shape[1], dtype=float)
    for _ in range(repeats):
        indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        positive += matrix[indices].mean(axis=0) > 0
    return pd.Series(positive / repeats, index=values.columns)


def score_evidence(
    membership: pd.DataFrame,
    deltas: pd.DataFrame,
    meta: pd.DataFrame,
    pathway_prior: pd.DataFrame,
    llm_prior: pd.DataFrame,
    recipe: dict[str, float],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    use = [gene for gene in membership["gene"] if gene in deltas.columns]
    membership = membership[membership["gene"].isin(use)].copy()
    expected = membership.set_index("gene")["expected_sign"]
    oriented = deltas[use].mul(expected, axis=1)
    block_labels = meta.set_index("context_id")["block_id"]
    drug_labels = meta.set_index("context_id")["drug_name"]
    by_block = oriented.groupby(block_labels).mean()
    by_drug = oriented.groupby(drug_labels).mean()
    membership["context_positive_fraction"] = membership["gene"].map((oriented > 0).mean())
    membership["block_positive_fraction"] = membership["gene"].map((by_block > 0).mean())
    membership["drug_positive_fraction"] = membership["gene"].map((by_drug > 0).mean())
    membership["mean_oriented_delta"] = membership["gene"].map(by_block.mean())
    membership["bootstrap_sign_probability"] = membership["gene"].map(
        bootstrap_positive_fraction(by_block, repeats, seed)
    )
    membership["effect_percentile"] = percentile(membership["mean_oriented_delta"])
    membership["gate_percentile"] = percentile(membership["selection_probability"])
    membership["rank_prior"] = 1.0 - (membership["rank"] - 1.0) / max(float(membership["rank"].max() - 1.0), 1.0)
    membership = membership.merge(pathway_prior, on=["program", "direction", "gene"], how="left")
    membership = membership.merge(llm_prior, on=["program", "gene"], how="left")
    membership["pathway_prior"] = membership["pathway_prior"].fillna(0.0)
    membership["llm_prior"] = membership["llm_prior"].fillna(0.0)
    membership["selection_score"] = 0.0
    for feature, weight in recipe.items():
        membership["selection_score"] += weight * membership[feature]
    if membership["llm_prior"].gt(0).any():
        membership["selection_score"] = (membership["selection_score"] + 0.05 * membership["llm_prior"]) / 1.05
    return membership


def select_members(score_table: pd.DataFrame, k_per_direction: int) -> pd.DataFrame:
    frames = []
    for _, group in score_table.groupby("direction", sort=False):
        frames.append(
            group.sort_values(
                ["selection_score", "selection_probability", "rank"],
                ascending=[False, False, True],
            ).head(k_per_direction)
        )
    return pd.concat(frames, ignore_index=True)


def program_scores(selected: pd.DataFrame, deltas: pd.DataFrame) -> pd.Series:
    use = [gene for gene in selected["gene"] if gene in deltas.columns]
    indexed = selected.set_index("gene").loc[use]
    weights = indexed["weight"].abs() * (0.75 + 0.50 * indexed["selection_score"])
    return deltas[use].mul(indexed["expected_sign"], axis=1).mul(weights, axis=1).sum(axis=1) / weights.sum()


def fold_map(blocks: list[str], folds: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(blocks), dtype=object)
    rng.shuffle(shuffled)
    return {block: index % folds for index, block in enumerate(shuffled)}


def validation_record(
    program: str,
    recipe_name: str,
    k: int,
    fold: int,
    selected: pd.DataFrame,
    deltas: pd.DataFrame,
    meta: pd.DataFrame,
) -> dict[str, object]:
    scores = program_scores(selected, deltas)
    block_scores = scores.groupby(meta.set_index("context_id")["block_id"]).mean()
    mean = float(block_scores.mean())
    sd = float(block_scores.std(ddof=1)) if len(block_scores) > 1 else 0.0
    return {
        "program": program,
        "recipe": recipe_name,
        "k_per_direction": k,
        "fold": fold,
        "n_selected": len(selected),
        "n_validation_blocks": len(block_scores),
        "validation_mean_oriented_score": mean,
        "validation_positive_fraction": float((block_scores > 0).mean()),
        "validation_standardized_mean": mean / max(sd, 1e-8),
    }


def choose_hyperparameters(screen: pd.DataFrame) -> pd.Series:
    grouped = (
        screen.groupby(["program", "recipe", "k_per_direction"], as_index=False)
        .agg(
            cv_positive_fraction=("validation_positive_fraction", "mean"),
            cv_mean_oriented_score=("validation_mean_oriented_score", "mean"),
            cv_standardized_mean=("validation_standardized_mean", "mean"),
        )
    )
    grouped["cv_objective"] = (
        grouped["cv_positive_fraction"]
        + 0.08 * np.tanh(grouped["cv_standardized_mean"])
        - 0.0005 * grouped["k_per_direction"]
    )
    best = (
        grouped.sort_values(
            ["program", "cv_objective", "cv_positive_fraction", "k_per_direction"],
            ascending=[True, False, False, True],
        )
        .groupby("program", as_index=False)
        .head(1)
        .copy()
    )
    return grouped, best


def internal_summary(selected: pd.DataFrame, deltas: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    block_labels = meta.set_index("context_id")["block_id"]
    drug_labels = meta.set_index("context_id")["drug_name"]
    for program, group in selected.groupby("program"):
        scores = program_scores(group, deltas)
        blocks = scores.groupby(block_labels).mean()
        drugs = scores.groupby(drug_labels).mean()
        rows.append(
            {
                "program": program,
                "n_genes": len(group),
                "n_up": int(group["direction"].eq("up").sum()),
                "n_down": int(group["direction"].eq("down").sum()),
                "context_mean_oriented_score": float(scores.mean()),
                "context_positive_fraction": float((scores > 0).mean()),
                "block_positive_fraction": float((blocks > 0).mean()),
                "drug_positive_fraction": float((drugs > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("program", key=lambda values: values.map(program_key))


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
    deltas, meta = discovery_gene_deltas(expression, contrasts, genes["gene"].tolist())
    blocks = sorted(meta["block_id"].unique())
    block_to_fold = fold_map(blocks, args.folds, args.seed)
    meta["fold"] = meta["block_id"].map(block_to_fold)
    rows = []
    programs = sorted(genes["program"].unique(), key=program_key)
    for program_index, program in enumerate(programs):
        membership = genes[genes["program"].eq(program)].copy()
        membership["expected_sign"] = np.sign(membership["weight"]) * discovery_sign.loc[program]
        for fold in range(args.folds):
            train_ids = meta.loc[meta["fold"].ne(fold), "context_id"]
            valid_ids = meta.loc[meta["fold"].eq(fold), "context_id"]
            train_meta = meta[meta["fold"].ne(fold)].copy()
            valid_meta = meta[meta["fold"].eq(fold)].copy()
            for recipe_name, recipe in RECIPES.items():
                evidence = score_evidence(
                    membership,
                    deltas.loc[train_ids],
                    train_meta,
                    pathway_prior,
                    llm_prior,
                    recipe,
                    args.bootstrap_repeats,
                    args.seed + 100 * program_index + fold,
                )
                for k in K_PER_DIRECTION:
                    selected = select_members(evidence, k)
                    rows.append(validation_record(program, recipe_name, k, fold, selected, deltas.loc[valid_ids], valid_meta))
    fold_screen = pd.DataFrame(rows)
    grouped_screen, best = choose_hyperparameters(fold_screen)
    selected_frames = []
    for program_index, program in enumerate(programs):
        chosen = best[best["program"].eq(program)].iloc[0]
        membership = genes[genes["program"].eq(program)].copy()
        membership["expected_sign"] = np.sign(membership["weight"]) * discovery_sign.loc[program]
        evidence = score_evidence(
            membership,
            deltas,
            meta,
            pathway_prior,
            llm_prior,
            RECIPES[str(chosen["recipe"])],
            args.bootstrap_repeats,
            args.seed + 1000 + program_index,
        )
        selected = select_members(evidence, int(chosen["k_per_direction"])).copy()
        selected["selected_recipe"] = str(chosen["recipe"])
        selected["selected_k_per_direction"] = int(chosen["k_per_direction"])
        selected["original_weight"] = selected["weight"]
        selected["weight"] = selected["weight"] * (0.75 + 0.50 * selected["selection_score"])
        selected["rank"] = selected.groupby("direction")["selection_score"].rank(ascending=False, method="first").astype(int)
        selected_frames.append(selected)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(
        ["program", "direction", "rank"],
        key=lambda values: values.map(program_key) if values.name == "program" else values,
    )
    summary = internal_summary(selected, deltas, meta)
    fold_screen.to_csv(outdir / f"{args.prefix}_fold_validation_detail.csv", index=False)
    grouped_screen.to_csv(outdir / f"{args.prefix}_hyperparameter_screen.csv", index=False)
    best.to_csv(outdir / f"{args.prefix}_selected_hyperparameters.csv", index=False)
    selected.to_csv(outdir / f"{args.prefix}_core_program_genes.csv", index=False)
    summary.to_csv(outdir / f"{args.prefix}_discovery_internal_summary.csv", index=False)
    meta.to_csv(outdir / f"{args.prefix}_discovery_context_folds.csv", index=False)
    with open(outdir / f"{args.prefix}_config.json", "w") as handle:
        json.dump(
            {
                "source_model": args.source_prefix,
                "selection": "discovery-only leave-block-out cross-validation",
                "external_validation_used_for_selection": False,
                "llm_prior_used": bool(args.llm_prior_file),
                "folds": args.folds,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    print(best[["program", "recipe", "k_per_direction", "cv_positive_fraction", "cv_objective"]].to_string(index=False))
    print("\nFrozen gene counts")
    print(summary[["program", "n_genes", "n_up", "n_down", "block_positive_fraction", "drug_positive_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
