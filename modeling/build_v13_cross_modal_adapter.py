#!/usr/bin/env python3
"""Build an explicitly calibrated bulk-to-scRNA program adapter.

The v13 adapter uses discovery bulk plus scRNA training pairs. Pair-level
cross-validation estimates internal cross-modal generalization. Held-out bulk
samples are intentionally not read here and remain available for evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.append(str(Path(__file__).resolve().parent))
import bulk_pre_sc_pipeline as base
from build_v10_discovery_stable_core import discovery_gene_deltas, load_llm_prior, load_pathway_prior
from build_v11_cross_study_core import RECIPES, program_scores, score_evidence, select_members


ADAPTER_RECIPES = {
    "adapter_balanced": {
        "discovery_score": 0.60,
        "scrna_positive_fraction": 0.15,
        "scrna_bootstrap_probability": 0.15,
        "scrna_effect_percentile": 0.10,
    },
    "adapter_support": {
        "discovery_score": 0.50,
        "scrna_positive_fraction": 0.25,
        "scrna_bootstrap_probability": 0.15,
        "scrna_effect_percentile": 0.10,
    },
    "adapter_effect": {
        "discovery_score": 0.50,
        "scrna_positive_fraction": 0.15,
        "scrna_bootstrap_probability": 0.10,
        "scrna_effect_percentile": 0.25,
    },
}
K_PER_DIRECTION = [5, 8, 10, 15, 20, 25, 30, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--source-model-dir", default="bulk_pre_sc/model_upgrade_v9_final")
    parser.add_argument("--source-prefix", default="our_anchored_sparse_attention_v9")
    parser.add_argument(
        "--scrna-gene-deltas",
        default="bulk_pre_sc/model_upgrade_v9_final/gene_direction_validation/scrna_all_adaptive_gene_delta_by_pair.csv",
    )
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade_v13_cross_modal_adapter")
    parser.add_argument("--prefix", default="our_cross_modal_adapter_v13")
    parser.add_argument("--llm-prior-file", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def program_key(value: str) -> int:
    match = re.search(r"_(\d+)$", str(value))
    return int(match.group(1)) if match else 10**6


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


def fold_map(units: list[int], folds: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(units), dtype=int)
    rng.shuffle(shuffled)
    return {int(unit): index % folds for index, unit in enumerate(shuffled)}


def add_scrna_evidence(
    discovery_evidence: pd.DataFrame,
    scrna_pairs: pd.DataFrame,
    adapter_recipe: dict[str, float],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    use = [gene for gene in discovery_evidence["gene"] if gene in scrna_pairs.columns]
    evidence = discovery_evidence[discovery_evidence["gene"].isin(use)].copy()
    expected = evidence.set_index("gene")["expected_sign"]
    oriented = scrna_pairs[use].mul(expected, axis=1)
    evidence["discovery_score"] = evidence["selection_score"]
    evidence["scrna_positive_fraction"] = evidence["gene"].map((oriented > 0).mean())
    evidence["scrna_mean_oriented_delta"] = evidence["gene"].map(oriented.mean())
    evidence["scrna_bootstrap_probability"] = evidence["gene"].map(
        bootstrap_positive_fraction(oriented, repeats, seed)
    )
    evidence["scrna_effect_percentile"] = percentile(evidence["scrna_mean_oriented_delta"])
    evidence["selection_score"] = 0.0
    for feature, weight in adapter_recipe.items():
        evidence["selection_score"] += weight * evidence[feature]
    return evidence


def safe_wilcoxon(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue)


def bh_fdr(values: pd.Series) -> pd.Series:
    raw = values.astype(float).to_numpy()
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def candidate_record(
    program: str,
    recipe: str,
    k: int,
    fold: int,
    selected: pd.DataFrame,
    validation_pairs: pd.DataFrame,
) -> dict[str, object]:
    scores = program_scores(selected, validation_pairs)
    mean = float(scores.mean())
    sd = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    return {
        "program": program,
        "adapter_recipe": recipe,
        "k_per_direction": k,
        "fold": fold,
        "n_validation_pairs": len(scores),
        "validation_mean_oriented_score": mean,
        "validation_positive_fraction": float((scores > 0).mean()),
        "validation_standardized_mean": mean / max(sd, 1e-8),
    }


def choose_hyperparameters(screen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        screen.groupby(["program", "adapter_recipe", "k_per_direction"], as_index=False)
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
    bulk_deltas, bulk_meta = discovery_gene_deltas(expression, contrasts, genes["gene"].tolist())
    scrna_long = pd.read_csv(args.scrna_gene_deltas)
    scrna_pairs = scrna_long.pivot_table(index="matched_pair_index", columns="gene", values="delta", aggfunc="mean")
    pair_to_fold = fold_map(scrna_pairs.index.astype(int).tolist(), args.folds, args.seed)
    programs = sorted(genes["program"].unique(), key=program_key)
    discovery_tables: dict[str, pd.DataFrame] = {}
    rows = []
    for program_index, program in enumerate(programs):
        membership = genes[genes["program"].eq(program)].copy()
        membership["expected_sign"] = np.sign(membership["weight"]) * discovery_sign.loc[program]
        discovery_tables[program] = score_evidence(
            membership,
            bulk_deltas,
            bulk_meta,
            pathway_prior,
            llm_prior,
            RECIPES["balanced"],
            args.bootstrap_repeats,
            args.seed + program_index,
        )
        for fold in range(args.folds):
            train_units = [unit for unit, assigned in pair_to_fold.items() if assigned != fold]
            valid_units = [unit for unit, assigned in pair_to_fold.items() if assigned == fold]
            for recipe_name, adapter_recipe in ADAPTER_RECIPES.items():
                evidence = add_scrna_evidence(
                    discovery_tables[program],
                    scrna_pairs.loc[train_units],
                    adapter_recipe,
                    args.bootstrap_repeats,
                    args.seed + 100 * program_index + fold,
                )
                for k in K_PER_DIRECTION:
                    selected = select_members(evidence, k)
                    rows.append(candidate_record(program, recipe_name, k, fold, selected, scrna_pairs.loc[valid_units]))
    fold_screen = pd.DataFrame(rows)
    grouped_screen, best = choose_hyperparameters(fold_screen)

    frozen_frames = []
    oof_rows = []
    for program_index, program in enumerate(programs):
        chosen = best[best["program"].eq(program)].iloc[0]
        recipe_name = str(chosen["adapter_recipe"])
        k = int(chosen["k_per_direction"])
        final_evidence = add_scrna_evidence(
            discovery_tables[program],
            scrna_pairs,
            ADAPTER_RECIPES[recipe_name],
            args.bootstrap_repeats,
            args.seed + 1000 + program_index,
        )
        frozen = select_members(final_evidence, k).copy()
        frozen["selected_adapter_recipe"] = recipe_name
        frozen["selected_k_per_direction"] = k
        frozen["original_weight"] = frozen["weight"]
        frozen["weight"] = frozen["weight"] * (0.75 + 0.50 * frozen["selection_score"])
        frozen["rank"] = frozen.groupby("direction")["selection_score"].rank(ascending=False, method="first").astype(int)
        frozen_frames.append(frozen)
        for fold in range(args.folds):
            train_units = [unit for unit, assigned in pair_to_fold.items() if assigned != fold]
            valid_units = [unit for unit, assigned in pair_to_fold.items() if assigned == fold]
            evidence = add_scrna_evidence(
                discovery_tables[program],
                scrna_pairs.loc[train_units],
                ADAPTER_RECIPES[recipe_name],
                args.bootstrap_repeats,
                args.seed + 2000 + 100 * program_index + fold,
            )
            selected = select_members(evidence, k)
            scores = program_scores(selected, scrna_pairs.loc[valid_units])
            for pair, score in scores.items():
                oof_rows.append(
                    {
                        "matched_pair_index": pair,
                        "fold": fold,
                        "program": program,
                        "oriented_score": float(score),
                        "delta_score": float(score * discovery_sign.loc[program]),
                    }
                )
    frozen = pd.concat(frozen_frames, ignore_index=True)
    frozen = frozen.sort_values(
        ["program", "direction", "rank"],
        key=lambda values: values.map(program_key) if values.name == "program" else values,
    )
    oof = pd.DataFrame(oof_rows)
    summary_rows = []
    for program, group in oof.groupby("program"):
        oriented = group["oriented_score"].to_numpy()
        summary_rows.append(
            {
                "program": program,
                "n_pairs": len(group),
                "mean_oriented_score": float(oriented.mean()),
                "mean_delta": float(group["delta_score"].mean()),
                "frac_expected_direction": float((oriented > 0).mean()),
                "p_value": safe_wilcoxon(oriented),
            }
        )
    oof_summary = pd.DataFrame(summary_rows).sort_values("program", key=lambda values: values.map(program_key))
    oof_summary["q_value"] = bh_fdr(oof_summary["p_value"])
    fold_screen.to_csv(outdir / f"{args.prefix}_pair_fold_validation_detail.csv", index=False)
    grouped_screen.to_csv(outdir / f"{args.prefix}_hyperparameter_screen.csv", index=False)
    best.to_csv(outdir / f"{args.prefix}_selected_hyperparameters.csv", index=False)
    frozen.to_csv(outdir / f"{args.prefix}_core_program_genes.csv", index=False)
    oof.to_csv(outdir / f"{args.prefix}_scrna_pair_oof_scores.csv", index=False)
    oof_summary.to_csv(outdir / f"{args.prefix}_scrna_pair_oof_summary.csv", index=False)
    with open(outdir / f"{args.prefix}_config.json", "w") as handle:
        json.dump(
            {
                "source_model": args.source_prefix,
                "selection": "discovery bulk plus scRNA pair-level cross-modal calibration",
                "heldout_bulk_used_for_selection": False,
                "scrna_role": "calibration with pair-level cross-validation, not external validation",
                "folds": args.folds,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    print(best[["program", "adapter_recipe", "k_per_direction", "cv_positive_fraction", "cv_objective"]].to_string(index=False))
    print("\nscRNA pair out-of-fold summary")
    print(oof_summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


if __name__ == "__main__":
    main()
