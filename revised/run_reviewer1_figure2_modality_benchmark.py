#!/usr/bin/env python3
"""Calculate Figure 2-style benchmark metrics for modality controls.

This reviewer-response script does not create figures or overwrite manuscript
files. It reuses the Figure 2 perturbation-prediction metrics and appends two
controls to the existing benchmark table:

1. Bulk-only shared-program predictor:
   supervised bulk perturbation model trained only on bulk contrasts.
2. scRNA-only dictionary predictor:
   gene-space dictionary learned from paired scRNA deltas, with fold-internal
   bulk training data used only to map sample context to scRNA-derived program
   activities.

The output is intended as a feasibility test for reviewer-response analyses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "bulk_pre_sc"
sys.path.append(str(SOURCE_DIR))

import torch_model_benchmark as tb  # noqa: E402
import train_v14_multimodal_dictionary as v14  # noqa: E402


METRIC_COLS = [
    "rmse",
    "mae",
    "cosine",
    "pearson",
    "spearman",
    "top50_overlap",
    "top100_overlap",
    "top100_sign_agreement",
    "magnitude_fidelity",
]

FIGURE2_METHOD_ORDER = [
    "matched_DEG_drug_mean",
    "official_scgen_latent_arithmetic",
    "official_cpa_compositional_autoencoder",
    "official_cellot_neural_ot",
    "official_trvae_conditional_vae",
    "official_scvidr_regressed_vae",
    "OUR_hybrid_teacher_program_v3",
    "bulk_only_shared_program",
    "scrna_only_dictionary_ridge",
]

METHOD_LABELS = {
    "matched_DEG_drug_mean": "Drug-mean baseline",
    "official_scgen_latent_arithmetic": "scGen",
    "official_cpa_compositional_autoencoder": "CPA",
    "official_cellot_neural_ot": "CellOT",
    "official_trvae_conditional_vae": "trVAE",
    "official_scvidr_regressed_vae": "scVIDR",
    "OUR_hybrid_teacher_program_v3": "PRISM-MM",
    "bulk_only_shared_program": "Bulk-only",
    "scrna_only_dictionary_ridge": "scRNA-only",
}

SPLIT_LABELS = {
    "random_5fold": "Random",
    "leave_entity_out": "Cell source",
    "leave_block_out": "Study",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--existing-figure2", default="bulk_pre_sc/main_figure_source_data/figure_2_benchmark_source_metrics.csv")
    parser.add_argument(
        "--scrna-gene-deltas",
        default="bulk_pre_sc/model_upgrade_v9_final/gene_direction_validation/scrna_all_adaptive_gene_delta_by_pair.csv",
    )
    parser.add_argument("--outdir", default="/tmp/prism_review1_test/figure2_modality_benchmark")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--programs", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--skip-bulk-only", action="store_true")
    parser.add_argument("--skip-scrna-only", action="store_true")
    return parser.parse_args()


def add_direction_score(fold: pd.DataFrame) -> pd.DataFrame:
    scored = fold.copy()
    scored["cosine_scaled"] = ((scored["cosine"].clip(-1, 1) + 1) / 2).astype(float)
    scored["pearson_scaled"] = ((scored["pearson"].clip(-1, 1) + 1) / 2).astype(float)
    scored["spearman_scaled"] = ((scored["spearman"].clip(-1, 1) + 1) / 2).astype(float)
    scored["correlation_score"] = scored[["cosine_scaled", "pearson_scaled", "spearman_scaled"]].mean(axis=1)
    scored["responsive_gene_score"] = scored[["top50_overlap", "top100_overlap"]].mean(axis=1)
    scored["direction_score"] = scored[
        ["correlation_score", "responsive_gene_score", "top100_sign_agreement"]
    ].mean(axis=1)
    return scored


def summarize(fold: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold.groupby(["split", "split_label", "model", "method_label"], observed=True)[METRIC_COLS + ["direction_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([part for part in col if part]) for col in summary.columns]
    return summary


def load_arrays(args: argparse.Namespace):
    tb_args = argparse.Namespace(
        expr=str(ROOT / args.expr),
        article_dir=str(ROOT / args.article_dir),
        n_genes=args.n_genes,
        seed=args.random_state,
    )
    return tb.load_arrays(tb_args)


def run_bulk_only(args: argparse.Namespace, meta: pd.DataFrame, x_control: np.ndarray, y_delta: np.ndarray) -> pd.DataFrame:
    rows = []
    device = tb.choose_device(args.device)
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"Bulk-only split: {split_name}", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            model = tb.OurSharedProgramTorch(
                n_programs=args.programs,
                epochs=args.epochs,
                random_state=args.random_state + 7919 + fold,
                device=device,
            )
            model.fit(x_control[train_idx], y_delta[train_idx], meta.iloc[train_idx].reset_index(drop=True))
            pred = model.predict(x_control[test_idx], meta.iloc[test_idx].reset_index(drop=True))
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "bulk_only_shared_program",
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    "source": "reviewer_response",
                    "n_genes": int(y_delta.shape[1]),
                    "n_train": int(len(train_idx)),
                    "unseen_drug_predictions": int(
                        len(set(meta.iloc[test_idx]["drug_token"].astype(str)) - set(meta.iloc[train_idx]["drug_token"].astype(str)))
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


class ScRNAOnlyDictionaryRidge:
    name = "scrna_only_dictionary_ridge"

    def __init__(self, components: np.ndarray, mask: np.ndarray, scrna_scaler: StandardScaler, alpha: float, random_state: int):
        self.components = components.astype("float32")
        self.mask = mask.astype(bool)
        self.scrna_scaler = scrna_scaler
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, x_control: np.ndarray, y_delta: np.ndarray, meta: pd.DataFrame):
        self.feat = tb.FeatureFeaturizer(random_state=self.random_state).fit(meta, x_control)
        x_feat = self.feat.transform(meta, x_control)
        y_mask = y_delta[:, self.mask]
        y_z = self.scrna_scaler.transform(y_mask).astype("float32")
        activity = y_z @ self.components.T
        self.regressor = Ridge(alpha=self.alpha, fit_intercept=True)
        self.regressor.fit(x_feat, activity)
        return self

    def predict(self, x_control: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        x_feat = self.feat.transform(meta, x_control)
        activity = self.regressor.predict(x_feat)
        pred_z = activity @ self.components
        pred_mask = self.scrna_scaler.inverse_transform(pred_z)
        pred = np.zeros((len(meta), len(self.mask)), dtype="float32")
        pred[:, self.mask] = pred_mask.astype("float32")
        return pred


def build_scrna_dictionary(args: argparse.Namespace, genes: list[str]) -> tuple[np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    scrna_raw, scrna_mask, scrna_pairs = v14.scrna_matrix(ROOT / args.scrna_gene_deltas, genes)
    mask = scrna_mask > 0
    if mask.sum() < args.programs:
        raise ValueError(f"Only {mask.sum()} scRNA-overlap genes are available, fewer than {args.programs} components.")
    scaler = StandardScaler().fit(scrna_raw[:, mask])
    z = scaler.transform(scrna_raw[:, mask]).astype("float32")
    pca = PCA(n_components=args.programs, random_state=args.random_state)
    pca.fit(z)
    info = pd.DataFrame(
        {
            "n_scrna_pairs": [len(scrna_pairs)],
            "n_scrna_delta_genes_total": [int((scrna_mask > 0).sum())],
            "n_overlap_benchmark_genes": [int(mask.sum())],
            "n_benchmark_genes": [len(genes)],
            "n_components": [args.programs],
            "explained_variance_sum": [float(pca.explained_variance_ratio_.sum())],
        }
    )
    return pca.components_.astype("float32"), mask.astype(bool), scaler, info


def run_scrna_only(
    args: argparse.Namespace,
    meta: pd.DataFrame,
    x_control: np.ndarray,
    y_delta: np.ndarray,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components, mask, scaler, info = build_scrna_dictionary(args, genes)
    rows = []
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"scRNA-only split: {split_name}", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            model = ScRNAOnlyDictionaryRidge(
                components=components,
                mask=mask,
                scrna_scaler=scaler,
                alpha=args.ridge_alpha,
                random_state=args.random_state + 104729 + fold,
            ).fit(x_control[train_idx], y_delta[train_idx], meta.iloc[train_idx].reset_index(drop=True))
            pred = model.predict(x_control[test_idx], meta.iloc[test_idx].reset_index(drop=True))
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "scrna_only_dictionary_ridge",
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    "source": "reviewer_response",
                    "n_genes": int(y_delta.shape[1]),
                    "n_train": int(len(train_idx)),
                    "unseen_drug_predictions": int(
                        len(set(meta.iloc[test_idx]["drug_token"].astype(str)) - set(meta.iloc[train_idx]["drug_token"].astype(str)))
                    ),
                    "n_scrna_overlap_genes": int(mask.sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), info


def load_existing(args: argparse.Namespace) -> pd.DataFrame:
    existing = pd.read_csv(ROOT / args.existing_figure2)
    existing = existing[existing["model"].isin(FIGURE2_METHOD_ORDER)].copy()
    keep = [col for col in existing.columns if col in set(METRIC_COLS + ["split", "model", "fold", "n_test", "source", "n_genes", "n_train", "unseen_drug_predictions"])]
    return existing[keep].copy()


def attach_labels(fold: pd.DataFrame) -> pd.DataFrame:
    out = fold.copy()
    out["method_label"] = out["model"].map(METHOD_LABELS).fillna(out["model"])
    out["split_label"] = out["split"].map(SPLIT_LABELS).fillna(out["split"])
    out["model"] = pd.Categorical(out["model"], FIGURE2_METHOD_ORDER, ordered=True)
    return out.sort_values(["split", "model", "fold"])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta, x_control, _, y_delta, genes = load_arrays(args)
    frames = []
    if not args.skip_bulk_only:
        frames.append(run_bulk_only(args, meta, x_control, y_delta))
    if not args.skip_scrna_only:
        scrna_fold, scrna_info = run_scrna_only(args, meta, x_control, y_delta, genes)
        frames.append(scrna_fold)
        scrna_info.to_csv(outdir / "scrna_only_dictionary_info.csv", index=False)

    new_fold = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    new_fold = attach_labels(add_direction_score(new_fold)) if len(new_fold) else new_fold
    new_fold.to_csv(outdir / "modality_control_fold_metrics.csv", index=False)

    existing = attach_labels(add_direction_score(load_existing(args)))
    combined = pd.concat([existing, new_fold], ignore_index=True, sort=False)
    combined = attach_labels(add_direction_score(combined))
    combined.to_csv(outdir / "combined_figure2_modality_fold_metrics.csv", index=False)

    summary = summarize(combined)
    summary.to_csv(outdir / "combined_figure2_modality_summary.csv", index=False)

    score = (
        combined.groupby(["model", "method_label"], observed=True)["direction_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    score.to_csv(outdir / "overall_program_recovery_score.csv", index=False)

    print("\nOverall Figure 2-style program-recovery score")
    print(score.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSplit-level summary for PRISM-MM and modality controls")
    focus = summary[summary["model"].astype(str).isin(["OUR_hybrid_teacher_program_v3", "bulk_only_shared_program", "scrna_only_dictionary_ridge"])]
    cols = [
        "split_label",
        "method_label",
        "cosine_mean",
        "pearson_mean",
        "spearman_mean",
        "top50_overlap_mean",
        "top100_overlap_mean",
        "top100_sign_agreement_mean",
        "magnitude_fidelity_mean",
        "direction_score_mean",
    ]
    print(focus[cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote: {outdir}")


if __name__ == "__main__":
    main()
