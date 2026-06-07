#!/usr/bin/env python3
"""Leakage-safe upgrade experiments for interpretable MM perturbation programs.

The upgraded model keeps the manuscript-facing program bottleneck while adding:
- compositional perturbation and basal-state encoders;
- signed-NMF initialization for the gene-program dictionary;
- basal-state gated program activities;
- a small low-rank residual expert for context-specific effects;
- block/drug-balanced robust loss;
- control-state coexpression graph regularization;
- multi-seed consensus program export.

Model selection is performed only by internal cross-validation. Held-out bulk
samples and scRNA data are intentionally evaluated by separate scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import torch_model_benchmark as tb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade")
    parser.add_argument("--prefix", default="our_contextual_v5")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--programs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[71, 113, 197])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["matched", "v2_pca", "v4_no_balance", "v5_anchored", "v5_anchored_strong", "v5_frozen_dictionary"],
        choices=[
            "matched",
            "v2_pca",
            "v4_full",
            "v4_no_nmf",
            "v4_no_balance",
            "v4_no_graph",
            "v4_no_residual",
            "v5_anchored",
            "v5_anchored_strong",
            "v5_frozen_dictionary",
        ],
    )
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--selected-variant", default="v5_frozen_dictionary")
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


@dataclass
class ProgramConfig:
    n_programs: int = 10
    residual_rank: int = 4
    hidden_dim: int = 64
    init: str = "signed_nmf"
    balanced_loss: bool = True
    graph_lambda: float = 0.012
    residual_lambda: float = 0.018
    program_l1: float = 0.005
    diversity_lambda: float = 0.045
    activity_lambda: float = 0.002
    cosine_lambda: float = 0.38
    residual_scale: float = 0.22
    graph_neighbors: int = 4
    anchor_lambda: float = 0.0
    freeze_programs: bool = False


def config_for(name: str, n_programs: int) -> ProgramConfig:
    base = ProgramConfig(n_programs=n_programs)
    if name == "v4_full":
        return base
    if name == "v4_no_nmf":
        return replace(base, init="pca")
    if name == "v4_no_balance":
        return replace(base, balanced_loss=False)
    if name == "v4_no_graph":
        return replace(base, graph_lambda=0.0)
    if name == "v4_no_residual":
        return replace(base, residual_rank=0, residual_lambda=0.0)
    if name == "v5_anchored":
        return replace(base, balanced_loss=False, graph_lambda=0.0, anchor_lambda=0.08)
    if name == "v5_anchored_strong":
        return replace(base, balanced_loss=False, graph_lambda=0.0, anchor_lambda=0.20)
    if name == "v5_frozen_dictionary":
        return replace(base, balanced_loss=False, graph_lambda=0.0, freeze_programs=True)
    raise KeyError(name)


class CompositionalFeaturizer:
    """Keep perturbation/covariate features separate from basal-state features."""

    def __init__(self, n_control_pcs: int = 24, random_state: int = 0):
        self.n_control_pcs = n_control_pcs
        self.random_state = random_state
        self.cat_cols = ["drug_token", "drug_class", "dose_label", "phenotype_label", "confidence"]
        self.num_cols = ["time_hours", "has_time"]

    def fit(self, meta: pd.DataFrame, x_control: np.ndarray):
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(meta[self.cat_cols])
        self.num_scaler = StandardScaler().fit(meta[self.num_cols])
        self.ctrl_scaler = StandardScaler().fit(x_control)
        k = min(self.n_control_pcs, x_control.shape[0] - 1, x_control.shape[1])
        self.ctrl_pca = PCA(n_components=k, random_state=self.random_state).fit(self.ctrl_scaler.transform(x_control))
        return self

    def transform(self, meta: pd.DataFrame, x_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cat = self.onehot.transform(meta[self.cat_cols])
        if sparse.issparse(cat):
            cat = cat.toarray()
        num = self.num_scaler.transform(meta[self.num_cols])
        perturb = np.hstack([cat, num]).astype("float32")
        basal = self.ctrl_pca.transform(self.ctrl_scaler.transform(x_control)).astype("float32")
        return perturb, basal


class ContextualSignedProgramNet(nn.Module):
    def __init__(self, n_perturb: int, n_basal: int, n_genes: int, cfg: ProgramConfig):
        super().__init__()
        h = cfg.hidden_dim
        p = cfg.n_programs
        self.cfg = cfg
        self.drug_encoder = nn.Sequential(
            nn.Linear(n_perturb, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.basal_encoder = nn.Sequential(
            nn.Linear(n_basal, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(h * 3, h * 2),
            nn.LayerNorm(h * 2),
            nn.GELU(),
            nn.Dropout(0.06),
            nn.Linear(h * 2, h),
            nn.GELU(),
        )
        self.drug_activity = nn.Linear(h, p)
        self.context_activity = nn.Linear(h, p)
        self.basal_gate = nn.Linear(h, p)
        self.program_interaction = nn.Linear(p, p, bias=False)
        self.scale_head = nn.Sequential(nn.Linear(h, 24), nn.GELU(), nn.Linear(24, 1))
        self.programs = nn.Parameter(torch.randn(p, n_genes) * 0.02)
        if cfg.residual_rank > 0:
            self.residual_activity = nn.Linear(h, cfg.residual_rank)
            self.residual_programs = nn.Parameter(torch.randn(cfg.residual_rank, n_genes) * 0.01)
        else:
            self.residual_activity = None
            self.residual_programs = None

    def forward(self, perturb: torch.Tensor, basal: torch.Tensor):
        drug_h = self.drug_encoder(perturb)
        basal_h = self.basal_encoder(basal)
        joint = self.context_encoder(torch.cat([drug_h, basal_h, drug_h * basal_h], dim=1))
        act = self.drug_activity(drug_h) + torch.sigmoid(self.basal_gate(basal_h)) * self.context_activity(joint)
        act = act + 0.16 * torch.tanh(self.program_interaction(act))
        raw = act @ self.programs
        residual_act = None
        if self.residual_programs is not None:
            residual_act = self.residual_activity(joint)
            raw = raw + self.cfg.residual_scale * (residual_act @ self.residual_programs)
        scale = F.softplus(self.scale_head(joint)) + 0.08
        return raw * scale, act, residual_act


def signed_nmf_init(yz: np.ndarray, n_programs: int, random_state: int) -> np.ndarray:
    x = np.hstack([np.clip(yz, 0, None), np.clip(-yz, 0, None)])
    x = Normalizer(norm="l2").fit_transform(x)
    model = NMF(
        n_components=n_programs,
        init="nndsvda",
        random_state=random_state,
        max_iter=1200,
        solver="mu",
        beta_loss="frobenius",
    )
    model.fit(np.maximum(x, 0))
    h = model.components_
    init = h[:, : yz.shape[1]] - h[:, yz.shape[1] :]
    return normalize_rows(init)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, denom, out=np.zeros_like(x), where=denom > 1e-8).astype("float32")


def init_programs(yz: np.ndarray, cfg: ProgramConfig, random_state: int) -> np.ndarray:
    if cfg.init == "signed_nmf":
        return signed_nmf_init(yz, cfg.n_programs, random_state)
    if cfg.init == "pca":
        k = min(cfg.n_programs, yz.shape[0] - 1, yz.shape[1])
        init = PCA(n_components=k, random_state=random_state).fit(yz).components_
        if k < cfg.n_programs:
            rng = np.random.default_rng(random_state)
            init = np.vstack([init, rng.normal(0, 1, size=(cfg.n_programs - k, yz.shape[1]))])
        return normalize_rows(init)
    rng = np.random.default_rng(random_state)
    return normalize_rows(rng.normal(0, 1, size=(cfg.n_programs, yz.shape[1])))


def coexpression_edges(x_control: np.ndarray, neighbors: int) -> np.ndarray:
    """Construct a compact positive-correlation graph among genes."""
    gene_profiles = StandardScaler().fit_transform(x_control).T
    n_neighbors = min(neighbors + 1, len(gene_profiles))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(gene_profiles)
    distance, index = nn_model.kneighbors(gene_profiles)
    edges: set[tuple[int, int]] = set()
    for source in range(len(gene_profiles)):
        for dist, target in zip(distance[source, 1:], index[source, 1:]):
            if dist < 1.0:
                edges.add(tuple(sorted((int(source), int(target)))))
    return np.asarray(sorted(edges), dtype=np.int64)


def context_weights(meta: pd.DataFrame, balanced: bool) -> np.ndarray:
    if not balanced:
        return np.ones(len(meta), dtype="float32")
    block_n = meta["block_id"].map(meta["block_id"].value_counts()).astype(float)
    drug_n = meta["drug_token"].map(meta["drug_token"].value_counts()).astype(float)
    weights = block_n.pow(-0.5) * drug_n.pow(-0.25)
    weights = weights / weights.mean()
    return weights.clip(lower=0.35, upper=4.0).to_numpy(dtype="float32")


class AdvancedProgramTorch:
    def __init__(self, cfg: ProgramConfig, epochs: int, random_state: int, device: torch.device, name: str = "v4_full"):
        self.cfg = cfg
        self.epochs = epochs
        self.random_state = random_state
        self.device = device
        self.name = f"OUR_contextual_signed_program_{name}"

    def fit(self, x_control: np.ndarray, y_delta: np.ndarray, meta: pd.DataFrame):
        tb.set_seed(self.random_state)
        self.feat = CompositionalFeaturizer(random_state=self.random_state).fit(meta, x_control)
        perturb, basal = self.feat.transform(meta, x_control)
        self.y_scaler = StandardScaler().fit(y_delta)
        yz = self.y_scaler.transform(y_delta).astype("float32")
        self.model = ContextualSignedProgramNet(perturb.shape[1], basal.shape[1], yz.shape[1], self.cfg)
        init = torch.tensor(init_programs(yz, self.cfg, self.random_state))
        with torch.no_grad():
            self.model.programs.copy_(init)
        if self.cfg.freeze_programs:
            self.model.programs.requires_grad_(False)
        edge_array = coexpression_edges(x_control, self.cfg.graph_neighbors) if self.cfg.graph_lambda > 0 else np.empty((0, 2), dtype=np.int64)
        edge_index = torch.tensor(edge_array.T, dtype=torch.long, device=self.device) if len(edge_array) else None
        weights = torch.tensor(context_weights(meta, self.cfg.balanced_loss), device=self.device)
        perturb_t = torch.tensor(perturb, device=self.device)
        basal_t = torch.tensor(basal, device=self.device)
        y_t = torch.tensor(yz, device=self.device)
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=7e-4, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred, act, residual_act = self.model(perturb_t, basal_t)
            huber = F.smooth_l1_loss(pred, y_t, reduction="none").mean(dim=1)
            cos = 1 - F.cosine_similarity(pred, y_t, dim=1)
            loss = weighted_mean(huber, weights) + self.cfg.cosine_lambda * weighted_mean(cos, weights)
            loss = loss + self.cfg.program_l1 * self.model.programs.abs().mean()
            loss = loss + self.cfg.activity_lambda * act.abs().mean()
            normalized = F.normalize(self.model.programs, dim=1)
            gram = normalized @ normalized.T
            eye = torch.eye(gram.shape[0], device=gram.device)
            loss = loss + self.cfg.diversity_lambda * ((gram - eye) ** 2).mean()
            if edge_index is not None:
                graph_diff = self.model.programs[:, edge_index[0]] - self.model.programs[:, edge_index[1]]
                loss = loss + self.cfg.graph_lambda * (graph_diff**2).mean()
            if self.cfg.anchor_lambda > 0:
                anchor = init.to(self.device)
                loss = loss + self.cfg.anchor_lambda * (1 - F.cosine_similarity(self.model.programs, anchor, dim=1).mean())
            if residual_act is not None:
                loss = loss + self.cfg.residual_lambda * (
                    residual_act.abs().mean() + self.model.residual_programs.abs().mean()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            if value < best_loss:
                best_loss = value
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x_control: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        perturb, basal = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            pred, _, _ = self.model(torch.tensor(perturb, device=self.device), torch.tensor(basal, device=self.device))
        return self.y_scaler.inverse_transform(pred.cpu().numpy()).astype("float32")

    def program_activity(self, x_control: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        perturb, basal = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            _, act, _ = self.model(torch.tensor(perturb, device=self.device), torch.tensor(basal, device=self.device))
        return act.cpu().numpy()

    def program_matrix_original_scale(self) -> np.ndarray:
        return self.model.programs.detach().cpu().numpy() * self.y_scaler.scale_[None, :]


def make_model(name: str, args: argparse.Namespace, device: torch.device, seed: int):
    if name == "matched":
        return tb.MatchedDrugMean()
    if name == "v2_pca":
        return tb.OurSharedProgramTorch(args.programs, args.epochs, seed, device)
    return AdvancedProgramTorch(config_for(name, args.programs), args.epochs, seed, device, name=name)


def splitters(meta: pd.DataFrame, random_state: int, n_splits: int):
    n_splits = max(3, n_splits)
    yield "random", list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(len(meta))))
    for name, col in [("cellline", "entity_norm"), ("study", "block_id"), ("drug", "drug_token")]:
        groups = meta[col].astype(str).to_numpy()
        yield name, list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(np.arange(len(meta)), groups=groups))


def run_cv(args: argparse.Namespace, device: torch.device, meta: pd.DataFrame, x_control: np.ndarray, y_delta: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in args.seeds:
        for split_name, folds in splitters(meta, seed, args.n_splits):
            print(f"[seed={seed}] split={split_name}", flush=True)
            for variant in args.variants:
                print(f"  variant={variant}", flush=True)
                for fold, (train_idx, test_idx) in enumerate(folds, start=1):
                    model = make_model(variant, args, device, seed + 1000 * fold)
                    model.fit(x_control[train_idx], y_delta[train_idx], meta.iloc[train_idx].reset_index(drop=True))
                    pred = model.predict(x_control[test_idx], meta.iloc[test_idx].reset_index(drop=True))
                    row = tb.metrics(y_delta[test_idx], pred)
                    row.update(retrieval_metrics(y_delta[test_idx], pred, meta.iloc[test_idx]["drug_token"].astype(str).to_numpy()))
                    row.update(
                        {
                            "seed": seed,
                            "split": split_name,
                            "variant": variant,
                            "model": model.name,
                            "fold": fold,
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_cv(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse",
        "mae",
        "cosine",
        "pearson",
        "spearman",
        "top50_overlap",
        "top100_overlap",
        "top100_sign_agreement",
        "magnitude_fidelity",
        "context_retrieval_top1",
        "context_retrieval_percentile",
        "drug_retrieval_top1",
        "drug_retrieval_percentile",
    ]
    summary = rows.groupby(["split", "variant", "model"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns]
    summary = summary.reset_index()
    summary["selection_score"] = (
        summary["cosine_mean"]
        + 0.45 * summary["pearson_mean"]
        + 0.20 * summary["top100_overlap_mean"]
        - 0.025 * summary["rmse_mean"]
    )
    return summary.sort_values(["split", "selection_score"], ascending=[True, False])


def retrieval_metrics(y_true: np.ndarray, y_pred: np.ndarray, drug_labels: np.ndarray) -> dict[str, float]:
    """Quantify whether predictions retain context- and drug-specific signals."""
    true_norm = y_true / np.maximum(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-8)
    pred_norm = y_pred / np.maximum(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-8)
    similarity = pred_norm @ true_norm.T
    context_top1 = []
    context_percentile = []
    drug_top1 = []
    drug_percentile = []
    for i in range(len(y_true)):
        order = np.argsort(-similarity[i])
        rank = int(np.flatnonzero(order == i)[0]) + 1
        denom = max(len(order) - 1, 1)
        context_top1.append(rank == 1)
        context_percentile.append(1.0 - (rank - 1) / denom)
        relevant = np.flatnonzero(drug_labels == drug_labels[i])
        relevant_ranks = [int(np.flatnonzero(order == j)[0]) + 1 for j in relevant]
        best_rank = min(relevant_ranks)
        drug_top1.append(best_rank == 1)
        drug_percentile.append(1.0 - (best_rank - 1) / denom)
    return {
        "context_retrieval_top1": float(np.mean(context_top1)),
        "context_retrieval_percentile": float(np.mean(context_percentile)),
        "drug_retrieval_top1": float(np.mean(drug_top1)),
        "drug_retrieval_percentile": float(np.mean(drug_percentile)),
    }


def align_programs(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = normalize_rows(reference)
    cand = normalize_rows(candidate)
    similarity = ref @ cand.T
    ref_idx, cand_idx = linear_sum_assignment(-np.abs(similarity))
    order = cand_idx[np.argsort(ref_idx)]
    signs = np.sign(similarity[np.arange(len(reference)), order])
    signs[signs == 0] = 1
    return candidate[order] * signs[:, None], order, signs


def program_gene_table(program_matrix: np.ndarray, genes: list[str], top_n: int = 80) -> pd.DataFrame:
    rows = []
    for i, weights in enumerate(program_matrix):
        series = pd.Series(weights, index=genes)
        for direction, values in [("up", series.sort_values(ascending=False)), ("down", series.sort_values(ascending=True))]:
            for rank, (gene, weight) in enumerate(values.head(top_n).items(), start=1):
                rows.append(
                    {
                        "program": f"Program_{i + 1}",
                        "direction": direction,
                        "rank": rank,
                        "gene": gene,
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows)


def train_consensus(
    args: argparse.Namespace,
    device: torch.device,
    meta: pd.DataFrame,
    x_control: np.ndarray,
    y_delta: np.ndarray,
    genes: list[str],
    outdir: Path,
) -> None:
    cfg = config_for(args.selected_variant, args.programs)
    models = []
    predictions = []
    activities = []
    matrices = []
    for seed in args.final_seeds:
        print(f"[final consensus] seed={seed}", flush=True)
        model = AdvancedProgramTorch(cfg, args.epochs * 2, seed, device, name=args.selected_variant)
        model.fit(x_control, y_delta, meta)
        models.append(model)
        predictions.append(model.predict(x_control, meta))
        activities.append(model.program_activity(x_control, meta))
        matrices.append(model.program_matrix_original_scale())
    reference = matrices[0]
    aligned_matrix = [reference]
    aligned_activity = [activities[0]]
    stability_rows = []
    for seed, matrix, activity in zip(args.final_seeds[1:], matrices[1:], activities[1:]):
        aligned, order, signs = align_programs(reference, matrix)
        aligned_matrix.append(aligned)
        aligned_activity.append(activity[:, order] * signs[None, :])
        ref_norm = normalize_rows(reference)
        aligned_norm = normalize_rows(aligned)
        for i in range(cfg.n_programs):
            top_ref = set(np.argsort(np.abs(reference[i]))[-50:])
            top_alt = set(np.argsort(np.abs(aligned[i]))[-50:])
            stability_rows.append(
                {
                    "seed": seed,
                    "program": f"Program_{i + 1}",
                    "signed_cosine": float(np.sum(ref_norm[i] * aligned_norm[i])),
                    "top50_jaccard": float(len(top_ref & top_alt) / len(top_ref | top_alt)),
                }
            )
    consensus_matrix = np.mean(aligned_matrix, axis=0)
    consensus_activity = np.mean(aligned_activity, axis=0)
    consensus_prediction = np.mean(predictions, axis=0)
    prefix = args.prefix
    pd.DataFrame([tb.metrics(y_delta, consensus_prediction)]).to_csv(outdir / f"{prefix}_fit_metrics.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(outdir / f"{prefix}_program_seed_stability.csv", index=False)
    gene_table = program_gene_table(consensus_matrix, genes)
    gene_table.to_csv(outdir / f"{prefix}_program_genes.csv", index=False)
    program_cols = [f"Program_{i + 1}" for i in range(cfg.n_programs)]
    activity_df = pd.DataFrame(consensus_activity, columns=program_cols)
    activity_df.insert(0, "context_id", meta["context_id"].to_numpy())
    activity_df = activity_df.merge(meta, on="context_id", how="left")
    activity_df.to_csv(outdir / f"{prefix}_program_activity.csv", index=False)
    activity_df.groupby(["drug_token", "drug_name", "drug_class"], as_index=False)[program_cols].mean().to_csv(
        outdir / f"{prefix}_program_drug_activity.csv", index=False
    )
    np.save(outdir / f"{prefix}_program_matrix.npy", consensus_matrix)
    np.save(outdir / f"{prefix}_training_prediction.npy", consensus_prediction)
    joblib.dump(models, outdir / f"{prefix}_seed_models.joblib")
    (outdir / f"{prefix}_config.json").write_text(
        json.dumps(
            {
                "variant": args.selected_variant,
                "config": asdict(cfg),
                "final_seeds": args.final_seeds,
                "n_training_contexts": len(meta),
                "n_genes": len(genes),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    safe_mkdir(outdir)
    device = tb.choose_device(args.device)
    print("Device:", device, flush=True)
    meta, x_control, _, y_delta, genes = tb.load_arrays(args)
    meta.to_csv(outdir / "upgrade_training_contrasts.csv", index=False)
    pd.Series(genes, name="gene").to_csv(outdir / "upgrade_genes.csv", index=False)
    if not args.skip_cv:
        fold_metrics = run_cv(args, device, meta, x_control, y_delta)
        fold_metrics.to_csv(outdir / "upgrade_cv_fold_metrics.csv", index=False)
        summarize_cv(fold_metrics).to_csv(outdir / "upgrade_cv_summary.csv", index=False)
    if not args.skip_final:
        train_consensus(args, device, meta, x_control, y_delta, genes, outdir)
    print(f"Done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
