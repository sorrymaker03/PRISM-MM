#!/usr/bin/env python3
"""Sparse attention-MoE perturbation model with stability-selected core programs.

V8 keeps the manuscript-facing drug-minus-control prediction task while
separating:
- sparse signed core programs for shared biological conclusions;
- private programs for context-dependent response;
- a routed low-rank expert residual for remaining heterogeneity.

Only discovery bulk data are used for model selection and program freezing.
Held-out bulk and scRNA evaluation are intentionally handled by separate
scripts.
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
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import advanced_program_model as adv
import torch_model_benchmark as tb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade_v8")
    parser.add_argument("--prefix", default="our_sparse_moe_v8")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[71, 113, 197])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["v5_frozen_dictionary", "v8_sparse_moe", "v8_sparse_moe_light", "v8_sparse_no_moe"],
        choices=[
            "matched",
            "v2_pca",
            "v5_frozen_dictionary",
            "v8_sparse_moe",
            "v8_sparse_moe_light",
            "v8_sparse_moe_strong",
            "v8_sparse_no_moe",
            "v8_sparse_no_invariance",
            "v8_dual_path",
            "v8_dual_path_no_moe",
            "v8_dual_path_strong",
        ],
    )
    parser.add_argument("--selected-variant", default="v8_sparse_moe")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--min-core", type=int, default=3)
    parser.add_argument("--max-core", type=int, default=5)
    parser.add_argument("--min-seed-cosine", type=float, default=0.72)
    parser.add_argument("--min-top50-jaccard", type=float, default=0.22)
    parser.add_argument("--min-drug-positive-fraction", type=float, default=0.60)
    parser.add_argument("--min-study-positive-fraction", type=float, default=0.55)
    parser.add_argument("--min-gene-selection-probability", type=float, default=0.67)
    return parser.parse_args()


@dataclass
class SparseMoEConfig:
    n_core_candidates: int = 8
    n_private: int = 8
    n_experts: int = 3
    expert_rank: int = 3
    hidden_dim: int = 64
    attention_heads: int = 4
    core_scale: float = 0.42
    private_scale: float = 0.82
    expert_scale: float = 0.16
    cosine_lambda: float = 0.38
    core_l0_lambda: float = 0.018
    core_weight_l1: float = 0.004
    core_activity_lambda: float = 0.0015
    private_l1: float = 0.005
    private_activity_lambda: float = 0.002
    diversity_lambda: float = 0.045
    orthogonal_lambda: float = 0.070
    expert_lambda: float = 0.020
    router_balance_lambda: float = 0.010
    conditional_invariance_lambda: float = 0.025
    hard_concrete_beta: float = 0.67
    hard_concrete_gamma: float = -0.10
    hard_concrete_zeta: float = 1.10
    use_predictor_backbone: bool = False
    backbone_residual_lambda: float = 0.018
    mechanism_reconstruction_lambda: float = 0.24
    distillation_lambda: float = 0.12


def config_for(name: str) -> SparseMoEConfig:
    base = SparseMoEConfig()
    if name == "v8_sparse_moe":
        return base
    if name == "v8_sparse_moe_light":
        return replace(base, core_l0_lambda=0.010)
    if name == "v8_sparse_moe_strong":
        return replace(base, core_l0_lambda=0.032)
    if name == "v8_sparse_no_moe":
        return replace(base, n_experts=0, expert_rank=0, expert_scale=0.0)
    if name == "v8_sparse_no_invariance":
        return replace(base, conditional_invariance_lambda=0.0)
    if name == "v8_dual_path":
        return replace(base, use_predictor_backbone=True, core_scale=0.18, private_scale=0.30, expert_scale=0.08)
    if name == "v8_dual_path_no_moe":
        return replace(base, use_predictor_backbone=True, core_scale=0.18, private_scale=0.30, n_experts=0, expert_rank=0, expert_scale=0.0)
    if name == "v8_dual_path_strong":
        return replace(base, use_predictor_backbone=True, core_scale=0.18, private_scale=0.30, expert_scale=0.08, core_l0_lambda=0.032)
    raise KeyError(name)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, denom, out=np.zeros_like(x), where=denom > 1e-8).astype("float32")


def orient_dictionary(dictionary: np.ndarray, y_delta: np.ndarray) -> np.ndarray:
    oriented = dictionary.copy()
    projection = y_delta @ normalize_rows(oriented).T
    for index in range(len(oriented)):
        if np.nanmedian(projection[:, index]) < 0:
            oriented[index] *= -1
    return oriented


def align_programs(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarity = normalize_rows(reference) @ normalize_rows(candidate).T
    ref_index, candidate_index = linear_sum_assignment(-np.abs(similarity))
    order = candidate_index[np.argsort(ref_index)]
    signs = np.sign(similarity[np.arange(len(reference)), order])
    signs[signs == 0] = 1
    return candidate[order] * signs[:, None], order, signs


def initialize_gate_logits(dictionary: np.ndarray) -> np.ndarray:
    absolute = np.abs(dictionary)
    scale = np.quantile(absolute, 0.94, axis=1, keepdims=True)
    scaled = np.divide(absolute, np.maximum(scale, 1e-8))
    return np.clip(-4.2 + 3.6 * scaled, -4.2, 1.8).astype("float32")


class SparseAttentionMoENet(nn.Module):
    def __init__(
        self,
        n_perturb: int,
        n_basal: int,
        n_genes: int,
        cfg: SparseMoEConfig,
        dictionary: np.ndarray,
        backbone_dictionary: np.ndarray | None = None,
    ):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.drug_encoder = nn.Sequential(
            nn.Linear(n_perturb, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.07),
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
        self.interaction_encoder = nn.Sequential(nn.Linear(h * 2, h), nn.LayerNorm(h), nn.GELU())
        self.attention = nn.MultiheadAttention(h, cfg.attention_heads, dropout=0.05, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.LayerNorm(h * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(h * 2, h),
            nn.GELU(),
        )
        self.core_activity = nn.Linear(h, cfg.n_core_candidates)
        self.core_gate = nn.Linear(h, cfg.n_core_candidates)
        self.core_weights = nn.Parameter(torch.tensor(dictionary, dtype=torch.float32))
        self.core_log_alpha = nn.Parameter(torch.tensor(initialize_gate_logits(dictionary)))
        self.private_activity = nn.Linear(h, cfg.n_private)
        self.private_programs = nn.Parameter(torch.randn(cfg.n_private, n_genes) * 0.02)
        if backbone_dictionary is not None:
            backbone_cfg = adv.config_for("v5_frozen_dictionary", len(backbone_dictionary))
            self.predictor_backbone = adv.ContextualSignedProgramNet(n_perturb, n_basal, n_genes, backbone_cfg)
            with torch.no_grad():
                self.predictor_backbone.programs.copy_(torch.tensor(backbone_dictionary))
            self.predictor_backbone.programs.requires_grad_(False)
        else:
            self.predictor_backbone = None
        if cfg.n_experts > 0:
            self.expert_router = nn.Linear(h, cfg.n_experts)
            self.expert_activity = nn.Linear(h, cfg.n_experts * cfg.expert_rank)
            self.expert_programs = nn.Parameter(torch.randn(cfg.n_experts, cfg.expert_rank, n_genes) * 0.01)
        else:
            self.expert_router = None
            self.expert_activity = None
            self.expert_programs = None

    def hard_concrete_gate(self) -> torch.Tensor:
        cfg = self.cfg
        if self.training:
            uniform = torch.rand_like(self.core_log_alpha).clamp(1e-6, 1 - 1e-6)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            soft = torch.sigmoid((logistic + self.core_log_alpha) / cfg.hard_concrete_beta)
        else:
            soft = torch.sigmoid(self.core_log_alpha)
        stretched = soft * (cfg.hard_concrete_zeta - cfg.hard_concrete_gamma) + cfg.hard_concrete_gamma
        return stretched.clamp(0.0, 1.0)

    def expected_gate_probability(self) -> torch.Tensor:
        cfg = self.cfg
        offset = cfg.hard_concrete_beta * np.log(-cfg.hard_concrete_gamma / cfg.hard_concrete_zeta)
        return torch.sigmoid(self.core_log_alpha - offset)

    def effective_core_programs(self) -> torch.Tensor:
        return self.core_weights * self.hard_concrete_gate()

    def encode(self, perturb: torch.Tensor, basal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drug_h = self.drug_encoder(perturb)
        basal_h = self.basal_encoder(basal)
        interaction_h = self.interaction_encoder(torch.cat([drug_h * basal_h, torch.abs(drug_h - basal_h)], dim=1))
        tokens = torch.stack([drug_h, basal_h, interaction_h], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        pooled = attended.mean(dim=1)
        fused = self.fusion(torch.cat([pooled, drug_h, basal_h, interaction_h], dim=1))
        return drug_h, fused

    def forward(self, perturb: torch.Tensor, basal: torch.Tensor):
        drug_h, fused = self.encode(perturb, basal)
        core_act = F.softplus(self.core_activity(drug_h) + torch.sigmoid(self.core_gate(fused)) * self.core_activity(fused))
        core_programs = self.effective_core_programs()
        mechanism_prediction = self.cfg.core_scale * (core_act @ core_programs)
        backbone_act = None
        backbone_residual_act = None
        if self.predictor_backbone is not None:
            prediction, backbone_act, backbone_residual_act = self.predictor_backbone(perturb, basal)
        else:
            prediction = mechanism_prediction
        private_act = self.private_activity(fused)
        mechanism_prediction = mechanism_prediction + self.cfg.private_scale * (private_act @ self.private_programs)
        router = None
        expert_act = None
        if self.expert_programs is not None:
            router = torch.softmax(self.expert_router(fused), dim=1)
            expert_act = self.expert_activity(fused).reshape(-1, self.cfg.n_experts, self.cfg.expert_rank)
            expert_delta = torch.einsum("be,ber,erg->bg", router, expert_act, self.expert_programs)
            mechanism_prediction = mechanism_prediction + self.cfg.expert_scale * expert_delta
        if self.predictor_backbone is None:
            prediction = mechanism_prediction
        return prediction, mechanism_prediction, core_act, private_act, router, expert_act, backbone_act, backbone_residual_act


def conditional_study_invariance(core_act: torch.Tensor, group_index: list[np.ndarray], drug_groups: list[list[int]]) -> torch.Tensor:
    if not group_index:
        return core_act.new_tensor(0.0)
    means = [core_act[torch.as_tensor(index, device=core_act.device)].mean(dim=0) for index in group_index]
    losses = []
    for group_ids in drug_groups:
        if len(group_ids) > 1:
            values = torch.stack([means[index] for index in group_ids])
            losses.append(((values - values.mean(dim=0, keepdim=True)) ** 2).mean())
    return torch.stack(losses).mean() if losses else core_act.new_tensor(0.0)


def make_invariance_groups(meta: pd.DataFrame) -> tuple[list[np.ndarray], list[list[int]]]:
    groups = []
    records = []
    for (drug, block), index in meta.groupby(["drug_token", "block_id"]).groups.items():
        groups.append(np.asarray(list(index), dtype=int))
        records.append({"drug_token": drug, "block_id": block, "group_index": len(groups) - 1})
    frame = pd.DataFrame(records)
    drug_groups = [subset["group_index"].astype(int).tolist() for _, subset in frame.groupby("drug_token")]
    return groups, drug_groups


class SparseMoETorch:
    def __init__(self, cfg: SparseMoEConfig, epochs: int, random_state: int, device: torch.device, name: str):
        self.cfg = cfg
        self.epochs = epochs
        self.random_state = random_state
        self.device = device
        self.name = f"OUR_sparse_attention_moe_{name}"

    def fit(self, x_control: np.ndarray, y_delta: np.ndarray, meta: pd.DataFrame):
        tb.set_seed(self.random_state)
        self.feat = adv.CompositionalFeaturizer(random_state=self.random_state).fit(meta, x_control)
        perturb, basal = self.feat.transform(meta, x_control)
        self.y_scaler = StandardScaler().fit(y_delta)
        yz = self.y_scaler.transform(y_delta).astype("float32")
        dictionary = orient_dictionary(adv.signed_nmf_init(yz, self.cfg.n_core_candidates, self.random_state), yz)
        backbone_dictionary = None
        if self.cfg.use_predictor_backbone:
            backbone_dictionary = orient_dictionary(adv.signed_nmf_init(yz, 10, self.random_state + 503), yz)
        self.model = SparseAttentionMoENet(
            perturb.shape[1],
            basal.shape[1],
            yz.shape[1],
            self.cfg,
            dictionary,
            backbone_dictionary=backbone_dictionary,
        )
        private = PCA(n_components=self.cfg.n_private, random_state=self.random_state).fit(yz).components_.astype("float32")
        with torch.no_grad():
            self.model.private_programs.copy_(torch.tensor(private))
        weights = torch.tensor(adv.context_weights(meta, balanced=True), device=self.device)
        perturb_t = torch.tensor(perturb, device=self.device)
        basal_t = torch.tensor(basal, device=self.device)
        y_t = torch.tensor(yz, device=self.device)
        group_index, drug_groups = make_invariance_groups(meta)
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=7e-4, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred, mechanism_pred, core_act, private_act, router, expert_act, backbone_act, backbone_residual_act = self.model(perturb_t, basal_t)
            huber = F.smooth_l1_loss(pred, y_t, reduction="none").mean(dim=1)
            cosine = 1 - F.cosine_similarity(pred, y_t, dim=1)
            loss = adv.weighted_mean(huber, weights) + self.cfg.cosine_lambda * adv.weighted_mean(cosine, weights)
            if self.model.predictor_backbone is not None:
                mechanism_huber = F.smooth_l1_loss(mechanism_pred, y_t, reduction="none").mean(dim=1)
                mechanism_cosine = 1 - F.cosine_similarity(mechanism_pred, y_t, dim=1)
                loss = loss + self.cfg.mechanism_reconstruction_lambda * (
                    adv.weighted_mean(mechanism_huber, weights) + 0.25 * adv.weighted_mean(mechanism_cosine, weights)
                )
                loss = loss + self.cfg.distillation_lambda * F.smooth_l1_loss(mechanism_pred, pred.detach())
            effective_core = self.model.effective_core_programs()
            core_norm = F.normalize(effective_core, dim=1)
            private_norm = F.normalize(self.model.private_programs, dim=1)
            core_gram = core_norm @ core_norm.T
            private_gram = private_norm @ private_norm.T
            loss = loss + self.cfg.core_l0_lambda * self.model.expected_gate_probability().mean()
            loss = loss + self.cfg.core_weight_l1 * effective_core.abs().mean()
            loss = loss + self.cfg.core_activity_lambda * core_act.mean()
            loss = loss + self.cfg.private_l1 * self.model.private_programs.abs().mean()
            loss = loss + self.cfg.private_activity_lambda * private_act.abs().mean()
            loss = loss + self.cfg.diversity_lambda * (
                ((core_gram - torch.eye(len(core_norm), device=self.device)) ** 2).mean()
                + 0.5 * ((private_gram - torch.eye(len(private_norm), device=self.device)) ** 2).mean()
            )
            loss = loss + self.cfg.orthogonal_lambda * ((private_norm @ core_norm.T) ** 2).mean()
            if self.cfg.conditional_invariance_lambda > 0:
                loss = loss + self.cfg.conditional_invariance_lambda * conditional_study_invariance(core_act, group_index, drug_groups)
            if router is not None:
                router_mean = router.mean(dim=0)
                uniform = torch.full_like(router_mean, 1 / len(router_mean))
                loss = loss + self.cfg.router_balance_lambda * ((router_mean - uniform) ** 2).mean()
                loss = loss + self.cfg.expert_lambda * (
                    expert_act.abs().mean() + self.model.expert_programs.abs().mean()
                )
            if self.model.predictor_backbone is not None:
                loss = loss + 0.002 * backbone_act.abs().mean()
                if self.model.predictor_backbone.residual_programs is not None:
                    loss = loss + self.cfg.backbone_residual_lambda * (
                        backbone_residual_act.abs().mean() + self.model.predictor_backbone.residual_programs.abs().mean()
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
            pred, _, _, _, _, _, _, _ = self.model(torch.tensor(perturb, device=self.device), torch.tensor(basal, device=self.device))
        return self.y_scaler.inverse_transform(pred.cpu().numpy()).astype("float32")

    def core_activity_values(self, x_control: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        perturb, basal = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            _, _, activity, _, _, _, _, _ = self.model(torch.tensor(perturb, device=self.device), torch.tensor(basal, device=self.device))
        return activity.cpu().numpy()

    def core_matrix_original_scale(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            matrix = self.model.effective_core_programs().cpu().numpy()
        return matrix * self.y_scaler.scale_[None, :]

    def core_gate_probability(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model.expected_gate_probability().cpu().numpy()


def make_model(name: str, args: argparse.Namespace, device: torch.device, seed: int):
    if name == "matched":
        return tb.MatchedDrugMean()
    if name == "v2_pca":
        return tb.OurSharedProgramTorch(10, args.epochs, seed, device)
    if name == "v5_frozen_dictionary":
        return adv.AdvancedProgramTorch(adv.config_for(name, 10), args.epochs, seed, device, name=name)
    return SparseMoETorch(config_for(name), args.epochs, seed, device, name)


def splitters(meta: pd.DataFrame, random_state: int, n_splits: int):
    n_splits = max(3, n_splits)
    yield "random", list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(len(meta))))
    for name, column in [("cellline", "entity_norm"), ("study", "block_id"), ("drug", "drug_token")]:
        groups = meta[column].astype(str).to_numpy()
        folds = GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(np.arange(len(meta)), groups=groups)
        yield name, list(folds)


def summarize_cv(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "mae", "cosine", "pearson", "spearman", "top50_overlap", "top100_overlap", "top100_sign_agreement", "magnitude_fidelity"]
    summary = rows.groupby(["split", "variant", "model"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join([str(item) for item in column if item]) for column in summary.columns]
    summary = summary.reset_index()
    summary["selection_score"] = (
        summary["cosine_mean"] + 0.45 * summary["pearson_mean"] + 0.20 * summary["top100_overlap_mean"] - 0.025 * summary["rmse_mean"]
    )
    return summary.sort_values(["split", "selection_score"], ascending=[True, False])


def run_cv(args, device, meta, x_control, y_delta) -> pd.DataFrame:
    rows = []
    for seed in args.seeds:
        for split_name, folds in splitters(meta, seed, args.n_splits):
            print(f"[seed={seed}] split={split_name}", flush=True)
            for variant in args.variants:
                print(f"  variant={variant}", flush=True)
                for fold, (train_index, test_index) in enumerate(folds, start=1):
                    model = make_model(variant, args, device, seed + 1000 * fold)
                    train_meta = meta.iloc[train_index].reset_index(drop=True)
                    test_meta = meta.iloc[test_index].reset_index(drop=True)
                    model.fit(x_control[train_index], y_delta[train_index], train_meta)
                    row = tb.metrics(y_delta[test_index], model.predict(x_control[test_index], test_meta))
                    row.update({"seed": seed, "split": split_name, "variant": variant, "model": model.name, "fold": fold, "n_test": len(test_index)})
                    rows.append(row)
    return pd.DataFrame(rows)


def dictionary_quality(dictionary: np.ndarray, y_delta: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    projection = y_delta @ normalize_rows(dictionary).T
    rows = []
    for index in range(dictionary.shape[0]):
        frame = pd.DataFrame({"score": projection[:, index], "drug_token": meta["drug_token"], "block_id": meta["block_id"]})
        drug = frame.groupby("drug_token")["score"].mean()
        study = frame.groupby("block_id")["score"].mean()
        rows.append(
            {
                "candidate": index,
                "drug_positive_fraction": float(np.mean(drug > 0)),
                "study_positive_fraction": float(np.mean(study > 0)),
                "context_positive_fraction": float(np.mean(frame["score"] > 0)),
                "mean_projection": float(frame["score"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["breadth"] = 0.45 * out["drug_positive_fraction"] + 0.35 * out["study_positive_fraction"] + 0.20 * out["context_positive_fraction"]
    return out


def choose_core_programs(quality: pd.DataFrame, dictionary: np.ndarray, args) -> pd.DataFrame:
    quality = quality.copy()
    quality["passes_stability"] = quality["signed_cosine_min"].ge(args.min_seed_cosine) & quality["top50_jaccard_min"].ge(args.min_top50_jaccard)
    quality["passes_common_breadth"] = (
        quality["drug_positive_fraction"].ge(args.min_drug_positive_fraction)
        & quality["study_positive_fraction"].ge(args.min_study_positive_fraction)
    )
    quality["core_selection_score"] = (
        0.42 * quality["signed_cosine_mean"].clip(lower=0)
        + 0.22 * quality["top50_jaccard_mean"]
        + 0.26 * quality["breadth"]
        + 0.10 * np.tanh(np.abs(quality["mean_projection"]))
    )
    ordered = quality.sort_values(["passes_stability", "passes_common_breadth", "core_selection_score"], ascending=[False, False, False])
    selected = []
    eligible = ordered[ordered["passes_stability"] & ordered["passes_common_breadth"]]
    for candidate in eligible["candidate"].astype(int):
        if all(abs(float(normalize_rows(dictionary)[candidate] @ normalize_rows(dictionary)[prior])) < 0.74 for prior in selected):
            selected.append(candidate)
        if len(selected) >= args.max_core:
            break
    if len(selected) < args.min_core:
        selected = ordered.head(args.min_core)["candidate"].astype(int).tolist()
    quality["selected_core"] = quality["candidate"].isin(selected)
    return quality


def variable_gene_table(matrix: np.ndarray, genes: list[str], gate_frequency: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for index, (weights, frequency) in enumerate(zip(matrix, gate_frequency), start=1):
        keep = frequency >= threshold
        if keep.sum() < 24:
            keep[np.argsort(np.abs(weights))[-24:]] = True
        frame = pd.DataFrame({"gene": genes, "weight": weights, "selection_probability": frequency, "keep": keep})
        frame = frame[frame["keep"]].copy()
        for direction, subset in [("up", frame[frame["weight"] > 0].sort_values("weight", ascending=False)), ("down", frame[frame["weight"] < 0].sort_values("weight"))]:
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                rows.append(
                    {
                        "program": f"CoreProgram_{index}",
                        "direction": direction,
                        "rank": rank,
                        "gene": row.gene,
                        "weight": float(row.weight),
                        "selection_probability": float(row.selection_probability),
                    }
                )
    return pd.DataFrame(rows)


def train_final(args, device, meta, x_control, y_delta, genes, outdir: Path) -> None:
    cfg = config_for(args.selected_variant)
    models = []
    matrices = []
    gates = []
    activities = []
    predictions = []
    for seed in args.final_seeds:
        print(f"[final v8] seed={seed}", flush=True)
        model = SparseMoETorch(cfg, args.epochs * 2, seed, device, args.selected_variant)
        model.fit(x_control, y_delta, meta)
        models.append(model)
        matrices.append(model.core_matrix_original_scale())
        gates.append(model.core_gate_probability())
        activities.append(model.core_activity_values(x_control, meta))
        predictions.append(model.predict(x_control, meta))
    reference = orient_dictionary(matrices[0], y_delta)
    aligned_matrix = [reference]
    aligned_gates = [gates[0]]
    aligned_activity = [activities[0]]
    stability_rows = []
    for seed, matrix, gate, activity in zip(args.final_seeds[1:], matrices[1:], gates[1:], activities[1:]):
        aligned, order, signs = align_programs(reference, matrix)
        aligned_matrix.append(aligned)
        aligned_gates.append(gate[order])
        aligned_activity.append(activity[:, order] * signs[None, :])
        similarity = np.sum(normalize_rows(reference) * normalize_rows(aligned), axis=1)
        for index in range(cfg.n_core_candidates):
            ref_top = set(np.argsort(np.abs(reference[index]))[-50:])
            alt_top = set(np.argsort(np.abs(aligned[index]))[-50:])
            stability_rows.append(
                {
                    "seed": seed,
                    "candidate": index,
                    "signed_cosine": float(similarity[index]),
                    "top50_jaccard": float(len(ref_top & alt_top) / len(ref_top | alt_top)),
                }
            )
    consensus_matrix = np.mean(aligned_matrix, axis=0)
    consensus_gates = np.mean(aligned_gates, axis=0)
    consensus_activity = np.mean(aligned_activity, axis=0)
    stability = pd.DataFrame(stability_rows)
    if stability.empty:
        stability_summary = pd.DataFrame(
            {
                "candidate": np.arange(cfg.n_core_candidates),
                "signed_cosine_mean": 1.0,
                "signed_cosine_min": 1.0,
                "top50_jaccard_mean": 1.0,
                "top50_jaccard_min": 1.0,
            }
        )
    else:
        stability_summary = stability.groupby("candidate", as_index=False).agg(
            signed_cosine_mean=("signed_cosine", "mean"),
            signed_cosine_min=("signed_cosine", "min"),
            top50_jaccard_mean=("top50_jaccard", "mean"),
            top50_jaccard_min=("top50_jaccard", "min"),
        )
    quality = dictionary_quality(consensus_matrix, y_delta, meta).merge(stability_summary, on="candidate")
    quality = choose_core_programs(quality, consensus_matrix, args)
    selected = quality[quality["selected_core"]].sort_values("core_selection_score", ascending=False)["candidate"].astype(int).tolist()
    core_matrix = consensus_matrix[selected]
    core_gates = consensus_gates[selected]
    core_activity = consensus_activity[:, selected]
    program_cols = [f"CoreProgram_{index}" for index in range(1, len(selected) + 1)]
    gene_table = variable_gene_table(core_matrix, genes, core_gates, args.min_gene_selection_probability)
    gate_rows = []
    for program, weights, frequency in zip(program_cols, core_matrix, core_gates):
        for gene, weight, probability in zip(genes, weights, frequency):
            gate_rows.append({"program": program, "gene": gene, "weight": float(weight), "selection_probability": float(probability)})
    pd.DataFrame(stability_rows).to_csv(outdir / f"{args.prefix}_candidate_seed_stability.csv", index=False)
    quality.to_csv(outdir / f"{args.prefix}_core_candidate_quality.csv", index=False)
    gene_table.to_csv(outdir / f"{args.prefix}_core_program_genes.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(outdir / f"{args.prefix}_core_gene_membership.csv", index=False)
    np.save(outdir / f"{args.prefix}_core_program_matrix.npy", core_matrix)
    np.save(outdir / f"{args.prefix}_all_candidate_program_matrix.npy", consensus_matrix)
    np.save(outdir / f"{args.prefix}_all_candidate_gene_gate_probability.npy", consensus_gates)
    activity_df = pd.DataFrame(core_activity, columns=program_cols)
    activity_df.insert(0, "context_id", meta["context_id"].to_numpy())
    activity_df = activity_df.merge(meta, on="context_id", how="left")
    activity_df.to_csv(outdir / f"{args.prefix}_core_program_activity.csv", index=False)
    activity_df.groupby(["drug_token", "drug_name", "drug_class"], as_index=False)[program_cols].mean().to_csv(
        outdir / f"{args.prefix}_core_program_drug_activity.csv", index=False
    )
    pd.DataFrame([tb.metrics(y_delta, np.mean(predictions, axis=0))]).to_csv(outdir / f"{args.prefix}_fit_metrics.csv", index=False)
    joblib.dump(models, outdir / f"{args.prefix}_seed_models.joblib")
    (outdir / f"{args.prefix}_config.json").write_text(
        json.dumps(
            {
                "variant": args.selected_variant,
                "config": asdict(cfg),
                "final_seeds": args.final_seeds,
                "selected_candidate_indices": selected,
                "selected_core_programs": len(selected),
                "gene_membership_probability_threshold": args.min_gene_selection_probability,
                "min_drug_positive_fraction": args.min_drug_positive_fraction,
                "min_study_positive_fraction": args.min_study_positive_fraction,
                "n_training_contexts": len(meta),
                "n_genes": len(genes),
            },
            indent=2,
        )
        + "\n"
    )
    print(quality.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nFrozen core programs={len(selected)}; variable exported genes={len(gene_table)}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = tb.choose_device(args.device)
    print("Device:", device, flush=True)
    meta, x_control, _, y_delta, genes = tb.load_arrays(args)
    meta.to_csv(outdir / "v8_training_contrasts.csv", index=False)
    pd.Series(genes, name="gene").to_csv(outdir / "v8_genes.csv", index=False)
    if not args.skip_cv:
        fold = run_cv(args, device, meta, x_control, y_delta)
        fold.to_csv(outdir / "v8_cv_fold_metrics.csv", index=False)
        summarize_cv(fold).to_csv(outdir / "v8_cv_summary.csv", index=False)
    if not args.skip_final:
        train_final(args, device, meta, x_control, y_delta, genes, outdir)
    print(f"Done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
