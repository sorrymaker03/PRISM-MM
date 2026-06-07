#!/usr/bin/env python3
"""Train a structured multimodal refinement head for shared MM programs.

V14 is a calibration-stage latent dictionary model layered on the v9 bulk
predictor. It jointly reconstructs discovery bulk contrasts and paired scRNA
pseudobulk deltas while separating:
- sparse anchored shared programs;
- low-rank modality adapters;
- modality-private programs;
- routed bulk expert residuals;
- study-invariant shared activities.

The paired scRNA cohort is a calibration cohort. Pair-level out-of-fold scores
are exported for internal cross-modal assessment. Held-out bulk samples are
not read by this training script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import advanced_program_model as adv
import sparse_moe_program_model as v8
import torch_model_benchmark as tb
from build_v10_discovery_stable_core import load_pathway_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--v9-dir", default="bulk_pre_sc/model_upgrade_v9_final")
    parser.add_argument("--v9-prefix", default="our_anchored_sparse_attention_v9")
    parser.add_argument("--anchor-genes", default="bulk_pre_sc/model_upgrade_v13_cross_modal_adapter/our_cross_modal_adapter_v13_core_program_genes.csv")
    parser.add_argument("--scrna-gene-deltas", default="bulk_pre_sc/model_upgrade_v9_final/gene_direction_validation/scrna_all_adaptive_gene_delta_by_pair.csv")
    parser.add_argument("--outdir", default="bulk_pre_sc/model_upgrade_v14_multimodal_dictionary")
    parser.add_argument("--prefix", default="our_multimodal_dictionary_v14")
    parser.add_argument(
        "--variant",
        default="v14_balanced",
        choices=["v14_balanced", "v14_anchor_strong", "v14_alignment", "v14_dictionary_expand"],
    )
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--oof-epochs", type=int, default=260)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    return parser.parse_args()


@dataclass
class MultimodalConfig:
    n_core: int = 10
    n_private: int = 5
    n_experts: int = 3
    expert_rank: int = 3
    adapter_rank: int = 3
    innovation_scale: float = 0.14
    adapter_scale: float = 0.10
    private_scale: float = 0.22
    expert_scale: float = 0.08
    bulk_reconstruction_lambda: float = 1.0
    scrna_reconstruction_lambda: float = 0.72
    anchor_cosine_lambda: float = 0.50
    membership_anchor_lambda: float = 0.025
    pathway_membership_lambda: float = 0.010
    core_l0_lambda: float = 0.006
    innovation_l1: float = 0.008
    adapter_l1: float = 0.014
    adapter_alignment_lambda: float = 0.030
    private_l1: float = 0.010
    activity_l1: float = 0.002
    diversity_lambda: float = 0.050
    orthogonal_lambda: float = 0.070
    expert_l1: float = 0.014
    router_balance_lambda: float = 0.010
    conditional_invariance_lambda: float = 0.025
    distribution_alignment_lambda: float = 0.035
    nonanchor_gate_logit: float = -4.50
    export_size_multiplier: float = 1.0
    hard_concrete_beta: float = 0.67
    hard_concrete_gamma: float = -0.10
    hard_concrete_zeta: float = 1.10


def config_for(name: str) -> MultimodalConfig:
    base = MultimodalConfig()
    if name == "v14_balanced":
        return base
    if name == "v14_anchor_strong":
        return replace(
            base,
            innovation_scale=0.09,
            adapter_scale=0.07,
            anchor_cosine_lambda=0.85,
            membership_anchor_lambda=0.040,
            scrna_reconstruction_lambda=0.62,
        )
    if name == "v14_alignment":
        return replace(
            base,
            innovation_scale=0.16,
            adapter_scale=0.13,
            scrna_reconstruction_lambda=0.90,
            adapter_alignment_lambda=0.045,
            distribution_alignment_lambda=0.070,
        )
    if name == "v14_dictionary_expand":
        return replace(
            base,
            innovation_scale=0.34,
            adapter_scale=0.16,
            scrna_reconstruction_lambda=0.95,
            anchor_cosine_lambda=0.22,
            membership_anchor_lambda=0.008,
            pathway_membership_lambda=0.016,
            core_l0_lambda=0.004,
            adapter_alignment_lambda=0.045,
            distribution_alignment_lambda=0.060,
            nonanchor_gate_logit=-2.10,
            export_size_multiplier=1.50,
        )
    raise KeyError(name)


def program_key(value: str) -> int:
    match = re.search(r"_(\d+)$", str(value))
    return int(match.group(1)) if match else 10**6


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, denom, out=np.zeros_like(x), where=denom > 1e-8).astype("float32")


def initialize_gate_logits(anchor: np.ndarray, nonanchor_logit: float) -> np.ndarray:
    present = np.abs(anchor) > 0
    logits = np.full_like(anchor, nonanchor_logit, dtype="float32")
    absolute = np.abs(anchor)
    for index in range(len(anchor)):
        values = absolute[index, present[index]]
        if len(values) == 0:
            continue
        scaled = values / max(float(np.quantile(values, 0.75)), 1e-8)
        logits[index, present[index]] = np.clip(-0.85 + scaled, -0.75, 2.40)
    return logits


def load_anchor(path: Path, genes: list[str], n_core: int) -> tuple[np.ndarray, dict[str, int]]:
    table = pd.read_csv(path)
    matrix = np.zeros((n_core, len(genes)), dtype="float32")
    sizes: dict[str, int] = {}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for row in table.itertuples(index=False):
        program = str(row.program)
        index = program_key(program) - 1
        gene = str(row.gene)
        if 0 <= index < n_core and gene in gene_index:
            matrix[index, gene_index[gene]] = float(row.weight)
            sizes[program] = sizes.get(program, 0) + 1
    missing = np.flatnonzero(np.linalg.norm(matrix, axis=1) < 1e-8)
    if len(missing):
        raise ValueError(f"Missing anchor programs: {missing.tolist()}")
    return normalize_rows(matrix), sizes


def pathway_matrix(path: Path, genes: list[str], n_core: int) -> np.ndarray:
    prior = load_pathway_prior(path)
    matrix = np.zeros((n_core, len(genes)), dtype="float32")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for row in prior.itertuples(index=False):
        index = program_key(str(row.program)) - 1
        if 0 <= index < n_core and str(row.gene) in gene_index:
            matrix[index, gene_index[str(row.gene)]] = max(matrix[index, gene_index[str(row.gene)]], float(row.pathway_prior))
    return matrix


def scrna_matrix(path: Path, genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long = pd.read_csv(path)
    wide = long.pivot_table(index="matched_pair_index", columns="gene", values="delta", aggfunc="mean")
    pairs = wide.index.to_numpy(dtype=int)
    matrix = np.zeros((len(wide), len(genes)), dtype="float32")
    mask = np.zeros(len(genes), dtype="float32")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for gene in wide.columns:
        if str(gene) in gene_index:
            index = gene_index[str(gene)]
            matrix[:, index] = wide[gene].fillna(0.0).to_numpy(dtype="float32")
            mask[index] = 1.0
    return matrix, mask, pairs


def pair_fold_map(pairs: np.ndarray, folds: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    return {int(pair): index % folds for index, pair in enumerate(shuffled)}


def safe_wilcoxon(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def bh_fdr(values: pd.Series) -> pd.Series:
    raw = values.astype(float).to_numpy()
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


class MultimodalDictionaryNet(nn.Module):
    def __init__(self, n_bulk: int, n_scrna: int, n_genes: int, cfg: MultimodalConfig, anchor: np.ndarray, pathway: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("anchor", torch.tensor(anchor, dtype=torch.float32))
        self.register_buffer("anchor_membership", torch.tensor(np.abs(anchor) > 0, dtype=torch.float32))
        self.register_buffer("pathway_prior", torch.tensor(pathway, dtype=torch.float32))
        self.core_innovation = nn.Parameter(torch.zeros_like(self.anchor))
        self.core_log_alpha = nn.Parameter(torch.tensor(initialize_gate_logits(anchor, cfg.nonanchor_gate_logit)))
        self.bulk_activity = nn.Parameter(torch.randn(n_bulk, cfg.n_core) * 0.06)
        self.scrna_activity = nn.Parameter(torch.randn(n_scrna, cfg.n_core) * 0.06)
        self.bulk_private_activity = nn.Parameter(torch.randn(n_bulk, cfg.n_private) * 0.04)
        self.scrna_private_activity = nn.Parameter(torch.randn(n_scrna, cfg.n_private) * 0.04)
        self.bulk_private_programs = nn.Parameter(torch.randn(cfg.n_private, n_genes) * 0.012)
        self.scrna_private_programs = nn.Parameter(torch.randn(cfg.n_private, n_genes) * 0.012)
        self.adapter_left = nn.Parameter(torch.randn(2, cfg.n_core, cfg.adapter_rank) * 0.012)
        self.adapter_right = nn.Parameter(torch.randn(2, cfg.adapter_rank, n_genes) * 0.012)
        self.expert_router_logits = nn.Parameter(torch.zeros(n_bulk, cfg.n_experts))
        self.expert_activity = nn.Parameter(torch.randn(n_bulk, cfg.n_experts, cfg.expert_rank) * 0.025)
        self.expert_programs = nn.Parameter(torch.randn(cfg.n_experts, cfg.expert_rank, n_genes) * 0.010)

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

    def shared_programs(self) -> torch.Tensor:
        ungated = self.anchor + self.cfg.innovation_scale * torch.tanh(self.core_innovation)
        return ungated * self.hard_concrete_gate()

    def modality_programs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_programs()
        adapters = torch.einsum("mpr,mrg->mpg", self.adapter_left, self.adapter_right)
        return shared, shared + self.cfg.adapter_scale * adapters[0], shared + self.cfg.adapter_scale * adapters[1]

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        shared, bulk_programs, scrna_programs = self.modality_programs()
        bulk = self.bulk_activity @ bulk_programs
        scrna = self.scrna_activity @ scrna_programs
        bulk = bulk + self.cfg.private_scale * (self.bulk_private_activity @ self.bulk_private_programs)
        scrna = scrna + self.cfg.private_scale * (self.scrna_private_activity @ self.scrna_private_programs)
        router = torch.softmax(self.expert_router_logits, dim=1)
        expert = torch.einsum("be,ber,erg->bg", router, self.expert_activity, self.expert_programs)
        bulk = bulk + self.cfg.expert_scale * expert
        return bulk, scrna, {
            "shared": shared,
            "bulk_programs": bulk_programs,
            "scrna_programs": scrna_programs,
            "adapters": scrna_programs - bulk_programs,
            "router": router,
        }


def masked_scrna_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    residual = (prediction - target)[:, mask > 0]
    return F.smooth_l1_loss(residual, torch.zeros_like(residual)) + 0.20 * (
        1 - F.cosine_similarity(prediction[:, mask > 0], target[:, mask > 0], dim=1).mean()
    )


def distribution_alignment(bulk: torch.Tensor, scrna: torch.Tensor) -> torch.Tensor:
    bulk_z = (bulk - bulk.mean(dim=0)) / (bulk.std(dim=0) + 1e-6)
    scrna_z = (scrna - scrna.mean(dim=0)) / (scrna.std(dim=0) + 1e-6)
    return ((bulk_z.mean(dim=0) - scrna_z.mean(dim=0)) ** 2).mean() + (
        (bulk_z.std(dim=0) - scrna_z.std(dim=0)) ** 2
    ).mean()


def train_model(
    bulk_z: np.ndarray,
    scrna_z: np.ndarray,
    scrna_mask: np.ndarray,
    meta: pd.DataFrame,
    anchor: np.ndarray,
    pathway: np.ndarray,
    cfg: MultimodalConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[MultimodalDictionaryNet, pd.DataFrame]:
    tb.set_seed(seed)
    model = MultimodalDictionaryNet(len(bulk_z), len(scrna_z), bulk_z.shape[1], cfg, anchor, pathway).to(device)
    bulk_t = torch.tensor(bulk_z, dtype=torch.float32, device=device)
    scrna_t = torch.tensor(scrna_z, dtype=torch.float32, device=device)
    mask_t = torch.tensor(scrna_mask, dtype=torch.float32, device=device)
    weights = torch.tensor(adv.context_weights(meta, balanced=True), dtype=torch.float32, device=device)
    group_index, drug_groups = v8.make_invariance_groups(meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    rows = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        bulk_pred, scrna_pred, extra = model()
        bulk_huber = F.smooth_l1_loss(bulk_pred, bulk_t, reduction="none").mean(dim=1)
        bulk_cosine = 1 - F.cosine_similarity(bulk_pred, bulk_t, dim=1)
        loss_bulk = adv.weighted_mean(bulk_huber + 0.24 * bulk_cosine, weights)
        loss_scrna = masked_scrna_loss(scrna_pred, scrna_t, mask_t)
        shared_norm = F.normalize(extra["shared"], dim=1)
        anchor_norm = F.normalize(model.anchor, dim=1)
        bulk_private_norm = F.normalize(model.bulk_private_programs, dim=1)
        scrna_private_norm = F.normalize(model.scrna_private_programs, dim=1)
        gate = model.expected_gate_probability()
        membership_target = (0.94 * model.anchor_membership + 0.04 + 0.10 * model.pathway_prior).clamp(0.0, 0.99)
        loss = cfg.bulk_reconstruction_lambda * loss_bulk + cfg.scrna_reconstruction_lambda * loss_scrna
        loss = loss + cfg.anchor_cosine_lambda * (1 - F.cosine_similarity(shared_norm, anchor_norm, dim=1).mean())
        loss = loss + cfg.membership_anchor_lambda * F.binary_cross_entropy(gate, membership_target)
        loss = loss - cfg.pathway_membership_lambda * (gate * model.pathway_prior).mean()
        loss = loss + cfg.core_l0_lambda * gate.mean() + cfg.innovation_l1 * model.core_innovation.abs().mean()
        loss = loss + cfg.adapter_l1 * (model.adapter_left.abs().mean() + model.adapter_right.abs().mean())
        loss = loss + cfg.adapter_alignment_lambda * (extra["adapters"] ** 2).mean()
        loss = loss + cfg.private_l1 * (model.bulk_private_programs.abs().mean() + model.scrna_private_programs.abs().mean())
        loss = loss + cfg.activity_l1 * (
            model.bulk_activity.abs().mean()
            + model.scrna_activity.abs().mean()
            + model.bulk_private_activity.abs().mean()
            + model.scrna_private_activity.abs().mean()
        )
        identity = torch.eye(cfg.n_core, device=device)
        loss = loss + cfg.diversity_lambda * ((shared_norm @ shared_norm.T - identity) ** 2).mean()
        loss = loss + cfg.orthogonal_lambda * (
            ((bulk_private_norm @ shared_norm.T) ** 2).mean() + ((scrna_private_norm @ shared_norm.T) ** 2).mean()
        )
        loss = loss + cfg.expert_l1 * (model.expert_activity.abs().mean() + model.expert_programs.abs().mean())
        router_mean = extra["router"].mean(dim=0)
        loss = loss + cfg.router_balance_lambda * ((router_mean - 1 / cfg.n_experts) ** 2).mean()
        loss = loss + cfg.conditional_invariance_lambda * v8.conditional_study_invariance(model.bulk_activity, group_index, drug_groups)
        loss = loss + cfg.distribution_alignment_lambda * distribution_alignment(model.bulk_activity, model.scrna_activity)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        rows.append({"epoch": epoch + 1, "loss": value, "bulk_loss": float(loss_bulk.detach().cpu()), "scrna_loss": float(loss_scrna.detach().cpu())})
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(rows)


def original_shared_matrix(model: MultimodalDictionaryNet, bulk_scaler: StandardScaler) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        matrix = model.shared_programs().cpu().numpy()
    return matrix * bulk_scaler.scale_[None, :]


def gene_table(
    matrix: np.ndarray,
    gates: np.ndarray,
    genes: list[str],
    anchor_sizes: dict[str, int],
    export_size_multiplier: float,
) -> pd.DataFrame:
    rows = []
    for index, (weights, probability) in enumerate(zip(matrix, gates), start=1):
        program = f"Program_{index}"
        target_total = max(10, int(round(anchor_sizes.get(program, 20) * export_size_multiplier)))
        target_each = max(5, target_total // 2)
        for direction, mask, order in [
            ("up", weights > 0, np.argsort(weights)[::-1]),
            ("down", weights < 0, np.argsort(weights)),
        ]:
            candidates = [item for item in order if mask[item]]
            keep = [item for item in candidates if probability[item] >= 0.50]
            if len(keep) < target_each:
                keep = candidates[:target_each]
            keep = keep[: max(target_each, min(40, target_total))]
            for rank, item in enumerate(keep, start=1):
                rows.append(
                    {
                        "program": program,
                        "direction": direction,
                        "rank": rank,
                        "gene": genes[item],
                        "weight": float(weights[item]),
                        "selection_probability": float(probability[item]),
                    }
                )
    return pd.DataFrame(rows)


def signature_scores(table: pd.DataFrame, matrix: np.ndarray, genes: list[str]) -> pd.DataFrame:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    rows = {}
    for program, group in table.groupby("program"):
        use = group[group["gene"].isin(gene_index)].copy()
        indices = [gene_index[gene] for gene in use["gene"]]
        weights = use["weight"].to_numpy(dtype=float)
        rows[program] = matrix[:, indices] @ weights / max(float(np.abs(weights).sum()), 1e-8)
    return pd.DataFrame(rows)


def export_oof(
    args: argparse.Namespace,
    cfg: MultimodalConfig,
    device: torch.device,
    meta: pd.DataFrame,
    bulk_z: np.ndarray,
    scrna_raw: np.ndarray,
    scrna_mask: np.ndarray,
    scrna_pairs: np.ndarray,
    genes: list[str],
    anchor: np.ndarray,
    anchor_sizes: dict[str, int],
    pathway: np.ndarray,
    bulk_scaler: StandardScaler,
    outdir: Path,
) -> None:
    pair_to_fold = pair_fold_map(scrna_pairs, args.folds, args.seed)
    rows = []
    for fold in range(args.folds):
        train_indices = [index for index, pair in enumerate(scrna_pairs) if pair_to_fold[int(pair)] != fold]
        test_indices = [index for index, pair in enumerate(scrna_pairs) if pair_to_fold[int(pair)] == fold]
        scrna_scaler = StandardScaler().fit(scrna_raw[train_indices])
        train_z = scrna_scaler.transform(scrna_raw[train_indices]).astype("float32")
        model, _ = train_model(
            bulk_z,
            train_z,
            scrna_mask,
            meta,
            anchor,
            pathway,
            cfg,
            args.oof_epochs,
            args.seed + 1000 + fold,
            device,
        )
        matrix = original_shared_matrix(model, bulk_scaler)
        gates = model.expected_gate_probability().detach().cpu().numpy()
        table = gene_table(matrix, gates, genes, anchor_sizes, cfg.export_size_multiplier)
        score = signature_scores(table, scrna_raw[test_indices], genes)
        for local_index, pair_index in enumerate(test_indices):
            for program in score.columns:
                rows.append(
                    {
                        "matched_pair_index": int(scrna_pairs[pair_index]),
                        "fold": fold,
                        "program": program,
                        "delta_score": float(score.iloc[local_index][program]),
                    }
                )
    long = pd.DataFrame(rows)
    long.to_csv(outdir / f"{args.prefix}_scrna_pair_oof_shift_long.csv", index=False)
    summary_rows = []
    for program, group in long.groupby("program"):
        values = group["delta_score"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "program": program,
                "n_pairs": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "frac_increase": float((values > 0).mean()),
                "p_value": safe_wilcoxon(values),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("program", key=lambda values: values.map(program_key))
    summary["q_value"] = bh_fdr(summary["p_value"])
    summary.to_csv(outdir / f"{args.prefix}_scrna_pair_oof_shift_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = config_for(args.variant)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = tb.choose_device(args.device)
    print("Device:", device, flush=True)
    meta, _, _, bulk_delta, genes = tb.load_arrays(args)
    anchor, anchor_sizes = load_anchor(Path(args.anchor_genes), genes, cfg.n_core)
    pathway = pathway_matrix(Path(args.v9_dir) / f"{args.v9_prefix}_core_program_enrichr_pathways.csv", genes, cfg.n_core)
    scrna_raw, scrna_mask, scrna_pairs = scrna_matrix(Path(args.scrna_gene_deltas), genes)
    bulk_scaler = StandardScaler().fit(bulk_delta)
    scrna_scaler = StandardScaler().fit(scrna_raw)
    bulk_z = bulk_scaler.transform(bulk_delta).astype("float32")
    scrna_z = scrna_scaler.transform(scrna_raw).astype("float32")
    model, history = train_model(bulk_z, scrna_z, scrna_mask, meta, anchor, pathway, cfg, args.epochs, args.seed, device)
    matrix = original_shared_matrix(model, bulk_scaler)
    gates = model.expected_gate_probability().detach().cpu().numpy()
    table = gene_table(matrix, gates, genes, anchor_sizes, cfg.export_size_multiplier)
    with torch.no_grad():
        bulk_fit, scrna_fit, _ = model()
    fit_metrics = tb.metrics(bulk_z, bulk_fit.cpu().numpy())
    fit_metrics["scrna_masked_rmse"] = float(np.sqrt(np.mean((scrna_fit.cpu().numpy()[:, scrna_mask > 0] - scrna_z[:, scrna_mask > 0]) ** 2)))
    meta.to_csv(outdir / "v14_training_contrasts.csv", index=False)
    pd.Series(genes, name="gene").to_csv(outdir / "v14_genes.csv", index=False)
    table.to_csv(outdir / f"{args.prefix}_core_program_genes.csv", index=False)
    history.to_csv(outdir / f"{args.prefix}_training_history.csv", index=False)
    pd.DataFrame([fit_metrics]).to_csv(outdir / f"{args.prefix}_fit_metrics.csv", index=False)
    np.save(outdir / f"{args.prefix}_core_program_matrix.npy", matrix)
    np.save(outdir / f"{args.prefix}_gene_gate_probability.npy", gates)
    export_oof(args, cfg, device, meta, bulk_z, scrna_raw, scrna_mask, scrna_pairs, genes, anchor, anchor_sizes, pathway, bulk_scaler, outdir)
    (outdir / f"{args.prefix}_config.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "config": asdict(cfg),
                "training": "v9 backbone followed by multimodal latent-dictionary refinement head",
                "scrna_role": "calibration cohort with pair-level out-of-fold assessment",
                "heldout_bulk_used_for_training": False,
                "anchor_genes": args.anchor_genes,
                "n_bulk_contexts": len(meta),
                "n_scrna_pairs": len(scrna_pairs),
                "n_genes": len(genes),
            },
            indent=2,
        )
        + "\n"
    )
    print("\nExported genes")
    print(table.groupby(["program", "direction"]).size().unstack(fill_value=0).assign(total=lambda frame: frame.sum(axis=1)).to_string())
    print(f"\nDone: {outdir}", flush=True)


if __name__ == "__main__":
    main()
