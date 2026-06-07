#!/usr/bin/env python3
"""PyTorch benchmark for MM bulk perturbation models.

This focuses on model effect, not manuscript figures. It compares:
- matched DEG/drug-mean baseline
- scGen-style autoencoder latent shift
- CPA-style supervised additive neural model
- CellOT-style neural control-to-treated map
- OUR upgraded shared-program neural model

The official scGen/CPA/CellOT packages are not called here; these are PyTorch
architectures adapted to the current bulk matched-control task.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import bulk_pre_sc_pipeline as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--outdir", default="bulk_pre_sc/torch_benchmark")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--programs", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["matched", "scgen", "cpa", "cellot", "ours_shared", "ours_hybrid"],
        choices=["matched", "scgen", "cpa", "cellot", "ours_shared", "ours_hybrid"],
    )
    parser.add_argument("--skip-final-export", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_arrays(args: argparse.Namespace):
    article_dir = Path(args.article_dir)
    contrasts = pd.read_csv(article_dir / "ordered_contrasts_core.csv")
    delta = pd.read_csv(article_dir / "delta_core_selected_genes.csv.gz", index_col=0)
    contrasts = contrasts[contrasts["context_id"].isin(delta.index)].copy()
    contrasts = contrasts.set_index("context_id").loc[delta.index].reset_index()
    if "context_id" not in contrasts.columns and "index" in contrasts.columns:
        contrasts = contrasts.rename(columns={"index": "context_id"})

    if args.n_genes < delta.shape[1]:
        genes = delta.var(axis=0).sort_values(ascending=False).head(args.n_genes).index.tolist()
        delta = delta[genes]
    else:
        genes = delta.columns.tolist()

    print(f"Loading log2 expression for {len(genes)} genes")
    expr = base.read_expression(Path(args.expr), human_symbol_like=True, exclude_technical_gene_families=True)
    expr = np.log2(expr.loc[genes] + 1.0).astype("float32")

    x_control = []
    y_treated = []
    for _, row in contrasts.iterrows():
        controls = [s for s in str(row["control_samples"]).split(";") if s in expr.columns]
        treated = [s for s in str(row["treated_samples"]).split(";") if s in expr.columns]
        x_control.append(expr[controls].mean(axis=1).to_numpy(dtype=np.float32))
        y_treated.append(expr[treated].mean(axis=1).to_numpy(dtype=np.float32))
    x_control = np.vstack(x_control).astype("float32")
    y_treated = np.vstack(y_treated).astype("float32")
    y_delta = (y_treated - x_control).astype("float32")

    for col in ["drug_token", "drug_class", "dose_label", "phenotype_label", "confidence", "entity_norm", "block_id"]:
        contrasts[col] = contrasts[col].fillna("NA").astype(str)
    contrasts["time_hours"] = contrasts["time_hours"].fillna(0.0).astype(float)
    contrasts["has_time"] = (contrasts["time_label"].fillna("NA").astype(str) != "NA").astype(int)
    return contrasts.reset_index(drop=True), x_control, y_treated, y_delta, genes


class FeatureFeaturizer:
    def __init__(self, n_control_pcs: int = 24, random_state: int = 0):
        self.n_control_pcs = n_control_pcs
        self.random_state = random_state

    def fit(self, meta: pd.DataFrame, x_control: np.ndarray):
        cat_cols = ["drug_token", "drug_class", "dose_label", "phenotype_label", "confidence"]
        num_cols = ["time_hours", "has_time"]
        self.transformer = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        ).fit(meta)
        k = min(self.n_control_pcs, x_control.shape[0] - 1, x_control.shape[1])
        self.ctrl_scaler = StandardScaler().fit(x_control)
        self.ctrl_pca = PCA(n_components=k, random_state=self.random_state).fit(self.ctrl_scaler.transform(x_control))
        return self

    def transform(self, meta: pd.DataFrame, x_control: np.ndarray) -> np.ndarray:
        z = self.transformer.transform(meta)
        if sparse.issparse(z):
            z = z.toarray()
        ctrl = self.ctrl_pca.transform(self.ctrl_scaler.transform(x_control))
        return np.hstack([z, ctrl]).astype("float32")


def rank_rows(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, axis=1)
    ranks = np.empty(order.shape, dtype="float32")
    ranks[np.arange(x.shape[0])[:, None], order] = np.arange(x.shape[1], dtype="float32")
    return ranks


def row_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(yt, axis=1) * np.linalg.norm(yp, axis=1)
    return np.divide(np.sum(yt * yp, axis=1), denom, out=np.zeros_like(denom), where=denom > 1e-12)


def topk_overlap_and_sign(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[float, float]:
    k = min(k, y_true.shape[1])
    overlaps = []
    sign_hits = []
    for t, p in zip(y_true, y_pred):
        ti = set(np.argpartition(np.abs(t), -k)[-k:])
        pi = set(np.argpartition(np.abs(p), -k)[-k:])
        overlaps.append(len(ti & pi) / k)
        idx = np.array(sorted(ti))
        true_sign = np.sign(t[idx])
        pred_sign = np.sign(p[idx])
        mask = true_sign != 0
        sign_hits.append(float(np.mean(true_sign[mask] == pred_sign[mask])) if mask.any() else 0.5)
    return float(np.mean(overlaps)), float(np.mean(sign_hits))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse_row = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mae_row = np.mean(np.abs(y_true - y_pred), axis=1)
    denom = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    cosine = np.divide(np.sum(y_true * y_pred, axis=1), denom, out=np.zeros_like(denom), where=denom > 1e-12)
    pearson = row_correlation(y_true, y_pred)
    spearman = row_correlation(rank_rows(y_true), rank_rows(y_pred))
    top50_overlap, _ = topk_overlap_and_sign(y_true, y_pred, 50)
    top100_overlap, top100_sign_agreement = topk_overlap_and_sign(y_true, y_pred, 100)
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    magnitude_fidelity = np.exp(-np.abs(np.log((pred_norm + 1e-8) / (true_norm + 1e-8))))
    return {
        "rmse": float(np.mean(rmse_row)),
        "mae": float(np.mean(mae_row)),
        "cosine": float(np.mean(cosine)),
        "pearson": float(np.mean(pearson)),
        "spearman": float(np.mean(spearman)),
        "top50_overlap": top50_overlap,
        "top100_overlap": top100_overlap,
        "top100_sign_agreement": top100_sign_agreement,
        "magnitude_fidelity": float(np.mean(magnitude_fidelity)),
    }


def train_loop(model: nn.Module, loss_fn, tensors: tuple[torch.Tensor, ...], device: torch.device, epochs: int, lr: float = 1e-3):
    model.to(device)
    tensors = tuple(t.to(device) for t in tensors)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model, tensors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return model


class AutoEncoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int):
        super().__init__()
        hidden = min(512, max(128, latent_dim * 8))
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MLPDelta(nn.Module):
    def __init__(self, n_features: int, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, n_genes),
        )

    def forward(self, x):
        return self.net(x)


class TransportNet(nn.Module):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class SharedProgramNet(nn.Module):
    def __init__(self, n_features: int, n_genes: int, n_programs: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, n_programs),
        )
        self.programs = nn.Parameter(torch.randn(n_programs, n_genes) * 0.025)
        self.scale_head = nn.Sequential(nn.Linear(n_features, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        act = self.encoder(x)
        raw = act @ self.programs
        scale = F.softplus(self.scale_head(x)) + 0.1
        return raw * scale, act


class MatchedDrugMean:
    name = "matched_DEG_drug_mean"

    def fit(self, x_control, y_delta, meta, **kwargs):
        self.global_mean = y_delta.mean(axis=0)
        self.drug_mean = {k: y_delta[list(v)].mean(axis=0) for k, v in meta.groupby("drug_token").groups.items()}
        self.class_mean = {k: y_delta[list(v)].mean(axis=0) for k, v in meta.groupby("drug_class").groups.items()}
        return self

    def predict(self, x_control, meta):
        rows = []
        for _, row in meta.iterrows():
            if row["drug_token"] in self.drug_mean:
                rows.append(self.drug_mean[row["drug_token"]])
            elif row["drug_class"] in self.class_mean:
                rows.append(self.class_mean[row["drug_class"]])
            else:
                rows.append(self.global_mean)
        return np.vstack(rows).astype("float32")


class ScGenTorch:
    name = "scGen_torch_latent_shift"

    def __init__(self, latent_dim: int, epochs: int, random_state: int, device: torch.device):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def fit(self, x_control, y_delta, meta, **kwargs):
        set_seed(self.random_state)
        y_treated = x_control + y_delta
        combined = np.vstack([x_control, y_treated])
        self.scaler = StandardScaler().fit(combined)
        combined_z = self.scaler.transform(combined).astype("float32")
        self.model = AutoEncoder(combined_z.shape[1], self.latent_dim)

        x_t = torch.tensor(combined_z)

        def loss_fn(model, tensors):
            (x,) = tensors
            recon = model(x)
            return F.mse_loss(recon, x)

        train_loop(self.model, loss_fn, (x_t,), self.device, self.epochs, lr=1e-3)
        self.model.to(self.device).eval()
        with torch.no_grad():
            zc = self.model.encoder(torch.tensor(self.scaler.transform(x_control).astype("float32"), device=self.device)).cpu().numpy()
            zt = self.model.encoder(torch.tensor(self.scaler.transform(y_treated).astype("float32"), device=self.device)).cpu().numpy()
        shifts = zt - zc
        self.global_shift = shifts.mean(axis=0)
        self.drug_shift = {k: shifts[list(v)].mean(axis=0) for k, v in meta.groupby("drug_token").groups.items()}
        self.class_shift = {k: shifts[list(v)].mean(axis=0) for k, v in meta.groupby("drug_class").groups.items()}
        return self

    def predict(self, x_control, meta):
        self.model.to(self.device).eval()
        with torch.no_grad():
            zc = self.model.encoder(torch.tensor(self.scaler.transform(x_control).astype("float32"), device=self.device)).cpu().numpy()
        shifts = []
        for _, row in meta.iterrows():
            shifts.append(self.drug_shift.get(row["drug_token"], self.class_shift.get(row["drug_class"], self.global_shift)))
        zt = zc + np.vstack(shifts)
        with torch.no_grad():
            pred_z = self.model.decoder(torch.tensor(zt.astype("float32"), device=self.device)).cpu().numpy()
        y_pred = self.scaler.inverse_transform(pred_z)
        return (y_pred - x_control).astype("float32")


class CPATorch:
    name = "CPA_torch_additive"

    def __init__(self, epochs: int, random_state: int, device: torch.device):
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def fit(self, x_control, y_delta, meta, **kwargs):
        set_seed(self.random_state)
        self.feat = FeatureFeaturizer(random_state=self.random_state).fit(meta, x_control)
        x_feat = self.feat.transform(meta, x_control)
        self.y_scaler = StandardScaler().fit(y_delta)
        yz = self.y_scaler.transform(y_delta).astype("float32")
        self.model = MLPDelta(x_feat.shape[1], yz.shape[1])

        def loss_fn(model, tensors):
            x, y = tensors
            pred = model(x)
            cos = 1 - F.cosine_similarity(pred, y, dim=1).mean()
            return F.mse_loss(pred, y) + 0.12 * cos

        train_loop(self.model, loss_fn, (torch.tensor(x_feat), torch.tensor(yz)), self.device, self.epochs, lr=1e-3)
        return self

    def predict(self, x_control, meta):
        x_feat = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(x_feat, device=self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(pred).astype("float32")


class CellOTTorch:
    name = "CellOT_torch_transport"

    def __init__(self, latent_dim: int, epochs: int, random_state: int, device: torch.device):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def fit(self, x_control, y_delta, meta, **kwargs):
        set_seed(self.random_state)
        y_treated = x_control + y_delta
        combined = np.vstack([x_control, y_treated])
        self.expr_scaler = StandardScaler().fit(combined)
        k = min(self.latent_dim, combined.shape[0] - 1, combined.shape[1])
        self.pca = PCA(n_components=k, random_state=self.random_state).fit(self.expr_scaler.transform(combined))
        zc = self.pca.transform(self.expr_scaler.transform(x_control)).astype("float32")
        zt = self.pca.transform(self.expr_scaler.transform(y_treated)).astype("float32")
        self.feat = FeatureFeaturizer(random_state=self.random_state).fit(meta, x_control)
        x_feat = np.hstack([zc, self.feat.transform(meta, x_control)]).astype("float32")
        self.model = TransportNet(x_feat.shape[1], zt.shape[1])

        def loss_fn(model, tensors):
            x, y = tensors
            pred = model(x)
            return F.mse_loss(pred, y) + 0.1 * (1 - F.cosine_similarity(pred, y, dim=1).mean())

        train_loop(self.model, loss_fn, (torch.tensor(x_feat), torch.tensor(zt)), self.device, self.epochs, lr=1e-3)
        return self

    def predict(self, x_control, meta):
        zc = self.pca.transform(self.expr_scaler.transform(x_control)).astype("float32")
        x_feat = np.hstack([zc, self.feat.transform(meta, x_control)]).astype("float32")
        self.model.to(self.device).eval()
        with torch.no_grad():
            zt = self.model(torch.tensor(x_feat, device=self.device)).cpu().numpy()
        y_pred = self.expr_scaler.inverse_transform(self.pca.inverse_transform(zt))
        return (y_pred - x_control).astype("float32")


class OurSharedProgramTorch:
    name = "OUR_shared_program_v2"

    def __init__(self, n_programs: int, epochs: int, random_state: int, device: torch.device):
        self.n_programs = n_programs
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    def fit(self, x_control, y_delta, meta, **kwargs):
        set_seed(self.random_state)
        self.feat = FeatureFeaturizer(random_state=self.random_state).fit(meta, x_control)
        x_feat = self.feat.transform(meta, x_control)
        self.y_scaler = StandardScaler().fit(y_delta)
        yz = self.y_scaler.transform(y_delta).astype("float32")
        self.model = SharedProgramNet(x_feat.shape[1], yz.shape[1], self.n_programs)

        # Initialize shared programs from PCA components of deltas for stability.
        k = min(self.n_programs, yz.shape[0] - 1, yz.shape[1])
        init = PCA(n_components=k, random_state=self.random_state).fit(yz).components_.astype("float32")
        if k < self.n_programs:
            init = np.vstack([init, np.random.normal(0, 0.01, size=(self.n_programs - k, yz.shape[1])).astype("float32")])
        with torch.no_grad():
            self.model.programs.copy_(torch.tensor(init))

        def loss_fn(model, tensors):
            x, y = tensors
            pred, act = model(x)
            mse = F.mse_loss(pred, y)
            cos = 1 - F.cosine_similarity(pred, y, dim=1).mean()
            l1_program = model.programs.abs().mean()
            act_penalty = act.abs().mean() * 0.002
            # Encourage programs to be distinct without forcing one-hot use.
            p = F.normalize(model.programs, dim=1)
            gram = p @ p.T
            diversity = ((gram - torch.eye(gram.shape[0], device=gram.device)) ** 2).mean()
            return mse + 0.35 * cos + 0.006 * l1_program + 0.04 * diversity + act_penalty

        train_loop(self.model, loss_fn, (torch.tensor(x_feat), torch.tensor(yz)), self.device, self.epochs, lr=8e-4)
        return self

    def predict(self, x_control, meta):
        x_feat = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            pred, _ = self.model(torch.tensor(x_feat, device=self.device))
            pred = pred.cpu().numpy()
        return self.y_scaler.inverse_transform(pred).astype("float32")

    def program_activity(self, x_control, meta):
        x_feat = self.feat.transform(meta, x_control)
        self.model.to(self.device).eval()
        with torch.no_grad():
            _, act = self.model(torch.tensor(x_feat, device=self.device))
        return act.cpu().numpy()

    def program_matrix_original_scale(self):
        prog = self.model.programs.detach().cpu().numpy()
        return prog * self.y_scaler.scale_[None, :]


class OurHybridTeacherProgramTorch:
    name = "OUR_hybrid_teacher_program_v3"

    def __init__(self, latent_dim: int, n_programs: int, epochs: int, random_state: int, device: torch.device):
        self.latent_dim = latent_dim
        self.n_programs = n_programs
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

    @staticmethod
    def _calibrate_scale(y_true: np.ndarray, pred: np.ndarray) -> float:
        denom = float(np.sum(pred * pred))
        if denom <= 1e-12:
            return 1.0
        scale = float(np.sum(y_true * pred) / denom)
        return float(np.clip(scale, 0.05, 1.5))

    def fit(self, x_control, y_delta, meta, **kwargs):
        # Directional teachers.
        self.scgen = ScGenTorch(self.latent_dim, self.epochs, self.random_state, self.device).fit(x_control, y_delta, meta)
        self.cellot = CellOTTorch(self.latent_dim, self.epochs, self.random_state + 1, self.device).fit(x_control, y_delta, meta)
        # Interpretable student/program branch.
        self.program = OurSharedProgramTorch(self.n_programs, self.epochs, self.random_state + 2, self.device).fit(x_control, y_delta, meta)
        self.mean = MatchedDrugMean().fit(x_control, y_delta, meta)

        preds = {
            "scgen": self.scgen.predict(x_control, meta),
            "cellot": self.cellot.predict(x_control, meta),
            "program": self.program.predict(x_control, meta),
            "mean": self.mean.predict(x_control, meta),
        }

        candidates = []
        grid = np.linspace(0, 1, 6)
        keys = list(preds)
        # Coarse simplex search over four branches.
        for ws in grid:
            for wc in grid:
                for wp in grid:
                    wm = 1.0 - ws - wc - wp
                    if wm < -1e-9:
                        continue
                    raw = ws * preds["scgen"] + wc * preds["cellot"] + wp * preds["program"] + wm * preds["mean"]
                    scale = self._calibrate_scale(y_delta, raw)
                    pred = scale * raw
                    m = metrics(y_delta, pred)
                    objective = (
                        m["cosine"]
                        + 0.45 * m["pearson"]
                        + 0.25 * m["top100_overlap"]
                        - 0.025 * m["rmse"]
                        + 0.02 * wp
                    )
                    candidates.append((objective, ws, wc, wp, wm, scale))
        _, self.ws, self.wc, self.wp, self.wm, self.scale = max(candidates, key=lambda x: x[0])
        return self

    def predict(self, x_control, meta):
        raw = (
            self.ws * self.scgen.predict(x_control, meta)
            + self.wc * self.cellot.predict(x_control, meta)
            + self.wp * self.program.predict(x_control, meta)
            + self.wm * self.mean.predict(x_control, meta)
        )
        return (self.scale * raw).astype("float32")

    def program_activity(self, x_control, meta):
        return self.program.program_activity(x_control, meta)

    def program_matrix_original_scale(self):
        return self.program.program_matrix_original_scale()


def splitters(meta: pd.DataFrame, random_state: int):
    n = len(meta)
    yield "random_5fold", list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(np.arange(n)))
    for name, col in [("leave_entity_out", "entity_norm"), ("leave_block_out", "block_id")]:
        groups = meta[col].astype(str).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            yield name, list(GroupKFold(n_splits=min(5, n_groups)).split(np.arange(n), groups=groups))


def model_factories(args: argparse.Namespace, device: torch.device):
    factories = {
        "matched": lambda: MatchedDrugMean(),
        "scgen": lambda: ScGenTorch(args.latent_dim, args.epochs, args.random_state, device),
        "cpa": lambda: CPATorch(args.epochs, args.random_state, device),
        "cellot": lambda: CellOTTorch(args.latent_dim, args.epochs, args.random_state, device),
        "ours_shared": lambda: OurSharedProgramTorch(args.programs, args.epochs, args.random_state, device),
        "ours_hybrid": lambda: OurHybridTeacherProgramTorch(args.latent_dim, args.programs, args.epochs, args.random_state, device),
    }
    return [factories[name] for name in args.models]


def run_cv(args, device, meta, x_control, y_delta):
    rows = []
    for split_name, splits in splitters(meta, args.random_state):
        print(f"Split: {split_name}")
        for make_model in model_factories(args, device):
            model_name = make_model().name
            print(f"  {model_name}")
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                model = make_model()
                model.fit(x_control[train_idx], y_delta[train_idx], meta.iloc[train_idx].reset_index(drop=True))
                pred = model.predict(x_control[test_idx], meta.iloc[test_idx].reset_index(drop=True))
                row = metrics(y_delta[test_idx], pred)
                row.update({"split": split_name, "model": model.name, "fold": fold, "n_test": len(test_idx)})
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    out = fold_metrics.groupby(["split", "model"], as_index=False)[metric_cols].agg(["mean", "std"])
    out.columns = ["_".join([x for x in col if x]) for col in out.columns]
    return out.reset_index().sort_values(["split", "cosine_mean"], ascending=[True, False])


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 5, figsize=(25, 6), sharey=True)
    for ax, (metric, title) in zip(
        axs,
        [
            ("cosine_mean", "Cosine"),
            ("pearson_mean", "Pearson"),
            ("spearman_mean", "Spearman"),
            ("top100_overlap_mean", "Top-100 overlap"),
            ("rmse_mean", "RMSE"),
        ],
    ):
        sns.barplot(data=summary, y="model", x=metric, hue="split", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.legend(fontsize=7, title="")
    fig.suptitle("PyTorch perturbation model benchmark", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "torch_benchmark_summary.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def export_program_model(model, args, device, meta, x_control, y_delta, genes, outdir: Path, prefix: str):
    pred = model.predict(x_control, meta)
    pd.DataFrame([metrics(y_delta, pred)]).to_csv(outdir / f"{prefix}_fit_metrics.csv", index=False)

    activity = model.program_activity(x_control, meta)
    program_cols = [f"Program_{i+1}" for i in range(activity.shape[1])]
    act_df = pd.DataFrame(activity, columns=program_cols)
    act_df.insert(0, "context_id", meta["context_id"].to_numpy())
    act_df = act_df.merge(meta, on="context_id", how="left")
    act_df.to_csv(outdir / f"{prefix}_program_activity.csv", index=False)

    drug_act = act_df.groupby(["drug_token", "drug_name", "drug_class"], as_index=False)[program_cols].mean()
    drug_act.to_csv(outdir / f"{prefix}_program_drug_activity.csv", index=False)

    prog = model.program_matrix_original_scale()
    rows = []
    for i, program in enumerate(program_cols):
        weights = pd.Series(prog[i], index=genes)
        for direction, vals in [("up", weights.sort_values(ascending=False)), ("down", weights.sort_values(ascending=True))]:
            top = vals.head(80)
            for rank, (gene, weight) in enumerate(top.items(), start=1):
                rows.append({"program": program, "direction": direction, "rank": rank, "gene": gene, "weight": float(weight)})
    pd.DataFrame(rows).to_csv(outdir / f"{prefix}_program_genes.csv", index=False)


def train_final_our_model(args, device, meta, x_control, y_delta, genes, outdir: Path):
    model = OurSharedProgramTorch(args.programs, args.epochs * 2, args.random_state + 99, device)
    model.fit(x_control, y_delta, meta)
    export_program_model(model, args, device, meta, x_control, y_delta, genes, outdir, "our_shared_v2")

    hybrid = OurHybridTeacherProgramTorch(args.latent_dim, args.programs, args.epochs * 2, args.random_state + 199, device)
    hybrid.fit(x_control, y_delta, meta)
    export_program_model(hybrid, args, device, meta, x_control, y_delta, genes, outdir, "our_hybrid_v3")
    return hybrid


def write_notes(summary: pd.DataFrame, outdir: Path) -> None:
    def table(df):
        display = df.astype(str)
        header = "| " + " | ".join(display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in display.to_numpy()])

    best = summary.sort_values(["split", "cosine_mean"], ascending=[True, False]).groupby("split").head(3)
    lines = [
        "# PyTorch Perturbation Benchmark",
        "",
        "This benchmark compares PyTorch adaptations of classic perturbation-model ideas on matched bulk deltas.",
        "",
        "Models:",
        "- `scGen_torch_latent_shift`: autoencoder latent shift.",
        "- `CPA_torch_additive`: supervised additive/covariate neural predictor.",
        "- `CellOT_torch_transport`: neural control-to-treated latent map.",
        "- `OUR_shared_program_v2`: supervised shared-program decoder designed to identify cross-drug programs.",
        "- `OUR_hybrid_teacher_program_v3`: upgraded hybrid using scGen/CellOT direction teachers plus shared-program decoder.",
        "",
        "Best by cosine:",
        table(best),
        "",
        "Interpretation:",
        "- Cosine/Pearson/top-gene overlap measure perturbation direction and signature recovery.",
        "- RMSE is conservative and often favors near-zero predictions because many genes have weak deltas.",
        "- A useful model for the planned claim should do well on leave-entity/block splits and export stable shared programs.",
    ]
    (outdir / "torch_benchmark_notes.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    safe_mkdir(outdir)
    device = choose_device(args.device)
    print("Device:", device)
    meta, x_control, y_treated, y_delta, genes = load_arrays(args)
    meta.to_csv(outdir / "torch_benchmark_contrasts.csv", index=False)
    pd.Series(genes, name="gene").to_csv(outdir / "torch_benchmark_genes.csv", index=False)

    fold_metrics = run_cv(args, device, meta, x_control, y_delta)
    fold_metrics.to_csv(outdir / "torch_benchmark_fold_metrics.csv", index=False)
    summary = summarize(fold_metrics)
    summary.to_csv(outdir / "torch_benchmark_summary.csv", index=False)
    plot_summary(summary, outdir)
    if not args.skip_final_export:
        train_final_our_model(args, device, meta, x_control, y_delta, genes, outdir)
    write_notes(summary, outdir)
    print(summary.to_string(index=False))
    print(f"Done. Outputs: {outdir}")


if __name__ == "__main__":
    main()
