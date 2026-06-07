#!/usr/bin/env python3
"""Benchmark official perturbation-model implementations on the MM bulk task."""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import torch_model_benchmark as tb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--outdir", default="bulk_pre_sc/official_benchmark")
    parser.add_argument("--models", nargs="+", required=True, choices=["scgen", "cpa", "cellot", "trvae", "scvidr"])
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--scgen-epochs", type=int, default=20)
    parser.add_argument("--cpa-epochs", type=int, default=20)
    parser.add_argument("--trvae-epochs", type=int, default=20)
    parser.add_argument("--scvidr-epochs", type=int, default=20)
    parser.add_argument("--cellot-iters", type=int, default=300)
    parser.add_argument("--cellot-inner-iters", type=int, default=5)
    parser.add_argument("--cellot-path", default="/tmp/cellot_official")
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def validation_indices(train_idx: np.ndarray, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    n_valid = max(1, int(round(len(train_idx) * 0.15)))
    return set(rng.choice(train_idx, size=n_valid, replace=False).tolist())


def build_cpa_anndata(meta, x_control, y_treated, genes, train_idx, test_idx, seed):
    test_set = set(np.asarray(test_idx).tolist())
    valid_set = validation_indices(np.asarray(train_idx), seed)
    x = np.vstack([x_control, y_treated]).astype("float32")
    obs = []
    for role in ["control", "treated"]:
        for i, row in meta.iterrows():
            split = "test" if i in test_set else ("valid" if i in valid_set else "train")
            condition = "control" if role == "control" else str(row["drug_token"])
            obs.append(
                {
                    "contrast_index": i,
                    "context_id": row["context_id"],
                    "drug_token": str(row["drug_token"]),
                    "condition": condition,
                    "dose": "1.0",
                    "split": split,
                    "role": role,
                }
            )
    return ad.AnnData(x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=genes), dtype=x.dtype)


def benchmark_cpa(args, meta, x_control, y_treated, y_delta, genes) -> pd.DataFrame:
    import cpa

    rows = []
    warnings.filterwarnings("ignore")
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"[CPA official] {split_name}")
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            adata = build_cpa_anndata(meta, x_control, y_treated, genes, train_idx, test_idx, args.random_state + fold)
            cpa.CPA.pert_encoder = None
            cpa.CPA.covars_encoder = None
            cpa.CPA.pert_smiles_map = {}
            cpa.CPA.setup_anndata(
                adata,
                perturbation_key="condition",
                control_group="control",
                dosage_key="dose",
                categorical_covariate_keys=[],
                is_count_data=False,
                max_comb_len=1,
            )
            model = cpa.CPA(
                adata,
                split_key="split",
                train_split="train",
                valid_split="valid",
                test_split="test",
                n_latent=args.latent_dim,
                n_hidden_encoder=64,
                n_layers_encoder=1,
                n_hidden_decoder=64,
                n_layers_decoder=1,
                recon_loss="gauss",
                variational=False,
                use_batch_norm_encoder=False,
                use_batch_norm_decoder=False,
                dropout_rate_encoder=0.05,
                dropout_rate_decoder=0.05,
            )
            model.train(
                max_epochs=args.cpa_epochs,
                use_gpu=False,
                batch_size=args.batch_size,
                check_val_every_n_epoch=5,
                early_stopping_patience=5,
                enable_progress_bar=False,
                logger=False,
            )
            query = adata[(adata.obs["role"].eq("control")) & (adata.obs["split"].eq("test"))].copy()
            query.obs["condition"] = query.obs["drug_token"].astype(str)
            model.predict(query, batch_size=args.batch_size, n_samples=1)
            pred_expr = np.asarray(query.obsm["CPA_pred"], dtype=np.float32)
            pos = {int(i): j for j, i in enumerate(query.obs["contrast_index"].astype(int).to_numpy())}
            pred = np.vstack([pred_expr[pos[int(i)]] - x_control[int(i)] for i in test_idx]).astype("float32")
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "official_cpa_compositional_autoencoder",
                    "fold": fold,
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    "n_train": len(train_idx),
                    "unseen_drug_predictions": 0,
                }
            )
            rows.append(row)
            print(f"  fold {fold}: cosine={row['cosine']:.3f}, top100={row['top100_overlap']:.3f}")
    return pd.DataFrame(rows)


def build_scgen_train_anndata(meta, x_control, y_treated, genes, train_idx):
    x = np.vstack([x_control[train_idx], y_treated[train_idx]]).astype("float32")
    obs = []
    for i in train_idx:
        row = meta.iloc[int(i)]
        obs.append(
            {
                "contrast_index": int(i),
                "context_id": row["context_id"],
                "condition": "control",
                "cell_type": "bulk",
                "drug_token": str(row["drug_token"]),
                "role": "control",
            }
        )
    for i in train_idx:
        row = meta.iloc[int(i)]
        obs.append(
            {
                "contrast_index": int(i),
                "context_id": row["context_id"],
                "condition": str(row["drug_token"]),
                "cell_type": "bulk",
                "drug_token": str(row["drug_token"]),
                "role": "treated",
            }
        )
    return ad.AnnData(x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=genes), dtype=x.dtype)


def scgen_global_predict(model, train_adata, query):
    import torch

    ctrl = train_adata[train_adata.obs["condition"].eq("control")].copy()
    stim = train_adata[~train_adata.obs["condition"].eq("control")].copy()
    delta = model._avg_vector(stim) - model._avg_vector(ctrl)
    latent = model.get_latent_representation(query)
    pred_latent = latent + delta
    return model.module.generative(torch.Tensor(pred_latent))["px"].cpu().detach().numpy()


def benchmark_scgen(args, meta, x_control, y_treated, y_delta, genes) -> pd.DataFrame:
    import scgen

    rows = []
    warnings.filterwarnings("ignore")
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"[scGen official] {split_name}")
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            adata = build_scgen_train_anndata(meta, x_control, y_treated, genes, train_idx)
            scgen.SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
            model = scgen.SCGEN(
                adata,
                n_hidden=64,
                n_latent=args.latent_dim,
                n_layers=1,
                dropout_rate=0.05,
            )
            model.train(
                max_epochs=args.scgen_epochs,
                use_gpu=False,
                batch_size=args.batch_size,
                early_stopping=False,
                logger=False,
            )
            train_drugs = set(adata.obs.loc[adata.obs["role"].eq("treated"), "condition"].astype(str))
            pred = np.zeros((len(test_idx), len(genes)), dtype=np.float32)
            unseen = 0
            for drug in pd.Series(meta.iloc[test_idx]["drug_token"].astype(str)).unique():
                local = np.where(meta.iloc[test_idx]["drug_token"].astype(str).to_numpy() == drug)[0]
                q_idx = test_idx[local]
                q = ad.AnnData(
                    x_control[q_idx].astype("float32"),
                    obs=pd.DataFrame({"condition": ["control"] * len(q_idx), "cell_type": ["bulk"] * len(q_idx)}),
                    var=pd.DataFrame(index=genes),
                    dtype="float32",
                )
                if drug in train_drugs:
                    pred_adata, _ = model.predict(ctrl_key="control", stim_key=drug, adata_to_predict=q)
                    pred_expr = np.asarray(pred_adata.X, dtype=np.float32)
                else:
                    unseen += len(q_idx)
                    pred_expr = np.asarray(scgen_global_predict(model, adata, q), dtype=np.float32)
                pred[local] = pred_expr - x_control[q_idx]
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "official_scgen_latent_arithmetic",
                    "fold": fold,
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    "n_train": len(train_idx),
                    "unseen_drug_predictions": unseen,
                }
            )
            rows.append(row)
            print(f"  fold {fold}: cosine={row['cosine']:.3f}, top100={row['top100_overlap']:.3f}, unseen={unseen}")
    return pd.DataFrame(rows)


def benchmark_cellot(args, meta, x_control, y_treated, y_delta, genes) -> pd.DataFrame:
    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cellot_path = Path(args.cellot_path)
    if cellot_path.exists():
        sys.path.insert(0, str(cellot_path))
    from cellot.models.cellot import compute_loss_f, compute_loss_g
    from cellot.networks.icnns import ICNN

    rows = []
    rng = np.random.default_rng(args.random_state)
    torch.manual_seed(args.random_state)
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"[CellOT official ICNN] {split_name}")
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            combined = np.vstack([x_control[train_idx], y_treated[train_idx]])
            scaler = StandardScaler().fit(combined)
            n_components = min(args.latent_dim, combined.shape[0] - 1, combined.shape[1])
            pca = PCA(n_components=n_components, random_state=args.random_state).fit(scaler.transform(combined))
            source = torch.tensor(pca.transform(scaler.transform(x_control[train_idx])).astype("float32"))
            target = torch.tensor(pca.transform(scaler.transform(y_treated[train_idx])).astype("float32"))

            f = ICNN(input_dim=n_components, hidden_units=[64, 64], softplus_W_kernels=False, fnorm_penalty=0)
            g = ICNN(input_dim=n_components, hidden_units=[64, 64], softplus_W_kernels=False, fnorm_penalty=1)
            opt_f = torch.optim.Adam(f.parameters(), lr=1e-4, betas=(0.5, 0.9))
            opt_g = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
            batch_size = min(args.batch_size, len(train_idx))
            for _ in range(args.cellot_iters):
                target_batch = target[rng.choice(len(target), size=batch_size, replace=True)]
                source_index = rng.choice(len(source), size=batch_size, replace=True)
                for _inner in range(args.cellot_inner_iters):
                    xb = source[source_index].clone().requires_grad_(True)
                    opt_g.zero_grad()
                    loss_g = compute_loss_g(f, g, xb).mean() + g.penalize_w()
                    loss_g.backward()
                    opt_g.step()
                xb = source[source_index].clone().requires_grad_(True)
                opt_f.zero_grad()
                loss_f = compute_loss_f(f, g, xb, target_batch).mean()
                loss_f.backward()
                opt_f.step()
                f.clamp_w()

            zc = torch.tensor(
                pca.transform(scaler.transform(x_control[test_idx])).astype("float32"),
                requires_grad=True,
            )
            g.eval()
            zt = g.transport(zc).detach().numpy()
            pred_expr = scaler.inverse_transform(pca.inverse_transform(zt))
            pred = (pred_expr - x_control[test_idx]).astype("float32")
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "official_cellot_neural_ot",
                    "fold": fold,
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    "n_train": len(train_idx),
                    "unseen_drug_predictions": 0,
                }
            )
            rows.append(row)
            print(f"  fold {fold}: cosine={row['cosine']:.3f}, top100={row['top100_overlap']:.3f}")
    return pd.DataFrame(rows)


def benchmark_trvae(args, meta, x_control, y_treated, y_delta, genes) -> pd.DataFrame:
    import torch
    from scarches.models import TRVAE

    rows = []
    warnings.filterwarnings("ignore")

    def decode(model, z: np.ndarray, condition: str) -> np.ndarray:
        device = next(model.model.parameters()).device
        if condition not in model.model.condition_encoder:
            condition = "control"
        label = model.model.condition_encoder[condition]
        c = torch.tensor(np.full(z.shape[0], label), device=device).long()
        zt = torch.tensor(z.astype("float32"), device=device)
        with torch.no_grad():
            return model.model.decoder(zt, c)[0].detach().cpu().numpy()

    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"[trVAE] {split_name}")
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            x = np.vstack([x_control[train_idx], y_treated[train_idx]]).astype("float32")
            obs = []
            for i in train_idx:
                row = meta.iloc[int(i)]
                obs.append({"condition": "control", "drug_token": str(row["drug_token"])})
            for i in train_idx:
                row = meta.iloc[int(i)]
                obs.append({"condition": str(row["drug_token"]), "drug_token": str(row["drug_token"])})
            adata = ad.AnnData(x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=genes), dtype=x.dtype)
            model = TRVAE(
                adata,
                condition_key="condition",
                hidden_layer_sizes=[64, 32],
                latent_dim=min(args.latent_dim, 32),
                recon_loss="mse",
                use_mmd=True,
                use_bn=False,
                use_ln=True,
            )
            model.train(n_epochs=args.trvae_epochs, batch_size=args.batch_size, early_stopping=False, verbose=False)
            z_control = model.get_latent(x_control[test_idx].astype("float32"), np.array(["control"] * len(test_idx)), mean=True)
            pred = np.zeros((len(test_idx), len(genes)), dtype=np.float32)
            for drug in pd.Series(meta.iloc[test_idx]["drug_token"].astype(str)).unique():
                local = np.where(meta.iloc[test_idx]["drug_token"].astype(str).to_numpy() == drug)[0]
                pred_expr = decode(model, z_control[local], drug)
                pred[local] = pred_expr - x_control[test_idx[local]]
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "official_trvae_conditional_vae",
                    "fold": fold,
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    "n_train": len(train_idx),
                    "unseen_drug_predictions": int(sum(str(meta.iloc[i]["drug_token"]) not in model.model.condition_encoder for i in test_idx)),
                }
            )
            rows.append(row)
            print(f"  fold {fold}: cosine={row['cosine']:.3f}, top100={row['top100_overlap']:.3f}")
    return pd.DataFrame(rows)


def benchmark_scvidr(args, meta, x_control, y_treated, y_delta, genes) -> pd.DataFrame:
    import torch
    from sklearn.linear_model import LinearRegression
    from scvi.data import setup_anndata
    from vidr import VIDR

    rows = []
    warnings.filterwarnings("ignore")
    for split_name, splits in tb.splitters(meta, args.random_state):
        print(f"[scVIDR] {split_name}")
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_meta = meta.iloc[train_idx].reset_index(drop=True)
            x = np.vstack([x_control[train_idx], y_treated[train_idx]]).astype("float32")
            obs = []
            for j, i in enumerate(train_idx):
                row = meta.iloc[int(i)]
                obs.append(
                    {
                        "dose": "control",
                        "celltype": str(row["entity_norm"]),
                        "drug_token": str(row["drug_token"]),
                        "contrast_local": j,
                    }
                )
            for j, i in enumerate(train_idx):
                row = meta.iloc[int(i)]
                obs.append(
                    {
                        "dose": str(row["drug_token"]),
                        "celltype": str(row["entity_norm"]),
                        "drug_token": str(row["drug_token"]),
                        "contrast_local": j,
                    }
                )
            adata = ad.AnnData(x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=genes), dtype=x.dtype)
            adata = setup_anndata(adata, copy=True, batch_key="dose", labels_key="celltype")
            model = VIDR(
                adata,
                hidden_dim=64,
                latent_dim=min(args.latent_dim, 32),
                n_hidden_layers=1,
                dropout_rate=0.05,
                linear_decoder=False,
            )
            model.train(
                max_epochs=args.scvidr_epochs,
                use_gpu=False,
                batch_size=args.batch_size,
                early_stopping=False,
                train_size=0.9,
                check_val_every_n_epoch=max(1, args.scvidr_epochs),
            )
            train_latent = model.get_latent_representation(adata, batch_size=args.batch_size)
            n_train = len(train_idx)
            ctrl_lat = train_latent[:n_train]
            treat_lat = train_latent[n_train:]
            global_delta = (treat_lat - ctrl_lat).mean(axis=0)
            if hasattr(adata.obs["celltype"], "cat"):
                known_labels = list(adata.obs["celltype"].cat.categories.astype(str))
            else:
                known_labels = sorted(adata.obs["celltype"].astype(str).unique().tolist())
            safe_label = "UNK" if "UNK" in known_labels else known_labels[0]
            query = ad.AnnData(
                x_control[test_idx].astype("float32"),
                obs=pd.DataFrame({"dose": ["control"] * len(test_idx), "celltype": [safe_label] * len(test_idx)}),
                var=pd.DataFrame(index=genes),
                dtype="float32",
            )
            query_latent = model.get_latent_representation(query, batch_size=args.batch_size)
            pred_latent = np.zeros_like(query_latent)
            unseen = 0
            for pos, idx in enumerate(test_idx):
                drug = str(meta.iloc[int(idx)]["drug_token"])
                mask = train_meta["drug_token"].astype(str).eq(drug).to_numpy()
                if mask.sum() == 0:
                    delta = global_delta
                    unseen += 1
                else:
                    drug_delta = treat_lat[mask] - ctrl_lat[mask]
                    if mask.sum() >= 3:
                        reg = LinearRegression().fit(ctrl_lat[mask], drug_delta)
                        delta = reg.predict(query_latent[[pos]])[0]
                    else:
                        delta = drug_delta.mean(axis=0)
                pred_latent[pos] = query_latent[pos] + delta
            with torch.no_grad():
                pred_expr = model.module.generative(torch.tensor(pred_latent.astype("float32")))["px"].detach().cpu().numpy()
            pred = (pred_expr - x_control[test_idx]).astype("float32")
            row = tb.metrics(y_delta[test_idx], pred)
            row.update(
                {
                    "split": split_name,
                    "model": "official_scvidr_regressed_vae",
                    "fold": fold,
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    "n_train": len(train_idx),
                    "unseen_drug_predictions": unseen,
                }
            )
            rows.append(row)
            print(f"  fold {fold}: cosine={row['cosine']:.3f}, top100={row['top100_overlap']:.3f}, unseen={unseen}")
    return pd.DataFrame(rows)


def write_outputs(outdir: Path, model_name: str, fold_metrics: pd.DataFrame) -> None:
    safe_mkdir(outdir)
    fold_metrics.to_csv(outdir / f"{model_name}_fold_metrics.csv", index=False)
    summary = summarize(fold_metrics)
    summary.to_csv(outdir / f"{model_name}_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    safe_mkdir(outdir)
    load_args = SimpleNamespace(article_dir=args.article_dir, expr=args.expr, n_genes=args.n_genes)
    meta, x_control, y_treated, y_delta, genes = tb.load_arrays(load_args)
    pd.Series(genes, name="gene").to_csv(outdir / "official_benchmark_genes.csv", index=False)

    runners = {
        "scgen": benchmark_scgen,
        "cpa": benchmark_cpa,
        "cellot": benchmark_cellot,
        "trvae": benchmark_trvae,
        "scvidr": benchmark_scvidr,
    }
    all_rows = []
    for model_name in args.models:
        fold_metrics = runners[model_name](args, meta, x_control, y_treated, y_delta, genes)
        write_outputs(outdir, model_name, fold_metrics)
        all_rows.append(fold_metrics)
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(outdir / "official_benchmark_fold_metrics.csv", index=False)
        summarize(combined).to_csv(outdir / "official_benchmark_summary.csv", index=False)


if __name__ == "__main__":
    main()
