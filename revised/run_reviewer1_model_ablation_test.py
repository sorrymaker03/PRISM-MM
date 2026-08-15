#!/usr/bin/env python3
"""Run reviewer-requested PRISM-MM ablation tests without drawing figures.

The goal is to test whether PRISM-MM's integrated representation is necessary:

1. Full PRISM-MM.
2. A bulk-only v9-anchor variant with scRNA/cross-modal losses disabled.
3. A no-bulk-loss variant to test whether bulk reconstruction contributes to
   drug-response recovery.
4. A no-alignment-loss variant to test whether cross-modal/study-alignment
   losses matter.

Outputs are written to a user-specified directory and do not overwrite
manuscript figure/table files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "bulk_pre_sc"
sys.path.append(str(SOURCE_DIR))

import train_v14_multimodal_dictionary as v14  # noqa: E402


VARIANTS: dict[str, dict[str, object]] = {
    "full_prism_mm": {
        "description": "Current PRISM-MM configuration.",
        "anchor_genes": "bulk_pre_sc/model_upgrade_v13_cross_modal_adapter/our_cross_modal_adapter_v13_core_program_genes.csv",
        "cfg_edits": {},
    },
    "bulk_only_v9_anchor": {
        "description": "Bulk-derived v9 anchor; scRNA and cross-modal losses disabled.",
        "anchor_genes": "bulk_pre_sc/model_upgrade_v9_final/our_anchored_sparse_attention_v9_core_program_genes.csv",
        "cfg_edits": {
            "scrna_reconstruction_lambda": 0.0,
            "adapter_alignment_lambda": 0.0,
            "distribution_alignment_lambda": 0.0,
            "conditional_invariance_lambda": 0.0,
        },
    },
    "no_bulk_loss": {
        "description": "PRISM-MM structure with bulk reconstruction disabled.",
        "anchor_genes": "bulk_pre_sc/model_upgrade_v13_cross_modal_adapter/our_cross_modal_adapter_v13_core_program_genes.csv",
        "cfg_edits": {
            "bulk_reconstruction_lambda": 0.0,
            "conditional_invariance_lambda": 0.0,
        },
    },
    "no_alignment_loss": {
        "description": "PRISM-MM structure without study/cross-modal alignment losses.",
        "anchor_genes": "bulk_pre_sc/model_upgrade_v13_cross_modal_adapter/our_cross_modal_adapter_v13_core_program_genes.csv",
        "cfg_edits": {
            "adapter_alignment_lambda": 0.0,
            "distribution_alignment_lambda": 0.0,
            "conditional_invariance_lambda": 0.0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/tmp/prism_review1_test/revised_ablation")
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--article-dir", default="bulk_pre_sc/article_results")
    parser.add_argument("--v9-dir", default="bulk_pre_sc/model_upgrade_v9_final")
    parser.add_argument("--v9-prefix", default="our_anchored_sparse_attention_v9")
    parser.add_argument(
        "--scrna-gene-deltas",
        default="bulk_pre_sc/model_upgrade_v9_final/gene_direction_validation/scrna_all_adaptive_gene_delta_by_pair.csv",
    )
    parser.add_argument("--variant", default="v14_balanced", choices=["v14_balanced", "v14_anchor_strong", "v14_alignment", "v14_dictionary_expand"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--oof-epochs", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps"])
    parser.add_argument("--skip-validation", action="store_true", help="Only train and export fit metrics; skip bulk/scRNA program validation.")
    parser.add_argument("--only", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    return parser.parse_args()


def apply_config_edits(cfg: v14.MultimodalConfig, edits: dict[str, float]) -> v14.MultimodalConfig:
    for key, value in edits.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg


def training_args(args: argparse.Namespace, outdir: Path, prefix: str, anchor_genes: str) -> SimpleNamespace:
    return SimpleNamespace(
        expr=str(ROOT / args.expr),
        article_dir=str(ROOT / args.article_dir),
        v9_dir=str(ROOT / args.v9_dir),
        v9_prefix=args.v9_prefix,
        anchor_genes=str(ROOT / anchor_genes),
        scrna_gene_deltas=str(ROOT / args.scrna_gene_deltas),
        outdir=str(outdir),
        prefix=prefix,
        variant=args.variant,
        epochs=args.epochs,
        oof_epochs=args.oof_epochs,
        n_genes=2000,
        folds=args.folds,
        seed=args.seed,
        device=args.device,
    )


def fit_one(args: argparse.Namespace, name: str, spec: dict[str, object]) -> pd.DataFrame:
    outdir = Path(args.outdir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = name
    local_args = training_args(args, outdir, prefix, str(spec["anchor_genes"]))
    cfg = apply_config_edits(v14.config_for(args.variant), dict(spec["cfg_edits"]))
    device = v14.tb.choose_device(args.device)

    meta, _, _, bulk_delta, genes = v14.tb.load_arrays(local_args)
    anchor, anchor_sizes = v14.load_anchor(Path(local_args.anchor_genes), genes, cfg.n_core)
    pathway = v14.pathway_matrix(Path(local_args.v9_dir) / f"{local_args.v9_prefix}_core_program_enrichr_pathways.csv", genes, cfg.n_core)
    scrna_raw, scrna_mask, scrna_pairs = v14.scrna_matrix(Path(local_args.scrna_gene_deltas), genes)

    bulk_scaler = StandardScaler().fit(bulk_delta)
    scrna_scaler = StandardScaler().fit(scrna_raw)
    bulk_z = bulk_scaler.transform(bulk_delta).astype("float32")
    scrna_z = scrna_scaler.transform(scrna_raw).astype("float32")

    model, history = v14.train_model(bulk_z, scrna_z, scrna_mask, meta, anchor, pathway, cfg, args.epochs, args.seed, device)
    matrix = v14.original_shared_matrix(model, bulk_scaler)
    gates = model.expected_gate_probability().detach().cpu().numpy()
    gene_table = v14.gene_table(matrix, gates, genes, anchor_sizes, cfg.export_size_multiplier)

    with v14.torch.no_grad():
        bulk_fit, scrna_fit, _ = model()
    fit_metrics = v14.tb.metrics(bulk_z, bulk_fit.cpu().numpy())
    fit_metrics["scrna_masked_rmse"] = float(np.sqrt(np.mean((scrna_fit.cpu().numpy()[:, scrna_mask > 0] - scrna_z[:, scrna_mask > 0]) ** 2)))
    fit_metrics["model"] = name
    fit_metrics["description"] = str(spec["description"])

    meta.to_csv(outdir / "v14_training_contrasts.csv", index=False)
    pd.Series(genes, name="gene").to_csv(outdir / "v14_genes.csv", index=False)
    gene_table.to_csv(outdir / f"{prefix}_core_program_genes.csv", index=False)
    history.to_csv(outdir / f"{prefix}_training_history.csv", index=False)
    pd.DataFrame([fit_metrics]).to_csv(outdir / f"{prefix}_fit_metrics.csv", index=False)
    np.save(outdir / f"{prefix}_core_program_matrix.npy", matrix)
    np.save(outdir / f"{prefix}_gene_gate_probability.npy", gates)
    v14.export_oof(local_args, cfg, device, meta, bulk_z, scrna_raw, scrna_mask, scrna_pairs, genes, anchor, anchor_sizes, pathway, bulk_scaler, outdir)

    (outdir / f"{prefix}_config.json").write_text(
        json.dumps(
            {
                "model": name,
                "description": spec["description"],
                "base_variant": args.variant,
                "anchor_genes": spec["anchor_genes"],
                "cfg_edits": spec["cfg_edits"],
                "epochs": args.epochs,
                "oof_epochs": args.oof_epochs,
                "note": "Reviewer-response ablation test. Outputs are not manuscript source files.",
            },
            indent=2,
        )
        + "\n"
    )
    return pd.DataFrame([fit_metrics])


def run_validation(args: argparse.Namespace, name: str) -> None:
    model_dir = Path(args.outdir) / name
    validation = model_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    prefix = name
    subprocess.run(
        [
            sys.executable,
            str(SOURCE_DIR / "evaluate_v6_core_programs.py"),
            "--model-dir",
            str(model_dir),
            "--prefix",
            prefix,
            "--outdir",
            str(validation),
            "--top-n",
            str(args.top_n),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "Rscript",
            str(SOURCE_DIR / "evaluate_v6_core_programs_scrna.R"),
            str(model_dir),
            prefix,
            str(validation),
            str(args.top_n),
        ],
        cwd=ROOT,
        check=True,
    )


def bh_summary(model_dir: Path, model: str) -> pd.DataFrame:
    validation = model_dir / "validation"
    discovery = pd.read_csv(validation / "discovery_core_program_shift_summary.csv").set_index("program")
    heldout = pd.read_csv(validation / "heldout_sample_core_program_shift_summary.csv").set_index("program")
    scrna = pd.read_csv(validation / "scrna_paired_core_program_shift_summary.csv").set_index("program")
    rows = []
    for program in discovery.index:
        row = {
            "model": model,
            "program": program,
            "discovery_mean": float(discovery.loc[program, "mean_delta"]),
            "discovery_fdr": float(discovery.loc[program, "fdr"]),
            "heldout_mean": float(heldout.loc[program, "mean_delta"]),
            "heldout_fdr": float(heldout.loc[program, "fdr"]),
            "calibration_scrna_mean": float(scrna.loc[program, "mean_delta"]),
            "calibration_scrna_q": float(scrna.loc[program, "q_value"]),
        }
        signs = np.sign([row["discovery_mean"], row["heldout_mean"], row["calibration_scrna_mean"]])
        row["triple_direction_concordant"] = bool(signs[0] == signs[1] == signs[2])
        row["threeway_sample_significant_fdr_0_10"] = bool(
            row["triple_direction_concordant"]
            and row["discovery_fdr"] < 0.10
            and row["heldout_fdr"] < 0.10
            and row["calibration_scrna_q"] < 0.10
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_outputs(args: argparse.Namespace, fit_frames: list[pd.DataFrame]) -> None:
    outdir = Path(args.outdir)
    fit = pd.concat(fit_frames, ignore_index=True)
    fit = fit[["model", "description", "rmse", "mae", "cosine", "pearson", "spearman", "top50_overlap", "top100_overlap", "top100_sign_agreement", "magnitude_fidelity", "scrna_masked_rmse"]]
    fit.to_csv(outdir / "fit_metrics_summary.csv", index=False)

    if args.skip_validation:
        print("\nFit metrics")
        print(fit.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
        return

    detail = pd.concat([bh_summary(outdir / name, name) for name in args.only], ignore_index=True)
    detail.to_csv(outdir / "three_layer_program_detail.csv", index=False)
    rows = []
    for model, group in detail.groupby("model", sort=False):
        direction = group["triple_direction_concordant"]
        significant = group["threeway_sample_significant_fdr_0_10"]
        rows.append(
            {
                "model": model,
                "triple_direction_concordant_n": int(direction.sum()),
                "triple_programs": ";".join(group.loc[direction, "program"]),
                "threeway_sample_significant_n": int(significant.sum()),
                "threeway_sample_significant_programs": ";".join(group.loc[significant, "program"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "three_layer_summary.csv", index=False)
    print("\nFit metrics")
    print(fit.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print("\nThree-layer program summary")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fit_frames = []
    for name in args.only:
        print(f"\n=== Running {name} ===", flush=True)
        fit_frames.append(fit_one(args, name, VARIANTS[name]))
        if not args.skip_validation:
            run_validation(args, name)
    summarize_outputs(args, fit_frames)


if __name__ == "__main__":
    main()
