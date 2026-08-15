#!/usr/bin/env python3
"""Build and evaluate a simple scRNA-only signed SVD program baseline.

This is a reviewer-response control for the claim that the integrated
PRISM-MM representation outperforms an scRNA-only representation. It learns
program weights only from paired scRNA drug-control gene deltas, exports a
program gene table, and evaluates those programs with the same discovery bulk,
held-out bulk, and calibration scRNA scoring scripts used for PRISM-MM.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "bulk_pre_sc"
sys.path.append(str(SOURCE_DIR))

import train_v14_multimodal_dictionary as v14  # noqa: E402


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
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def export_gene_table(matrix: np.ndarray, genes: list[str], outdir: Path, prefix: str, top_n: int) -> None:
    rows = []
    for component_idx, weights in enumerate(matrix, start=1):
        program = f"Program_{component_idx}"
        up_order = np.argsort(weights)[::-1]
        down_order = np.argsort(weights)
        for direction, order in [("up", up_order), ("down", down_order)]:
            keep = [idx for idx in order if (weights[idx] > 0 if direction == "up" else weights[idx] < 0)][:top_n]
            for rank, idx in enumerate(keep, start=1):
                rows.append(
                    {
                        "program": program,
                        "direction": direction,
                        "rank": rank,
                        "gene": genes[idx],
                        "weight": float(weights[idx]),
                        "selection_probability": 1.0,
                    }
                )
    pd.DataFrame(rows).to_csv(outdir / f"{prefix}_core_program_genes.csv", index=False)


def summarize(model_dir: Path, model: str) -> pd.DataFrame:
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
    detail = pd.DataFrame(rows)
    detail.to_csv(model_dir / "three_layer_program_detail.csv", index=False)
    direction = detail["triple_direction_concordant"]
    significant = detail["threeway_sample_significant_fdr_0_10"]
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "triple_direction_concordant_n": int(direction.sum()),
                "triple_programs": ";".join(detail.loc[direction, "program"]),
                "threeway_sample_significant_n": int(significant.sum()),
                "threeway_sample_significant_programs": ";".join(detail.loc[significant, "program"]),
            }
        ]
    )
    summary.to_csv(model_dir / "three_layer_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    model = "scrna_only_svd"
    prefix = model
    model_dir = Path(args.outdir) / model
    model_dir.mkdir(parents=True, exist_ok=True)

    tb_args = argparse.Namespace(
        expr=str(ROOT / args.expr),
        article_dir=str(ROOT / args.article_dir),
        n_genes=2000,
        seed=args.seed,
    )
    _, _, _, _, genes = v14.tb.load_arrays(tb_args)
    scrna_raw, _, _ = v14.scrna_matrix(ROOT / args.scrna_gene_deltas, genes)
    z = StandardScaler().fit_transform(scrna_raw).astype("float32")
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    scores = pca.fit_transform(z)
    components = pca.components_.astype("float32")
    for idx in range(len(components)):
        if float(scores[:, idx].mean()) < 0:
            components[idx] *= -1

    export_gene_table(components, genes, model_dir, prefix, args.top_n)
    pd.DataFrame({"gene": genes}).to_csv(model_dir / "v14_genes.csv", index=False)
    pd.DataFrame({"component": np.arange(1, len(components) + 1), "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
        model_dir / "scrna_only_svd_explained_variance.csv", index=False
    )
    np.save(model_dir / f"{prefix}_core_program_matrix.npy", components)

    validation = model_dir / "validation"
    validation.mkdir(parents=True, exist_ok=True)
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
    summary = summarize(model_dir, model)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
