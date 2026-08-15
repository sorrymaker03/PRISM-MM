#!/usr/bin/env python3
"""Create Figure 2 benchmark for the shared-program prediction backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


PALETTE = {
    "ink": "#000000",
    "muted": "#000000",
    "grid": "#D0D5DD",
    "light": "#F8FAFC",
    "blue": "#2B6CB0",
    "teal": "#0F766E",
    "orange": "#C2410C",
    "red": "#B42318",
    "purple": "#7A5AF8",
    "pink": "#C11574",
    "gray": "#98A2B3",
}

METHOD_LABELS = {
    "matched_DEG_drug_mean": "Drug-mean\nbaseline",
    "official_scgen_latent_arithmetic": "scGen",
    "official_cpa_compositional_autoencoder": "CPA",
    "official_cellot_neural_ot": "CellOT",
    "official_trvae_conditional_vae": "trVAE",
    "official_scvidr_regressed_vae": "scVIDR",
    "OUR_hybrid_teacher_program_v3": "PRISM-MM",
}

METHOD_ORDER = [
    "matched_DEG_drug_mean",
    "official_scgen_latent_arithmetic",
    "official_cpa_compositional_autoencoder",
    "official_cellot_neural_ot",
    "official_trvae_conditional_vae",
    "official_scvidr_regressed_vae",
    "OUR_hybrid_teacher_program_v3",
]

METHOD_COLORS = {
    "Drug-mean\nbaseline": PALETTE["gray"],
    "scGen": PALETTE["blue"],
    "CPA": PALETTE["orange"],
    "CellOT": PALETTE["teal"],
    "trVAE": PALETTE["purple"],
    "scVIDR": PALETTE["pink"],
    "PRISM-MM": PALETTE["red"],
}

SPLIT_LABELS = {
    "random_5fold": "Random",
    "leave_entity_out": "Cell source",
    "leave_block_out": "Study",
}

SPLIT_ORDER = ["Cell source", "Study", "Random"]

DIRECTION_SCORE_COMPONENTS = [
    "cosine",
    "pearson",
    "spearman",
    "top50_overlap",
    "top100_overlap",
    "top100_sign_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-dir", default="bulk_pre_sc/torch_benchmark")
    parser.add_argument("--official-dir", default="bulk_pre_sc/official_benchmark")
    parser.add_argument("--outdir", default="bulk_pre_sc/main_figures")
    parser.add_argument("--panel-dir", default="bulk_pre_sc/main_figure_panels")
    parser.add_argument("--supp-panel-dir", default="bulk_pre_sc/supplement/figure_panels")
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.06,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def load_benchmark(torch_dir: Path, official_dir: Path) -> pd.DataFrame:
    rows = []
    torch_fold = pd.read_csv(torch_dir / "torch_benchmark_fold_metrics.csv")
    keep_torch = torch_fold[
        torch_fold["model"].isin(["matched_DEG_drug_mean", "OUR_hybrid_teacher_program_v3"])
    ].copy()
    keep_torch["source"] = "current_study"
    rows.append(keep_torch)

    for path in sorted(official_dir.glob("*_fold_metrics.csv")):
        if path.name == "official_benchmark_fold_metrics.csv":
            continue
        df = pd.read_csv(path)
        df["source"] = "official_implementation"
        rows.append(df)

    fold = pd.concat(rows, ignore_index=True, sort=False)
    fold = fold[fold["model"].isin(METHOD_ORDER)].copy()
    fold["method_label"] = fold["model"].map(METHOD_LABELS)
    fold["split_label"] = fold["split"].map(SPLIT_LABELS)
    fold["model"] = pd.Categorical(fold["model"], METHOD_ORDER, ordered=True)
    fold["method_label"] = pd.Categorical(
        fold["method_label"],
        [METHOD_LABELS[m] for m in METHOD_ORDER if m in set(fold["model"].astype(str))],
        ordered=True,
    )
    return fold.sort_values(["split", "model", "fold"])


def summarize(fold: pd.DataFrame) -> pd.DataFrame:
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
    summary = fold.groupby(["split", "split_label", "model", "method_label"], observed=True)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns]
    return summary.reset_index()


def add_direction_score(fold: pd.DataFrame) -> pd.DataFrame:
    """Calculate a balanced score for recovery of perturbation programs."""
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


def metric_barplot(
    ax: plt.Axes,
    fold: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    panel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    sns.barplot(
        data=fold,
        x="split_label",
        y=metric,
        hue="method_label",
        order=SPLIT_ORDER,
        hue_order=list(fold["method_label"].cat.categories),
        errorbar="sd",
        palette=METHOD_COLORS,
        ax=ax,
    )
    panel_label(ax, panel)
    ax.set_title(title, loc="left", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0, labelsize=8.2)
    ax.tick_params(axis="y", labelsize=8.2)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.55, alpha=0.75)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)


def direction_score_barplot(ax: plt.Axes, fold: pd.DataFrame, panel: str) -> None:
    score_summary = (
        fold.groupby("method_label", observed=True)["direction_score"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    labels = score_summary["method_label"].astype(str).str.replace("\n", " ", regex=False)
    colors = [METHOD_COLORS[str(label)] for label in score_summary["method_label"]]
    bars = ax.barh(
        labels,
        score_summary["mean"],
        xerr=score_summary["std"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, value, err in zip(bars, score_summary["mean"], score_summary["std"]):
        ax.text(value + err + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7.8)
    panel_label(ax, panel)
    ax.set_title("Program-recovery score", loc="left", fontsize=10.4, fontweight="bold")
    ax.set_xlabel("Program-recovery score")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 0.62)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.55, alpha=0.75)
    ax.tick_params(axis="both", labelsize=8.2)
    sns.despine(ax=ax)


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_3(fold: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(18.2, 10.7))
    gs = fig.add_gridspec(2, 4, wspace=0.34, hspace=0.44)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    metric_barplot(axes[0], fold, "cosine", "Cosine similarity", "Perturbation-direction recovery", "A")
    metric_barplot(axes[1], fold, "pearson", "Pearson correlation", "Gene-wise response correlation", "B")
    metric_barplot(axes[2], fold, "spearman", "Spearman correlation", "Gene-rank response correlation", "C")
    metric_barplot(axes[3], fold, "top50_overlap", "Top-50 overlap", "Top response-gene recovery", "D")
    metric_barplot(axes[4], fold, "top100_overlap", "Top-100 overlap", "Strong response-gene recovery", "E")
    metric_barplot(axes[5], fold, "top100_sign_agreement", "Sign agreement", "Direction of top response genes", "F", ylim=(0.0, 1.0))
    axes[5].axhline(0.5, color=PALETTE["grid"], lw=1.0, ls="--")
    metric_barplot(axes[6], fold, "magnitude_fidelity", "Magnitude fidelity", "Effect-size calibration", "G", ylim=(0.0, 1.02))
    direction_score_barplot(axes[7], fold, "H")

    handles = [
        Patch(facecolor=METHOD_COLORS[label], edgecolor="white", label=label.replace("\n", " "))
        for label in fold["method_label"].cat.categories
    ]
    fig.legend(
        handles=handles,
        title="",
        ncol=len(handles),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.975),
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.suptitle("Prediction-backbone benchmark against published perturbation models", fontsize=16, fontweight="bold", x=0.02, y=0.99, ha="left")
    save_figure(fig, outdir, "figure_2_model_benchmark")


def export_panels(fold: pd.DataFrame, panel_dir: Path, supp_panel_dir: Path) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    supp_panel_dir.mkdir(parents=True, exist_ok=True)
    for stale in panel_dir.glob("figure2_*.pdf"):
        stale.unlink()
    for stale_name in ("figureS1_A_rmse.pdf", "figureS1_B_mae.pdf"):
        stale = supp_panel_dir / stale_name
        if stale.exists():
            stale.unlink()

    metric_specs = [
        ("A", "cosine", "Cosine similarity", "Perturbation-direction recovery", None),
        ("B", "pearson", "Pearson correlation", "Gene-wise response correlation", None),
        ("C", "spearman", "Spearman correlation", "Gene-rank response correlation", None),
        ("D", "top50_overlap", "Top-50 overlap", "Top response-gene recovery", None),
        ("E", "top100_overlap", "Top-100 overlap", "Strong response-gene recovery", None),
        ("F", "top100_sign_agreement", "Sign agreement", "Direction of top response genes", (0.0, 1.0)),
        ("G", "magnitude_fidelity", "Magnitude fidelity", "Effect-size calibration", (0.0, 1.02)),
    ]
    for panel, metric, ylabel, title, ylim in metric_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        metric_barplot(ax, fold, metric, ylabel, title, panel, ylim=ylim)
        if panel == "A":
            handles = [
                Patch(facecolor=METHOD_COLORS[label], edgecolor="white", label=label.replace("\n", " "))
                for label in fold["method_label"].cat.categories
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
                fontsize=7.2,
                handlelength=1.1,
                columnspacing=0.9,
            )
        if panel == "F":
            ax.axhline(0.5, color=PALETTE["grid"], lw=1.0, ls="--")
        fig.savefig(panel_dir / f"figure2_{panel}_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    direction_score_barplot(ax, fold, "H")
    fig.savefig(panel_dir / "figure2_H_program_recovery_score.pdf", bbox_inches="tight")
    plt.close(fig)

    supp_specs = [
        ("A", "rmse", "RMSE", "Expression-scale error (lower better)", None),
        ("B", "mae", "MAE", "Absolute expression error (lower better)", None),
    ]
    for panel, metric, ylabel, title, ylim in supp_specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        metric_barplot(ax, fold, metric, ylabel, title, panel, ylim=ylim)
        if panel in {"A", "B"}:
            ax.set_ylim(bottom=0)
        fig.savefig(supp_panel_dir / f"figureS1_{panel}_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_notes(fold: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    source_dir = outdir.parent / "main_figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    fold.to_csv(source_dir / "figure_2_benchmark_source_metrics.csv", index=False)
    summary.to_csv(source_dir / "figure_2_benchmark_summary.csv", index=False)
    score_summary = (
        fold.groupby("method_label", observed=True)["direction_score"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    score_summary.to_csv(source_dir / "figure_2_program_recovery_score.csv", index=False)
    best = summary.sort_values(["split", "cosine_mean"], ascending=[True, False]).groupby("split", observed=True).head(3)
    best_table = best[
        ["split_label", "method_label", "cosine_mean", "pearson_mean", "top100_overlap_mean", "rmse_mean"]
    ].copy()
    best_table["method_label"] = best_table["method_label"].astype(str).str.replace("\n", " ", regex=False)
    header = "| " + " | ".join(best_table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(best_table.columns)) + " |"
    body = []
    for row in best_table.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    lines = [
        "# Figure 2 Benchmark Notes",
        "",
        "Figure labels use published model names wherever an official implementation was run.",
        "",
        "Method provenance:",
        "- Mean drug response: matched drug-level bulk delta baseline from this study. It is retained as a strong shrinkage comparator.",
        "- scGen: official scGen package run on the bulk pseudo-AnnData task.",
        "- CPA: official cpa-tools package run on the bulk pseudo-AnnData task.",
        "- CellOT: official CellOT ICNN/transport implementation run in PCA latent space for this high-dimensional bulk task.",
        "- trVAE: scArches trVAE conditional VAE run on the bulk pseudo-AnnData task.",
        "- scVIDR: official scVIDR VAE with latent response regression adapted to the paired bulk task.",
        "- PRISM-MM: prediction backbone from this study combining latent direction teachers with a shared-program decoder.",
        "",
        "All models use the same five folds for Random, Cell source, and Study evaluation. Figure 2 benchmarks the predictive backbone; the multimodal dictionary refinement and external program validation are evaluated separately.",
        "",
        "Main Figure 2 reports seven individual metrics and a hierarchical program-recovery score. The score balances three recovery domains: correlation recovery, responsive-gene recovery, and sign agreement. Correlation recovery is the mean of scaled cosine similarity, scaled Pearson correlation, and scaled Spearman correlation. Responsive-gene recovery is the mean of Top-50 and Top-100 overlap. Correlations are mapped from [-1, 1] to [0, 1]; overlap and sign agreement are already bounded in [0, 1].",
        "",
        "Magnitude fidelity is displayed as a separate calibration metric but is not mixed into the program-recovery score. RMSE and MAE are retained as Supplementary Figure S1. The drug-mean baseline can remain competitive on calibration and scale-error metrics because the evaluated folds permit the same drug to occur in training and testing contexts.",
        "",
        "Top methods by cosine:",
        "\n".join([header, sep] + body),
    ]
    (source_dir / "figure_2_benchmark_notes.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    setup_style()
    outdir = Path(args.outdir)
    fold = load_benchmark(Path(args.torch_dir), Path(args.official_dir))
    fold = add_direction_score(fold)
    summary = summarize(fold)
    figure_3(fold, outdir)
    export_panels(fold, Path(args.panel_dir), Path(args.supp_panel_dir))
    write_notes(fold, summary, outdir)
    print(f"Saved Figure 2 to {outdir / 'figure_2_model_benchmark.png'}")


if __name__ == "__main__":
    main()
