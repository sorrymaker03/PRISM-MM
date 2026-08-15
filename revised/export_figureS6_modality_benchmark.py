#!/usr/bin/env python3
"""Export Figure S6: PRISM-MM versus bulk-only and scRNA-only controls.

The figure uses the same benchmark metrics as Figure 2 but shows only the
current PRISM-MM model and two reviewer-requested modality controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


METRIC_SPECS = [
    ("A", "cosine", "Cosine similarity", "Perturbation-direction recovery", None),
    ("B", "pearson", "Pearson correlation", "Gene-wise response correlation", None),
    ("C", "spearman", "Spearman correlation", "Gene-rank response correlation", None),
    ("D", "top50_overlap", "Top-50 overlap", "Top response-gene recovery", None),
    ("E", "top100_overlap", "Top-100 overlap", "Strong response-gene recovery", None),
    ("F", "top100_sign_agreement", "Sign agreement", "Direction of top response genes", (0.0, 1.0)),
    ("G", "magnitude_fidelity", "Magnitude fidelity", "Effect-size calibration", (0.0, 0.78)),
]

MODEL_ORDER = ["OUR_hybrid_teacher_program_v3", "bulk_only_shared_program", "scrna_only_dictionary_ridge"]
MODEL_LABELS = {
    "OUR_hybrid_teacher_program_v3": "PRISM-MM",
    "bulk_only_shared_program": "Bulk-only",
    "scrna_only_dictionary_ridge": "scRNA-only",
}
SPLIT_ORDER = ["Cell source", "Study", "Random"]

COLORS = {
    "PRISM-MM": "#B42318",
    "Bulk-only": "#2B6CB0",
    "scRNA-only": "#0F766E",
    "grid": "#D0D5DD",
    "ink": "#000000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="/tmp/prism_review1_test/figure2_modality_benchmark/combined_figure2_modality_fold_metrics.csv",
    )
    parser.add_argument(
        "--out",
        default="/Users/mingkewu/Documents/vscode/supplementary_panels/figureS6_modality_benchmark.pdf",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_metrics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["model"].astype(str).isin(MODEL_ORDER)].copy()
    data["method_label"] = data["model"].astype(str).map(MODEL_LABELS)
    data["method_label"] = pd.Categorical(data["method_label"], [MODEL_LABELS[m] for m in MODEL_ORDER], ordered=True)
    data["split_label"] = pd.Categorical(data["split_label"], SPLIT_ORDER, ordered=True)
    return data.sort_values(["split_label", "method_label", "fold"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )


def metric_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel: str,
    metric: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None,
) -> None:
    sns.barplot(
        data=data,
        x="split_label",
        y=metric,
        hue="method_label",
        order=SPLIT_ORDER,
        hue_order=[MODEL_LABELS[m] for m in MODEL_ORDER],
        palette=COLORS,
        errorbar="sd",
        err_kws={"linewidth": 0.8},
        capsize=0.10,
        ax=ax,
    )
    panel_label(ax, panel)
    ax.set_title(title, loc="left", fontsize=9.6, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=8.8)
    ax.tick_params(axis="x", labelsize=8.0, rotation=0)
    ax.tick_params(axis="y", labelsize=8.0)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, alpha=0.75)
    if metric == "top100_sign_agreement":
        ax.axhline(0.5, color=COLORS["grid"], lw=0.9, ls="--", zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)


def overall_score_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    score = (
        data.groupby("method_label", observed=True)["direction_score"]
        .agg(["mean", "std"])
        .reindex([MODEL_LABELS[m] for m in MODEL_ORDER])
        .reset_index()
    )
    y = range(len(score))
    labels = score["method_label"].astype(str).tolist()
    colors = [COLORS[label] for label in labels]
    bars = ax.barh(
        y,
        score["mean"],
        xerr=score["std"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 0.8, "capsize": 3},
    )
    for bar, value, err in zip(bars, score["mean"], score["std"]):
        ax.text(value + err + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.0)
    panel_label(ax, "H")
    ax.set_title("Overall program-recovery score", loc="left", fontsize=9.6, fontweight="bold", pad=6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel("Program-recovery score", fontsize=8.8)
    ax.set_ylabel("")
    ax.set_xlim(0.20, 0.64)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, alpha=0.75)
    ax.tick_params(axis="x", labelsize=8.0)
    sns.despine(ax=ax)


def main() -> None:
    args = parse_args()
    setup_style()
    data = load_metrics(Path(args.metrics))

    fig = plt.figure(figsize=(13.2, 7.1))
    gs = fig.add_gridspec(2, 4, wspace=0.46, hspace=0.58)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]
    for ax, spec in zip(axes[:7], METRIC_SPECS):
        metric_panel(ax, data, *spec)
    overall_score_panel(axes[7], data)

    handles = [
        Patch(facecolor=COLORS[MODEL_LABELS[m]], edgecolor="white", label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        fontsize=9.0,
        handlelength=1.3,
        columnspacing=1.8,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
