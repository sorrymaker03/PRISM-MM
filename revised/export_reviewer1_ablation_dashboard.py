#!/usr/bin/env python3
"""Export a reviewer-response ablation dashboard figure.

The figure summarizes whether the integrated PRISM-MM representation improves
bulk response recovery and cross-layer program reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("/tmp/prism_review1_test/revised_ablation")
DEFAULT_OUT = ROOT / "supplementary_panels"

MODEL_LABELS = {
    "prism_mm": "PRISM-MM",
    "bulk_only_v9": "Bulk-only\nv9",
    "scrna_only_svd": "scRNA-only\nSVD",
    "no_bulk_loss": "No bulk\nloss",
}
VALIDATION_MODEL_ORDER = ["prism_mm", "bulk_only_v9", "scrna_only_svd"]
FIT_MODEL_ORDER = ["prism_mm", "bulk_only_v9", "no_bulk_loss"]
PROGRAMS = [f"Program_{i}" for i in range(1, 11)]

COLORS = {
    "ink": "#000000",
    "grid": "#D0D5DD",
    "soft_grid": "#EAECF0",
    "blue": "#2F6690",
    "teal": "#0F766E",
    "orange": "#C47A2C",
    "red": "#B93A32",
    "gray": "#A3ADBB",
    "light_blue": "#D7E7F3",
    "light_teal": "#D9EEE9",
    "pale_gray": "#F2F4F7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUT))
    parser.add_argument("--basename", default="figureS_reviewer1_ablation_dashboard")
    return parser.parse_args()


def setup() -> None:
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")


def summary_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in detail.groupby("model", sort=False):
        direction = group["triple_direction_concordant"].astype(bool)
        significant = group["threeway_sample_significant_fdr_0_10"].astype(bool)
        rows.append(
            {
                "model": model,
                "triple_direction_concordant_n": int(direction.sum()),
                "triple_programs": ";".join(group.loc[direction, "program"].astype(str)),
                "threeway_sample_significant_n": int(significant.sum()),
                "threeway_sample_significant_programs": ";".join(group.loc[significant, "program"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def full_prism_detail() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "bulk_pre_sc/model_upgrade_v14_multimodal_dictionary/v14_variant_program_detail.csv")
    frame = frame[frame["variant"].eq("balanced")].copy()
    frame = frame.rename(
        columns={
            "heldout_bulk_sample_mean": "heldout_mean",
            "heldout_bulk_sample_fdr": "heldout_fdr",
            "scrna_oof_mean": "calibration_scrna_mean",
            "scrna_oof_q": "calibration_scrna_q",
        }
    )
    frame["model"] = "prism_mm"
    return frame[
        [
            "model",
            "program",
            "discovery_mean",
            "discovery_fdr",
            "heldout_mean",
            "heldout_fdr",
            "calibration_scrna_mean",
            "calibration_scrna_q",
            "triple_direction_concordant",
            "threeway_sample_significant_fdr_0_10",
        ]
    ]


def bulk_only_detail() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "bulk_pre_sc/model_upgrade_v10_discovery_core/v9_v10_external_validation_program_detail.csv")
    frame = frame[frame["variant"].eq("v9_baseline")].copy()
    frame = frame.rename(
        columns={
            "heldout_bulk_sample_mean": "heldout_mean",
            "heldout_bulk_sample_fdr": "heldout_fdr",
            "scrna_mean": "calibration_scrna_mean",
            "scrna_q": "calibration_scrna_q",
            "sample_scrna_significant_both_fdr_0_10": "threeway_sample_significant_fdr_0_10",
        }
    )
    frame["model"] = "bulk_only_v9"
    frame["triple_direction_concordant"] = frame["triple_direction_concordant"].astype(bool)
    frame["threeway_sample_significant_fdr_0_10"] = frame["threeway_sample_significant_fdr_0_10"].astype(bool)
    return frame[
        [
            "model",
            "program",
            "discovery_mean",
            "heldout_mean",
            "heldout_fdr",
            "calibration_scrna_mean",
            "calibration_scrna_q",
            "triple_direction_concordant",
            "threeway_sample_significant_fdr_0_10",
        ]
    ].assign(discovery_fdr=np.nan)


def scrna_only_detail(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "scrna_only_svd" / "three_layer_program_detail.csv"
    frame = pd.read_csv(path).copy()
    frame["model"] = "scrna_only_svd"
    return frame


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_fit = pd.read_csv(ROOT / "bulk_pre_sc/model_upgrade_v14_multimodal_dictionary/balanced/our_multimodal_dictionary_v14_fit_metrics.csv")
    full_fit["model"] = "prism_mm"
    v9_fit = pd.read_csv(ROOT / "bulk_pre_sc/model_upgrade_v9_final/our_anchored_sparse_attention_v9_fit_metrics.csv")
    v9_fit["model"] = "bulk_only_v9"
    smoke_fit = pd.read_csv(input_dir / "fit_metrics_summary.csv")
    no_bulk_fit = smoke_fit[smoke_fit["model"].eq("no_bulk_loss")].copy()
    fit = pd.concat([full_fit, v9_fit, no_bulk_fit], ignore_index=True, sort=False)
    fit["model"] = pd.Categorical(fit["model"], categories=FIT_MODEL_ORDER, ordered=True)

    detail = pd.concat([full_prism_detail(), bulk_only_detail(), scrna_only_detail(input_dir)], ignore_index=True, sort=False)
    detail["model"] = pd.Categorical(detail["model"], categories=VALIDATION_MODEL_ORDER, ordered=True)
    detail["program"] = pd.Categorical(detail["program"], categories=PROGRAMS, ordered=True)
    detail = detail.sort_values(["model", "program"])
    summary = summary_from_detail(detail)
    summary["model"] = pd.Categorical(summary["model"], categories=VALIDATION_MODEL_ORDER, ordered=True)
    return fit.sort_values("model"), summary.sort_values("model"), detail


def plot_response_metrics(ax: plt.Axes, fit: pd.DataFrame) -> None:
    metrics = [
        ("cosine", "Cosine"),
        ("pearson", "Pearson"),
        ("top100_sign_agreement", "Top100 sign"),
    ]
    plot = fit.melt(id_vars="model", value_vars=[m[0] for m in metrics], var_name="metric", value_name="value")
    plot["metric"] = plot["metric"].map(dict(metrics))
    plot["model_label"] = plot["model"].astype(str).map(MODEL_LABELS)
    metric_order = [m[1] for m in metrics]
    model_order = [MODEL_LABELS[m] for m in FIT_MODEL_ORDER if m in set(fit["model"].astype(str))]
    sns.barplot(
        data=plot,
        y="model_label",
        x="value",
        hue="metric",
        hue_order=metric_order,
        order=model_order,
        palette=[COLORS["blue"], COLORS["teal"], COLORS["orange"]],
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
    )
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("Bulk response recovery")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLORS["soft_grid"], lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5, title=None)
    ax.tick_params(axis="both", labelsize=8.3)
    sns.despine(ax=ax, left=True)
    panel_label(ax, "A")


def plot_program_counts(ax: plt.Axes, summary: pd.DataFrame) -> None:
    plot = summary.copy()
    plot["model_label"] = plot["model"].astype(str).map(MODEL_LABELS)
    plot = plot.dropna(subset=["model_label"])
    model_order = [MODEL_LABELS[m] for m in VALIDATION_MODEL_ORDER if m in set(plot["model"].astype(str))]
    long = plot.melt(
        id_vars=["model", "model_label"],
        value_vars=["triple_direction_concordant_n", "threeway_sample_significant_n"],
        var_name="test",
        value_name="n_programs",
    )
    long["test"] = long["test"].map(
        {
            "triple_direction_concordant_n": "Direction\nconcordant",
            "threeway_sample_significant_n": "Direction +\nsignificant",
        }
    )
    sns.barplot(
        data=long,
        y="model_label",
        x="n_programs",
        hue="test",
        order=model_order,
        palette=[COLORS["light_blue"], COLORS["blue"]],
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8.0, color=COLORS["ink"])
    ax.set_xlim(0, 10.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Programs recovered across layers")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLORS["soft_grid"], lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5, title=None)
    ax.tick_params(axis="both", labelsize=8.3)
    sns.despine(ax=ax, left=True)
    panel_label(ax, "B")


def plot_program_status(ax: plt.Axes, detail: pd.DataFrame) -> None:
    status = detail.pivot_table(
        index="model",
        columns="program",
        values="threeway_sample_significant_fdr_0_10",
        aggfunc="max",
        observed=False,
    )
    concord = detail.pivot_table(
        index="model",
        columns="program",
        values="triple_direction_concordant",
        aggfunc="max",
        observed=False,
    )
    status = status.reindex(VALIDATION_MODEL_ORDER).dropna(how="all")
    concord = concord.reindex(status.index)
    code = pd.DataFrame(0, index=status.index, columns=PROGRAMS, dtype=float)
    code = code.where(~concord.reindex(columns=PROGRAMS).fillna(False), 1)
    code = code.where(~status.reindex(columns=PROGRAMS).fillna(False), 2)
    cmap = ListedColormap([COLORS["pale_gray"], COLORS["light_blue"], COLORS["blue"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    sns.heatmap(
        code,
        cmap=cmap,
        norm=norm,
        linewidths=0.7,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([MODEL_LABELS.get(str(m), str(m)) for m in code.index], rotation=0)
    ax.set_xticklabels([p.replace("Program_", "P") for p in PROGRAMS], rotation=0)
    ax.tick_params(axis="both", labelsize=8.2, length=0)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", lw=0, markersize=8, markerfacecolor=COLORS["pale_gray"], markeredgecolor="white", label="Not concordant"),
        plt.Line2D([0], [0], marker="s", lw=0, markersize=8, markerfacecolor=COLORS["light_blue"], markeredgecolor="white", label="Direction concordant"),
        plt.Line2D([0], [0], marker="s", lw=0, markersize=8, markerfacecolor=COLORS["blue"], markeredgecolor="white", label="Direction + significant"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.4)
    panel_label(ax, "C")


def plot_shift_triads(ax: plt.Axes, detail: pd.DataFrame) -> None:
    use = detail[detail["model"].astype(str).eq("prism_mm")].copy()
    use = use[use["program"].isin(["Program_6", "Program_8", "Program_9"])]
    use = use.sort_values("program")
    rows = []
    for row in use.itertuples(index=False):
        rows.extend(
            [
                {"program": row.program, "layer": "Discovery\nbulk", "delta": row.discovery_mean},
                {"program": row.program, "layer": "Held-out\nbulk", "delta": row.heldout_mean},
                {"program": row.program, "layer": "Calibration\nscRNA", "delta": row.calibration_scrna_mean},
            ]
        )
    plot = pd.DataFrame(rows)
    plot["program_label"] = plot["program"].str.replace("Program_", "P", regex=False)
    layers = ["Discovery\nbulk", "Held-out\nbulk", "Calibration\nscRNA"]
    x = np.arange(len(use))
    offsets = {"Discovery\nbulk": -0.23, "Held-out\nbulk": 0.0, "Calibration\nscRNA": 0.23}
    colors = {"Discovery\nbulk": COLORS["gray"], "Held-out\nbulk": COLORS["blue"], "Calibration\nscRNA": COLORS["teal"]}
    ax.axhline(0, color=COLORS["grid"], lw=0.9, zorder=0)
    for layer in layers:
        sub = plot[plot["layer"].eq(layer)]
        vals = sub["delta"].to_numpy(dtype=float)
        pos = x + offsets[layer]
        ax.scatter(pos, vals, s=38, color=colors[layer], edgecolor="white", linewidth=0.55, label=layer, zorder=3)
        for px, val in zip(pos, vals):
            ax.plot([px, px], [0, val], color=colors[layer], lw=0.85, alpha=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(use["program"].astype(str).str.replace("Program_", "P", regex=False))
    ax.set_ylabel("Drug - Ctrl score shift")
    ax.set_xlabel("")
    ax.grid(axis="y", color=COLORS["soft_grid"], lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=7.4, ncol=1)
    ax.tick_params(axis="both", labelsize=8.2)
    sns.despine(ax=ax)
    panel_label(ax, "D")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup()
    fit, summary, detail = load_inputs(input_dir)

    fig = plt.figure(figsize=(13.2, 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.18], wspace=0.34, hspace=0.43)
    plot_response_metrics(fig.add_subplot(gs[0, 0]), fit)
    plot_program_counts(fig.add_subplot(gs[0, 1]), summary)
    plot_program_status(fig.add_subplot(gs[1, 0]), detail)
    plot_shift_triads(fig.add_subplot(gs[1, 1]), detail)
    pdf = outdir / f"{args.basename}.pdf"
    png = outdir / f"{args.basename}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
