#!/usr/bin/env python3
"""Export Figure S7 for reviewer-requested drug-program sensitivity.

The figure visualizes whether the held-out bulk classifier result is driven by
specific drug-program contributions requested by the reviewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


SCENARIO_LABELS = {
    "baseline_selected_programs": "Baseline",
    "zero both-cohorts bortezomib x Program_1": "BTZ x P1",
    "zero both-cohorts bortezomib x Program_9": "BTZ x P9",
    "zero both-cohorts bortezomib x Program_1+Program_9": "BTZ x P1+P9",
    "zero both-cohorts dexamethasone x Program_1": "DEX x P1",
    "zero both-cohorts dexamethasone x Program_9": "DEX x P9",
    "zero both-cohorts dexamethasone x Program_1+Program_9": "DEX x P1+P9",
}
SCENARIO_ORDER = list(SCENARIO_LABELS)

COLORS = {
    "ink": "#000000",
    "grid": "#D0D5DD",
    "gray": "#98A2B3",
    "light_gray": "#E5E7EB",
    "blue": "#2F6690",
    "teal": "#0F766E",
    "red": "#B93A32",
    "orange": "#C47A2C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/tmp/prism_review1_test/figure4_exact_prediction_curve_sensitivity.csv",
    )
    parser.add_argument(
        "--out",
        default="/Users/mingkewu/Documents/vscode/supplementary_panels/figureS7_drug_removal_sensitivity.pdf",
    )
    parser.add_argument(
        "--four-layer",
        default="/tmp/prism_review1_test/program_direction_four_layer_summary.csv",
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


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["scenario"].isin(SCENARIO_ORDER)].copy()
    data["scenario"] = pd.Categorical(data["scenario"], SCENARIO_ORDER, ordered=True)
    data["label"] = data["scenario"].astype(str).map(SCENARIO_LABELS)
    return data.sort_values("scenario")


def load_four_layer(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    layer_cols = [
        ("discovery_mean", "Discovery\nbulk"),
        ("calib_scrna_mean", "Calibration\nscRNA"),
        ("heldout_mean", "Held-out\nbulk"),
        ("external_scrna_mean", "External\nscRNA"),
    ]
    rows = []
    for row in data.itertuples(index=False):
        program = str(row.program).replace("Program_", "P")
        for column, layer in layer_cols:
            value = float(getattr(row, column))
            rows.append(
                {
                    "program": program,
                    "layer": layer,
                    "mean_shift": value,
                    "direction": 1 if value > 0 else -1 if value < 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def auc_ap_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = np.arange(len(data))
    width = 0.34
    auc = ax.bar(
        x - width / 2,
        data["auc"],
        width=width,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        label="AUC",
    )
    ap = ax.bar(
        x + width / 2,
        data["ap"],
        width=width,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.8,
        label="AP",
    )
    ax.axhline(0.5, color=COLORS["grid"], lw=1.0, ls="--", zorder=0)
    for bars in [auc, ap]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.014,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )
    panel_label(ax, "A")
    ax.set_ylabel("Classifier performance")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], fontsize=7.4, rotation=28, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, alpha=0.75)
    if ax.legend_ is not None:
        ax.legend_.remove()
    handles = [
        plt.Line2D([0], [0], marker="s", lw=0, color=COLORS["blue"], markersize=7, label="AUC"),
        plt.Line2D([0], [0], marker="s", lw=0, color=COLORS["red"], markersize=7, label="AP"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.98),
        fontsize=8.2,
        handlelength=0.9,
        borderaxespad=0,
    )
    sns.despine(ax=ax)


def delta_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    base_auc = float(data.loc[data["scenario"].astype(str).eq("baseline_selected_programs"), "auc"].iloc[0])
    base_ap = float(data.loc[data["scenario"].astype(str).eq("baseline_selected_programs"), "ap"].iloc[0])
    use = data[~data["scenario"].astype(str).eq("baseline_selected_programs")].copy()
    use["delta_auc"] = use["auc"] - base_auc
    use["delta_ap"] = use["ap"] - base_ap
    y = np.arange(len(use))
    ax.axvline(0, color=COLORS["grid"], lw=1.0, zorder=0)
    ax.scatter(use["delta_auc"], y + 0.13, s=46, color=COLORS["blue"], edgecolor="white", linewidth=0.6, label="AUC")
    ax.scatter(use["delta_ap"], y - 0.13, s=46, color=COLORS["red"], edgecolor="white", linewidth=0.6, label="AP")
    for metric, offset, color in [("delta_auc", 0.13, COLORS["blue"]), ("delta_ap", -0.13, COLORS["red"])]:
        for value, ypos in zip(use[metric], y + offset):
            ax.plot([0, value], [ypos, ypos], color=color, lw=1.0, alpha=0.9)
            if abs(value) < 0.015:
                text_x = value - 0.018
                ha = "right"
            else:
                text_x = value + (-0.015 if value < 0 else 0.015)
                ha = "right" if value < 0 else "left"
            ax.text(
                text_x,
                ypos,
                f"{value:+.2f}",
                ha=ha,
                va="center",
                fontsize=7.4,
            )
    panel_label(ax, "B")
    ax.set_xlabel("Change from baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(use["label"], fontsize=8.2)
    xmin = min(-0.24, float(use[["delta_auc", "delta_ap"]].min().min()) - 0.04)
    xmax = max(0.08, float(use[["delta_auc", "delta_ap"]].max().max()) + 0.04)
    ax.set_xlim(xmin, xmax + 0.025)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, alpha=0.75)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)


def four_layer_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    programs = [f"P{i}" for i in range(1, 11)]
    layers = ["Discovery\nbulk", "Calibration\nscRNA", "Held-out\nbulk", "External\nscRNA"]
    raw = (
        data.pivot_table(index="layer", columns="program", values="mean_shift", aggfunc="first")
        .reindex(index=layers, columns=programs)
        .fillna(0.0)
    )
    plot = raw.div(raw.abs().max(axis=1).replace(0, np.nan), axis=0).fillna(0.0).T
    cmap = LinearSegmentedColormap.from_list("blue_white_red", [COLORS["blue"], "#FFFFFF", COLORS["red"]])
    sns.heatmap(
        plot,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.7,
        linecolor="white",
        cbar=True,
        cbar_kws={"shrink": 0.62, "pad": 0.025, "label": "scaled score"},
        ax=ax,
    )
    for program in ["P6", "P8", "P9"]:
        y = programs.index(program)
        ax.add_patch(plt.Rectangle((0, y), len(layers), 1, fill=False, edgecolor=COLORS["ink"], linewidth=1.2))
    panel_label(ax, "C")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(layers, rotation=0, fontsize=7.0)
    ax.set_yticklabels(programs, rotation=0, fontsize=7.3)
    for tick in ax.get_yticklabels():
        if tick.get_text() in {"P6", "P8", "P9"}:
            tick.set_fontweight("bold")
    ax.tick_params(axis="both", length=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6.6, length=2)
    cbar.set_label("scaled score", fontsize=7.0)


def main() -> None:
    args = parse_args()
    setup_style()
    data = load_data(Path(args.input))
    four_layer = load_four_layer(Path(args.four_layer))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.0, 4.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 0.92], wspace=0.50)
    auc_ap_panel(fig.add_subplot(gs[0, 0]), data)
    delta_panel(fig.add_subplot(gs[0, 1]), data)
    four_layer_panel(fig.add_subplot(gs[0, 2]), four_layer)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
