#!/usr/bin/env python3
"""Export Figure S8: K-means clustering baseline for Figure 6 scRNA data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns


CLUSTER_ORDER = [f"Cluster {i}" for i in range(1, 11)]
DATASET_ORDER = ["GSE161195", "GSE161801"]
PHASE_ORDER = ["Ctrl", "Drug"]

CLUSTER_COLORS = {
    "Cluster 1": "#2F6690",
    "Cluster 2": "#B93A32",
    "Cluster 3": "#0F766E",
    "Cluster 4": "#7A5AF8",
    "Cluster 5": "#CBD2DC",
    "Cluster 6": "#C47A2C",
    "Cluster 7": "#4D7C3A",
    "Cluster 8": "#C11574",
    "Cluster 9": "#7C6D5F",
    "Cluster 10": "#D6A21F",
}
PHASE_COLORS = {"Ctrl": "#2F6690", "Drug": "#B93A32"}
PALETTE = {"ink": "#000000", "grid": "#D0D5DD", "gray": "#667085"}
DELTA_CMAP = sns.diverging_palette(240, 12, s=85, l=48, as_cmap=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        default="/tmp/prism_review1_test/scrna_kmeans_baseline/figureS8_scrna_kmeans_cells.csv",
    )
    parser.add_argument(
        "--out",
        default="/Users/mingkewu/Documents/vscode/supplementary_panels/figureS8_scrna_kmeans_cluster_baseline.pdf",
    )
    parser.add_argument(
        "--source-outdir",
        default="/tmp/prism_review1_test/scrna_kmeans_baseline",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.5,
        fontweight="bold",
    )


def load_cells(path: Path) -> pd.DataFrame:
    cells = pd.read_csv(path)
    cells = cells[cells["dataset"].isin(DATASET_ORDER)].copy()
    cells["dataset"] = pd.Categorical(cells["dataset"], DATASET_ORDER, ordered=True)
    cells["phase_display"] = pd.Categorical(cells["phase_display"], PHASE_ORDER, ordered=True)
    cells["kmeans_cluster"] = pd.Categorical(cells["kmeans_cluster"], CLUSTER_ORDER, ordered=True)
    return cells.sort_values(["dataset", "kmeans_cluster", "phase_display"])


def composition(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = (
        cells.groupby(["dataset", "phase_display", "kmeans_cluster"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    comp["phase_cells"] = comp.groupby(["dataset", "phase_display"], observed=True)["n_cells"].transform("sum")
    comp["fraction"] = comp["n_cells"] / comp["phase_cells"]
    comp["percent"] = 100 * comp["fraction"]

    stats_rows = []
    for dataset, sub in comp.groupby("dataset", observed=True):
        table = sub.pivot(index="phase_display", columns="kmeans_cluster", values="n_cells").loc[PHASE_ORDER, CLUSTER_ORDER]
        chi2, p_value, _, _ = chi2_contingency(table.to_numpy(dtype=float))
        deltas = (
            sub.pivot(index="kmeans_cluster", columns="phase_display", values="fraction")
            .reindex(CLUSTER_ORDER)
            .assign(drug_minus_ctrl=lambda x: x["Drug"] - x["Ctrl"])
        )
        stats_rows.append(
            {
                "dataset": dataset,
                "chi_square": chi2,
                "p_value": p_value,
                "max_abs_fraction_delta": float(deltas["drug_minus_ctrl"].abs().max()),
                "n_clusters_abs_delta_ge_2pct": int((deltas["drug_minus_ctrl"].abs() >= 0.02).sum()),
                "n_clusters_abs_delta_ge_5pct": int((deltas["drug_minus_ctrl"].abs() >= 0.05).sum()),
            }
        )
    return comp, pd.DataFrame(stats_rows)


def draw_umap(ax: plt.Axes, cells: pd.DataFrame, dataset: str, label: str) -> None:
    sub = cells[cells["dataset"].astype(str).eq(dataset)].copy()
    rng = np.random.default_rng(20260806)
    order = rng.permutation(len(sub))
    sub = sub.iloc[order]
    for cluster in CLUSTER_ORDER:
        part = sub[sub["kmeans_cluster"].astype(str).eq(cluster)]
        ax.scatter(
            part["UMAP1"],
            part["UMAP2"],
            s=1.05,
            c=CLUSTER_COLORS[cluster],
            alpha=0.62,
            linewidth=0,
            rasterized=True,
        )
    panel_label(ax, label)
    ax.set_title(dataset, loc="left", fontsize=11.0, fontweight="bold", pad=5)
    ax.set_xlabel("UMAP1", fontsize=9.2)
    ax.set_ylabel("UMAP2", fontsize=9.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    sns.despine(ax=ax, left=True, bottom=True)


def draw_composition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    comp = comp.copy()
    comp["cluster_label"] = comp["kmeans_cluster"].astype(str).str.replace("Cluster ", "C", regex=False)
    sns.barplot(
        data=comp,
        x="cluster_label",
        y="percent",
        hue="phase_display",
        order=[c.replace("Cluster ", "C") for c in CLUSTER_ORDER],
        hue_order=PHASE_ORDER,
        palette=PHASE_COLORS,
        errorbar=None,
        ax=ax,
    )
    panel_label(ax, "C")
    ax.set_xlabel("K-means cluster", fontsize=9.2)
    ax.set_ylabel("Cell fraction (%)", fontsize=9.2)
    ax.tick_params(axis="x", labelsize=8.4)
    ax.tick_params(axis="y", labelsize=8.0)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.55, alpha=0.75)
    ax.set_ylim(0, max(68, comp["percent"].max() * 1.12))
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)


def draw_pooled_composition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    pooled = (
        comp.groupby(["phase_display", "kmeans_cluster"], observed=True)["fraction"]
        .mean()
        .reset_index()
    )
    pooled["percent"] = 100 * pooled["fraction"]
    pooled["cluster_label"] = pooled["kmeans_cluster"].astype(str).str.replace("Cluster ", "C", regex=False)
    sns.barplot(
        data=pooled,
        x="cluster_label",
        y="percent",
        hue="phase_display",
        order=[c.replace("Cluster ", "C") for c in CLUSTER_ORDER],
        hue_order=PHASE_ORDER,
        palette=PHASE_COLORS,
        errorbar=None,
        ax=ax,
    )
    panel_label(ax, "C")
    ax.set_title("Pooled scRNA datasets", loc="left", fontsize=10.4, fontweight="bold", pad=5)
    ax.set_xlabel("K-means cluster", fontsize=9.0)
    ax.set_ylabel("Cell fraction (%)", fontsize=9.0)
    ax.set_ylim(0, max(45, pooled["percent"].max() * 1.12))
    ax.tick_params(axis="x", labelsize=8.2)
    ax.tick_params(axis="y", labelsize=7.8)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.55, alpha=0.75)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)
    handles = [
        Patch(facecolor=PHASE_COLORS["Ctrl"], edgecolor="white", label="Ctrl"),
        Patch(facecolor=PHASE_COLORS["Drug"], edgecolor="white", label="Drug"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        fontsize=8.4,
        handlelength=1.1,
    )

    return pooled


def draw_bottom(fig: plt.Figure, gs_cell, comp: pd.DataFrame) -> None:
    ax = fig.add_subplot(gs_cell)
    draw_pooled_composition(ax, comp)


def cluster_labels() -> list[str]:
    return [cluster.replace("Cluster ", "C") for cluster in CLUSTER_ORDER]


def add_colorbar_axis(ax: plt.Axes, size: str = "2.2%", pad: float = 0.06) -> plt.Axes:
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size=size, pad=pad)


def draw_matrix(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    row_col: str,
    value_col: str,
    label: str,
    title: str,
    row_order: list[str],
    vlim: float,
    cbar_label: str,
    ytick_size: float = 6.5,
    separator_after: list[int] | None = None,
) -> None:
    mat = (
        df.assign(cluster=lambda x: pd.Categorical(x["cluster"], CLUSTER_ORDER, ordered=True))
        .pivot_table(index=row_col, columns="cluster", values=value_col, aggfunc="mean", observed=False)
        .reindex(index=row_order, columns=CLUSTER_ORDER)
    )
    mat.columns = cluster_labels()
    cax = add_colorbar_axis(ax)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=DELTA_CMAP,
        vmin=-vlim,
        vmax=vlim,
        center=0,
        linewidths=0.28,
        linecolor="white",
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": cbar_label},
    )
    panel_label(ax, label)
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", pad=5)
    ax.set_xlabel("K-means cluster", fontsize=8.7)
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels(mat.columns, rotation=0)
    ax.set_yticks(np.arange(len(mat.index)) + 0.5)
    ax.set_yticklabels(mat.index, rotation=0)
    ax.tick_params(axis="x", labelsize=7.5, rotation=0)
    ax.tick_params(axis="y", labelsize=ytick_size, rotation=0)
    cax.tick_params(labelsize=6.8, length=2)
    cax.yaxis.label.set_size(7.2)
    if separator_after:
        for pos in separator_after:
            ax.axhline(pos, color="white", lw=1.7)


def load_downstream_tables(source_outdir: Path) -> dict[str, pd.DataFrame]:
    return {
        "D": pd.read_csv(source_outdir / "figureS8_D_marker_gene_delta.csv"),
        "E": pd.read_csv(source_outdir / "figureS8_E_tf_activity_delta.csv"),
        "F": pd.read_csv(source_outdir / "figureS8_F_gsea_pooled.csv"),
        "G": pd.read_csv(source_outdir / "figureS8_G_progeny_activity_delta.csv"),
        "H": pd.read_csv(source_outdir / "figureS8_H_ligand_receptor_delta.csv"),
    }


def draw_marker_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    groups = [
        ("Program 6 markers", ["TMEM156", "CTSF", "IDUA", "UGT8", "LAMP3", "PRDM1", "CD38"]),
        ("Program 8 markers", ["CD44", "RGS2", "TYMP", "CD74", "HLA-DQB1", "STAT1", "BIRC3", "NFKBIA"]),
        ("Program 9 markers", ["DUSP1", "CDKN1C", "KLRD1", "FOS", "JUNB", "ZFP36", "IL32"]),
    ]
    row_order = []
    separators = []
    for _, genes in groups:
        row_order.extend([gene for gene in genes if gene in set(df["gene"])])
        separators.append(len(row_order))
    draw_matrix(
        ax,
        df,
        row_col="gene",
        value_col="delta",
        label="D",
        title="Marker-gene expression change",
        row_order=row_order,
        vlim=0.65,
        cbar_label="Drug - Ctrl",
        ytick_size=6.7,
        separator_after=separators[:-1],
    )


def draw_tf_activity_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    tf_order = (
        df.assign(max_abs=lambda x: x.groupby("TF")["activity_delta"].transform(lambda v: v.abs().max()))
        .sort_values(["max_abs", "TF"], ascending=[False, True])["TF"]
        .drop_duplicates()
        .tolist()
    )
    draw_matrix(
        ax,
        df,
        row_col="TF",
        value_col="activity_delta",
        label="E",
        title="TF activity change",
        row_order=tf_order,
        vlim=2.2,
        cbar_label="Drug - Ctrl",
        ytick_size=6.8,
    )


def draw_gsea_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    pathway_order = [
        "Plasma-cell identity",
        "Protein homeostasis",
        "Secretory/UPR",
        "Oxidative phosphorylation",
        "NF-kB survival",
        "Interferon response",
        "Antigen presentation",
        "Adhesion/migration",
        "Stress/dormancy",
    ]
    pathway_order = [p for p in pathway_order if p in set(df["pathway_label"])]
    draw_matrix(
        ax,
        df,
        row_col="pathway_label",
        value_col="NES",
        label="F",
        title="Pathway enrichment change",
        row_order=pathway_order,
        vlim=1.6,
        cbar_label="NES",
        ytick_size=6.8,
    )


def draw_progeny_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    pathway_order = ["TNFa", "NF-kB", "JAK-STAT", "p53", "Hypoxia", "TGFb"]
    pathway_order = [p for p in pathway_order if p in set(df["pathway_label"])]
    draw_matrix(
        ax,
        df,
        row_col="pathway_label",
        value_col="activity_delta",
        label="G",
        title="Tumor pathway activity change",
        row_order=pathway_order,
        vlim=3.0,
        cbar_label="Drug - Ctrl",
        ytick_size=6.9,
    )


def draw_lr_delta(ax: plt.Axes, df: pd.DataFrame) -> None:
    pair_order = [
        "MIF-CD74",
        "MIF-CXCR4",
        "VIM-CD44",
        "TIMP1-CD63",
        "B2M-HLA-F",
        "B2M-LILRB1",
        "B2M-KLRD1",
        "HLA-B-KLRD1",
    ]
    pair_order = [p for p in pair_order if p in set(df["pair_label"])]
    draw_matrix(
        ax,
        df,
        row_col="pair_label",
        value_col="plot_delta",
        label="H",
        title="Ligand-receptor potential change",
        row_order=pair_order,
        vlim=1.1,
        cbar_label="Drug - Ctrl",
        ytick_size=6.9,
    )


def main() -> None:
    args = parse_args()
    setup_style()
    cells = load_cells(Path(args.cells))
    comp, stats = composition(cells)

    source_outdir = Path(args.source_outdir)
    source_outdir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(source_outdir / "figureS8_scrna_kmeans_composition.csv", index=False)
    stats.to_csv(source_outdir / "figureS8_scrna_kmeans_stats.csv", index=False)
    downstream = load_downstream_tables(source_outdir)

    fig = plt.figure(figsize=(13.2, 17.2))
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.0, 0.58, 1.28, 0.92, 0.9],
        wspace=0.28,
        hspace=0.52,
    )
    draw_umap(fig.add_subplot(gs[0, 0]), cells, "GSE161195", "A")
    draw_umap(fig.add_subplot(gs[0, 1]), cells, "GSE161801", "B")
    draw_bottom(fig, gs[1, :], comp)
    draw_marker_delta(fig.add_subplot(gs[2, :]), downstream["D"])
    draw_tf_activity_delta(fig.add_subplot(gs[3, 0]), downstream["E"])
    draw_gsea_delta(fig.add_subplot(gs[3, 1]), downstream["F"])
    draw_progeny_delta(fig.add_subplot(gs[4, 0]), downstream["G"])
    draw_lr_delta(fig.add_subplot(gs[4, 1]), downstream["H"])

    handles = [
        Patch(facecolor=CLUSTER_COLORS[cluster], edgecolor="white", label=cluster.replace("Cluster ", "C"))
        for cluster in CLUSTER_ORDER
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=10,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.005),
        fontsize=8.1,
        handlelength=1.0,
        columnspacing=0.85,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
