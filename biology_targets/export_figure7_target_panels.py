from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "biology_interpretation" / "gene_target_trial"
PANEL_DIR = ROOT / "main_figure_panels"
SOURCE_DIR = ROOT / "main_figure_source_data"
PRISM_PANEL_DIR = ROOT.parent / "PRISM-MM" / "figures" / "main_figure_panels"

PROGRAM_COLORS = {
    "Program_6": "#B93A32",
    "Program_8": "#0F766E",
    "Program_9": "#7A5AF8",
}
PROGRAM_LABELS = {
    "Program_6": "Program 6",
    "Program_8": "Program 8",
    "Program_9": "Program 9",
}
PALETTE = {
    "ink": "#111111",
    "muted": "#4B5563",
    "grid": "#DDE3EA",
    "blue": "#2F6690",
    "red": "#B93A32",
}
SELECTED = {
    "Program_6": ["TMEM156", "CTSF", "IDUA", "UGT8", "LAMP3"],
    "Program_8": ["RGS2", "TYMP", "CD44"],
    "Program_9": ["KLRD1", "CDKN1C", "DUSP1"],
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.edgecolor": "#9AA6B2",
            "axes.linewidth": 0.6,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    PRISM_PANEL_DIR.mkdir(parents=True, exist_ok=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["ink"],
    )


def save_panel(fig: plt.Figure, filename: str) -> None:
    for outdir in [PANEL_DIR, PRISM_PANEL_DIR]:
        fig.savefig(outdir / filename, bbox_inches="tight")
    plt.close(fig)


def clean_program(value: str) -> str:
    value = str(value)
    return value if value.startswith("Program_") else value.replace(" ", "_")


def load_priority() -> pd.DataFrame:
    df = pd.read_csv(TRIAL / "trial_gene_target_prioritization.csv")
    df["program"] = df["program"].map(clean_program)
    return df


def add_program_legend(ax: plt.Axes, x: float = 0.02, y: float = 1.02) -> None:
    for i, (program, color) in enumerate(PROGRAM_COLORS.items()):
        ax.scatter(x + i * 0.23, y, s=42, color=color, transform=ax.transAxes, clip_on=False)
        ax.text(
            x + i * 0.23 + 0.025,
            y,
            PROGRAM_LABELS[program],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.5,
            color=PALETTE["muted"],
        )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str = "#AAB4BF",
    fontsize: float = 8.5,
    weight: str = "normal",
) -> None:
    box = mpl.patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.75,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["ink"],
        fontweight=weight,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#64748B", shrinkA=3, shrinkB=3),
    )


def panel_a_nomination_scheme() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_label(ax, "A")
    rounded_box(
        ax,
        (0.035, 0.47),
        0.16,
        0.25,
        "PRISM-MM\nsigned programs",
        "#F8FAFC",
        edge="#CBD5E1",
        weight="bold",
    )
    for j, program in enumerate(["Program_6", "Program_8", "Program_9"]):
        ax.add_patch(
            mpl.patches.Circle(
                (0.078 + j * 0.038, 0.34),
                0.012,
                color=PROGRAM_COLORS[program],
                alpha=0.95,
            )
        )
    ax.text(0.116, 0.27, "P6 down; P8/P9 up", ha="center", va="center", fontsize=7.2, color=PALETTE["muted"])

    rounded_box(
        ax,
        (0.275, 0.47),
        0.18,
        0.25,
        "Gene evidence\nintegration",
        "#EEF6FF",
        edge="#93B7D8",
        weight="bold",
    )
    factors = [
        ("model weight", 0.258, 0.33),
        ("marker", 0.386, 0.33),
        ("replication", 0.258, 0.22),
        ("metadata", 0.386, 0.22),
    ]
    for label, x, y in factors:
        rounded_box(ax, (x, y), 0.112, 0.065, label, "#FFFFFF", edge="#C7D7E8", fontsize=6.7)

    rounded_box(
        ax,
        (0.530, 0.47),
        0.17,
        0.25,
        "Virtual\nperturbation",
        "#F1F8F5",
        edge="#9AC7B7",
        weight="bold",
    )
    ax.text(
        0.615,
        0.32,
        "network propagation\nrestore score",
        ha="center",
        va="center",
        fontsize=6.9,
        color=PALETTE["muted"],
    )

    rounded_box(
        ax,
        (0.775, 0.47),
        0.19,
        0.25,
        "Prioritized\ncandidate genes",
        "#F7F3FF",
        edge="#B8A8F8",
        weight="bold",
    )
    ax.text(
        0.87,
        0.32,
        r"$S_g = S_{evidence} + S_{recovery}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=PALETTE["ink"],
    )
    arrow(ax, (0.195, 0.56), (0.275, 0.56))
    arrow(ax, (0.455, 0.56), (0.530, 0.56))
    arrow(ax, (0.700, 0.56), (0.775, 0.56))
    save_panel(fig, "figure7_A_target_nomination_scheme.pdf")


def panel_b_priority() -> None:
    df = load_priority()
    rows = []
    for program, genes in SELECTED.items():
        sub = df[(df["program"] == program) & (df["gene"].isin(genes))].copy()
        rows.append(sub)
    plot = pd.concat(rows, ignore_index=True)
    plot["label"] = plot["gene"] + "  " + plot["program"].map(PROGRAM_LABELS)
    plot["score"] = plot["score"].fillna(0)
    plot["rank_order"] = plot["program"].map({"Program_6": 0, "Program_8": 1, "Program_9": 2}) * 100 - plot["score"]
    plot = plot.sort_values("rank_order", ascending=True).reset_index(drop=True)
    plot.to_csv(SOURCE_DIR / "figure7_B_target_prioritization.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.9, 4.1))
    y = np.arange(len(plot))
    colors = [PROGRAM_COLORS[p] for p in plot["program"]]
    ax.barh(y, plot["score"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, plot["gene"])
    ax.invert_yaxis()
    ax.set_xlabel("Integrated target-prioritization score")
    ax.set_ylabel("")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    xmax = max(7.8, float(plot["score"].max()) + 0.6)
    ax.set_xlim(0, xmax)
    add_program_legend(ax)
    panel_label(ax, "B")
    ax.spines[["top", "right"]].set_visible(False)
    save_panel(fig, "figure7_B_target_prioritization.pdf")


def panel_c_virtual_perturbation() -> None:
    df = pd.read_csv(TRIAL / "trial_network_propagation_virtual_perturbation_summary.csv")
    df["program"] = df["program"].map(clean_program)
    rows = []
    for program, genes in SELECTED.items():
        action = "activate" if program == "Program_6" else "knockdown"
        sub = df[(df["program"] == program) & (df["target"].isin(genes)) & (df["action"] == action)].copy()
        rows.append(sub)
    plot = pd.concat(rows, ignore_index=True)
    plot["program_label"] = plot["program"].map(PROGRAM_LABELS)
    plot["action_label"] = plot["action"].map({"activate": "activation", "knockdown": "knockdown"})
    plot["display"] = plot["target"] + " (" + plot["action_label"] + ")"
    plot["rank_order"] = plot["program"].map({"Program_6": 0, "Program_8": 1, "Program_9": 2}) * 100 - plot["recovery_score"]
    plot = plot.sort_values("rank_order").reset_index(drop=True)
    plot.to_csv(SOURCE_DIR / "figure7_C_virtual_perturbation.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 4.1))
    y = np.arange(len(plot))
    colors = [PROGRAM_COLORS[p] for p in plot["program"]]
    ax.barh(y, plot["recovery_score"], color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#AAB4BF", linewidth=0.8)
    ax.set_yticks(y, plot["display"])
    ax.invert_yaxis()
    ax.set_xlabel("Predicted restoration score")
    ax.set_ylabel("")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlim(min(-0.03, float(plot["recovery_score"].min()) - 0.04), float(plot["recovery_score"].max()) + 0.08)
    add_program_legend(ax)
    panel_label(ax, "C")
    ax.spines[["top", "right"]].set_visible(False)
    save_panel(fig, "figure7_C_virtual_perturbation.pdf")


def panel_d_sctenifold() -> None:
    df = pd.read_csv(TRIAL / "trial_scTenifoldNet_program_high_vs_other.csv")
    df["program"] = df["program"].map(clean_program)
    df = df[df["dataset"] == "GSE161801"].copy()
    candidates = set(sum(SELECTED.values(), [])) | {"CD63", "SEC11C", "HLA-B", "B2M", "NEAT1", "TMEM59"}
    df = df[df["gene"].isin(candidates)].copy()
    df["neglog10p"] = -np.log10(df["p.value"].clip(lower=1e-300))
    top_genes = (
        df.groupby("gene")["neglog10p"]
        .max()
        .sort_values(ascending=False)
        .head(14)
        .index.tolist()
    )
    plot = df[df["gene"].isin(top_genes)].copy()
    gene_order = (
        plot.groupby("gene")["neglog10p"].max().sort_values(ascending=True).index.tolist()
    )
    program_order = ["Program_6", "Program_8", "Program_9"]
    plot["gene"] = pd.Categorical(plot["gene"], gene_order, ordered=True)
    plot["program"] = pd.Categorical(plot["program"], program_order, ordered=True)
    plot.to_csv(SOURCE_DIR / "figure7_D_sctenifold_network_nodes.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    x = plot["program"].cat.codes
    y = plot["gene"].cat.codes
    sizes = 22 + np.clip(plot["neglog10p"], 0, 5) * 26
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=plot["FC"],
        cmap="Reds",
        vmin=1,
        vmax=max(12, np.nanpercentile(plot["FC"], 95)),
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xticks(np.arange(len(program_order)), [PROGRAM_LABELS[p] for p in program_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(gene_order)), gene_order)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(color=PALETTE["grid"], linewidth=0.45)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Network fold-change")
    for s, label in [(50, "1"), (100, "3"), (150, "5")]:
        ax.scatter([], [], s=s, color="#B93A32", edgecolor="white", linewidth=0.35, label=label)
    ax.legend(title="-log10 p", frameon=False, loc="lower right", bbox_to_anchor=(1.32, 0.0), fontsize=7.2, title_fontsize=7.5)
    panel_label(ax, "D")
    ax.spines[["top", "right"]].set_visible(False)
    save_panel(fig, "figure7_D_sctenifold_network_nodes.pdf")


def panel_d_gene_high_metadata() -> None:
    enr = pd.read_csv(TRIAL / "trial_gene_high_cutoff_enrichment.csv")
    selected_genes = ["TMEM156", "CTSF", "IDUA", "RGS2", "TYMP", "CD44", "KLRD1", "CDKN1C", "DUSP1"]
    features = ["Drug", "1q gain", "CD138_pos", "CD138_neg", "WBM"]
    df = enr[(enr["dataset"] == "GSE161801") & (enr["gene"].isin(selected_genes)) & (enr["feature"].isin(features))].copy()
    mat = df.pivot_table(index="gene", columns="feature", values="log2_or", aggfunc="mean").reindex(selected_genes)
    mat = mat.reindex(columns=features)
    mat.to_csv(SOURCE_DIR / "figure7_D_gene_high_metadata.csv")

    fig, ax = plt.subplots(figsize=(4.7, 3.8))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(np.arange(len(features)), features, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(selected_genes)), selected_genes)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.0, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log2 odds ratio")
    # Light program color bars at left.
    for i, gene in enumerate(selected_genes):
        program = next((p for p, genes in SELECTED.items() if gene in genes), None)
        ax.add_patch(plt.Rectangle((-0.72, i - 0.42), 0.12, 0.84, transform=ax.transData, color=PROGRAM_COLORS.get(program, "#AAAAAA"), clip_on=False))
    panel_label(ax, "D")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_panel(fig, "figure7_D_gene_high_metadata.pdf")


def main() -> None:
    setup()
    panel_a_nomination_scheme()
    panel_b_priority()
    panel_c_virtual_perturbation()
    panel_d_sctenifold()
    print(f"Saved Figure 7 panels to {PANEL_DIR} and {PRISM_PANEL_DIR}")


if __name__ == "__main__":
    main()
