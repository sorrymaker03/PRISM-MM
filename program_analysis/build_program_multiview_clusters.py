#!/usr/bin/env python3
"""Cluster learned programs across discovery and validation response contexts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "bulk_pre_sc" / "main_figure_source_data"
SUPP = ROOT / "bulk_pre_sc" / "supplement" / "source_data"
OUT = ROOT / "bulk_pre_sc" / "biology_interpretation"


def load_view(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame[[f"Program_{i}" for i in range(1, 11)]].apply(pd.to_numeric)
    frame.index = [f"{label}:{value}" for value in frame.index]
    return frame


def main() -> None:
    views = [
        load_view(FIG / "figure3_D_source_program_shift.csv", "discovery_source"),
        load_view(FIG / "figure3_E_study_program_shift.csv", "discovery_study"),
        load_view(FIG / "figure4_E_drug_program_shift.csv", "heldout_drug"),
        load_view(SUPP / "figureS2_C_calibration_scRNA_pair_heatmap.csv", "calibration_scRNA_pair"),
        load_view(FIG / "figure5_E_external_pair_heatmap.csv", "external_scRNA_pair"),
    ]

    contexts = pd.concat(views, axis=0)
    standardized = contexts.sub(contexts.mean(axis=1), axis=0)
    row_sd = contexts.std(axis=1).replace(0, np.nan)
    standardized = standardized.div(row_sd, axis=0).fillna(0)

    program_profiles = standardized.T
    corr = program_profiles.T.corr()
    distance = (1 - corr).clip(lower=0)
    distance_array = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_array, 0)
    tree = linkage(squareform(distance_array, checks=False), method="average")
    cluster = fcluster(tree, t=4, criterion="maxclust")
    order = leaves_list(tree)

    response_summary = pd.DataFrame(index=program_profiles.index)
    for view in views:
        label = view.index[0].split(":", 1)[0]
        response_summary[f"{label}_mean"] = view.mean(axis=0)
        response_summary[f"{label}_positive_fraction"] = (view > 0).mean(axis=0)

    result = response_summary.copy()
    result.insert(0, "cluster", cluster)
    result.insert(1, "cluster_order", np.argsort(order) + 1)
    result.index.name = "program"

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "program_multiview_clusters.csv")
    corr.to_csv(OUT / "program_multiview_correlation.csv")
    program_profiles.to_csv(OUT / "program_multiview_standardized_profiles.csv")

    print(result.sort_values(["cluster", "cluster_order"]).round(3).to_string())
    print("\nProgram correlation:")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
