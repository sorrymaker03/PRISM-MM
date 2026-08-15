#!/usr/bin/env python3
"""Regenerate only Figure 3D/E for the revised study-level display."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "bulk_pre_sc" / "main_figure_code"))

from export_figure3_4_panel_pdfs import (  # noqa: E402
    PROGRAMS,
    SOURCE,
    clustered_heatmap_panel,
    setup,
)


DELTA_FILE = ROOT / "bulk_pre_sc" / "main_figure_source_data" / "figure3_model_bulk_weighted_program_deltas.csv"

# The expression matrix used to build the final contrasts no longer stores GEO
# accessions per individual sample. This map reconstructs study-level groups
# from the final curated source table and the block-level cell/drug contexts.
BLOCK_TO_STUDY = {
    "B001_mc_OPM1": "GSE123660",
    "B002_mc_MM1S": "GSE124510",
    "B003_mc_AMO1": "GSE186445",
    "B004_unk_MM1S": "GSE126463",
    "B007_mc_MMTUMOR": "GSE136725",
    "B008_mc_ARP1": "GSE143406",
    "B009_mc_RPMI": "GSE160572",
    "B011_mc_AMO1": "GSE160572",
    "B012_mc_L363": "GSE160572",
    "B013_mc_RPMI": "GSE144249",
    "B014_mc_ARH77": "GSE160572",
    "B015_cd138_UNK": "GSE162403",
    "B016_unk_KMS21": "GSE165557",
    "B017_unk_KMS27": "GSE165557",
    "B018_unk_KMS34": "GSE165557",
    "B019_unk_AMO1STINGWT": "GSE160572",
    "B020_mc_MM1S": "GSE178340",
    "B021_mc_H929": "GSE184029",
    "B022_mc_RPMI": "GSE184029",
    "B023_mc_AMO1": "GSE124510",
    "B024_unk_KMS12": "GSE189365",
    "B025_unk_MM1S": "GSE189365",
    "B026_mc_SKMM1": "GSE196231",
    "B027_mc_XG2": "GSE196231",
    "B028_mc_MM1S": "GSE214668",
    "B029_unk_NCUMM1": "GSE222411",
    "B030_unk_U266": "GSE222411",
    "B032_mc_MM1S": "GSE31451",
    "B033_mc_OPM2": "GSE31421",
    "B034_unk_KMS27": "GSE269245",
    "B035_unk_KMS12PE": "GSE269245",
    "B036_mc_JJN3": "GSE281182",
    "B037_mc_XG20": "GSE281182",
    "B038_mc_XG7": "GSE281182",
    "B039_unk_MM1S": "GSE246435",
    "B040_unk_OPM2": "GSE246435",
    "B041_mc_JJN3": "GSE37302",
    "B042_mc_MM1S": "GSE136725",
    "B043_mc_U266": "GSE136725",
    "B044_mc_595SP": "GSE41930",
    "B045_mc_638BM": "GSE41930",
    "B046_mc_589BM": "GSE41930",
    "B047_mc_595SP": "GSE41930",
    "B049_unk_BM": "GSE60742",
    "B050_unk_KAS61": "GSE62237",
    "B051_unk_ANBL6": "GSE62237",
    "B052_unk_U266": "GSE62237",
    "B053_unk_OPM2": "GSE62237",
    "B054_mc_UNK": "GSE8546",
}


def main() -> None:
    setup()
    delta = pd.read_csv(DELTA_FILE)
    missing = sorted(set(delta["block_id"]) - set(BLOCK_TO_STUDY))
    if missing:
        raise ValueError(f"Missing block-to-study mapping: {missing}")

    delta["study"] = delta["block_id"].map(BLOCK_TO_STUDY)

    source_mat = delta.groupby("entity_norm")[PROGRAMS].mean().loc[delta["entity_norm"].value_counts().index]
    study_counts = delta["study"].value_counts()
    study_mat = delta.groupby("study")[PROGRAMS].mean().loc[study_counts.index]

    source_ordered = clustered_heatmap_panel(
        source_mat,
        "figure3_D_source_program_shift.pdf",
        "D",
        6.0,
        show_ylabels=True,
        ylabel="Cell source",
    )
    study_ordered = clustered_heatmap_panel(
        study_mat,
        "figure3_E_study_program_shift.pdf",
        "E",
        6.1,
        show_ylabels=True,
        ylabel="Study",
    )

    SOURCE.mkdir(parents=True, exist_ok=True)
    source_ordered.to_csv(SOURCE / "figure3_D_source_program_shift.csv")
    study_ordered.to_csv(SOURCE / "figure3_E_study_program_shift.csv")
    pd.DataFrame(
        sorted(BLOCK_TO_STUDY.items()),
        columns=["block_id", "study"],
    ).to_csv(SOURCE / "figure3_E_block_to_study_mapping.csv", index=False)


if __name__ == "__main__":
    main()
