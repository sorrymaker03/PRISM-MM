#!/usr/bin/env python3
"""Bulk RNA perturbation program pipeline before single-cell validation.

The pipeline is intentionally conservative:
1. Parse sample names into metadata.
2. Log-transform and filter the expression matrix.
3. Run PCA/QC and simple variance attribution.
4. Build matched control-treatment perturbation deltas.
5. Derive shared cross-drug programs with signed NMF.
6. Export gene programs and sample scores for downstream scRNA validation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import hypergeom, ttest_ind
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import Normalizer, StandardScaler


DRUG_INFO = {
    "ctrl": ("control", "control"),
    "control": ("control", "control"),
    "btz": ("bortezomib", "proteasome_inhibitor"),
    "car": ("carfilzomib", "proteasome_inhibitor"),
    "len": ("lenalidomide", "imid"),
    "pom": ("pomalidomide", "imid"),
    "thal": ("thalidomide", "imid"),
    "dex": ("dexamethasone", "glucocorticoid"),
    "ven": ("venetoclax", "bcl2_inhibitor"),
    "selinexor": ("selinexor", "xpo1_inhibitor"),
    "everolimus": ("everolimus", "mtor_inhibitor"),
    "flavopiridol": ("flavopiridol", "cdk_inhibitor"),
    "dox": ("doxorubicin", "dna_damage"),
    "pds": ("pds", "other_unknown"),
    "runxi": ("runxi", "other_unknown"),
    "ms177": ("ms177", "other_unknown"),
    "ms177n": ("ms177n", "other_unknown"),
    "c24": ("c24", "other_unknown"),
}

CONTROL_TOKENS = {"ctrl", "control"}
DRUG_TOKENS = set(DRUG_INFO)


PATHWAY_SETS = {
    "proteasome_UPR_stress": {
        "HSPA5", "HSP90B1", "XBP1", "DDIT3", "ATF4", "ATF6", "ERN1",
        "EIF2AK3", "HERPUD1", "DNAJB9", "HSPA1A", "HSPA1B", "HSP90AA1",
        "HSP90AB1", "HYOU1", "DERL1", "EDEM1", "EDEM2", "SEL1L",
        "SYVN1", "VCP", "UBC", "SQSTM1", "PSMA1", "PSMA2", "PSMA3",
        "PSMA4", "PSMA5", "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5",
        "PSMB6", "PSMB7", "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD11",
    },
    "interferon_inflammation": {
        "STAT1", "STAT2", "IRF1", "IRF7", "IRF9", "ISG15", "IFI6",
        "IFI27", "IFI44", "IFI44L", "IFIT1", "IFIT2", "IFIT3", "MX1",
        "MX2", "OAS1", "OAS2", "OAS3", "RSAD2", "CXCL10", "CXCL9",
        "GBP1", "GBP2", "HLA-A", "HLA-B", "HLA-C", "B2M",
    },
    "cell_cycle_proliferation": {
        "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5",
        "MCM6", "MCM7", "CDK1", "CDK2", "CCNA2", "CCNB1", "CCNB2",
        "CCNE1", "AURKA", "AURKB", "BUB1", "BUB1B", "CDC20", "CDC45",
        "CDC6", "E2F1", "E2F2", "E2F7", "TYMS", "RRM2",
    },
    "apoptosis_p53": {
        "TP53", "CDKN1A", "BAX", "BAK1", "BCL2L11", "BBC3", "PMAIP1",
        "FAS", "FASLG", "CASP3", "CASP7", "CASP8", "CASP9", "PARP1",
        "MDM2", "GADD45A", "DDB2", "BTG2", "TNFRSF10B",
    },
    "nfkb_tnf": {
        "NFKB1", "NFKB2", "RELA", "REL", "RELB", "NFKBIA", "NFKBIE",
        "TNF", "TNFAIP3", "TRAF1", "TRAF2", "BIRC2", "BIRC3", "ICAM1",
        "CXCL8", "IL6", "IL1B", "CCL2", "JUNB", "NFKBIZ",
    },
    "oxidative_phosphorylation": {
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5", "NDUFB1",
        "NDUFB2", "NDUFB3", "NDUFS1", "NDUFS2", "SDHA", "SDHB",
        "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "COX6A1", "ATP5F1A",
        "ATP5F1B", "ATP5MC1", "ATP5MC2", "ATP5ME",
    },
    "hypoxia_metabolic_stress": {
        "HIF1A", "EPAS1", "VEGFA", "SLC2A1", "HK1", "HK2", "PFKP",
        "ALDOA", "ENO1", "LDHA", "PDK1", "BNIP3", "NDRG1", "CA9",
        "EGLN1", "ANGPTL4",
    },
    "adhesion_niche_ecm": {
        "CXCR4", "ITGA4", "ITGB1", "VCAM1", "ICAM1", "FN1", "VIM",
        "CD44", "SELL", "SPP1", "LGALS1", "LGALS3", "COL1A1", "COL1A2",
        "COL6A1", "MMP2", "MMP9", "LAMB1", "LAMC1",
    },
    "plasma_cell_identity": {
        "SDC1", "MZB1", "XBP1", "PRDM1", "IRF4", "TNFRSF17", "SLAMF7",
        "CD38", "CD79A", "CD79B", "JCHAIN", "DERL3", "FKBP11", "IGKC",
        "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2",
    },
    "myc_translation": {
        "MYC", "MAX", "NPM1", "FBL", "RPLP0", "RPL3", "RPL4", "RPL5",
        "RPL7", "RPL10", "RPS3", "RPS6", "RPS8", "EIF4E", "EIF4A1",
        "EIF3A", "NCL", "NOP56",
    },
    "dna_damage_repair": {
        "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "RAD51",
        "RAD50", "MRE11", "NBN", "PARP1", "XRCC5", "XRCC6", "H2AFX",
        "GADD45A", "DDB2", "RPA1", "RPA2", "FANCD2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", default="raw data/RNAfinal.csv")
    parser.add_argument("--outdir", default="bulk_pre_sc/results")
    parser.add_argument("--min-detected-frac", type=float, default=0.03)
    parser.add_argument("--top-variable-genes", type=int, default=6000)
    parser.add_argument("--top-delta-genes", type=int, default=5000)
    parser.add_argument("--n-programs", type=int, default=6)
    parser.add_argument("--top-genes-per-program", type=int, default=80)
    parser.add_argument("--human-symbol-like", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-technical-gene-families", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-non-mm-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_entity(entity: str) -> str:
    entity = (entity or "unknown").strip()
    return re.sub(r"[^A-Za-z0-9]+", "", entity).upper() or "UNKNOWN"


def parse_sample_name(sample: str) -> dict:
    parts = sample.split("_")
    low = [p.lower() for p in parts]
    drug_idx = None
    for i, token in enumerate(low):
        if token in DRUG_TOKENS:
            drug_idx = i
            break
    if drug_idx is None:
        prefix = parts[0] if parts else "unknown"
        entity = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        drug_token = "unknown"
        suffix = ""
    else:
        prefix = parts[0]
        entity = "_".join(parts[1:drug_idx]) or "unknown"
        drug_token = low[drug_idx]
        suffix = "_".join(parts[drug_idx + 1 :])

    drug_name, drug_class = DRUG_INFO.get(drug_token, (drug_token, "other_unknown"))
    time_match = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?)h(?=$|[_\-.])", suffix, flags=re.I)
    time_hours = float(time_match.group(1)) if time_match else np.nan

    dose_label = "none"
    if re.search(r"\blow", suffix, flags=re.I):
        dose_label = "low"
    elif re.search(r"\bhigh", suffix, flags=re.I):
        dose_label = "high"

    phenotype_parts = []
    for token in ["VR", "CR", "WT", "lenresponsive"]:
        if re.search(rf"(^|[_\-.]){re.escape(token)}($|[_\-.])", suffix, flags=re.I):
            phenotype_parts.append(token.upper())
    phenotype_label = "+".join(sorted(set(phenotype_parts))) or "none"

    patient_id = None
    patient_match = re.search(r"\b(P\d+|MPM\d+|MPR\d+)\b", suffix, flags=re.I)
    if patient_match:
        patient_id = patient_match.group(1).upper()
    elif prefix.lower() == "cd138":
        compact_suffix = suffix.replace("_", "")
        if re.match(r"^\d+[A-Z]?$", compact_suffix, flags=re.I):
            patient_id = compact_suffix.upper()
    elif entity.lower() == "unk":
        patient_match = re.search(r"\b(P\d+)\b", suffix, flags=re.I)
        if patient_match:
            patient_id = patient_match.group(1).upper()

    entity_norm = normalize_entity(entity)
    subject_id = patient_id or entity_norm
    time_label = "NA" if pd.isna(time_hours) else f"{time_hours:g}h"

    non_mm_models = {"JURKAT", "JEKO1"}
    is_mm_like = entity_norm not in non_mm_models

    return {
        "sample": sample,
        "prefix": prefix,
        "prefix_lower": prefix.lower(),
        "entity": entity,
        "entity_norm": entity_norm,
        "drug_token": drug_token,
        "drug_name": drug_name,
        "drug_class": drug_class,
        "is_control": drug_token in CONTROL_TOKENS,
        "is_treated": drug_token not in CONTROL_TOKENS,
        "suffix": suffix,
        "time_hours": time_hours,
        "time_label": time_label,
        "dose_label": dose_label,
        "phenotype_label": phenotype_label,
        "patient_id": patient_id or "",
        "subject_id": subject_id,
        "has_patient_id": patient_id is not None,
        "is_mm_like": is_mm_like,
    }


def build_metadata(samples: list[str], exclude_non_mm_models: bool) -> pd.DataFrame:
    meta = pd.DataFrame([parse_sample_name(s) for s in samples])
    meta["analysis_include"] = True
    if exclude_non_mm_models:
        meta.loc[~meta["is_mm_like"], "analysis_include"] = False
    return meta


def valid_gene_mask(index: pd.Index, human_symbol_like: bool, exclude_technical_gene_families: bool) -> pd.Series:
    s = pd.Series(index.astype(str), index=index)
    mask = s.ne("") & ~s.str.contains(r"\?", regex=True) & ~s.eq("---")
    if human_symbol_like:
        mask &= ~s.str.match(r"^ENS", case=False, na=False)
        mask &= ~s.str.match(r"^\d", na=False)
        mask &= ~s.str.contains(r"RIK$", case=False, regex=True)
        mask &= s.str.len().between(2, 40)
    if exclude_technical_gene_families:
        technical_regex = (
            r"^(LOC\d+|LINC\d+|MIR\d+|MIRLET|SNOR|SNORA|SNORD|RNU|RNY|RN7S|RNA\d|"
            r"RNVU|TRE-|MGC\d+|RP\d+[-_]|AF\d{5,}|MTRNR|MT-|MALAT1$|NEAT1$|XIST$|"
            r"IGH[-]?[VDJCGAEM]|IGK[-]?[VDJC]|IGL[-]?[VDJC]|TR[ABDG][-]?[VDJC])"
        )
        mask &= ~s.str.match(technical_regex, case=False, na=False)
    return mask


def read_expression(expr_path: Path, human_symbol_like: bool, exclude_technical_gene_families: bool) -> pd.DataFrame:
    expr = pd.read_csv(expr_path, index_col=0)
    expr.index = expr.index.fillna("").astype(str)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    expr = expr.loc[valid_gene_mask(expr.index, human_symbol_like, exclude_technical_gene_families)]
    return expr


def filter_expression(expr: pd.DataFrame, min_detected_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    detected_frac = (expr > 0).mean(axis=1)
    mean_raw = expr.mean(axis=1)
    keep = (detected_frac >= min_detected_frac) & (mean_raw > 0)
    qc = pd.DataFrame({
        "gene": expr.index,
        "detected_frac": detected_frac.to_numpy(),
        "mean_raw": mean_raw.to_numpy(),
    })
    return expr.loc[keep], qc


def select_top_variable_genes(expr_log: pd.DataFrame, n: int) -> list[str]:
    gene_var = expr_log.var(axis=1)
    n = min(n, len(gene_var))
    return gene_var.sort_values(ascending=False).head(n).index.tolist()


def categorical_r2(y: np.ndarray, groups: pd.Series) -> float:
    keep = groups.notna().to_numpy()
    y = y[keep]
    groups = groups[keep].astype(str)
    if len(y) == 0:
        return np.nan
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return np.nan
    sse = 0.0
    for _, idx in groups.groupby(groups).groups.items():
        vals = y[list(idx)]
        sse += float(np.sum((vals - vals.mean()) ** 2))
    return max(0.0, min(1.0, 1.0 - sse / total))


def run_pca(expr_log: pd.DataFrame, genes: list[str], meta: pd.DataFrame, outdir: Path, random_state: int) -> pd.DataFrame:
    x = expr_log.loc[genes, meta["sample"]].T
    x_scaled = StandardScaler().fit_transform(x)
    n_components = min(12, x_scaled.shape[0] - 1, x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(x_scaled)
    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    coords = pd.DataFrame(pcs, columns=pc_cols)
    coords["sample"] = meta["sample"].to_numpy()
    coords = coords.merge(meta, on="sample", how="left")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        coords.attrs[f"PC{i}_variance_ratio"] = float(ratio)

    coords.to_csv(outdir / "sample_pca_coordinates.csv", index=False)
    with open(outdir / "pca_explained_variance.json", "w") as f:
        json.dump({f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)}, f, indent=2)

    plot_pca(coords, outdir, "drug_token", "pca_by_drug.png")
    plot_pca(coords, outdir, "prefix_lower", "pca_by_prefix.png")
    plot_pca(coords, outdir, "drug_class", "pca_by_drug_class.png")
    plot_pca(coords, outdir, "phenotype_label", "pca_by_phenotype.png")

    variables = [
        "prefix_lower", "entity_norm", "drug_token", "drug_class", "is_treated",
        "phenotype_label", "has_patient_id", "time_label",
    ]
    rows = []
    for pc in pc_cols[:8]:
        for var in variables:
            rows.append({"pc": pc, "variable": var, "r2": categorical_r2(coords[pc].to_numpy(), coords[var])})
    var_df = pd.DataFrame(rows)
    var_df.to_csv(outdir / "variance_explained_by_metadata.csv", index=False)
    return coords


def plot_pca(coords: pd.DataFrame, outdir: Path, color_var: str, filename: str) -> None:
    plt.figure(figsize=(8.2, 6.6))
    plot_df = coords.copy()
    if plot_df[color_var].nunique() > 14:
        top = plot_df[color_var].value_counts().head(13).index
        plot_df[color_var] = np.where(plot_df[color_var].isin(top), plot_df[color_var], "other")
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=color_var,
        style="is_treated" if "is_treated" in plot_df else None,
        s=38,
        linewidth=0,
        alpha=0.82,
    )
    plt.title(f"Bulk expression PCA colored by {color_var}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=220)
    plt.close()


@dataclass
class DeltaContext:
    context_id: str
    drug_token: str
    drug_name: str
    drug_class: str
    prefix_lower: str
    entity_norm: str
    subject_id: str
    phenotype_label: str
    time_label: str
    dose_label: str
    n_treated: int
    n_control: int
    treated_samples: str
    control_samples: str


def choose_controls(ctrl_meta: pd.DataFrame, time_label: str) -> pd.DataFrame:
    if ctrl_meta.empty:
        return ctrl_meta
    exact = ctrl_meta[ctrl_meta["time_label"].eq(time_label)]
    if not exact.empty:
        return exact
    baseline = ctrl_meta[ctrl_meta["time_label"].isin(["0h", "NA"])]
    if not baseline.empty:
        return baseline
    return ctrl_meta


def build_delta_contexts(meta: pd.DataFrame) -> tuple[list[DeltaContext], pd.DataFrame]:
    use_meta = meta[meta["analysis_include"]].copy()
    ctrls = use_meta[use_meta["is_control"]]
    treated = use_meta[~use_meta["is_control"]]
    group_cols = [
        "prefix_lower", "entity_norm", "subject_id", "phenotype_label",
        "drug_token", "time_label", "dose_label",
    ]
    contexts = []
    skipped_rows = []
    for keys, group in treated.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        ctrl_pool = ctrls[
            ctrls["prefix_lower"].eq(row["prefix_lower"])
            & ctrls["entity_norm"].eq(row["entity_norm"])
            & ctrls["subject_id"].eq(row["subject_id"])
            & ctrls["phenotype_label"].eq(row["phenotype_label"])
        ]
        chosen_ctrls = choose_controls(ctrl_pool, row["time_label"])
        if chosen_ctrls.empty:
            skipped_rows.append({
                **row,
                "n_treated": len(group),
                "reason": "no matched control",
                "treated_samples": ";".join(group["sample"].tolist()),
            })
            continue
        drug_token = row["drug_token"]
        drug_name, drug_class = DRUG_INFO.get(drug_token, (drug_token, "other_unknown"))
        context_id = "|".join([
            row["prefix_lower"], row["entity_norm"], row["subject_id"],
            row["phenotype_label"], drug_token, row["time_label"], row["dose_label"],
        ])
        contexts.append(DeltaContext(
            context_id=context_id,
            drug_token=drug_token,
            drug_name=drug_name,
            drug_class=drug_class,
            prefix_lower=row["prefix_lower"],
            entity_norm=row["entity_norm"],
            subject_id=row["subject_id"],
            phenotype_label=row["phenotype_label"],
            time_label=row["time_label"],
            dose_label=row["dose_label"],
            n_treated=len(group),
            n_control=len(chosen_ctrls),
            treated_samples=";".join(group["sample"].tolist()),
            control_samples=";".join(chosen_ctrls["sample"].tolist()),
        ))
    skipped = pd.DataFrame(skipped_rows)
    return contexts, skipped


def compute_delta_matrix(expr_log: pd.DataFrame, contexts: list[DeltaContext]) -> pd.DataFrame:
    sample_expr = expr_log.T
    deltas = []
    context_ids = []
    for ctx in contexts:
        treated = ctx.treated_samples.split(";")
        controls = ctx.control_samples.split(";")
        delta = sample_expr.loc[treated].mean(axis=0) - sample_expr.loc[controls].mean(axis=0)
        deltas.append(delta.to_numpy(dtype=np.float32))
        context_ids.append(ctx.context_id)
    return pd.DataFrame(deltas, index=context_ids, columns=expr_log.index)


def save_gzip_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=index)


def summarize_drug_signatures(delta_df: pd.DataFrame, context_df: pd.DataFrame, outdir: Path, top_n: int = 150) -> pd.DataFrame:
    drug_mean = delta_df.groupby(context_df.set_index("context_id").loc[delta_df.index, "drug_token"]).mean()
    save_gzip_csv(drug_mean, outdir / "drug_signature_matrix_log2delta.csv.gz")

    context_counts = context_df.groupby("drug_token")["context_id"].nunique().to_dict()
    rows = []
    for drug, vals in drug_mean.iterrows():
        n_contexts = context_counts.get(drug, 0)
        drug_name, drug_class = DRUG_INFO.get(drug, (drug, "other_unknown"))
        ordered_up = vals.sort_values(ascending=False).head(top_n)
        ordered_down = vals.sort_values(ascending=True).head(top_n)
        for rank, (gene, val) in enumerate(ordered_up.items(), start=1):
            rows.append({
                "drug_token": drug, "drug_name": drug_name, "drug_class": drug_class,
                "n_contexts": n_contexts, "direction": "up", "rank": rank,
                "gene": gene, "mean_log2_delta": float(val),
            })
        for rank, (gene, val) in enumerate(ordered_down.items(), start=1):
            rows.append({
                "drug_token": drug, "drug_name": drug_name, "drug_class": drug_class,
                "n_contexts": n_contexts, "direction": "down", "rank": rank,
                "gene": gene, "mean_log2_delta": float(val),
            })
    top_df = pd.DataFrame(rows)
    top_df.to_csv(outdir / "drug_signature_top_genes.csv", index=False)
    return drug_mean


def run_nmf_programs(
    delta_df: pd.DataFrame,
    context_df: pd.DataFrame,
    outdir: Path,
    top_delta_genes: int,
    n_programs: int,
    top_genes_per_program: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if delta_df.shape[0] < 3:
        raise RuntimeError("Need at least 3 matched perturbation contexts for NMF.")
    gene_var = delta_df.var(axis=0).sort_values(ascending=False)
    selected_genes = gene_var.head(min(top_delta_genes, len(gene_var))).index.tolist()
    delta_use = delta_df[selected_genes]
    save_gzip_csv(delta_use, outdir / "context_delta_matrix_selected_genes.csv.gz")

    x_pos = np.clip(delta_use.to_numpy(dtype=np.float32), 0, None)
    x_neg = np.clip(-delta_use.to_numpy(dtype=np.float32), 0, None)
    x_signed = np.hstack([x_pos, x_neg])
    x_signed = Normalizer(norm="l2").fit_transform(x_signed)

    k = min(n_programs, max(2, delta_df.shape[0] - 1))
    nmf = NMF(
        n_components=k,
        init="nndsvda",
        random_state=random_state,
        max_iter=3000,
        solver="mu",
        beta_loss="frobenius",
    )
    w = nmf.fit_transform(np.maximum(x_signed, 0))
    h = nmf.components_

    program_cols = [f"Program_{i+1}" for i in range(k)]
    activity = pd.DataFrame(w, index=delta_df.index, columns=program_cols).reset_index(names="context_id")
    activity = activity.merge(context_df, on="context_id", how="left")
    activity.to_csv(outdir / "context_program_activity.csv", index=False)

    drug_activity = activity.groupby(["drug_token", "drug_name", "drug_class"], as_index=False)[program_cols].mean()
    drug_activity.to_csv(outdir / "drug_program_activity.csv", index=False)

    plot_program_heatmap(drug_activity, program_cols, outdir)

    program_rows = []
    n_genes = len(selected_genes)
    for p_idx, program in enumerate(program_cols):
        up_weights = pd.Series(h[p_idx, :n_genes], index=selected_genes)
        down_weights = pd.Series(h[p_idx, n_genes:], index=selected_genes)
        for direction, weights in [("up", up_weights), ("down", down_weights)]:
            for rank, (gene, weight) in enumerate(weights.sort_values(ascending=False).head(top_genes_per_program).items(), start=1):
                program_rows.append({
                    "program": program,
                    "direction": direction,
                    "rank": rank,
                    "gene": gene,
                    "weight": float(weight),
                })
    program_genes = pd.DataFrame(program_rows)
    program_genes.to_csv(outdir / "nmf_program_genes.csv", index=False)
    with pd.ExcelWriter(outdir / "nmf_program_genes.xlsx") as writer:
        for program in program_cols:
            program_genes[program_genes["program"].eq(program)].to_excel(writer, sheet_name=program, index=False)

    pathway_df = pathway_overlaps(program_genes, set(delta_df.columns))
    pathway_df.to_csv(outdir / "program_pathway_overlaps.csv", index=False)
    export_sc_validation_assets(program_genes, pathway_df, outdir)

    return activity, program_genes


def plot_program_heatmap(drug_activity: pd.DataFrame, program_cols: list[str], outdir: Path) -> None:
    heat = drug_activity.set_index("drug_token")[program_cols].T
    plt.figure(figsize=(max(7, 0.55 * heat.shape[1]), 4.8))
    sns.heatmap(heat, cmap="viridis", linewidths=0.3, linecolor="white")
    plt.title("Mean NMF program activity by drug")
    plt.xlabel("Drug")
    plt.ylabel("Program")
    plt.tight_layout()
    plt.savefig(outdir / "drug_program_activity_heatmap.png", dpi=240)
    plt.close()


def pathway_overlaps(program_genes: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    rows = []
    m = max(len(universe), 1)
    for (program, direction), sub in program_genes.groupby(["program", "direction"]):
        genes = set(sub.sort_values("rank").head(60)["gene"])
        n = len(genes)
        for pathway, pathway_genes in PATHWAY_SETS.items():
            pathway_in_universe = set(pathway_genes) & universe
            k = len(pathway_in_universe)
            overlap = genes & pathway_in_universe
            if k == 0 or n == 0:
                pval = 1.0
            else:
                pval = float(hypergeom.sf(len(overlap) - 1, m, k, n))
            rows.append({
                "program": program,
                "direction": direction,
                "pathway": pathway,
                "overlap_n": len(overlap),
                "program_gene_n": n,
                "pathway_gene_n": k,
                "p_value": pval,
                "overlap_genes": ";".join(sorted(overlap)),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_within_program_direction"] = out.groupby(["program", "direction"])["p_value"].rank(method="first")
    return out.sort_values(["program", "direction", "p_value", "overlap_n"], ascending=[True, True, True, False])


def export_sc_validation_assets(program_genes: pd.DataFrame, pathway_df: pd.DataFrame, outdir: Path, top_n: int = 50) -> None:
    gene_sets = {}
    gmt_lines = []
    prompt_lines = [
        "# LLM-assisted annotation prompts",
        "",
        "Use these prompts only for hypothesis generation. Confirm labels with pathway enrichment and single-cell validation.",
        "",
    ]
    programs = sorted(program_genes["program"].unique(), key=lambda x: int(x.split("_")[1]))
    for program in programs:
        gene_sets[program] = {}
        prompt_lines.append(f"## {program}")
        for direction in ["up", "down"]:
            genes = (
                program_genes[
                    program_genes["program"].eq(program)
                    & program_genes["direction"].eq(direction)
                ]
                .sort_values("rank")
                .head(top_n)["gene"]
                .tolist()
            )
            gene_sets[program][direction] = genes
            gmt_lines.append("\t".join([f"{program}_{direction.upper()}", f"Bulk NMF {program} {direction} genes"] + genes))
            prompt_lines.append(f"- {direction.upper()} genes: {', '.join(genes[:35])}")
        if not pathway_df.empty:
            top_pathways = (
                pathway_df[pathway_df["program"].eq(program)]
                .sort_values(["p_value", "overlap_n"], ascending=[True, False])
                .head(6)
            )
            if not top_pathways.empty:
                prompt_lines.append("- Candidate pathway overlaps:")
                for _, row in top_pathways.iterrows():
                    prompt_lines.append(
                        f"  - {row['direction']} / {row['pathway']}: overlap={row['overlap_n']}, genes={row.get('overlap_genes', '')}"
                    )
        prompt_lines.append(
            "Prompt: Given the up/down genes and candidate pathway overlaps above, propose a concise biological label for this multiple-myeloma drug perturbation program, list the supporting mechanisms, and flag any genes that look technical or batch-driven."
        )
        prompt_lines.append("")

    with open(outdir / "sc_validation_gene_sets.json", "w") as f:
        json.dump(gene_sets, f, indent=2)
    (outdir / "sc_validation_gene_sets.gmt").write_text("\n".join(gmt_lines) + "\n")
    (outdir / "program_annotation_prompts.md").write_text("\n".join(prompt_lines))


def score_samples(expr_log: pd.DataFrame, program_genes: pd.DataFrame, meta: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    programs = sorted(program_genes["program"].unique(), key=lambda x: int(x.split("_")[1]))
    needed_genes = sorted(set(program_genes["gene"]) & set(expr_log.index))
    x = expr_log.loc[needed_genes, meta["sample"]]
    z = pd.DataFrame(
        StandardScaler().fit_transform(x.T).T,
        index=x.index,
        columns=x.columns,
    )

    scores = pd.DataFrame({"sample": meta["sample"]})
    for program in programs:
        pgenes = program_genes[program_genes["program"].eq(program)]
        up = pgenes[pgenes["direction"].eq("up")].sort_values("rank").head(50)["gene"].tolist()
        down = pgenes[pgenes["direction"].eq("down")].sort_values("rank").head(50)["gene"].tolist()
        up = [g for g in up if g in z.index]
        down = [g for g in down if g in z.index]
        up_score = z.loc[up].mean(axis=0) if up else 0
        down_score = z.loc[down].mean(axis=0) if down else 0
        scores[program] = np.asarray(up_score) - np.asarray(down_score)
    scores = scores.merge(meta, on="sample", how="left")
    scores.to_csv(outdir / "sample_program_scores.csv", index=False)

    assoc = []
    program_cols = programs
    for variable in ["drug_token", "drug_class", "phenotype_label", "is_treated"]:
        for value, sub in scores.groupby(variable):
            if len(sub) < 3:
                continue
            rest = scores[~scores.index.isin(sub.index)]
            if len(rest) < 3:
                continue
            for program in program_cols:
                stat, pval = ttest_ind(sub[program], rest[program], equal_var=False, nan_policy="omit")
                assoc.append({
                    "variable": variable,
                    "level": value,
                    "n_level": len(sub),
                    "n_rest": len(rest),
                    "program": program,
                    "mean_level": float(sub[program].mean()),
                    "mean_rest": float(rest[program].mean()),
                    "difference": float(sub[program].mean() - rest[program].mean()),
                    "p_value": float(pval) if not np.isnan(pval) else np.nan,
                })
    pd.DataFrame(assoc).sort_values("p_value").to_csv(outdir / "program_score_group_differences.csv", index=False)
    return scores


def write_summary(
    outdir: Path,
    expr: pd.DataFrame,
    expr_filtered: pd.DataFrame,
    meta: pd.DataFrame,
    contexts: list[DeltaContext],
    skipped: pd.DataFrame,
    delta_df: pd.DataFrame,
    program_genes: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        display = df.copy()
        display = display.astype(str)
        header = "| " + " | ".join(display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
        return "\n".join([header, sep] + rows)

    context_df = pd.DataFrame([c.__dict__ for c in contexts])
    lines = []
    lines.append("# Bulk pre-single-cell analysis summary")
    lines.append("")
    lines.append(f"- Raw matrix: {expr.shape[0]:,} genes/features x {expr.shape[1]:,} samples")
    lines.append(f"- Filtered matrix: {expr_filtered.shape[0]:,} genes/features x {expr_filtered.shape[1]:,} samples")
    lines.append(f"- Samples included for perturbation analysis: {meta['analysis_include'].sum():,} / {len(meta):,}")
    lines.append(f"- Matched treatment-control perturbation contexts: {len(contexts):,}")
    lines.append(f"- Skipped treated groups without matched controls: {len(skipped):,}")
    lines.append("")
    lines.append("## Drug counts")
    drug_counts = meta.groupby(["drug_token", "drug_name", "drug_class"]).size().reset_index(name="n_samples").sort_values("n_samples", ascending=False)
    lines.append(markdown_table(drug_counts))
    lines.append("")
    if not context_df.empty:
        lines.append("## Matched delta contexts by drug")
        context_counts = context_df.groupby(["drug_token", "drug_name", "drug_class"]).agg(n_contexts=("context_id", "nunique"), n_treated=("n_treated", "sum"), n_control=("n_control", "sum")).reset_index().sort_values("n_contexts", ascending=False)
        lines.append(markdown_table(context_counts))
    lines.append("")
    lines.append("## Program top genes preview")
    preview = program_genes[program_genes["rank"] <= 8].pivot_table(
        index=["program", "direction"],
        values="gene",
        aggfunc=lambda x: ", ".join(x),
    ).reset_index()
    lines.append(markdown_table(preview))
    lines.append("")
    lines.append("## Outputs for single-cell validation")
    lines.append("- `nmf_program_genes.csv`: top up/down genes per bulk-derived program.")
    lines.append("- `sample_program_scores.csv`: program scores in each bulk sample.")
    lines.append("- `drug_program_activity.csv`: cross-drug activity of each program.")
    lines.append("- `context_delta_matrix_selected_genes.csv.gz`: matched treatment-control deltas for reproducibility.")
    lines.append("")
    (outdir / "analysis_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    expr_path = Path(args.expr)
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    print(f"[1/8] Reading expression matrix: {expr_path}")
    expr = read_expression(expr_path, args.human_symbol_like, args.exclude_technical_gene_families)
    samples = expr.columns.tolist()
    meta = build_metadata(samples, args.exclude_non_mm_models)
    meta.to_csv(outdir / "sample_metadata_parsed.csv", index=False)
    meta.groupby(["drug_token", "drug_name", "drug_class"], as_index=False).size().rename(columns={"size": "n_samples"}).sort_values("n_samples", ascending=False).to_csv(outdir / "sample_counts_by_drug.csv", index=False)

    print("[2/8] Filtering and log-transforming genes")
    expr_filtered, gene_qc = filter_expression(expr, args.min_detected_frac)
    gene_qc.to_csv(outdir / "gene_qc_before_filtering.csv", index=False)
    expr_log = np.log2(expr_filtered + 1.0).astype("float32")
    included_samples = meta.loc[meta["analysis_include"], "sample"].tolist()
    expr_log = expr_log[included_samples]
    meta_included = meta[meta["analysis_include"]].reset_index(drop=True)

    print("[3/8] PCA and metadata variance diagnostics")
    variable_genes = select_top_variable_genes(expr_log, args.top_variable_genes)
    pd.Series(variable_genes, name="gene").to_csv(outdir / "top_variable_genes_for_pca.csv", index=False)
    run_pca(expr_log, variable_genes, meta_included, outdir, args.random_state)

    print("[4/8] Building matched treatment-control contexts")
    contexts, skipped = build_delta_contexts(meta)
    context_df = pd.DataFrame([c.__dict__ for c in contexts])
    context_df.to_csv(outdir / "matched_delta_contexts.csv", index=False)
    skipped.to_csv(outdir / "skipped_treated_groups_without_controls.csv", index=False)
    if len(contexts) < 3:
        raise RuntimeError("Too few matched control-treatment contexts after parsing.")

    print("[5/8] Computing log2 treatment deltas")
    delta_df = compute_delta_matrix(expr_log, contexts)
    save_gzip_csv(delta_df, outdir / "context_delta_matrix_all_filtered_genes.csv.gz")

    print("[6/8] Summarizing drug perturbation signatures")
    summarize_drug_signatures(delta_df, context_df, outdir)

    print("[7/8] Learning shared cross-drug programs with signed NMF")
    activity, program_genes = run_nmf_programs(
        delta_df=delta_df,
        context_df=context_df,
        outdir=outdir,
        top_delta_genes=args.top_delta_genes,
        n_programs=args.n_programs,
        top_genes_per_program=args.top_genes_per_program,
        random_state=args.random_state,
    )

    print("[8/8] Scoring samples and writing summary")
    score_samples(expr_log, program_genes, meta_included, outdir)
    write_summary(outdir, expr, expr_filtered, meta, contexts, skipped, delta_df, program_genes)
    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
