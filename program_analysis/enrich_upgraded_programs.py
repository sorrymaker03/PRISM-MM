#!/usr/bin/env python3
"""Run Enrichr Hallmark and Reactome enrichment for upgraded programs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


LIBRARIES = ["MSigDB_Hallmark_2020", "Reactome_2022"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="bulk_pre_sc/model_upgrade")
    parser.add_argument("--prefix", default="our_contextual_v4")
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--genes-file")
    parser.add_argument("--output-file")
    return parser.parse_args()


def add_gene_list(genes: list[str], description: str) -> int:
    boundary = "----CodexEnrichrBoundary"
    parts = []
    for name, value in [("list", "\n".join(genes)), ("description", description)]:
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        parts.append(value)
        parts.append("\r\n")
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")
    request = Request(
        "https://maayanlab.cloud/Enrichr/addList",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return int(json.loads(response.read().decode("utf-8"))["userListId"])


def enrich(list_id: int, library: str) -> list:
    query = urlencode({"userListId": list_id, "backgroundType": library})
    with urlopen(f"https://maayanlab.cloud/Enrichr/enrich?{query}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8")).get(library, [])


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    genes_path = Path(args.genes_file) if args.genes_file else model_dir / f"{args.prefix}_program_genes.csv"
    output_path = Path(args.output_file) if args.output_file else model_dir / f"{args.prefix}_program_enrichr_pathways.csv"
    genes = pd.read_csv(genes_path)
    programs = genes["program"].dropna().drop_duplicates().tolist()
    rows = []
    for program in programs:
        for direction in ["up", "down"]:
            gene_list = (
                genes[genes["program"].eq(program) & genes["direction"].eq(direction)]
                .sort_values("rank")
                .head(args.top_n)["gene"]
                .dropna()
                .astype(str)
                .tolist()
            )
            list_id = add_gene_list(gene_list, f"{program}_{direction}")
            time.sleep(0.12)
            for library in LIBRARIES:
                for item in enrich(list_id, library)[:30]:
                    rows.append(
                        {
                            "program": program,
                            "direction": direction,
                            "library": library,
                            "rank": item[0],
                            "term": item[1],
                            "p_value": item[2],
                            "z_score": item[3],
                            "combined_score": item[4],
                            "overlap_genes": ";".join(item[5]) if isinstance(item[5], list) else str(item[5]),
                            "adjusted_p_value": item[6],
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"Enrichment rows={len(out)}")


if __name__ == "__main__":
    main()
