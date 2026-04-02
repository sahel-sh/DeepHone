#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def extract_metrics_row(run_folder: Path, metrics_json_file: Path) -> Dict[str, str]:
    with metrics_json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    return {
        "run folder": str(run_folder),
        "NDCG@5": str(metrics.get("ndcg_cut_5", "")),
        "NDCG@10": str(metrics.get("ndcg_cut_10", "")),
        "Recall@5": str(metrics.get("recall_5", "")),
        "Recall@10": str(metrics.get("recall_10", "")),
    }


def collect_rows(retrieval_output_dir: Path, suffix: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run_folder in sorted(retrieval_output_dir.iterdir()):
        if not run_folder.is_dir():
            continue

        metric_files = sorted(run_folder.glob(f"*.metrics.{suffix}.json"))
        if not metric_files:
            continue

        if len(metric_files) > 1:
            print(
                f"Warning: multiple {suffix} metric files in {run_folder}; "
                f"using {metric_files[0].name}"
            )

        rows.append(extract_metrics_row(run_folder, metric_files[0]))
    return rows


def write_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run folder", "NDCG@5", "NDCG@10", "Recall@5", "Recall@10"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate gold/evidence metric JSON files into two CSV outputs."
    )
    parser.add_argument(
        "--retrieval-output-dir",
        default="/u6/s8sharif/BrowseComp-Plus/retrieval_output",
        help="Root retrieval output directory containing run folders.",
    )
    parser.add_argument(
        "--gold-output-csv",
        default="/u6/s8sharif/BrowseComp-Plus/retrieval_output/aggregated_metrics_gold.csv",
        help="Output CSV path for gold metrics.",
    )
    parser.add_argument(
        "--evidence-output-csv",
        default="/u6/s8sharif/BrowseComp-Plus/retrieval_output/aggregated_metrics_evidence.csv",
        help="Output CSV path for evidence metrics.",
    )
    args = parser.parse_args()

    retrieval_output_dir = Path(args.retrieval_output_dir).resolve()
    if not retrieval_output_dir.is_dir():
        raise FileNotFoundError(
            f"Retrieval output directory not found: {retrieval_output_dir}"
        )

    gold_rows = collect_rows(retrieval_output_dir, "gold")
    evidence_rows = collect_rows(retrieval_output_dir, "evidence")

    gold_output_csv = Path(args.gold_output_csv).resolve()
    evidence_output_csv = Path(args.evidence_output_csv).resolve()

    write_csv(gold_output_csv, gold_rows)
    write_csv(evidence_output_csv, evidence_rows)

    print(f"Wrote {len(gold_rows)} rows to {gold_output_csv}")
    print(f"Wrote {len(evidence_rows)} rows to {evidence_output_csv}")


if __name__ == "__main__":
    main()
