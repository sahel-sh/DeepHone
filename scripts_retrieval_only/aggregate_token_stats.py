#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MILLION = 1_000_000.0


def find_aggregated_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/invocat*_history/aggregated_token_stats.jsonl"))


def parse_rank_depth(run_folder_name: str) -> Optional[int]:
    rerank_match = re.search(r"_rerank_.*_k_(\d+)_rtb_", run_folder_name)
    if rerank_match:
        return int(rerank_match.group(1))

    all_k = re.findall(r"_k_(\d+)", run_folder_name)
    if all_k:
        return int(all_k[-1])
    return None


def get_nested_int(container: Dict, path: Iterable[str], default: int = 0) -> int:
    node = container
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    if isinstance(node, (int, float)):
        return int(node)
    return default


def extract_summary_token_stats(path: Path) -> Optional[Dict]:
    summary = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("aid") == "all":
                summary = record.get("token_stats", {})
    if summary is None:
        return None
    return summary


def to_millions(value: int) -> str:
    return f"{value / MILLION:.6f}"


def build_row(run_folder: Path, token_stats: Dict) -> Dict[str, str]:
    input_tokens = token_stats.get("prompt_tokens", token_stats.get("input_tokens", 0))
    output_tokens = token_stats.get(
        "completion_tokens", token_stats.get("output_tokens", 0)
    )
    total_tokens = token_stats.get("total_tokens", input_tokens + output_tokens)

    cached_tokens = get_nested_int(
        token_stats, ("prompt_tokens_details", "cached_tokens"), default=0
    )
    if cached_tokens == 0:
        cached_tokens = get_nested_int(
            token_stats, ("input_tokens_details", "cached_tokens"), default=0
        )

    reasoning_tokens = get_nested_int(
        token_stats, ("completion_tokens_details", "reasoning_tokens"), default=0
    )
    if reasoning_tokens == 0:
        reasoning_tokens = get_nested_int(
            token_stats, ("output_tokens_details", "reasoning_tokens"), default=0
        )

    rank_depth = parse_rank_depth(run_folder.name)

    return {
        "run folder path": str(run_folder),
        "rank_depth": "" if rank_depth is None else str(rank_depth),
        "input_tokens (cached)": f"{to_millions(int(input_tokens))} ({to_millions(cached_tokens)})",
        "output_tokens (reasoning)": f"{to_millions(int(output_tokens))} ({to_millions(reasoning_tokens)})",
        "total_tokens": to_millions(int(total_tokens)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate invocation_history token stats into a single CSV."
    )
    parser.add_argument(
        "--retrieval-output-dir",
        default="/u6/s8sharif/BrowseComp-Plus/retrieval_output",
        help="Root retrieval output directory containing run folders.",
    )
    parser.add_argument(
        "--output-csv",
        default="/u6/s8sharif/BrowseComp-Plus/retrieval_output/aggregated_token_stats.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    retrieval_output_dir = Path(args.retrieval_output_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    files = find_aggregated_files(retrieval_output_dir)
    rows: List[Dict[str, str]] = []
    for stats_file in files:
        run_folder = stats_file.parent.parent
        summary_stats = extract_summary_token_stats(stats_file)
        if not summary_stats:
            continue
        rows.append(build_row(run_folder, summary_stats))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run folder path",
        "rank_depth",
        "input_tokens (cached)",
        "output_tokens (reasoning)",
        "total_tokens",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
