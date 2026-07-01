"""
Encode BrowseComp-Plus corpus with an embedding model via Tevatron and build
a FAISS-ready pickle index.

Supports multi-GPU sharding: each shard encodes a disjoint slice of the
corpus in parallel and writes its own .pkl file.  The resulting shards can
be loaded by FaissSearcher with a glob pattern like:
    --index-path 'indexes/qwen3-embedding-8b/corpus.shard.*.pkl'

Usage (single-GPU):
    python scripts_build_index/encode_dense_corpus.py \
        --model Qwen/Qwen3-Embedding-8B \
        --output-dir indexes/qwen3-embedding-8b

Usage (2-GPU parallel, run in one shell):
    python scripts_build_index/encode_dense_corpus.py \
        --model Qwen/Qwen3-Embedding-8B \
        --output-dir indexes/qwen3-embedding-8b \
        --num-shards 2 --shard-index 0 --gpu 0 &
    python scripts_build_index/encode_dense_corpus.py \
        --model Qwen/Qwen3-Embedding-8B \
        --output-dir indexes/qwen3-embedding-8b \
        --num-shards 2 --shard-index 1 --gpu 1 &
    wait
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode BrowseComp-Plus corpus into FAISS-ready pickle shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace model name for the embedding model.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Tevatron/browsecomp-plus-corpus",
        help="HuggingFace dataset containing the corpus.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional HuggingFace dataset config (e.g., corpus).",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Optional HuggingFace dataset split (e.g., wiki, ecommerce).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output .pkl shard(s).",
    )
    parser.add_argument(
        "--passage-max-len",
        type=int,
        default=4096,
        help="Maximum passage token length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Per-device eval batch size for encoding.",
    )
    parser.add_argument(
        "--pooling",
        default="eos",
        help="Pooling strategy (eos, mean, cls).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Floating-point precision for encoding.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of corpus shards (for multi-GPU parallelism).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard (0-based).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index. If set, CUDA_VISIBLE_DEVICES is overridden.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation to use (default: model's default).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_shards > 1:
        output_path = os.path.join(
            args.output_dir, f"corpus.shard.{args.shard_index}.pkl"
        )
    else:
        output_path = os.path.join(args.output_dir, "corpus.pkl")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        sys.executable, "-m", "tevatron.retriever.driver.encode",
        "--model_name_or_path", args.model,
        "--dataset_name", args.dataset_name,
        "--encode_output_path", output_path,
        "--passage_max_len", str(args.passage_max_len),
        "--pooling", args.pooling,
        "--passage_prefix", "",
        "--per_device_eval_batch_size", str(args.batch_size),
        f"--{args.dtype}",
    ]

    if args.dataset_config:
        cmd.extend(["--dataset_config", args.dataset_config])

    if args.dataset_split:
        cmd.extend(["--dataset_split", args.dataset_split])

    if args.normalize:
        cmd.append("--normalize")

    if args.attn_implementation:
        cmd.extend(["--attn_implementation", args.attn_implementation])

    if args.num_shards > 1:
        cmd.extend([
            "--dataset_number_of_shards", str(args.num_shards),
            "--dataset_shard_index", str(args.shard_index),
        ])

    print(f"Running: {' '.join(cmd)}")
    print(f"  Output: {output_path}")
    if args.gpu is not None:
        print(f"  GPU: {args.gpu}")
    if args.num_shards > 1:
        print(f"  Shard: {args.shard_index}/{args.num_shards}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: Encoding failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Done. Index shard written to: {output_path}")


if __name__ == "__main__":
    main()
