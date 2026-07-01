---
name: setup-benchmark
description: Set up the DeepHone / BrowseComp-Plus benchmark — install the uv environment (Python 3.10, Java 21, flash-attn), decrypt the dataset, and download or build the BM25 / Qwen3-Embedding-8B retrieval indexes. Use this before running any experiment or evaluation.
---

# Setup the benchmark

Prepare the environment, data, and indexes so experiments can run end-to-end.

## 1. Environment (uv, Python 3.10)
```bash
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn        # for faiss / dense retrieval
conda install -c conda-forge openjdk=21               # Java 21 for Pyserini/BM25
```

## 2. Dataset (queries, answers, qrels)
```bash
python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
```
Requires a Hugging Face login (`huggingface-cli login`) with access to `Tevatron/browsecomp-plus`.
Relevance judgments already ship in `topics-qrels/qrel_evidence.txt` (evidence) and `qrel_golds.txt` (gold).

## 3. Indexes
Fastest — download pre-built BM25 + Qwen3-Embedding indexes:
```bash
bash scripts_build_index/download_indexes.sh
```
Or build the paper's dense index (Qwen3-Embedding-8B) yourself (multi-GPU sharded):
```bash
python scripts_build_index/encode_dense_corpus.py --model Qwen/Qwen3-Embedding-8B \
    --output-dir indexes/qwen3-embedding-8b --num-shards 2 --shard-index 0 --gpu 0 &
python scripts_build_index/encode_dense_corpus.py --model Qwen/Qwen3-Embedding-8B \
    --output-dir indexes/qwen3-embedding-8b --num-shards 2 --shard-index 1 --gpu 1 &
wait
```
The FAISS shards are then loaded with `--index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl"`.

## Verify
- `ls data/browsecomp_plus_decrypted.jsonl topics-qrels/queries.tsv`
- `ls indexes/bm25 indexes/qwen3-embedding-8b`
- Next: **serve-vllm-models**, then **run-reranking-experiment**.
