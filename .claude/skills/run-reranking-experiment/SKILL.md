---
name: run-reranking-experiment
description: Run the paper's core experiments — one-shot reranking effectiveness (Table 1) and end-to-end deep-research with listwise or cross-encoder reranking at depth d in {10,20,50} and search reasoning in {low,medium,high} (Tables 2/4, Figures 2/3). Use to produce run directories over BrowseComp-Plus.
---

# Run a reranking experiment

Requires the indexes (**setup-benchmark**) and running model servers (**serve-vllm-models**).
`--first-stage-k` is the reranking depth `d`; candidates are truncated to 512 tokens; the agent sees top-`--k 5`.

## One-shot reranking effectiveness (Table 1 / Figure 1)
Retrieve top-`d`, rerank, write a TREC run, then score it:
```bash
python scripts_retrieval_only/retrieve.py \
    --searcher-type faiss --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name Qwen/Qwen3-Embedding-8B --normalize \
    --query topics-qrels/queries.tsv --output-dir retrieval_output --k 5 \
    --reranker-type batch_listwise_vllm --reranker-model openai/gpt-oss-20b \
    --reranker-base-url http://localhost:44713/v1 --first-stage-k 50 \
    --candidate-max-tokens 512 --prompt-template-path reasonrank_template_low.yaml

python -m pyserini.eval.trec_eval -c -m recall.5,10 -m ndcg_cut.5,10 \
    topics-qrels/qrel_evidence.txt retrieval_output/<run>.trec
```

## End-to-end deep-research with reranking (Tables 2/4, Figures 2/3)
```bash
python search_agent/oss_client.py \
    --model openai/gpt-oss-20b --model-url http://localhost:34713/v1 \
    --searcher-type faiss --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name Qwen/Qwen3-Embedding-8B --normalize --k 5 --snippet-max-tokens 512 \
    --reranker-type batch_listwise_vllm --reranker-model openai/gpt-oss-20b \
    --reranker-base-url http://localhost:44713/v1 \
    --first-stage-k 20 --candidate-max-tokens 512 --reasoning-token-budget 2048 \
    --prompt-template-path reasonrank_template_low.yaml \
    --reasoning-effort high --max-tokens 16384 --num-threads 32 \
    --output-dir runs/qwen3-embedding-8b/gpt-oss-20b/rerank_l_k_20_search_high \
    --invocation-history-dir runs/qwen3-embedding-8b/gpt-oss-20b/rerank_l_k_20_search_high/invocation_history
```

## Sweeps used in the paper
- **Reranking depth**: `--first-stage-k` ∈ {10, 20, 50}; the no-rerank baseline omits all `--reranker-*` flags.
- **Search reasoning**: `--reasoning-effort` ∈ {low, medium, high} with `--max-tokens` 2048 / 8192 / 16384.
- **Heterogeneous cross-encoder** (Table 4 / Fig 3): `--reranker-type batch_pointwise_vllm --reranker-model Qwen/Qwen3-Reranker-0.6B` (no prompt template / reasoning budget needed).
- **Agent size**: swap `--model` / `--reranker-model` between `openai/gpt-oss-20b` and `openai/gpt-oss-120b`.

Output is one JSON per query under `--output-dir` plus reranker call logs under `--invocation-history-dir`
(needed for token accounting). Next: **evaluate-and-etc**.
