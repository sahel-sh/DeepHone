# DeepHone — Rerank Before You Reason

| [📄 Paper (arXiv 2601.14224)](https://arxiv.org/abs/2601.14224) | [🏛️ ACL 2026 Findings](https://arxiv.org/abs/2601.14224) | [🤗 BrowseComp-Plus Dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) |

Code for **"Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in
Deep Search Agents"** (Sahel Sharifymoghaddam & Jimmy Lin, *ACL 2026 Findings*).

We study how to spend a compute budget in a deep-search (Deep-Research) agent: on **search-time
reasoning**, or on **listwise / cross-encoder reranking** of retrieved documents *before* the agent
reasons over them. Using the [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) benchmark
(830 reasoning-intensive queries over a fixed ~100K-document corpus), we isolate the retriever, the
reranker, and the agent, and measure the tradeoff with a new **Effective Token Cost (ETC)** metric.

**Key findings.** Reranking consistently improves both retrieval quality and end-to-end accuracy, and a
*moderate* amount of reranking is often more cost-effective than adding search-time reasoning — at
substantially lower token cost. Small cross-encoder rerankers retain most of the accuracy gains at a
fraction of the cost.

> This repository is a fork of **BrowseComp-Plus** (Chen et al., 2025, arXiv 2508.06600). It keeps
> BrowseComp-Plus's benchmark, retrievers, Deep-Research agent clients, and LLM-as-judge evaluation, and
> **adds** the reranking integration and the ETC analysis used in our paper. Please cite **both** papers
> (see [Citation](#citation)).

---

## What's in this repo

- **Rerankers** (`searcher/rerankers/`) — listwise reranking via [RankLLM](https://github.com/castorini/rank_llm)
  (`batch_listwise_vllm`, window 20 / stride 10) and a pointwise **cross-encoder** reranker
  (`batch_pointwise_vllm`, binary yes/no) for the heterogeneous setup.
- **Deep-Research agents** (`search_agent/`) — the gpt-oss agent used in the paper plus the inherited
  BrowseComp-Plus agent clients (OpenAI, Anthropic, Gemini, Qwen, GLM, Search-R1, Tongyi).
- **Retrievers** (`searcher/searchers/`) — BM25 (Lucene/Pyserini), dense/FAISS (Qwen3-Embedding), hybrid.
- **Evaluation** (`scripts_evaluation/`) — LLM-as-judge (gpt-oss-120b or Qwen3-32B) + TREC retrieval metrics.
- **ETC analysis** (`scripts_rerank/`, `scripts_retrieval_only/`) — one-shot reranking effectiveness,
  token-usage aggregation, Effective Token Cost, and the paper's figures.
- **Agent skills** (`.claude/skills/`) — one-command wrappers for setup, serving, running, and evaluating.

The models used in the paper: agents **gpt-oss-20b / gpt-oss-120b** (low/medium/high reasoning);
rerankers **gpt-oss-20b/120b** (listwise) and **Qwen3-Reranker-0.6B** (cross-encoder); retriever
**Qwen3-Embedding-8B**; judge **gpt-oss-120b**.

---

## Installation

`uv` with Python 3.10 is used to manage the environment:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn      # needed for faiss / dense retrieval
```
This repo also depends on Java 21 (for Pyserini/BM25):
```bash
conda install -c conda-forge openjdk=21     # or: sudo apt install -y openjdk-21-jdk
```
Listwise reranking uses RankLLM; the paper's reasoning-rank prompt template ships here as
[`reasonrank_template_low.yaml`](reasonrank_template_low.yaml).

---

## Data & indexes

Download the (obfuscated) BrowseComp-Plus queries, answers, and relevance judgments:
```bash
python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
```
> You may need `huggingface-cli login` first.

Get the retrieval index. Pre-built BM25 + Qwen3-Embedding indexes:
```bash
bash scripts_build_index/download_indexes.sh
```
Or build the dense (Qwen3-Embedding-8B) index yourself:
```bash
python scripts_build_index/encode_dense_corpus.py \
    --model Qwen/Qwen3-Embedding-8B \
    --output-dir indexes/qwen3-embedding-8b \
    --num-shards 2 --shard-index 0 --gpu 0 &
python scripts_build_index/encode_dense_corpus.py \
    --model Qwen/Qwen3-Embedding-8B \
    --output-dir indexes/qwen3-embedding-8b \
    --num-shards 2 --shard-index 1 --gpu 1 &
wait
```

---

## Reproduce the paper

The pipeline is: **serve models → (one-shot rerank | deep-research rerank) → evaluate → ETC/plots**.
The [`.claude/skills/`](.claude/skills) wrap each step; the raw commands are below. See also the
per-model guides under [`docs/`](docs).

### 1. Serve the vLLM models

Deep-research with reranking uses two GPUs — one for the **reranker**, one for the **search agent**:
```bash
# GPU 0 — reranker
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b --port 44713 \
  --dtype auto --gpu-memory-utilization 0.8 --enable-prefix-caching --max-model-len 32768
# GPU 1 — search agent (needs tool calling)
CUDA_VISIBLE_DEVICES=1 vllm serve openai/gpt-oss-20b --port 34713 \
  --dtype auto --gpu-memory-utilization 0.8 --tool-call-parser openai --enable-auto-tool-choice
```
For the cross-encoder (heterogeneous) setup, serve `Qwen/Qwen3-Reranker-0.6B` as the reranker instead.
The judge runs separately: `CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-120b --port 21292 --dtype auto`.

### 2a. One-shot reranking effectiveness (Table 1)

Retrieve top-`d` and rerank to produce a TREC run, then score with `trec_eval`:
```bash
python scripts_retrieval_only/retrieve.py \
    --searcher-type faiss --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name Qwen/Qwen3-Embedding-8B --normalize \
    --query topics-qrels/queries.tsv --output-dir retrieval_output \
    --k 5 --reranker-type batch_listwise_vllm --reranker-model openai/gpt-oss-20b \
    --reranker-base-url http://localhost:44713/v1 --first-stage-k 50 \
    --candidate-max-tokens 512 --prompt-template-path reasonrank_template_low.yaml

python -m pyserini.eval.trec_eval -c -m recall.5,10 -m ndcg_cut.5,10 \
    topics-qrels/qrel_evidence.txt retrieval_output/<run>.trec
```

### 2b. Deep-Research with reranking (Table 2 / Figure 2)

Run the gpt-oss agent over BrowseComp-Plus with a reranker at depth `d ∈ {10, 20, 50}` and search
reasoning effort ∈ {low, medium, high}:
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
    --invocation-history-dir runs/.../invocation_history
```
`--first-stage-k` is the reranking depth `d`; search token budgets are low=2048, medium=8192, high=16384.
Set `--reranker-type batch_pointwise_vllm --reranker-model Qwen/Qwen3-Reranker-0.6B` for the
**heterogeneous cross-encoder** setup (Table 4 / Figure 3). Omit `--reranker-*` for the no-rerank baseline.

### 3. Evaluate (LLM-as-judge + retrieval)

The paper uses **gpt-oss-120b** as the judge via an OpenAI-compatible vLLM server:
```bash
python scripts_evaluation/evaluate_run_vllm.py \
    --input_dir runs/qwen3-embedding-8b/gpt-oss-20b/rerank_l_k_20_search_high \
    --ground_truth data/browsecomp_plus_decrypted.jsonl \
    --model openai/gpt-oss-120b --base_url http://localhost:21292/v1 \
    --qrel_evidence topics-qrels/qrel_evidence.txt \
    --temperature 0.0 --top_p 0.95 --max_output_tokens 4096 --batch_size 64
```
> The BrowseComp-Plus default judge (Qwen3-32B, self-hosted) is available via
> `scripts_evaluation/evaluate_run.py --tensor_parallel_size <num_gpus>`; see [docs/llm_as_judge.md](docs/llm_as_judge.md).

Aggregate token usage (input / cached / output / reasoning) per experiment:
```bash
python scripts_rerank/aggregate_token_stats_per_experiment.py runs/.../rerank_l_k_20_search_high
python scripts_rerank/aggregate_token_stats_per_experiment.py runs/.../invocation_history
```

### 4. Effective Token Cost & figures

ETC combines non-cached input, cached input (× α), and output/reasoning tokens (× β):

```
ETC          = Input_noncached + α · Input_cached + β · Output_total
ETC (hetero) = ETC_agent + γ · ETC_reranker
```
with α ∈ {0.1, 0.3, 0.5}, β ∈ {3, 5, 7}, and γ = 0.32 (relative FLOPs of Qwen3-Reranker-0.6B vs gpt-oss-20b).
Reproduce the ETC accuracy-vs-cost figures (Figures 1–3) from the aggregated per-experiment CSV:
```bash
python scripts_rerank/plot_etc_deepsearch.py --csv <aggregated_results.csv> --metric accuracy
python scripts_rerank/plot_etc_ranking.py    --csv <one_shot_results.csv>          # one-shot (Table 1 / Fig 1)
```

---

## Evaluating your own run

Format your Deep-Research agent's output as one JSON file per query under `runs/<name>/` with at least:
`query_id`, `tool_call_counts`, `status`, `retrieved_docids`, and `result` (final `output_text`). Then run
the evaluation script above. For retrieval-only, produce a TREC run and score against
`topics-qrels/qrel_evidence.txt` (evidence docs) or `topics-qrels/qrel_golds.txt` (gold docs).

## Available components

- **Searchers** (`--searcher-type`): `bm25`, `faiss`, `reasonir`, `hybrid`, `custom`.
- **Rerankers** (`--reranker-type`): `listwise_vllm`, `batch_listwise_vllm`, `batch_pointwise_vllm`.
- **Agents**: see [`docs/`](docs) — `oss.md` (gpt-oss, used in the paper), `openai.md`, `anthropic.md`,
  `gemini.md`, `qwen.md`, `glm.md`, `search-r1.md`, `tongyi.md`, and `custom_retriever.md`.

To plug in your own retriever or reranker, see [`.claude/skills/extend-searcher-reranker`](.claude/skills/extend-searcher-reranker)
and [docs/custom_retriever.md](docs/custom_retriever.md).

---

## Citation

If you use this code, please cite our paper and BrowseComp-Plus:

```bibtex
@inproceedings{sharifymoghaddam2026rerank,
  title   = {Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents},
  author  = {Sharifymoghaddam, Sahel and Lin, Jimmy},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year    = {2026},
  eprint  = {2601.14224},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR}
}

@article{chen2025browsecompplus,
  title   = {BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent},
  author  = {Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
  journal = {arXiv preprint arXiv:2508.06600},
  year    = {2025}
}
```

## Acknowledgements

Built on [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) and
[RankLLM](https://github.com/castorini/rank_llm). BrowseComp-Plus sources its queries from OpenAI's
[BrowseComp](https://openai.com/index/browsecomp).
