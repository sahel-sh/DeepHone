# DeepHone — repo guide for agents

Code for *"Rerank Before You Reason"* (arXiv 2601.14224, ACL 2026 Findings) — a study of listwise /
cross-encoder reranking in deep-search agents on the BrowseComp-Plus benchmark, plus the Effective Token
Cost (ETC) metric. This is a fork of [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus); see
[README.md](README.md).

## Reproduction flow (prefer the skills in `.claude/skills/`)
1. `setup-benchmark` — env + dataset + indexes.
2. `serve-vllm-models` — vLLM servers for agent, reranker, judge.
3. `run-reranking-experiment` — one-shot (Table 1) and end-to-end reranking runs (Tables 2/4, Figs 2/3).
4. `evaluate-and-etc` — LLM-as-judge + token aggregation + ETC figures (Figs 1–3).
5. `extend-searcher-reranker` — add a custom retriever/reranker.

## Key layout
- `search_agent/` — Deep-Research agent clients (`oss_client.py` is the paper's agent).
- `searcher/searchers/` — retrievers (`bm25`, `faiss`, `hybrid`, …); `searcher/rerankers/` — rerankers
  (`batch_listwise_vllm`, `batch_pointwise_vllm`). Components are selected by CLI name via the
  `SearcherType` / `RerankerType` enums in each package's `__init__.py`.
- `scripts_retrieval_only/` — one-shot retrieval+rerank → TREC. `scripts_rerank/` — token aggregation & ETC plots.
- `scripts_evaluation/` — LLM-as-judge. `topics-qrels/` — queries & relevance judgments. `docs/` — per-agent guides.

## Conventions
- Run from the repo root; scripts add the root to `sys.path`. Environment is managed by `uv` (Python 3.10) + Java 21.
- `runs/`, `indexes/`, `data/`, `evals/` are gitignored outputs — don't commit them.
- The reranker is optional plumbing in `searcher/searchers/base.py` (`search()` reranks only when a reranker is set).
