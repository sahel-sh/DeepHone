---
name: extend-searcher-reranker
description: Add a custom retriever (searcher) or reranker to DeepHone by implementing the BaseSearcher / BaseReranker interface and registering it in the SearcherType / RerankerType enum. Use when integrating a new retrieval or reranking method into the benchmark.
---

# Add a custom searcher or reranker

The agent clients select retrieval and reranking components by CLI name through two enums, so adding a
component is: implement the interface → register it in the enum. See also `docs/custom_retriever.md`.

## Add a searcher (retriever)
1. Implement a subclass of `BaseSearcher` in `searcher/searchers/` (see `searcher/searchers/base.py`).
   Required: `parse_args(parser)`, `_retrieve(query, k)`, `retrieve_batch(queries, qids, k)`, `get_document(docid)`.
   Reranking is handled for you: `BaseSearcher.search()` calls `self.reranker.rerank(...)` when a reranker is set.
   Return results as `{"docid": str, "score": float, "text"/"snippet": str}`.
2. Register it in `searcher/searchers/__init__.py`:
   ```python
   from .my_searcher import MySearcher
   class SearcherType(Enum):
       ...
       MY = ("my_searcher", MySearcher)
   ```
   It is now selectable via `--searcher-type my_searcher`.

## Add a reranker
1. Implement a subclass of `BaseReranker` in `searcher/rerankers/` (see `searcher/rerankers/base.py`).
   Required: `parse_args(parser)`, `__init__(args)`, `rerank(query, retrieved_documents, query_id, k, reasoning=None)`,
   and the `rerank_type` property. Reuse `searcher/rerankers/vllm_openai_utils.py` for vLLM inference and the
   batching pattern in `batch_listwise_reranker_vllm.py` / `batch_pointwise_reranker_vllm.py`.
2. Register it in `searcher/rerankers/__init__.py`:
   ```python
   from .my_reranker import MyReranker
   class RerankerType(Enum):
       ...
       MY = ("my_reranker", MyReranker)
   ```
   It is now selectable via `--reranker-type my_reranker`.

## Verify
```bash
python -c "from searcher.searchers import SearcherType; print(SearcherType.get_choices())"
python -c "from searcher.rerankers import RerankerType; print(RerankerType.get_choices())"
```
Then run a small experiment (**run-reranking-experiment**) with your new `--searcher-type` / `--reranker-type`.
