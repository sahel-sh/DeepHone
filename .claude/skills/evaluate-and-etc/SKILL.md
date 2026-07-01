---
name: evaluate-and-etc
description: Evaluate a DeepHone deep-research run with the gpt-oss-120b LLM-as-judge (accuracy / recall / calibration), aggregate token usage, and compute the Effective Token Cost (ETC) plus the paper's accuracy-vs-cost figures (Figures 1-3). Use after producing run directories with run-reranking-experiment.
---

# Evaluate a run & compute Effective Token Cost

## 1. LLM-as-judge (paper: gpt-oss-120b)
Needs the judge server from **serve-vllm-models** (port 21292):
```bash
python scripts_evaluation/evaluate_run_vllm.py \
    --input_dir runs/qwen3-embedding-8b/gpt-oss-20b/rerank_l_k_20_search_high \
    --ground_truth data/browsecomp_plus_decrypted.jsonl \
    --model openai/gpt-oss-120b --base_url http://localhost:21292/v1 \
    --qrel_evidence topics-qrels/qrel_evidence.txt \
    --temperature 0.0 --top_p 0.95 --max_output_tokens 4096 --batch_size 64
```
This reports **Accuracy** (LLM-judged), **Recall**, **search-call counts**, and **Calibration Error**.
> BrowseComp-Plus's default self-hosted judge (Qwen3-32B) is `scripts_evaluation/evaluate_run.py
> --tensor_parallel_size <num_gpus>`; see `docs/llm_as_judge.md`.

Retrieval-only metrics use `trec_eval` against `topics-qrels/qrel_evidence.txt` (evidence) or
`qrel_golds.txt` (gold).

## 2. Aggregate token usage
```bash
python scripts_rerank/aggregate_token_stats_per_experiment.py runs/.../rerank_l_k_20_search_high
python scripts_rerank/aggregate_token_stats_per_experiment.py runs/.../rerank_l_k_20_search_high/invocation_history
python scripts_rerank/aggregate_reasoning_token_stats_across_all.py   # roll up reasoning tokens across runs
python scripts_rerank/aggregate_results.py                            # combine accuracy + tokens into one CSV
```

## 3. Effective Token Cost (ETC) & figures
ETC weights cached input by α and output/reasoning by β:
```
ETC          = Input_noncached + α · Input_cached + β · Output_total       # α∈{0.1,0.3,0.5}, β∈{3,5,7}
ETC (hetero) = ETC_agent + γ · ETC_reranker                                # γ = 0.32 for Qwen3-Reranker-0.6B vs gpt-oss-20b
```
Generate the paper's accuracy-vs-cost plots from the aggregated CSV:
```bash
python scripts_rerank/plot_etc_deepsearch.py --csv <aggregated_results.csv> --metric accuracy   # Figures 2/3
python scripts_rerank/plot_etc_ranking.py    --csv <one_shot_results.csv>                        # Figure 1 (one-shot)
```
The CSV must contain the per-config accuracy plus `*_input_tokens`, `*_cached_tokens`, `*_output_tokens`
columns produced by the aggregation scripts, keyed by reranking depth `k` ∈ {0, 10, 20, 50}.
