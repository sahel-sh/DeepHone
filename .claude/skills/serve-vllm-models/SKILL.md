---
name: serve-vllm-models
description: Launch the vLLM servers needed for DeepHone experiments — the gpt-oss search agent (with tool calling), the reranker (gpt-oss listwise or Qwen3-Reranker-0.6B cross-encoder), and the gpt-oss-120b LLM-as-judge. Use before running or evaluating deep-research runs.
---

# Serve the vLLM models

Deep-research with reranking needs two model servers (reranker + search agent), each on its own GPU;
evaluation needs a judge server. Start each in the background / a separate terminal.

## Reranker server (GPU 0)
Listwise (paper default):
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b --port 44713 \
  --dtype auto --gpu-memory-utilization 0.8 --enable-prefix-caching --max-model-len 32768
```
Cross-encoder (heterogeneous setup) — serve the reranker instead as:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Reranker-0.6B --port 44713 --dtype auto
```

## Search-agent server (GPU 1) — must support tool calling
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve openai/gpt-oss-20b --port 34713 \
  --dtype auto --gpu-memory-utilization 0.8 --tool-call-parser openai --enable-auto-tool-choice
```
Use `openai/gpt-oss-120b` (and `--tensor-parallel-size N`) for the larger agent.

## Judge server (for evaluation)
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-120b --port 21292 --dtype auto --gpu-memory-utilization 0.8
```

## Wait until ready
Poll `/v1/models` before launching a run:
```bash
until curl -sf "http://localhost:44713/v1/models" | grep -q '"id"'; do sleep 10; done
```

Ports are arbitrary but must match the `--reranker-base-url`, `--model-url`, and `--base_url` flags used in
**run-reranking-experiment** and **evaluate-and-etc**. For OpenAI/Anthropic/Gemini/etc. agents (no local
serving), see the guides in `docs/`.
