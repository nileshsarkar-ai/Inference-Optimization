# SaturateLLM controlled inference experiment

The Qwen2.5-1.5B and 7B controls completed on the supplied A100 PCIe 40 GB. The current-generation Gemma 4 and Qwen3.5 experiments use the isolated `model_suite` scripts. See `experiment_status.json` and [W&B](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility) for status.

## Question and design

Does manually matching graph capture sizes change completed output throughput or capture-memory cost relative to the current vLLM default? This is one configuration-sensitivity study, not an implemented automatic selector or proof of higher GPU utilization.

Three configurations run in fresh processes: current default, a coarse powers-of-two grid, and that coarse grid plus the measured batch sizes. All share a capture ceiling of 256, BF16, tensor parallelism 1, maximum 128 sequences, 512-token context, 4,096 batched tokens, 85% GPU-memory budget, disabled prefix caching and enabled chunked prefill.

Real WikiText-2 paragraphs supply 128-token prompts. Each request generates exactly 128 tokens, ignoring EOS. Batches are 24, 31, 48, 63, 80, 95, 112 and 127, with three warmed repetitions per shape. These deliberately selected cases are not a production trace. The matched set knows the evaluated batches in advance.

## Runtime and reproducibility

A100-PCIE-40GB, driver 595.58.03; Python 3.12; vLLM 0.28.0; PyTorch 2.13.0; Transformers 5.16.1; W&B 0.29.0; datasets 5.0.1. Each run records its complete package list, immutable model/data revisions, prompt checksum, source checksum and resolved settings. Frozen source copies accompany completed datasets.

Use an otherwise idle BF16-capable GPU, public model/data access and valid W&B authentication. Credentials stay outside the repository and artifacts. The remote runtime is `/home/SaturateLLM/.venv`; `bootstrap.sh` records installation. Do not reinstall it for every run.

## Run

```bash
python pilot.py prepare --model Qwen/Qwen2.5-7B --out prepared
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture default --out runs/default > default.log 2>&1
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture matched --out runs/matched > matched.log 2>&1
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture coarse --out runs/coarse > coarse.log 2>&1
python analyze.py runs/default runs/coarse runs/matched --out figures
```

`orchestrate.py --model MODEL_ID` runs this sequence with W&B, followed by separate default/matched profiler calls at batch 63. It expects the private credential file `/home/.saturatellm-auth/wandb.key`. The detached 7B launch was:

```bash
jl run --on 500717 --json --yes -- bash -lc 'cd /home/SaturateLLM/graph_size_pilot && /home/SaturateLLM/.venv/bin/python orchestrate.py --model Qwen/Qwen2.5-7B'
```

Use `jl run logs RUN_ID --tail 20` for bounded checks. The user requested instance destruction after the studies and verified local backup; later reproduction will need an available instance ID.

## What the data supports

Timing covers completed generation including prefill and host work. Initialization and profiler calls are excluded. Plots show all repetitions and means; three within-process samples are not independent confidence intervals. The process order is default, matched, coarse. Small effects require independent runs with order reversal before claiming improvement.

`nominal_next_graph` is a configuration-derived proxy, not observed dispatch. Profiler traces confirm replay but cannot substitute for unprofiled timings. Device memory, nominal KV capacity and rounded engine-reported graph memory are separate measurements. Shared graph pools and runtime profiling prevent graph-count × assumed cost from being a valid memory estimate.

Every call must return the requested number of 128-token outputs. Hash differences are reported; exact output equivalence and unchanged quality are not established. The modern suite additionally saves actual token IDs for inspection.

The code executes real autoregressive generation. Work scales with configurations × repetitions × batch cases × generated tokens, plus initialization and warmup. Memory includes weights, KV/state cache, graph pools and workspaces. This study does not establish online p99 latency, achieved SM/HBM utilization, an automatic selector, workload-shift robustness or universal generalization.

## Sources

- [vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [vLLM profiler](https://docs.vllm.ai/en/stable/api/vllm/config/profiler/)
- [NVIDIA graph-size tradeoff study](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html)
- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
