# SaturateLLM: first GPU experiment

**Status: prepared; no GPU execution or experimental result yet.** Run only on the GPU the researcher supplies. The scripts have not been validated against an installed GPU runtime.

The first experiment asks whether CUDA-graph capture sizes make a measurable difference to inference on a small dense model. It is a configuration-sensitivity pilot, not the complete proposed memory-aware selector. A successful pilot motivates that selector; it cannot establish its novelty or generalization.

## Claim and fair comparison

Hypothesis: workload-matched capture sizes may increase useful output throughput without unacceptable KV-cache cost. The experiment can return a negative result.

Compare three configurations in the same vLLM build:

| Configuration | Purpose |
|---|---|
| Current default size-selection rule, with a common capture ceiling of 256 | Strong practical baseline; preserve its resolved sizes in the output. |
| Powers-of-two grid | Coarse diagnostic, not the main baseline. |
| Coarse grid plus the measured batch sizes | Manual workload-matched configuration; not an implemented automatic selector. |

Use one NVIDIA GPU, Qwen2.5-1.5B in BF16, TP=1, max 128 requests, 128 real-text prompt tokens and exactly 128 generated tokens. WikiText-2 test paragraphs supply the prompts. All configurations receive the same immutable token IDs, scheduler settings and 85% total GPU-memory budget. Prefix caching is disabled. We force fixed generation length to keep request lifetimes comparable; this is a controlled workload, not a production trace.

Batch sizes are 24, 31, 48, 63, 80, 95, 112 and 127. Some align with the current default stride and others fall immediately below a captured size. These are deliberate sensitivity cases, not a representative workload distribution. The matched grid is given the batch sizes in advance; this pilot is an upper-bound-style diagnostic for workload matching, not a held-out evaluation of an adaptive policy.

## What the experiment actually measures

- Wall time of completed fixed-batch generation, including prefill and host work; useful output tokens/s, three warmed repetitions per batch.
- Effective graph mode and capture-size list; actual worker allocated/reserved memory, KV-block count and nominal token capacity when exposed by the runtime.
- Output-length validity and output-token hashes. Inspect any hash differences before claiming equivalence.
- A separate optional profiler run to check actual graph replay and execution gaps. Profiled timings never enter the performance plot.

The plotting script displays all observed points and the mean, with a zero-based throughput axis. It does not fabricate error bars or treat three within-process samples as independent confidence intervals. `nominal_next_graph` is explicitly a configuration-derived proxy; it is not a measured padding trace.

Do not interpret the GPU memory fraction setting as GPU utilization. Do not call allocated memory “graph memory”; it includes other allocations. CUDA graph pools can share storage, so graph-count × assumed per-graph cost is not a valid memory measurement.

## Requirements and run procedure

Use an otherwise idle NVIDIA GPU with BF16 support, preferably at least 24 GB for this planned setup. The exact driver, CUDA and vLLM build must be selected and pinned after the supplied GPU is inspected. Use one environment for all configurations. Required packages are `vllm`, `datasets`, `transformers`, `huggingface_hub`, and `matplotlib`; do not blindly replace a working server's GPU packages. Internet access is needed once for public model/data downloads. The script records installed packages and immutable model/data revisions.

From this directory, after the GPU environment is ready:

```bash
python pilot.py prepare --out prepared
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture default --out runs/default > default.log 2>&1
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture coarse --out runs/coarse > coarse.log 2>&1
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture matched --out runs/matched > matched.log 2>&1
python analyze.py runs/default runs/coarse runs/matched --out figures
```

These commands run sequentially. Each process captures its graphs independently. Initialization is excluded from throughput timing and saved separately; it is not a fair cold-start comparison because model/compiler caches may differ. An interrupted/failed run remains failed; do not insert zero or synthetic measurements. Use new output directories for reruns.

For the selected batch-63 mechanism check, after the timing comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture default --profile --out profiles/default > profile-default.log 2>&1
CUDA_VISIBLE_DEVICES=0 python pilot.py run --prepared prepared --capture matched --profile --out profiles/matched > profile-matched.log 2>&1
```

Inspect the traces and engine logs to confirm the intended graph sizes are actually replayed. Runtime configuration/resource access is version-sensitive and must be checked on the supplied GPU before the full sweep. If an API fails, preserve the failure and update the script for that pinned runtime; do not silently substitute an estimate.

## Interpretation decided before seeing results

1. A gain only over the coarse grid does not establish an opportunity over current serving practice.
2. A gain over the current default must be repeatable, use complete outputs, and have a plausible mechanism in the trace. Repeat the configuration order in reverse before treating a small difference as real.
3. If current defaults tie or win, report that this small-model regime does not support the proposed improvement. Do not select only favourable batch sizes for the PPT.
4. If gains require reduced KV capacity, report the tradeoff. A short-context pilot may not reveal its cost; longer-context/load sweeps are required before a memory-aware claim.
5. This offline pilot cannot establish p99 TTFT/ITL preservation, workload-shift robustness, higher achieved SM/HBM utilization, or benefits on H100/B200. Those remain the next experiments.

The script performs real autoregressive generation with identical length controls. Cost scales with the number of configurations × repetitions × batch cases × generated tokens, plus model loading and graph capture. Memory includes weights, KV cache, graph pools and workspaces. No runtime or gain estimate is promised before the GPU is known.

## Sources for the setup

- [vLLM CUDA-graph design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [vLLM LLM API and worker RPC](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/)
- [vLLM profiler configuration](https://docs.vllm.ai/en/stable/api/vllm/config/profiler/)
- [vLLM cache configuration](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [Qwen2.5-1.5B official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [NVIDIA's graph-size tuning report](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html)

Future PPT charts must cite the raw run files, GPU SKU, model revision, vLLM version, settings and measured repetitions. Published authors' results remain separately labelled.
