# Scheduling validation protocol — fixed before execution

This is a test of a simple scheduling hypothesis, not a claimed new SOTA algorithm. The broader project studies useful GPU utilization across inference workloads. LLMs are the first test bed; this experiment does not establish transfer to other model types or multi-GPU routing.

## Question

Can a request admission policy that accounts for context demand improve SLO-compliant completed work over current vLLM continuous batching and a fixed concurrency limit during a long-context burst?

## Controlled settings

- One A100 PCIe 40 GB; Gemma 4 12B; BF16; vLLM 0.28.0; TP1.
- Same default graph configuration, chunked prefill, disabled prefix cache, 4,096-token context/batched-token limits, 128 maximum sequences and requested 0.85 GPU-memory fraction.
- Each trace has 240 requests: 96 with 128 input tokens, 48 with 2,048 input tokens, then 96 with 128 input tokens. Every request produces exactly 128 output tokens.
- Real WikiText-2 test text supplies contiguous token slices. Text may cross paragraph boundaries. This is a controlled workload with real text, not a production arrival trace.
- Exponential inter-arrival times at 8 requests/s, seeds 20260908–20260910. The same saved trace is replayed for every policy within each repetition.
- Three independent engine processes. Policy order rotates across repetitions: default/fixed/context; fixed/context/default; context/default/fixed. All requests drain between trials. Common warmup is excluded.

## Policies

1. **Current default:** submit each request at its arrival to vLLM's existing continuous-batching scheduler.
2. **Fixed limit:** an external FIFO admission queue permits at most 32 unfinished requests in the engine.
3. **Context budget:** an external queue reserves `input tokens + 128 output tokens` for each admitted request, with a 20,000-token budget. It admits the oldest fitting request. After a pending request has waited two seconds, newly available room is reserved for that oldest request before bypassing it. This tests a context-based resource estimate; it does not measure or control actual KV bytes directly.

The token budget is a declared conservative operating point motivated by the earlier Gemma control's roughly 25,833 engine-reported effective KV tokens. It is not tuned on this test. The different 4,096-token context limit means actual runtime memory must still be recorded. Known fixed output length is an experimental simplification.

## Outcomes and decision

- Count latency from scheduled arrival, including external admission waiting. Report timing of submission, first streamed output, completion and every stream chunk.
- A request meets the declared SLO if TTFT ≤ 2 seconds and average TPOT ≤ 100 ms. These are experiment targets, not universal service requirements.
- **SLO goodput:** number of requests meeting both targets divided by the full replay-and-drain duration.
- Also report completed output tokens/s, p99 TTFT, p99 TPOT, p99 stream gap, output validity and per-class latency. A stream chunk can contain multiple tokens; chunk gaps must not be mislabeled as exact token timestamps.
- Sample NVML GPU kernel-busy percentage, memory-busy percentage and occupied device memory every 200 ms during replay. Kernel-busy time does not measure achieved SM utilization or useful compute.
- **Predeclared success rule:** context policy must improve SLO goodput by at least 5% over both controls in each of the three paired repetitions, with no worsening of p99 TTFT or p99 stream gap in any paired run. All requested outputs must complete. Report failure or a tradeoff if the criterion is not met.

The 5% effect threshold is a practical criterion, not a statistical significance test. Three independent repetitions remain a small sample. Save all data and negative results. This finite, phased workload cannot establish steady-state capacity, quality equivalence or generalization.

## Run and reproduce

`queue_validation.py` is a detached, run-specific queue that waits for managed run `r_7a2c52f8`, then prepares immutable traces and runs repetitions 0–2. It reads existing W&B authentication from `/home/.saturatellm-auth/wandb.key`, outside the repository. For a fresh machine, adapt paths and the dependency ID deliberately or invoke the prepare/run commands directly.

```bash
python validate_scheduling.py prepare --out prepared
SATURATE_GROUP=scheduling-reproduction python validate_scheduling.py run --prepared prepared --out repeat-0 --repeat 0
```

Repeat with indices 1 and 2, using separate output directories. Valid W&B authentication and the pinned packages recorded with the original results are required. The analysis script expects the three `repeat-N` directories under one result folder.

The implementation is a small asyncio dispatcher around the actual installed vLLM streaming API. It does real autoregressive generation and checks request/output counts. Policy scans are O(n²) in the worst case for n queued requests; model computation dominates. Memory consists of model weights, runtime workspaces, graph pools and KV cache, plus the saved requests and streamed outputs.

## Primary background

- [vLLM streaming example](https://docs.vllm.ai/en/latest/examples/deployment/async_llm_streaming/)
- [vLLM CUDA-graph design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [SOLA: state-aware scheduling, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf)
- [DuetServe, ICML 2026](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)
- [NVIDIA NVML utilization definition](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)
