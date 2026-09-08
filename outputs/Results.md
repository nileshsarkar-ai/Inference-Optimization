# Inference Optimization measured results

All experiments used the supplied A100 PCIe 40 GB. This report distinguishes successful execution from evidence of an improvement.

## Scheduling result

The context-budget policy failed the predeclared improvement criterion in every comparison. Its SLO goodput is 20.0–24.0% lower than current vLLM and 6.6–11.5% lower than the fixed-32 control across the three paired repetitions. First-token tail latency increases. Stream gaps decrease, showing a tradeoff rather than an overall improvement.

| Policy | SLO goodput, req/s | Output tokens/s | p99 TTFT, seconds | p99 stream gap, seconds | Mean NVML busy, % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current vLLM | 2.588 | 766.629 | 13.705 | 0.565 | 99.459 |
| Fixed limit 32 | 2.202 | 665.293 | 14.989 | 0.308 | 99.672 |
| Context budget | 2.013 | 608.318 | 23.894 | 0.302 | 99.781 |

These are arithmetic means of three independent engine-run metrics. In particular, the latency columns average the three per-run p99 values; they are not pooled p99 values or confidence intervals.

All policies keep the GPU nearly continuously busy. The candidate even has slightly higher average NVML busy time while completing less useful work. This directly illustrates why maximizing that percentage alone is an inadequate objective. Achieved SM utilization was not measured.

## Execution and controls

The complete scheduling study contains 2,160 requests: 240 requests × three policies × three independent engine repetitions. All return exactly 128 output tokens. The same saved arrival trace is used for each policy within a repetition; policy order rotates between repetitions. TTFT is measured from scheduled arrival and includes time waiting in the external admission gate. The frozen success criterion requires at least 5% higher SLO goodput than both controls in each repetition, without worse p99 TTFT or p99 stream gap.

The controlled trace consists of 96 short prompts, 48 long prompts and 96 short prompts. Input lengths are 128 and 2,048 tokens, generated from actual WikiText-2 test text. Arrivals have exponential inter-arrival times averaging 8 requests/s. Targets are TTFT ≤2 seconds and mean TPOT ≤100 ms. These declared targets are not universal production service requirements.

The context policy reserves input length plus the known 128-token output budget under a 20,000-token limit, with two-second age protection. This estimate is not actual KV-cache memory. The fixed control permits at most 32 unfinished admitted requests. The principal baseline submits requests directly to current vLLM continuous batching.

Raw-record checks confirmed trace/request identity, output lengths and hashes, timestamp ordering, and exact SLO accounting for every request. Output hashes differ for 95–130 of 240 requests in the nondefault comparisons. Valid output length does not establish semantic-quality equivalence.

## Three-model capture comparison

Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B each completed 72 timed calls on the same A100, comparing current-default, matched and coarse graph capture sets. Inputs and outputs each contain 128 tokens. Eight fixed batch sizes and three warmed repetitions are used per configuration.

Matched settings differ from default by approximately −0.6% to +0.3%. Coarse settings lose up to 3.8%. Static manual matching does not establish a useful improvement over current defaults. Repetitions within a configuration share an engine process and do not provide independent-run confidence intervals.

For Gemma 4 12B, throughput falls 10.1% between batches 95 and 127. Engine logs reach 99.7% KV occupancy, including warmup. This is an association rather than a causal diagnosis, and KV occupancy is not GPU compute utilization.

Older Qwen2.5-1.5B and Qwen2.5-7B runs remain supplementary implementation controls. They are not presented as current-generation models in the main PPT.

## Burst-recovery diagnosis

Post-hoc analysis of the saved requests found the same pattern in all nine policy runs: all 96 initial short requests meet both latency targets, but none of the 96 short requests after the long-context burst do. Only 4–9 of 48 long requests pass. The candidate waits approximately 20–22 seconds after the final arrival to finish the workload, versus 10–12 seconds for current vLLM. Nearly the same number of requests pass, so much of the goodput difference reflects the longer full-run denominator. These are descriptive findings from existing records, not a new GPU experiment or a sustainable-capacity estimate.

The 20,000-token gate permits at most nine simultaneously admitted all-long requests (20,000 divided by 2,176), compared with 32 in the fixed control. Admitted requests can still queue inside vLLM. The comparison changes restriction, ordering and age protection together, so it cannot isolate their causal effects. See `scheduling_results/analysis/phase_diagnostics.json` and `Methodology_Assessment.md`.

## Research implication

Current vLLM is the strongest measured goodput control. This simple context-budget setting does not validate an improved scheduler. The open question is whether a decision informed by measured service cost and latency slack can preserve useful concurrency while managing a long-context burst. The one eventual experiment compares a frozen adaptive rule against current vLLM and a calibrated simple limit on unseen burst traces at several predeclared arrival rates. Calibration and evaluation data remain separate, with matching fairness conditions and a version without the adaptive decision to isolate its effect.

The data covers one GPU and a limited controlled workload. It does not establish steady-state serving capacity, model-quality equivalence, multi-worker routing or transfer beyond LLM inference. No statistical-significance claim is made.

## Reproduction and sources

- The frozen scheduling protocol and scripts are in `scheduling_validation/` and exact executed copies in `scheduling_results/reproduction/`.
- Full records, per-run summaries, package versions, GPU metadata and model/data revisions are in `scheduling_results/`.
- Modern capture scripts are in `model_suite/`, records in the three `experiment_data_*` folders, and aggregate data in `current_model_summary/`.
- Scientific plots are in `figures/` and `scheduling_results/analysis/`, with PNG and PDF formats and source footers.
- `backup_receipts/` contains local/remote SHA256 verification records. Failed setup attempts and detached-run logs are also preserved.
- Paper references and interpretation boundaries are in `References.md` and the PPT speaker notes.

Scheduling W&B runs:

- [Independent repetition 0](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility/runs/bqi747dz)
- [Independent repetition 1](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility/runs/8uxhnbwe)
- [Independent repetition 2](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility/runs/jxmojkaz)
