# Inference Optimization: useful GPU utilization through inference scheduling

## Problem statement

**Can better admission decisions increase SLO goodput during bursts of long prompts? SLO goodput is the number of requests completed within latency targets per second.**

The comparison uses the same GPU and model, with current vLLM continuous batching and a calibrated simple admission limit as controls. The workload changes from short prompts to a burst of long prompts and back to short prompts.

The next hypothesis is that estimated processing cost, system state and remaining latency budget can guide admission to increase SLO goodput beyond both controls. Improvement must include tail-latency and long-request fairness checks.

LLM inference on one A100 40 GB is the first test bed. Request and stage routing, and inference beyond LLMs, belong to the wider project scope. The current experiments do not establish transfer to those settings. The context-budget gate is a simple experimental candidate, not a claimed new SOTA algorithm.

## Meaning of useful utilization

The objective in this investigation is more requests meeting their latency SLOs per second on the same GPU. NVML's GPU utilization measures sampled time during which a kernel executes. It does not measure achieved SM efficiency, FLOP efficiency or useful work. A nearly continuously busy GPU can still deliver poor tail latency. [NVIDIA NVML definition](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html).

We measure output tokens/s and **SLO goodput**: requests meeting both declared latency targets divided by the full replay-and-drain time. Latencies include admission waiting from the request's scheduled arrival. Occupied VRAM alone is not a benefit.

## What SOTA already does

There is no universal SOTA GPU-utilization percentage. Results depend on workload, hardware, service targets and baseline implementation.

| Prior work | Established approach and evidence | Implication |
| --- | --- | --- |
| [SOLA, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf) | Uses request and system state. Figure 1 reports 65% to 98% SLO attainment for Llama3-70B on four A100 GPUs, ShareGPT at 4.6 requests/s. | State-aware scheduling already exists. |
| [DuetServe, ICML 2026](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) | Adaptively shares SM resources between prefill and decode, with look-ahead execution. AzureCode Qwen3-8B on one H100: 12.43 to 13.57 completed requests/s at 16 offered requests/s, relative to SGLang-Default. | Resource interference and execution choices matter. Its baselines differ from our engine version. |
| [NVIDIA Dynamo, current documentation](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing) | Routes using reusable cache state and projected live load. | Cache- and load-aware routing is existing practice. Our admission test uses one engine. |
| [Prism DLRM, NSDI 2025](https://www.usenix.org/system/files/nsdi25-yang.pdf) | Separates CPU/GPU-heavy recommendation stages. Figure 17 reports 5–9× GPU-node goodput with added CPU nodes. The highest Model-XS case uses MIG. | GPU underuse can originate outside GPU kernels, including in non-LLM inference. |
| [NVIDIA CUDA-graph tuning, August 2026](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html) | Reports graph-padding/memory tradeoffs and proposes automatic size selection from concurrency logs. | This engineering report provides a reference for the capture-size tradeoff. |

Also relevant: [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf), [Aegaeon](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf), and the distinct [Prism LLM memory-ballooning system, OSDI 2026](https://www.usenix.org/conference/osdi26/technical-sessions). These studies cannot be ranked using their raw headline gains across different setups.

## Hypothesis and execution study

**Next hypothesis:** admission informed by estimated processing cost, system state and remaining latency budget may improve SLO goodput during a workload shift. **Competing explanation:** conservative admission may reduce useful concurrency, and current vLLM may already make better decisions.

The first experiment compared current-default, manually matched and coarse CUDA-graph capture sets on Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B on the same A100 40 GB. Each model has 72 timed calls: eight fixed batch sizes, three configurations and three warmed repetitions. Inputs and outputs each contain 128 tokens. Precision is BF16, engine vLLM 0.28.0. Initialization and separate profiling are excluded.

Across 216 measurements, matched settings differ from default by approximately −0.6% to +0.3%. Coarse settings lose up to 3.8%. Gemma 4 12B throughput falls 10.1% between batches 95 and 127, while its logs reach 99.7% KV occupancy. This association does not isolate a causal mechanism. The comparisons establish configuration sensitivity, but no useful improvement from manual capture matching. Repetitions are within-process for each configuration.

## Scheduling validation

The second experiment compares current vLLM submission, a fixed FIFO limit of 32 unfinished requests, and an oldest-fitting context-budget gate reserving at most 20,000 estimated tokens with two-second age protection. Cost is input length plus the known 128-token output budget, not measured KV bytes.

Each trace contains 96 short prompts, 48 long prompts and 96 short prompts: 128/2,048 input tokens and 128 output tokens. Exponential arrivals average eight requests/s. Three independent engine processes use fixed seeds and rotate policy order. Every policy replays the same trace within each repetition, giving 2,160 generated requests overall.

Declared targets: TTFT ≤2 seconds and mean TPOT ≤100 ms. Success requires at least 5% higher SLO goodput than both controls in every paired repetition, without worsening p99 TTFT or p99 stream gap. This practical threshold is not a significance test. The frozen protocol is in scheduling_validation/PROTOCOL.md. Execution status and final measured outcomes are recorded separately in experiment_status.json and the results report.

## What the completed test exposes

Near-100% GPU busy time coexisted with missed latency targets, while the tested context-budget gate reduced SLO goodput by 20–24% relative to default vLLM. We therefore do not yet have a better scheduler. The research progression is an observed problem, measured baselines, a tested simple hypothesis, rejection of that setting and a better-controlled next hypothesis. The negative result concerns this gate and its settings; it does not rule out every size-based policy. [Measured comparison](scheduling_results/analysis/validation.json).

A post-hoc analysis of the saved request records found that all 96 initial short requests meet the declared targets, while all 96 short requests after the long-context burst miss them, in every policy and repetition. The context-budget candidate also takes longer to drain the workload. This establishes a reproducible burst-recovery problem under the tested conditions. It does not establish that scheduling can eliminate the misses at the offered load. [Phase analysis](scheduling_results/analysis/phase_diagnostics.json).

## One eventual experiment

Run one controlled comparison of an adaptive admission rule against current vLLM and a calibrated fixed admission limit. Use separate data to diagnose queueing and processing cost, identify useful system-state signals, select baseline settings, and calibrate the candidate. Freeze the rules before replaying unseen burst traces across a predeclared range of arrival rates on the same model and GPU. Use the same fairness rule for controlled admission comparisons and compare a version without the adaptive decision to isolate what it adds.

Measure SLO goodput and output throughput, phase-specific attainment, external waiting and long-request tail latency. Preserve complete output accounting, report any drops, and use balanced independent engine repetitions. Keep detailed GPU profiling separate from timing if claiming changes in hardware efficiency. Latency targets and evaluation conditions must be fixed before evaluation. The earlier 8-request/s test alone does not establish sustainable serving capacity.

The next investigation will measure whether the policy increases SLO goodput and identify where it succeeds or fails.

Research and source check: 8 September 2026. Exact model/dataset revisions, package versions, traces, output token IDs, code and measurements are saved locally and tracked in [W&B](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility).
