# Inference Optimization: investigating recoverable GPU underutilization

## Problem statement and extension

**Which limits to GPU utilization can we reduce to improve LLM throughput within latency targets?**

The completed studies test CUDA graph configuration and admission during bursts of long prompts. They remain the empirical starting point. The extension investigates the cause of a performance limit before selecting one scheduling or execution change. Neither the existence nor the magnitude of recoverable capacity has been established.

The initial admission question remains: can better admission decisions increase SLO goodput during long-prompt bursts? Estimated processing cost, system state and remaining latency budget remain possible admission signals. They are one route within the broader investigation, rather than a committed solution.

Start with a fixed model on an A100 40 GB and current vLLM as the baseline. Keep the model revision, precision and workload fixed across conditions. Multiple-replica routing and inference beyond LLMs remain possible extensions; the current evidence does not establish transfer.

The illustrative 47% utilization / 53% remainder is not a recorded experimental baseline. A percentage must identify its counter and denominator. NVML busy time, achieved compute throughput and fleet utilization cannot be interpreted interchangeably.

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

## Completed execution study and admission hypothesis

**Admission-specific hypothesis:** admission informed by estimated processing cost, system state and remaining latency budget may improve SLO goodput during a workload shift. **Competing explanation:** conservative admission may reduce useful concurrency, and current vLLM may already make better decisions.

The first experiment compared current-default, manually matched and coarse CUDA-graph capture sets on Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B on the same A100 40 GB. Each model has 72 timed calls: eight fixed batch sizes, three configurations and three warmed repetitions. Inputs and outputs each contain 128 tokens. Precision is BF16, engine vLLM 0.28.0. Initialization and separate profiling are excluded.

Across 216 measurements, matched settings differ from default by approximately −0.6% to +0.3%. Coarse settings lose up to 3.8%. Gemma 4 12B throughput falls 10.1% between batches 95 and 127, while its logs reach 99.7% KV occupancy. This association does not isolate a causal mechanism. The comparisons establish configuration sensitivity, but no useful improvement from manual capture matching. Repetitions are within-process for each configuration.

## Scheduling validation

The second experiment compares current vLLM submission, a fixed FIFO limit of 32 unfinished requests, and an oldest-fitting context-budget gate reserving at most 20,000 estimated tokens with two-second age protection. Cost is input length plus the known 128-token output budget, not measured KV bytes.

Each trace contains 96 short prompts, 48 long prompts and 96 short prompts: 128/2,048 input tokens and 128 output tokens. Exponential arrivals average eight requests/s. Three independent engine processes use fixed seeds and rotate policy order. Every policy replays the same trace within each repetition, giving 2,160 generated requests overall.

Declared targets: TTFT ≤2 seconds and mean TPOT ≤100 ms. Success requires at least 5% higher SLO goodput than both controls in every paired repetition, without worsening p99 TTFT or p99 stream gap. This practical threshold is not a significance test. The frozen protocol is in scheduling_validation/PROTOCOL.md. Execution status and final measured outcomes are recorded separately in experiment_status.json and the results report.

## What the completed test exposes

Near-100% GPU busy time coexisted with missed latency targets, while the tested context-budget gate reduced SLO goodput by 20–24% relative to default vLLM. We therefore do not yet have a better scheduler. The research progression is an observed problem, measured baselines, a tested simple hypothesis, rejection of that setting and a better-controlled next hypothesis. The negative result concerns this gate and its settings; it does not rule out every size-based policy. [Measured comparison](scheduling_results/analysis/validation.json).

A post-hoc analysis of the saved request records found that all 96 initial short requests meet the declared targets, while all 96 short requests after the long-context burst miss them, in every policy and repetition. The context-budget candidate also takes longer to drain the workload. This establishes a reproducible burst-recovery problem under the tested conditions. It does not establish that scheduling can eliminate the misses at the offered load. [Phase analysis](scheduling_results/analysis/phase_diagnostics.json).

## Industry work and its relation to our experiments

[Tensormux](https://www.tensormux.com/) describes an inference control plane above engines. Its [gateway](https://github.com/KrxGu/Tensormux) handles routing and reliability, and explicitly excludes engine-level batching and GPU scheduling. The website's GPU-spend percentage is a directional target. The [company benchmark](https://www.tensormux.com/blogs/sla-benchmark) is evidence of SLA compliance under a specified workload, with similar throughput across routing strategies. It does not measure a recovered 53% capacity gap.

[TensorPath](https://github.com/tensormux/Tensorpath) reports operation-level kernel results and identifies serving-runtime integration as unfinished. Our capture study evaluates complete generation through the engine. The relevant next step is to compare a selected change against the actual engine implementation and measure complete serving, rather than infer model-wide gains from an isolated operation.

## One eventual experiment

Run a diagnostic phase followed by one controlled intervention on the same model and GPU configuration. Sweep arrival rates below and near observed serving capacity while retaining the short/long/short workload structure. Separate idle time caused by low demand from gaps that occur with pending work. Profile CPU and GPU execution with [Nsight Systems](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), and use [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) on representative expensive kernels to distinguish compute and memory limits. Do not sum overlapping CPU and GPU times into an overhead budget.

Choose one mechanism from the profile. Admission or batching is a candidate if scheduling restricts ready work. A kernel or runtime change is a candidate if execution cost dominates. This decision belongs to the diagnostic phase. Use separate calibration traces, freeze the candidate and settings, and evaluate on unseen traces at predeclared loads with balanced independent engine repetitions.

Compare current vLLM, the candidate and a version with its targeted change disabled. For an admission intervention, also include a calibrated simple limit and match fairness conditions. Keep model revision, precision and request accounting fixed, check output correctness, include all waiting and report any dropped requests. Collect detailed profiles separately from timing runs.

Success requires a repeatable increase in output throughput and SLO goodput within fixed latency targets, supported by evidence that the identified bottleneck decreases. A higher busy percentage alone does not establish success. The result may be no improvement or a gain restricted to particular loads. Hardware and model generalization require additional evidence.

Research extension and industry sources checked: 9 September 2026. Completed experiment records and the frozen original protocol remain unchanged. Exact revisions, traces, output records and measurements are saved locally and tracked in [W&B](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility).
