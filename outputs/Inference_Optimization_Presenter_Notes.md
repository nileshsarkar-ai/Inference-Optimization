# Inference Optimization — Presenter notes

Extracted directly from the latest 22-slide PowerPoint on 9 September 2026. The notes retain their original wording and citations. Slide headings follow presentation order.

## Slide 1 — Inference Optimization

Our project is Inference Optimization. The goal is to improve how GPU resources support complete inference while meeting response-time targets. We have completed two studies on an A100 40 GB: CUDA graph configuration and request admission during a long-prompt burst.

Neither study establishes a better optimization algorithm. They give us measured baselines, unsuccessful settings and a reproducible workload. We now extend this work by investigating where capacity remains unused and whether a targeted change can recover some of it.

The presentation distinguishes published research, self-reported industry evidence, our own measurements and the next hypothesis. Admission remains one possible mechanism. Profiling will guide the choice.

---

## Slide 2 — Research question

The broad research question is which limits to GPU utilization we can reduce to improve LLM throughput while meeting latency targets. Our initial admission question remains part of this: can better admission decisions improve SLO goodput during long-prompt bursts?

The new extension investigates the cause of a performance limit before choosing its remedy. Possible mechanisms include scheduling and batching, coordination between CPU and GPU, or execution inside a kernel. We have not selected a winning mechanism.

Our hypothesis is that a measured bottleneck can sometimes be reduced enough to improve complete serving performance. An alternative outcome is that the workload already reaches a memory, compute or latency limit that the chosen change cannot improve.

SLO goodput counts requests meeting both targets divided by replay and drain duration. It remains an outcome measure alongside output tokens per second. Hardware profiling is needed to explain a claimed change in GPU efficiency.

Research scope: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Research_Question.md

---

## Slide 3 — Previous Literature Review

This section contains results from other researchers. For each paper, I will explain the objective, the experimental setup, the axes and the result.

Their results provide background for our investigation. They are not results from our GPU, and their headline gains cannot be compared directly because the hardware, models and workloads differ.

The section also includes Tensormux and TensorPath, which are industry projects rather than peer-reviewed papers. Their reported results have narrower meanings than a general GPU-utilization improvement.

---

## Slide 4 — What existing inference systems optimize

The literature shows several different places where inference can be improved. SOLA studies request scheduling and the amount of work performed in each iteration. DuetServe studies how prompt processing and output generation share GPU execution resources. Prism studies how recommendation inference uses CPU and GPU resources across machines.

We selected these examples to understand why a GPU utilization problem may come from scheduling, execution overlap or another part of the system. This is a map of selected approaches, not a claim that one system wins every benchmark.

The next three slides each show an original published figure with its own conditions.

Sources: SOLA https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf
DuetServe https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf
Prism https://www.usenix.org/system/files/nsdi25-yang.pdf

---

## Slide 5 — SOLA: scheduling to meet latency targets

SOLA tries to increase the proportion of requests meeting response-time targets. It uses request and system state to change execution order and work per iteration.

Read the two panels from left to right: the paper's vLLM baseline and SOLA. Each point is a request. Moving right means waiting longer for the first token. Moving up means more time per output token. The shaded lower-left region meets both targets.

In this Llama3-70B test on four A100 GPUs, the reported attainment increases from 65% to 98%. This supports investigating state-aware scheduling, but it does not establish the same gain on our model or newer vLLM version.

Source, Figure 1: https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf

---

## Slide 6 — DuetServe: sharing GPU resources across inference stages

DuetServe aims to keep token generation progressing while prompt processing also uses the GPU. It adaptively shares execution resources between these phases.

The horizontal axis is the incoming request rate, called QPS. The vertical axis is the rate of completed requests. The orange curve is DuetServe. The panel uses AzureCode requests and Qwen3-8B on one H100 80 GB.

At sixteen incoming requests per second, section 5.2 reports 13.57 completed requests per second for DuetServe and 12.43 for SGLang-Default. The 9.2 percent difference is our calculation from those reported values. This is throughput, not an SLO-attainment percentage. The paper also discusses a first-token versus later-token latency tradeoff.

Source, Figure 6 and section 5.2: https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf

---

## Slide 7 — Prism: recommendation inference across CPU and GPU nodes

Prism shows that inference efficiency also matters outside language models. Recommendation models can need substantial CPU work and memory, which limits how much of a GPU node can be used.

The graph compares two recommendation models. Different bars represent different deployments. In Prism's labels, the two numbers count CPU-side and GPU-side inference instances. Taller bars mean more goodput under the paper's twenty-five-millisecond latency condition.

The reported five-to-nine-times gains use added CPU nodes. The nine-times case also partitions GPUs with MIG. Therefore this is evidence about resource balance, not a claim that unchanged total hardware became nine times faster. It does not validate our admission rule beyond language models.

Source, Figure 17 and section 5.3: https://www.usenix.org/system/files/nsdi25-yang.pdf

---

## Slide 8 — Tensormux: routing across inference replicas

Tensormux works above inference engines and distributes requests among replicas. Its commercial platform describes scaling and cache features, while the public gateway implements routing, failover and observability and explicitly excludes engine-level batching and GPU scheduling.

The table reproduces the company's reported first-token latency values. Lower is better. All five fit its declared target, with nearly equal throughput. The uniform workload offers little routing advantage. Autoscaling was disabled. The report does not specify a hardware counter for its utilization percentage.

For our project, the relevant idea is to examine where requests wait and how work is assigned. This study does not demonstrate a before/after recovery of unused capacity, and it is not a benchmark against our single A100 setup. The website's forty-percent GPU-spend figure is labeled a directional target, not an established improvement.

Benchmark, Table 1: https://www.tensormux.com/blogs/sla-benchmark
Platform claims: https://www.tensormux.com/
Open-source scope: https://github.com/KrxGu/Tensormux

---

## Slide 9 — TensorPath: optimizing GPU kernels

TensorPath is a separate project in the Tensormux ecosystem. Its Forge component generates Triton kernels and checks correctness and execution time against a reference. RMSNorm is an operation that normalizes intermediate model activations.

The displayed speedup applies to a single RMSNorm implementation against PyTorch eager, on the listed GPU, shape and precision. It does not imply the same speedup for a full model. The repository states that runtime integration remains future work.

Our graph experiment measured complete generation using the serving engine, so the experiment units differ. We would first identify an expensive operation in the actual engine and use its existing optimized implementation as the baseline. A faster isolated kernel matters to this research only if integration improves complete inference without violating correctness and latency requirements.

TensorPath README, Forge results and integration scope: https://github.com/tensormux/Tensorpath

---

## Slide 10 — Our Experiments

From this point, the experiments and measurements are our own. We used one rented A100 40 GB GPU and saved the code, inputs, configurations and outputs.

There are two different studies. The first changes CUDA-graph capture sizes on five models. The second changes when requests enter vLLM during a burst, using Gemma 4 12B. They answer different questions and their throughput numbers should not be directly compared.

---

## Slide 11 — Our experimental system and performance measures

This is the system we actually used. All experiments ran on the same A100 PCIe 40 GB GPU, with BF16 precision and one GPU per model. The table records the installed software versions. CPU model and total host RAM were not reliably recorded, so we do not claim those specifications.

Throughput counts output tokens per second. SLO goodput counts requests that meet both response-time targets per second. In the scheduling study, the denominator includes replaying arrivals and finishing outstanding work.

GPU busy time comes from NVIDIA telemetry. It reports time with a kernel running, not achieved compute efficiency. More VRAM occupied is not itself a performance improvement.

Additional settings: NVIDIA driver 595.58.03, Transformers 5.16.1, engine memory fraction 0.85, maximum 128 sequences, 4096 batched tokens, prefix cache off and chunked prefill on. Capture maximum context is 512; scheduling maximum context is 4096.

Evidence: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Experiment_Inventory.md
Metric: https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html

---

## Slide 12 — Our experiment 1: CUDA graph configuration

We started with a bounded execution-setting experiment before designing a scheduler. CUDA graphs record GPU operations so they can be replayed. A captured size that does not fit the workload exactly can involve padding, while storing many graphs has a memory cost.

We compared the engine's default set, a coarse set and a set including our exact tested batch sizes. Everything else was held fixed within each model. Each request had 128 input and 128 output tokens. We timed eight batch sizes with three warmed repetitions for each configuration.

A fresh engine process was used per configuration, but configuration order was fixed. The repetitions within a configuration are not independent process repetitions. Timing includes prompt processing and host work. The separate profiler runs are excluded.

Why this experiment: it tests whether an accessible execution setting leaves a useful improvement before adding queueing logic.

Protocol: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/model_suite/README.md
Engineering background: https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html

---

## Slide 13 — Our experiment 2: admission during a long-prompt burst

The second study introduces changing arrivals and queueing, which the fixed-batch experiment does not test. We replay the same short-long-short request trace for each policy within a repetition.

Direct submission lets vLLM receive each request on arrival. Fixed 32 adds a FIFO gate. The candidate reserves input length plus the known 128-token output budget, with at most twenty thousand estimated tokens admitted. This is a token heuristic, not actual measured compute cost or live KV bytes.

Our targets are a first token within two seconds and mean subsequent-token time within one hundred milliseconds. All waiting counts from the scheduled arrival. Three independent engine processes each run all policies, with rotated policy order. Every request completes before moving to the next policy.

The declared success criterion requires at least five percent more goodput than both controls in every paired repetition, without worsening p99 first-token delay or p99 stream gap. A p99 is a ninety-ninth-percentile value.

Protocol: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_validation/PROTOCOL.md

---

## Slide 14 — Our Results

The following graphs are our measurements, taken from the saved experiment records. The CUDA-graph comparison comes first, followed by the request-admission comparison.

The chart points show actual repetitions. The lines or diamonds show calculated means. We will separate what was observed from what it suggests, and we will not describe the unsuccessful candidate as an improvement.

---

## Slide 15 — Our experiment 1 results: three current models

These three panels show Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B. The horizontal axis is the number of requests submitted in a fixed batch. The vertical axis is generated output tokens per second. Compare configurations within each panel because the vertical ranges differ.

Black shows vLLM's default capture sizes, blue the exact matched set and gray the coarse grid. Every recorded repeat appears as a point, and lines connect the means. Closely overlapping points are expected.

Matched settings differ from default by approximately minus 0.6 to plus 0.3 percent. This study does not establish a useful gain from manual matching. Coarse grids lose up to 3.8 percent. Gemma 12B throughput falls at the largest batches, but high KV occupancy in logs does not prove the cause.

The peaks label the best tested default batch for each model. They are not general model benchmarks or online serving capacity.

Measurements: https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/current_model_summary

---

## Slide 16 — Our experiment 1 results: two earlier model controls

We also retained our earlier Qwen2.5 controls. They use the same batch-size and capture-size comparison, with all 144 recorded timing calls shown.

Read the axes and colors the same way as the previous slide. The smaller model achieves more tokens per second in these tests, but the purpose here is the configuration comparison within a model. These are older controls and are not presented as the newest available models.

The five-model coverage applies to the CUDA-graph study. It does not mean that the scheduling candidate has been evaluated on five models. Scheduling was tested on Gemma 4 12B only.

Raw records: https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/experiment_data
https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/experiment_data_qwen7b_complete

---

## Slide 17 — Our experiment 2 results: throughput, latency and GPU activity

These are the four performance measures for our scheduling comparison. Each point represents a policy in one independent engine repetition, and each diamond is its mean. Some points overlap.

The upper-left panel is SLO goodput: requests meeting both latency targets per second, where higher is better. The upper-right is total generated tokens per second. The lower-left is first-token tail latency, where lower is better. The lower-right is the GPU-busy percentage.

The context-budget rule performs worse on SLO goodput and output throughput and increases first-token delay. Yet GPU busy time is near one hundred percent for every policy. This shows why that percentage alone is an inadequate optimization target.

These are timed burst workloads. Do not compare the tokens-per-second values directly against the fixed-batch graphs as a performance regression because the inputs, arrival behavior and timing procedure differ.

Data: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/validation.json

---

## Slide 18 — Our results: what the scheduling comparison means

This table gives the means behind the scheduling graphs. Default vLLM produces about 2.588 requests meeting the latency targets per second, compared with 2.013 for the context-budget rule. Across paired repetitions, the rule reduces SLO goodput by twenty to twenty-four percent. The latency column averages each run's p99; it is not a percentile pooled across all requests.

We established two observations under these test conditions. Near-one-hundred-percent GPU busy time coexists with requests missing their latency targets. Also, our first context-budget admission rule performs worse than default vLLM. GPU activity alone therefore does not tell us how much timely service the system provides.

The phase analysis identifies the burst-recovery problem. All ninety-six initial short requests meet the targets, while none of the ninety-six short requests after the long burst meet them in any policy or repetition. Similar numbers pass across policies, but the candidate takes longer to drain the workload, which explains much of its lower SLO goodput.

These results reject this tested setting. They do not isolate the causal bottleneck or establish sustainable capacity at eight requests per second.

Results: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/validation.json
Phase analysis: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/phase_diagnostics.json

---

## Slide 19 — How our experiments extend this work

The two completed experiments examine different levels of serving. The graph study tested one execution configuration, while the admission study tested how requests enter vLLM. Neither is a direct reproduction of Tensormux or TensorPath.

The connection is methodological. Tensormux motivates examining request distribution and waiting. TensorPath motivates examining operation execution cost. Our work starts with a fixed model on one GPU so changes can be evaluated against the same engine and workload.

The graph study found no established benefit from manually matching capture sizes on the three current models. The context-budget gate reduced goodput by twenty to twenty-four percent even though NVML busy time stayed close to one hundred percent. This motivates investigating the bottleneck, but it does not prove that recoverable spare capacity exists under every tested load.

The earlier forty-seven-percent example was illustrative. We did not measure a fifty-three-percent overhead budget. NVML busy time measures kernel activity, while achieved compute and memory throughput require other counters. These percentages have different denominators and must not be subtracted or compared as if they were the same quantity.

Our results: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Results.md
Industry evidence: https://www.tensormux.com/blogs/sla-benchmark and https://github.com/tensormux/Tensorpath
NVIDIA definition: https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html

---

## Slide 20 — Next experiment: recoverable GPU underutilization

This is one study with a diagnostic phase followed by a controlled intervention. We will retain the existing model and burst structure first, and sweep arrival rates below and near measured serving capacity. Low demand alone can leave a GPU idle, so we must determine whether pending work exists during idle intervals.

Use Nsight Systems to relate CPU work, launches and GPU execution. Use Nsight Compute on representative expensive kernels to examine compute and memory limits. Collect these profiles separately from final timing because profiling can change execution behavior. Do not add overlapping CPU and GPU times and call the sum removable overhead.

Select one mechanism based on that evidence. If scheduling limits ready work, an admission or batching change is a candidate. If execution dominates, a kernel or runtime change is a candidate. Cost, current state and remaining latency budget are still possible admission signals, not a committed solution. A routing intervention requires multiple replicas and would be a later extension.

Use separate calibration traces, freeze the candidate and baseline settings, and then run unseen traces at predeclared loads in balanced independent engine repetitions. Compare the current engine, the candidate and a version with its targeted change disabled. If admission is selected, include a calibrated fixed limit and matched fairness rules. Keep model revision, precision and output accounting fixed, check correctness and count all waiting and requests.

We seek a repeatable improvement in throughput and SLO goodput within the declared latency targets, together with evidence that the identified bottleneck decreases. A higher busy percentage alone is insufficient. No gain, or a gain restricted to one load regime, is a valid outcome. Any future generalization claim requires additional models and hardware.

Profiling references:
https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
Research protocol: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Research_Question.md

---

## Slide 21 — References and reproducibility records

These links identify the original published figures, official engineering documentation and our reproducibility records. The three papers discussed in detail are SOLA, DuetServe and the recommendation-serving Prism system.

NVIDIA's graph-tuning article is an engineering report, not a peer-reviewed paper. The vLLM and NVML links document implementation behavior and metric definitions. The repository contains model and dataset revisions, the executed code, raw requests, measurements, plots and the full bibliography.

The extension now asks which measured scheduling or execution bottlenecks are recoverable. The next slide supplies the additional industry and profiling references.

Full references: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/References.md

---

## Slide 22 — Industry sources and profiling references

These sources cover the new extension. The Tensormux website describes the commercial platform. Its benchmark supplies the reported serving measurements, and the gateway repository states the narrower scope of the open-source component. TensorPath's README supplies the operation benchmark and its integration limitations.

The NVIDIA references define the profiling methods we propose using. None of these sources establishes that our illustrative fifty-three-percent gap exists or is removable.

Sources checked on 9 September 2026:
https://www.tensormux.com/
https://www.tensormux.com/blogs/sla-benchmark
https://github.com/KrxGu/Tensormux
https://github.com/tensormux/Tensorpath
https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
