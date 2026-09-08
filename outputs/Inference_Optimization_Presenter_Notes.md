# Inference Optimization — Presenter notes

Extracted directly from the latest 18-slide PowerPoint on 9 September 2026. The notes retain their original wording and citations. Slide headings follow presentation order.

## Slide 1 — Inference Optimization

Our project is called Inference Optimization. We want better admission decisions to increase SLO goodput: the number of requests completed within their latency targets per second on the same GPU.

The completed tests give us a research progression. We observed a burst-recovery problem, measured default vLLM and a simple admission limit, and tested a context-budget rule. That rule performed worse, so we do not yet have a better scheduler.

Our next question is whether estimated processing cost, system state and remaining latency budget can guide better admission decisions. The presentation separates published results, our measurements and this untested next hypothesis.

---

## Slide 2 — Research question

Can better admission decisions increase SLO goodput during bursts of long prompts? SLO goodput means the number of requests completed within latency targets per second. In our finite replay experiment, the denominator includes the arrival period and the time required to finish all outstanding work.

Imagine short questions arriving, followed by long documents and then more short questions. The later questions may spend their latency budget waiting for the backlog to clear. Admission decides when and how many waiting requests enter the serving engine.

Our hypothesis is that estimated processing cost, system state and remaining latency budget can improve SLO goodput without worsening tail latency. System state could include queued work and available resources; the useful signals still need to be tested. The competing explanation is that an extra gate suppresses useful concurrency and delays work that vLLM handles more effectively.

Evidence: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Research_Question.md

---

## Slide 3 — Previous Literature Review

This section contains results from other researchers. For each paper, I will explain the objective, the experimental setup, the axes and the result.

Their results provide background for our investigation. They are not results from our GPU, and their headline gains cannot be compared directly because the hardware, models and workloads differ.

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

## Slide 8 — Our Experiments

From this point, the experiments and measurements are our own. We used one rented A100 40 GB GPU and saved the code, inputs, configurations and outputs.

There are two different studies. The first changes CUDA-graph capture sizes on five models. The second changes when requests enter vLLM during a burst, using Gemma 4 12B. They answer different questions and their throughput numbers should not be directly compared.

---

## Slide 9 — Our experimental system and performance measures

This is the system we actually used. All experiments ran on the same A100 PCIe 40 GB GPU, with BF16 precision and one GPU per model. The table records the installed software versions. CPU model and total host RAM were not reliably recorded, so we do not claim those specifications.

Throughput counts output tokens per second. SLO goodput counts requests that meet both response-time targets per second. In the scheduling study, the denominator includes replaying arrivals and finishing outstanding work.

GPU busy time comes from NVIDIA telemetry. It reports time with a kernel running, not achieved compute efficiency. More VRAM occupied is not itself a performance improvement.

Additional settings: NVIDIA driver 595.58.03, Transformers 5.16.1, engine memory fraction 0.85, maximum 128 sequences, 4096 batched tokens, prefix cache off and chunked prefill on. Capture maximum context is 512; scheduling maximum context is 4096.

Evidence: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Experiment_Inventory.md
Metric: https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html

---

## Slide 10 — Our experiment 1: CUDA graph configuration

We started with a bounded execution-setting experiment before designing a scheduler. CUDA graphs record GPU operations so they can be replayed. A captured size that does not fit the workload exactly can involve padding, while storing many graphs has a memory cost.

We compared the engine's default set, a coarse set and a set including our exact tested batch sizes. Everything else was held fixed within each model. Each request had 128 input and 128 output tokens. We timed eight batch sizes with three warmed repetitions for each configuration.

A fresh engine process was used per configuration, but configuration order was fixed. The repetitions within a configuration are not independent process repetitions. Timing includes prompt processing and host work. The separate profiler runs are excluded.

Why this experiment: it tests whether an accessible execution setting leaves a useful improvement before adding queueing logic.

Protocol: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/model_suite/README.md
Engineering background: https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html

---

## Slide 11 — Our experiment 2: admission during a long-prompt burst

The second study introduces changing arrivals and queueing, which the fixed-batch experiment does not test. We replay the same short-long-short request trace for each policy within a repetition.

Direct submission lets vLLM receive each request on arrival. Fixed 32 adds a FIFO gate. The candidate reserves input length plus the known 128-token output budget, with at most twenty thousand estimated tokens admitted. This is a token heuristic, not actual measured compute cost or live KV bytes.

Our targets are a first token within two seconds and mean subsequent-token time within one hundred milliseconds. All waiting counts from the scheduled arrival. Three independent engine processes each run all policies, with rotated policy order. Every request completes before moving to the next policy.

The declared success criterion requires at least five percent more goodput than both controls in every paired repetition, without worsening p99 first-token delay or p99 stream gap. A p99 is a ninety-ninth-percentile value.

Protocol: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_validation/PROTOCOL.md

---

## Slide 12 — Our Results

The following graphs are our measurements, taken from the saved experiment records. The CUDA-graph comparison comes first, followed by the request-admission comparison.

The chart points show actual repetitions. The lines or diamonds show calculated means. We will separate what was observed from what it suggests, and we will not describe the unsuccessful candidate as an improvement.

---

## Slide 13 — Our experiment 1 results: three current models

These three panels show Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B. The horizontal axis is the number of requests submitted in a fixed batch. The vertical axis is generated output tokens per second. Compare configurations within each panel because the vertical ranges differ.

Black shows vLLM's default capture sizes, blue the exact matched set and gray the coarse grid. Every recorded repeat appears as a point, and lines connect the means. Closely overlapping points are expected.

Matched settings differ from default by approximately minus 0.6 to plus 0.3 percent. This study does not establish a useful gain from manual matching. Coarse grids lose up to 3.8 percent. Gemma 12B throughput falls at the largest batches, but high KV occupancy in logs does not prove the cause.

The peaks label the best tested default batch for each model. They are not general model benchmarks or online serving capacity.

Measurements: https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/current_model_summary

---

## Slide 14 — Our experiment 1 results: two earlier model controls

We also retained our earlier Qwen2.5 controls. They use the same batch-size and capture-size comparison, with all 144 recorded timing calls shown.

Read the axes and colors the same way as the previous slide. The smaller model achieves more tokens per second in these tests, but the purpose here is the configuration comparison within a model. These are older controls and are not presented as the newest available models.

The five-model coverage applies to the CUDA-graph study. It does not mean that the scheduling candidate has been evaluated on five models. Scheduling was tested on Gemma 4 12B only.

Raw records: https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/experiment_data
https://github.com/nileshsarkar-ai/Inference-Optimization/tree/master/outputs/experiment_data_qwen7b_complete

---

## Slide 15 — Our experiment 2 results: throughput, latency and GPU activity

These are the four performance measures for our scheduling comparison. Each point represents a policy in one independent engine repetition, and each diamond is its mean. Some points overlap.

The upper-left panel is SLO goodput: requests meeting both latency targets per second, where higher is better. The upper-right is total generated tokens per second. The lower-left is first-token tail latency, where lower is better. The lower-right is the GPU-busy percentage.

The context-budget rule performs worse on SLO goodput and output throughput and increases first-token delay. Yet GPU busy time is near one hundred percent for every policy. This shows why that percentage alone is an inadequate optimization target.

These are timed burst workloads. Do not compare the tokens-per-second values directly against the fixed-batch graphs as a performance regression because the inputs, arrival behavior and timing procedure differ.

Data: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/validation.json

---

## Slide 16 — Our results: what the scheduling comparison means

This table gives the means behind the scheduling graphs. Default vLLM produces about 2.588 requests meeting the latency targets per second, compared with 2.013 for the context-budget rule. Across paired repetitions, the rule reduces SLO goodput by twenty to twenty-four percent. The latency column averages each run's p99; it is not a percentile pooled across all requests.

We established two observations under these test conditions. Near-one-hundred-percent GPU busy time coexists with requests missing their latency targets. Also, our first context-budget admission rule performs worse than default vLLM. GPU activity alone therefore does not tell us how much timely service the system provides.

The phase analysis identifies the burst-recovery problem. All ninety-six initial short requests meet the targets, while none of the ninety-six short requests after the long burst meet them in any policy or repetition. Similar numbers pass across policies, but the candidate takes longer to drain the workload, which explains much of its lower SLO goodput.

These results reject this tested setting. They do not isolate the causal bottleneck or establish sustainable capacity at eight requests per second.

Results: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/validation.json
Phase analysis: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/scheduling_results/analysis/phase_diagnostics.json

---

## Slide 17 — Research progression and next hypothesis

Our research progression is: problem observed, baselines measured, a simple hypothesis tested, the tested rule rejected, then a better-controlled next hypothesis.

We do not yet have a better scheduler. We have a reproducible burst-recovery problem and evidence that this simple size-based gate is insufficient under the tested conditions. The next question is whether admission decisions informed by estimated processing cost, system state and remaining latency budget can increase SLO goodput beyond both default vLLM and simpler admission limits.

We propose one controlled comparison. Use separate data to estimate processing cost, identify useful state signals and calibrate a strong simple admission limit. Freeze the candidate and controls before testing on unseen burst traces at predeclared arrival rates, with independent engine repetitions on the same GPU and model. Include a version with adaptation removed to test what the adaptive decision adds.

All requests and waiting time must count. Keep fairness conditions matched and check tail latency for long requests as well as SLO goodput. Report any dropped requests. The current gate admits at most nine all-long requests and changes throttling, ordering and age protection together, so its failure does not identify the best replacement or rule out every size-based policy.

The outcome remains open: the adaptive candidate may help, fail, or reveal a load beyond what this hardware can serve within the targets. One model and workload do not establish generalization or improved GPU compute efficiency.

Assessment: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/Methodology_Assessment.md

---

## Slide 18 — References and reproducibility records

These links identify the original published figures, official engineering documentation and our reproducibility records. The three papers discussed in detail are SOLA, DuetServe and the recommendation-serving Prism system.

NVIDIA's graph-tuning article is an engineering report, not a peer-reviewed paper. The vLLM and NVML links document implementation behavior and metric definitions. The repository contains model and dataset revisions, the executed code, raw requests, measurements, plots and the full bibliography.

The final research message is that we have measured a burst-recovery failure and rejected one simple admission rule. The next hypothesis tests whether processing cost, system state and remaining latency budget can guide admission to increase SLO goodput beyond default vLLM and calibrated simple limits. The algorithm and any improvement remain to be established.

Full references: https://github.com/nileshsarkar-ai/Inference-Optimization/blob/master/outputs/References.md
