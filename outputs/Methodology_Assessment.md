# Inference Optimization: research question and evidence assessment

The completed experiments support a bounded research investigation. They reject the tested context-budget policy. They do not demonstrate increased GPU compute utilization or establish that an adaptive scheduler will succeed.

## The problem we can defend

**Can inference scheduling complete more requests within latency targets on the same GPU when a burst of long prompts delays subsequent short requests?**

LLM inference is the first test bed. The project objective is more useful inference work per GPU. The measurable objective here is SLO goodput, with throughput, tail latency and fairness constraints. NVML's GPU percentage reports time with a kernel executing, so it cannot establish achieved compute efficiency. [NVIDIA's metric definition](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html).

The hypothesis is that measured processing cost and remaining time before a deadline might improve admission decisions. The competing explanation is that an extra gate reduces useful concurrency and delays work that vLLM already schedules effectively.

## What we did and learned

The CUDA-graph study tested three capture configurations over eight batch sizes on five models, with 360 timing calls. Manual matching showed no established improvement on the three current models. The tiny differences require independent, balanced engine runs before interpretation. The larger-batch Gemma throughput decline coincides with high KV occupancy, but the test does not isolate memory pressure as its cause. [Measured results](Results.md).

The scheduling comparison replayed a short/long/short prompt sequence through default vLLM, a fixed limit of 32 admitted requests and a 20,000-token budget. All policies used the same saved trace within each repetition. Three independent engine repetitions, rotated policy order, fixed outputs and inclusion of external waiting make this a useful controlled comparison. All 2,160 requests completed. The candidate's goodput was 20–24% below default and first-token tail latency worsened. [Original decision and measurements](scheduling_results/analysis/validation.json).

A descriptive analysis of those same records found, in every trial:

- All 96 initial short requests met the latency targets.
- None of the 96 post-burst short requests met them.
- Only 4–9 of 48 long requests met them.
- The candidate needed about 21 seconds after the last arrival to finish, versus 10–12 seconds for default vLLM.

This exposes poor recovery under the tested burst. Almost the same number of requests passed across policies, while completion time changed. Consequently, much of the goodput difference comes from the full replay-and-drain denominator. It is not a steady-state capacity measurement. [Phase evidence](scheduling_results/analysis/phase_diagnostics.json), [reproduction script](scheduling_validation/analyze_phases.py).

## Where the current attack is weak

The token budget mixes several changes. It restricts concurrency, accounts for prompt length, reorders requests and adds age protection. With only long requests, 20,000 / (2,048 + 128) permits nine admitted requests, versus 32 in the fixed control. These are admitted requests, which may wait inside the engine, rather than simultaneous GPU executions. Reduced admission and longer waiting are observed; their exact causal contribution remains unmeasured.

The reservation holds the full input-plus-output estimate until completion. It is neither measured service cost nor actual live KV bytes. Its two-second age protection starts when the two-second first-token target is already exhausted. These choices can explain why the rule is worth revising, but the negative result alone does not establish the correct replacement.

One fixed arrival rate and one finite workload cannot tell us where extra load becomes unsustainable. No scheduler can be assumed to eliminate overload without sufficient processing capacity or an explicit admission/rejection policy. Dropping hard requests must not silently improve reported goodput. All requests and both prompt classes must remain accounted for.

Scheduling was tested on one model. The five-model capture study does not demonstrate cross-model scheduling generalization. Output hashes differ across policies, so fixed output length does not establish answer-quality equivalence. GPU-busy telemetry cannot identify achieved SM efficiency, memory bandwidth saturation or the underlying bottleneck.

## One next experiment

Use one comparison to test whether adaptive admission adds value beyond a calibrated simple limit. On separate calibration data, measure waiting and processing behavior, select baseline settings and fit a service-cost estimate. Freeze the resulting candidate before evaluation. Replay unseen burst traces at several predeclared arrival rates on the same GPU and model.

Compare default vLLM, the strongest calibrated fixed rule and the adaptive candidate. Keep fairness conditions equal where comparing admission rules, and include a version with the adaptive decision disabled to isolate its effect. Record goodput, throughput, phase-specific target attainment, external waiting and long-request tail latency. Preserve complete outputs and report any dropped requests. Use independent engine repetitions and keep detailed GPU profiling separate from timed runs.

The experiment can succeed, fail or show a workload-dependent tradeoff. Its purpose is to determine whether a measured-cost decision helps, and at what load. A failure would still delimit the useful scope of admission control.

The direction is consistent with established work on throughput–latency tradeoffs, including [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) and [SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf). Their existence motivates strong baselines; it does not validate our proposed policy. [Full references](References.md).

## Local reproduction

Run `python3 outputs/scheduling_validation/analyze_phases.py` from the repository root. It reads the completed request, telemetry and summary records and writes `outputs/scheduling_results/analysis/phase_diagnostics.json`. This analysis uses the Python standard library and does not access a GPU. The original GPU instance was destroyed after verified backup.
