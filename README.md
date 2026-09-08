# Inference Optimization

Research on completing more useful inference work per GPU while meeting latency targets. Large language models are the first test bed. The current study investigates GPU execution settings and request admission on one A100 PCIe 40 GB.

## Research question

When long prompts arrive in a burst, can admission decisions based on estimated processing time, available resources and time remaining before a latency target improve useful completed work over current vLLM?

This is an open investigation. The completed tests did not establish an improved scheduler. The saved request records expose poor recovery after the long-prompt burst: all 96 subsequent short requests miss the declared targets in every tested policy and repetition. This is a finding for the controlled workload, not a claim about sustainable serving capacity.

## Presentation and research records

- [Final presentation](Inference_Optimization_Final.pptx): the single current 22-slide PPT, organized into the question, literature, system and experiment protocols, results, Tensormux/TensorPath evidence, and one profiling/intervention study. Includes all nine distinct experimental chart panels, references and native presenter notes for every slide.
- [Plain-language research explainer](outputs/Inference_Optimization_Explained.pdf): the problem, system, experiment designs and actual results, with graphs from saved measurements.
- [Keynote presenter setup](outputs/Keynote_Presenter_Setup.md): speaker notes and classroom display instructions. Keynote import was checked; the physical projector must be checked in the room.
- [Experiment inventory](outputs/Experiment_Inventory.md): study folders, graphs, raw data, reproduction scripts and slide coverage.
- [Research question](outputs/Research_Question.md)
- [Measured results](outputs/Results.md)
- [Methodology assessment and next comparison](outputs/Methodology_Assessment.md)
- [References](outputs/References.md)
- [File manifest](FILE_MANIFEST.json): SHA256 and size of the committed research files.

## What we ran

| Study | Models / workload | Observations |
| --- | --- | --- |
| Current-model capture comparison | Gemma 4 12B, Qwen3.5-9B, Gemma 4 E4B | 216 timed calls, 72 per model |
| Earlier implementation controls | Qwen2.5-1.5B, Qwen2.5-7B | 144 timed calls, 72 per model |
| Admission scheduling comparison | Gemma 4 12B, phased short/long/short prompts | 2,160 requests across three policies and three independent engine runs |

The capture comparison changes CUDA-graph capture sizes while keeping the engine setup fixed within each model. CUDA graphs replay recorded GPU operations. Throughput charts show all three warmed timing repetitions and their means. Profiling runs are separate and are excluded from timed throughput.

The scheduling test compares current vLLM submission, a FIFO limit of 32 unfinished requests, and a context-budget gate. All policies replay the same saved request arrivals within each repetition. First-token latency includes admission waiting. The output length is fixed at 128 tokens.

## Main measured findings

- On the three current models, matched capture sizes differ from current defaults by about −0.6% to +0.3%. Coarse grids lose up to 3.8%.
- The context-budget admission setting produces 20–24% less SLO goodput than default vLLM across the three paired repetitions, with worse first-token tail latency. It fails the predeclared success criterion.
- GPU busy time is about 99–100% across scheduling runs, despite different useful completion rates. This metric alone does not measure achieved compute efficiency.

Here, SLO goodput is the number of requests meeting both targets, divided by the entire replay-and-drain duration. The targets are time to first token ≤2 seconds and mean time per subsequent output token ≤100 ms. These are declared experimental conditions, not universal service requirements.

See the [results report](outputs/Results.md) for every outcome, exact comparisons and limitations. Three independent scheduling repetitions and one controlled workload do not establish generalization or semantic-quality equivalence.

## Code and reproduction

- [Capture study](outputs/model_suite/README.md), with the source scripts in `outputs/model_suite/`.
- [Scheduling protocol and run commands](outputs/scheduling_validation/PROTOCOL.md).
- `outputs/remote_reproduction/` and each result folder's `reproduction/` preserve the exact executed source copies.
- Raw results include model and dataset revisions, environment versions, request/output token IDs, logs and profiler traces.
- [Presentation source](outputs/presentation_source/README.md), including editable chart generation and all 22 native presenter-note scripts.
- [Explainer source](outputs/document_source/README.md), with two PNG/PDF figure families that replot existing measurements.

Original GPU: one NVIDIA A100-PCIE-40GB, 40,960 MiB reported memory, 250 W power limit and driver 595.58.03. CPU model and installed host RAM were not recorded. Original environment: Python 3.12.13, vLLM 0.28.0, PyTorch 2.13.0 and Transformers 5.16.1. Full package lists are saved with each run. A compatible NVIDIA GPU and valid W&B authentication are needed for reruns. Credentials and model-weight caches are not included.

The queue launchers retain the original machine paths and predecessor run IDs. Use the direct experiment commands in the protocol on a fresh machine. No active GPU is required to inspect the saved results or regenerate the analysis.

## Provenance

[W&B experiment project](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility). Historical run names retain the original project identifier so the saved records continue to match W&B.

The original A100 instance was destroyed after completion and verified local backup. All eight result directories, containing 525 files, matched remote SHA256 hashes before shutdown. The repository preserves those receipts, failed setup attempts, and detached-run logs.

Paper figures and published results have their own source links and experimental context. They are separate from our measurements. The bibliography distinguishes peer-reviewed papers from official engineering documentation.
