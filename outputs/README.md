# Inference Optimization research records

## Current presentation and references

- `../Inference_Optimization_Final.pptx`: current 18-slide presentation, organized into the research question, previous literature, our system and experiment protocols, measured results, and one next comparison. It retains all nine distinct experimental chart panels. Original paper figures have linked source footers; references are visible on slide 18. Each slide includes native presenter notes.
- `Inference_Optimization_Explained.pdf`: a plain-language explanation of the problem, system, experiments, graphs and measured conclusions.
- `Keynote_Presenter_Setup.md`: presenter-note and extended-display setup for a classroom presentation. Keynote import was checked; a physical HDMI/projector test remains room-specific.
- `References.md`: consolidated primary-source bibliography with slide mappings and experimental context.
- `Research_Question.md`: research question, SOTA boundary, hypotheses and one eventual experiment.
- `Results.md`: complete measured findings and limitations.
- `Methodology_Assessment.md`: what the tests support, design weaknesses and one controlled next comparison.

The candidate context-budget scheduler did not improve SLO goodput. Current vLLM is the strongest measured goodput control. The research question remains open.

## Actual results and code

- `scheduling_results/`: all 2,160 scheduling requests, exact traces, streamed output token IDs, telemetry, per-run summaries, package versions and pinned revisions. `analysis/` contains full-resolution PNG/PDF graphs, CSV and the predeclared decision outcome.
- `scheduling_validation/`: directly runnable research scripts and the frozen protocol. Exact executed copies are also in `scheduling_results/reproduction/`.
- `experiment_data_gemma12b/`, `experiment_data_qwen35/`, `experiment_data_gemma_e4b/`: the 216 current-model capture measurements, runtime logs and separate profiler traces.
- `current_model_summary/` and `figures/`: aggregate capture data and full-resolution graphs.
- `model_suite/`: capture study scripts and analysis.
- `experiment_data/` and `experiment_data_qwen7b_complete/`: older Qwen2.5 supplementary controls. These are not current-generation evidence.
- `remote_reproduction/`: exact remote experiment scripts and queue state.
- `managed_run_records/` and `archived_attempts/`: detached-run logs, exit codes and unsuccessful setup attempts.
- `research_sources/`: original paper PDFs, source figure excerpts and provenance.
- `presentation_source/`: current slide builder, per-slide presenter notes and zipped template assets.
- `document_source/`: explainer builder and two PNG/PDF figure families that replot existing measurements; these are not additional experiments.

W&B project: https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility

## Reproduction

Read the experiment-specific protocol/README before running. The original environment was Python 3.12.13, vLLM 0.28.0, PyTorch 2.13.0 and Transformers 5.16.1 on A100 PCIe 40 GB. Full version lists and model/dataset revisions are saved with the records. Authentication is not included in this archive. Queue scripts contain run-specific paths/dependency IDs; use the direct prepare/run commands in the protocol for a fresh machine.

## Local backup and GPU lifecycle

All eight remote result directories, totaling 525 files, matched local SHA256 hashes before shutdown. Scripts and managed-run records also matched checksum comparisons. The canonical local codebase is `/Users/nileshsarkar/Documents/SaturateLLM/`. Experiment records are under `outputs/`; the single final PPT is at the codebase root.

JarvisLabs instance 500717 was destroyed after successful completion and backup. It was not paused. `experiment_status.json` and `backup_receipts/gpu_shutdown.json` record the final state.
