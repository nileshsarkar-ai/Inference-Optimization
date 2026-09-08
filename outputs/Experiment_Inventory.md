# Inference Optimization — complete experiment inventory

This inventory describes the files actually saved on 8 September 2026. Paths are relative to the directory containing this document. The completed work comprises **five CUDA-graph configuration studies, 360 unprofiled timing calls, ten separate profiler passes, and a scheduling comparison with 2,160 completed requests**. No additional GPU experiments are implied by the presentation update.

**Final PPT coverage verified:** slide 9 contains the three modern-model graphs, slide 10 contains the two earlier Qwen controls, and slide 13 contains all four scheduling metrics. All 360 capture observations and 36 scheduling metric observations match the saved data in nine editable charts, with means shown separately. Every final slide was individually reviewed.

## 1. Five completed configuration studies

Each study contains a `complete.json` marker. File-level checks found exactly 24 records in each of `default/measurements.jsonl`, `coarse/measurements.jsonl` and `matched/measurements.jsonl`: eight batch sizes × three warmed repetitions = **72 timing calls per model**.

| Model | Complete saved study | Timing calls | Separate profile passes | Backup receipt |
|---|---|---:|---:|---|
| Qwen2.5-1.5B | [experiment_data/](experiment_data/) | 72 | 2 | [qwen15b.json](backup_receipts/qwen15b.json) |
| Qwen2.5-7B | [experiment_data_qwen7b_complete/](experiment_data_qwen7b_complete/) | 72 | 2 | [qwen7b.json](backup_receipts/qwen7b.json) |
| Gemma 4 12B | [experiment_data_gemma12b/](experiment_data_gemma12b/) | 72 | 2 | [gemma12b.json](backup_receipts/gemma12b.json) |
| Qwen3.5-9B | [experiment_data_qwen35/](experiment_data_qwen35/) | 72 | 2 | [qwen35.json](backup_receipts/qwen35.json) |
| Gemma 4 E4B | [experiment_data_gemma_e4b/](experiment_data_gemma_e4b/) | 72 | 2 | [gemma_e4b.json](backup_receipts/gemma_e4b.json) |

Common settings were one A100 PCIe 40 GB, BF16, tensor parallelism 1, vLLM 0.28.0, 128 input tokens and 128 generated tokens. Fixed batch sizes were **24, 31, 48, 63, 80, 95, 112 and 127**. Real WikiText-2 test text supplied prompts. The model and dataset revisions, tokenized-prompt checksum, seed, resolved capture sizes and software versions are saved with each study. The three modern models use text-only execution.

The independent variable was the CUDA-graph capture-size configuration: current default, a coarse grid, and a manually matched set. Each configuration used a fresh engine process; the three timing repetitions within a configuration were warmed repetitions, not independent engine runs. Timing includes prefill and host work. The nominal next graph in each record is a configuration-derived proxy, not an observed dispatch trace.

Every study retains:

- `prepared/manifest.json` and `prepared/prompts.json` for exact model/data identity and inputs.
- `default/`, `coarse/` and `matched/`, each with `measurements.jsonl`, `environment.json`, `requested-config.json`, `package-versions.txt`, `gpu.csv`, `wandb-url.txt` and `worker-resources-final.json`.
- `figures/throughput.png`, `throughput.pdf`, `summary.csv`, `provenance.json` and `output-consistency.json`.
- `default.log`, `coarse.log`, `matched.log`, `profile-default.log` and `profile-matched.log`.
- `profile-default/` and `profile-matched/`: ten profile directories in total, each containing its own environment/configuration, GPU record, W&B link, compressed PyTorch trace and `profiler_out_0.txt`. These diagnostic passes are separate from the timing plots.

The three modern studies additionally save 24 `outputs-b{batch}-r{repeat}.json` files per configuration: **216 full output-record files**. Earlier Qwen controls retain output counts and SHA-256 hashes in the measurement records; they do not contain these additional token-ID files. This distinction matters when reproducing output comparisons. Output hashes are not a model-quality evaluation.

## 2. Completed scheduling comparison

The complete record is in [scheduling_results/](scheduling_results/), including its root completion marker and one completion marker for each of three independent engine runs.

| Controlled element | Saved design |
|---|---|
| Hardware and model | Same A100 40 GB; Gemma 4 12B at revision `023679ed352de9bb66cc873c9009ce3482585c08`; BF16; vLLM 0.28.0 |
| Engine settings | 4,096-token maximum context and batched-token limits; 128 maximum sequences; requested GPU-memory fraction 0.85; default graph configuration; chunked prefill enabled; prefix cache disabled |
| Trace | 240 requests: 96 × 128-input-token requests, then 48 × 2,048-input-token requests, then 96 × 128-input-token requests; every request generates 128 tokens |
| Arrivals | Exponential inter-arrival times at 8 requests/s; seeds 20260908, 20260909 and 20260910; identical saved trace for all policies within each engine repetition |
| Default control | Submit each arriving request to current vLLM continuous batching |
| Fixed-limit control | External FIFO gate allowing at most 32 unfinished requests in the engine |
| Context-budget candidate | Reserve input tokens + 128 output tokens per admitted request; 20,000-token budget; oldest fitting request with two-second age protection |
| Replication | Three independent engine processes; policy order rotates default/fixed/context, fixed/context/default, context/default/fixed; common warmup excluded and all requests drain between trials |
| Service target | Time to first token (TTFT) ≤ 2 seconds and average time per output token (TPOT) ≤ 100 ms; includes external admission waiting |
| Decision rule | At least 5% greater SLO goodput than both controls in every paired repetition, with no worse p99 TTFT or p99 stream gap |

Actual-file checks found **240 request records in every policy/run directory**, with 192 short and 48 long requests. Every recorded output has 128 token IDs: **3 policies × 3 engine runs × 240 = 2,160 completed requests**.

The result directory contains:

- `prepared/manifest.json` and `prepared/trace-0.json` through `trace-2.json`, with immutable trace checksums.
- `repeat-0/`, `repeat-1/` and `repeat-2/`, each retaining `environment.json`, `package-versions.txt`, `gpu.csv`, `wandb-url.txt` and `complete.json`.
- Within every repetition, `default/`, `fixed32/` and `context_budget/`, each containing `requests.json`, `telemetry.json` and `summary.json`. Request records include admission waiting, streaming chunks, completion timings and generated token IDs.
- `repeat-0.log` through `repeat-2.log`.
- [analysis/validation.json](scheduling_results/analysis/validation.json), [analysis/summary.csv](scheduling_results/analysis/summary.csv), and the four-panel PNG/PDF result figure. The JSON also retains per-class results, paired decisions and output-hash differences.
- [reproduction/](scheduling_results/reproduction/), containing the exact saved scripts and protocol.

The candidate **did not meet the predeclared criterion**. GPU-busy readings were approximately 99–100% across policies, while useful completed work differed. NVML kernel-busy time is not achieved SM utilization; this test does not establish a rise in achieved compute utilization or generalization beyond this model and trace.

A later descriptive analysis of the same saved requests is in [analysis/phase_diagnostics.json](scheduling_results/analysis/phase_diagnostics.json), reproduced by [analyze_phases.py](scheduling_validation/analyze_phases.py). It adds phase-specific SLO attainment, admission waiting, admitted-request counts and drain time. It does not add GPU runs or alter the original protocol. Slide 14 summarizes its burst-recovery finding. [Methodology assessment](Methodology_Assessment.md) explains what the design supports and the limits of causal interpretation.

## 3. Complete graph coverage

There are **seven saved experiment figure families**, each available as a PNG/PDF pair. Five are individual-model configuration figures, one combines the three modern models, and one contains the scheduling metrics. The combined modern figure reuses the same measurements as the three individual modern figures. PNG and PDF are alternate formats, not separate experiments.

| Figure family and available files | Underlying observations | Verified final PPT coverage |
|---|---|---|
| Qwen2.5-1.5B: [PNG](experiment_data/figures/throughput.png), [PDF](experiment_data/figures/throughput.pdf) | 72 timing calls; all repetitions and means | Slide 10, earlier control panel |
| Qwen2.5-7B: [PNG](experiment_data_qwen7b_complete/figures/throughput.png), [PDF](experiment_data_qwen7b_complete/figures/throughput.pdf) | 72 timing calls; all repetitions and means | Slide 10, earlier control panel |
| Gemma 4 12B: [PNG](experiment_data_gemma12b/figures/throughput.png), [PDF](experiment_data_gemma12b/figures/throughput.pdf) | 72 timing calls; all repetitions and means | Slide 9, modern-model panel |
| Qwen3.5-9B: [PNG](experiment_data_qwen35/figures/throughput.png), [PDF](experiment_data_qwen35/figures/throughput.pdf) | 72 timing calls; all repetitions and means | Slide 9, modern-model panel |
| Gemma 4 E4B: [PNG](experiment_data_gemma_e4b/figures/throughput.png), [PDF](experiment_data_gemma_e4b/figures/throughput.pdf) | 72 timing calls; all repetitions and means | Slide 9, modern-model panel |
| Three-model overview: [PNG](figures/throughput_all_models.png), [PDF](figures/throughput_all_models.pdf) | The same 216 modern-model timing calls; no additional experiment | Slide 9 covers these same three panels |
| Scheduling: [PNG](scheduling_results/analysis/scheduling_results.png), [PDF](scheduling_results/analysis/scheduling_results.pdf) | SLO goodput, completed output tokens/s, p99 TTFT and mean NVML GPU busy; nine observations per metric plus means | Slide 13, all four metrics and all repetitions |

Complete unique coverage therefore means **five model-throughput panels plus four scheduling-metric panels**. It does not require inserting duplicates of each file format or both the individual and combined representations. The modern overview is derived from [current_model_summary/suite.json](current_model_summary/suite.json) and [summary.csv](current_model_summary/summary.csv); the plotting provenance lists the exact underlying JSONL paths. Scheduling plots are derived from the nine saved policy/run summaries.

## 4. Code, reproduction records and sources

| Purpose | Saved location |
|---|---|
| Earlier Qwen experiment code and setup | [pilot/](pilot/): `pilot.py`, `orchestrate.py`, `analyze.py`, `bootstrap.sh`, `README.md` |
| Modern-model experiment and summary/plot code | [model_suite/](model_suite/): `pilot.py`, `orchestrate.py`, queue scripts, `analyze.py`, `summarize_suite.py`, `plot_suite.py`, `README.md` |
| Scheduling code and declared protocol | [scheduling_validation/](scheduling_validation/): `validate_scheduling.py`, `queue_validation.py`, `analyze_scheduling.py`, `PROTOCOL.md` |
| Copies downloaded from the GPU | [remote_reproduction/](remote_reproduction/): `graph_size_pilot/`, `model_suite/`, `scheduling_validation/`; includes modern-suite queue status |
| Run-local reproduction snapshots | [experiment_data/reproduction/](experiment_data/reproduction/); `experiment_data_qwen7b_complete/pilot.py` and `orchestrate.py`; [scheduling_results/reproduction/](scheduling_results/reproduction/) |
| Presentation source | [presentation_source/](presentation_source/): current slide builder, zipped template source assets, source notes, report builder and README |
| Original paper assets | [research_sources/](research_sources/): SOLA, DuetServe and Prism DLRM PDFs and original figure crops; the directory README records sources |
| W&B tracking | [Project](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility); exact run links are retained in each `wandb-url.txt` |

Model revisions and package versions are pinned in the environment records. Dataset revision is `b08601e04326c79dfdd32d625aee71d232d685c3`, `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split. The scheduling workload uses contiguous token slices of real text and controlled arrivals; it is not a production trace.

## 5. Failed attempts, execution logs and backup evidence

Setup failures are retained in [archived_attempts/](archived_attempts/). `setup_144751/` contains the missing-`pip` environment-recording failure; `setup_145003/` contains the resource-query callable serialization failure. Neither folder contains a completed timing comparison. Their prepared inputs, logs and any pre-failure records are preserved rather than included as successful measurements.

[managed_run_records/](managed_run_records/) retains nine detached-run records. Every directory contains `meta.json`, `output.log`, `run.sh`, `pid` and `exit_code`.

| Run ID | Saved exit code | Role |
|---|---:|---|
| `r_3ef68da2` | 0 | Qwen2.5-1.5B completed study |
| `r_280f8ded` | 0 | Qwen2.5-7B completed study |
| `r_739e084f` | 0 | Gemma 4 12B and Qwen3.5-9B completed suite |
| `r_7a2c52f8` | 0 | Gemma 4 E4B completed study |
| `r_6d73eb1e` | 0 | Scheduling comparison |
| `r_789b10e3`, `r_b317ce96`, `r_ee3f787d` | 1 | Unsuccessful setup/launch records retained for traceability |
| `r_9c5021fe` | 127 | Unsuccessful setup/launch record retained for traceability |

[backup_receipts/](backup_receipts/) contains eight SHA-256 comparison receipts: all five model studies, the scheduling study, and both archived setup attempts. They collectively record **525 remote result files** verified against the local copies before shutdown. Some result folders also contain additional local reproduction copies, so their present local file counts need not equal the remote receipt count.

The ninth receipt, [gpu_shutdown.json](backup_receipts/gpu_shutdown.json), records successful destruction of instance **500717** after verified backup and its absence from the instance list. It also records the local codebase-backup destination. This inventory is a navigational record; the separate full-file manifest supplies per-file checksums for the final committed package.
