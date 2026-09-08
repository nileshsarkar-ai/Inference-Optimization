# Research explainer source

`build_explainer.py` creates `../Inference_Optimization_Explained.pdf`, a five-page student-oriented explanation. It uses the preserved local experiment records; it does not execute a model or require a GPU/network connection.

Run from the repository root:

```bash
python3 outputs/document_source/build_explainer.py
```

Requirements: Python 3, `reportlab`, `matplotlib`, and `numpy`. Arial is embedded when available on macOS; the builder falls back to Helvetica on other systems. It writes temporary rendering/cache files under `work/pdf_explainer/` and reproducible chart assets under `outputs/document_source/figures/`.

The two document figures are layouts of **existing measurements**, not additional experiments:

- `capture_measurements`: all 216 warmed timing observations from the three current models, read directly from `experiment_data_gemma12b`, `experiment_data_qwen35`, and `experiment_data_gemma_e4b`, plus within-configuration means.
- `scheduling_measurements`: four metrics from nine policy/run summaries (36 points), plus arithmetic means across three independent engine runs for each policy.

The text also uses the preserved phase diagnostics, GPU/environment records, and older-control counts documented in `../Experiment_Inventory.md`. The numerical scheduling table is computed from the nine raw summaries. `build_summary.json` records the resulting values, plotting counts, source links, and page-content extents.

To render for local visual review:

```bash
pdftoppm -r 120 -png outputs/Inference_Optimization_Explained.pdf work/pdf_explainer/page
```

The final five pages were inspected individually after rendering. The explanation includes the intuition for burst backlog, CUDA-graph padding and memory costs, and how a restrictive gate can reduce useful batching while the GPU remains busy. Suggested mechanisms are explicitly distinguished from measured causal findings. Clickable citations link to original papers, official metric/engine documentation, and the project’s actual result records. The document preserves the negative candidate result and distinguishes kernel-busy time from achieved GPU compute efficiency.
