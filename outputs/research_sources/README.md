# Figure provenance

These figures are excerpts from the original papers, not reconstructed experimental results. Each relevant slide has a linked source footer. SaturateLLM's own graphs are generated from the supplied measured JSONL records.

| Local asset | Original source and context |
| --- | --- |
| `sola-figure1.png` | Hong et al., SOLA, MLSys 2025, Figure 1. Llama3-70B, four A100 GPUs, ShareGPT at 4.6 requests/s. The paper reports 65% and 98% SLO attainment for its compared systems. |
| `duet-dispatch-figure-complete.png` | Gao et al., DuetServe, ICML 2026, Figure 5. Original schematic of conventional dispatch and look-ahead execution. |
| `duet-figure6-azure-throughput.png`, `duet-figure6-legend.png` | DuetServe Figure 6, AzureCode throughput panel and its original legend. Qwen3-8B, tensor parallelism 1, H100 80 GB. Precise 12.43 and 13.57 completed requests/s at 16 offered requests/s are from Section 5.2. |

- [SOLA original proceedings PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf)
- [DuetServe original author PDF](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)

The PDFs are saved alongside their excerpts. Their experiments have different workloads, hardware and baseline versions; the deck does not rank systems across these papers.

The NVIDIA graph-size results are presented as a cited table. Their source is an engineering report, not a peer-reviewed paper: [Tuning CUDA Graph Batch Sizes for Higher Output Throughput](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html). It explicitly identifies concurrency-log-based automatic size selection as future work, so that broad idea is not claimed as our novelty.

The source links and qualifications are in the 15-slide deck's speaker notes and `SaturateLLM_Problem_Statement.md`.

## Non-LLM evidence

`prism-figure17.png` is the original Figure 17 from Yang et al., *GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale*, NSDI 2025. The source PDF is saved as `prism_dlrm.pdf`. It reports GPU-node goodput with separate CPU nodes added in Prism, using an eight-A100-80-GB GPU node. The highest Model-XS result uses MIG. The 5–9× range is not a GPU-busy percentage and does not establish transfer of SaturateLLM to recommendation models.

[Original Prism DLRM paper](https://www.usenix.org/system/files/nsdi25-yang.pdf). This system is distinct from *Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning*, OSDI 2026.
