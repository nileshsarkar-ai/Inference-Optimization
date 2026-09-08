# Inference Optimization references

Checked 8 September 2026. The 18-slide PPT includes clickable source footers, a visible references slide and citations in native presenter notes. Original paper excerpts remain separate from our own measured graphs. Cross-paper gains are not directly comparable.

## Published systems research

1. **Hong et al. (2025). SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling. MLSys 2025.** [Proceedings PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf). Slides 4–5. Original Figure 1: 65% to 98% SLO attainment, Llama3-70B, four A100s, ShareGPT at 4.6 requests/s. Targets: TTFT 500 ms and TPOT 200 ms. Local PDF: research_sources/sola.pdf.

2. **Gao et al. (2026). DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing. ICML 2026.** [Author PDF](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf). Slides 4 and 6. Original Figure 6 AzureCode throughput panel and legend. Section 5.2 supplies 12.43 and 13.57 completed requests/s at 16 offered requests/s, relative to SGLang-Default, Qwen3-8B on one H100 80 GB. The deck's 9.2% is calculated from these values. Local PDF: research_sources/duetserve.pdf.

3. **Yang et al. (2025). GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale. NSDI 2025.** [USENIX PDF](https://www.usenix.org/system/files/nsdi25-yang.pdf). Slides 4 and 7. Original Figure 17: 5–9× GPU-node goodput in the reported configurations. Eight A100 80 GB GPUs per node, with separate CPU nodes added by Prism. The highest Model-XS result uses MIG. Local PDF: research_sources/prism_dlrm.pdf.

4. **Agrawal et al. (2024). Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. OSDI 2024.** [USENIX PDF](https://www.usenix.org/system/files/osdi24-agrawal.pdf). Additional background on chunked prefill and batching; not a named system in the current PPT.

5. **Aegaeon (2025). Effective GPU Pooling for Concurrent LLM Serving on the Market. SOSP 2025.** [Author PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf). Additional background on token-level model autoscaling and GPU pooling; not a named system in the current PPT. No Aegaeon numeric result is used.

6. **Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning (2026). OSDI 2026.** [USENIX technical-session listing](https://www.usenix.org/conference/osdi26/technical-sessions). Additional background, not a named system in the current PPT. This LLM system is distinct from the NSDI 2025 recommendation-model system above. No numeric result is used in the deck.

7. **Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.** [Paper](https://arxiv.org/abs/2309.06180). Supporting background on vLLM's memory management.

## Official engineering documentation

8. **NVIDIA Dynamo, KV-Aware Routing.** [Architecture and policy](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing), [routing concepts](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts). Additional background, not a named system in the current PPT. Cache reuse and projected load inform current routing decisions. These are product documents, not peer-reviewed results.

9. **NVIDIA (August 2026). Tuning CUDA Graph Batch Sizes for Higher Output Throughput.** [TensorRT-LLM engineering report](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html). Background for the capture protocol and results on slides 10 and 13–14. Discusses padding, graph-memory cost and automatic selection from concurrency logs. This engineering report describes the configuration tradeoff in our control study.

10. **vLLM CUDA Graphs.** [Official design documentation](https://docs.vllm.ai/en/stable/design/cuda_graphs/). [Async streaming example](https://docs.vllm.ai/en/latest/examples/deployment/async_llm_streaming/). Implementation background. The experiment records the actual installed version, vLLM 0.28.0, and complete requested settings.

11. **NVIDIA NVML utilization definition.** [nvmlUtilization_t](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html). Slide 9. Kernel-busy percentage is different from achieved SM efficiency or useful completed work.

## Models and data

12. Official model repositories: [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B), [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Slides 9–11 and 13. The result folders record immutable revisions and exact package versions. E4B indicates effective active parameters; it should not be described as a four-billion-total-parameter model.

13. [Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext). Real WikiText-2 test text supplies the prompts. Request arrivals are a declared controlled trace, not a production workload. Dataset revision and exact tokenized traces are saved with the experiments.

## Our actual measurements

14. [Recorded W&B experiment project](https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility). Slides 9–17. Capture-comparison JSONL records, raw output token IDs, runtime configuration and independent scheduling request/telemetry records are saved locally. The summary CSVs and full-resolution PNG/PDF graphs are supplied with the deck. The accompanying explainer uses the same saved measurements, replotted under `document_source/figures/`; it does not add experiments. These results are our measurements, not published-paper results.

No claim of a solved algorithm, universal GPU-utilization improvement or validated transfer beyond LLMs follows from these sources or experiments.
