"""SaturateLLM controlled CUDA-graph configuration experiment.

API sources:
https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/
https://docs.vllm.ai/en/stable/design/cuda_graphs/
https://docs.vllm.ai/en/stable/api/vllm/config/profiler/
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "Salesforce/wikitext"
BATCHES = [24, 31, 48, 63, 80, 95, 112, 127]
PROMPT_TOKENS = 128
OUTPUT_TOKENS = 128
MAX_SEQS = 128
CAPTURE_CEILING = 256
REPEATS = 3
SEED = 20260908
COARSE = [1, 2, 4, 8, 16, 32, 64, 128, CAPTURE_CEILING]
MATCHED = sorted(set(COARSE + BATCHES))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(folder, model):
    """Resolve immutable revisions and prepare real-text prompts once, on CPU."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    folder.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    model_revision = api.model_info(model).sha
    data_revision = api.dataset_info(DATASET).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=model_revision)
    data = load_dataset(DATASET, "wikitext-2-raw-v1", split="test", revision=data_revision)
    # Shuffle real paragraphs, then take fixed-length prefixes; no random-token data.
    candidates = list(range(len(data)))
    random.Random(SEED).shuffle(candidates)
    prompts = []
    for row in candidates:
        ids = tokenizer.encode(data[row]["text"], add_special_tokens=False)
        if len(ids) >= PROMPT_TOKENS:
            prompts.append({"row": row, "prompt_token_ids": ids[:PROMPT_TOKENS]})
        if len(prompts) == MAX_SEQS:
            break
    if len(prompts) != MAX_SEQS:
        raise ValueError("Insufficient real-text paragraphs for the requested batch sizes.")
    write_json(folder / "prompts.json", prompts)
    write_json(folder / "manifest.json", {
        "model": model, "model_revision": model_revision,
        "dataset": DATASET, "dataset_revision": data_revision,
        "dataset_config": "wikitext-2-raw-v1", "split": "test",
        "seed": SEED, "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS, "batches": BATCHES,
        "prompts_sha256": hashlib.sha256((folder / "prompts.json").read_bytes()).hexdigest(),
    })


def resource_snapshot(llm):
    """Read device memory and exposed cache metadata without worker serialization."""
    cache = llm.llm_engine.vllm_config.cache_config
    blocks = cache.num_gpu_blocks
    block_size = cache.block_size
    used = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return {
        "device_used_mib": float(used.strip()),
        "kv_blocks": blocks, "kv_block_size": block_size,
        "kv_capacity_tokens": None if blocks is None else blocks * block_size,
    }


def run(args):
    # No CUDA initialization in the parent before vLLM creates its workers.
    from vllm import LLM, SamplingParams
    import wandb

    manifest = json.loads((args.prepared / "manifest.json").read_text())
    prompts_bytes = (args.prepared / "prompts.json").read_bytes()
    if hashlib.sha256(prompts_bytes).hexdigest() != manifest["prompts_sha256"]:
        raise ValueError("Prompt snapshot changed after preparation.")
    prompts = [{"prompt_token_ids": p["prompt_token_ids"]} for p in json.loads(prompts_bytes)]
    args.out.mkdir(parents=True, exist_ok=False)
    gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version,power.limit", "--format=csv"], text=True)
    (args.out / "gpu.csv").write_text(gpu_info)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.out / "package-versions.txt").write_text("\n".join(packages) + "\n")
    compilation = {"max_cudagraph_capture_size": CAPTURE_CEILING}
    if args.capture != "default":
        compilation["cudagraph_capture_sizes"] = COARSE if args.capture == "coarse" else MATCHED
    engine_args = {
        "model": manifest["model"], "revision": manifest["model_revision"],
        "tokenizer_revision": manifest["model_revision"], "dtype": "bfloat16",
        "tensor_parallel_size": 1, "seed": SEED,
        "max_model_len": 512, "max_num_seqs": MAX_SEQS,
        "max_num_batched_tokens": 4096, "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": False, "enable_chunked_prefill": True,
        "compilation_config": compilation, "disable_log_stats": False,
    }
    if args.profile:
        engine_args["profiler_config"] = {
            "profiler": "torch", "torch_profiler_dir": str((args.out / "traces").resolve()),
            "torch_profiler_record_shapes": True,
        }
    write_json(args.out / "requested-config.json", engine_args)
    wb = wandb.init(project="saturatellm-feasibility", group=os.environ.get("SATURATE_GROUP", "a100-40gb-pilot"),
                    name=f"{args.capture}-{'profile' if args.profile else 'timing'}",
                    config={"engine": engine_args, "manifest": manifest, "capture": args.capture,
                            "batches": BATCHES, "repeats": REPEATS, "measurement_type": "offline_fixed_batch"})
    (args.out / "wandb-url.txt").write_text(wb.url + "\n")
    t0 = time.perf_counter()
    llm = LLM(**engine_args)
    startup_s = time.perf_counter() - t0
    cfg = llm.llm_engine.vllm_config.compilation_config
    resolved_sizes = list(cfg.cudagraph_capture_sizes)
    if max(resolved_sizes) != CAPTURE_CEILING:
        raise ValueError(f"Capture coverage changed: {resolved_sizes}")
    resources = resource_snapshot(llm)
    wb.config.update({"resolved_capture_sizes": resolved_sizes, "resolved_graph_mode": str(cfg.cudagraph_mode)})
    wb.summary["startup_s"] = startup_s
    wb.summary["resources_after_init"] = resources
    write_json(args.out / "environment.json", {
        "manifest": manifest, "capture": args.capture,
        "python": sys.version, "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": importlib.metadata.version("torch"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_capture_sizes": resolved_sizes, "resolved_graph_mode": str(cfg.cudagraph_mode),
        "startup_s": startup_s, "resources_after_init": resources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_type": "offline fixed-batch generation; includes prefill and host work",
    })
    params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS, ignore_eos=True)
    # Warm every measured shape outside the timing region.
    batches = [63] if args.profile else BATCHES
    for batch in batches:
        llm.generate(prompts[:batch], params, use_tqdm=False)
    if args.profile:
        llm.start_profile()
        llm.generate(prompts[:63], params, use_tqdm=False)
        llm.stop_profile()
        wb.summary["profile_complete"] = True
        wb.finish()
        return  # Profiled timings must never enter the performance chart.

    cases = [(rep, batch) for rep in range(REPEATS) for batch in BATCHES]
    random.Random(SEED).shuffle(cases)
    with (args.out / "measurements.jsonl").open("w") as rows:
        for rep, batch in cases:
            start = time.perf_counter()
            results = llm.generate(prompts[:batch], params, use_tqdm=False)
            elapsed_s = time.perf_counter() - start
            ids = [list(r.outputs[0].token_ids) for r in results]
            if len(ids) != batch or any(len(x) != OUTPUT_TOKENS for x in ids):
                raise ValueError("Incomplete outputs; this run is not a valid comparison.")
            measured = {
                "capture": args.capture, "repeat": rep, "batch": batch,
                "elapsed_s": elapsed_s, "output_tokens": sum(map(len, ids)),
                "output_tokens_s": sum(map(len, ids)) / elapsed_s,
                "output_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                # This is a configuration-based proxy, not measured dispatch/padding.
                "nominal_next_graph": min(x for x in resolved_sizes if x >= batch),
            }
            rows.write(json.dumps(measured) + "\n")
            rows.flush()
            print(json.dumps(measured), flush=True)
            wb.log({k: v for k, v in measured.items() if isinstance(v, (int, float))})
    write_json(args.out / "worker-resources-final.json", resource_snapshot(llm))
    artifact = wandb.Artifact(f"pilot-{args.capture}", type="measurements")
    for filename in ["measurements.jsonl", "environment.json", "requested-config.json", "gpu.csv", "package-versions.txt", "worker-resources-final.json"]:
        artifact.add_file(str(args.out / filename))
    wb.log_artifact(artifact)
    wb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--model", default=MODEL)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--prepared", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--capture", choices=["default", "coarse", "matched"], required=True)
    run_parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.model)
    else:
        run(args)


if __name__ == "__main__":
    main()
