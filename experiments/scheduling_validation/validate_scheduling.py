"""A controlled, streaming comparison of three admission policies on one GPU."""
import argparse
import asyncio
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

MODEL = 'google/gemma-4-12B'
MODEL_REVISION = '023679ed352de9bb66cc873c9009ce3482585c08'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
SEED = 20260908
REQUESTS = 240
RATE = 8.0
OUTPUT_TOKENS = 128
FIXED_LIMIT = 32
TOKEN_BUDGET = 20000
AGE_LIMIT_S = 2.0
TTFT_SLO_S = 2.0
TPOT_SLO_S = 0.1
POLICIES = ['default', 'fixed32', 'context_budget']


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(out):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                        revision=DATA_REVISION, split='test')
    corpus = '\n\n'.join(s for s in data['text'] if s.strip())
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    if len(tokens) < 2048:
        raise ValueError('Insufficient real source text.')
    # One declared workload shift: short requests, a long-context burst, then short.
    lengths = [128] * 96 + [2048] * 48 + [128] * 96
    for rep in range(3):
        rng = random.Random(SEED + rep)
        arrival = 0.0
        requests = []
        for i, length in enumerate(lengths):
            if i:
                arrival += rng.expovariate(RATE)
            start = rng.randrange(len(tokens) - length)
            requests.append({'id': i, 'arrival_s': arrival, 'input_tokens': length,
                             'prompt_token_ids': tokens[start:start + length]})
        save(out / f'trace-{rep}.json', requests)
    save(out / 'manifest.json', {
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': MODEL, 'model_revision': MODEL_REVISION,
        'dataset': 'Salesforce/wikitext', 'dataset_revision': DATA_REVISION,
        'dataset_config': 'wikitext-2-raw-v1', 'split': 'test',
        'prompt_construction': 'Contiguous token slices of concatenated real WikiText paragraphs; slices may cross paragraph boundaries.',
        'seed': SEED, 'requests_per_trace': REQUESTS, 'offered_rate_rps': RATE,
        'trace_phases': '96 short (128), 48 long (2048), 96 short (128)',
        'output_tokens': OUTPUT_TOKENS, 'policies': POLICIES,
        'fixed_limit': FIXED_LIMIT, 'estimated_token_budget': TOKEN_BUDGET,
        'age_protection_s': AGE_LIMIT_S, 'ttft_slo_s': TTFT_SLO_S,
        'tpot_slo_s': TPOT_SLO_S,
        'decision_rule': 'Candidate must improve SLO goodput by at least 5% over both controls in each of three balanced independent engine runs, without worsening p99 TTFT or p99 stream gap in any paired run.',
        'limitations': 'A controlled finite burst, not a production trace. Token reservation uses the declared output budget, not a prediction of unknown production output length. NVML busy time is not SM utilization.',
        'trace_sha256': {str(rep): hashlib.sha256((out / f'trace-{rep}.json').read_bytes()).hexdigest() for rep in range(3)},
    })


async def replay(engine, params, requests, policy, out, run_prefix):
    import numpy as np
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start = time.perf_counter()
    finished = False
    telemetry = []
    pending = []
    active = {}
    records = []
    next_request = 0
    reserved = 0

    async def sample_gpu():
        while not finished:
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            telemetry.append({'time_s': time.perf_counter() - start,
                              'gpu_busy_percent': u.gpu, 'memory_busy_percent': u.memory,
                              'device_memory_bytes': m.used, 'pending_requests': len(pending),
                              'admitted_requests': len(active), 'reserved_token_estimate': reserved})
            await asyncio.sleep(0.2)

    async def execute(request, admitted):
        chunks = []
        ids = []
        async for output in engine.generate(
                {'prompt_token_ids': request['prompt_token_ids']}, params,
                request_id=f"{run_prefix}-{request['id']}"):
            new_ids = list(output.outputs[0].token_ids)
            if new_ids:
                stamp = time.perf_counter() - start
                chunks.append({'time_s': stamp, 'tokens': len(new_ids)})
                ids.extend(new_ids)
        completed = time.perf_counter() - start
        if len(ids) != OUTPUT_TOKENS or not chunks:
            raise ValueError(f"Incomplete streaming request: {request['id']}, {len(ids)} tokens")
        ttft = chunks[0]['time_s'] - request['arrival_s']
        tpot = (chunks[-1]['time_s'] - chunks[0]['time_s']) / (OUTPUT_TOKENS - 1)
        return {'id': request['id'], 'input_tokens': request['input_tokens'],
                'arrival_s': request['arrival_s'], 'admitted_s': admitted,
                'completed_s': completed, 'admission_wait_s': admitted - request['arrival_s'],
                'ttft_s': ttft, 'tpot_s': tpot,
                'slo_met': ttft <= TTFT_SLO_S and tpot <= TPOT_SLO_S,
                'stream_chunks': chunks, 'output_token_ids': ids,
                'output_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}

    sampler = asyncio.create_task(sample_gpu())
    while len(records) < len(requests):
        now = time.perf_counter() - start
        while next_request < len(requests) and requests[next_request]['arrival_s'] <= now:
            pending.append(requests[next_request])
            next_request += 1
        for key in [k for k, (task, cost) in active.items() if task.done()]:
            task, cost = active.pop(key)
            records.append(task.result())
            reserved -= cost
        while pending:
            selected = 0
            if policy == 'fixed32' and len(active) >= FIXED_LIMIT:
                break
            if policy == 'context_budget':
                room = TOKEN_BUDGET - reserved
                oldest_cost = pending[0]['input_tokens'] + OUTPUT_TOKENS
                if now - pending[0]['arrival_s'] >= AGE_LIMIT_S:
                    if oldest_cost > room:
                        break  # Reserve the next available room for the aged request.
                else:
                    selected = next((i for i, r in enumerate(pending)
                                     if r['input_tokens'] + OUTPUT_TOKENS <= room), None)
                    if selected is None:
                        break
            request = pending.pop(selected)
            cost = request['input_tokens'] + OUTPUT_TOKENS
            admitted = time.perf_counter() - start
            task = asyncio.create_task(execute(request, admitted))
            active[request['id']] = (task, cost)
            reserved += cost
        if len(records) == len(requests):
            break
        timeout = None if next_request == len(requests) else max(0, requests[next_request]['arrival_s'] - (time.perf_counter() - start))
        if active:
            await asyncio.wait([v[0] for v in active.values()], timeout=timeout,
                               return_when=asyncio.FIRST_COMPLETED)
        elif timeout is not None:
            await asyncio.sleep(timeout)
        else:
            raise ValueError('Admission deadlock: the declared budget cannot fit a request.')
    duration = time.perf_counter() - start
    finished = True
    await sampler
    pynvml.nvmlShutdown()
    records.sort(key=lambda r: r['id'])
    if [r['id'] for r in records] != list(range(len(requests))):
        raise ValueError('Missing or duplicate completed requests.')
    gaps = [b['time_s'] - a['time_s'] for r in records
            for a, b in zip(r['stream_chunks'], r['stream_chunks'][1:])]
    summary = {'policy': policy, 'requests': len(records), 'duration_s': duration,
               'output_tokens_s': len(records) * OUTPUT_TOKENS / duration,
               'slo_goodput_rps': sum(r['slo_met'] for r in records) / duration,
               'slo_attainment_percent': 100 * sum(r['slo_met'] for r in records) / len(records),
               'p99_ttft_s': float(np.percentile([r['ttft_s'] for r in records], 99)),
               'p99_tpot_s': float(np.percentile([r['tpot_s'] for r in records], 99)),
               'p99_stream_gap_s': float(np.percentile(gaps, 99)),
               'max_stream_chunk_tokens': max(c['tokens'] for r in records for c in r['stream_chunks']),
               'mean_gpu_busy_percent': float(np.mean([r['gpu_busy_percent'] for r in telemetry])),
               'max_device_memory_bytes': max(r['device_memory_bytes'] for r in telemetry)}
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'requests.json', records)
    save(out / 'telemetry.json', telemetry)
    save(out / 'summary.json', summary)
    return summary


async def run(prepared, out, rep):
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    import wandb
    manifest = json.loads((prepared / 'manifest.json').read_text())
    trace_file = prepared / f'trace-{rep}.json'
    if hashlib.sha256(trace_file.read_bytes()).hexdigest() != manifest['trace_sha256'][str(rep)]:
        raise ValueError('Trace changed after preparation.')
    requests = json.loads(trace_file.read_text())
    out.mkdir(parents=True, exist_ok=False)
    order = POLICIES[rep:] + POLICIES[:rep]
    settings = dict(model=MODEL, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
                    dtype='bfloat16', tensor_parallel_size=1, seed=SEED,
                    max_model_len=4096, max_num_seqs=128, max_num_batched_tokens=4096,
                    gpu_memory_utilization=0.85, enable_prefix_caching=False,
                    enable_chunked_prefill=True, language_model_only=True,
                    compilation_config={'max_cudagraph_capture_size': 256},
                    disable_log_stats=False)
    wb = wandb.init(project='saturatellm-feasibility', group=os.environ['SATURATE_GROUP'],
                    name=f'scheduling-repetition-{rep}', config={'manifest': manifest, 'engine': settings, 'order': order})
    (out / 'wandb-url.txt').write_text(wb.url + '\n')
    (out / 'package-versions.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + '\n')
    (out / 'gpu.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version,power.limit', '--format=csv'], text=True))
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
    params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS,
                            ignore_eos=True, output_kind=RequestOutputKind.DELTA)
    save(out / 'environment.json', {'manifest': manifest, 'requested_engine': settings, 'order': order,
                                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   'python': sys.version, 'vllm_version': importlib.metadata.version('vllm'),
                                   'resolved_graph_sizes': engine.vllm_config.compilation_config.cudagraph_capture_sizes})
    # Common untimed warmup covers both context lengths and several concurrency levels.
    async def warm_one(r, key):
        async for _ in engine.generate({'prompt_token_ids': r['prompt_token_ids']}, params, request_id=key):
            pass
    for length in [128, 2048]:
        sample = next(r for r in requests if r['input_tokens'] == length)
        for batch in [8, 32]:
            await asyncio.gather(*(warm_one(sample, f'warm-{length}-{batch}-{i}') for i in range(batch)))
    for policy in order:
        summary = await replay(engine, params, requests, policy, out / policy, f'r{rep}-{policy}')
        summary['repeat'] = rep
        wb.log(summary)
        print(json.dumps(summary), flush=True)
    engine.shutdown()
    artifact = wandb.Artifact(f'scheduling-repeat-{rep}', type='measurements')
    artifact.add_dir(str(out))
    wb.log_artifact(artifact)
    wb.finish()
    save(out / 'complete.json', {'repeat': rep, 'status': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--prepared', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--repeat', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        asyncio.run(run(args.prepared, args.out, args.repeat))
