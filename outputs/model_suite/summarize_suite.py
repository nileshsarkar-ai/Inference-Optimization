"""Summarize completed model comparisons; never substitute missing measurements."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folders', nargs='+', type=Path)
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
studies = []
flat = []
for folder in args.folders:
    configs = {}
    for name in ['default', 'matched', 'coarse']:
        env = json.loads((folder / name / 'environment.json').read_text())
        rows = [json.loads(line) for line in (folder / name / 'measurements.jsonl').read_text().splitlines()]
        log = (folder / f'{name}.log').read_text()
        batches = env['manifest']['batches']
        if len(rows) != len(batches) * 3:
            raise ValueError(f'Incomplete comparison: {folder} / {name}')
        stats = []
        for batch in batches:
            selected = [r for r in rows if r['batch'] == batch]
            if {r['repeat'] for r in selected} != {0, 1, 2} or len(selected) != 3:
                raise ValueError(f'Missing or duplicate repetitions: {folder} / {name} / {batch}')
            values = [r['output_tokens_s'] for r in selected]
            stat = {'batch': batch, 'mean': statistics.mean(values), 'min': min(values), 'max': max(values), 'values': values}
            stats.append(stat)
            flat.append({'model': env['manifest']['model'], 'capture': name, 'batch': batch,
                         'mean_output_tokens_s': stat['mean'], 'min_output_tokens_s': stat['min'],
                         'max_output_tokens_s': stat['max'], 'repetitions': 3})
        memory = re.findall(r'Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB', log)
        occupancy = [float(v) for v in re.findall(r'GPU KV cache usage: ([\d.]+)%', log)]
        cache_tokens = re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)
        configs[name] = {'environment': env, 'stats': stats,
                         'wandb_url': (folder / name / 'wandb-url.txt').read_text().strip(),
                         'graph_memory_gib': float(memory[-1]) if memory else None,
                         'max_logged_kv_usage_percent': max(occupancy) if occupancy else None,
                         'engine_reported_kv_tokens': int(cache_tokens[-1].replace(',', '')) if cache_tokens else None,
                         'measurement_source': str((folder / name / 'measurements.jsonl').resolve())}
    ref = configs['default']['environment']
    for c in configs.values():
        for key in ['manifest', 'script_sha256', 'vllm_version', 'torch_version']:
            if c['environment'][key] != ref[key]:
                raise ValueError(f'Uncontrolled comparison: {folder} / {key}')
    delta = [100 * (m['mean'] / d['mean'] - 1) for m, d in zip(configs['matched']['stats'], configs['default']['stats'])]
    studies.append({'model': ref['manifest']['model'], 'folder': str(folder.resolve()),
                    'configs': configs, 'matched_delta_percent': delta,
                    'hash_mismatches': json.loads((folder / 'figures/output-consistency.json').read_text()),
                    'interpretation': 'Within-model descriptive fixed-batch comparison. Includes prefill and host work. Logged KV occupancy includes warmup and is not GPU utilization. No independent-process significance or model-quality claim.'})
args.out.mkdir(parents=True, exist_ok=False)
(args.out / 'suite.json').write_text(json.dumps(studies, indent=2) + '\n')
with (args.out / 'summary.csv').open('w') as dest:
    writer = csv.DictWriter(dest, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
print('Summarized', len(studies), 'complete model comparisons into', args.out)
