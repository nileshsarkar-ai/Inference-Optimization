"""Analyze every declared policy and repetition, including negative results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
args = parser.parse_args()
policies = ['default', 'fixed32', 'context_budget']
labels = ['vLLM default', 'Fixed limit 32', 'Context budget']
colors = ['#222222', '#8995A5', '#2878D6']
rows = []
outputs = {}
references = []
environments = []
class_results = []
for rep in range(3):
    folder = args.root / f'repeat-{rep}'
    if not (folder / 'complete.json').exists():
        raise ValueError(f'Incomplete engine repetition: {rep}')
    env = json.loads((folder / 'environment.json').read_text())
    environments.append(env)
    references.append((folder / 'wandb-url.txt').read_text().strip())
    for policy in policies:
        data = folder / policy
        summary = json.loads((data / 'summary.json').read_text())
        records = json.loads((data / 'requests.json').read_text())
        if summary['requests'] != 240 or len(records) != 240:
            raise ValueError('Missing requests in the declared test.')
        summary['repeat'] = rep
        rows.append(summary)
        outputs[(rep, policy)] = records
        for length in [128, 2048]:
            selected = [r for r in records if r['input_tokens'] == length]
            class_results.append({'repeat': rep, 'policy': policy, 'input_tokens': length,
                                  'requests': len(selected),
                                  'p99_ttft_s': float(np.percentile([r['ttft_s'] for r in selected], 99)),
                                  'slo_attainment_percent': 100 * sum(r['slo_met'] for r in selected) / len(selected)})
for env in environments[1:]:
    for key in ['manifest', 'requested_engine', 'script_sha256', 'vllm_version', 'resolved_graph_sizes']:
        if env[key] != environments[0][key]:
            raise ValueError(f'Uncontrolled engine comparison: {key}')

lookup = {(r['repeat'], r['policy']): r for r in rows}
comparisons = []
for rep in range(3):
    candidate = lookup[(rep, 'context_budget')]
    for baseline in ['default', 'fixed32']:
        control = lookup[(rep, baseline)]
        goodput_pass = candidate['slo_goodput_rps'] >= 1.05 * control['slo_goodput_rps'] and candidate['slo_goodput_rps'] > control['slo_goodput_rps']
        latency_pass = candidate['p99_ttft_s'] <= control['p99_ttft_s'] and candidate['p99_stream_gap_s'] <= control['p99_stream_gap_s']
        comparisons.append({'repeat': rep, 'baseline': baseline,
                            'goodput_delta_percent': (100 * (candidate['slo_goodput_rps'] / control['slo_goodput_rps'] - 1)) if control['slo_goodput_rps'] else None,
                            'goodput_rule_pass': goodput_pass, 'latency_rule_pass': latency_pass,
                            'pass': goodput_pass and latency_pass})
hash_differences = {f'repeat-{rep}/{policy}': sum(a['output_sha256'] != b['output_sha256'] for a, b in zip(outputs[(rep, 'default')], outputs[(rep, policy)]))
                    for rep in range(3) for policy in ['fixed32', 'context_budget']}
result = {'status': 'complete', 'predeclared_rule_pass': all(c['pass'] for c in comparisons),
          'comparisons': comparisons, 'output_hash_mismatches': hash_differences,
          'wandb_runs': references, 'observations': rows, 'per_class': class_results,
          'provenance': {'root': str(args.root.resolve()), 'script_sha256': environments[0]['script_sha256'],
                         'manifest': environments[0]['manifest']},
          'interpretation': 'Three independent engine repetitions with balanced policy order. Success is a predeclared practical threshold, not a significance test. NVML busy time is not SM utilization. No model-quality or non-LLM generalization claim.'}
out = args.root / 'analysis'
out.mkdir(exist_ok=False)
(out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
with (out / 'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
metrics = [('slo_goodput_rps', 'SLO goodput (requests/s)'),
           ('output_tokens_s', 'Completed output tokens/s'),
           ('p99_ttft_s', 'p99 time to first token (s)'),
           ('mean_gpu_busy_percent', 'Mean NVML GPU busy (%)')]
for ax, (metric, title) in zip(axes.flat, metrics):
    for i, policy in enumerate(policies):
        values = [r[metric] for r in rows if r['policy'] == policy]
        ax.bar(i, statistics.mean(values), color=colors[i], width=0.58, alpha=0.75)
        ax.scatter([i - .08, i, i + .08], values, color=colors[i], edgecolors='white', linewidths=.5, s=36, zorder=4)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(3), labels, fontsize=10)
    ax.set_ylim(bottom=0)
    if metric == 'mean_gpu_busy_percent':
        ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=.2)
fig.suptitle('A100 scheduling validation: identical phased request traces', fontsize=16, fontweight='bold')
fig.text(.5, .928, 'Dots: three independent engine repetitions · Bars: means · TTFT includes admission waiting', ha='center', fontsize=10)
fig.text(.5, .027, 'SLO: TTFT ≤ 2 s and average TPOT ≤ 100 ms · NVML busy time is not achieved SM utilization', ha='center', fontsize=9)
fig.text(.5, .009, 'Source: saved request/telemetry records · wandb.ai/nileshsarkar-ai/saturatellm-feasibility', ha='center', fontsize=8)
fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.11, hspace=.38, wspace=.3)
fig.savefig(out / 'scheduling_results.png', dpi=220)
fig.savefig(out / 'scheduling_results.pdf')
print(json.dumps({'predeclared_rule_pass': result['predeclared_rule_pass'], 'output': str(out)}), flush=True)
