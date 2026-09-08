"""Publication-style plots from completed measured-study summaries."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('summary', type=Path)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
studies = json.loads(args.summary.read_text())
colors = {'default': '#222222', 'matched': '#2878D6', 'coarse': '#8995A5'}
labels = {'default': 'Current default', 'matched': 'Manual matched set', 'coarse': 'Coarse grid'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, len(studies), figsize=(5 * len(studies), 4.8), squeeze=False)
for ax, study in zip(axes[0], studies):
    for name in ['default', 'matched', 'coarse']:
        stats = study['configs'][name]['stats']
        batches = [s['batch'] for s in stats]
        for s in stats:
            ax.scatter([s['batch']] * len(s['values']), s['values'], s=15,
                       color=colors[name], alpha=0.45, zorder=3)
        ax.plot(batches, [s['mean'] for s in stats], color=colors[name],
                marker='o', markersize=4, linewidth=1.6, label=labels[name])
    ax.set_title(study['model'].split('/')[-1], fontsize=13, fontweight='bold')
    ax.set_xlabel('Requests submitted in fixed batch')
    ax.set_ylabel('Completed output tokens/s')
    ax.set_ylim(bottom=0)
    ax.set_xticks(batches)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', alpha=0.2)
handles, names = axes[0][0].get_legend_handles_labels()
fig.legend(handles, names, loc='lower center', bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
fig.suptitle('CUDA-graph capture configuration on A100 40 GB', fontsize=16, fontweight='bold')
fig.text(0.5, 0.9, 'Points: all three warmed repetitions · Lines: means · Compare configurations within each model', ha='center', fontsize=10)
fig.text(0.5, 0.012, 'Source: SaturateLLM raw measurements and W&B · wandb.ai/nileshsarkar-ai/saturatellm-feasibility', ha='center', fontsize=8)
fig.subplots_adjust(left=0.065, right=0.99, top=0.8, bottom=0.24, wspace=0.32)
args.out.mkdir(parents=True, exist_ok=False)
fig.savefig(args.out / 'throughput_all_models.png', dpi=240)
fig.savefig(args.out / 'throughput_all_models.pdf')
(args.out / 'provenance.json').write_text(json.dumps({
    'summary_source': str(args.summary.resolve()),
    'measurement_sources': [s['configs'][c]['measurement_source'] for s in studies for c in ['default','matched','coarse']],
    'interpretation': 'Unprofiled end-to-end fixed-batch throughput, includes prefill and host work. No independent-run confidence interval or model-quality comparison.'
}, indent=2) + '\n')
