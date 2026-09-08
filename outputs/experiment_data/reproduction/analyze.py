"""Plot actual pilot files only; no example or fallback measurements."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs=3, type=Path, help="default, coarse, matched run directories")
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
data = {}
environments = {}
for folder in args.runs:
    env = json.loads((folder / "environment.json").read_text())
    rows = [json.loads(line) for line in (folder / "measurements.jsonl").read_text().splitlines()]
    name = env["capture"]
    if name in data:
        raise ValueError("Provide one run of each capture configuration.")
    data[name] = rows
    environments[name] = env
if set(data) != {"default", "coarse", "matched"}:
    raise ValueError("All three configurations are required.")
reference = environments["default"]
for env in environments.values():
    for key in ["manifest", "vllm_version", "torch_version", "script_sha256"]:
        if env[key] != reference[key]:
            raise ValueError(f"Runs differ in {key}; do not pool them.")

batches = reference["manifest"]["batches"]
colors = {"default": "#222222", "coarse": "#8C96A5", "matched": "#2878D6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9.4, 5.2), layout="constrained")
summary = []
for name in ["default", "coarse", "matched"]:
    means = []
    for b in batches:
        cases = [x for x in data[name] if x["batch"] == b]
        if len(cases) != 3 or {x["repeat"] for x in cases} != {0, 1, 2}:
            raise ValueError(f"Missing or duplicate repetitions: {name}, batch {b}")
        values = [x["output_tokens_s"] for x in cases]
        mean = statistics.mean(values)
        means.append(mean)
        ax.scatter([b] * len(values), values, color=colors[name], alpha=0.45, s=20)
        summary.append({"capture": name, "batch": b, "mean_output_tokens_s": mean,
                        "min_output_tokens_s": min(values), "max_output_tokens_s": max(values),
                        "repeats": len(values)})
    ax.plot(batches, means, color=colors[name], marker="o", label=name)
ax.set(xlabel="Requests in fixed batch", ylabel="Successful output tokens/s",
       title="CUDA-graph configuration sensitivity\nPoints: measured repetitions · Lines: within-process means")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
args.out.mkdir(parents=True, exist_ok=False)
fig.savefig(args.out / "throughput.png", dpi=240)
fig.savefig(args.out / "throughput.pdf")
with (args.out / "summary.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

hashes = {name: {(x["batch"], x["repeat"]): x["output_ids_sha256"] for x in rows} for name, rows in data.items()}
differences = {name: sum(h != hashes["default"][key] for key, h in mapping.items()) for name, mapping in hashes.items()}
(args.out / "output-consistency.json").write_text(json.dumps(differences, indent=2) + "\n")
(args.out / "provenance.json").write_text(json.dumps({
    "source_files": [str((p / "measurements.jsonl").resolve()) for p in args.runs],
    "environments": environments,
    "interpretation": "Descriptive pilot repetitions, not independent-process confidence intervals. Includes prefill. Does not establish p99 serving latency, SM saturation, generalization or selector novelty.",
}, indent=2) + "\n")
