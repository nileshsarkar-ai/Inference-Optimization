"""Run two modern-model comparisons after the existing A100 job exits."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS_EXIT = Path('/home/jl-runs/r_280f8ded/exit_code')
MODELS = ['google/gemma-4-12B', 'Qwen/Qwen3.5-9B']

print('Waiting for the existing 7B job to release the GPU.', flush=True)
while not PREVIOUS_EXIT.exists():
    time.sleep(20)
print('Previous job exit code:', PREVIOUS_EXIT.read_text().strip(), flush=True)
statuses = []
for model in MODELS:
    print('Starting:', model, flush=True)
    # A failed model is recorded; the independent next model still gets its own run.
    completed = subprocess.run([sys.executable, str(ROOT / 'orchestrate.py'), '--model', model], cwd=ROOT)
    statuses.append({'model': model, 'exit_code': completed.returncode})
    (ROOT / 'queue-status.json').write_text(json.dumps(statuses, indent=2) + '\n')
    print('Finished:', model, 'exit code:', completed.returncode, flush=True)
sys.exit(1 if any(s['exit_code'] != 0 for s in statuses) else 0)
