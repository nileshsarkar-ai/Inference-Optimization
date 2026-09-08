"""Run the three predeclared configurations sequentially in fresh processes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import wandb

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
args = parser.parse_args()
os.chdir(root)
os.environ["WANDB_API_KEY"] = Path('/home/.saturatellm-auth/wandb.key').read_text().strip()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
group = args.model.split('/')[-1] + datetime.datetime.now(datetime.timezone.utc).strftime('-a100-%Y%m%dT%H%M%SZ')
os.environ["SATURATE_GROUP"] = group
result = root / "results" / group
result.mkdir(parents=True)
# Fail early on auth rather than silently running without the requested tracking.
wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
print("W&B authentication verified. Group:", group, flush=True)
subprocess.run([sys.executable, "pilot.py", "prepare", "--model", args.model, "--out", str(result / "prepared")], check=True)
# Predeclared order; small effects require a later order-reversal check.
for capture in ['default', 'matched', 'coarse']:
    with (result / f'{capture}.log').open('w') as log:
        subprocess.run([sys.executable, "pilot.py", "run", "--prepared", str(result / "prepared"),
                        "--capture", capture, "--out", str(result / capture)], stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Completed:", capture, flush=True)
subprocess.run([sys.executable, "analyze.py", *[str(result / n) for n in ['default', 'coarse', 'matched']],
                "--out", str(result / "figures")], check=True)
print("Timing comparison and figures complete:", result, flush=True)
with (result / 'profile-default.log').open('w') as log:
    subprocess.run([sys.executable, "pilot.py", "run", "--prepared", str(result / "prepared"),
                    "--capture", "default", "--profile", "--out", str(result / "profile-default")],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (result / 'profile-matched.log').open('w') as log:
    subprocess.run([sys.executable, "pilot.py", "run", "--prepared", str(result / "prepared"),
                    "--capture", "matched", "--profile", "--out", str(result / "profile-matched")],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
(result / 'complete.json').write_text(json.dumps({'status':'complete','group':group})+'\n')
print('Controlled experiment complete:', result, flush=True)
