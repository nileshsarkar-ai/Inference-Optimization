"""Run the authorized scheduling validation after the current model queue releases the GPU."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS_EXIT = Path('/home/jl-runs/r_7a2c52f8/exit_code')
print('Waiting for the current Gemma E4B study to release the A100.', flush=True)
while not PREVIOUS_EXIT.exists():
    time.sleep(20)
print('Previous study exit code:', PREVIOUS_EXIT.read_text().strip(), flush=True)
os.chdir(ROOT)
os.environ['WANDB_API_KEY'] = Path('/home/.saturatellm-auth/wandb.key').read_text().strip()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
group = datetime.datetime.now(datetime.timezone.utc).strftime('scheduling-a100-%Y%m%dT%H%M%SZ')
os.environ['SATURATE_GROUP'] = group
out = ROOT / 'results' / group
out.mkdir(parents=True)
source = out / 'reproduction'
source.mkdir()
for name in ['validate_scheduling.py', 'analyze_scheduling.py', 'queue_validation.py', 'PROTOCOL.md']:
    shutil.copy2(ROOT / name, source / name)
print('Scheduling validation group:', group, flush=True)
subprocess.run([sys.executable, 'validate_scheduling.py', 'prepare', '--out', str(out / 'prepared')], check=True)
for rep in range(3):
    with (out / f'repeat-{rep}.log').open('w') as log:
        subprocess.run([sys.executable, 'validate_scheduling.py', 'run', '--repeat', str(rep),
                        '--prepared', str(out / 'prepared'), '--out', str(out / f'repeat-{rep}')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed independent repetition:', rep, flush=True)
subprocess.run([sys.executable, 'analyze_scheduling.py', str(out)], check=True)
(out / 'complete.json').write_text(json.dumps({'status': 'complete', 'group': group}) + '\n')
print('Scheduling validation complete:', out, flush=True)
