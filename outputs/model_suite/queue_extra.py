"""Complete the current-generation small/large Gemma comparison sequentially."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS_EXIT = Path('/home/jl-runs/r_739e084f/exit_code')
print('Waiting for the Gemma 12B and Qwen3.5 queue to release the GPU.', flush=True)
while not PREVIOUS_EXIT.exists():
    time.sleep(20)
print('Previous queue exit code:', PREVIOUS_EXIT.read_text().strip(), flush=True)
subprocess.run([sys.executable, str(ROOT / 'orchestrate.py'), '--model', 'google/gemma-4-E4B'], cwd=ROOT, check=True)
