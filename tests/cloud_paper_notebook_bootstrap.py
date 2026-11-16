"""Cloud-only integration check: no local dependency materialization."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
book = json.loads((repo / 'notebooks/zuco_paper_classification.ipynb').read_text())
source = ''.join(next(c for c in book['cells'] if c.get('id') == 'paper-setup')['source'])
block = source.split('# BOOTSTRAP_START: also exercised verbatim in cloud CI.\n')[1].split('# BOOTSTRAP_END')[0]
with tempfile.TemporaryDirectory() as temporary:
    env = Path(temporary) / 'scientific'
    # Simulate the incomplete environment left by Colab's failed ensurepip.
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True)
    block = block.replace("Path('/content/word-eeg-paper-classification-python')", f'Path({str(env)!r})')
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    scope = dict(Path=Path, sys=sys, os=os, subprocess=subprocess, REPO=repo)
    exec(compile(block, '<actual-notebook-bootstrap>', 'exec'), scope)
    # A second setup must reuse the environment successfully.
    exec(compile(block, '<notebook-bootstrap-rerun>', 'exec'), scope)
print('Notebook scientific environment and rerun verified.')
