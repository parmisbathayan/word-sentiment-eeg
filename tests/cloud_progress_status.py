"""Heartbeats never replace/delete Drive files; scientific resume is separate."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from colab_progress import run_with_progress
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'status.json'
    command=[sys.executable,'-u','-c',"import time;print('one',flush=True);time.sleep(.12);print('two',flush=True)"]
    with patch.object(Path,'replace',side_effect=AssertionError('No rename for advisory status')):
        run_with_progress(command,path,heartbeat_seconds=.03,activity='Synthetic heartbeat test')
    assert json.loads(path.read_text())['status']=='completed'
    assert list(Path(tmp).iterdir())==[path]
print('PASS single in-place heartbeat, no temporary/deleted status generations')
