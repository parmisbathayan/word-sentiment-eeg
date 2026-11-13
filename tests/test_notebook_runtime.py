"""Runtime guards must precede setup and reject the wrong hardware."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GPU = ('cnn', 'zuco_paper_analyze_gpu', 'zuco_paper_recover_analyze_gpu',
       'teco_paper_analyze_gpu', 'zuco_recovered_followups_cnn', 'zuco_agent_paper_subset')
CPU = ('zuco_paper_prepare_cpu', 'zuco_paper_recover_prepare_cpu',
       'teco_paper_prepare_cpu', 'zuco_recovered_followups',
       'zuco_sliding_reader_maps', 'zuco_paper_pilot')

class NotebookRuntimeTests(unittest.TestCase):
    def test_all_guards(self):
        for name in GPU + CPU:
            book = json.loads((ROOT / 'notebooks' / (name + '.ipynb')).read_text())
            cell = next(c for c in book['cells'] if c['cell_type'] == 'code')
            self.assertEqual(cell['id'], 'runtime-check')
            self.assertEqual(cell['outputs'], [])
            code = ''.join(cell['source'])
            ast.parse(code)
            required_gpu = name in GPU
            self.assertEqual(book['metadata'].get('accelerator'), 'GPU' if required_gpu else None)
            if required_gpu:
                self.assertEqual(book['metadata']['colab']['gpuType'], 'T4')
            cases = [
                (FileNotFoundError(), {}, not required_gpu),
                (subprocess.CompletedProcess([], 0, 'Tesla T4\n', ''), {}, required_gpu),
                (subprocess.CompletedProcess([], 0, 'NVIDIA A100\n', ''), {}, required_gpu),
                (subprocess.CompletedProcess([], 1, '', 'failed'), {}, False),
                (subprocess.TimeoutExpired('nvidia-smi', 15), {}, False),
                (FileNotFoundError(), {'COLAB_TPU_ADDR': 'test'}, False),
            ]
            for result, env, allowed in cases:
                with self.subTest(notebook=name, result=str(result), env=env):
                    options = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
                    with patch.dict(os.environ, env, clear=True), patch('subprocess.run', **options), contextlib.redirect_stdout(io.StringIO()):
                        if allowed:
                            exec(code, {})
                        else:
                            with self.assertRaises(RuntimeError):
                                exec(code, {})

if __name__ == '__main__':
    unittest.main()
