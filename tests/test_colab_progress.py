"""Verify live child output, quiet-stage heartbeats and error propagation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import time
from threading import Event
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_progress import run_with_progress


class ProgressTests(unittest.TestCase):
    def test_output_heartbeat_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            out=io.StringIO()
            with redirect_stdout(out):
                run_with_progress([sys.executable,'-u','-c',
                    'import time; print("Extract SR/ZAB", flush=True); time.sleep(0.2); print("saved", flush=True)'],
                    path, heartbeat_seconds=.05)
            self.assertIn('Extract SR/ZAB',out.getvalue())
            self.assertIn('[Still running',out.getvalue())
            self.assertIn('saved',out.getvalue())
            status=json.loads(path.read_text())
            self.assertEqual(status['status'],'completed')
            self.assertEqual(status['last_output'],'saved')
            self.assertEqual(status['returncode'],0)

    def test_child_error_is_visible_and_not_marked_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            out=io.StringIO()
            with redirect_stdout(out),self.assertRaises(subprocess.CalledProcessError):
                run_with_progress([sys.executable,'-u','-c',
                    'import sys; print("source unavailable",file=sys.stderr); sys.exit(7)'],path)
            self.assertIn('source unavailable',out.getvalue())
            status=json.loads(path.read_text())
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['returncode'],7)

    def test_monitor_write_failure_does_not_stop_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            path.mkdir()  # replacing this directory with a file must fail
            out=io.StringIO()
            with redirect_stdout(out):
                run_with_progress([sys.executable,'-u','-c','print("extraction continued")'],path)
            self.assertIn('Progress-file update unavailable',out.getvalue())
            self.assertIn('extraction continued',out.getvalue())

    def test_slow_drive_write_does_not_block_live_output(self):
        release=Event();observed=[];start=time.monotonic()
        class Output(io.StringIO):
            def write(self,text):
                if 'live-child-output' in text:
                    observed.append(time.monotonic()-start)
                    release.set()
                return super().write(text)
        original=Path.write_text
        def slow_write(path,*args,**kwargs):
            release.wait(timeout=1)
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch.object(Path,'write_text',slow_write),redirect_stdout(Output()):
            run_with_progress([sys.executable,'-u','-c','print("live-child-output")'],Path(directory)/'progress.json')
        self.assertTrue(observed)
        self.assertLess(observed[0],.8)


if __name__=='__main__':
    unittest.main()
