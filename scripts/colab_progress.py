"""Relay child output into Colab and publish a liveness heartbeat to Drive.

This wrapper does not alter extraction code or its checkpoint fingerprint.
A heartbeat confirms the child is alive, not that an EEG stage is advancing.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
import time


def run_with_progress(command, status_path, heartbeat_seconds=30):
    if heartbeat_seconds <= 0:
        raise ValueError('heartbeat_seconds must be positive')
    status_path = Path(status_path)
    started = time.monotonic()
    last_output_clock=started
    state = {'status':'starting', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'last_output':'Waiting for extraction output', 'last_output_at_utc':None}

    # Drive/FUSE can block rather than raise OSError. Keep it off the
    # thread that relays logs and prints heartbeats into the notebook.
    pending=Queue(maxsize=1)
    def persist():
        while True:
            snapshot=pending.get()
            try:
                status_path.parent.mkdir(parents=True, exist_ok=True)
                tmp=status_path.with_suffix('.tmp.json')
                tmp.write_text(json.dumps(snapshot, indent=2)+'\n')
                tmp.replace(status_path)
            except OSError as error:
                print(f'Progress-file update unavailable: {error}', flush=True)
            if snapshot['status'] in {'completed','failed','interrupted'}:
                return
    publisher=Thread(target=persist,daemon=True)
    publisher.start()

    def save(status, process=None):
        state.update(status=status, heartbeat_at_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic()-started, 1),
                     child_pid=process.pid if process else None,
                     returncode=process.poll() if process else None)
        state['seconds_since_last_output']=round(time.monotonic()-last_output_clock,1)
        # Keep only the newest snapshot if a previous Drive write is blocked.
        try:
            pending.get_nowait()
        except Empty:
            pass
        pending.put_nowait(dict(state))

    print('Starting word EEG extraction. Live logs follow; a status message appears every 30 seconds during quiet stages.', flush=True)
    save('starting')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    queue = Queue()

    def read_output():
        try:
            for line in process.stdout:
                queue.put(line)
        finally:
            queue.put(None)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    next_heartbeat = time.monotonic()+heartbeat_seconds
    save('running', process)
    try:
        while True:
            try:
                line = queue.get(timeout=max(.01,next_heartbeat-time.monotonic()))
            except Empty:
                line = ''
            if line is None:
                break
            if line:
                print(line, end='', flush=True)
                state['last_output'] = line.strip()
                state['last_output_at_utc'] = datetime.now(timezone.utc).isoformat()
                last_output_clock=time.monotonic()
            if time.monotonic() >= next_heartbeat:
                minutes = (time.monotonic()-started)/60
                print(f'[Still running | {minutes:.1f} min elapsed] Last log: {state["last_output"]}', flush=True)
                print(f'  No new log for {time.monotonic()-last_output_clock:.0f}s. A live child is not proof of advancing extraction.', flush=True)
                save('running' if process.poll() is None else 'finishing', process)
                next_heartbeat = time.monotonic()+heartbeat_seconds
        code = process.wait()
        save('completed' if code==0 else 'failed', process)
        if code:
            raise subprocess.CalledProcessError(code, command)
        print('Extraction command finished successfully. The next cell displays the saved report and figures.', flush=True)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        save('interrupted', process)
        print('Run interrupted. Completed reader checkpoints remain saved.', flush=True)
        raise
    finally:
        reader.join(timeout=1)
        process.stdout.close()
        publisher.join(timeout=2)  # a blocked Drive upload must not hang notebook completion
