"""Real CLI saved-trial handoff for both shapes; no extraction or slow full fit."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import atomic_npz,write_json,file_digest

rng=np.random.default_rng(4)
with tempfile.TemporaryDirectory() as tmp:
    for dataset,C in [('zuco',105),('teco',126)]:
        root=Path(tmp)/dataset;root.mkdir();readers=['A','B','C'];marks={}
        for subject in readers:
            folder=root/'readers'/subject;folder.mkdir(parents=True)
            x=rng.normal(size=((2,12,C,201) if dataset=='zuco' else (12,C,201))).astype(np.float32)
            neg=np.arange(6);pos=np.arange(6,12);d=x[:,pos]-x[:,neg] if dataset=='zuco' else x[pos]-x[neg]
            atomic_npz(folder/'prepared_trials.npz',signal=x,labels=np.r_[np.zeros(6),np.ones(6)].astype(np.int8),negative_pair_index=neg,positive_pair_index=pos)
            atomic_npz(folder/'statistics.npz',pair_sum=d.sum(axis=1 if dataset=='zuco' else 0,dtype=np.float64),pair_squares=(d.astype(float)**2).sum(axis=1 if dataset=='zuco' else 0),pairs=np.array(6),times_ms=np.arange(-75,126)*8,counts=np.array([6,6]))
            (folder/'accepted_trial_metadata.csv.gz').write_bytes(b'SYNTHETIC metadata checkpoint')
            write_json(folder/'prepared.json',dict(status='prepared',subject=subject,accepted_counts=[6,6],counts=[6,6],pairs=6,checksums={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}))
            marks[subject]=file_digest(folder/'prepared.json')
        code={'src/word_sentiment_eeg/paper_figure_stats.py':file_digest(ROOT/'src/word_sentiment_eeg/paper_figure_stats.py')}
        protocol=dict(code=code,subjects=readers,budget_bytes=100_000_000)
        if dataset=='zuco':protocol.update(tct_permutations=4,cluster_permutations=4)
        else:protocol.update(tct=4,cluster=4)
        write_json(root/'protocol.json',protocol)
        state=dict(status='completed',protocol_sha256=file_digest(root/'protocol.json'))
        if dataset=='zuco':state.update(subjects=readers,reader_checkpoint_sha256=marks)
        else:state.update(readers=marks)
        write_json(root/'cpu_preparation_manifest.json',state)
        # Need full reserve headroom for the production guard.
        protocol['budget_bytes']=500_000_000;write_json(root/'protocol.json',protocol);state['protocol_sha256']=file_digest(root/'protocol.json');write_json(root/'cpu_preparation_manifest.json',state)
        immutable={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
        command=[sys.executable,str(ROOT/'scripts/analyze_paper_small_storage.py'),'--thesis-root',str(Path(tmp)),'--run-dir',str(root),'--work-root',str(Path(tmp)/'local'),'--dataset',dataset,'--free-drive-bytes','300000000','--cpu','--skip-models']
        subprocess.run(command,check=True)
        for p,b in immutable.items():assert p.read_bytes()==b,'CPU archive/protocol was modified'
        snapshot={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        subprocess.run(command,check=True)
        assert snapshot=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()},'Resume rewrote committed Drive files'
        assert not (root/'null_work').exists()
        print('PASS saved CPU -> compact GPU handoff and resume:',dataset)
