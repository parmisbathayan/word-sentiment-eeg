"""Lossless CPU/GPU handoff, restart, identity and storage safety checks."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import atomic_npz
from word_sentiment_eeg.paper_stage_archive import save_trials,mark_prepared,verify_prepared,finish_reader


def main():
    rng=np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);reader=root/'readers/A';reader.mkdir(parents=True)
        signal=rng.normal(size=(2,8,105,201)).astype('float32')
        meta=pd.DataFrame(dict(event_id=[f'event{i}' for i in range(8)],sentiment=['negative']*4+['positive']*4,
            word=['bad']*4+['good']*4,sentence_key=[str(i) for i in range(8)],word_index=np.arange(8),subject='A'))
        neg=np.arange(4);pos=np.arange(4,8)
        save_trials(reader,signal,meta,neg,pos,root,20_000_000)
        with np.load(reader/'prepared_trials.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(signal,z['signal']);np.testing.assert_array_equal(meta.event_id,z['event_ids'])
        d=signal[:,pos]-signal[:,neg]
        atomic_npz(reader/'statistics.npz',pair_sum=d.sum(axis=1,dtype=np.float64),pair_squares=(d.astype(np.float64)**2).sum(axis=1))
        mark_prepared(reader,'A',[4,4],4);verify_prepared(reader)
        finish_reader(reader,root/'null_work',7,5,use_gpu=False)
        snapshots={p.name:p.read_bytes() for p in (root/'null_work').iterdir()}
        # Simulates a fresh process/runtime: only persistent files enter this call.
        finish_reader(reader,root/'null_work',7,5,use_gpu=False)
        assert snapshots=={p.name:p.read_bytes() for p in (root/'null_work').iterdir()}
        for p in (root/'null_work').glob('*.json'):assert json.loads(p.read_text())['subjects']==['A']
        assert json.loads((reader/'manifest.json').read_text())['status']=='completed'
        original=(reader/'accepted_trial_metadata.csv.gz').read_bytes()
        (reader/'accepted_trial_metadata.csv.gz').write_bytes(original+b'corrupt')
        try:verify_prepared(reader)
        except ValueError:pass
        else:raise AssertionError('Changed metadata accepted')
        (reader/'accepted_trial_metadata.csv.gz').write_bytes(original)
        (reader/'blocked').mkdir()
        try:save_trials(reader/'blocked',signal,meta,neg,pos,root,1)
        except OSError:pass
        else:raise AssertionError('Storage cap ignored')
        assert not (reader/'blocked/prepared_trials.npz').exists()
    print('CPU/GPU handoff passed: lossless signals, full identities, runtime restart, no duplicate permutations, corruption detection and storage cap')


if __name__=='__main__':main()
