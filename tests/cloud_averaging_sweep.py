"""Small cloud test for exact-size averaging, fixed readers and cache label identity."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import zuco_averaging_sweep as s
from word_sentiment_eeg.common import write_json


def main():
    rng=np.random.default_rng(8);rows=[]
    for reader in range(6):
        for label in ['negative','positive']:
            for j in range(205):rows.append(dict(subject=f'R{reader}',sentiment=label,event_id=f'R{reader}/{label}/{j}'))
    meta=pd.DataFrame(rows);x=rng.normal(0,.3,(len(meta),105)).astype('float32');x[:,0]+=np.where(meta.sentiment.eq('positive'),2,-2)
    xx,mm,cohort=s.common_cohort(x,meta)
    assert len(xx)==len(x) and len(cohort['included_readers'])==6
    folds=s.grouped_folds(mm.sentiment.eq('positive').to_numpy(int),mm.subject.to_numpy())
    for size in s.SIZES:
        score,pred=s.evaluate(xx,mm,size,0,folds)
        assert score['test_readers']==6 and score['test_original_used']==len(pred)*size
        assert len(pred)==6*2*(205//size)
        assert score['balanced_accuracy']>.9
        import json
        members=[v for encoded in pred.member_event_ids for v in json.loads(encoded)]
        assert len(members)==len(set(members))
        assert all(len(json.loads(v))==size for v in pred.member_event_ids)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);work=tmp/'work';work.mkdir();parent=tmp/'parent';subset=tmp/'subset';(subset/'selection').mkdir(parents=True);folder=parent/'readers/R0';folder.mkdir(parents=True)
        metadata=pd.DataFrame(dict(event_id=['a','b'],sentiment=['negative','positive']))
        metadata.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False)
        # Agent labels deliberately reverse both parent labels: reading the audit
        # column sentiment instead of lexical_class would silently give wrong y.
        audit=pd.DataFrame(dict(event_id=['a','b'],subject=['R0','R0'],parent_row=[0,1],parent_sentiment=['negative','positive'],
            sentiment=['negative','positive'],lexical_class=['positive','negative'],contextual_class=['positive','negative'],subset_status=['retained','retained']))
        audit.to_csv(subset/'selection/trial_selection.csv.gz',index=False)
        signal=rng.normal(size=(2,2,105,201)).astype('float32');times=np.arange(-75,126)*8
        np.savez_compressed(folder/'prepared_trials.npz',signal=signal,times_ms=times,event_ids=['a','b'],labels=[0,1])
        checks={f:s.file_digest(folder/f) for f in ['accepted_trial_metadata.csv.gz','prepared_trials.npz']}
        write_json(folder/'prepared.json',dict(status='prepared',checksums=checks))
        write_json(parent/'protocol.json',{'test':1});write_json(parent/'cpu_preparation_manifest.json',dict(status='completed',protocol_sha256=s.file_digest(parent/'protocol.json'),reader_checkpoint_sha256={'R0':s.file_digest(folder/'prepared.json')}))
        write_json(subset/'complete.json',dict(status='completed',checksums={'selection/trial_selection.csv.gz':s.file_digest(subset/'selection/trial_selection.csv.gz')}))
        store=s.ImmutableStore(tmp/'out',20_000_000)
        features,labels,identity=s.load_features(parent,subset,store,work)
        np.testing.assert_array_equal(features,signal[1][:,:,(times>=224)&(times<=304)].mean(axis=2))
        assert labels.sentiment.tolist()==['positive','negative']
        with patch.object(s.shutil,'copyfile',side_effect=AssertionError('EEG reloaded after caching')):
            again,labels2,identity2=s.load_features(parent,subset,store,work)
            np.testing.assert_array_equal(again,features);assert identity2==identity and labels2.equals(labels)
        # Full orchestration, small repeated grouping setting, then unchanged resume.
        with patch.dict(s.SPEC,grouping_repeats=2):
            sweepstore=s.ImmutableStore(tmp/'sweep',20_000_000)
            summary=s.run_sweep(xx,mm,sweepstore,work)
            assert summary['size'].tolist()==s.SIZES and summary.test_readers.eq(6).all()
            assert (sweepstore.root/'averaging_curve.png').is_file()
            assert s.read(sweepstore.root/'complete.json')['status']=='completed'
            before={str(p):p.stat().st_mtime_ns for p in sweepstore.root.rglob('*') if p.is_file()}
            with patch.object(s,'evaluate',side_effect=AssertionError('Completed cases retrained')):s.run_sweep(xx,mm,sweepstore,work)
            assert before=={str(p):p.stat().st_mtime_ns for p in sweepstore.root.rglob('*') if p.is_file()}
    print('PASS exact sizes1/10..100, all folds/readers, no duplicate events, agent/parent-label distinction, cached-feature reuse, result publication and resume')

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
