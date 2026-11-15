"""Cloud-only synthetic checks: pooled task identity, holdouts, averaging, CNN and resume."""
from pathlib import Path
import hashlib
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import classify_zuco_pooled as p


def main():
    torch.set_num_threads(1)
    rows=[]
    # The same sentences/words/readers deliberately repeat across all tasks.
    for task in p.TASKS:
        for reader in range(12):
            for sentence in range(18):
                for label in range(2):
                    rows.append(dict(task=task,subject=f'R{reader}',_recording=f'{task}/R{reader}',
                        truth=['negative','positive'][label],sentence_key=f's{sentence}',word_identity=f'w{sentence}_{label}',
                        sentence_position=sentence,onset_sample_sentence_0based=100+label*1500,dataset='zuco',
                        event_id=f'{task}/{reader}/{sentence}/{label}'))
    meta=pd.DataFrame(rows);y=meta.truth.eq('positive').to_numpy(int)
    for mode,column in [('sentence','sentence_key'),('word','word_identity'),('reader','subject')]:
        folds,spec=p.folds_for(meta,mode);seen=[]
        for train,fit,valid,test in folds:
            for left,right in [(fit,valid),(fit,test),(valid,test),(train,test)]:
                assert not set(meta.iloc[left][column])&set(meta.iloc[right][column])
                assert not set(left)&set(right)
            seen.extend(test)
            for part in [fit,valid,test]:
                groups,labels,table=p.memberships(meta,part,40,2026)
                assert sorted(np.concatenate(groups).tolist())==sorted(part.tolist())
                assert set(table.task)==set(p.TASKS)
                for g in groups:
                    assert len(g)<=40
                    assert len(meta.iloc[g][['subject','task','truth']].drop_duplicates())==1
        assert sorted(seen)==list(range(len(meta)))
    # Native sentence numbers collide across tasks, but unrelated tasks must not be purged.
    small=meta.iloc[:20].copy().reset_index(drop=True)
    small.loc[:9,'task']='NR';small.loc[:9,'_recording']='NR/R0'
    small.loc[10:,'task']='SR';small.loc[10:,'_recording']='SR/R0'
    small['subject']='R0';small['sentence_position']=0;small['onset_sample_sentence_0based']=100
    purged,_=p.purged(small,small.truth.eq('positive').to_numpy(int),[(np.arange(10),np.arange(10,20))])
    assert len(purged[0][0])==10
    groups,_,table=p.memberships(meta,np.arange(len(meta)),40,2026)
    assert table.is_remainder.all() and table.original_trials.eq(18).all()
    # End-to-end models and immutable checkpoint reuse; tiny CPU run in CI only.
    rng=np.random.default_rng(3);x=rng.normal(0,.1,(len(meta),4,100)).astype('float32')
    wave=np.sin(np.arange(100)*2*np.pi/25)
    x[:,0,:]+=np.where(y[:,None]==1,2.,-2.)*wave
    folds,_=p.folds_for(meta,'reader')
    with tempfile.TemporaryDirectory() as tmp:
        store=p.ImmutableStore(Path(tmp)/'results',100_000_000)
        for model in ['dummy','task_only','logistic','cnn']:
            result=p.run_case(x,meta,'reader',40,model,folds,store,device='cpu',epochs=2)
            assert result['pooled']['original_trials']==len(meta)
            assert set(result['per_task'])==set(p.TASKS)
            assert result['equal_task_balanced_accuracy'] is not None
            before={str(f):f.stat().st_mtime_ns for f in store.root.rglob('*') if f.is_file()}
            again=p.run_case(x,meta,'reader',40,model,folds,store,device='cpu',epochs=2)
            assert again==result
            assert before=={str(f):f.stat().st_mtime_ns for f in store.root.rglob('*') if f.is_file()}
        # A learned non-constant CNN, rather than only shape checks.
        xx=x[:200];yy=y[:200]
        prob=p.fit_cnn(xx,yy,np.arange(120),np.arange(120,160),np.arange(160,200),store,Path('learn'),3,'cpu',20,8)
        assert ((prob>=.5)==yy[160:]).mean()>.9
        with patch.object(p.c,'CompactEEGNet',side_effect=AssertionError('retrained completed fold')):
            np.testing.assert_array_equal(prob,p.fit_cnn(xx,yy,np.arange(120),np.arange(120,160),np.arange(160,200),store,Path('learn'),3,'cpu',20,8))
        # Saved waveform loading and row identity, followed by read-free stage reuse.
        archive=Path(tmp)/'archive';archive.mkdir();reader=archive/'reader';reader.mkdir()
        montage=[f'E{i}' for i in range(105)]
        pd.DataFrame({'channel':montage}).to_csv(archive/'zuco_channel_regions.csv',index=False)
        eeg=rng.normal(size=(3,105,126)).astype('float32');ids=np.array(['a','b','c'])
        shard=reader/'part.npz';np.savez_compressed(shard,eeg_uV=eeg,event_id=ids,channel_names=montage,times_ms=np.arange(-25,101)*8)
        p.write_json(reader/'checkpoint.json',dict(blocks=[dict(checksums={'part.npz':p.file_digest(shard)})]))
        selection=pd.DataFrame(dict(task=['SR','SR'],subject=['R0','R0'],_reader_dir=[str(reader)]*2,
            epoch_file=['part.npz']*2,epoch_row=[2,0],event_id=['c','a'],_stage_row=[0,1],
            signal_sha256=[hashlib.sha256(eeg[i].tobytes()).hexdigest() for i in [2,0]]))
        staged=p.stage(selection,archive,Path(tmp)/'temp','test')
        np.testing.assert_array_equal(staged,eeg[[2,0],:,25:125])
        with patch.object(p.a,'strict_path',side_effect=AssertionError('reloaded saved shards')):
            reused=p.stage(selection,archive,Path(tmp)/'temp','test')
            np.testing.assert_array_equal(staged,reused)
    print('PASS pooled holdouts, task-aware purge, all-event averaging, OOF models, CNN learning, checkpoint reuse and saved EEG staging')

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
