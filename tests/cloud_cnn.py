"""CPU synthetic CNN learning, group leakage and checkpoint reuse checks."""
from pathlib import Path
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import classify_cnn as c


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(3);y=np.tile([0,1],100)
    X=rng.normal(0,.15,(200,4,100)).astype('float32')
    wave=np.sin(np.arange(100)*2*np.pi/25).astype('float32')
    X[:,0,:]+=np.where(y[:,None]==1,2.,-2.)*wave
    fit=np.arange(120);valid=np.arange(120,160);test=np.arange(160,200)
    mean,sd=c.train_stats(X,fit)
    np.testing.assert_allclose(mean,X[fit].mean(axis=(0,2),dtype=float),atol=1e-6)
    np.testing.assert_allclose(sd,X[fit].std(axis=(0,2),dtype=float),atol=1e-6)
    with tempfile.TemporaryDirectory() as directory:
        p=c.train_fold(X,y,fit,valid,test,Path(directory),max_epochs=20,patience=8,batch_size=32)
        assert ((p>=.5)==y[test]).mean()>.9
        before=(Path(directory)/'best.pt').stat().st_mtime_ns
        p2=c.train_fold(X,y,fit,valid,test,Path(directory),max_epochs=20,patience=8,batch_size=32)
        np.testing.assert_array_equal(p,p2)
        assert (Path(directory)/'best.pt').stat().st_mtime_ns==before
        try:c.train_fold(X,y,fit,valid,test,Path(directory),max_epochs=19,patience=8,batch_size=32)
        except ValueError:pass
        else:raise AssertionError('Changed model configuration silently reused')
    meta=pd.DataFrame(dict(sentence_key=[f's{i//4}' for i in range(200)],word_identity=[f'w{i//4}' for i in range(200)],
                           subject=[f'R{i//20}' for i in range(200)],dataset='zuco',sentence_position=np.arange(200),
                           onset_sample_sentence_0based=200,event_id=[str(i) for i in range(200)]))
    for mode in ['sentence','word','reader']:
        folds,_=c.nested_folds(meta,y,mode);seen=[]
        for tr,va,te in folds:
            assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
            seen.extend(te)
        assert sorted(seen)==list(range(200))
    assert sum(p.numel() for p in c.CompactEEGNet(105).parameters())<10000
    # Exercise the actual archive-to-temporary-stage adapter, not just training.
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);archive=root/'archive';archive.mkdir();temporary=root/'temporary';temporary.mkdir()
        protocol={'test':True};(archive/'protocol.json').write_text(json.dumps(protocol))
        pd.DataFrame(dict(channel=['C1','C2','C3','C4'])).to_csv(archive/'zuco_channel_regions.csv',index=False)
        reader=archive/'zuco/readers/SR/A';reader.mkdir(parents=True)
        events=pd.DataFrame([dict(event_id=f'event{i}',status='archived',sentence_key=f'fake{i}',word_index=0,word='example',
                                  subject='A',epoch_file='epochs.npz',epoch_row=i) for i in range(6)])
        events.to_csv(reader/'events.csv.gz',index=False,compression='gzip')
        data=rng.normal(size=(6,4,126)).astype('float32')
        np.savez_compressed(reader/'epochs.npz',eeg_uV=data,event_id=events.event_id.to_numpy(dtype=str),
                            channel_names=np.array(['C1','C2','C3','C4']),times_ms=np.arange(-25,101)*8)
        checkpoint=dict(status='completed',epochs=6,identity={'protocol_sha256':c.digest(protocol)},
                        events_sha256=c.file_digest(reader/'events.csv.gz'),
                        blocks=[dict(epoch_files=['epochs.npz'],checksums={'epochs.npz':c.file_digest(reader/'epochs.npz')})])
        (reader/'checkpoint.json').write_text(json.dumps(checkpoint))
        roster=[dict(dataset='zuco',task='SR',subject='A',epochs=6)]
        staged,meta,identity,_=c.stage_task(archive,roster,root,temporary)
        np.testing.assert_array_equal(staged,data[:,:,25:125]);assert meta.event_id.tolist()==events.event_id.tolist()
        del staged
        (reader/'epochs.npz').rename(reader/'source_hidden.npz')
        staged,meta,identity2,_=c.stage_task(archive,roster,root,temporary)
        assert identity==identity2 # a validated stage never needs to reopen EEG
        np.testing.assert_array_equal(staged,data[:,:,25:125]);del staged
    print('CNN checks passed: learns signal, train-only scaling, grouped validation, OOF coverage, resume and parameter budget')


if __name__=='__main__':main()
