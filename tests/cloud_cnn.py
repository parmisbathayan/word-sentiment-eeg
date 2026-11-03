"""CPU synthetic CNN learning, group leakage and checkpoint reuse checks."""
from pathlib import Path
import sys
import tempfile
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
    print('CNN checks passed: learns signal, train-only scaling, grouped validation, OOF coverage, resume and parameter budget')


if __name__=='__main__':main()
