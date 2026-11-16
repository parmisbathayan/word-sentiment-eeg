"""Small full-path protocol test; no private EEG or heavyweight published grid."""
from pathlib import Path
import sys,tempfile
from unittest.mock import patch
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg import paper_protocol_decoding as d
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore


def main():
    rng=np.random.default_rng(8);rows=[]
    #328examples/24jointstrata: reproduce why requiring all24cells is impossible,
    # while bounded marginal proportions and reader overlap remain feasible.
    for i in range(328):rows.append(dict(subject=f'R{i%12}',sentiment=['negative','positive'][(i//12)%2],average_id=f'a{i}'))
    meta=pd.DataFrame(rows)
    for seed in range(100):
        train,test,a=d.random_split(meta,seed)
        assert len(test)==17 and not set(train)&set(test)
        assert set(meta.iloc[train].subject)==set(meta.iloc[test].subject)==set(meta.subject)
        assert abs(meta.iloc[test].sentiment.eq('positive').sum()-meta.sentiment.eq('positive').mean()*17)<=.5+1e-9
    original=pd.DataFrame([dict(subject=f'R{r}',sentiment=c,event_id=f'{r}/{c}/{i}',sentence_key=f's{i%7}',word=f'w{i%11}',word_length=1+i%9) for r in range(3) for c in ['negative','positive'] for i in range(43)])
    x=rng.normal(size=(len(original),6)).astype('float32')
    for size in [1,10,20,40]:
        xx,mm,members=d.averages(x,original,size)
        assert len(members)==len(original) and not members.event_id.duplicated().any()
        assert mm.actual_trials.between(1,size).all()
        for _,g in members.groupby('average_id'):
            assert len(g.subject.unique())==len(g.sentiment.unique())==1 and not g.sentence_key.duplicated().any()
    # A tiny nested search still exercises both model grids and train-only NCA.
    rows=[]
    for reader in range(4):
        for label in ['negative','positive']:
            for i in range(15):rows.append(dict(average_id=f'{reader}/{label}/{i}',subject=f'R{reader}',sentiment=label,actual_trials=1))
    meta=pd.DataFrame(rows);y=meta.sentiment.eq('positive').to_numpy();x=rng.normal(0,.2,(len(meta),6)).astype('float32')+np.where(y[:,None],3,-3)
    membership=meta[['average_id','subject','sentiment']].copy();membership['event_id']=['event'+str(i) for i in range(len(meta))];membership['sentence_key']=[f's{i%8}' for i in range(len(meta))];membership['word']=[f'w{i%5}' for i in range(len(meta))]
    with tempfile.TemporaryDirectory() as tmp:
        store=ImmutableStore(Path(tmp),10_000_000);relative=Path('decoding/size40/repeat000')
        real=d.nca_transform;calls=[]
        def monitored(a,y,z,k,seed,max_iter=100):
            calls.append((a.shape,z.shape,k));np.testing.assert_allclose(a.mean(axis=0),0,atol=1e-5)
            return real(a,y,z,k,seed,max_iter)
        options=dict(dims=(2,3),cs=(.1,1),gammas=(.1,1),inner_folds=2,nca_max_iter=20)
        with patch.object(d,'nca_transform',side_effect=monitored):result=d.run_repeat(x,meta,membership,store,relative,0,**options)
        assert len(calls)<=6 #4innerfit+up to2outer, never perC/gamma
        assert min(v['accuracy'] for v in result['scores'] if v['model']!='dummy')>.8
        split=d.read(store.root/relative/'split.json');assert split['original_event_overlap']==0 and len(split['readers_shared'])==4
        before={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
        with patch.object(d,'nca_transform',side_effect=AssertionError('Completed repeat refit')):again=d.run_repeat(x,meta,membership,store,relative,0,**options)
        assert again==result and before=={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
        #The actual algorithm must reject a changed feature matrix on resume.
        try:d.run_repeat(x+1,meta,membership,store,relative,0,**options)
        except ValueError:pass
        else:raise AssertionError('Changed features reused')
        summary=d.summary([dict(size=40,**s) for s in result['scores']]);assert set(summary.model)=={'logistic','svm_rbf','dummy'}
    print('PASS100proportional95/5splits, sharedreaders, smalljointstrata, exactmembership, retainedremainders, bothnestedgrids, train-onlyNCA/scaling and resume')

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
