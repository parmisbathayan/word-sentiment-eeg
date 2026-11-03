"""Small synthetic validation of train-only NCA, split feasibility and resume."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg import paper_decoding as d


def main():
    assert d.decoding_plan('published_grid')['classifier_fits']==880200
    rng=np.random.default_rng(8);rows=[];members=[];X=[]
    for reader in ['A','B','C','D']:
        for label,value in [('negative',-5),('positive',5)]:
            for i in range(25):
                identifier=f'{reader}/{label}/{i}'
                rows.append(dict(average_id=identifier,subject=reader,sentiment=label))
                members.append(dict(average_id=identifier,event_id='event/'+identifier,subject=reader,sentiment=label))
                X.append(rng.normal(value,.1,6))
    meta=pd.DataFrame(rows);membership=pd.DataFrame(members);X=np.asarray(X,np.float32)
    pipe=d.estimator('logistic','reported_fixed',[1],nca_max_iter=100)
    pipe.fit(X[:100],meta.sentiment.to_numpy()[:100])
    np.testing.assert_allclose(pipe.named_steps['scaler'].mean_,X[:100].mean(axis=0,dtype=float))
    before=pipe.named_steps['scaler'].mean_.copy();pipe.predict(np.ones((2,6))*9999)
    np.testing.assert_array_equal(before,pipe.named_steps['scaler'].mean_)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        frame=d.run_decoding(X,meta,membership,out,stratification='joint_reader_sentiment',repeats=2,components=[1],nca_max_iter=100)
        assert len(frame)==4 and frame.accuracy.min()>.95
        original=d.estimator
        def no_refit(*args,**kwargs):raise AssertionError('Completed decoder refit')
        d.estimator=no_refit
        try:d.run_decoding(X,meta,membership,out,stratification='joint_reader_sentiment',repeats=2,components=[1],nca_max_iter=100)
        finally:d.estimator=original
        try:d.run_decoding(X,meta,membership,out/'large',stratification='sentiment',mode='published_grid',max_classifier_fits=100)
        except ValueError as error:assert 'budget' in str(error)
        else:raise AssertionError('Unbudgeted full grid executed')
    # Paper's reported 328 forty-trial averages cannot jointly cover 24 strata
    # in a 5% test set. This must not silently fall back to another split.
    small=pd.DataFrame([dict(subject=f'R{i%12}',sentiment='positive' if i%2 else 'negative') for i in range(328)])
    # Ensure all 24 joint strata are actually represented.
    small=pd.DataFrame([dict(subject=f'R{i%12}',sentiment='positive' if (i//12)%2 else 'negative') for i in range(328)])
    try:d.split_averages(small,'joint_reader_sentiment')
    except ValueError as error:assert '24' in str(error)
    else:raise AssertionError('Infeasible joint stratification accepted')
    print('Paper decoder synthetic checks passed: normalization, NCA, membership, checkpoint reuse, grid budget and stratification')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
