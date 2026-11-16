from pathlib import Path
import sys,tempfile
from unittest.mock import patch
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import zuco_original_cohort_sweep as s
from zuco_recovered_followups import decode_case

def main():
    rng=np.random.default_rng(8);rows=[]
    for reader in range(6):
        for cls in ['negative','positive']:
            for j in range(40 if reader==0 else 125):rows.append(dict(subject=f'R{reader}',sentiment=cls,event_id=f'{reader}/{cls}/{j}',sentence_key=f's{j}',word_key=f'w{j}'))
    meta=pd.DataFrame(rows);y=meta.sentiment.eq('positive').to_numpy(int);x=rng.normal(0,.3,(len(meta),105)).astype('float32');x+=np.where(y[:,None],2,-2)*np.r_[np.ones(50),-np.ones(50),np.zeros(5)][None,:]
    folds=s.grouped_folds(y,meta.subject.to_numpy())
    old=decode_case(x,y,meta,'reader',40,'logistic','published_224_304','cleaned')
    actual,pred=s.evaluate(x,meta,40,0,folds);assert s.reference_check(actual,old)['status']=='verified'
    hundred,p=s.evaluate(x,meta,100,0,folds);assert hundred['test_readers']==5 and hundred['missing_readers']==['R0']
    bad=dict(old,pooled=dict(old['pooled'],balanced_accuracy=.1))
    try:s.reference_check(actual,bad)
    except ValueError:pass
    else:raise AssertionError('Reference mismatch silently accepted')
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)/'work';work.mkdir();store=s.ImmutableStore(Path(tmp)/'out',15_000_000)
        with patch.dict(s.SPEC,groupings=2):
            summary=s.run(x,meta,old,store,work)
            assert summary['size'].tolist()==s.b.SIZES
            assert s.b.read(store.root/'reference_verification.json')['status']=='verified'
            assert (store.root/'averaging_curve.png').exists()
            before={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
            # Only the deliberately repeated original-case audit may fit again.
            original_eval=s.evaluate
            def check(*args):
                assert args[2:4]==(40,0),'Completed sweep was retrained'
                return original_eval(*args)
            with patch.object(s,'evaluate',side_effect=check):s.run(x,meta,old,store,work)
            assert before=={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
    print('PASS original decode_case parity, exact membership digests, cohort retention, visible missing-reader coverage, curve publication and resume')
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
