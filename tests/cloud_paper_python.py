"""Validate clock rejection, FIR event expansion and continuous solver in cloud."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.paper_python import synchronize,eye_to_seconds,time_expanded_design,fit_nuisance


def main():
    codes=np.tile([10,11,12,13,15],6)
    eeg_samples=np.arange(1,len(codes)+1)*1000+1
    eye_clock=np.arange(1,len(codes)+1)*2000+900000
    sync=synchronize(codes,eeg_samples,np.column_stack([eye_clock,codes]),500)
    np.testing.assert_allclose(eye_to_seconds(eye_clock,sync),(eeg_samples-1)/500,atol=1e-10)
    bad=codes.copy();bad[3]=10
    try:synchronize(codes,eeg_samples,np.column_stack([eye_clock,bad]),500)
    except ValueError:pass
    else:raise AssertionError('Mismatched trigger sequence accepted')
    noisy=eeg_samples.astype(float);noisy[10]+=10
    try:synchronize(codes,noisy,np.column_stack([eye_clock,codes]),500)
    except ValueError:pass
    else:raise AssertionError('Large synchronization error accepted')
    diagnostic=synchronize(codes,eeg_samples+np.where(np.arange(len(codes))==10,2,0),
                           np.column_stack([eye_clock,codes]),500,max_error_ms=8)
    assert not diagnostic['native_one_sample_target_met']
    rng=np.random.default_rng(5);rows=[]
    for kind,n in [('fixation',60),('saccade',60),('blink',15),('keypress',15)]:
        for i,t in enumerate(np.sort(rng.uniform(1,49,n))):
            rows.append(dict(event_type=kind,onset_seconds=t,duration_ms=100+i,amplitude_deg=.5+i*.1))
    events=pd.DataFrame(rows)
    design,info=time_expanded_design(events,6250,limits=(-.04,.08))
    assert set(p.split('/')[0] for p in info['predictors'])=={'fixation','saccade','blink','keypress'}
    beta=rng.normal(size=(design.shape[1],2))
    signal=np.asarray(design@beta).T.astype(np.float32)
    residual,recovered,diagnostics=fit_nuisance(signal,design,atol=1e-9,maxiter=3000)
    assert np.linalg.norm(residual)/np.linalg.norm(signal)<1e-5
    assert all(r['stop'] in [0,1,2,4,5] for r in diagnostics)
    np.testing.assert_allclose(recovered@design.T,signal,atol=1e-4)
    print('Python pilot: exact-clock recovery, mismatch/noise rejection and overlapping synthetic-response recovery passed')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
