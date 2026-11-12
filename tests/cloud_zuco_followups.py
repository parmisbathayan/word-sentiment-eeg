"""Cloud numerical/integrity tests; synthetic data only, no native EEG downloads."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import zuco_recovered_followups as f
from word_sentiment_eeg.common import atomic_npz,file_digest,write_json
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

if sys.platform=='darwin':raise SystemExit('Cloud only')
rng=np.random.default_rng(73)
# Exact reader-level test: all sign configurations, two-sided spatial magnitude.
pattern=np.arange(105,dtype=float);pattern-=pattern.mean()
r=f.scalp_test(np.tile(pattern,(12,1)))
assert r['assignments']==4096 and r['p_exact']==2/4096
assert np.allclose(f.holm([.01,.02,.5,.9]),[.04,.06,1,1])

n=360
meta=pd.DataFrame(dict(subject=np.repeat(['ZAB','ZDM','ZDN'],120),event_id=[f'e{i}' for i in range(n)],
    sentence_key=[f's{i//6}' for i in range(n)],word_key=[f'w{i%12}' for i in range(n)],
    sentiment=np.where(np.arange(n)%2,'positive','negative')))
y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy()
features=rng.normal(size=(n,10)).astype(np.float32);features[:,0]+=y*2
for mode,col in [('sentence','sentence_key'),('word','word_key'),('reader','subject')]:
    for train,test in f.grouped_folds(y,meta[col]):
        assert not set(meta.iloc[train][col])&set(meta.iloc[test][col])
        a,ya,_,ma=f.averaged(features,y,meta,train,20,1)
        b,yb,_,mb=f.averaged(features,y,meta,test,20,2)
        assert len(ma)==len(set(ma)) and len(mb)==len(set(mb)) and not set(ma)&set(mb)
    case=f.decode_case(features,y,meta,mode,1,'logistic','synthetic','raw')
    assert case['status']=='completed' and case['pooled']['n']==n
    assert case['pooled']['balanced_accuracy']>.6
print('PASS exact window test, Holm, group isolation, disjoint pseudotrials, real linear fits')

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);run=tmp/'input';work=tmp/'runtime';work.mkdir();out=tmp/'output';subjects=['ZAB','ZDM','ZDN'];marks={}
    for subject in subjects:
        folder=run/'readers'/subject;folder.mkdir(parents=True)
        n=24;y=np.arange(n)%2;ids=[subject+str(i) for i in range(n)]
        m=pd.DataFrame(dict(event_id=ids,subject=subject,sentence_key=[str(i//4) for i in range(n)],word_index=np.arange(n)%4,
            word=np.where(y,'good','bad'),label_5=np.where(y,1,-1),original_label_method='single lexical leaf (punctuation-only companions permitted)',
            block_id=1,word_length=4,word_frequency=10,duration_ms=150+np.arange(n),previous_gap_ms=200,next_gap_ms=200,
            sentiment=np.where(y,'positive','negative'),selection_status='accepted'))
        m.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False);m.to_csv(folder/'fixation_audit.csv.gz',index=False)
        signal=rng.normal(size=(2,n,105,201)).astype(np.float32)
        atomic_npz(folder/'prepared_trials.npz',signal=signal,labels=y,event_ids=np.array(ids),times_ms=np.arange(-75,126)*8,
            negative_pair_index=np.flatnonzero(y==0),positive_pair_index=np.flatnonzero(y==1))
        write_json(folder/'source_mapping.json',dict(blocks=[dict(block_id='1',status='included')]))
        write_json(folder/'prepared.json',dict(status='prepared',subject=subject,accepted_counts=[12,12],pairs=12,
            checksums={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}));marks[subject]=file_digest(folder/'prepared.json')
    write_json(run/'protocol.json',dict(synthetic=True));write_json(run/'cpu_preparation_manifest.json',dict(status='completed',subjects=subjects,
        protocol_sha256=file_digest(run/'protocol.json'),reader_checkpoint_sha256=marks));write_json(run/'manifest.json',dict(status='completed'))
    before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    store=ImmutableStore(out,50_000_000);x,m,p,provenance=f.stage(run,work,subjects,store,'test')
    assert x.shape==(2,72,105,201) and len(p)==3
    for window in f.SPEC['classification_windows']:assert np.isfinite(f.features(x[0],np.arange(-75,126)*8,window)).all()
    with patch.object(f,'label_audit',return_value={'synthetic':True}):f.diagnostics(x,m,p,provenance,store,work)
    assert len(json.loads((out/'diagnostics/window_tests.json').read_text())['tests'])==4
    assert (out/'figures/reader_map_correlations.png').is_file()
    assert len(json.loads((out/'diagnostics/repeated_words.json').read_text())['rows'])==6
    # Real classifier loop in reduced coverage. No fabricated final results.
    small=dict(f.SPEC,classification_windows=['published_224_304'],averaging=[1],splits=['sentence'])
    with patch.object(f,'SPEC',small):f.run_classifiers(x,m,store,work)
    saved={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    with patch.object(f,'SPEC',small):f.run_classifiers(x,m,store,work)
    assert saved=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    for path,data in before.items():assert path.read_bytes()==data
    assert not list(out.rglob('*.npz')) and store.used<50_000_000
    # Corruption is a hard failure; never silently skip a changed archive.
    target=run/'readers/ZAB/prepared_trials.npz';payload=target.read_bytes();target.write_bytes(payload[:-1]+bytes([payload[-1]^1]))
    try:f.stage(run,work,subjects,store,'test')
    except ValueError as error:assert 'checksum' in str(error)
    else:raise AssertionError('Corrupt trials accepted')
print('PASS saved-reader staging, all diagnostic outputs, immutable source/results, resume and corruption refusal')

# CNN actual CPU fit/resume parity, small synthetic input, no GPU claim.
import zuco_recovered_cnn as cnn
import torch
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);data=rng.normal(size=(360,8,100)).astype(np.float32)
    meta=pd.DataFrame(dict(subject=np.repeat(['A','B','C'],120),sentence_key=[f's{i//6}' for i in range(360)],
        word_key=[f'w{i%12}' for i in range(360)],event_id=[str(i) for i in range(360)]))
    y=np.arange(360)%2;train,test=f.grouped_folds(y,meta.sentence_key)[0]
    torch.set_num_threads(2)
    settings=dict(cnn.CNN_SPEC,epochs=2,patience=99)
    # Every epoch persists in this short regression only.
    settings['checkpoint_every']=1
    original=cnn.SmallNet
    with patch.object(cnn,'CNN_SPEC',settings):
        direct=cnn.fit_fold(data,y,meta,'sentence',train,test,ImmutableStore(tmp/'direct',50_000_000),Path('cnn/test'),0,device='cpu')
        first=dict(settings,epochs=1)
        # Restore identical identity settings after interrupting a committed frame.
        store=ImmutableStore(tmp/'resume',50_000_000);publish=store.file
        def interrupted(relative,source,*args,**kwargs):
            result=publish(relative,source,*args,**kwargs)
            if str(relative).endswith('epoch_001.pt'):raise KeyboardInterrupt()
            return result
        with patch.object(store,'file',interrupted):
            try:cnn.fit_fold(data,y,meta,'sentence',train,test,store,Path('cnn/test'),0,device='cpu')
            except KeyboardInterrupt:pass
            else:raise AssertionError('Interruption not reached')
        resumed=cnn.fit_fold(data,y,meta,'sentence',train,test,ImmutableStore(tmp/'resume',50_000_000),Path('cnn/test'),0,device='cpu')
    assert direct==resumed
print('PASS actual CNN training and interrupted frame resume parity on CPU')
