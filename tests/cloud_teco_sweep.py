"""Synthetic cloud integration and split/selection integrity checks. No private data."""
from pathlib import Path
import sys,tempfile,json
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_classification_sweep as t
rng=np.random.default_rng(12)
meta=pd.DataFrame([dict(event_id=f'r{r}_{j}',subject=f'r{r}',truth='positive' if j%2 else 'negative',
    sentence_key=f's{j//4}',sentence_position=j//4,word_identity=f'w{j%16}',word_index=j%4,dataset='teco',
    onset_sample_sentence_0based=(j%4)*2400) for r in range(12) for j in range(80)])
y=meta.truth.eq('positive').to_numpy(dtype=int)
montage=pd.DataFrame(dict(channel=[f'C{i}' for i in range(12)],region=np.repeat(t.a.REGIONS,2)))
eeg=rng.normal(size=(len(meta),12,126)).astype(np.float32);times=np.arange(-25,101)*8
features=t.feature_set(eeg,times,montage)
assert len(features)==21 # includes pre-control; twenty EEG inputs
np.testing.assert_allclose(features['channels_224_304'],eeg[:,:,(times>=224)&(times<=304)].mean(2))
assert features['regional_bins'].shape==(len(meta),48)
for mode in t.MODES:
    pairs=t.folds(meta,y,np.arange(len(meta)),mode,3,t.SEED)
    for tr,te,purged in pairs:
        assert not set(tr)&set(te)
        if mode!='mixed':
            c={'reader':'subject','word':'word_identity','sentence':'sentence_key'}[mode]
            assert not set(meta[c].iloc[tr])&set(meta[c].iloc[te])
        # Every kept same-sentence training event is >1000ms from test.
        for i in tr:
            same=meta.iloc[te];same=same[(same.subject==meta.subject.iloc[i])&(same.sentence_position==meta.sentence_position.iloc[i])]
            assert ((same.onset_sample_sentence_0based-meta.onset_sample_sentence_0based.iloc[i]).abs()>1200).all()
tr,te,_=t.folds(meta,y,np.arange(len(meta)),'reader',3,t.SEED)[0]
gt=t.memberships(meta,y,tr,5,2);ge=t.memberships(meta,y,te,5,3)
assert gt.shape[1]==5 and ge.shape[1]==5 and not set(gt.ravel())&set(ge.ravel())
assert all(len(set(y[g]))==1 and meta.subject.iloc[g].nunique()==1 for g in ge)
# Every offered estimator fits and produces valid binary predictions.
X=features['regional_bins']
for model in t.MODELS:
    score=t.score_one(meta,y,X,(gt,ge),model,detailed=False)
    assert 0<=score['balanced_accuracy']<=1 and 'truth' not in score
residual=np.concatenate([X,rng.normal(size=(len(meta),6))],axis=1)
assert 0<=t.score_one(meta,y,residual,(gt,ge),'lr_1',True)['balanced_accuracy']<=1
# Correlated events must be removed in inner as well as outer mixed splits.
near=meta.copy();near['onset_sample_sentence_0based']=(near.word_index*120).to_numpy()
pairs=t.folds(near,y,np.arange(len(near)),'word',3,t.SEED)
assert any(pr>0 for _,_,pr in pairs)
with tempfile.TemporaryDirectory() as tmp:
    store=t.ImmutableStore(Path(tmp),50_000_000)
    subset={k:features[k] for k in ['channels_224_304','regional_bins','pre_control']};subset['nuisance']=rng.normal(size=(len(meta),6))
    rows,dev=t.evaluate(meta,subset,store,modes=['reader'],sizes=[1,5],models=['lr_1','lda'])
    assert len(rows)==3*2*5 and all(r['status']=='completed' for r in rows)
    assert all(r['stage'] in ['input_screen','model_search'] for r in dev)
    t.report(rows,dev,store)
    report=pd.read_csv(store.root/'summary.csv');assert report.completed_folds.eq(3).all()
    assert (store.root/'sweep.png').stat().st_size>1000
    before={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
    rows2,dev2=t.evaluate(meta,subset,store,modes=['reader'],sizes=[1,5],models=['lr_1','lda'])
    assert rows==rows2 and dev==dev2
    assert before=={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
    # Outer test signal and truth cannot affect input/model selection for fold zero.
    changed={k:v.copy() for k,v in subset.items()}
    for v in changed.values():v[te]=999
    changed_meta=meta.copy();changed_meta.loc[te,'truth']=np.where(y[te]==1,'negative','positive')
    # Reader folds are fixed by reader identifiers and must be independent of test labels.
    # Test selection directly through fixed outer/inner memberships rather than rebuilding stratification.
    inner=t.folds(meta,y,tr,'reader',2,t.SEED+100)
    for a,b,_ in inner:assert not set(np.r_[a,b])&set(te)
    original_folds=t.folds
    def frozen_folds(m,yy,ix,mode,n,seed):return original_folds(meta,y,ix,mode,n,seed)
    with tempfile.TemporaryDirectory() as altered_tmp,patch.object(t,'folds',side_effect=frozen_folds):
        altered_store=t.ImmutableStore(Path(altered_tmp),50_000_000)
        _,changed_dev=t.evaluate(changed_meta,changed,altered_store,modes=['reader'],sizes=[1,5],models=['lr_1','lda'])
        assert [r for r in dev if r['fold']==0]==[r for r in changed_dev if r['fold']==0], 'Held-out data affected development selection'
    # Prefix corruption is refused, never silently counted as completed.
    marker=next(store.root.glob('fits/*/fold0/size1/dev/*/*/inner0/result.json'))
    result=json.loads(marker.read_text());result['balanced_accuracy']=.999;marker.write_text(json.dumps(result))
    try:t.save_case(store,str(marker.parent.relative_to(store.root)),lambda:{})
    except ValueError:pass
    else:raise AssertionError('Corrupt checkpoint accepted')
print('TeCo sweep: features, all models, purging, grouped splits, exact averaging, checkpoint resume and report passed.')
