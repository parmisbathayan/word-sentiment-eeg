"""Cloud-only multi-target integration: multiclass, familiarity, purging, no target leakage."""
from pathlib import Path
import sys,tempfile,json
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_other_targets as t
rng=np.random.default_rng(13)
# Three repeated source occurrences per word identity, with well separated onsets.
meta=pd.DataFrame([dict(event_id=f'{r}-{j}',subject=f'R{r}',sentence_key=f'S{j%8}',word_identity=f'W{j%16}',
    sentence_position=j%8,onset_sample_sentence_0based=(j//8)*2400,dataset='teco',
    first_fixation_duration_ms=120+(j%3)*100,signal_sha256=f'h{r}-{j}') for r in range(12) for j in range(96)])
post=rng.normal(size=(len(meta),48)).astype(np.float32);pre=rng.normal(size=(len(meta),12)).astype(np.float32)
features=t.regional_inputs(post,pre);features['metadata_control']=rng.normal(size=(len(meta),4)).astype(np.float32)
# Validate region-major original cache geometry.
np.testing.assert_array_equal(features['region_anterior'],post[:,:8]);np.testing.assert_array_equal(features['regional_200_400'],post.reshape(-1,6,8)[:,:,2:4].reshape(len(meta),-1))
info=[]
for target in ['reader_identity','fixation_150','fixation_200','fixation_250']:
    rows,y,X,classes,definition=t.prepare_target(Path('/nonexistent'),meta,features,target)
    assert len(classes)==(12 if target=='reader_identity' else 2)
    assert 'metadata_control' in X and X['metadata_control'].shape[1]==4
    assert not any('duration' in key or key.startswith('residual_') for key in X)
    info.append((rows,y,X,classes,definition))
reader,y,_,classes,_=info[0]
for mode in ['familiar','sentence']:
    pairs=t.split(reader,y,np.arange(len(reader)),mode,3,t.SEED)
    for tr,te,audit in pairs:
        assert not set(tr)&set(te)
        if mode=='familiar':assert all(v==1 for v in audit['test_seen_fraction'].values())
        if mode=='sentence':assert not set(meta.sentence_key.iloc[tr])&set(meta.sentence_key.iloc[te])
        for i in tr:
            g=meta.iloc[te];g=g[(g.subject==meta.subject.iloc[i])&(g.sentence_position==meta.sentence_position.iloc[i])]
            assert ((g.onset_sample_sentence_0based-meta.onset_sample_sentence_0based.iloc[i]).abs()>1200).all()
tr,te,_=t.split(reader,y,np.arange(len(reader)),'sentence',3,t.SEED)[0]
gt=t.membership(meta,y,tr,10,7);ge=t.membership(meta,y,te,10,8)
assert all(len(g)==10 and len(set(y[g]))==1 and meta.subject.iloc[g].nunique()==1 for g in gt+ge)
cap=t.membership(meta,y,te,'max40',8)
assert sum(map(len,cap))==len(te) and all(len(g)<=40 for g in cap)
# Every estimator supports twelve-way reader identity (not accidentally binary LDA).
for name in t.MODELS:
    result=t.score(meta,y,features['regional_bins'],(gt,ge),name,12)
    assert np.array(result['confusion']).shape==(12,12) and 0<=result['balanced_accuracy']<=1
# Binary duration models use the original binary logistic solver; all model families fit.
fix,fy,fx,fc,_=info[2]
trf,tef,_=t.split(fix,fy,np.arange(len(fix)),'sentence',3,t.SEED)[0]
fg=(t.membership(fix,fy,trf,1,7),t.membership(fix,fy,tef,1,8))
for name in t.MODELS:
    result=t.score(fix,fy,fx['regional_bins'],fg,name,2)
    assert np.array(result['confusion']).shape==(2,2)
assert t.estimator('lr_1',2).steps[-1][1].solver=='liblinear'
# Fixed duration thresholds, exact boundary and invalid duration accounting.

boundary=meta.iloc[:4].copy();boundary['first_fixation_duration_ms']=[199,200,np.nan,-1]
_,dy,_,dc,definition=t.prepare_target(Path('/nonexistent'),boundary,{k:v[:4] for k,v in features.items()},'fixation_200')
assert definition['excluded_invalid_duration']==2 and [dc[i] for i in dy]==['short','long']
with tempfile.TemporaryDirectory() as tmp:
    store=t.ImmutableStore(Path(tmp),50_000_000)
    subset={k:features[k] for k in ['regional_bins','region_anterior','pre_control','metadata_control']}
    rows,dev,missing=t.evaluate(meta,y,subset,classes,dict(modes=['sentence','familiar'],sizes=[1,10,'max40']),store,'reader_identity',models=['lr_1','lda'])
    assert rows and all(r['status']=='completed' for r in rows)
    summary=t.summarize(rows);assert summary.completed_folds.eq(3).all();t.figures(store,summary,rows)
    assert (store.root/'reader_confusion.png').stat().st_size>1000
    before={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
    again,dev2,missing2=t.evaluate(meta,y,subset,classes,dict(modes=['sentence','familiar'],sizes=[1,10,'max40']),store,'reader_identity',models=['lr_1','lda'])
    assert rows==again and dev==dev2 and missing==missing2
    assert before=={str(p):p.stat().st_mtime_ns for p in store.root.rglob('*') if p.is_file()}
    # No outer test influence on choice when test memberships are fixed.
    original=t.split
    def fixed(m,yy,ix,mode,n,seed):return original(meta,y,ix,mode,n,seed)
    altered={k:v.copy() for k,v in subset.items()}
    for v in altered.values():v[te]=999
    with tempfile.TemporaryDirectory() as tmp2,patch.object(t,'split',side_effect=fixed):
        other=t.ImmutableStore(Path(tmp2),50_000_000)
        _,d3,_=t.evaluate(meta,y,altered,classes,dict(modes=['sentence'],sizes=[1]),other,'reader_identity',models=['lr_1','lda'])
        assert [r for r in dev if r['mode']=='sentence' and r['fold']==0 and r['size']=='1']==[r for r in d3 if r['fold']==0]
    assert store.used<50_000_000
# Broader-cache loader and frozen sentiment rosters: no live annotation lookup or raw EEG fallback.
with tempfile.TemporaryDirectory() as tmp:
    thesis=Path(tmp);archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz';source=thesis/'Results/word_sentiment_eeg/classification_similarity'
    source.mkdir(parents=True);(source/'manifest.json').write_text(json.dumps(dict(status='completed')));(source/'protocol.json').write_text('{}')
    records=[dict(dataset='teco',task='task1',subject=f'R{i}',epochs=96) for i in range(12)]
    def cached(ar,src,record):
        ix=np.flatnonzero(meta.subject.to_numpy()==record['subject']);return meta.iloc[ix].copy(),post[ix],pre[ix],np.zeros((len(ix),6)),dict(reader=record['subject'],cache='fixture')
    with patch.object(t.a,'load_completed',return_value=(records,{})),patch.object(t.old,'cached_features_only',side_effect=cached) as loader:
        broad,vectors,provenance=t.load_all(thesis)
    assert loader.call_count==12 and len(broad)==len(meta) and vectors['metadata_control'].shape[1]==4
    # Actual saved-roster reading restores per-event labels and validates immutable checksums.
    for target in ['binary','three_class']:
        labels=np.where(np.arange(len(broad))%2,'positive','negative') if target=='binary' else np.array(['negative','neutral','positive'])[np.arange(len(broad))%3]
        roster=broad[['event_id','subject']].copy();roster['truth']=labels
        folder=source/'teco/task1/cases'/f'word_agreement__{target}__sentence__post__logistic';folder.mkdir(parents=True)
        path=folder/'predictions.csv.gz';roster.to_csv(path,index=False)
        counts={c:int((labels==c).sum()) for c in set(labels)}
        (folder/'result.json').write_text(json.dumps(dict(checksums={'predictions.csv.gz':t.a.file_digest(path)},score=dict(trials=len(roster),class_counts=counts))))
    for target in ['sentiment_binary','sentiment_three_class','neutral_nonneutral']:
        rows,yy,xx,cc,definition=t.prepare_target(thesis,broad,vectors,target)
        assert len(rows)==len(broad) and len(cc)==(3 if target=='sentiment_three_class' else 2)
        assert len(xx['metadata_control'])==len(rows)
    # Corruption is fatal, not a fallback to live labels.
    path=source/'teco/task1/cases/word_agreement__binary__sentence__post__logistic/predictions.csv.gz';path.write_bytes(b'broken')
    try:t.frozen_sentiment(thesis,broad,'sentiment_binary')
    except ValueError as error:assert 'checksum' in str(error)
    else:raise AssertionError('Corrupt frozen label source was accepted')

# Familiarity can exclude a unique word, not relabel it or pretend it was seen.
unique=meta.copy();unique.loc[0,'word_identity']='singleton'
for tr,te,audit in t.split(unique,y,np.arange(len(unique)),'familiar',3,t.SEED):assert 0 not in te
print('Other-target sweep: multiclass models, fixed duration cutoffs, target-safe inputs, strict familiarity, overlap purge, nested isolation, resume and plots passed.')
