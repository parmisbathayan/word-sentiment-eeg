"""Cloud integration: same rosters/features, nested reader holdout, averaging and resume."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import teco_sentiment_unseen_readers as s
t = s.t
rng = np.random.default_rng(27)
meta = pd.DataFrame([dict(event_id=f'R{r}-E{j}', subject=f'R{r:02}', sentence_key=f'S{j//4}',
    word_identity=f'W{j%12}', sentence_position=j//4, onset_sample_sentence_0based=(j%4)*200,
    signal_sha256=f'h{r}-{j}', dataset='teco') for r in range(12) for j in range(126)])
post = rng.normal(size=(len(meta),48)).astype(np.float32)
features = t.regional_inputs(post, rng.normal(size=(len(meta),12)).astype(np.float32))
features['metadata_control'] = rng.normal(size=(len(meta),4)).astype(np.float32)
source = dict(readers=['synthetic fixture only'], source_protocol='test')
prepared = {}
for name in s.PLAN:
    classes = ['negative','positive'] if name=='sentiment_binary' else (['negative','neutral','positive'] if name=='sentiment_three_class' else ['neutral','nonneutral'])
    y = np.tile(np.arange(126)%len(classes),12)
    x = {k:features[k] for k in ['regional_bins','pre_control','metadata_control']}
    if name=='sentiment_binary':x['advanced_channels_224_304']=post[:,:5]
    info=dict(classes=classes, class_counts={c:int((y==i).sum()) for i,c in enumerate(classes)}, trials=len(meta), readers=12)
    prepared[name]=(meta,y,x,classes,info)
    # Reader identity never appears across either outer or inner partition.
    outer=t.split(meta,y,np.arange(len(meta)),'reader',3,t.SEED)
    for train,test,_ in outer:
        assert not set(meta.subject.iloc[train])&set(meta.subject.iloc[test])
        for tr,te,_ in t.split(meta,y,train,'reader',2,t.SEED+100):
            assert not set(meta.subject.iloc[tr])&set(meta.subject.iloc[te])
        exact=t.membership(meta,y,test,40,0);maximum=t.membership(meta,y,test,'max40',0)
        assert all(len(g)==40 for g in exact)
        assert sum(map(len,maximum))==len(test)
        for groups in [exact,maximum]:
            assert all(meta.subject.iloc[g].nunique()==1 and len(set(y[g]))==1 for g in groups)
evaluate=t.evaluate
def small_evaluate(*args,**kwargs):return evaluate(*args,**kwargs,models=['lr_1','lda'])
with tempfile.TemporaryDirectory() as tmp:
    thesis=Path(tmp);ref=thesis/'Results/word_sentiment_eeg/teco_other_targets'
    store=t.ImmutableStore(ref,50_000_000)
    store.json('protocol.json',dict(source=source))
    paths=['protocol.json']
    for name,(m,y,x,c,info) in prepared.items():
        store.json(f'targets/{name}/definition.json',info)
        export=m[['event_id','subject','sentence_key','word_identity','sentence_position','onset_sample_sentence_0based']].copy();export['target_code']=y
        t.old.csv(store,f'targets/{name}/events.csv.gz',export)
        paths += [f'targets/{name}/definition.json',f'targets/{name}/events.csv.gz']
    store.json('complete.json',dict(status='completed',checksums={f:t.a.file_digest(ref/f) for f in paths}))
    before={p.relative_to(ref):p.read_bytes() for p in ref.rglob('*') if p.is_file()}
    # Scientific driver runs all12 requested conditions. No archive or waveform exists here.
    with patch.object(t,'load_all',return_value=(meta,features,source)), patch.object(t,'prepare_target',side_effect=lambda _,m,x,n:prepared[n]), patch.object(t,'evaluate',side_effect=small_evaluate):
        s.run(thesis,50_000_000)
    out=thesis/'Results/word_sentiment_eeg/teco_sentiment_unseen_readers'
    summary=pd.read_csv(out/'summary.csv',dtype={'size':str})
    assert len(summary)==60 and summary.completed_folds.eq(3).all()
    assert set(summary['size'])=={'1','10','40','max40'} and set(summary['mode'])=={'reader'}
    assert summary[['balanced_accuracy','macro_f1']].notna().all().all()
    assert t.read(out/'complete.json')['complete_conditions']==12
    choices=pd.read_csv(out/'selected_inputs_models.csv');assert len(choices)==36
    import gzip
    results=json.loads(gzip.decompress((out/'outer_results.json.gz').read_bytes()))
    assert all(r['input']=='advanced_channels_224_304' for r in results if r['target']=='sentiment_binary' and r['variant']=='fixed_eeg')
    assert all(r['input']=='regional_bins' for r in results if r['target']!='sentiment_binary' and r['variant']=='fixed_eeg')
    assert before=={p.relative_to(ref):p.read_bytes() for p in ref.rglob('*') if p.is_file()}
    # Completed rerun may load vectors for identity verification, but never fits again.
    with patch.object(t,'load_all',return_value=(meta,features,source)), patch.object(t,'evaluate',side_effect=AssertionError('unexpected refit')):
        s.run(thesis,50_000_000)
    m,y,x,c,info=prepared['sentiment_binary']
    for bad_y,bad_x in [(1-y,x),(y,{k:v for k,v in x.items() if k!='advanced_channels_224_304'})]:
        try:s.verify_target(ref,'sentiment_binary',m,bad_y,bad_x,info)
        except ValueError:pass
        else:raise AssertionError('Changed labels or missing fixed input accepted')
    # Low quota fails before any feature read.
    with patch.object(t,'load_all',side_effect=AssertionError('loaded before quota check')):
        try:s.run(thesis,-1)
        except OSError:pass
        else:raise AssertionError('Quota limit ignored')
print('Reader-held-out sentiment integration passed:12conditions,60scores,reference preserved,resume and identity checks.')
