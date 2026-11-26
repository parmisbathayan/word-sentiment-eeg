"""Cloud-only tests for occurrence controls, numeric EEG regression and true refixation pairs."""
from pathlib import Path
import tempfile,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import review_controls as c
import review_fixations as f
import experiment_review_followup as main
t=c.t
rng=np.random.default_rng(37)
meta=pd.DataFrame([dict(event_id=f'{r}/{j}',dataset='teco',task='task1',subject=f'R{r}',
    sentence_key=f'S{j//8}',sentence_position=j//8,word_index=j%8,word=f'w{j%10}',word_identity=f'w{j%10}',
    onset_sample_sentence_0based=(j%8)*3000,word_length=2+j%10,relative_word_position=(j%8)/7,
    sentence_length=8,is_control=bool(j%2),first_fixation_duration_ms=130+10*r+30*(j%3))
    for r in range(12) for j in range(96)])
y=np.tile(np.arange(96)%3,12)
collapsed,cy,x,conflict=c.collapse(meta,y,['a','b','c'])
assert len(collapsed)==96 and len(conflict)==0
assert collapsed.occurrence_key.nunique()==96
# The measured duration differs across readers, but is not an input to the four-feature controls.
expected=meta.set_index('event_id').loc[collapsed.event_id].reset_index()
assert x.shape==(96,4) and np.array_equal(x,c.metadata_four(expected),equal_nan=True)
bad=meta.copy();bad.loc[0,'word_length']=999
rr,_,_,conflicts=c.collapse(bad,y,['a','b','c']);assert len(rr)==95 and len(conflicts)==1
bad_y=y.copy();bad_y[0]=1
rr,_,_,conflicts=c.collapse(meta,bad_y,['a','b','c']);assert len(rr)==95 and conflicts[0]['label_conflict']
for mode in ['occurrence_random','sentence','word']:
    for tr,te in c.standalone_folds(collapsed,cy,mode):
        assert not set(collapsed.occurrence_key.iloc[tr])&set(collapsed.occurrence_key.iloc[te])
gy=np.zeros(95,dtype=int)
assert [len(g) for g in c.independent_groups(gy,np.arange(95),40,0)]==[40,40]
assert [len(g) for g in c.independent_groups(gy,np.arange(95),'max40',0)]==[40,40,15]
# One test average/class is insufficient, even though no class is absent.
try:c.score_control(np.ones((8,4)),np.array([0,0,0,0,1,1,1,1]),[np.array([0]),np.array([4])],[np.array([1]),np.array([5])],['a','b'],'metadata_four')
except ValueError:pass
else:raise AssertionError('Too-few test averages accepted')
with tempfile.TemporaryDirectory() as tmp:
    store=t.ImmutableStore(Path(tmp),50_000_000)
    # Legacy linguistic completion records omitted events.csv.gz. Require an exact
    # reconstruction, including row order/labels, instead of bypassing verification.
    legacy=Path(tmp)/'legacy';(legacy/'targets/pos').mkdir(parents=True)
    definition=legacy/'targets/pos/definition.json';definition.write_text('{}')
    columns=['event_id','dataset','task','subject','sentence_key','word_identity','sentence_position','onset_sample_sentence_0based']
    roster=meta[columns].copy();roster['target_code']=y
    path=legacy/'targets/pos/events.csv.gz';roster.to_csv(path,index=False,compression='gzip')
    state={'checksums':{'targets/pos/definition.json':t.a.file_digest(definition)}}
    assert main.checked_events(legacy,'targets/pos/events.csv.gz',state,'linguistic_targets',(meta,y)).startswith('exact_reconstruction')
    for altered in [roster.iloc[::-1],roster.assign(target_code=(y+1)%3),roster.iloc[:-1]]:
        altered.to_csv(path,index=False,compression='gzip')
        try:main.checked_events(legacy,'targets/pos/events.csv.gz',state,'linguistic_targets',(meta,y))
        except ValueError:pass
        else:raise AssertionError('Changed legacy roster accepted')
    roster.to_csv(path,index=False,compression='gzip')
    state['checksums']['targets/pos/events.csv.gz']='bad checksum'
    try:main.checked_events(legacy,'targets/pos/events.csv.gz',state,'linguistic_targets',(meta,y))
    except ValueError:pass
    else:raise AssertionError('Present but wrong checksum bypassed')
    protocol_store=t.ImmutableStore(Path(tmp)/'protocol',50_000_000)
    old={'spec':main.SPEC,'code':{'scripts/experiment_review_followup.py':main.LEGACY_REVIEW_SHA256,'other':'unchanged'}}
    protocol_store.json('protocol.json',old)
    new=dict(old,code=dict(old['code'],**{'scripts/experiment_review_followup.py':'new audit code'}))
    main.record_protocol(protocol_store,new);main.record_protocol(protocol_store,new)
    assert t.read(protocol_store.root/'protocol.json')==old
    try:main.record_protocol(protocol_store,dict(new,spec={'changed':True}))
    except ValueError:pass
    else:raise AssertionError('Changed scientific protocol accepted')
    rows,missing=c.run_standalone(meta,y,['a','b','c'],'pos',store,sizes=[1,10,'max40'])
    assert rows and all(r['unit']=='unique source occurrences' for r in rows)
    audit=c.audit_partition(meta,y,np.arange(96),np.arange(96,192),'test','pos','reader',1,1)
    assert audit['fraction_test_events_with_seen_source']==1.
    assert audit['test_events_with_seen_four_feature_label_pair']==96
    # True same-occurrence pairing, native fixation order and onsets.
    raw=rng.normal(0,.2,(126,6000)).astype(np.float32);onsets=[900,2400];duration=(144-1)*1000/1200
    segments=[raw[:,o:o+144].copy() for o in onsets];durations=np.full(8,duration)
    word=dict(nFixations=2,fixPositions=[3,8],rawEEG=segments,content='example')
    pair=f.confirmed_pair(raw,word,durations)
    assert [r['onset'] for r in pair]==onsets
    reverse=dict(word,fixPositions=[8,3],rawEEG=segments[::-1]);assert [r['onset'] for r in f.confirmed_pair(raw,reverse,durations)]==onsets
    try:f.confirmed_pair(raw,dict(word,fixPositions=[3,3]),durations)
    except ValueError:pass
    else:raise AssertionError('Duplicate native fixation order accepted')
    regions=np.array(t.a.REGIONS*21)
    filtered=f.preprocess40(raw,1200);epoch,_,_=f.make_epoch(filtered,900,1200)
    post,pre=t.a.feature_blocks(epoch[None],np.arange(-25,101)*8,regions)
    import hashlib
    sentence='a private synthetic sentence';key=hashlib.sha256(sentence.encode()).hexdigest()[:24]
    one=meta.iloc[:1].copy();one['sentence_key']=key;one['sentence_id_teco']=1;one['word_index']=0;one['word']='example';one['onset_sample_sentence_0based']=900
    data={0:dict(sentenceId=1,persian_sentence=sentence,raw_data=raw,allFixations=dict(duration=durations),word={0:word})}
    pairs,px,pb,trace=f.reader_pairs(data,one,post,regions)
    assert len(pairs)==1 and px.shape==(1,2,48) and trace[0]['status']=='accepted'
    # Mismatched cached preparation is a hard error, never a silent exclusion.
    try:f.reader_pairs(data,one,post+1,regions)
    except RuntimeError:pass
    else:raise AssertionError('Mismatched EEG preparation accepted')
    # Identical metadata for first/later, matched word groups: necessarily chance BA.
    paired=np.repeat(rng.normal(size=(30,1,4)),2,axis=1)
    result=f.score_pairs(paired,[np.array([i]) for i in range(20)],[np.array([i]) for i in range(20,30)],'lr_1')
    assert t.balanced_accuracy_score(result['truth'],result['prediction'])==.5
    # Regressions see EEG vectors only; outcome properties are never appended to X.
    rx=rng.normal(size=(120,48));ry=2*rx[:,0]+rng.normal(size=120)*.1
    for model in ['ridge1','ridge100','extra_trees','dummy']:
        result=c.regression_score(rx,ry,np.arange(90),np.arange(90,120),model)
        assert len(result['prediction'])==30 and result['test_events']==30
    # Pair-level source word/reader/sentence holdout and full-pair overlap purge.
    pm=meta.iloc[:240].copy().reset_index(drop=True);pm['pair_id']=[f'P{i}' for i in range(len(pm))];pm['first_onset']=pm.onset_sample_sentence_0based;pm['later_onset']=pm.first_onset+400
    for mode in ['reader','sentence','word']:
        if (pm.subject if mode=='reader' else pm.sentence_key if mode=='sentence' else pm.word_identity).nunique()<3:continue
        for tr,te in f.pair_folds(pm,mode):assert not set(pm.pair_id.iloc[tr])&set(pm.pair_id.iloc[te])
print('Experiment review tests passed: occurrence independence, label conflicts, group counts, native pair validation, preparation identity, paired controls, regression and leakage guards.')
