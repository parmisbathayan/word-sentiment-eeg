"""Small cloud-only integration checks; no private data or local installs."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_spatial40 as t
import audit_zuco_overlap as audit


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


rng=np.random.default_rng(41)
meta=pd.DataFrame([dict(event_id=f'{r}-{j}',subject=f'R{r}',truth='positive' if j%2 else 'negative',
    sentence_key=f's{j}',word_identity=f'w{j%20}',sentence_position=j,block_id='1',onset_seconds=j*.1,
    dataset='teco',onset_sample_sentence_0based=300,epoch_file='epoch.npz',epoch_row=j)
    for r in range(12) for j in range(84)])
baseline=dict(post=rng.normal(size=(len(meta),48)),pre=rng.normal(size=(len(meta),12)),nuisance=rng.normal(size=(len(meta),6)))
# Eligibility is decided from counts, never scores. Check exclusions explicitly.
small=meta.copy();small.loc[(small.subject=='R11')&small.truth.eq('negative')&small.epoch_row.lt(6),'truth']='positive'
kept,_,count=t.cohort(small,baseline);assert kept.subject.nunique()==11 and not count.loc[count.subject=='R11','included'].iloc[0]
cohort,baseline,counts=t.cohort(meta,baseline);assert len(cohort)==len(meta) and counts.included.all()
montage=pd.DataFrame(dict(channel=[f'C{i}' for i in range(12)],region=np.repeat(t.a.REGIONS,2)))
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);archive=root/'archive';archive.mkdir();montage.to_csv(archive/'teco_channel_regions.csv',index=False)
    expected_post=[];expected_pre=[];frames=[];times=np.arange(-25,101)*8
    for subject,g in cohort.groupby('subject',sort=True):
        folder=archive/'teco/readers/task1'/subject;folder.mkdir(parents=True)
        eeg=rng.normal(size=(len(g),12,126)).astype(np.float32)
        eeg[g.truth.eq('positive').to_numpy(),0,:]+=0.1
        np.savez_compressed(folder/'epoch.npz',eeg_uV=eeg,event_id=g.event_id.to_numpy(dtype=str),
            channel_names=montage.channel.to_numpy(dtype=str),times_ms=times)
        import hashlib
        h=g.copy();h['signal_sha256']=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in eeg];frames.append(h)
        save(folder/'checkpoint.json',dict(blocks=[dict(checksums={'epoch.npz':t.a.file_digest(folder/'epoch.npz')})]))
        expected_post.append((g.index,eeg[:,:,(times>=224)&(times<=304)].mean(axis=2)))
        expected_pre.append((g.index,eeg[:,:,(times>=-88)&(times<=-8)].mean(axis=2)))
    meta=pd.concat(frames).sort_index()
    store=t.ImmutableStore(root/'out',20_000_000)
    post,pre=t.cached_channel_means(archive,meta,montage,store)
    for ix,x in expected_post:np.testing.assert_allclose(post[ix],x)
    for ix,x in expected_pre:np.testing.assert_allclose(pre[ix],x)
    original_read=Path.read_bytes
    def no_epoch_reads(path):
        if path.name=='epoch.npz':raise AssertionError('Repeated epoch read')
        return original_read(path)
    with patch.object(Path,'read_bytes',no_epoch_reads):
        again,_=t.cached_channel_means(archive,meta,montage,store)
    np.testing.assert_array_equal(post,again)
    features=t.spatial_features(post,pre,baseline,montage)
    assert features['whole_head'].shape==(len(meta),12) and features['six_region_means'].shape==(len(meta),6)
    y=meta.truth.eq('positive').to_numpy(dtype=int)
    folds=t.z.grouped_folds(y,meta.subject.to_numpy());train,test=folds[0]
    chosen,inner=t.select_region(meta,y,features,train,40,20261113)
    altered={k:v.copy() for k,v in features.items()}
    for v in altered.values():v[test]=1e9
    changed=y.copy();changed[test]=1-changed[test]
    other,other_inner=t.select_region(meta,changed,altered,train,40,20261113)
    assert chosen==other and inner==other_inner,'Outer test affected region selection'
    with patch.dict(t.SPEC,regroupings40=2):rows=t.evaluate(meta,features,store)
    assert len(rows)==len([*features,'selected_region'])*3
    for score in rows:
        assert score['test_examples']==len(meta) if score['size']==1 else score['test_examples']==24
    sizes=pd.read_csv(root/'out/size40/repeat00/memberships.csv.gz')
    assert sizes.trials.eq(40).all()
    for fold,g in sizes.groupby('fold'):
        ids={partition:{event for s in table.event_ids for event in json.loads(s)} for partition,table in g.groupby('partition')}
        assert not ids['train']&ids['test']
        lookup=meta.set_index('event_id')
        assert not set(lookup.loc[list(ids['train']),'subject'])&set(lookup.loc[list(ids['test']),'subject'])
    before={str(p):p.stat().st_mtime_ns for p in (root/'out').rglob('*') if p.is_file()}
    with patch.dict(t.SPEC,regroupings40=2):resumed=t.evaluate(meta,features,store)
    assert rows==resumed and before=={str(p):p.stat().st_mtime_ns for p in (root/'out').rglob('*') if p.is_file()}
    # Overlap audit distinguishes a long epoch from the actual narrow predictor.
    simple=pd.DataFrame(dict(subject=['A','A','A','B'],block_id=['1']*4,onset_seconds=[0,.1,.04,.01]))
    assert audit.count_overlap(simple,[0],[1],10)['cross_partition_event_pairs']==0
    assert audit.count_overlap(simple,[0],[1],99)['cross_partition_event_pairs']==1
    assert audit.count_overlap(simple,[0],[2],10)['cross_partition_event_pairs']==1
    assert audit.count_overlap(simple,[0],[3],99)['cross_partition_event_pairs']==0
    assert audit.audit(root/'no_data',store)['unverified']
    # Reconstruct a fully checksum-verified historical metadata-only run.
    thesis=root/'history';parent=thesis/'Results/word_sentiment_eeg/zuco_paper_figures_recovered'
    report=thesis/'Results/word_sentiment_eeg/zuco_recovered_followups'
    historical=[];subjects=sorted(meta.subject.unique());prepared={}
    for subject in subjects:
        g=meta[meta.subject==subject].copy();g['word']=g.word_identity;g['sentiment']=g.truth
        folder=parent/'readers'/subject;folder.mkdir(parents=True)
        g.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False)
        save(folder/'prepared.json',dict(checksums={'accepted_trial_metadata.csv.gz':t.a.file_digest(folder/'accepted_trial_metadata.csv.gz')}))
        prepared[subject]=t.a.file_digest(folder/'prepared.json');historical.append(g)
    save(parent/'cpu_preparation_manifest.json',dict(subjects=subjects,reader_checkpoint_sha256=prepared))
    history=pd.concat(historical,ignore_index=True);history['word_key']=history.word.map(t.z.clean_word)
    labels=history.sentiment.eq('positive').to_numpy(dtype=int);proof={}
    for mode,column in [('sentence','sentence_key'),('word','word_key'),('reader','subject')]:
        for window in ['published_224_304','post_0_800']:
            for size in [1,40]:
                f=[]
                for fi,(tr,te) in enumerate(t.z.grouped_folds(labels,history[column].to_numpy())):
                    _,_,_,ma=t.z.averaged(np.zeros((len(history),1)),labels,history,tr,size,t.z.SEED+fi)
                    _,_,_,mb=t.z.averaged(np.zeros((len(history),1)),labels,history,te,size,t.z.SEED+100+fi)
                    f.append(dict(train_event_digest=t.a.digest(sorted(history.event_id.iloc[ma].tolist())),
                                  test_event_digest=t.a.digest(sorted(history.event_id.iloc[mb].tolist()))))
                for rep in ['raw','cleaned']:
                    relative=f'classification/{rep}__{window}__{mode}__avg{size}__logistic.json'
                    save(report/relative,dict(folds=f));proof[relative]=t.a.file_digest(report/relative)
    save(report/'cpu_complete.json',dict(checksums=proof))
    audited=audit.audit(thesis,store);assert len(audited['verified_rows'])==36
    assert all(v['cross_partition_event_pairs']==0 for v in audited['verified_rows'] if v['window']=='published_224_304' or v['split']=='reader')
    assert any(v['cross_partition_event_pairs']>0 for v in audited['verified_rows'] if v['window']=='post_0_800' and v['split']=='word')
    relative='classification/raw__post_0_800__word__avg40__logistic.json'
    changed=json.loads((report/relative).read_text());changed['folds'][0]['train_event_digest']='bad'
    save(report/relative,changed);proof[relative]=t.a.file_digest(report/relative)
    save(report/'cpu_complete.json',dict(checksums=proof))
    refused=audit.audit(thesis,store)
    assert len(refused['verified_rows'])==33 and any('membership' in v['reason'] for v in refused['unverified'])
    # Run the delivered reporting cells against the completed synthetic results.
    t.csv(store,'scores.csv',pd.DataFrame(rows));t.csv(store,'reader_cohort.csv',counts)
    store.json('zuco_overlap_audit.json',refused)
    store.json('complete.json',dict(status='completed',checksums={'scores.csv':t.a.file_digest(store.root/'scores.csv')}))
    from matplotlib.figure import Figure
    figures=[]
    def display(value):
        if isinstance(value,Figure):figures.append(value)
    scope=dict(OUT=store.root,json=json,pd=pd,display=display)
    book=json.loads((ROOT/'notebooks/teco_spatial40.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code' and cell['id'] in ['spatial-results','spatial-primary','spatial-regions','spatial-readers','spatial-overlap']:
            exec(compile(''.join(cell['source']),'<spatial-report-cell>','exec'),scope)
    assert len(figures)==2 and all(len(f.axes)==1 for f in figures)


print('Spatial means, persistent reuse, exact averaging, nested selection, reader isolation and overlap geometry passed.')
