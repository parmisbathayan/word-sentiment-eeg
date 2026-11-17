"""Cloud-only numerical checks for partitioned TeCo averaging and resume."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import teco_average40 as t


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


# Full cache loader regression: labels are ONLY in the original completed
# predictions. The old annotation CSV can be missing or replaced.
with tempfile.TemporaryDirectory() as tmp:
    thesis=Path(tmp);root=thesis/'Results/word_sentiment_eeg'
    source=root/'classification_similarity';archive=root/'trial_archive_125hz'
    save(source/'manifest.json',dict(status='completed'))
    save(source/'protocol.json',dict(original=True))
    save(archive/'protocol.json',dict(test_archive=True))
    records=[];roster=[]
    for r in range(12):
        subject=f'R{r}';ids=[f'{subject}-neg',f'{subject}-pos']
        record=dict(dataset='teco',task='task1',subject=subject,epochs=2);records.append(record)
        m=pd.DataFrame(dict(event_id=ids,subject=[subject]*2,dataset=['teco']*2,task=['task1']*2,
            sentence_key=['s0','s1'],sentence_position=[0,1],onset_sample_sentence_0based=[300,300],
            signal_sha256=ids,is_control=['false','true'],word_identity=['bad','good'],
            **{c:[1.,2.] for c in t.a.NUISANCE if c!='control_numeric'}))
        reader=archive/'teco/readers/task1'/subject;reader.mkdir(parents=True)
        m.to_csv(reader/'events.csv.gz',index=False)
        checkpoint=dict(status='completed',epochs=2,identity=dict(protocol_sha256=t.a.digest(read_archive:=dict(test_archive=True))),
            events_sha256=t.a.file_digest(reader/'events.csv.gz'))
        save(reader/'checkpoint.json',checkpoint)
        cache=source/'features/teco/task1'/subject;cache.mkdir(parents=True)
        m.to_csv(cache/'events.csv.gz',index=False)
        np.savez(cache/'features.npz',post=np.full((2,48),r,dtype=float),pre=np.zeros((2,12)),event_id=np.array(ids))
        signature=t.a.digest(dict(spec=t.a.FEATURE_SPEC,checkpoint=t.a.file_digest(reader/'checkpoint.json')))
        save(cache/'manifest.json',dict(signature=signature,checksums={n:t.a.file_digest(cache/n) for n in ['features.npz','events.csv.gz']}))
        roster.extend(dict(event_id=e,subject=subject,truth=c) for e,c in zip(ids,['negative','positive']))
    # Reverse frozen ordering to prove row/event alignment, not incidental order.
    frozen=pd.DataFrame(roster[::-1]);case=source/'teco/task1/cases/word_agreement__binary__sentence__post__logistic'
    case.mkdir(parents=True);frozen.to_csv(case/'predictions.csv.gz',index=False)
    save(case/'result.json',dict(checksums={'predictions.csv.gz':t.a.file_digest(case/'predictions.csv.gz')},
        score=dict(trials=24,readers=12,class_counts=dict(negative=12,positive=12))))
    with patch.object(t.a,'load_completed',return_value=(records,{})), \
         patch.object(t.a,'reader_features',side_effect=AssertionError('Live helper must not run')), \
         patch.object(t.a,'attach_available_labels',side_effect=AssertionError('Live labels must not run')):
        meta,features,provenance=t.load_cached(thesis)
        assert meta.event_id.tolist()==frozen.event_id.tolist() and meta.truth.tolist()==frozen.truth.tolist()
        assert features['post'][:,0].tolist()==[int(s[1:]) for s in frozen.subject]
        assert provenance['frozen_labels']['class_counts']==dict(negative=12,positive=12)
        # An unrelated revised annotation must not change the frozen selection.
        current=archive/'annotations/teco_word_labels.csv';current.parent.mkdir();current.write_text('revised,labels\nwrong,negative\n')
        again,_,_=t.load_cached(thesis);pd.testing.assert_frame_equal(meta,again)
        (case/'predictions.csv.gz').write_bytes(b'corrupt')
        try:t.load_cached(thesis)
        except ValueError as e:assert 'checksum' in str(e)
        else:raise AssertionError('Corrupt frozen labels reused')

rng=np.random.default_rng(47)
meta=pd.DataFrame([dict(event_id=f'{r}-{j}',subject=f'R{r}',truth='positive' if j%2 else 'negative',
    sentence_key=f's{j}',word_identity=f'w{j%20}',sentence_position=j,
    dataset='teco',onset_sample_sentence_0based=300) for r in range(12) for j in range(80)])
x=rng.normal(size=(len(meta),48));x[:,0]+=(meta.truth=='positive').to_numpy()*0.5
for n in [1,40]:
    g=t.groups(meta,np.arange(len(meta)),n,2026)
    assert max(map(len,g))<=n and sum(map(len,g))==len(meta)
    assert [v.tolist() for v in g]==[v.tolist() for v in t.groups(meta,np.arange(len(meta)),n,2026)]
    _,y=t.average(x,meta,g)
    for ix,label in zip(g,y):
        assert meta.truth.iloc[ix].eq(label).all() and meta.subject.iloc[ix].nunique()==1

with tempfile.TemporaryDirectory() as tmp:
    store=t.ImmutableStore(tmp,10_000_000)
    arrays=dict(post=x,pre=rng.normal(size=(len(meta),12)),nuisance=rng.normal(size=(len(meta),6)))
    rows,skips=t.analyze(meta,arrays,store)
    assert len(rows)==18 and not skips
    assert all(r['original_test_trials']==len(meta) for r in rows)
    for mode in ['sentence','word','reader']:
        path=Path(tmp)/f'cases/{mode}__max40__post__logistic'
        m=pd.read_csv(path/'memberships.csv.gz')
        lookup=meta.set_index('event_id')
        for fold,f in m.groupby('fold'):
            ids={part:{v for s in g.event_ids for v in json.loads(s)} for part,g in f.groupby('partition')}
            assert not ids['train']&ids['test']
            column={'sentence':'sentence_key','word':'word_identity','reader':'subject'}[mode]
            assert not set(lookup.loc[list(ids['train']),column])&set(lookup.loc[list(ids['test']),column])
        d=pd.read_csv(Path(tmp)/f'cases/{mode}__max40__post__dummy/memberships.csv.gz')
        pd.testing.assert_frame_equal(m,d)
    # Execute the actual notebook plotting cells against completed synthetic
    # outputs. CI has no Colab/IPython, so inject only its display function.
    t.csv(store,'scores.csv',pd.DataFrame(rows))
    paths=['scores.csv']+[str(p.relative_to(tmp)) for p in Path(tmp).glob('cases/*/result.json')]
    store.json('complete.json',dict(status='completed',cases=18,checksums={k:t.a.file_digest(Path(tmp)/k) for k in paths}))
    book=json.loads((ROOT/'notebooks/teco_average40_results.ipynb').read_text())
    from matplotlib.figure import Figure
    figures=[]
    def show(value):
        if isinstance(value,Figure):figures.append(value)
    scope=dict(display=show)
    for cell in book['cells']:
        if cell['cell_type']!='code' or cell['id']=='teco40-report-mount':continue
        code=''.join(cell['source']).replace('from IPython.display import display\n','')
        code=code.replace("Path('/content/drive/MyDrive/Thesis/Results/word_sentiment_eeg/teco_average40')",f'Path({tmp!r})')
        exec(compile(code,'<saved-result-plot-cell>','exec'),scope)
    assert len(figures)==3 and [len(f.axes) for f in figures]==[3,3,6]
    times={str(p):p.stat().st_mtime_ns for p in Path(tmp).rglob('*') if p.is_file()}
    resumed,_=t.analyze(meta,arrays,store)
    assert rows==resumed and times=={str(p):p.stat().st_mtime_ns for p in Path(tmp).rglob('*') if p.is_file()}
    corrupted=Path(tmp)/'cases/sentence__max40__post__logistic/predictions.csv.gz'
    corrupted.write_bytes(b'corrupt')
    try:t.analyze(meta,arrays,store)
    except ValueError as e:assert 'checksum' in str(e)
    else:raise AssertionError('Corrupt result reused')
print('TeCo averaging, holdouts, membership matching, resume and corruption checks passed.')
