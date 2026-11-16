"""Cloud-only numerical checks for partitioned TeCo averaging and resume."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import teco_average40 as t

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
    times={str(p):p.stat().st_mtime_ns for p in Path(tmp).rglob('*') if p.is_file()}
    resumed,_=t.analyze(meta,arrays,store)
    assert rows==resumed and times=={str(p):p.stat().st_mtime_ns for p in Path(tmp).rglob('*') if p.is_file()}
    corrupted=Path(tmp)/'cases/sentence__max40__post__logistic/predictions.csv.gz'
    corrupted.write_bytes(b'corrupt')
    try:t.analyze(meta,arrays,store)
    except ValueError as e:assert 'checksum' in str(e)
    else:raise AssertionError('Corrupt result reused')
print('TeCo averaging, holdouts, membership matching, resume and corruption checks passed.')
