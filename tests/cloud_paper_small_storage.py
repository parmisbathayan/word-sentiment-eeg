"""Cloud regression tests: exact null tests, restarts, corruption, no Drive deletes."""
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.common import atomic_npz,file_digest,write_json
from word_sentiment_eeg.paper_figure_stats import permutation_reader,finish_permutations,mixed_models
from word_sentiment_eeg.teco_paper_stats import add_reader,finish
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore,permutations,model_checkpoints

rng=np.random.default_rng(19);C,T=9,15;subjects=['A','B','C'];adj=np.ones((C,C),bool)^np.eye(C,dtype=bool);times=np.arange(T)*8
D=[rng.normal(size=(2,7,C,T)).astype(np.float32) for _ in subjects]
for d in D:d[:,:,0:6,5:10]+=1.5
sums=np.sum([d.sum(axis=1,dtype=np.float64) for d in D],axis=0);sq=np.sum([(d.astype(float)**2).sum(axis=1) for d in D],axis=0);n=21
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);legacy=root/'legacy';drive=root/'drive'
    for subject,d in zip(subjects,D):permutation_reader(d[0],d[1],legacy,subject,7,5,use_gpu=False,chunk=3)
    old,clusters=finish_permutations(legacy,sums,sq,n,adj,subjects)
    store=ImmutableStore(drive,20_000_000)
    real_replace=os.replace;real_unlink=Path.unlink
    def replace(a,b,*args,**kwargs):
        if drive.resolve() in Path(b).resolve().parents:raise AssertionError('Rename/replace on Drive creates Trash versions')
        return real_replace(a,b,*args,**kwargs)
    def unlink(path,*args,**kwargs):
        if drive.resolve() in Path(path).resolve().parents:raise AssertionError('Deleting Drive checkpoints creates Trash')
        return real_unlink(path,*args,**kwargs)
    with patch('os.replace',replace),patch.object(Path,'unlink',unlink):
        assert permutations(D,subjects,sums,sq,n,adj,times,store,'same',7,5,chunk=3,use_gpu=False,max_new_chunks=1)==(None,None)
        initial={p:p.read_bytes() for p in drive.rglob('*.json')}
        actual,newclusters=permutations(D,subjects,sums,sq,n,adj,times,ImmutableStore(drive,20_000_000),'same',7,5,chunk=3,use_gpu=False)
        for name in old:np.testing.assert_array_equal(old[name],actual[name],err_msg=name)
        assert [(x['mass'],x['p'],x['samples']) for x in clusters]==[(x['mass'],x['p'],x['samples']) for x in newclusters]
        assert any(x['mass']>0 for x in newclusters),'Exercise nonempty spatial-temporal clusters'
        for p,b in initial.items():assert p.read_bytes()==b
        snapshot={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in drive.rglob('*') if p.is_file()}
        permutations(D,subjects,sums,sq,n,adj,times,ImmutableStore(drive,20_000_000),'same',7,5,chunk=3,use_gpu=False)
        assert snapshot=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in drive.rglob('*') if p.is_file()}
        assert not list(drive.rglob('*.npz')),'No large null maps on Drive'
        assert sum(p.stat().st_size for p in drive.rglob('*') if p.is_file())<30_000
        # An interrupted prefix is reconstructed in place, with no delete/replace.
        first=next(drive.rglob('tct_*.json'));original=first.read_bytes();first.write_bytes(original[:30])
        permutations(D,subjects,sums,sq,n,adj,times,ImmutableStore(drive,20_000_000),'same',7,5,chunk=3,use_gpu=False)
        assert first.read_bytes()==original
    try:permutations(D,subjects,sums,sq,n,adj,times,store,'changed',7,5,chunk=3,use_gpu=False)
    except ValueError:pass
    else:raise AssertionError('Changed identity accepted')
    corrupt=json.loads(first.read_text());corrupt['begin']+=1;first.write_text(json.dumps(corrupt))
    try:permutations(D,subjects,sums,sq,n,adj,times,store,'same',7,5,chunk=3,use_gpu=False)
    except ValueError:pass
    else:raise AssertionError('Corrupt completed batch accepted')
    # TeCo's single raw representation must match its previous null engine too.
    ds=[d[:1] for d in D];single=root/'single';legacy_single=root/'single_old'
    for subject,d in zip(subjects,ds):add_reader(d[0],legacy_single,subject,7,5,use_gpu=False,chunk=3)
    old1,_=finish(legacy_single,sums[0],sq[0],n,adj,subjects,7,5,times)
    new1,_=permutations(ds,subjects,sums[:1],sq[:1],n,adj,times,ImmutableStore(single,20_000_000),'single',7,5,chunk=3,use_gpu=False)
    for name in old1:np.testing.assert_array_equal(old1[name],new1[name][0],err_msg='TeCo '+name)
    # ML checkpoint interrupted after 200 features; resume on a fresh local disk.
    y=rng.normal(size=(3,2,10,201));counts=np.full((3,2),10);sy=y.sum(axis=2);ss=(y*y).sum(axis=2)
    expected=mixed_models(counts,sy,ss)
    model_drive=root/'model_drive';local=root/'local1';local.mkdir()
    class Interrupted(ImmutableStore):
        def file(self,relative,source,**kwargs):
            result=super().file(relative,source,**kwargs)
            if 'model_frames' in str(relative):raise KeyboardInterrupt('simulated disconnect after committed model frame')
            return result
    try:
        with model_checkpoints(Interrupted(model_drive,10_000_000),local):mixed_models(counts,sy,ss,checkpoint=local/'raw_ml_checkpoint.npz')
    except KeyboardInterrupt:pass
    else:raise AssertionError('Interruption not injected')
    local2=root/'local2';local2.mkdir()
    with model_checkpoints(ImmutableStore(model_drive,10_000_000),local2):got=mixed_models(counts,sy,ss,checkpoint=local2/'raw_ml_checkpoint.npz')
    for k in expected:np.testing.assert_array_equal(expected[k],got[k],err_msg='ML resume '+k)
    assert len(list((model_drive/'analysis_state/model_frames/raw_ml_checkpoint').glob('*.npz')))==2
print('PASS: exact ZuCo + TeCo null results, nonempty clusters, fresh-process batch resume, torn writes, corruption/identity refusal, unchanged ML checkpoint resume, no null maps/Drive deletes')
