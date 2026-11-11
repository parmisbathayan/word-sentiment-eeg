"""Synthetic cloud checks: all fixations, controls, matching, saved handoff and nulls.

No private TeCo inputs are downloaded or published by this test.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.teco_paper import extract,prepare,verify,montage,TIMES
from word_sentiment_eeg.teco_paper_stats import add_reader,finish
from word_sentiment_eeg.paper_figure_stats import scalp_geometry
from word_sentiment_eeg.common import atomic_npz,write_json
from render_teco_paper import render


def fixture(reader='test',seed=1):
    rng=np.random.default_rng(seed);data={};lookup={}
    for sid in range(1,7):
        sentence='SYNTHETIC '+str(sid);key=hashlib.sha256(sentence.encode()).hexdigest()[:24]
        raw=rng.normal(size=(126,6000)).astype(np.float32);words={}
        for wi,(token,c,starts) in enumerate([('NEG','negative',[1400,1800]),('POS','positive',[3000,3400])]):
            words[wi]=dict(content=token,nFixations=2,fixPositions=[wi*2+1,wi*2+2],rawEEG=[raw[:,s:s+180].copy() for s in starts])
            lookup[sid,key,wi,token]=SimpleNamespace(lexical_class=c,contextual_class=c,sentence_context_class='mixed',alignment_warning='',is_control=sid==1)
        data[sid]=dict(sentenceId=sid,persian_sentence=sentence,raw_data=raw,word=words,allFixations={'duration':[150]*4})
    return data,lookup


with tempfile.TemporaryDirectory() as d:
    root=Path(d);data,lookup=fixture();x,meta,audit,orders,policy=extract(data,'test',lookup)
    assert x.shape==(24,126,201) and x.dtype==np.float32
    assert set(meta.fixation_index)=={0,1},'Must retain rereading fixations'
    assert meta.is_control.astype(str).eq('True').sum()==4,'Control reading must survive'
    assert len(orders[0])==len(orders[1])==12
    # A corrupt second fixation must not affect the first or masquerade as matched.
    data[1]['word'][0]['rawEEG'][1]=np.full((126,180),np.nan)
    _,_,a,_,_=extract(data,'test',lookup)
    assert a.status.eq('invalid_stored_segment').sum()==1
    # Recovering the same physical onset for different words is ambiguous.
    data,lookup=fixture();data[1]['word'][1]['rawEEG'][0]=data[1]['word'][0]['rawEEG'][0].copy()
    _,_,a,_,_=extract(data,'test',lookup)
    assert a.status.eq('ambiguous_duplicate_onset').sum()==2
    # Full native order/duration corroboration is required, not just waveform length.
    data,lookup=fixture();data[1]['allFixations']['duration'][0]=20
    _,_,a,_,_=extract(data,'test',lookup)
    assert a.status.eq('fixation_duration_or_order_disagreement').sum()==1
    stats=[];audits=[];total=np.zeros((126,201));squares=total.copy();n=0;readers=['one','two','three']
    for i,reader in enumerate(readers):
        data,lookup=fixture(seed=10+i);folder=root/'readers'/reader
        prepare(data,reader,lookup,folder,root,100_000_000)
        state=verify(folder)
        with np.load(folder/'statistics.npz',allow_pickle=False) as z:st={k:z[k].copy() for k in z.files};stats.append(st)
        with np.load(folder/'prepared_trials.npz',allow_pickle=False) as z:
            diff=z['signal'][z['positive_pair_index']]-z['signal'][z['negative_pair_index']]
        assert np.array_equal(diff.sum(axis=0,dtype=np.float64),st['pair_sum'])
        add_reader(diff,root/'null_work',reader,tct=4,cluster=4,use_gpu=False,chunk=2)
        before={p.name:p.read_bytes() for p in (root/'null_work').glob('*')}
        add_reader(diff,root/'null_work',reader,tct=4,cluster=4,use_gpu=False,chunk=2)
        assert before=={p.name:p.read_bytes() for p in (root/'null_work').glob('*')},'Resume must not double-add'
        if i==0:
            try:add_reader(diff+1,root/'null_work',reader,tct=4,cluster=4,use_gpu=False,chunk=2)
            except ValueError:pass
            else:raise AssertionError('Changed pairs must be refused')
        total+=st['pair_sum'];squares+=st['pair_squares'];n+=int(st['pairs'])
        audits.append(pd.read_csv(folder/'fixation_audit.csv.gz'))
    xy,adj=scalp_geometry(montage());perm,clusters=finish(root/'null_work',total,squares,n,adj,readers,4,4,TIMES)
    assert perm['cluster_mask'].shape==(126,201) and perm['tct_p'].shape==(201,)
    assert xy.shape==(126,2) and np.isfinite(xy).all()
    analysis=root/'analysis';analysis.mkdir()
    atomic_npz(analysis/'group_results.npz',counts=np.stack([s['counts'] for s in stats]),sums=np.stack([s['sums'] for s in stats]),
        squares=np.stack([s['squares'] for s in stats]),p=np.ones((127,201)),xy=xy,times_ms=TIMES)
    atomic_npz(analysis/'permutation_results.npz',**perm);write_json(analysis/'clusters.json',clusters)
    render(root,readers,stats,pd.concat(audits,ignore_index=True))
    assert len(list((root/'figures').glob('*.png')))==3
    # Checksums detect changed archives before a subsequent GPU use.
    target=root/'readers/one/statistics.npz';target.write_bytes(b'corrupt')
    try:verify(root/'readers/one')
    except ValueError:pass
    else:raise AssertionError('Corrupt saved trials/statistics must fail')
print('PASS: all stored fixations, controls, duration/order checks, duplicate onsets, persistent trial checkpoints, resume, permutations, projection and three figure assemblies')
