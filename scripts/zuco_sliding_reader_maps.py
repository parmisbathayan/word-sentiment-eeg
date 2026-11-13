#!/usr/bin/env python3
"""Exact reader sign-flip sliding scalp-map test; saved sufficient stats only.

Fixed80ms endpoint span,8ms step across -600..1000ms, raw+clean together.
A new exploratory analysis, not original TCT/TANOVA or electrode cluster test.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import digest,file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

SPEC=dict(span_ms=80,step_ms=8,epoch_ms=[-600,1000],representations=['raw','cleaned'],
    spatial_mean_removed=True,reader_weighting='equal',correction='single maximum over both representations and all windows',
    statistic='mean contrast map RMS squared / expected sign-flip RMS squared; scale fixed under reader sign flips',
    warning='Exploratory, already-inspected dataset; not original TCT/TANOVA, not proof of sentiment specificity',budget_bytes=10_000_000)

def scan(maps,times,span=80,step=8,batch=128):
    """Shared exact sign flips across raw/clean and overlapping windows.

    Per-window sign-invariant scale makes window magnitudes comparable in the
    maximum statistic. This leaves each window's uncorrected permutation p
    unchanged. All-zero windows have statistic0/p1.12readers ->4096assignments.
    Reader contrast symmetry/exchangeability remains an assumption.
    """
    maps=np.asarray(maps,float);times=np.asarray(times)
    if maps.ndim!=4 or not np.isfinite(maps).all() or maps.shape[-1]!=len(times) or not 3<=maps.shape[1]<=16:raise ValueError('Expected finite representation×reader×channel×time contrasts')
    if not np.array_equal(times,np.arange(-75,126)*8) or span%8 or step%8 or not 0<span<=1600 or step<=0 or batch<1:raise ValueError('Invalid fixed sampling/window configuration')
    starts=np.arange(0,len(times)-span//8,step//8);L=span//8+1
    m=np.stack([maps[:,:,:,j:j+L].mean(axis=-1) for j in starts],axis=2) # R,S,W,C
    m=m-m.mean(axis=-1,keepdims=True)
    R,S,W,C=m.shape;signs=(1-2*((np.arange(2**S)[:,None]>>np.arange(S))&1)).astype(float)
    scale=np.mean(np.sum(m*m,axis=1),axis=-1)/(S*S)
    observed_rms=np.sqrt(np.mean(m.mean(axis=1)**2,axis=-1))
    stat=np.divide(observed_rms**2,scale,out=np.zeros((R,W)),where=scale>0)
    exceeds=np.zeros((R,W),int);maxima=np.zeros(len(signs))
    for r in range(R):
        for begin in range(0,len(signs),batch):
            end=min(begin+batch,len(signs));v=(signs[begin:end]@m[r].reshape(S,-1)/S).reshape(end-begin,W,C)
            null=np.divide(np.mean(v*v,axis=-1),scale[r],out=np.zeros((end-begin,W)),where=scale[r]>0)
            exceeds[r]+=np.sum(null>=stat[r][None]-1e-12,axis=0)
            maxima[begin:end]=np.maximum(maxima[begin:end],null.max(axis=1))
    p=exceeds/len(signs)
    adjusted=np.mean(maxima[:,None,None]>=stat[None]-1e-12,axis=0)
    return dict(start_ms=times[starts],end_ms=times[starts+L-1],center_ms=(times[starts]+times[starts+L-1])/2,
        rms_uV=observed_rms,statistic=stat,p_uncorrected=p,p_familywise=adjusted,
        assignments=len(signs),samples_per_window=L,windows=W,reader_maps=m)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud only; no data/dependency download on Mac')
    run=args.run_dir or args.thesis_root/'Results/word_sentiment_eeg/zuco_paper_figures_recovered'
    out=args.output or args.thesis_root/'Results/word_sentiment_eeg/zuco_sliding_reader_maps'
    if out.resolve()==run.resolve() or run.resolve() in out.resolve().parents:raise ValueError('Separate output required')
    state=json.loads((run/'cpu_preparation_manifest.json').read_text())
    if state['status']!='completed' or state['protocol_sha256']!=file_digest(run/'protocol.json'):raise ValueError('Incomplete/changed parent preparation')
    subjects=state['subjects'];input_hashes={};maps=[]
    for i,s in enumerate(subjects,1):
        folder=run/'readers'/s;mark=folder/'prepared.json'
        if file_digest(mark)!=state['reader_checkpoint_sha256'][s]:raise ValueError('Changed reader checkpoint')
        metadata=json.loads(mark.read_text());path=folder/'statistics.npz';sha=file_digest(path)
        if sha!=metadata['checksums']['statistics.npz']:raise ValueError('Corrupt reader statistics')
        input_hashes[s]=sha
        # No single-trial EEG archive is opened by this script.
        with np.load(path,allow_pickle=False) as z:
            times=z['times_ms'];n=int(z['pairs']);value=z['pair_sum'].copy()
            if value.shape!=(2,105,201) or n<2 or not np.array_equal(times,np.arange(-75,126)*8):raise ValueError('Bad paired-map schema')
            maps.append(value/n)
        print(f'LOAD SMALL READER STATISTICS {i}/{len(subjects)} {s}',flush=True)
    protocol=dict(spec=SPEC,inputs=input_hashes,code=file_digest(Path(__file__)),software={n:version(n) for n in ['numpy','pandas','matplotlib']})
    store=ImmutableStore(out,SPEC['budget_bytes']);store.json('protocol.json',protocol)
    mark=out/'complete.json'
    if mark.exists():
        for relative,sha in json.loads(mark.read_text())['checksums'].items():
            if file_digest(out/relative)!=sha:raise ValueError('Changed saved scan output')
        print('KEEP COMPLETED SCAN: results verified',flush=True);return
    result=scan(np.stack(maps,axis=1),times);rows=[]
    for r,rep in enumerate(['raw','cleaned']):
        for j in range(result['windows']):
            rows.append(dict(representation=rep,start_ms=int(result['start_ms'][j]),end_ms=int(result['end_ms'][j]),
                center_ms=float(result['center_ms'][j]),rms_contrast_uV=float(result['rms_uV'][r,j]),
                statistic=float(result['statistic'][r,j]),p_uncorrected=float(result['p_uncorrected'][r,j]),
                p_familywise=float(result['p_familywise'][r,j]),significant=bool(result['p_familywise'][r,j]<.05)))
    store.json('window_results.json',dict(spec=SPEC,subjects=subjects,assignments=result['assignments'],samples_per_window=result['samples_per_window'],windows_per_representation=result['windows'],rows=rows))
    with tempfile.TemporaryDirectory() as temporary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        frame=pd.DataFrame(rows);local=Path(temporary)/'window_results.csv';frame.to_csv(local,index=False);store.file(local.name,local)
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for r,rep in enumerate(['raw','cleaned']):
            v=frame[frame.representation.eq(rep)]
            axes[r,0].plot(v.center_ms,v.rms_contrast_uV,color='#323232');axes[r,0].set_ylabel('Mean contrast-map RMS (µV)')
            axes[r,1].plot(v.center_ms,v.p_uncorrected,label='Per-window uncorrected',alpha=.6)
            axes[r,1].plot(v.center_ms,v.p_familywise,label='Corrected across complete scan',color='#a12a2a')
            axes[r,1].axhline(.05,color='k',ls=':');axes[r,1].set_ylim(0,1);axes[r,1].set_ylabel('Permutation p-value');axes[r,1].legend(fontsize=8)
            for ax in axes[r]:
                ax.axvline(0,color='#777777',ls='--',lw=.7);ax.set_title(rep.title()+' ·80-ms span,8-ms step');ax.set_xlabel('Window center relative to fixation onset (ms)')
        fig.suptitle('Exploratory reader-level map scan · all windows and raw/clean jointly corrected')
        local=Path(temporary)/'sliding_reader_maps.png';fig.savefig(local,dpi=140);plt.close(fig);store.file(local.name,local)
    store.json('complete.json',dict(status='completed',checksums={name:file_digest(out/name) for name in ['window_results.json','window_results.csv','sliding_reader_maps.png']}))
    print('SCAN COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
