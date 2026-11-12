#!/usr/bin/env python3
"""Analyze saved ZuCo/TeCo paper trials with compact, resumable Drive checkpoints.

CPU scientific protocol/trials are read-only. No source download or extraction.
Null maps and atomic model rewrites stay on local runtime disk/in memory. Drive
receives only immutable batch summaries, small ML frames and final outputs.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.common import atomic_npz,file_digest,digest,write_json
from word_sentiment_eeg.paper_stage_archive import disk_bytes,verify_prepared
from word_sentiment_eeg.paper_figure_stats import mixed_models,scalp_geometry
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore,permutations,model_checkpoints


def read(path):return json.loads(Path(path).read_text())


def validate_handoff(out,dataset):
    protocol=read(out/'protocol.json');state=read(out/'cpu_preparation_manifest.json')
    if state.get('status')!='completed' or state['protocol_sha256']!=file_digest(out/'protocol.json'):raise ValueError('CPU preparation incomplete or protocol changed')
    for name,sha in protocol['code'].items():
        if file_digest(ROOT/name)!=sha:raise ValueError('Saved CPU scientific implementation differs: '+name)
    for name,v in protocol.get('software',{}).items():
        if version(name)!=v:raise ValueError(f'CPU software {name}={v}, current={version(name)}; do not mix environments')
    subjects=state['subjects'] if dataset=='zuco' else protocol['subjects']
    hashes=state['reader_checkpoint_sha256'] if dataset=='zuco' else state['readers']
    if set(subjects)!=set(hashes):raise ValueError('CPU reader roster mismatch')
    for s in subjects:
        if file_digest(out/'readers'/s/'prepared.json')!=hashes[s]:raise ValueError('CPU reader identity changed: '+s)
    return protocol,subjects,hashes


def verified_complete(out,dataset):
    if not (out/'manifest.json').is_file():return False
    done=read(out/'manifest.json')
    if done.get('status')!='completed':raise ValueError('Invalid final manifest')
    if dataset=='zuco':
        checks=dict(done['figure_checksums']);analysis=read(out/'analysis/analysis_manifest.json')
        checks['analysis/analysis_manifest.json']=done['analysis_manifest_sha256']
        checks.update({'analysis/'+k:v for k,v in analysis['checksums'].items()})
    else:checks=done['checksums']
    for name,sha in checks.items():
        if file_digest(out/name)!=sha:raise ValueError('Completed result changed: '+name)
    print('KEEP COMPLETED RESULTS: final statistics and figures verified; no GPU work repeated',flush=True);return True


def stage_readers(out,work,subjects,hashes,dataset):
    stats=[];diffs=[];marks=[]
    for i,s in enumerate(subjects,1):
        source=out/'readers'/s;target=work/'readers'/s;target.mkdir(parents=True,exist_ok=True)
        state=read(source/'prepared.json')
        if state.get('status')!='prepared':raise ValueError('Reader is not prepared: '+s)
        print(f'LOAD SAVED READER {i}/{len(subjects)} {s}: CPU trials only; no source EEG/cleaning',flush=True)
        for name,sha in dict(state['checksums'],**{'prepared.json':hashes[s]}).items():
            if Path(name).name!=name:raise ValueError('Unexpected reader checkpoint path')
            dest=target/name
            if not dest.is_file() or file_digest(dest)!=sha:
                if not (source/name).is_file():raise ValueError('Missing CPU checkpoint: '+s+'/'+name)
                shutil.copyfile(source/name,dest)
                if file_digest(dest)!=sha:raise ValueError('CPU checkpoint changed/corrupt: '+s+'/'+name)
        with np.load(target/'statistics.npz',allow_pickle=False) as z:st={k:z[k].copy() for k in z.files}
        with np.load(target/'prepared_trials.npz',allow_pickle=False) as z:
            signal=z['signal'];neg=z['negative_pair_index'];pos=z['positive_pair_index'];labels=z['labels']
            if len(neg)!=len(pos) or len(pos)!=int(st['pairs']) or not (labels[neg]==0).all() or not (labels[pos]==1).all():raise ValueError('Saved trial pairing/labels changed')
            d=signal[:,pos]-signal[:,neg] if dataset=='zuco' else (signal[pos]-signal[neg])[None]
        # Local arrays are held in RAM; GPU tensors are loaded lazily by engine.
        stats.append(st);diffs.append(d.astype(np.float32,copy=False))
        gpu=dict(status='completed',subject=s,accepted_counts=state.get('accepted_counts',state.get('counts')),pairs=int(st['pairs']),
            checksums=dict(state['checksums'],**{'prepared.json':hashes[s]}),permutation_engine='immutable_batch_summaries_v1')
        write_json(target/'manifest.json',gpu);marks.append(gpu)
    return stats,diffs,marks


def analyze_teco(work,subjects,stats,perm,clusters):
    analysis=work/'analysis';counts=np.stack([s['counts'] for s in stats]);sums=np.stack([s['sums'] for s in stats]);sq=np.stack([s['squares'] for s in stats])
    model=mixed_models(counts,sums,sq,checkpoint=analysis/'mixed_model_checkpoint.npz',progress=lambda a,b:print(f'MIXED MODELS {a}/{b}',flush=True))
    from word_sentiment_eeg.teco_paper import montage,TIMES
    xy,_=scalp_geometry(montage());atomic_npz(analysis/'group_results.npz',counts=counts,sums=sums,squares=sq,xy=xy,**model,times_ms=TIMES)
    audit=pd.concat([pd.read_csv(work/'readers'/s/'fixation_audit.csv.gz') for s in subjects],ignore_index=True)
    audit.selection_status.value_counts().rename_axis('outcome').reset_index(name='fixations').to_csv(analysis/'exclusion_counts.csv',index=False)
    pd.DataFrame([dict(reader=s,negative=int(n[0]),positive=int(n[1]),total=int(n.sum())) for s,n in zip(subjects,counts)]).to_csv(analysis/'accepted_counts.csv',index=False)
    from render_teco_paper import render
    render(work,subjects,stats,audit)
    checks={str(p.relative_to(work)):file_digest(p) for d in ['analysis','figures'] for p in (work/d).glob('*') if p.is_file()}
    write_json(work/'manifest.json',dict(status='completed',checksums=checks,total_trials=int(counts.sum()),pairs=sum(int(s['pairs']) for s in stats),
        significant_clusters=sum(c['p']<.05 for c in clusters),minimum_cluster_p=min((c['p'] for c in clusters),default=None),
        nonconverged_features=int((model['converged']!=1).sum()),scope='raw FRP adaptation; full nuisance correction unavailable'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--dataset',choices=['zuco','teco'],required=True)
    p.add_argument('--run-dir',type=Path);p.add_argument('--work-root',type=Path);p.add_argument('--free-drive-bytes',type=int,required=True)
    p.add_argument('--cpu',action='store_true',help='Engineering validation only; actual analysis uses GPU')
    p.add_argument('--skip-models',action='store_true',help='Engineering test: stop after verified permutation results')
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud/Colab only; no local data loading or installation')
    out=args.run_dir or args.thesis_root/('Results/word_sentiment_eeg/'+('zuco_paper_figures_recovered' if args.dataset=='zuco' else 'teco_paper_raw_staged'))
    protocol,subjects,hashes=validate_handoff(out,args.dataset)
    if verified_complete(out,args.dataset):return
    implementation=['scripts/analyze_paper_small_storage.py','src/word_sentiment_eeg/paper_small_checkpoints.py','scripts/render_zuco_paper_figures.py','scripts/render_teco_paper.py']
    execution=dict(engine='immutable_batch_summaries_v1',dataset=args.dataset,cpu_protocol_sha256=file_digest(out/'protocol.json'),reader_checkpoints=hashes,
        code={n:file_digest(ROOT/n) for n in implementation},seed='unchanged [2026, subject, kind, permutation]',
        storage='Null maps only in runtime RAM; compact all-reader batch summaries and immutable ML checkpoint frames persist on Drive')
    identity=digest(execution);budget=protocol['budget_bytes'];store=ImmutableStore(out,budget)
    if (out/'null_work').exists() and any(p.is_file() for p in (out/'null_work').rglob('*')):
        raise ValueError('Older full-map GPU work exists. Stop here for a storage migration review; CPU trials are retained and nothing was deleted.')
    reserve=200_000_000
    analysis_used=sum(disk_bytes(out/d) for d in ['analysis_state','analysis','figures'])
    needed=max(0,reserve-analysis_used)
    if store.used+needed>budget or args.free_drive_bytes<needed:raise OSError(f'Need {needed/1e6:.1f} MB additional verified free space for compact inference/checkpoints; CPU archives retained')
    # Enforce the incremental analysis allowance as well as the original CPU cap.
    store.budget=min(budget,store.used+needed)
    store.json('analysis_state/execution.json',execution)
    workbase=args.work_root or Path(tempfile.gettempdir())/'word_eeg_paper_local'
    work=workbase/identity
    if '/content/drive' in str(work.resolve()) or out.resolve()==work.resolve() or out.resolve() in work.resolve().parents:raise ValueError('Mutable analysis workspace must be on local runtime disk, not Drive')
    work.mkdir(parents=True,exist_ok=True)
    source_size=sum((out/'readers'/s/name).stat().st_size for s in subjects for name in read(out/'readers'/s/'prepared.json')['checksums'])
    if shutil.disk_usage(work).free<source_size*1.3:raise OSError('Insufficient local runtime disk space; Drive trials remain unchanged')
    stats,diffs,marks=stage_readers(out,work,subjects,hashes,args.dataset)
    analysis=work/'analysis';analysis.mkdir(exist_ok=True)
    if args.dataset=='zuco':
        xy,adj=scalp_geometry(pd.read_csv(ROOT/'assets/zuco_chanlocs.csv'));tct=protocol['tct_permutations'];cluster=protocol['cluster_permutations']
        total=np.sum([s['pair_sum'] for s in stats],axis=0);squares=np.sum([s['pair_squares'] for s in stats],axis=0)
    else:
        from word_sentiment_eeg.teco_paper import montage
        xy,adj=scalp_geometry(montage());tct=protocol['tct'];cluster=protocol['cluster']
        total=np.sum([s['pair_sum'] for s in stats],axis=0)[None];squares=np.sum([s['pair_squares'] for s in stats],axis=0)[None]
    n=sum(int(s['pairs']) for s in stats);times=stats[0]['times_ms']
    perm,clusters=permutations(diffs,subjects,total,squares,n,adj,times,store,identity,tct,cluster,use_gpu=not args.cpu)
    if args.dataset=='teco':
        perm={k:v[0] for k,v in perm.items()};clusters=[{k:v for k,v in c.items() if k!='representation'} for c in clusters]
    atomic_npz(analysis/'permutation_results.npz',**perm);write_json(analysis/'clusters.json',clusters)
    # Publish exact completed permutation results before any reader completion mark.
    for name in ['permutation_results.npz','clusters.json']:store.file('analysis/'+name,analysis/name)
    for s in subjects:store.file('readers/'+s+'/manifest.json',work/'readers'/s/'manifest.json')
    store.json('preparation_manifest.json',dict(status='completed',subjects=subjects))
    write_json(work/'preparation_manifest.json',dict(status='completed',subjects=subjects))
    if args.skip_models:print('ENGINEERING HANDOFF COMPLETE: compact permutations verified; models/figures not run',flush=True);return
    with model_checkpoints(store,analysis):
        if args.dataset=='zuco':
            from render_zuco_paper_figures import analyze_and_render
            analyze_and_render(work)
        else:analyze_teco(work,subjects,stats,perm,clusters)
    done=read(work/'manifest.json');figure_hashes={}
    for d in ['analysis','figures']:
        for path in sorted((work/d).glob('*')):
            if not path.is_file():continue
            relative=str(path.relative_to(work));sha=store.file(relative,path,reuse_committed=d=='figures')
            if d=='figures':figure_hashes[relative]=sha
    if args.dataset=='zuco':done['figure_checksums']=figure_hashes;done['analysis_manifest_sha256']=file_digest(out/'analysis/analysis_manifest.json')
    else:
        done['checksums'].update(figure_hashes);done['readers']=hashes
    done['storage_engine']='immutable_batch_summaries_v1'
    store.json('manifest.json',done)
    print(f'ANALYSIS COMPLETE: {out}. CPU archives preserved; no null-map generations written to or deleted from Drive.',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
