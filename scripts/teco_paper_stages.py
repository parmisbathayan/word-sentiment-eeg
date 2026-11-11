#!/usr/bin/env python3
"""TeCo Task 1 paper-style RAW-FRP adaptation; CPU archive -> GPU inference."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from word_sentiment_eeg.common import atomic_npz,file_digest,write_json
from word_sentiment_eeg.teco import find_raw_files
from word_sentiment_eeg.teco_erp import READERS,load_labels
from word_sentiment_eeg.teco_paper import prepare,verify,montage,TIMES
from word_sentiment_eeg.paper_stage_archive import disk_bytes
from word_sentiment_eeg.paper_figure_stats import mixed_models,scalp_geometry
from word_sentiment_eeg.teco_paper_stats import add_reader,finish

DIFFERENCES='''# TeCo adaptation: scope and departures
- Task 1 only: native Task 2/3 EEG has not been validated for this route.
- All recoverable stored word-associated fixations, including rereadings; not just the first fixation.
- Reading EEG only, including control sentences. Answer EEG is never used.
- Private agent word labels: lexical and word-in-context must agree. No whole-sentence agreement requirement; neutral/uncertain words excluded for this binary paper-style comparison. These are not original English SST or human-consensus labels.
- No complete blink/saccade/keypress timeline in validated native pickle schema. No blink-predecessor exclusion, full neighbor-timing matching, eye-velocity control, or regression-cleaned Figure 3 is claimed. Unavailable panels say unavailable.
- Word-context agreement handles some polarity reversals but is NOT the paper's explicit preceding-negation exclusion. No claim of equivalent negation control.
- Exact word-length / target-fixation-duration-tertile balancing replaces unavailable previous/next fixation timing matching. Frequency is diagnosed, not balanced.
- Native 126-channel, 1200-Hz sentence reading EEG; average reference, fourth-order Butterworth 1–50 Hz, polyphase 125 Hz. Native amplitude units assumed microvolts, not independently calibrated.
- Epoch −600…1000 ms; no baseline subtraction. Sentence edges, rather than continuous block edges, limit epochs; bandpass on short sentences can introduce edge effects.
- ≥100-ms segments, corroborated against native fixation order/duration within 5 ms; finite waveform matching across all channels. ±90 assumed microvolt rejection. Epoch quality rejection occurs before matching here.
- Mixed models: same Gaussian ML sufficient-statistics implementation as ZuCo; correlated reader intercept/sentiment slope, asymptotic Wald tests. Repeated trials within readers do not add a word/sentence random effect.
- TCT: 5000 independent channel permutations per ranked within-reader trial pair; ≥6 consecutive significant 125-Hz samples. Cluster correction: 1000 within-pair sign flips, max absolute spatial-temporal mass, at least five supra-threshold neighbors. Trials are not independent lexical items; inference is exploratory.
- 224–304-ms scalp maps are a predefined paper reference window, not an effect detected in TeCo. No published significance is copied.
- Individual accepted matched positive/negative trials retained, float32 all channels/time points. This archive does not contain neutral or rejected events' waveforms.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','analyze'],required=True);p.add_argument('--labels',type=Path)
    p.add_argument('--subjects',nargs='+');p.add_argument('--free-drive-bytes',type=int,required=True)
    p.add_argument('--budget-bytes',type=int,default=2_000_000_000);p.add_argument('--cpu',action='store_true')
    p.add_argument('--tct-permutations',type=int,default=5000);p.add_argument('--cluster-permutations',type=int,default=1000)
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud/Colab only; no local EEG loading')
    if not 0<args.budget_bytes<=2_000_000_000 or min(args.tct_permutations,args.cluster_permutations)<1:raise ValueError('Invalid budget/permutation count')
    root=args.thesis_root/'Results/word_sentiment_eeg/teco_paper_raw_staged';root.mkdir(parents=True,exist_ok=True)
    subjects=list(args.subjects or READERS)
    if len(set(subjects))!=len(subjects) or not set(subjects)<=set(READERS):raise ValueError('Unknown/duplicate reader')
    labelpath=args.labels
    if labelpath is None:
        base=args.thesis_root/'gpt uploaded files/word_sentiment_eeg/teco'
        labelpath=next((base/name for name in ['word_labels_three_level.csv','word_labels.csv'] if (base/name).is_file()),None)
    if labelpath is None or not labelpath.is_file():raise FileNotFoundError('Private labels missing under Thesis/gpt uploaded files/word_sentiment_eeg/teco; pass --labels for an existing private CSV')
    _,lookup=load_labels(labelpath)
    names=['scripts/teco_paper_stages.py','scripts/render_teco_paper.py','src/word_sentiment_eeg/teco_paper.py',
        'src/word_sentiment_eeg/teco_paper_stats.py','src/word_sentiment_eeg/teco.py','src/word_sentiment_eeg/teco_erp.py',
        'src/word_sentiment_eeg/erp.py','src/word_sentiment_eeg/common.py','src/word_sentiment_eeg/paper_figure_stats.py','assets/teco_montage.txt']
    protocol=dict(version='teco_paper_raw_v1',subjects=subjects,labels_sha256=file_digest(labelpath),budget_bytes=args.budget_bytes,
        tct=args.tct_permutations,cluster=args.cluster_permutations,code={n:file_digest(ROOT/n) for n in names},method=DIFFERENCES)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed; refusing to mix existing trials/results')
    nullbytes=(args.tct_permutations+args.cluster_permutations)*126*201*4
    reserve=nullbytes+150_000_000;prepbudget=args.budget_bytes-reserve
    if prepbudget<100_000_000:raise ValueError('Insufficient trial storage after inference reserve')
    required=max(0,args.budget_bytes-disk_bytes(root))
    if (root/'manifest.json').exists():required=0
    if args.free_drive_bytes<required:raise OSError(f'Need {required/1e9:.2f} GB additional Drive capacity; 2 GB total cap including temporary nulls. No EEG read.')
    write_json(root/'protocol.json',protocol);(root/'METHOD_DIFFERENCES.md').write_text(DIFFERENCES)
    if (root/'manifest.json').exists():
        state=json.loads((root/'manifest.json').read_text())
        for name,sha in state['checksums'].items():
            if file_digest(root/name)!=sha:raise ValueError('Changed final result: '+name)
        for subject in subjects:
            verify(root/'readers'/subject)
            if file_digest(root/'readers'/subject/'prepared.json')!=state['readers'][subject]:raise ValueError('Final reader identity changed')
        print('KEEP COMPLETED TECO: archive and figures verified; no EEG loading or inference repeated',flush=True);return
    if args.phase=='prepare':
        sources,alternates=find_raw_files(args.thesis_root/'Data')
        missing=[s for s in subjects if s not in sources]
        if missing:raise FileNotFoundError('Missing native task1.pickle readers: '+str(missing))
        for i,subject in enumerate(subjects,1):
            folder=root/'readers'/subject
            if (folder/'prepared.json').exists():verify(folder);print(f'KEEP CPU READER {i}/{len(subjects)} {subject}',flush=True);continue
            print(f'CPU PREPARE READER {i}/{len(subjects)} {subject}: {sources[subject]}',flush=True)
            with sources[subject].open('rb') as stream:data=pickle.load(stream)
            prepare(data,subject,lookup,folder,root,prepbudget,progress=lambda a,b:print(f'SENTENCES {subject}: {a}/{b}',flush=True) if a%10==0 or a==b else None,source_info=dict(path=str(sources[subject]),sha256=file_digest(sources[subject])))
            del data
            state=json.loads((folder/'prepared.json').read_text())
            print(f'CPU PREPARED {subject}: {state["counts"]}; saved individual fixation trials verified',flush=True)
        write_json(root/'cpu_preparation_manifest.json',dict(status='completed',protocol_sha256=file_digest(root/'protocol.json'),
            readers={s:file_digest(root/'readers'/s/'prepared.json') for s in subjects}))
        print('CPU PREPARATION COMPLETE — switch to GPU analysis notebook',flush=True);return
    state=json.loads((root/'cpu_preparation_manifest.json').read_text())
    if state['protocol_sha256']!=file_digest(root/'protocol.json') or state['status']!='completed':raise ValueError('CPU handoff changed/incomplete')
    if disk_bytes(root)-disk_bytes(root/'null_work')+reserve>args.budget_bytes:raise OSError('Insufficient GPU reserve; trials retained')
    stats=[];audits=[]
    for i,subject in enumerate(subjects,1):
        folder=root/'readers'/subject;verify(folder)
        if file_digest(folder/'prepared.json')!=state['readers'][subject]:raise ValueError('CPU reader identity changed')
        print(f'GPU READER {i}/{len(subjects)} {subject}: saved trials only',flush=True)
        with np.load(folder/'prepared_trials.npz',allow_pickle=False) as z:
            x=z['signal'];diff=x[z['positive_pair_index']]-x[z['negative_pair_index']]
        add_reader(diff,root/'null_work',subject,args.tct_permutations,args.cluster_permutations,use_gpu=not args.cpu)
        with np.load(folder/'statistics.npz',allow_pickle=False) as z:stats.append({k:z[k].copy() for k in z.files})
        audits.append(pd.read_csv(folder/'fixation_audit.csv.gz'))
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    counts=np.stack([s['counts'] for s in stats]);sums=np.stack([s['sums'] for s in stats]);squares=np.stack([s['squares'] for s in stats])
    model=mixed_models(counts,sums,squares,checkpoint=analysis/'mixed_model_checkpoint.npz',progress=lambda a,b:print(f'MIXED MODELS {a}/{b}',flush=True))
    xy,adj=scalp_geometry(montage());pairs=sum(int(s['pairs']) for s in stats)
    perm,clusters=finish(root/'null_work',sum(s['pair_sum'] for s in stats),sum(s['pair_squares'] for s in stats),pairs,adj,subjects,args.tct_permutations,args.cluster_permutations,TIMES)
    atomic_npz(analysis/'group_results.npz',counts=counts,sums=sums,squares=squares,xy=xy,**model,times_ms=TIMES)
    atomic_npz(analysis/'permutation_results.npz',**perm);write_json(analysis/'clusters.json',clusters)
    audits=pd.concat(audits,ignore_index=True);audits.selection_status.value_counts().rename_axis('outcome').reset_index(name='fixations').to_csv(analysis/'exclusion_counts.csv',index=False)
    pd.DataFrame([dict(reader=s,negative=int(n[0]),positive=int(n[1]),total=int(n.sum())) for s,n in zip(subjects,counts)]).to_csv(analysis/'accepted_counts.csv',index=False)
    from render_teco_paper import render
    render(root,subjects,stats,audits)
    checks={str(p.relative_to(root)):file_digest(p) for d in ['analysis','figures'] for p in (root/d).glob('*') if p.is_file()}
    write_json(root/'manifest.json',dict(status='completed',readers=state['readers'],checksums=checks,total_trials=int(counts.sum()),
        pairs=pairs,significant_clusters=sum(c['p']<.05 for c in clusters),minimum_cluster_p=min((c['p'] for c in clusters),default=None),
        nonconverged_features=int((model['converged']!=1).sum()),scope='raw FRP adaptation; full nuisance correction unavailable'))
    shutil.rmtree(root/'null_work',ignore_errors=True)
    print(f'TECO ANALYSIS COMPLETE: {root}',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
