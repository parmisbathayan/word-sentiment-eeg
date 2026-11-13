#!/usr/bin/env python3
"""Paper-style statistics on old agent labels intersecting saved accepted FRPs.

Not a reconstruction of every agent-labelled word: the source archive was
already selected/matched/rejected with SST labels. No source EEG or new cleaning.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import digest,file_digest,atomic_npz,write_json
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore,permutations,model_checkpoints
from word_sentiment_eeg.paper_figure_stats import mixed_models,scalp_geometry
from word_sentiment_eeg.paper_trials import matched_trials
from zuco_recovered_followups import read,scalp_test,holm,run_classifiers

POLICY=dict(labels='annotations/word_labels.csv; original v2 agent review',
    eligibility='lexical == contextual in positive/negative; no alignment warning; agent_reviewed_not_human_gold',
    context='word meaning in context, not full-sentence sentiment; no all-three agreement or content-word restriction',
    fixations='all archived fixations of eligible word occurrences, including rereading; not first fixation only',
    parent='SST-selected, matched, rejected and regression-cleaned archive; intersection only',
    matching='rematch within reader/exact length/previous and next gap tertiles, seed2026; rank retained classes by length/duration/event ID',
    negation='inherits parent five-token preceding-three-word exclusion; contractions not added',
    budget=200_000_000,tct=5000,cluster=1000)

def norm(w):return unicodedata.normalize('NFKC',str(w)).casefold().strip()

def labels_table(path):
    labels=pd.read_csv(path,keep_default_na=False)
    required={'sentence_key','word_index','word','tasks','lexical_class','contextual_class','alignment_warning','annotation_status'}
    if not required.issubset(labels):raise ValueError('Missing original agent annotation fields')
    labels=labels[labels.tasks.str.split('[;|]',regex=True).map(lambda x:'SR' in x)].copy()
    labels['word_index']=pd.to_numeric(labels.word_index).astype(int)
    keys=['sentence_key','word_index']
    if labels.duplicated(keys).any():raise ValueError('Duplicate source occurrence annotation')
    labels['agent_eligible']=(labels.lexical_class.isin(['positive','negative'])&labels.lexical_class.eq(labels.contextual_class)&
        labels.alignment_warning.eq('')&labels.annotation_status.eq('agent_reviewed_not_human_gold'))
    return labels

def select(meta,labels):
    """Exact occurrence join; spelling checked, no nearest-word/index fallback."""
    meta=meta.copy();meta['word_index']=pd.to_numeric(meta.word_index).astype(int)
    meta['parent_sentiment']=meta.sentiment;meta['parent_selection_status']=meta.selection_status
    right=labels[['sentence_key','word_index','word','lexical_class','contextual_class','agent_eligible','annotation_flags']].rename(columns={'word':'annotation_word'})
    joined=meta.merge(right,on=['sentence_key','word_index'],how='left',validate='many_to_one',sort=False)
    if joined.event_id.tolist()!=meta.event_id.tolist():raise ValueError('Join reordered trial rows')
    joined['join_status']=np.where(joined.annotation_word.isna(),'missing_annotation',
        np.where(joined.word.map(norm).eq(joined.annotation_word.map(norm)),'matched','word_text_mismatch'))
    joined['subset_status']=np.where(joined.join_status.ne('matched'),joined.join_status,
        np.where(joined.agent_eligible.eq(True),'eligible_before_matching','agent_neutral_disagreement_or_warning'))
    joined['parent_row']=np.arange(len(joined));eligible=joined[joined.subset_status.eq('eligible_before_matching')].copy()
    eligible['sentiment']=eligible.lexical_class
    # Keep the rematching operational definition identical to the paper route.
    if not eligible.empty:
        for col in ['word_length','previous_gap_ms','next_gap_ms','duration_ms']:
            eligible[col]=pd.to_numeric(eligible[col],errors='coerce')
        good=eligible.word_length.between(1,20)&np.isfinite(eligible[['previous_gap_ms','next_gap_ms','duration_ms']]).all(axis=1)
        joined.loc[eligible.index[~good],'subset_status']='unusable_matching_covariate';eligible=eligible[good].copy()
    if not eligible.empty:
        for col in ['previous_gap_ms','next_gap_ms']:
            edges=np.unique(np.quantile(eligible[col],[1/3,2/3]));eligible[col+'_stratum']=np.searchsorted(edges,eligible[col],side='right')
        ix,_,_=matched_trials(eligible,['subject','word_length','previous_gap_ms_stratum','next_gap_ms_stratum'])
        retained=eligible.iloc[ix].copy()
        joined.loc[eligible.index,'subset_status']='excluded_by_subset_matching'
        joined.loc[retained.index,'subset_status']='retained'
    else:retained=eligible
    return joined,retained

def inventory(run,labels,subjects,work,store):
    selections=[];rows=[];coverage=[];ledger=[]
    eligible=labels[labels.agent_eligible]
    source_counts=eligible.lexical_class.value_counts().to_dict()
    for i,s in enumerate(subjects,1):
        folder=run/'readers'/s;state=read(folder/'prepared.json');p=folder/'accepted_trial_metadata.csv.gz'
        if file_digest(p)!=state['checksums'][p.name]:raise ValueError('Changed parent metadata: '+s)
        meta=pd.read_csv(p,keep_default_na=False);meta['subject']=s
        joined,selected=select(meta,labels)
        counts=selected.sentiment.value_counts();usable=min(counts.get('negative',0),counts.get('positive',0))>=2
        joined.loc[joined.subset_status.eq('retained')&(not usable),'subset_status']='reader_insufficient_classes'
        selections.append(dict(subject=s,selected=selected if usable else selected.iloc[:0],state=state))
        rows.append(dict(reader=s,parent_accepted=len(meta),annotation_joined=int(joined.join_status.eq('matched').sum()),
            available_agent_eligible=int((joined.join_status.eq('matched')&joined.agent_eligible.eq(True)).sum()),
            matched_negative=int(counts.get('negative',0)),matched_positive=int(counts.get('positive',0)),
            subset_negative=int(counts.get('negative',0)) if usable else 0,
            subset_positive=int(counts.get('positive',0)) if usable else 0,included=bool(usable)))
        present=set(zip(joined.loc[joined.join_status.eq('matched'),'sentence_key'],joined.loc[joined.join_status.eq('matched'),'word_index']))
        for cls in ['negative','positive']:
            pool=eligible[eligible.lexical_class.eq(cls)];available=sum((k,w) in present for k,w in zip(pool.sentence_key,pool.word_index))
            coverage.append(dict(reader=s,agent_class=cls,source_word_occurrences=len(pool),available_in_parent=available,
                absent_from_parent=len(pool)-available,note='Absent may reflect SST selection, matching, missing fixation/source or EEG rejection; cause not inferred'))
        ledger.append(joined)
        print(f'INVENTORY {i}/{len(subjects)} {s}: agent subset NEG={counts.get("negative",0)}, POS={counts.get("positive",0)}; included={usable}',flush=True)
    report=dict(policy=POLICY,source_word_occurrences=source_counts,readers=rows,coverage=coverage,
        included_readers=[v['subject'] for v in selections if not v['selected'].empty],
        warning='Only the intersection with the preselected SST archive is available. Missing agent-labelled words have no saved waveform in this archive.')
    store.json('selection/inventory.json',report)
    local=work/'selection.csv.gz';pd.concat(ledger,ignore_index=True).to_csv(local,index=False,compression={'method':'gzip','mtime':0})
    store.file('selection/trial_selection.csv.gz',local)
    return [v for v in selections if not v['selected'].empty],report

def stage(run,selections,work):
    n=sum(len(v['selected']) for v in selections);shape=(2,n,105,201)
    if shutil.disk_usage(work).free<int(np.prod(shape))*4+500_000_000:raise OSError('Insufficient runtime disk; no Drive EEG copy made')
    x=np.memmap(work/'subset_signals.dat',dtype='float32',mode='w+',shape=shape)
    stats=[];diffs=[];metadata=[];pairs=[];offset=0
    times=np.arange(-75,126)*8
    for i,v in enumerate(selections,1):
        s=v['subject'];meta=v['selected'].copy().reset_index(drop=True);rows=meta.parent_row.to_numpy(int)
        print(f'LOAD SUBSET {i}/{len(selections)} {s}: {len(rows)} fixations from saved parent archive',flush=True)
        local=work/'parent.npz';shutil.copyfile(run/'readers'/s/'prepared_trials.npz',local)
        if file_digest(local)!=v['state']['checksums']['prepared_trials.npz']:raise ValueError('Corrupt parent waveform archive')
        with np.load(local,allow_pickle=False) as z:
            raw=z['signal']
            if raw.shape[0]!=2 or raw.shape[2:]!=(105,201) or not np.array_equal(z['times_ms'],times):raise ValueError('Unexpected parent waveform schema')
            if z['event_ids'][rows].astype(str).tolist()!=meta.event_id.astype(str).tolist():raise ValueError('Selected event/waveform row mismatch')
            if not np.array_equal(z['labels'][rows],meta.parent_sentiment.map({'negative':0,'positive':1}).to_numpy()):raise ValueError('Parent labels changed')
            signal=raw[:,rows].copy()
        del raw;local.unlink()
        if not np.isfinite(signal).all():raise ValueError('Nonfinite stored trial')
        counts=np.array([meta.sentiment.eq(c).sum() for c in ['negative','positive']])
        value=np.concatenate([signal,signal.std(axis=2)[:,:,None,:]],axis=2).astype(np.float64)
        sums=np.stack([value[:,meta.sentiment.eq(c)].sum(axis=1) for c in ['negative','positive']],axis=1)
        squares=np.stack([(value[:,meta.sentiment.eq(c)]**2).sum(axis=1) for c in ['negative','positive']],axis=1)
        orders=[meta[meta.sentiment.eq(c)].sort_values(['word_length','duration_ms','event_id'],kind='stable').index.to_numpy() for c in ['negative','positive']]
        k=min(map(len,orders));neg,pos=[a[:k] for a in orders];d=(signal[:,pos]-signal[:,neg]).astype(np.float32)
        map_sums=[];map_squares=[]
        for r,stop in enumerate([280,304]):
            m=signal[r][:,:,(times>=224)&(times<=stop)].mean(axis=2).astype(np.float64)
            map_sums.append(np.stack([m[meta.sentiment.eq(c)].sum(axis=0) for c in ['negative','positive']]))
            map_squares.append(np.stack([(m[meta.sentiment.eq(c)]**2).sum(axis=0) for c in ['negative','positive']]))
        stats.append(dict(counts=counts,sums=sums,squares=squares,map_sums=np.stack(map_sums),map_squares=np.stack(map_squares),
            pair_sum=d.sum(axis=1,dtype=np.float64),pair_squares=(d.astype(np.float64)**2).sum(axis=1),pairs=k,times_ms=times))
        diffs.append(d);pairs.append((s,neg+offset,pos+offset));x[:,offset:offset+len(meta)]=signal
        metadata.append(meta);offset+=len(meta);del signal,value
    x.flush();meta=pd.concat(metadata,ignore_index=True)
    from zuco_recovered_followups import clean_word
    meta['word_key']=meta.word.map(clean_word)
    return x,meta,stats,diffs,pairs

def analyze(stats,diffs,subjects,store,work,tct,cluster,use_gpu=True):
    analysis=work/'analysis';analysis.mkdir(exist_ok=True)
    xy,adj=scalp_geometry(pd.read_csv(ROOT/'assets/zuco_chanlocs.csv'));times=stats[0]['times_ms']
    n=sum(s['pairs'] for s in stats);total=np.sum([s['pair_sum'] for s in stats],axis=0);sq=np.sum([s['pair_squares'] for s in stats],axis=0)
    p,clusters=permutations(diffs,subjects,total,sq,n,adj,times,store,digest(read(store.root/'protocol.json')),tct,cluster,use_gpu=use_gpu)
    atomic_npz(analysis/'permutation_results.npz',**p);write_json(analysis/'clusters.json',clusters)
    for name in ['permutation_results.npz','clusters.json']:store.file('analysis/'+name,analysis/name)
    counts=np.stack([s['counts'] for s in stats]);sums=np.stack([s['sums'] for s in stats]);squares=np.stack([s['squares'] for s in stats]);models=[];maps=[]
    with model_checkpoints(store,analysis):
        for r,name in enumerate(['raw','cleaned']):
            models.append(mixed_models(counts,sums[:,r],squares[:,r],checkpoint=analysis/f'{name}_ml_checkpoint.npz',progress=lambda a,b:print(f'MIXED MODEL {name} {a}/{b}',flush=True)))
            maps.append(mixed_models(counts,np.stack([s['map_sums'][r] for s in stats]),np.stack([s['map_squares'][r] for s in stats]),checkpoint=analysis/f'{name}_map_ml.npz'))
    group=dict(counts=counts,sums=sums,squares=squares,xy=xy,times_ms=times,
        p=np.stack([v['p'] for v in models]),effect=np.stack([v['effect'] for v in models]),map_p=np.stack([v['p'] for v in maps]))
    atomic_npz(analysis/'group_results.npz',**group)
    summary=dict(status='completed',included_readers=subjects,accepted_negative=int(counts[:,0].sum()),accepted_positive=int(counts[:,1].sum()),pairs=n,
        minimum_cluster_p={name:min([c['p'] for c in clusters if c['representation']==r],default=1) for r,name in enumerate(['raw','cleaned'])},
        significant_clusters={name:sum(c['p']<.05 and c['representation']==r for c in clusters) for r,name in enumerate(['raw','cleaned'])},
        nonconverged_features=[int((m['converged']!=1).sum()) for m in models],tct_permutations=tct,cluster_permutations=cluster,
        scope='Agent lexical/context agreement within previously SST-selected archive; shared full-reader cleaning; exploratory comparison')
    store.json('analysis/summary.json',summary);store.file('analysis/group_results.npz',analysis/'group_results.npz')
    import render_zuco_paper_figures as renderer
    save=renderer.save
    def tagged(fig,out,name,caption):
        if fig._suptitle:fig._suptitle.set_text(fig._suptitle.get_text()+' · agent-label intersection')
        save(fig,out,name,caption+'\nOld agent lexical/context agreement; SST-preselected intersection, not complete agent cohort. Prior full-reader cleaning retained.')
    renderer.save=tagged
    try:
        for r in range(2):renderer.figure_frp(work,group,p,r)
    finally:renderer.save=save
    for image in (work/'figures').glob('*'):store.file('figures/'+image.name,image)
    return summary

def selection_controls(meta,report,store,work):
    from zuco_recovered_followups import balance
    store.json('analysis/covariate_balance.json',balance(meta))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');table=pd.DataFrame(report['readers'])
    a=axes[0];table.set_index('reader')[['subset_negative','subset_positive']].plot.bar(ax=a,color=['#333333','#cb3335'])
    a.set_title('Agent agreement: retained subset per reader');a.set_ylabel('Fixation trials after subset rematching')
    a=axes[1];a.axis('off');a.text(.02,.95,
        'Selection and interpretation\n\n'
        f'Parent accepted SST trials: {table.parent_accepted.sum():,}\n'
        f'Available agent-agreement trials before rematching: {table.available_agent_eligible.sum():,}\n'
        f'New subset after matching: {len(meta):,}\n\n'
        'Source word occurrences absent from parent are listed\nin selection/inventory.json. No waveform is inferred.\n\n'
        'All inherited EEG quality and original negation exclusions\nremain. All stored eligible refixations are candidates.\n\n'
        'Subset-specific continuous gaze/eye-velocity controls are\nunavailable from this saved trial archive. Parent controls\nare not reused as if they describe the subset.',va='top',fontsize=10)
    (work/'figures').mkdir(exist_ok=True);p=work/'figures/subset_selection_controls.png';fig.savefig(p,dpi=140);plt.close(fig)
    store.file('figures/'+p.name,p)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--run-dir',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--work-root',type=Path);p.add_argument('--cpu',action='store_true',help='Cloud engineering tests only')
    p.add_argument('--skip-decoding',action='store_true');p.add_argument('--tct',type=int,default=5000);p.add_argument('--cluster',type=int,default=1000)
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; no local dataset loading or dependencies')
    if min(args.tct,args.cluster)<1:raise ValueError('Positive permutation counts required')
    run=args.run_dir or args.thesis_root/'Results/word_sentiment_eeg/zuco_paper_figures_recovered'
    out=args.output or args.thesis_root/'Results/word_sentiment_eeg/zuco_agent_paper_subset'
    if run.resolve()==out.resolve() or run.resolve() in out.resolve().parents:raise ValueError('Separate output folder required')
    cpu=read(run/'cpu_preparation_manifest.json')
    if cpu['status']!='completed' or cpu['protocol_sha256']!=file_digest(run/'protocol.json') or read(run/'manifest.json')['status']!='completed':raise ValueError('Parent reproduction incomplete/changed')
    for s in cpu['subjects']:
        if file_digest(run/'readers'/s/'prepared.json')!=cpu['reader_checkpoint_sha256'][s]:raise ValueError('Parent reader identity changed')
    path=ROOT/'annotations/word_labels.csv';labels=labels_table(path)
    names=['scripts/zuco_agent_paper_subset.py','scripts/zuco_recovered_followups.py','src/word_sentiment_eeg/paper_small_checkpoints.py',
        'src/word_sentiment_eeg/paper_figure_stats.py','src/word_sentiment_eeg/paper_trials.py','scripts/render_zuco_paper_figures.py','annotations/word_labels.csv']
    protocol=dict(policy=POLICY,parent_cpu=file_digest(run/'cpu_preparation_manifest.json'),code={n:file_digest(ROOT/n) for n in names},
        software={n:version(n) for n in ['numpy','scipy','pandas','scikit-learn']},tct=args.tct,cluster=args.cluster,decoding=not args.skip_decoding)
    store=ImmutableStore(out,POLICY['budget']);store.json('protocol.json',protocol)
    marker=out/'complete.json'
    if marker.exists():
        for name,sha in read(marker)['checksums'].items():
            if file_digest(out/name)!=sha:raise ValueError('Completed subset result changed: '+name)
        print('KEEP COMPLETED AGENT SUBSET: saved outputs verified; no trial reload',flush=True);return
    root=args.work_root or Path('/content/zuco-agent-subset-work')
    if 'drive' in root.resolve().parts or args.thesis_root.resolve() in root.resolve().parents:raise ValueError('Work folder must be runtime-local')
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        work=Path(tmp);(work/'figures').mkdir()
        selected,report=inventory(run,labels,cpu['subjects'],work,store)
        if len(selected)<3:raise ValueError('Insufficient readers with >=2 trials per class; inventory retained, no inference run')
        x,meta,stats,diffs,pairs=stage(run,selected,work);subjects=[v['subject'] for v in selected]
        selection_controls(meta,report,store,work)
        analyze(stats,diffs,subjects,store,work,args.tct,args.cluster,use_gpu=not args.cpu)
        tests=[]
        for r,name in enumerate(['raw','cleaned']):
            for stop in [280,304]:
                maps=np.stack([x[r,pos][:,:,(stats[0]['times_ms']>=224)&(stats[0]['times_ms']<=stop)].mean(axis=(0,2))-x[r,neg][:,:,(stats[0]['times_ms']>=224)&(stats[0]['times_ms']<=stop)].mean(axis=(0,2)) for s,neg,pos in pairs])
                tests.append(dict(representation=name,start_ms=224,end_ms=stop,**scalp_test(maps)))
        for t,adjusted in zip(tests,holm([t['p_exact'] for t in tests])):t['p_holm_four_tests']=float(adjusted)
        store.json('analysis/window_tests.json',tests)
        if not args.skip_decoding:run_classifiers(x,meta,store,work)
        del x
    checks={str(f.relative_to(out)):file_digest(f) for d in ['selection','analysis','figures','classification'] for f in (out/d).rglob('*') if f.is_file()}
    store.json('complete.json',dict(status='completed',checksums=checks,protocol=digest(protocol)))
    print('AGENT SUBSET COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
