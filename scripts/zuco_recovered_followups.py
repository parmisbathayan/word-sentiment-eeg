#!/usr/bin/env python3
"""Fixed, exploratory follow-ups on the completed recovered ZuCo SR archive.

Cloud only. Source EEG and existing results are read-only. Features and training
data live on runtime disk. Drive receives immutable, bounded small results.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from word_sentiment_eeg.common import digest,file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

SEED=20261113
SPEC=dict(version=1,seed=SEED,folds=3,representations=['raw','cleaned'],
          windows=[[224,280],[224,304]],feature_bins_ms=40,
          classification_windows=['published_224_304','post_0_800','pre_minus200_0'],
          splits=['sentence','word','reader'],averaging=[1,20,40],
          selection='all saved accepted binary trials, no relabeling or additional rejection',
          interpretation='exploratory; full-reader preprocessing is transductive',
          drive_cap_bytes=50_000_000)

def read(p):return json.loads(Path(p).read_text())

def clean_word(w):
    w=unicodedata.normalize('NFKC',str(w)).casefold().strip()
    while w and unicodedata.category(w[0]).startswith('P'):w=w[1:]
    while w and unicodedata.category(w[-1]).startswith('P'):w=w[:-1]
    return w

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);v=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    out=np.empty_like(p);out[order]=np.minimum(v,1);return out

def scalp_test(maps):
    """Exact reader sign-flip test of equal-reader mean contrast-map magnitude.

    No electrode selection. Spatial mean removed, no GFP normalization. This is
    not the original trial-paired TCT/TANOVA or a test of specific generators.
    Symmetry/exchangeability of reader contrasts is an assumption.
    """
    x=np.asarray(maps,float);x=x-x.mean(axis=1,keepdims=True)
    if len(x)<3 or len(x)>16 or not np.isfinite(x).all():raise ValueError('Invalid reader contrast maps')
    signs=1-2*((np.arange(2**len(x))[:,None]>>np.arange(len(x)))&1)
    null=np.mean((signs@x/len(x))**2,axis=1)
    observed=float(np.mean(x.mean(axis=0)**2))
    p=float(np.mean(null>=observed-1e-12*max(1,observed)))
    return dict(statistic_rms_uV=float(np.sqrt(observed)),p_exact=p,assignments=len(signs),readers=len(x))

def balance(meta):
    rows=[]
    for col in ['word_length','word_frequency','duration_ms','previous_gap_ms','next_gap_ms']:
        for subject,g in [('ALL_TRIALS',meta),*list(meta.groupby('subject',sort=True))]:
            a=pd.to_numeric(g.loc[g.sentiment.eq('positive'),col],errors='coerce').dropna().to_numpy(float)
            b=pd.to_numeric(g.loc[g.sentiment.eq('negative'),col],errors='coerce').dropna().to_numpy(float)
            if not len(a) or not len(b):continue
            pooled=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2) if min(len(a),len(b))>1 else 0
            rows.append(dict(reader=subject,covariate=col,positive_n=len(a),negative_n=len(b),
                positive_mean=float(a.mean()),negative_mean=float(b.mean()),
                standardized_difference=float((a.mean()-b.mean())/pooled) if pooled>0 else None))
    return rows

def grouped_folds(y,groups,n=3):
    y=np.asarray(y);groups=np.asarray(groups).astype(str)
    if len(set(groups))<n:raise ValueError('Too few independent groups')
    splits=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,groups))
    for train,test in splits:
        if set(groups[train])&set(groups[test]) or len(set(y[train]))<2 or len(set(y[test]))<2:
            raise ValueError('Invalid grouped fold: overlapping groups or missing class')
    return splits

def averaged(x,y,meta,indices,size,seed):
    """Disjoint averages inside partition, reader and class. No crossing folds.

    Test labels form class-known averages as in a pseudotrial experiment. These
    predictions are not deployment/single-event accuracy. Remainders discarded.
    """
    indices=np.asarray(indices)
    if size==1:return x[indices],y[indices],meta.iloc[indices].subject.to_numpy(),indices.tolist()
    rng=np.random.default_rng(seed);values=[];labels=[];readers=[];members=[]
    for subject in sorted(meta.iloc[indices].subject.unique()):
        for c in [0,1]:
            ix=indices[(meta.subject.to_numpy()[indices]==subject)&(y[indices]==c)]
            ix=rng.permutation(ix)
            for start in range(0,len(ix)-size+1,size):
                block=ix[start:start+size];values.append(x[block].mean(axis=0));labels.append(c);readers.append(subject);members.extend(block.tolist())
    shape=(0,)+x.shape[1:]
    return np.asarray(values,dtype=np.float32) if values else np.empty(shape,np.float32),np.asarray(labels,int),np.asarray(readers),members

def metrics(y,pred,score):
    return dict(n=int(len(y)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        auc=float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def decode_case(x,y,meta,mode,size,model,window,rep):
    groups=meta[{'sentence':'sentence_key','word':'word_key','reader':'subject'}[mode]].to_numpy()
    folds=grouped_folds(y,groups);records=[];all_y=[];all_pred=[];all_score=[];reader_rows=[]
    for fi,(train,test) in enumerate(folds):
        a,ya,_,ma=averaged(x,y,meta,train,size,SEED+fi)
        b,yb,readers,mb=averaged(x,y,meta,test,size,SEED+100+fi)
        row=dict(fold=fi+1,train_original=len(train),test_original=len(test),train_instances=len(ya),test_instances=len(yb),
                 train_groups=len(set(groups[train])),test_groups=len(set(groups[test])),
                 train_event_digest=digest(sorted(meta.iloc[ma].event_id.tolist())),test_event_digest=digest(sorted(meta.iloc[mb].event_id.tolist())),
                 sentence_overlap=len(set(meta.iloc[train].sentence_key)&set(meta.iloc[test].sentence_key)),
                 word_overlap=len(set(meta.iloc[train].word_key)&set(meta.iloc[test].word_key)),
                 reader_overlap=len(set(meta.iloc[train].subject)&set(meta.iloc[test].subject)))
        if not len(ya) or not len(yb) or len(set(ya))<2 or len(set(yb))<2:
            records.append(dict(row,status='insufficient_pseudotrials'));continue
        estimator=(make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                      LogisticRegression(C=1,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=SEED))
                   if model=='logistic' else DummyClassifier(strategy='prior'))
        estimator.fit(a,ya);pred=estimator.predict(b);score=estimator.predict_proba(b)[:,1]
        records.append(dict(row,status='completed',**metrics(yb,pred,score)))
        all_y.extend(yb.tolist());all_pred.extend(pred.tolist());all_score.extend(score.tolist())
        for subject in np.unique(readers):
            pick=readers==subject
            if len(set(yb[pick]))==2:reader_rows.append(dict(reader=str(subject),fold=fi+1,**metrics(yb[pick],pred[pick],score[pick])))
    return dict(status='completed' if all(v['status']=='completed' for v in records) else 'incomplete_coverage',
        representation=rep,window=window,split=mode,average_trials=size,model=model,folds=records,
        pooled=metrics(np.asarray(all_y),np.asarray(all_pred),np.asarray(all_score)) if all_y else None,
        reader_fold_scores=reader_rows,interpretation='class-known disjoint pseudotrials' if size>1 else 'individual fixation trials')

def stage(run,work,subjects,store,identity):
    """One saved reader at a time; local-only full trial mmap, never new Drive EEG."""
    all_meta=[];total=0;states={}
    for subject in subjects:
        folder=run/'readers'/subject;state=read(folder/'prepared.json');states[subject]=state
        p=folder/'accepted_trial_metadata.csv.gz'
        if file_digest(p)!=state['checksums'][p.name]:raise ValueError('Metadata checksum changed: '+subject)
        m=pd.read_csv(p,keep_default_na=False);m['subject']=subject
        if m.event_id.duplicated().any() or not m.sentiment.isin(['negative','positive']).all():raise ValueError('Bad trial metadata')
        all_meta.append(m);total+=len(m)
    meta=pd.concat(all_meta,ignore_index=True)
    if meta.event_id.duplicated().any() or meta.sentence_key.eq('').any():raise ValueError('Missing/duplicate trial identity')
    meta['word_key']=meta.word.map(clean_word)
    if meta.word_key.eq('').any():raise ValueError('Missing lexical identity')
    for col in ['word_length','word_frequency','duration_ms','previous_gap_ms','next_gap_ms']:
        if col not in meta:raise ValueError('Missing audit covariate: '+col)
    shape=(2,total,105,201);required=int(np.prod(shape))*4+300_000_000
    if shutil.disk_usage(work).free<required:raise OSError(f'Need {required/1e9:.2f} GB free on runtime disk; no extra EEG on Drive')
    x=np.memmap(work/'signals.dat',mode='w+',dtype='float32',shape=shape);offset=0;paired=[];provenance=[];audit_rows=[]
    for i,(subject,m) in enumerate(zip(subjects,all_meta),1):
        folder=run/'readers'/subject;state=states[subject];p=folder/'prepared_trials.npz'
        print(f'LOAD SAVED TRIALS {i}/{len(subjects)} {subject}: {len(m)} fixations; runtime disk only',flush=True)
        local=work/'reader.npz';shutil.copyfile(p,local)
        if file_digest(local)!=state['checksums'][p.name]:raise ValueError('Trial checksum changed: '+subject)
        with np.load(local,allow_pickle=False) as z:
            s=z['signal'];times=z['times_ms'];labels=z['labels'];neg=z['negative_pair_index'];pos=z['positive_pair_index']
            if s.shape!=(2,len(m),105,201) or not np.isfinite(s).all() or not np.array_equal(times,np.arange(-75,126)*8):raise ValueError('Invalid saved waveform')
            if z['event_ids'].astype(str).tolist()!=m.event_id.astype(str).tolist() or not np.array_equal(labels,m.sentiment.map({'negative':0,'positive':1})):raise ValueError('Trial row/label mismatch')
            if not (labels[neg]==0).all() or not (labels[pos]==1).all() or len(neg)!=len(pos):raise ValueError('Invalid saved ranked pairs')
            x[:,offset:offset+len(m)]=s;paired.append((subject,neg+offset,pos+offset))
        mapping=read(folder/'source_mapping.json')
        # Derive recovered status from the source audit's declared recovery flag,
        # not from large residuals or from a guessed list of source filenames.
        for b in mapping['blocks']:
            provenance.append(dict(reader=subject,block=str(b['block_id']),status=b['status'],
                                   audit=b.get('synchronization',{}),source_keys=list(b)))
        offset+=len(m);del s
        local.unlink() # runtime disk only
        audit=folder/'fixation_audit.csv.gz'
        if file_digest(audit)!=state['checksums'][audit.name]:raise ValueError('Event audit checksum changed')
        ev=pd.read_csv(audit,keep_default_na=False)
        audit_rows.extend(dict(reader=subject,outcome=str(k),events=int(v)) for k,v in ev.selection_status.value_counts().items())
    x.flush();meta['block_id']=meta.block_id.astype(str).str.removesuffix('.0')
    store.json('diagnostics/selection_counts.json',audit_rows)
    return x,meta,paired,provenance

def label_audit(meta,work):
    """Recheck original word dictionary lookups; do not invent sentence joins."""
    from paper_sst_source import acquire_sst
    from word_sentiment_eeg.paper_labels import SST
    (work/'sst').mkdir(exist_ok=True)
    path,source=acquire_sst(work/'sst');sst=SST(path);rows=[]
    for (word,label,method),g in meta.groupby(['word','label_5','original_label_method'],dropna=False,sort=True):
        found,rule=sst.lookup_word(word)
        rows.append(dict(word=str(word),stored_label_5=int(label),stored_method=str(method),
            direct_dictionary_label_5=found['label_5'] if found else None,direct_dictionary_rule=rule,
            matches_direct_dictionary=(found['label_5']==int(label)) if found else None,
            accepted_fixations=len(g),readers=g.subject.nunique(),source_word_occurrences=g[['sentence_key','word_index']].drop_duplicates().shape[0]))
    return dict(source=source,rows=rows,limitations=[
        'Dictionary disagreement is a review flag, not proof the stored SST leaf label is wrong.',
        'Exact source-word-to-SST leaf joins and full sentence labels cannot always be reconstructed from accepted word metadata alone.',
        'No agent relabeling, sentence-label substitution or automatic label change is performed.'])

def diagnostics(x,meta,paired,provenance,store,work):
    times=np.arange(-75,126)*8;y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy();subjects=[p[0] for p in paired]
    tests=[];maps=[];similarity=[]
    for r,name in enumerate(['raw','cleaned']):
        for start,stop in SPEC['windows']:
            win=(times>=start)&(times<=stop)
            reader_maps=np.stack([(x[r,pos][:,:,win].mean(axis=(0,2))-x[r,neg][:,:,win].mean(axis=(0,2))) for _,neg,pos in paired])
            result=scalp_test(reader_maps);tests.append(dict(representation=name,start_ms=start,end_ms=stop,**result))
            maps.append(reader_maps)
            for i,subject in enumerate(subjects):
                other=np.delete(reader_maps,i,axis=0).mean(axis=0);a=reader_maps[i]
                denom=np.linalg.norm(a)*np.linalg.norm(other)
                similarity.append(dict(reader=subject,representation=name,start_ms=start,end_ms=stop,
                    cosine_to_other_readers=float(a@other/denom) if denom else None,
                    rms_contrast_uV=float(np.sqrt(np.mean(a*a)))))
    for t,p in zip(tests,holm([t['p_exact'] for t in tests])):t['p_holm_four_tests']=float(p)
    store.json('diagnostics/window_tests.json',dict(tests=tests,method=scalp_test.__doc__,
        warning='Additional exploratory reader-level test, not a replacement for the completed mixed models/TCT/cluster tests.'))
    store.json('diagnostics/reader_consistency.json',similarity)
    store.json('diagnostics/covariate_balance.json',balance(meta))
    audit_path=store.root/'diagnostics/label_audit.json'
    if not audit_path.exists():store.json('diagnostics/label_audit.json',label_audit(meta,work))
    # Independently validated recovered clocks are listed explicitly in proof.
    proof=read(ROOT/'docs/paper_clock_recovery_verified.json')['blocks']
    from word_sentiment_eeg.paper_verified_recovery import approved_clock
    approved={(v['subject'],str(v['block'])) for v in proof if approved_clock(v)}
    # If evidence schema changes, refuse to silently call every block original.
    if len(approved)!=34:raise ValueError('Expected 34 approved clock-recovery blocks; review evidence schema')
    meta['block_set']=['recovered' if (s,b) in approved else 'original' for s,b in zip(meta.subject,meta.block_id)]
    rows=[]
    for (subject,kind),m in meta.groupby(['subject','block_set']):
        rows.append(dict(reader=subject,block_set=kind,negative=int(m.sentiment.eq('negative').sum()),positive=int(m.sentiment.eq('positive').sum())))
    store.json('diagnostics/block_set_counts.json',rows)
    block_maps=[]
    for r,name in enumerate(['raw','cleaned']):
        for subject in subjects:
            maps_by_kind={}
            for kind in ['original','recovered']:
                ix=np.flatnonzero((meta.subject==subject)&(meta.block_set==kind));a=ix[y[ix]==1];b=ix[y[ix]==0]
                if min(len(a),len(b))<2:continue
                win=(times>=224)&(times<=304)
                maps_by_kind[kind]=x[r,a][:,:,win].mean(axis=(0,2))-x[r,b][:,:,win].mean(axis=(0,2))
            if len(maps_by_kind)==2:
                a,b=maps_by_kind.values();den=np.linalg.norm(a)*np.linalg.norm(b)
                block_maps.append(dict(reader=subject,representation=name,cosine_original_recovered=float(a@b/den) if den else None))
    store.json('diagnostics/block_set_consistency.json',dict(rows=block_maps,
        warning='Descriptive subset comparisons; same full-reader cleaning model, different sentence/trial pools. Not independent replications.'))
    # Repeated word: different source occurrences, same reader. Label alone is
    # insufficient: compare within-word versus same-class different-word pairs.
    rng=np.random.default_rng(SEED);repeat=[];win=(times>=224)&(times<=304)
    for subject,g in meta.groupby('subject'):
        for r,name in enumerate(['raw','cleaned']):
            same=[];control=[]
            for word,h in g.groupby('word_key'):
                h=h.drop_duplicates(['sentence_key','word_index'])
                if len(h)<2:continue
                ii=rng.choice(h.index.to_numpy(),size=2,replace=False);a,b=ii
                pool=g[(g.word_key!=word)&(g.sentiment==meta.loc[a,'sentiment'])]
                if pool.empty:continue
                c=int(rng.choice(pool.index.to_numpy()))
                va=x[r,a,:,win].mean(axis=0);vb=x[r,b,:,win].mean(axis=0);vc=x[r,c,:,win].mean(axis=0)
                def corr(u,v):
                    return float(np.corrcoef(u,v)[0,1]) if min(np.std(u),np.std(v))>0 else None
                sa,co=corr(va,vb),corr(va,vc)
                if sa is not None and co is not None:same.append(sa);control.append(co)
            repeat.append(dict(reader=subject,representation=name,word_comparisons=len(same),
                same_word_mean_r=float(np.mean(same)) if same else None,
                same_class_other_word_mean_r=float(np.mean(control)) if control else None))
    store.json('diagnostics/repeated_words.json',dict(rows=repeat,warning='Descriptive scalp-map correlation at 224–304 ms; shared reader/context and noise remain possible explanations.'))
    make_plots(tests,maps,similarity,store,work)
    return meta

def make_plots(tests,maps,similarity,store,work):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[p['reader'] for p in similarity[:len(maps[0])]]
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,t,m in zip(axes.flat,tests,maps):
        corr=np.corrcoef(m);im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(names)),names,rotation=90);ax.set_yticks(range(len(names)),names)
        ax.set_title(f"{t['representation']} {t['start_ms']}–{t['end_ms']} ms\nreader map correlation; Holm p={t['p_holm_four_tests']:.3f}")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Scalp contrast-map correlation (descriptive)')
    p=work/'reader_map_correlations.png';fig.savefig(p,dpi=140);plt.close(fig);store.file('figures/'+p.name,p)

def features(x,times,window):
    if window=='published_224_304':return x[:,:, (times>=224)&(times<=304)].mean(axis=2)
    lo,hi=(0,800) if window=='post_0_800' else (-200,0)
    return np.stack([x[:,:,(times>=a)&(times<a+40)].mean(axis=2) for a in range(lo,hi,40)],axis=2).reshape(len(x),-1)

def run_classifiers(x,meta,store,work):
    y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy();times=np.arange(-75,126)*8
    summary=[]
    for r,rep in enumerate(['raw','cleaned']):
        for window in SPEC['classification_windows']:
            f=features(x[r],times,window)
            for split in SPEC['splits']:
                for size in SPEC['averaging']:
                    if window=='pre_minus200_0' and size!=1:continue
                    for model in ['logistic','dummy']:
                        name=f'{rep}__{window}__{split}__avg{size}__{model}'
                        path=store.root/'classification'/f'{name}.json'
                        if path.is_file():result=read(path);print('KEEP CLASSIFICATION '+name,flush=True)
                        else:
                            print('FIT '+name,flush=True)
                            result=decode_case(f,y,meta,split,size,model,window,rep)
                            store.json('classification/'+name+'.json',result)
                        summary.append({k:result[k] for k in ['representation','window','split','average_trials','model','status','pooled']})
            del f
    # Metadata-only control assesses separability without EEG. Word/sentence IDs
    # are not predictors; the same predeclared group splits are used.
    cov=meta[['word_length','word_frequency','duration_ms','previous_gap_ms','next_gap_ms']].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    for split in SPEC['splits']:
        p=store.root/'classification'/f'metadata_only__{split}.json'
        if not p.exists():store.json(p.relative_to(store.root),decode_case(cov,y,meta,split,1,'logistic','metadata_only','no_EEG'))
        result=read(p);summary.append({k:result[k] for k in ['representation','window','split','average_trials','model','status','pooled']})
    store.json('classification/summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--phase',choices=['diagnostics','classification','cnn'],required=True)
    parser.add_argument('--run-dir',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--work-root',type=Path)
    args=parser.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; no native EEG or trial downloads on Mac')
    run=args.run_dir or args.thesis_root/'Results/word_sentiment_eeg/zuco_paper_figures_recovered'
    out=args.output or args.thesis_root/'Results/word_sentiment_eeg/zuco_recovered_followups'
    if run.resolve()==out.resolve() or run.resolve() in out.resolve().parents:raise ValueError('Follow-ups must have a separate output folder')
    state=read(run/'cpu_preparation_manifest.json');final=read(run/'manifest.json')
    if state['status']!='completed' or final['status']!='completed':raise ValueError('Finish existing CPU/GPU paper stages first')
    if state['protocol_sha256']!=file_digest(run/'protocol.json'):raise ValueError('CPU protocol changed')
    for s in state['subjects']:
        if file_digest(run/'readers'/s/'prepared.json')!=state['reader_checkpoint_sha256'][s]:raise ValueError('Reader completion changed')
    code=['scripts/zuco_recovered_followups.py','src/word_sentiment_eeg/paper_labels.py','scripts/paper_sst_source.py','scripts/zuco_recovered_cnn.py']
    protocol=dict(spec=SPEC,cpu_manifest=file_digest(run/'cpu_preparation_manifest.json'),
        final_manifest=file_digest(run/'manifest.json'),code={p:file_digest(ROOT/p) for p in code},
        software={p:version(p) for p in ['numpy','scipy','pandas','scikit-learn']})
    store=ImmutableStore(out,SPEC['drive_cap_bytes']);store.json('protocol.json',protocol)
    marker=out/f'{args.phase}_complete.json'
    if marker.exists():
        done=read(marker)
        for p,sha in done['checksums'].items():
            if file_digest(out/p)!=sha:raise ValueError('Completed follow-up changed: '+p)
        print(f'KEEP COMPLETED {args.phase}: verified outputs; no trial loading or fitting',flush=True);return
    root=args.work_root or Path('/content/zuco-followup-work')
    if 'drive' in root.resolve().parts or str(root.resolve()).startswith(str(args.thesis_root.resolve())):raise ValueError('Work disk must be runtime-local')
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        work=Path(temporary);x,meta,paired,provenance=stage(run,work,state['subjects'],store,digest(protocol))
        if args.phase=='diagnostics':diagnostics(x,meta,paired,provenance,store,work)
        elif args.phase=='classification':run_classifiers(x,meta,store,work)
        else:
            from zuco_recovered_cnn import run_cnn
            run_cnn(x,meta,store,work)
        del x
    directories={'diagnostics':['diagnostics','figures'],'classification':['classification'],'cnn':['cnn']}[args.phase]
    checks={str(p.relative_to(out)):file_digest(p) for d in directories for p in (out/d).rglob('*') if p.is_file()}
    store.json(marker.name,dict(status='completed',protocol=digest(protocol),checksums=checks))
    print(f'COMPLETE {args.phase}: {out}; existing preparation and figures untouched',flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
