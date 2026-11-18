#!/usr/bin/env python3
"""TeCo reader, fixation-duration and familiar-identity sentiment sweeps. Cache-only."""
import argparse,gzip,json,warnings
from io import BytesIO
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import analyze_trial_archive as a
import teco_average40 as old
import teco_classification_sweep as previous
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score
SEED=20261119
MODELS=['lr_0.1','lr_1','lr_10','lda','rbf','trees','forest','mlp','knn']
PLAN={
 'reader_identity':dict(modes=['familiar','sentence'],sizes=[1,10,40,'max40']),
 'fixation_200':dict(modes=['familiar','sentence','reader'],sizes=[1,10,40,'max40']),
 'fixation_150':dict(modes=['familiar'],sizes=[1,10]),
 'fixation_250':dict(modes=['familiar'],sizes=[1,10]),
 'sentiment_binary':dict(modes=['familiar'],sizes=[1,5,10,20,40,'max40']),
 'sentiment_three_class':dict(modes=['familiar'],sizes=[1,10,40,'max40']),
 'neutral_nonneutral':dict(modes=['familiar'],sizes=[1,10,40,'max40'])}
SPEC=dict(version=1,seed=SEED,plan=PLAN,models=MODELS,budget=50_000_000,
 data='TeCo task1 only; all accepted cached epochs for reader/duration; frozen original prediction rosters for sentiment',
 selection='3 outer folds,2 inner folds; C1 logistic screens all inputs; top3 advance to9 model settings; test never selects',
 familiar='Every retained test reader, sentence_key and word_identity occurs in training AFTER full-epoch overlap purging; report discarded test events',
 purge='Remove training events within1000ms of heldout events in same reader/sentence, including inner folds',
 duration='Fixed thresholds150,200,250ms; <threshold short, >=threshold long; nonpositive/nonfinite durations excluded',
 duration_leakage='Never input fixation duration, next-fixation delay, residualized features containing those covariates, identities or word text',
 average='Within reader and target class, after splitting; exact sizes discard remainders; max40 retains and reports remainders',
 limitations=['Class labels construct sentiment/duration averages; offline pseudotrials, not individual-event deployment',
 'Reader identity can reflect recording/session characteristics; one recording/session per reader cannot separate these',
 'Duration prediction can reflect eye movements and overlapping responses, not a pure cognitive state',
 'No intentional word/sentence/reader holdout in familiar setting; separate EEG events always held out',
 'Not exact paper replication: grouped nested selection,3 folds, no NCA or100random95/5splits',
 'Uses previously explored dataset; no new independent confirmation or significance claims',
 'No word-length classification; no raw EEG reads, extraction, relabeling or new large feature cache'])


def read(path):return json.loads(Path(path).read_text())


def zipped_json(store,path,value):
    payload=(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    store.bytes(path,gzip.compress(payload,mtime=0))


def case(store,path,fn):
    path+='/result.json.gz';target=store.root/path
    if target.is_file():
        r=json.loads(gzip.decompress(target.read_bytes()));payload={k:v for k,v in r.items() if k!='checksum'}
        if r['checksum']!=a.digest(payload):raise ValueError('Corrupt completed case: '+path)
        return payload
    try:r=dict(status='completed',**fn())
    except ValueError as error:r=dict(status='unavailable',reason=str(error))
    zipped_json(store,path,dict(**r,checksum=a.digest(r)));return r


def load_all(thesis):
    archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz';source=thesis/'Results/word_sentiment_eeg/classification_similarity'
    if read(source/'manifest.json')['status']!='completed':raise ValueError('Original feature preparation incomplete')
    records,_=a.load_completed(archive);records=sorted([r for r in records if r['dataset']=='teco' and r['task']=='task1'],key=lambda r:r['subject'])
    if len(records)!=12:raise ValueError('Expected12TeCo readers')
    tables=[];post=[];pre=[];nuisance=[];identities=[]
    for i,record in enumerate(records,1):
        print(f'LOAD EXISTING CACHE reader {i}/{len(records)}: {record["subject"]}; no EEG shard reads',flush=True)
        m,x,b,n,identity=old.cached_features_only(archive,source,record)
        tables.append(m);post.append(x);pre.append(b);nuisance.append(n);identities.append(identity)
    meta=pd.concat(tables,ignore_index=True);post=np.concatenate(post);pre=np.concatenate(pre);nuisance=np.concatenate(nuisance)
    if meta.event_id.duplicated().any():raise ValueError('Duplicate event ID')
    duplicate=meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False)|meta.signal_sha256.duplicated(keep=False)
    keep=~duplicate.to_numpy();meta=meta.loc[keep].reset_index(drop=True)
    features=regional_inputs(post[keep],pre[keep]);features['metadata_control']=nuisance[keep][:,[0,1,2,5]].astype(np.float32)
    return meta,features,dict(readers=identities,discarded_duplicate_events=int(duplicate.sum()),source_protocol=a.file_digest(source/'protocol.json'))


def regional_inputs(post,pre):
    # Original cache is region-major: six regions x eight100ms bins, NOT window-major.
    if post.ndim!=2 or post.shape[1]!=48 or pre.shape!=(len(post),12):raise ValueError('Unexpected cached feature layout')
    if not np.isfinite(post).all() or not np.isfinite(pre).all():raise ValueError('Nonfinite EEG feature')
    v=post.reshape(-1,6,8);out={'regional_bins':post.astype(np.float32),'pre_control':pre.astype(np.float32)}
    for lo,hi in [(0,200),(200,400),(300,500),(400,800),(0,500)]:out[f'regional_{lo}_{hi}']=v[:,:,lo//100:hi//100].reshape(len(v),-1)
    for j,r in enumerate(a.REGIONS):out['region_'+r]=v[:,j,:]
    out['regional_means']=v.mean(2);out['regional_shape']=np.concatenate([v.mean(2),v.std(2),np.ptp(v,axis=2)],axis=1)
    out['unit_norm_bins']=post/np.maximum(np.linalg.norm(post,axis=1,keepdims=True),1e-8)
    return {k:np.asarray(x,dtype=np.float32) for k,x in out.items()}


def frozen_sentiment(thesis,meta,kind):
    source=thesis/'Results/word_sentiment_eeg/classification_similarity/teco/task1/cases'
    target='binary' if kind=='sentiment_binary' else 'three_class'
    folder=source/f'word_agreement__{target}__sentence__post__logistic';marker=read(folder/'result.json');path=folder/'predictions.csv.gz'
    if a.file_digest(path)!=marker['checksums']['predictions.csv.gz']:raise ValueError('Frozen sentiment label checksum mismatch')
    roster=pd.read_csv(path,keep_default_na=False,dtype={'event_id':str,'subject':str})
    if roster.event_id.duplicated().any() or len(roster)!=marker['score']['trials']:raise ValueError('Invalid frozen sentiment roster')
    expected={'negative','positive'} if target=='binary' else {'negative','neutral','positive'}
    if set(roster.truth)!=expected:raise ValueError('Unexpected original sentiment classes')
    counts={c:int(roster.truth.eq(c).sum()) for c in expected}
    if counts!=marker['score']['class_counts']:raise ValueError('Frozen sentiment counts mismatch')
    positions=pd.Index(meta.event_id).get_indexer(roster.event_id)
    if (positions<0).any():raise ValueError('Frozen sentiment events missing from valid caches')
    if meta.subject.iloc[positions].tolist()!=roster.subject.tolist():raise ValueError('Frozen reader identity mismatch')
    labels=roster.truth.to_numpy(dtype=str)
    if kind=='neutral_nonneutral':labels=np.where(labels=='neutral','neutral','nonneutral')
    return positions,labels,dict(label_source=str(path),sha256=a.file_digest(path))


def advanced_binary(thesis,rows):
    """Reuse20existing frozen-binary input vectors; never rebuild or read EEG."""
    root=thesis/'Results/word_sentiment_eeg/teco_classification_sweep';state=read(root/'complete.json')
    if state['status']!='completed' or a.file_digest(root/'protocol.json')!=state['checksums']['protocol.json']:raise ValueError('Complete prior TeCo sweep required for advanced binary inputs')
    protocol=read(root/'protocol.json')
    if protocol['code_sha256']['scripts/teco_classification_sweep.py']!=a.file_digest(ROOT/'scripts/teco_classification_sweep.py'):raise ValueError('Prior feature code differs from tested input specification')
    result={};seen=set()
    for subject,g in rows.groupby('subject',sort=True):
        prefix=root/'features'/subject;marker_path=prefix/'complete.json'
        if a.file_digest(marker_path)!=state['checksums'][str(marker_path.relative_to(root))]:raise ValueError('Prior feature marker checksum mismatch')
        marker=read(marker_path);path=prefix/'vectors.npz'
        if a.file_digest(path)!=marker['sha256']:raise ValueError('Corrupt prior compact feature cache')
        with np.load(path,allow_pickle=False) as f:
            ids=f['event_id'].astype(str);ix=pd.Index(ids).get_indexer(g.event_id)
            if (ix<0).any():raise ValueError('Prior advanced features missing selected events')
            for k in f.files:
                if k in ['event_id','pre_control'] or k.startswith('residual_'):continue
                v=f[k][ix]
                if not np.isfinite(v).all():raise ValueError('Invalid advanced EEG input')
                if k not in result:result[k]=np.empty((len(rows),v.shape[1]),np.float32)
                result[k][g.index]=v
        seen.update(g.index)
    if len(seen)!=len(rows):raise ValueError('Incomplete advanced feature rows')
    return result


def prepare_target(thesis,meta,features,name):
    provenance={};removed=0
    if name=='reader_identity':ix=np.arange(len(meta));labels=meta.subject.astype(str).to_numpy()
    elif name.startswith('fixation_'):
        duration=pd.to_numeric(meta.first_fixation_duration_ms,errors='coerce').to_numpy(dtype=float)
        ix=np.flatnonzero(np.isfinite(duration)&(duration>0));removed=len(meta)-len(ix);threshold=int(name.split('_')[1])
        labels=np.where(duration[ix]<threshold,'short','long');provenance=dict(threshold_ms=threshold,excluded_invalid_duration=removed)
    else:ix,labels,provenance=frozen_sentiment(thesis,meta,name)
    rows=meta.iloc[ix].reset_index(drop=True);classes=sorted(set(labels));y=pd.Categorical(labels,categories=classes).codes.astype(int)
    out={k:v[ix] for k,v in features.items()}
    if name=='sentiment_binary':
        try:out.update({'advanced_'+k:v for k,v in advanced_binary(thesis,rows).items()});provenance['advanced_inputs']='verified prior compact sweep cache'
        except FileNotFoundError:provenance['advanced_inputs']='absent; use original regional inputs, no new preparation'
    if name.startswith('fixation_'):
        # The only non-EEG comparator has word length/position, sentence length/control flag.
        # No measured duration or timing proxy is a predictor, nor a residualization covariate.
        if any(k.startswith('residual_') for k in out):raise AssertionError('Duration leaked into input transform')
    counts={c:int((y==i).sum()) for i,c in enumerate(classes)}
    return rows,y,out,classes,dict(**provenance,classes=classes,class_counts=counts,trials=len(rows),readers=rows.subject.nunique())


def split(meta,y,indices,mode,n,seed):
    from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
    indices=np.asarray(indices,dtype=int)
    if mode=='familiar':
        strata=meta.subject.iloc[indices].astype(str).to_numpy()+'_'+y[indices].astype(str)
        labels,counts=np.unique(strata,return_counts=True)
        if counts.min()<n:raise ValueError('Too few examples in a reader/class for stratification')
        raw=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(indices)),strata)
    else:
        col={'sentence':'sentence_key','reader':'subject'}[mode];groups=meta[col].iloc[indices].astype(str).to_numpy()
        raw=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(indices)),y[indices],groups)
    results=[];required=set(y[indices])
    for train,test in raw:
        train,test=indices[train],indices[test];before=len(train)
        clean,_=a.purge_overlapping_train(meta,y,[(train,test)]);train=clean[0][0]
        eligible=np.ones(len(test),dtype=bool)
        if mode=='familiar':
            for col in ['subject','sentence_key','word_identity']:eligible &= meta[col].iloc[test].isin(set(meta[col].iloc[train])).to_numpy()
        removed=int((~eligible).sum());test=test[eligible]
        if set(y[train])!=required or set(y[test])!=required:raise ValueError('Missing target classes after overlap purge/familiar-identity restriction')
        if set(train)&set(test):raise AssertionError('Event leakage')
        fractions={c:float(meta[c].iloc[test].isin(set(meta[c].iloc[train])).mean()) for c in ['subject','sentence_key','word_identity']}
        if mode=='familiar' and any(v!=1 for v in fractions.values()):raise AssertionError('Familiar-identity guarantee failed')
        if mode!='familiar':
            col={'sentence':'sentence_key','reader':'subject'}[mode]
            if set(meta[col].iloc[train])&set(meta[col].iloc[test]):raise AssertionError('Held-out group leakage')
        results.append((train,test,dict(purged_training=before-len(train),discarded_unfamiliar_test=removed,test_seen_fraction=fractions)))
    return results


def membership(meta,y,indices,size,seed):
    rng=np.random.default_rng(seed);groups=[];cap=40 if size=='max40' else int(size)
    for reader in sorted(meta.subject.iloc[indices].unique()):
        for c in sorted(set(y)):
            ix=indices[(meta.subject.to_numpy()[indices]==reader)&(y[indices]==c)];ix=rng.permutation(ix)
            for start in range(0,len(ix),cap):
                g=ix[start:start+cap]
                if len(g)==cap or (size=='max40' and len(g)):groups.append(g)
    if not groups:raise ValueError('No averaging groups')
    flat=np.concatenate(groups)
    if len(flat)!=len(set(flat)) or not set(flat)<=set(indices):raise AssertionError('Repeated/cross-partition averaging event')
    return groups


def means(X,groups):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.stack([np.nanmean(X[g],axis=0) for g in groups]).astype(np.float32)


def estimator(name,nclasses):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.kernel_approximation import Nystroem
    from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier
    base=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
    if name.startswith('lr_'):model=LogisticRegression(C=float(name[3:]),class_weight='balanced',solver='lbfgs',max_iter=600,random_state=SEED)
    elif name=='lda':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(nclasses)/nclasses)
    elif name=='rbf':
        base.append(Nystroem(kernel='rbf',gamma=.02,n_components=64,random_state=SEED))
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=600,random_state=SEED)
    elif name in ['trees','forest']:
        cls=ExtraTreesClassifier if name=='trees' else RandomForestClassifier
        model=cls(n_estimators=48,max_depth=10,min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=SEED)
    elif name=='mlp':model=MLPClassifier(hidden_layer_sizes=(32,),alpha=1,max_iter=120,early_stopping=False,random_state=SEED)
    elif name=='knn':model=KNeighborsClassifier(n_neighbors=5,weights='distance')
    elif name=='dummy':model=DummyClassifier(strategy='prior')
    else:raise ValueError(name)
    return make_pipeline(*base,model)


def score(meta,y,X,groups,model_name,nclasses,detailed=False):
    train,test=groups;yt=np.array([y[g[0]] for g in train]);ye=np.array([y[g[0]] for g in test]);required=set(range(nclasses))
    if set(yt)!=required or set(ye)!=required or min(np.bincount(ye,minlength=nclasses))<2:raise ValueError('Too few examples per target class after averaging')
    model=estimator(model_name,nclasses)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(means(X,train),yt)
    pred=model.predict(means(X,test));sizes=[len(g) for g in test]
    result=dict(balanced_accuracy=float(balanced_accuracy_score(ye,pred)),macro_f1=float(f1_score(ye,pred,average='macro',zero_division=0)),
        confusion=confusion_matrix(ye,pred,labels=list(range(nclasses))).tolist(),test_examples=len(test),test_events=int(sum(sizes)),train_examples=len(train),
        actual_group_min=int(min(sizes)),actual_group_max=int(max(sizes)),actual_group_mean=float(np.mean(sizes)),warnings=sorted(set(str(w.message)[:250] for w in caught)))
    if detailed:result.update(truth=ye.tolist(),prediction=pred.astype(int).tolist())
    return result


def pooled(cases):
    if not cases or any(r['status']!='completed' for r in cases):return None
    matrix=np.sum([r['confusion'] for r in cases],axis=0)
    return float(np.mean(np.diag(matrix)/matrix.sum(axis=1)))


def save_members(store,path,train,test,audit):
    # Compressed index lists are enough to reproduce groups; target events.csv.gz maps indices to IDs.
    zipped_json(store,path,dict(train=[g.tolist() for g in train],test=[g.tolist() for g in test],audit=audit))


def evaluate(meta,y,features,classes,plan,store,target,models=MODELS):
    rows=[];development=[];skipped=[];inputs=sorted(k for k in features if k not in ['metadata_control','pre_control']);nclasses=len(classes)
    for mode in plan['modes']:
        try:outer=split(meta,y,np.arange(len(meta)),mode,3,SEED)
        except ValueError as e:skipped.append(dict(target=target,mode=mode,reason=str(e)));continue
        for fi,(train,test,audit) in enumerate(outer):
            try:inner=split(meta,y,train,mode,2,SEED+100+fi)
            except ValueError as e:skipped.append(dict(target=target,mode=mode,fold=fi,reason=str(e)));continue
            for size in plan['sizes']:
                prefix=f'fits/{target}/{mode}/fold{fi}/size{size}'
                print(f'TARGET {target} | {mode} | outer fold {fi+1}/3 | averaging {size} | {len(train)} train/{len(test)} test events',flush=True)
                try:gs=[(membership(meta,y,tr,size,SEED+fi*10+j),membership(meta,y,te,size,SEED+fi*10+j+1000)) for j,(tr,te,_) in enumerate(inner)]
                except ValueError as e:skipped.append(dict(target=target,mode=mode,fold=fi,size=str(size),reason=str(e)));continue
                for j,((gt,ge),(_,_,audit_inner)) in enumerate(zip(gs,inner)):save_members(store,prefix+f'/inner{j}/members.json.gz',gt,ge,audit_inner)
                candidates=[]
                def dev(inp,model_name):
                    results=[case(store,prefix+f'/development/{inp}/{model_name}/inner{j}',lambda groups=groups:score(meta,y,features[inp],groups,model_name,nclasses)) for j,groups in enumerate(gs)]
                    value=pooled(results);r=dict(target=target,mode=mode,fold=fi,size=str(size),input=inp,model=model_name,validation_ba=value)
                    candidates.append(r)
                for ci,inp in enumerate(inputs):
                    dev(inp,'lr_1')
                    if ci%5==0 or ci==len(inputs)-1:print(f'  screen input {ci+1}/{len(inputs)}',flush=True)
                ranked=sorted([r for r in candidates if r['validation_ba'] is not None],key=lambda r:(-r['validation_ba'],r['input'],r['model']))
                if not ranked:skipped.append(dict(target=target,mode=mode,fold=fi,size=str(size),reason='Insufficient development class coverage'));development.extend(candidates);continue
                for inp in [r['input'] for r in ranked[:3]]:
                    for name in models:
                        if name!='lr_1':dev(inp,name)
                    print('  model search complete:',inp,flush=True)
                development.extend(candidates);winner=sorted([r for r in candidates if r['validation_ba'] is not None],key=lambda r:(-r['validation_ba'],r['input'],r['model']))[0]
                try:gt=membership(meta,y,train,size,SEED+fi+2000);ge=membership(meta,y,test,size,SEED+fi+3000)
                except ValueError as e:skipped.append(dict(target=target,mode=mode,fold=fi,size=str(size),reason=str(e)));continue
                save_members(store,prefix+'/outer_members.json.gz',gt,ge,audit)
                fixed='advanced_channels_224_304' if 'advanced_channels_224_304' in features else 'regional_bins'
                for variant,inp,model_name in [('selected_eeg',winner['input'],winner['model']),('fixed_eeg',fixed,'lr_1'),('pre_control','pre_control','lr_1'),('metadata_control','metadata_control','lr_1'),('dummy',fixed,'dummy')]:
                    result=case(store,prefix+'/test/'+variant,lambda:score(meta,y,features[inp],(gt,ge),model_name,nclasses,True))
                    rows.append(dict(target=target,mode=mode,fold=fi,size=str(size),variant=variant,input=inp,model=model_name,classes=classes,chance=1/nclasses,**result))
                store.json(prefix+'/choice.json',dict(winner=winner,used_outer_test=False))
                print('SAVED TEST RESULT:',target,mode,fi+1,size,flush=True)
    return rows,development,skipped


def summarize(rows):
    if not rows:return pd.DataFrame()
    table=pd.DataFrame(rows);result=[]
    for (target,mode,size,variant),g in table.groupby(['target','mode','size','variant']):
        completed=g[g.status.eq('completed')];r=dict(target=target,mode=mode,size=size,variant=variant,completed_folds=len(completed),chance=float(g.chance.iloc[0]))
        if len(completed)==3:
            truth=np.concatenate(completed.truth.tolist());pred=np.concatenate(completed.prediction.tolist())
            r.update(balanced_accuracy=float(balanced_accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),test_examples=len(truth),
                test_events=int(completed.test_events.sum()),fold_min=float(completed.balanced_accuracy.min()),fold_max=float(completed.balanced_accuracy.max()),
                actual_group_min=int(completed.actual_group_min.min()),actual_group_max=int(completed.actual_group_max.max()))
        result.append(r)
    return pd.DataFrame(result)


def figures(store,summary,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=summary.target.unique();fig,axes=plt.subplots(len(targets),1,figsize=(11,3*len(targets)),squeeze=False)
    for ax,target in zip(axes[:,0],targets):
        g=summary[(summary.target==target)&summary.completed_folds.eq(3)&summary.variant.isin(['selected_eeg','fixed_eeg','pre_control','metadata_control'])].copy()
        labels=(g['mode']+' / '+g['size']+' / '+g.variant).tolist();ax.bar(np.arange(len(g)),g.balanced_accuracy*100)
        ax.axhline(float(summary[summary.target==target].chance.iloc[0])*100,ls=':',color='black');ax.set(title=target,ylabel='Balanced accuracy (%)',ylim=(0,100));ax.set_xticks(np.arange(len(g)),labels,rotation=65,ha='right',fontsize=7)
    fig.tight_layout();b=BytesIO();fig.savefig(b,format='png',dpi=120);plt.close(fig);store.bytes('overview.png',b.getvalue())
    identity=[r for r in rows if r['target']=='reader_identity' and r['mode']=='sentence' and r['size']=='1' and r['variant']=='selected_eeg' and r['status']=='completed']
    if len(identity)==3:
        cm=np.sum([r['confusion'] for r in identity],axis=0);norm=cm/cm.sum(1,keepdims=True)
        fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=np.arange(len(cm)),yticks=np.arange(len(cm)),xticklabels=identity[0]['classes'],yticklabels=identity[0]['classes'],xlabel='Predicted reader',ylabel='True reader',title='Reader identity: unseen sentences, individual trials');plt.setp(ax.get_xticklabels(),rotation=60,ha='right');fig.colorbar(im,ax=ax);fig.tight_layout();b=BytesIO();fig.savefig(b,format='png',dpi=130);plt.close(fig);store.bytes('reader_confusion.png',b.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    out=args.thesis_root/'Results/word_sentiment_eeg/teco_other_targets';store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve at most50MB for new compressed results; no analysis started')
    import fcntl,tempfile
    lock=Path(tempfile.gettempdir(),'teco-other-'+a.digest(str(out))[:16]+'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta,features,source=load_all(args.thesis_root)
    files=['scripts/teco_other_targets.py','scripts/teco_average40.py','scripts/teco_classification_sweep.py','scripts/analyze_trial_archive.py']
    store.json('protocol.json',dict(spec=SPEC,source=source,code={f:a.file_digest(ROOT/f) for f in files},versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']}))
    if (out/'complete.json').is_file():
        for path,sha in read(out/'complete.json')['checksums'].items():
            if a.file_digest(out/path)!=sha:raise ValueError('Corrupt completed output')
        print('KEEP COMPLETE OTHER-TARGET SWEEP: outputs verified, no refits',flush=True);return
    rows=[];development=[];skipped=[]
    for ti,(name,plan) in enumerate(PLAN.items(),1):
        print(f'BEGIN TARGET {ti}/{len(PLAN)}: {name}',flush=True)
        try:table,y,vectors,classes,info=prepare_target(args.thesis_root,meta,features,name)
        except FileNotFoundError as e:skipped.append(dict(target=name,reason='Required saved label roster absent: '+str(e)));continue
        store.json(f'targets/{name}/definition.json',info)
        export=table[['event_id','subject','sentence_key','word_identity','sentence_position','onset_sample_sentence_0based']].copy();export['target_code']=y
        old.csv(store,f'targets/{name}/events.csv.gz',export)
        result,dev,unavailable=evaluate(table,y,vectors,classes,plan,store,name);rows.extend(result);development.extend(dev);skipped.extend(unavailable)
    if not rows:raise ValueError('No usable target evaluations')
    summary=summarize(rows);old.csv(store,'summary.csv',summary);old.csv(store,'development.csv.gz',pd.DataFrame(development));old.csv(store,'unavailable.csv',pd.DataFrame(skipped,columns=['target','mode','fold','size','reason']))
    zipped_json(store,'outer_results.json.gz',rows);figures(store,summary,rows)
    paths=['protocol.json','summary.csv','development.csv.gz','unavailable.csv','outer_results.json.gz','overview.png']+[str(p.relative_to(out)) for p in out.rglob('result.json.gz')]+[str(p.relative_to(out)) for p in out.glob('targets/*/definition.json')]+[str(p.relative_to(out)) for p in out.glob('targets/*/events.csv.gz')]
    if (out/'reader_confusion.png').is_file():paths.append('reader_confusion.png')
    store.json('complete.json',dict(status='completed',checksums={f:a.file_digest(out/f) for f in paths},outer_cases=len(rows)))
    print('TECO OTHER TARGETS COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
