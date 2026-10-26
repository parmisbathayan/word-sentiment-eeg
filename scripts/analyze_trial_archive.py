#!/usr/bin/env python3
"""Fixed classification and similarity battery on the completed 125-Hz archive.

No source EEG, re-epoching, relabeling, hyperparameter search or trial balancing.
Stream each shard once into small reusable features. All outputs stay private.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import atomic_npz,digest,file_digest,write_json

VERSION='word_trial_classification_similarity_v1'
SEED=2026
CLASSES=('negative','neutral','positive')
REGIONS=('anterior','central','posterior','occipital','left_lateral','right_lateral')
POST_WINDOWS=tuple((a,a+100) for a in range(0,800,100))
PRE_WINDOWS=((-200,-100),(-100,0))
POLICIES=('word_agreement','all_three_content','lexical_content','contextual_content','sentence_content')
NUISANCE=('word_length','relative_word_position','sentence_length','first_fixation_duration_ms','next_word_first_fixation_ms','control_numeric')
FEATURE_SPEC={'version':2,'regions':REGIONS,'post_windows_ms':POST_WINDOWS,'pre_windows_ms':PRE_WINDOWS,
              'nuisance':NUISANCE,'endpoint':'left inclusive, right exclusive','expected_hz':125,'expected_samples':126}


def boolean(values):
    return values.astype(str).str.strip().str.lower().isin(['true','1','yes'])


def strict_path(root,relative):
    path=(root/str(relative)).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Archive relative path leaves reader directory')
    return path


def feature_blocks(eeg,times,region_names):
    if eeg.ndim!=3 or eeg.shape[2]!=126 or not np.array_equal(times,np.arange(-25,101)*8):
        raise ValueError('Expected all-channel 125Hz epochs on -200:8:800 grid')
    if not np.isfinite(eeg).all():raise ValueError('Nonfinite archived signal')
    picks=[np.flatnonzero(np.asarray(region_names)==region) for region in REGIONS]
    if any(not len(p) for p in picks):raise ValueError('Empty scalp region')
    regional=np.stack([eeg[:,p,:].mean(axis=1) for p in picks],axis=1)
    def windows(ranges):
        return np.stack([regional[:,:,(times>=a)&(times<b)].mean(axis=2) for a,b in ranges],axis=2).reshape(len(eeg),-1).astype(np.float32)
    return windows(POST_WINDOWS),windows(PRE_WINDOWS)


def load_completed(archive):
    m=json.loads((archive/'manifest.json').read_text());protocol=json.loads((archive/'protocol.json').read_text())
    if m.get('status')!='completed' or m.get('completed_reader_recordings')!=48 or m.get('expected_reader_recordings')!=48:
        raise ValueError('Wait for all 48 reader/task recordings and manifest status=completed; no partial archive is analyzed')
    if protocol.get('output_sample_rate_hz')!=125 or protocol.get('lowpass_hz')!=40:raise ValueError('This analysis expects the prepared 40Hz/125Hz archive')
    records=m['readers'];ids=[(r['dataset'],r['task'],r['subject']) for r in records]
    expected={('zuco',t) for t in ['SR','NR','TSR']}|{('teco','task1')}
    if len(ids)!=len(set(ids)) or set((d,t) for d,t,s in ids)!=expected:raise ValueError('Duplicate/unexpected recording roster')
    if any(sum((r['dataset'],r['task'])==pair for r in records)!=12 for pair in expected):raise ValueError('Each dataset/task requires 12 readers')
    return records,protocol


def attach_available_labels(meta,archive,thesis,dataset,task):
    # Join exact source identities; this does not reinterpret a word or label it.
    paths=[archive/'annotations'/f'{dataset}_word_labels.csv']
    if dataset=='zuco':paths.append(ROOT/'annotations/word_labels_three_level.csv')
    else:paths.append(thesis/'gpt uploaded files/word_sentiment_eeg/teco/word_labels_three_level.csv')
    source=next((p for p in paths if p.is_file()),None)
    if source is None:return meta,None
    frame=pd.read_csv(source,keep_default_na=False)
    if 'contextual_word_class' not in frame and 'contextual_class' in frame:frame['contextual_word_class']=frame.contextual_class
    if dataset=='zuco':
        frame=frame[frame.tasks.map(lambda s:task in str(s).split(';'))]
        keys=['sentence_key','word_index','word']
    else:keys=['sentence_id_teco','sentence_key','word_index','word']
    for key in keys:
        meta[key]=meta[key].astype(str);frame[key]=frame[key].astype(str)
    if frame.duplicated(keys).any():raise ValueError('Nonunique annotation join')
    columns=[c for c in ['lexical_class','contextual_word_class','sentence_context_class','grammatical_class','alignment_warning','annotation_status','is_control'] if c in frame]
    joined=meta.merge(frame[keys+columns],on=keys,how='left',validate='many_to_one',suffixes=('','_joined'),sort=False)
    if joined.event_id.tolist()!=meta.event_id.tolist():raise ValueError('Label join changed event order')
    for c in columns:
        new=c+'_joined' if c in meta else c
        target='annotation_alignment_warning' if c=='alignment_warning' else c
        if target not in joined:joined[target]='unknown'
        present=joined[new].notna()
        joined.loc[present,target]=joined.loc[present,new].astype(str)
        if new!=target:joined=joined.drop(columns=new)
    return joined,{'path':str(source),'sha256':file_digest(source)}


def reader_features(archive,record,out,thesis):
    d,t,s=record['dataset'],record['task'],record['subject']
    reader=archive/d/'readers'/t/s
    checkpoint=json.loads((reader/'checkpoint.json').read_text())
    if checkpoint['status']!='completed' or checkpoint['epochs']!=record['epochs']:raise ValueError('Incomplete/inconsistent reader checkpoint')
    if checkpoint['identity']['protocol_sha256']!=digest(json.loads((archive/'protocol.json').read_text())):raise ValueError('Reader/archive protocol mismatch')
    if file_digest(reader/'events.csv.gz')!=checkpoint['events_sha256']:raise ValueError('Reader event table checksum mismatch')
    signature=digest({'spec':FEATURE_SPEC,'checkpoint':file_digest(reader/'checkpoint.json')})
    cache=out/'features'/d/t/s;cache.mkdir(parents=True,exist_ok=True)
    mark=cache/'manifest.json'
    if mark.is_file():
        saved=json.loads(mark.read_text())
        if saved['signature']!=signature:raise ValueError('Feature cache differs from archive/specification; refusing mixed reuse')
        if any(file_digest(cache/k)!=sha for k,sha in saved['checksums'].items()):raise ValueError('Feature cache checksum mismatch')
        with np.load(cache/'features.npz',allow_pickle=False) as a:
            post=a['post'].copy();pre=a['pre'].copy();ids=a['event_id'].astype(str).tolist()
        meta=pd.read_csv(cache/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
        if ids!=meta.event_id.tolist() or len(post)!=len(meta):raise ValueError('Feature cache identity mismatch')
        print(f'KEEP FEATURES {d}/{t}/{s}: {len(meta)} trials',flush=True)
    else:
        meta=pd.read_csv(reader/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
        meta=meta[meta.status=='archived'].copy().reset_index(drop=True)
        if meta.event_id.duplicated().any() or len(meta)!=record['epochs']:raise ValueError('Invalid archived trial roster')
        m=pd.read_csv(archive/f'{d}_channel_regions.csv');channels=m.channel.astype(str).tolist()
        blocks=checkpoint['blocks'];post_parts=[];pre_parts=[];ordered=[];signal_hashes=[]
        for bi,b in enumerate(blocks):
            for relative in b['epoch_files']:
                path=strict_path(reader,relative)
                # One bounded byte read; checksum + NPZ decode use the same bytes.
                payload=path.read_bytes()
                import hashlib
                if hashlib.sha256(payload).hexdigest()!=b['checksums'][relative]:raise ValueError('Corrupt epoch shard: '+relative)
                rows=meta[meta.epoch_file==relative].sort_values('epoch_row',key=lambda v:v.astype(int))
                with np.load(BytesIO(payload),allow_pickle=False) as a:
                    if a['event_id'].astype(str).tolist()!=rows.event_id.tolist() or rows.epoch_row.astype(int).tolist()!=list(range(len(rows))):
                        raise ValueError('EEG/event join mismatch')
                    if a['channel_names'].astype(str).tolist()!=channels:raise ValueError('Channel/montage mismatch')
                    post,pre=feature_blocks(a['eeg_uV'],a['times_ms'],m.region.to_numpy())
                    signal_hashes.extend(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in a['eeg_uV'])
                ordered.extend(rows.event_id);post_parts.append(post);pre_parts.append(pre)
            if bi%10==0:print(f'FEATURES {d}/{t}/{s}: {bi+1}/{len(blocks)} blocks',flush=True)
        if len(ordered)!=len(set(ordered)) or set(ordered)!=set(meta.event_id):raise ValueError('Missing/repeated feature events')
        meta=meta.set_index('event_id').loc[ordered].reset_index()
        meta['signal_sha256']=signal_hashes
        post=np.concatenate(post_parts);pre=np.concatenate(pre_parts)
        # Full sentences remain in the original archive; compact feature caches
        # retain exact sentence IDs and tokens, not repeated sentence text.
        meta=meta.drop(columns=['sentence'],errors='ignore')
        atomic_npz(cache/'features.npz',post=post,pre=pre,event_id=np.array(ordered))
        meta.to_csv(cache/'events.csv.gz',index=False,compression='gzip')
        write_json(mark,{'signature':signature,'checksums':{k:file_digest(cache/k) for k in ['features.npz','events.csv.gz']},'trials':len(meta)})
    meta,label_source=attach_available_labels(meta,archive,thesis,d,t)
    meta['control_numeric']=np.where(meta.is_control.astype(str).str.lower().isin(['true','1']),1.,
                                    np.where(meta.is_control.astype(str).str.lower().isin(['false','0']),0.,np.nan))
    nuisance=np.stack([pd.to_numeric(meta[c],errors='coerce').to_numpy(dtype=float) for c in NUISANCE],axis=1)
    return meta,post,pre,nuisance,label_source


def selected(meta,policy,target):
    quality=~boolean(meta.amplitude_over_100_uV)
    alignment=meta.annotation_alignment_warning.astype(str).isin(['','nan'])
    quality &= alignment
    lex=meta.lexical_class.astype(str);ctx=meta.contextual_word_class.astype(str);sent=meta.sentence_context_class.astype(str)
    if policy=='word_agreement':labels=ctx;keep=quality & lex.eq(ctx)
    else:
        keep=quality & meta.grammatical_class.eq('content')
        if policy=='all_three_content':labels=lex;keep &=lex.eq(ctx)&lex.eq(sent)
        else:labels={'lexical_content':lex,'contextual_content':ctx,'sentence_content':sent}[policy]
    allowed=('negative','positive') if target=='binary' else CLASSES
    keep &=labels.isin(allowed)
    return np.flatnonzero(keep.to_numpy()),labels[keep].to_numpy(dtype=str),allowed


def grouped_splits(y,groups,max_folds=3):
    from sklearn.model_selection import StratifiedGroupKFold
    classes=set(y)
    support=min(len(set(groups[y==c])) for c in classes)
    for k in range(min(max_folds,support),1,-1):
        for offset in range(5):
            splitter=StratifiedGroupKFold(n_splits=k,shuffle=True,random_state=SEED+offset)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                folds=list(splitter.split(np.zeros((len(y),1)),y,groups))
            if all(set(y[tr])==classes and set(y[te])==classes and not set(groups[tr])&set(groups[te]) for tr,te in folds):
                return folds,{'folds':k,'seed':SEED+offset,'group_counts_by_class':{c:len(set(groups[y==c])) for c in sorted(classes)}}
    raise ValueError('Too few independent groups to put every class in training and test in each grouped fold')


def purge_overlapping_train(meta,y,folds):
    # Word-group folds can otherwise share raw EEG samples between adjacent
    # fixations. Purge only training events, keeping the same held-out words.
    subject=meta.subject.to_numpy();sentence=meta.sentence_position.astype(str).to_numpy()
    rate=np.where(meta.dataset.to_numpy()=='zuco',500.,1200.)
    onset=pd.to_numeric(meta.onset_sample_sentence_0based).to_numpy(dtype=float)*1000/rate
    result=[];removed=[]
    for train,test in folds:
        forbidden=defaultdict(list)
        for j in test:forbidden[(subject[j],sentence[j])].append(onset[j])
        forbidden={k:np.sort(v) for k,v in forbidden.items()}
        keep=[]
        for i in train:
            v=forbidden.get((subject[i],sentence[i]))
            if v is None:keep.append(i);continue
            pos=np.searchsorted(v,onset[i]);neighbors=v[max(0,pos-1):min(len(v),pos+1)]
            if not np.any(np.abs(neighbors-onset[i])<=1000):keep.append(i)
        keep=np.asarray(keep,dtype=int)
        if set(y[keep])!=set(y) or any((y[keep]==c).sum()<5 for c in set(y)):
            raise ValueError('Word-held-out folds have insufficient training coverage after purging overlapping EEG windows')
        removed.append(len(train)-len(keep));result.append((keep,test))
    return result,removed


def model_for(name,nclasses):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier
    model={'logistic':lambda:LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=SEED),
           'shrinkage_lda':lambda:LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(nclasses)/nclasses),
           'dummy':lambda:DummyClassifier(strategy='prior')}[name]()
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model)


def metrics(y,pred,classes,prob=None):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
    result={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=list(classes),average='macro',zero_division=0))}
    if len(classes)==2 and prob is not None:result['roc_auc']=float(roc_auc_score(np.asarray(y)=='positive',prob))
    return result


def evaluate(meta,X,y,classes,folds,model_name):
    from sklearn.metrics import confusion_matrix
    pred=np.full(len(y),'',dtype='<U12');prob=np.full(len(y),np.nan);fold_ids=np.full(len(y),-1)
    seen=np.zeros(len(y),bool);fit_warnings=[]
    for fi,(train,test) in enumerate(folds):
        if set(meta.iloc[train].event_id)&set(meta.iloc[test].event_id):raise ValueError('Same EEG event in train/test')
        model=model_for(model_name,len(classes))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X[train],y[train])
        fit_warnings.extend(str(w.message) for w in caught)
        pred[test]=model.predict(X[test]);fold_ids[test]=fi
        if len(classes)==2:prob[test]=model.predict_proba(X[test])[:,list(model.classes_).index('positive')]
        known=set(zip(meta.iloc[train].subject,meta.iloc[train].word_identity))
        seen[test]=[(a,b) in known for a,b in zip(meta.iloc[test].subject,meta.iloc[test].word_identity)]
    if (fold_ids<0).any() or (pred=='').any():raise ValueError('OOF predictions do not cover every eligible trial')
    score=metrics(y,pred,classes,prob if len(classes)==2 else None)
    per_reader=[]
    for reader in sorted(meta.subject.unique()):
        ix=np.flatnonzero(meta.subject.to_numpy()==reader)
        if set(y[ix])!=set(classes):continue
        per_reader.append({'subject':reader,'trials':len(ix),**metrics(y[ix],pred[ix],classes,prob[ix] if len(classes)==2 else None)})
    score.update(trials=len(y),readers=int(meta.subject.nunique()),readers_with_all_classes=len(per_reader),
                 equal_reader_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in per_reader])) if per_reader else None,
                 classes=list(classes),class_counts={c:int((y==c).sum()) for c in classes},fit_warnings=sorted(set(fit_warnings)))
    predictions=pd.DataFrame({'event_id':meta.event_id.to_numpy(),'subject':meta.subject.to_numpy(),'truth':y,'prediction':pred,
                              'fold':fold_ids,'same_word_same_reader_seen_in_train':seen})
    if len(classes)==2:predictions['probability_positive']=prob.astype(np.float32)
    diagnostic=None
    if seen.any() and set(y[seen])==set(classes):diagnostic={'trials':int(seen.sum()),**metrics(y[seen],pred[seen],classes,prob[seen] if len(classes)==2 else None)}
    return score,predictions,per_reader,confusion_matrix(y,pred,labels=list(classes)),diagnostic


def pearson_rows(X):
    z=X.astype(np.float64)-X.mean(axis=1,keepdims=True)
    norms=np.linalg.norm(z,axis=1);valid=norms>1e-12
    z[valid]/=norms[valid,None];z[~valid]=np.nan
    return z


def similarity_pairs(meta,X,y,max_pairs=300):
    """Within-reader correlations; distinct sentence texts prevent overlapping
    epochs and exact same-sentence repeats from being used as independent pairs."""
    rng=np.random.default_rng(SEED);z=pearson_rows(X);records=[];repeat=[]
    subjects=meta.subject.to_numpy();sent=meta.sentence_key.to_numpy();words=meta.word_identity.to_numpy();length=meta.word_length.to_numpy(dtype=int)
    for reader in sorted(set(subjects)):
        ix=np.flatnonzero((subjects==reader)&np.isfinite(z).all(axis=1))
        for ai,a in enumerate(CLASSES):
            for b in CLASSES[ai:]:
                left=ix[y[ix]==a];right=ix[y[ix]==b]
                if not len(left) or not len(right):continue
                seen=set();count=0
                for _ in range(max_pairs*30):
                    i=int(rng.choice(left));j=int(rng.choice(right));pair=tuple(sorted((i,j)))
                    if i==j or sent[i]==sent[j] or pair in seen:continue
                    seen.add(pair);count+=1
                    records.append(dict(subject=reader,class_a=a,class_b=b,event_a=meta.iloc[i].event_id,event_b=meta.iloc[j].event_id,
                                        correlation=float(z[i]@z[j]),same_word=bool(words[i]==words[j])))
                    if count>=max_pairs:break
        groups=defaultdict(list)
        for i in ix:groups[(words[i],y[i])].append(i)
        eligible=[v for v in groups.values() if len({sent[i] for i in v})>=2]
        seen=set()
        for _ in range(5000):
            if not eligible or len(seen)>=500:break
            group=eligible[int(rng.integers(len(eligible)))];i,j=map(int,rng.choice(group,2,replace=False));pair=tuple(sorted((i,j)))
            if sent[i]==sent[j] or pair in seen:continue
            # Same reader, same sentiment and similar word length, different
            # word and sentence. This remains a descriptive matched comparison.
            controls=ix[(y[ix]==y[i])&(words[ix]!=words[i])&(sent[ix]!=sent[i])&(np.abs(length[ix]-length[i])<=2)]
            if not len(controls):continue
            k=int(rng.choice(controls));seen.add(pair)
            repeat.append(dict(subject=reader,class_label=y[i],word=meta.iloc[i].word,event_a=meta.iloc[i].event_id,event_b=meta.iloc[j].event_id,
                               control_event=meta.iloc[k].event_id,same_word_r=float(z[i]@z[j]),different_word_r=float(z[i]@z[k])))
    return pd.DataFrame(records),pd.DataFrame(repeat)


def check_budget(out,limit):
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if used>limit:raise OSError(f'Analysis output budget exceeded ({used/1e6:.1f} MB); retained files are resumable, archive unchanged')


def run_task(meta,post,pre,nuisance,out):
    out.mkdir(parents=True,exist_ok=True)
    scores=[];skipped=[];readers=[];diagnostics=[];region_scores=[];split_specs=[]
    for policy in POLICIES:
        for target in ['binary','three_class']:
            ix,y,classes=selected(meta,policy,target)
            counts={c:int((y==c).sum()) for c in classes}
            if any(n<10 for n in counts.values()):
                skipped.append(dict(policy=policy,target=target,reason='Fewer than 10 eligible trials in a class',class_counts=counts));continue
            table=meta.iloc[ix].reset_index(drop=True)
            modes=['sentence','word','reader'] if policy=='word_agreement' else ['sentence']
            for mode in modes:
                groups=table[{'sentence':'sentence_key','word':'word_identity','reader':'subject'}[mode]].to_numpy(dtype=str)
                try:
                    folds,spec=grouped_splits(y,groups)
                    if mode=='word':folds,spec['purged_training_trials_per_fold']=purge_overlapping_train(table,y,folds)
                except ValueError as error:
                    skipped.append(dict(policy=policy,target=target,split=mode,reason=str(error),class_counts=counts));continue
                split_specs.append(dict(policy=policy,target=target,split=mode,**spec))
                variants=[('post','logistic'),('post','shrinkage_lda'),('post','dummy')] if policy=='word_agreement' else [('post','logistic')]
                if policy=='word_agreement' and mode=='sentence':variants +=[('pre','logistic'),('nuisance','logistic')]
                if policy=='word_agreement' and mode=='sentence' and target=='binary':variants +=[(region,'logistic') for region in REGIONS]
                for feature,model_name in variants:
                    case=f'{policy}__{target}__{mode}__{feature}__{model_name}'
                    path=out/'cases'/case;path.mkdir(parents=True,exist_ok=True)
                    mark=path/'result.json'
                    if mark.is_file():
                        r=json.loads(mark.read_text())
                        for name,sha in r['checksums'].items():
                            if file_digest(path/name)!=sha:raise ValueError('Corrupt completed analysis case')
                        score=r['score'];per_reader=r['per_reader'];diagnostic=r['diagnostic'];conf=np.array(r['confusion'])
                        print('KEEP '+case,flush=True)
                    else:
                        X=None
                        if feature=='post':X=post[ix]
                        elif feature=='pre':X=pre[ix]
                        elif feature=='nuisance':X=nuisance[ix]
                        if feature in REGIONS:X=post[ix,REGIONS.index(feature)*8:(REGIONS.index(feature)+1)*8]
                        print(f'FIT {case}: {len(ix)} trials / {len(folds)} folds',flush=True)
                        score,predictions,per_reader,conf,diagnostic=evaluate(table,X,y,classes,folds,model_name)
                        predictions.to_csv(path/'predictions.csv.gz',index=False,compression='gzip')
                        write_json(mark,dict(score=score,per_reader=per_reader,diagnostic=diagnostic,confusion=conf.tolist(),
                                             checksums={'predictions.csv.gz':file_digest(path/'predictions.csv.gz')}))
                    row=dict(policy=policy,target=target,split=mode,feature=feature,model=model_name,**score)
                    scores.append(row)
                    readers.extend(dict(policy=policy,target=target,split=mode,feature=feature,model=model_name,**r) for r in per_reader)
                    if policy=='word_agreement' and mode=='sentence' and feature=='post' and diagnostic:
                        diagnostics.append(dict(target=target,model=model_name,**diagnostic))
                    if feature in REGIONS:region_scores.append(row)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(readers).to_csv(out/'reader_scores.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'same_word_seen_diagnostic.csv',index=False)
    write_json(out/'skipped.json',skipped);write_json(out/'split_specs.json',split_specs)
    ix,y,_=selected(meta,'word_agreement','three_class')
    pairs,repeated=similarity_pairs(meta.iloc[ix].reset_index(drop=True),post[ix],y)
    pairs.to_csv(out/'class_similarity_pairs.csv.gz',index=False,compression='gzip')
    repeated.to_csv(out/'repeated_word_pairs.csv.gz',index=False,compression='gzip')
    plots(meta.iloc[ix].reset_index(drop=True),post[ix],y,pd.DataFrame(scores),pairs,repeated,out)
    reader_and_policy_plots(out)
    return scores,skipped


def plots(meta,X,y,scores,pairs,repeated,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    folder=out/'plots';folder.mkdir(exist_ok=True)
    colors=['#c43b45','#777777','#267d50']
    for target,classes in [('binary',('negative','positive')),('three_class',CLASSES)]:
        fig,axes=plt.subplots(3,3,figsize=(13,11))
        for ri,mode in enumerate(['sentence','word','reader']):
            for ci,model in enumerate(['logistic','shrinkage_lda','dummy']):
                path=out/'cases'/f'word_agreement__{target}__{mode}__post__{model}'/'result.json'
                ax=axes[ri,ci]
                if not path.is_file():ax.text(.5,.5,'Insufficient coverage',ha='center',transform=ax.transAxes);ax.axis('off');continue
                r=json.loads(path.read_text());cm=np.asarray(r['confusion'],float);cm/=cm.sum(axis=1,keepdims=True)
                ConfusionMatrixDisplay(cm,display_labels=classes).plot(ax=ax,colorbar=False,values_format='.2f',cmap='Blues')
                ax.set_title(f'{mode}-held-out / {model}\nBalanced accuracy {r["score"]["balanced_accuracy"]:.3f}',fontsize=9)
        fig.suptitle(f'{out.name}: {target} — pooled out-of-fold confusion, row proportions');fig.tight_layout();fig.savefig(folder/f'{target}_confusions.png',dpi=130);plt.close(fig)
    if len(scores):
        main=scores[(scores.policy=='word_agreement')&(scores.feature=='post')]
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for ax,target in zip(axes,['binary','three_class']):
            part=main[main.target==target];labels=[f'{r.split}\n{r.model}' for r in part.itertuples()]
            ax.bar(np.arange(len(part)),part.balanced_accuracy,color=([colors[0],colors[2],colors[1]]*3)[:len(part)])
            ax.set_xticks(np.arange(len(part)),labels,fontsize=7);ax.axhline(.5 if target=='binary' else 1/3,color='k',ls='--',lw=1)
            ax.set(ylim=(0,1),title=target,ylabel='Balanced accuracy (pooled OOF)')
        fig.tight_layout();fig.savefig(folder/'classification_overview.png',dpi=135);plt.close(fig)
        part=scores[(scores.policy=='word_agreement')&(scores.target=='binary')&(scores.split=='sentence')&(scores.model=='logistic')]
        fig,ax=plt.subplots(figsize=(11,4));ax.bar(part.feature,part.balanced_accuracy,color='#087f82');ax.axhline(.5,color='k',ls='--')
        ax.set(ylim=(0,1),ylabel='Balanced accuracy',title='Same sentence-held-out folds: post-onset, pre-onset, nuisance and six regions')
        ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(folder/'regions_and_controls.png',dpi=135);plt.close(fig)
    # Equal-reader aggregation; diagonal is different-trial same-class pairs,
    # not self-correlations of a trial with itself.
    grouped=pairs.groupby(['subject','class_a','class_b']).correlation.mean().reset_index() if len(pairs) else pd.DataFrame()
    mat=np.full((3,3),np.nan)
    if len(grouped):
        for (a,b),v in grouped.groupby(['class_a','class_b']):mat[CLASSES.index(a),CLASSES.index(b)]=mat[CLASSES.index(b),CLASSES.index(a)]=v.correlation.mean()
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    im=axes[0].imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');axes[0].set_xticks(range(3),CLASSES);axes[0].set_yticks(range(3),CLASSES)
    axes[0].set_title('Within-reader class-pair similarity\nEqual reader means; different sentences');fig.colorbar(im,ax=axes[0],shrink=.7)
    for a in range(3):
        for b in range(3):
            if np.isfinite(mat[a,b]):axes[0].text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center')
    if len(pairs):
        p=pairs.assign(pair_kind=np.where(pairs.class_a==pairs.class_b,'Same class','Different class')).groupby(['subject','pair_kind']).correlation.mean().unstack()
        for _,r in p.iterrows():axes[1].plot([0,1],[r.get('Same class',np.nan),r.get('Different class',np.nan)],'o-',alpha=.6)
    axes[1].set_xticks([0,1],['Same class','Different class']);axes[1].set(ylim=(-1,1),ylabel='Mean pattern correlation',title='One line per reader')
    if len(repeated):
        repeat_means=repeated.groupby('subject')[['same_word_r','different_word_r']].mean()
        repeat_means.to_csv(out/'repeated_word_reader_means.csv')
        for _,r in repeat_means.iterrows():axes[2].plot([0,1],r.to_numpy(),'o-',alpha=.6)
    axes[2].set_xticks([0,1],['Repeated word','Different word']);axes[2].set(ylim=(-1,1),title='Same reader/class, similar length\nDifferent sentence texts')
    fig.suptitle('Pearson correlation across 48 region × time-window features; descriptive');fig.tight_layout();fig.savefig(folder/'similarity_overview.png',dpi=135);plt.close(fig)
    # Small illustrative matrix, not an O(N²) matrix over all stored trials.
    rng=np.random.default_rng(SEED);chosen=[]
    for c in CLASSES:
        ix=np.flatnonzero(y==c);chosen.extend(rng.choice(ix,min(45,len(ix)),replace=False).tolist())
    if chosen:
        z=pearson_rows(X[chosen]);matrix=z@z.T;fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set(title='Illustrative trial correlation matrix\nUp to 45 trials/class, sorted negative / neutral / positive',xlabel='Trial',ylabel='Trial')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(folder/'sampled_trial_matrix.png',dpi=135);plt.close(fig)
    # Reproducibility of class-average feature patterns across readers.
    subjects=sorted(meta.subject.unique())
    if not subjects:return
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,c in zip(axes,CLASSES):
        means=np.stack([X[(meta.subject.to_numpy()==s)&(y==c)].mean(axis=0) if ((meta.subject.to_numpy()==s)&(y==c)).any() else np.full(X.shape[1],np.nan) for s in subjects])
        z=pearson_rows(means);ax.imshow(z@z.T,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(c);ax.set_xticks(range(len(subjects)),subjects,rotation=90,fontsize=7);ax.set_yticks(range(len(subjects)),subjects,fontsize=7)
    fig.suptitle('Between-reader correlation of class-average feature patterns (diagonal is self)');fig.tight_layout();fig.savefig(folder/'reader_reproducibility.png',dpi=130);plt.close(fig)


def reader_and_policy_plots(out):
    import matplotlib.pyplot as plt
    try:readers=pd.read_csv(out/'reader_scores.csv');scores=pd.read_csv(out/'scores.csv')
    except pd.errors.EmptyDataError:return
    if not len(readers):return
    part=readers[(readers.policy=='word_agreement')&(readers.feature=='post')]
    subjects=sorted(part.subject.unique())
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    columns=[mode+'/'+model for mode in ['sentence','word','reader'] for model in ['logistic','shrinkage_lda','dummy']]
    for ax,target in zip(axes,['binary','three_class']):
        sub=part[part.target==target].copy();sub['column']=sub['split']+'/'+sub.model
        table=sub.pivot(index='subject',columns='column',values='balanced_accuracy').reindex(index=subjects,columns=columns)
        im=ax.imshow(table.to_numpy(),vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_title(target)
        ax.set_xticks(range(len(columns)),columns,rotation=70,ha='right',fontsize=8);ax.set_yticks(range(len(subjects)),subjects)
        for r in range(len(subjects)):
            for c in range(len(columns)):
                v=table.iloc[r,c]
                if pd.notna(v):ax.text(c,r,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v<.65 else 'black')
    fig.suptitle('Reader-specific balanced accuracy: one reader per row; blank = missing class/coverage')
    fig.tight_layout();fig.savefig(out/'plots/reader_score_matrix.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,target in zip(axes,['binary','three_class']):
        sub=scores[(scores.target==target)&(scores['split']=='sentence')&(scores.feature=='post')&(scores.model=='logistic')].set_index('policy').reindex(POLICIES)
        ax.bar(range(len(POLICIES)),sub.balanced_accuracy,color='#087f82');ax.axhline(.5 if target=='binary' else 1/3,color='k',ls='--')
        ax.set_xticks(range(len(POLICIES)),POLICIES,rotation=25,ha='right',fontsize=8);ax.set(ylim=(0,1),title=target,ylabel='Sentence-held-out balanced accuracy')
    fig.suptitle('Label policies use different targets and trial pools; blanks indicate insufficient coverage')
    fig.tight_layout();fig.savefig(out/'plots/label_policy_comparison.png',dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--max-output-mb',type=int,default=400);args=p.parse_args()
    archive=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz'
    out=args.thesis_root/'Results/word_sentiment_eeg/classification_similarity'
    records,archive_protocol=load_completed(archive);out.mkdir(parents=True,exist_ok=True)
    protocol={'version':VERSION,'feature_spec':FEATURE_SPEC,'seed':SEED,'max_folds':3,'model_settings':'C1 balanced logistic; auto-shrinkage LDA uniform priors; prior dummy',
              'archive_manifest_sha256':file_digest(archive/'manifest.json'),'archive_protocol_sha256':digest(archive_protocol),
              'code_sha256':file_digest(Path(__file__)),'control_reading':'included','function_words':'primary includes; four auxiliary policies require known content grammar',
              'neutral_downsampling':False,'amplitude_rule':'exclude native-rate 40Hz peak >100 assumed µV, recorded in archive',
              'no_raw_source_read':True,'max_output_mb':args.max_output_mb}
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Analysis configuration/code/archive changed. Reconcile existing outputs before rerunning; no silent mixing.')
    write_json(out/'protocol.json',protocol)
    def state(stage,done,status='incomplete'):
        write_json(out/'manifest.json',dict(status=status,stage=stage,tasks_complete=done,expected_tasks=4,updated_at_utc=datetime.now(timezone.utc).isoformat()))
    state('features',[]);by_task=defaultdict(list);label_sources=[]
    for record in records:
        meta,post,pre,nuisance,label_source=reader_features(archive,record,out,args.thesis_root)
        by_task[(record['dataset'],record['task'])].append((meta,post,pre,nuisance))
        if label_source and label_source not in label_sources:label_sources.append(label_source)
        check_budget(out,args.max_output_mb*1000000)
    if (out/'label_sources.json').is_file() and json.loads((out/'label_sources.json').read_text())!=label_sources:
        raise ValueError('Label snapshot changed; reconcile prior analysis before reuse. Feature caches remain usable.')
    write_json(out/'label_sources.json',label_sources)
    all_scores=[];all_skips=[];done=[];coverage=[];quality=[]
    for (dataset,task),parts in sorted(by_task.items()):
        meta=pd.concat([x[0] for x in parts],ignore_index=True);post=np.concatenate([x[1] for x in parts]);pre=np.concatenate([x[2] for x in parts]);nuisance=np.concatenate([x[3] for x in parts])
        if meta.event_id.duplicated().any():raise ValueError('Repeated input trial across reader archives')
        physical=meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False)
        identical=meta.signal_sha256.duplicated(keep=False)
        duplicate=physical|identical
        for reader in sorted(meta.subject.unique()):
            local=meta.subject.eq(reader)
            quality.append(dict(dataset=dataset,task=task,subject=reader,archived_trials=int(local.sum()),
                                amplitude_over_100=int((local&boolean(meta.amplitude_over_100_uV)).sum()),
                                duplicate_signal_or_event=int((local&duplicate).sum()),
                                unknown_grammar=int((local&~meta.grammatical_class.isin(['content','function'])).sum())))
        if duplicate.any():
            folder=out/dataset/task;folder.mkdir(parents=True,exist_ok=True)
            meta.loc[duplicate].to_csv(folder/'excluded_duplicate_signals.csv.gz',index=False,compression='gzip')
            print(f'QC {dataset}/{task}: exclude all {int(duplicate.sum())} members of exact duplicate signal/event groups',flush=True)
            retain=~duplicate.to_numpy();meta=meta.loc[retain].reset_index(drop=True);post=post[retain];pre=pre[retain];nuisance=nuisance[retain]
        for policy in POLICIES:
            for target in ['binary','three_class']:
                ix,y,classes=selected(meta,policy,target)
                for reader in ['ALL_READERS',*sorted(meta.subject.unique())]:
                    local=np.ones(len(ix),bool) if reader=='ALL_READERS' else meta.iloc[ix].subject.to_numpy()==reader
                    for c in classes:coverage.append(dict(dataset=dataset,task=task,policy=policy,target=target,subject=reader,class_label=c,trials=int(((y==c)&local).sum()),
                                                          unique_source_occurrences=int(meta.iloc[ix].loc[(y==c)&local,['sentence_key','word_index']].drop_duplicates().shape[0])))
        folder=out/dataset/task;scores,skips=run_task(meta,post,pre,nuisance,folder)
        all_scores.extend(dict(dataset=dataset,task=task,**r) for r in scores);all_skips.extend(dict(dataset=dataset,task=task,**r) for r in skips)
        done.append(dataset+'/'+task);state('analysis',done);check_budget(out,args.max_output_mb*1000000)
    pd.DataFrame(quality).to_csv(out/'quality_counts.csv',index=False)
    pd.DataFrame(all_scores).to_csv(out/'scores.csv',index=False);pd.DataFrame(coverage).to_csv(out/'input_counts.csv',index=False);write_json(out/'skipped.json',all_skips)
    (out/'REPORT.md').write_text(REPORT)
    state('finished',done,'completed');print('ANALYSIS COMPLETE:',out,flush=True)


REPORT='''# Fixed word-EEG classification and similarity analyses

This uses the completed 40Hz /125Hz trial archive, not historical30Hz means.
No source downloads, extraction, new labels, neural networks or tuning sweep.
Main input: 48 features, six scalp-region means × eight100ms post-onset windows.
Pre-onset control: six regions × two100ms windows. Nuisance control: word length,
relative word position, sentence length, first-fixation duration, next recovered
word-first interval and control status. These are separate predictors, not mixed
into the main EEG classifier. The source archive remains intact for other features.

Primary population: lexical and contextual-word labels agree, alignment has no
warning, finite archived EEG, native40Hz peak <=100 assumed µV. Function words
and control reading are retained here. Neutral is not downsampled.
Four auxiliary content-word policies: all three agree, lexical only, contextual
word only, sentence only. Unknown grammar/labels do not become invented labels;
missing or sparse classes/group coverage are reported in skipped.json.

Main models: balanced regularized logistic regression (C=1), shrinkage LDA
(auto covariance shrinkage, uniform class priors), and a prior-only dummy.
Auxiliary policies use the fixed logistic model. Standardization and median
imputation fit on training folds only. No word identity is an EEG predictor.
Three grouped folds by default, reduced to two only for feasibility. A partition
seed is chosen solely for valid class coverage, never for better accuracy.

Sentence-held-out: all readers/repeats of a sentence stay together; words and
readers can recur across folds. Word-held-out: a normalized word identity stays
together; overlapping training EEG windows from the same reader/source sentence
are purged. Readers and sentence context can still recur; a word comparison may
be skipped if purging leaves too few training cases. Reader-held-out: readers stay
together, while texts/words can recur. These answer different questions; none
is claimed to hold out both readers, words and sentences simultaneously.
Binarypositive/negative is primary; three-class neutral is secondary.
Scores: balanced accuracy, macroF1, accuracy, binaryAUC; pooled out-of-fold and
per-reader scores. Chance balanced accuracy is .5 binary and1/3 three-class.
Fold dependence and correlated word/reader events preclude treating folds or
all trials as independent statistical replicates. No significance claim is made.

Pre-onset and nuisance models diagnose baseline/class confounding; pre-onset
performance is not automatically proof of leakage (anticipation/context/overlap
can contribute). Region-specific binary logistic models share sentence folds;
these exploratory comparisons are not independent tests or a selected winner.

Same-word/same-reader diagnostic uses the sentence-held-out predictions only
for test occurrences whose word was present for the same reader in training.
The trained model is pooled across readers, not a reader-only classifier.
Never the same event, never the same sentence text across those folds. This
is a seen-word diagnostic, separate from unseen-word generalization.

Correlation is Pearson similarity across48 region/time-window features, not
spectral coherence or direct sample-by-sample full-waveform correlation.
Class-pair comparisons sample bounded numbers within each reader and use
separate sentence texts (no self/overlapping same-sentence pairs). Correlation
matrix diagonal is same-class DIFFERENT-trial pairs; illustrative trial matrix
and reader reproducibility matrix have ordinary self-diagonals. Group summaries
weight readers equally. Repeated-word pairs share reader/class and use different
sentence texts; different-word controls match reader/class and word length±2.
They do not fully control word frequency, meaning, sentence context or timing.
Pair counts are not independent subject counts. These are descriptive analyses.

All outputs are private under classification_similarity; small feature caches
allow restarts without rereading EEG shards. Predictions and case results are
checkpointed. Completion requires four dataset/task analyses; individual cases
may be explicitly skipped for inadequate coverage. A skip is never a zero score.
Exact duplicated waveforms or duplicated physical-event groups are excluded in full
from analysis and reported separately; the archive remains unchanged.
No saved class means or trial-archive files are modified. An initial read of the
archive can take time; subsequent reruns reuse validated compact feature caches.

Methods references:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
https://scikit-learn.org/stable/common_pitfalls.html
'''

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
