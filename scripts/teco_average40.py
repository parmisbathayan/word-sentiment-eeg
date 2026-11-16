#!/usr/bin/env python3
"""CPU-only TeCo binary averaging follow-up. Read existing feature caches only."""
import argparse
import io
import json
from pathlib import Path
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import analyze_trial_archive as a
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

SPEC = dict(version=1, dataset='teco', task='task1', target='positive versus negative',
            policy='word_agreement', sizes=[1,40], seed=2026, folds=3,
            features='48 cached voltages: six regions x eight 100-ms bins, 0–800ms',
            averaging='within reader/class and train/test partition; retain smaller remainders',
            model='C1 balanced logistic; train-only median imputation and scaling',
            controls='dummy; pre-onset logistic and word/fixation-property logistic on max40',
            budget=10_000_000, no_eeg_reads=True)


def read(path):
    return json.loads(Path(path).read_text())


def csv(store, relative, frame):
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, compression={'method':'gzip','mtime':0} if relative.endswith('.gz') else None)
    store.bytes(relative, buffer.getvalue())


def load_cached(thesis):
    source = thesis/'Results/word_sentiment_eeg/classification_similarity'
    archive = thesis/'Results/word_sentiment_eeg/trial_archive_125hz'
    if read(source/'manifest.json').get('status') != 'completed':
        raise ValueError('The original classification feature preparation must be completed first.')
    records, _ = a.load_completed(archive)
    records = sorted([r for r in records if r['dataset']=='teco' and r['task']=='task1'], key=lambda r:r['subject'])
    if len(records)!=12:raise ValueError('Expected 12 TeCo task1 recordings')
    # Preflight ALL caches before calling the helper, whose missing-cache branch
    # would otherwise create features. This notebook must never enter that branch.
    for record in records:
        cache=source/'features/teco/task1'/record['subject']
        if not (cache/'manifest.json').is_file():
            raise FileNotFoundError(f'Missing saved cache: {cache}. No extraction/feature preparation started.')
    known={v['sha256'] for v in read(source/'label_sources.json')}
    tables=[];arrays=[[],[],[]];identities=[]
    for i,record in enumerate(records,1):
        print(f'LOAD SAVED FEATURES {i}/{len(records)}: TeCo/{record["subject"]}',flush=True)
        meta,post,pre,nuisance,label=a.reader_features(archive,record,source,thesis)
        if not label or label['sha256'] not in known:raise ValueError('Labels differ from completed original run')
        if not meta.dataset.eq('teco').all() or not meta.task.eq('task1').all() or not meta.subject.eq(record['subject']).all():
            raise ValueError('Reader/task metadata mismatch')
        tables.append(meta)
        for dst,x in zip(arrays,[post,pre,nuisance]):dst.append(x)
        cache=source/'features/teco/task1'/record['subject']
        identities.append(dict(reader=record['subject'],cache=a.file_digest(cache/'manifest.json'),labels=label))
    meta=pd.concat(tables,ignore_index=True);post,pre,nuisance=[np.concatenate(v) for v in arrays]
    if meta.event_id.duplicated().any():raise ValueError('Duplicate event IDs')
    duplicate=meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False)
    duplicate |= meta.signal_sha256.duplicated(keep=False)
    good=~duplicate.to_numpy();meta=meta.loc[good].reset_index(drop=True)
    post,pre,nuisance=[x[good] for x in [post,pre,nuisance]]
    ix,y,_=a.selected(meta,'word_agreement','binary')
    meta=meta.iloc[ix].reset_index(drop=True);meta['truth']=y
    if post.shape[1]!=48 or pre.shape[1]!=12 or nuisance.shape[1]!=6:raise ValueError('Unexpected cached feature layout')
    if len(set(y))!=2 or min(np.sum(y==c) for c in set(y))<10:raise ValueError('Too few binary trials')
    return meta,dict(post=post[ix],pre=pre[ix],nuisance=nuisance[ix]),dict(
        sources=identities,source_protocol=a.file_digest(source/'protocol.json'),
        excluded_duplicate_observations=int(duplicate.sum()),selected_trials=len(meta))


def groups(meta, indices, maximum, seed):
    """Every selected event enters one group, with no mixed reader or class."""
    rng=np.random.default_rng(seed);result=[]
    for _,g in meta.iloc[indices].groupby(['subject','truth'],sort=True):
        order=g.index.to_numpy().copy()
        if maximum>1:rng.shuffle(order)
        result.extend(order[j:j+maximum] for j in range(0,len(order),maximum))
    flat=np.concatenate(result)
    if len(flat)!=len(set(flat)) or set(flat)!=set(indices):raise ValueError('Missing/repeated averaging member')
    return result


def average(X, meta, membership):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        values=np.stack([np.nanmean(X[g],axis=0) for g in membership])
    y=np.array([meta.truth.iloc[g[0]] for g in membership])
    if not all(meta.truth.iloc[g].nunique()==1 and meta.subject.iloc[g].nunique()==1 for g in membership):
        raise ValueError('Mixed reader/class average')
    return values,y


def analyze(meta,features,store):
    from sklearn.metrics import confusion_matrix
    rows=[];skips=[]
    for mode,column in [('sentence','sentence_key'),('word','word_identity'),('reader','subject')]:
        try:
            folds,spec=a.grouped_splits(meta.truth.to_numpy(),meta[column].astype(str).to_numpy())
            if mode=='word':folds,spec['purged_training_trials']=a.purge_overlapping_train(meta,meta.truth.to_numpy(),folds)
        except ValueError as error:
            skips.append(dict(split=mode,reason=str(error)));continue
        store.json(f'splits/{mode}.json',dict(spec=spec,folds=[dict(train_event_ids=meta.event_id.iloc[tr].tolist(),
            test_event_ids=meta.event_id.iloc[te].tolist()) for tr,te in folds]))
        cases=[(n,'post',m) for n in [1,40] for m in ['logistic','dummy']]+[(40,f,'logistic') for f in ['pre','nuisance']]
        for maximum,feature,model_name in cases:
            name=f'{mode}__max{maximum}__{feature}__{model_name}';relative=f'cases/{name}'
            marker=store.root/relative/'result.json'
            if marker.is_file():
                saved=read(marker)
                if any(a.file_digest(store.root/k)!=sha for k,sha in saved['checksums'].items()):
                    raise ValueError('Saved case checksum mismatch')
                print('KEEP COMPLETED CASE:',name,flush=True);rows.append(saved['score']);continue
            predictions=[];members=[];fold_rows=[];fit_warnings=[]
            print(f'FIT {name}: {len(folds)} folds',flush=True)
            for fi,(train,test) in enumerate(folds):
                if set(meta[column].iloc[train])&set(meta[column].iloc[test]):raise ValueError('Holdout leakage')
                if set(meta.event_id.iloc[train])&set(meta.event_id.iloc[test]):raise ValueError('Event leakage')
                # Same memberships for every feature/model within a size/fold.
                gt=groups(meta,train,maximum,SPEC['seed']+100*fi)
                ge=groups(meta,test,maximum,SPEC['seed']+100*fi+1)
                xt,yt=average(features[feature],meta,gt);xe,ye=average(features[feature],meta,ge)
                if set(yt)!=set(ye) or len(set(yt))!=2:raise ValueError('Insufficient averaged class coverage')
                model=a.model_for(model_name,2)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(xt,yt)
                fit_warnings.extend(str(w.message) for w in caught)
                pred=model.predict(xe);prob=model.predict_proba(xe)[:,list(model.classes_).index('positive')]
                fold_rows.append(dict(fold=fi,**a.metrics(ye,pred,('negative','positive'),prob),
                    train_examples=len(gt),test_examples=len(ge)))
                for partition,gs in [('train',gt),('test',ge)]:
                    for j,g in enumerate(gs):
                        members.append(dict(fold=fi,partition=partition,example=j,reader=str(meta.subject.iloc[g[0]]),
                            truth=str(meta.truth.iloc[g[0]]),actual_trials=len(g),event_ids=json.dumps(meta.event_id.iloc[g].tolist())))
                for j,g in enumerate(ge):predictions.append(dict(fold=fi,example=j,reader=str(meta.subject.iloc[g[0]]),
                    truth=str(ye[j]),prediction=str(pred[j]),probability_positive=float(prob[j]),actual_trials=len(g)))
                print(f'  FOLD {fi+1}/{len(folds)}: {len(gt)} training averages, {len(ge)} test averages',flush=True)
            p=pd.DataFrame(predictions)
            score=dict(split=mode,maximum_trials=maximum,feature=feature,model=model_name,
                **a.metrics(p.truth.to_numpy(),p.prediction.to_numpy(),('negative','positive'),p.probability_positive.to_numpy()),
                test_examples=len(p),original_test_trials=int(p.actual_trials.sum()),
                actual_mean_trials=float(p.actual_trials.mean()),actual_min_trials=int(p.actual_trials.min()),
                full40_test_examples=int((p.actual_trials==40).sum()),folds=len(folds))
            csv(store,relative+'/predictions.csv.gz',p);csv(store,relative+'/memberships.csv.gz',pd.DataFrame(members))
            reader_rows=[]
            for subject,g in p.groupby('reader'):
                reader_rows.append(dict(reader=str(subject),examples=len(g),both_classes=g.truth.nunique()==2,
                    balanced_accuracy=float(a.metrics(g.truth.to_numpy(),g.prediction.to_numpy(),('negative','positive'))['balanced_accuracy']) if g.truth.nunique()==2 else None))
            paths=[relative+'/'+n for n in ['predictions.csv.gz','memberships.csv.gz']]
            store.json(relative+'/result.json',dict(score=score,fold_scores=fold_rows,per_reader=reader_rows,
                fit_warnings=fit_warnings,confusion=confusion_matrix(p.truth,p.prediction,labels=['negative','positive']).tolist(),
                checksums={k:a.file_digest(store.root/k) for k in paths}))
            rows.append(score)
    return rows,skips


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--thesis-root',required=True,type=Path)
    parser.add_argument('--free-drive-bytes',required=True,type=int);args=parser.parse_args()
    out=args.thesis_root/'Results/word_sentiment_eeg/teco_average40'
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    if args.free_drive_bytes<max(0,SPEC['budget']-used):raise OSError('Reserve at most 10 MB for new results. No EEG read.')
    import fcntl,tempfile,os
    lock=Path(tempfile.gettempdir(), 'teco-average40-'+a.digest(str(out))[:16]+'.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta,features,sources=load_cached(args.thesis_root)
    store=ImmutableStore(out,SPEC['budget'])
    store.json('protocol.json',dict(spec=SPEC,sources=sources,code=a.file_digest(Path(__file__)),helper=a.file_digest(Path(a.__file__)),
        software={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']}))
    counts=meta.groupby('truth').agg(reader_trials=('event_id','size'),readers=('subject','nunique')).reset_index()
    csv(store,'input_counts.csv',counts)
    rows,skips=analyze(meta,features,store)
    csv(store,'scores.csv',pd.DataFrame(rows));store.json('skipped.json',skips)
    paths=['scores.csv','input_counts.csv','protocol.json','skipped.json']+[str(p.relative_to(out)) for p in (out/'cases').glob('*/result.json')]
    store.json('complete.json',dict(status='completed' if len(rows)==18 and not skips else 'incomplete',cases=len(rows),
        checksums={k:a.file_digest(out/k) for k in paths}))
    print('TECO AVERAGING:',len(rows),'/18 cases.',out,flush=True)
    if len(rows)!=18 or skips:raise RuntimeError('Some holdouts were unavailable; see skipped.json. Completed cases retained.')


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
