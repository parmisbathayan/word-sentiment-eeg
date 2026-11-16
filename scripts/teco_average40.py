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


def frozen_roster(source):
    case=source/'teco/task1/cases/word_agreement__binary__sentence__post__logistic'
    marker=read(case/'result.json');path=case/'predictions.csv.gz'
    expected=marker['checksums']['predictions.csv.gz']
    if a.file_digest(path)!=expected:raise ValueError('Original TeCo trial/label roster checksum mismatch')
    roster=pd.read_csv(path,keep_default_na=False,dtype={'event_id':str,'subject':str})
    if roster.event_id.duplicated().any() or set(roster.truth)!=set(['negative','positive']):
        raise ValueError('Invalid saved original binary trial/label roster')
    score=marker['score']
    if len(roster)!=score['trials'] or roster.subject.nunique()!=score['readers']:
        raise ValueError('Saved original roster/count mismatch')
    counts={c:int(roster.truth.eq(c).sum()) for c in ['negative','positive']}
    if counts!=score['class_counts']:raise ValueError('Saved original label/count mismatch')
    return roster[['event_id','subject','truth']],dict(path=str(path),sha256=expected,
        result_sha256=a.file_digest(case/'result.json'),class_counts=counts)


def cached_features_only(archive,source,record):
    """Read-only equivalent of the old cache branch, without any live label join."""
    subject=record['subject'];reader=archive/'teco/readers/task1'/subject
    checkpoint_path=reader/'checkpoint.json';checkpoint=read(checkpoint_path)
    if checkpoint['status']!='completed' or checkpoint['epochs']!=record['epochs']:
        raise ValueError('Incomplete/inconsistent reader checkpoint')
    if checkpoint['identity']['protocol_sha256']!=a.digest(read(archive/'protocol.json')):
        raise ValueError('Reader/archive protocol mismatch')
    if a.file_digest(reader/'events.csv.gz')!=checkpoint['events_sha256']:
        raise ValueError('Reader event table checksum mismatch')
    cache=source/'features/teco/task1'/subject;saved=read(cache/'manifest.json')
    signature=a.digest({'spec':a.FEATURE_SPEC,'checkpoint':a.file_digest(checkpoint_path)})
    if saved['signature']!=signature:raise ValueError('Feature cache differs from archive/specification')
    if not {'features.npz','events.csv.gz'}<=set(saved['checksums']):raise ValueError('Incomplete cache checksums')
    for name,sha in saved['checksums'].items():
        if a.file_digest(a.strict_path(cache,name))!=sha:raise ValueError('Feature cache checksum mismatch')
    with np.load(cache/'features.npz',allow_pickle=False) as z:
        post=z['post'].copy();pre=z['pre'].copy();ids=z['event_id'].astype(str).tolist()
    meta=pd.read_csv(cache/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str,'subject':str})
    if ids!=meta.event_id.tolist() or meta.event_id.duplicated().any() or len(meta)!=record['epochs']:
        raise ValueError('Feature cache identity mismatch')
    if post.shape!=(len(meta),48) or pre.shape!=(len(meta),12):raise ValueError('Unexpected cached feature layout')
    if not meta.dataset.eq('teco').all() or not meta.task.eq('task1').all() or not meta.subject.eq(subject).all():
        raise ValueError('Reader/task metadata mismatch')
    meta['control_numeric']=np.where(meta.is_control.astype(str).str.lower().isin(['true','1']),1.,
        np.where(meta.is_control.astype(str).str.lower().isin(['false','0']),0.,np.nan))
    nuisance=np.stack([pd.to_numeric(meta[c],errors='coerce').to_numpy(dtype=float) for c in a.NUISANCE],axis=1)
    return meta,post,pre,nuisance,dict(reader=subject,cache=a.file_digest(cache/'manifest.json'))


def load_cached(thesis):
    source=thesis/'Results/word_sentiment_eeg/classification_similarity'
    archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz'
    if read(source/'manifest.json').get('status')!='completed':
        raise ValueError('The original classification feature preparation must be completed first.')
    roster,label_source=frozen_roster(source)
    records,_=a.load_completed(archive)
    records=sorted([r for r in records if r['dataset']=='teco' and r['task']=='task1'],key=lambda r:r['subject'])
    if len(records)!=12:raise ValueError('Expected 12 TeCo task1 recordings')
    if set(roster.subject)!={r['subject'] for r in records}:raise ValueError('Original roster/reader mismatch')
    tables=[];arrays=[[],[],[]];identities=[]
    for i,record in enumerate(records,1):
        print(f'LOAD SAVED FEATURES {i}/{len(records)}: TeCo/{record["subject"]}',flush=True)
        meta,post,pre,nuisance,identity=cached_features_only(archive,source,record)
        tables.append(meta)
        for dst,x in zip(arrays,[post,pre,nuisance]):dst.append(x)
        identities.append(identity)
    meta=pd.concat(tables,ignore_index=True);post,pre,nuisance=[np.concatenate(v) for v in arrays]
    if meta.event_id.duplicated().any():raise ValueError('Duplicate event IDs')
    index=pd.Index(meta.event_id).get_indexer(roster.event_id)
    if (index<0).any():raise ValueError('Saved original selected trials missing from caches')
    meta=meta.iloc[index].reset_index(drop=True)
    if meta.subject.tolist()!=roster.subject.tolist():raise ValueError('Saved original event/reader mismatch')
    if meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False).any() or meta.signal_sha256.duplicated(keep=False).any():
        raise ValueError('Original selected roster contains duplicate signal/events')
    meta['truth']=roster.truth.to_numpy()
    print(f'RESTORED ORIGINAL LABELS: {len(meta)} trials, {label_source["class_counts"]}; no annotation CSV read.',flush=True)
    return meta,dict(post=post[index],pre=pre[index],nuisance=nuisance[index]),dict(
        sources=identities,source_protocol=a.file_digest(source/'protocol.json'),
        frozen_labels=label_source,selection='exact original completed binary case; no current annotation lookup',selected_trials=len(meta))


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
