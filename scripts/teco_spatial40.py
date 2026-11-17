#!/usr/bin/env python3
"""ZuCo-inspired TeCo reader-held-out spatial experiment; saved epochs only."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from importlib.metadata import version
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import teco_average40 as old
import analyze_trial_archive as a
import zuco_recovered_followups as z
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

SPEC=dict(version=1,dataset='TeCo task1',target='original frozen positive/negative word-agreement roster',
    seed=20261113,outer='three reader-held-out folds; words/sentences may overlap',sizes=[1,40],regroupings40=5,
    cohort='at least40 original selected events per class per reader; fixed before any fits',
    post_window_ms=[224,304],pre_window_ms=[-88,-8],mean_endpoints='inclusive, 11 samples at125Hz',
    averaging='exact groups within reader/class/partition; discard remainders',
    model='C1 balanced liblinear logistic; training-only imputation and scaling',
    regions=list(a.REGIONS),selected_region='choose by inner reader-grouped CV using outer training readers only',
    baseline='original48regional0–800ms features on identical cohort/folds/groups',
    limitations=['first stored fixation, not all word fixations','no additional paper regression cleaning or trial matching',
                 'test labels form class averages','fixed region results exploratory; no significance tests'],budget=20_000_000)


def read(path):return json.loads(Path(path).read_text())


def csv(store,name,frame):old.csv(store,name,frame)


def cohort(meta,features):
    counts=meta.groupby(['subject','truth']).size().unstack(fill_value=0).reindex(columns=['negative','positive'],fill_value=0)
    counts['included']=counts.min(axis=1)>=40
    counts['reason']=np.where(counts.included,'eligible_before_fitting','fewer_than40_in_a_class')
    keep=meta.subject.isin(counts.index[counts.included]).to_numpy()
    if counts.included.sum()<6:raise ValueError('Fewer than6 readers have40 trials/class. Exact40 reader evaluation cannot proceed reliably; no partial-group fallback. Counts: '+counts.to_string())
    return meta.loc[keep].reset_index(drop=True),{k:v[keep] for k,v in features.items()},counts.reset_index()


def cached_channel_means(archive,meta,montage,store):
    post=np.empty((len(meta),len(montage)),dtype=np.float32);pre=np.empty_like(post)
    times=np.arange(-25,101)*8;postmask=(times>=224)&(times<=304);premask=(times>=-88)&(times<=-8)
    if postmask.sum()!=11 or premask.sum()!=11:raise AssertionError('Unexpected window grid')
    subjects=sorted(meta.subject.unique());channels=montage.channel.astype(str).tolist()
    for ri,subject in enumerate(subjects,1):
        rows=meta[meta.subject==subject];reader=archive/'teco/readers/task1'/subject
        checkpoint=read(reader/'checkpoint.json')
        identity=dict(reader_checkpoint=a.file_digest(reader/'checkpoint.json'),event_ids=rows.event_id.tolist(),
                      post=SPEC['post_window_ms'],pre=SPEC['pre_window_ms'],montage=a.file_digest(archive/'teco_channel_regions.csv'))
        prefix=f'channel_features/{subject}';marker=store.root/prefix/'complete.json';path=store.root/prefix/'means.npz'
        if marker.is_file():
            saved=read(marker)
            if saved['identity']!=identity or a.file_digest(path)!=saved['sha256']:raise ValueError('Changed/corrupt compact channel cache')
            with np.load(path,allow_pickle=False) as f:
                if f['event_id'].astype(str).tolist()!=rows.event_id.tolist() or f['post'].shape!=(len(rows),len(channels)) or f['pre'].shape!=(len(rows),len(channels)):
                    raise ValueError('Compact channel cache identity/shape mismatch')
                post[rows.index]=f['post'];pre[rows.index]=f['pre']
            print(f'KEEP CHANNEL MEANS {ri}/{len(subjects)}: {subject}; no epoch reads',flush=True);continue
        print(f'PREPARE CHANNEL MEANS {ri}/{len(subjects)}: {subject}; {len(rows)} selected epochs',flush=True)
        checks={k:v for b in checkpoint['blocks'] for k,v in b['checksums'].items()}
        localpost=np.empty((len(rows),len(channels)),np.float32);localpre=np.empty_like(localpost)
        local_positions={int(index):j for j,index in enumerate(rows.index)};seen=set()
        files=list(rows.groupby('epoch_file',sort=False))
        for fi,(relative,g) in enumerate(files,1):
            path=a.strict_path(reader,relative);payload=path.read_bytes()
            import hashlib
            if hashlib.sha256(payload).hexdigest()!=checks[relative]:raise ValueError('Corrupt saved EEG shard')
            with np.load(BytesIO(payload),allow_pickle=False) as f:
                positions=g.epoch_row.astype(int).to_numpy();ids=f['event_id'].astype(str)
                if (positions<0).any() or (positions>=len(ids)).any() or ids[positions].tolist()!=g.event_id.tolist():raise ValueError('Saved EEG/event mismatch')
                if f['channel_names'].astype(str).tolist()!=channels or not np.array_equal(f['times_ms'],times):raise ValueError('Montage/time mismatch')
                eeg=f['eeg_uV'][positions]
                if eeg.shape!=(len(g),len(channels),126) or not np.isfinite(eeg).all():raise ValueError('Invalid saved EEG')
                hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in eeg]
                if hashes!=g.signal_sha256.tolist():raise ValueError('EEG differs from validated original cache')
                eeg=eeg.astype(np.float32)
                dst=[local_positions[int(i)] for i in g.index]
                localpost[dst]=eeg[:,:,postmask].mean(axis=2);localpre[dst]=eeg[:,:,premask].mean(axis=2)
                if seen.intersection(dst):raise ValueError('Repeated saved epoch')
                seen.update(dst)
            if fi==1 or fi%20==0 or fi==len(files):print(f'  {subject}: {fi}/{len(files)} saved shards',flush=True)
        if len(seen)!=len(rows):raise ValueError('Missing channel-feature events')
        buffer=BytesIO();np.savez_compressed(buffer,post=localpost,pre=localpre,event_id=rows.event_id.to_numpy(dtype=str))
        store.bytes(prefix+'/means.npz',buffer.getvalue());cachepath=store.root/prefix/'means.npz'
        store.json(prefix+'/complete.json',dict(identity=identity,sha256=a.file_digest(cachepath)))
        post[rows.index]=localpost;pre[rows.index]=localpre
        print(f'SAVED CHANNEL MEANS {ri}/{len(subjects)}: {subject}; reusable after disconnect',flush=True)
    return post,pre


def spatial_features(post,pre,baseline,montage):
    result={'whole_head':post,'cached_0_800':baseline['post'],'pre_control':pre,'word_eye_control':baseline['nuisance']}
    for region in a.REGIONS:
        picks=np.flatnonzero(montage.region.astype(str).to_numpy()==region)
        if not len(picks):raise ValueError('Empty predefined region: '+region)
        result['region_'+region]=post[:,picks]
    result['six_region_means']=np.stack([result['region_'+r].mean(axis=1) for r in a.REGIONS],axis=1)
    result['dummy']=post
    return result


def memberships(meta,y,indices,size,seed):
    # Reuse the exact-size averaging implementation from the ZuCo follow-up.
    _,_,_,flat=z.averaged(np.zeros((len(meta),1),np.float32),y,meta,indices,size,seed)
    group=np.asarray(flat,dtype=int).reshape(-1,size)
    if group.size==0:raise ValueError('No complete groups')
    if len(np.unique(group))!=group.size or not set(group.ravel())<=set(indices):raise ValueError('Invalid averaging membership')
    for g in group:
        if len(set(y[g]))!=1 or meta.subject.iloc[g].nunique()!=1:raise ValueError('Mixed class/reader average')
    return group


def averaged(X,groups):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(X[groups],axis=1)


def estimator(dummy=False):
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if dummy:return DummyClassifier(strategy='prior')
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
        LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SPEC['seed']))


def select_region(meta,y,features,outer_train,size,repeat_seed):
    # No outer-test data, labels, reader or scores are used for region choice.
    inner=z.grouped_folds(y[outer_train],meta.subject.iloc[outer_train].to_numpy())
    scores={}
    for region in a.REGIONS:
        truth=[];pred=[]
        for fi,(tr,te) in enumerate(inner):
            train=outer_train[tr];valid=outer_train[te]
            if set(meta.subject.iloc[train])&set(meta.subject.iloc[valid]):raise ValueError('Inner reader leakage')
            gt=memberships(meta,y,train,size,repeat_seed+fi)
            ge=memberships(meta,y,valid,size,repeat_seed+100+fi)
            model=estimator();X=features['region_'+region]
            model.fit(averaged(X,gt),y[gt[:,0]])
            truth.extend(y[ge[:,0]].tolist());pred.extend(model.predict(averaged(X,ge)).tolist())
        scores[region]=float(balanced_accuracy_score(truth,pred))
    chosen=max(a.REGIONS,key=lambda r:scores[r])
    return chosen,scores


def evaluate(meta,features,store):
    y=meta.truth.eq('positive').to_numpy(dtype=int)
    folds=z.grouped_folds(y,meta.subject.to_numpy())
    store.json('reader_folds.json',[dict(fold=i+1,train_readers=sorted(meta.subject.iloc[tr].unique()),
        test_readers=sorted(meta.subject.iloc[te].unique())) for i,(tr,te) in enumerate(folds)])
    summaries=[]
    for size in SPEC['sizes']:
        repeats=range(1 if size==1 else SPEC['regroupings40'])
        for repeat in repeats:
            seed=SPEC['seed']+repeat*10000
            groups=[];member_rows=[]
            for fi,(train,test) in enumerate(folds):
                gt=memberships(meta,y,train,size,seed+fi);ge=memberships(meta,y,test,size,seed+100+fi)
                if set(meta.subject.iloc[train])&set(meta.subject.iloc[test]):raise ValueError('Reader leakage')
                if set(gt.ravel())&set(ge.ravel()):raise ValueError('Epoch leakage')
                groups.append((gt,ge))
                for partition,gs in [('train',gt),('test',ge)]:
                    for j,g in enumerate(gs):member_rows.append(dict(fold=fi+1,partition=partition,example=j,
                        reader=str(meta.subject.iloc[g[0]]),truth=int(y[g[0]]),trials=size,event_ids=json.dumps(meta.event_id.iloc[g].tolist())))
            prefix=f'size{size:02d}/repeat{repeat:02d}'
            csv(store,prefix+'/memberships.csv.gz',pd.DataFrame(member_rows))
            for variant in [*features,'selected_region']:
                case=prefix+'/'+variant;mark=store.root/case/'result.json'
                if mark.is_file():
                    saved=read(mark)
                    if a.file_digest(store.root/saved['prediction_path'])!=saved['predictions_sha256']:raise ValueError('Corrupt completed prediction')
                    summaries.append(saved['score']);print('KEEP CASE:',case,flush=True);continue
                print(f'FIT size={size}, grouping={repeat+1}/{len(repeats)}, {variant}',flush=True)
                rows=[];selection=[];fit_warnings=[]
                for fi,((train,test),(gt,ge)) in enumerate(zip(folds,groups)):
                    if variant=='selected_region':
                        chosen,inner=select_region(meta,y,features,train,size,seed+1000+fi*10)
                        X=features['region_'+chosen];selection.append(dict(fold=fi+1,chosen=chosen,inner_training_reader_ba=inner))
                    else:X=features[variant]
                    model=estimator(variant=='dummy')
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');model.fit(averaged(X,gt),y[gt[:,0]])
                    fit_warnings.extend(str(w.message) for w in caught)
                    xe=averaged(X,ge);pred=model.predict(xe);prob=model.predict_proba(xe)[:,list(model.classes_).index(1)]
                    for j,g in enumerate(ge):rows.append(dict(fold=fi+1,example=j,reader=str(meta.subject.iloc[g[0]]),
                        truth=int(y[g[0]]),prediction=int(pred[j]),probability_positive=float(prob[j]),trials=size))
                frame=pd.DataFrame(rows);truth=frame.truth.to_numpy();pred=frame.prediction.to_numpy()
                readers=[dict(reader=str(subject),examples=len(g),balanced_accuracy=float(balanced_accuracy_score(g.truth,g.prediction))) for subject,g in frame.groupby('reader')]
                score=dict(size=size,regrouping=repeat,variant=variant,balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
                    equal_reader_ba=float(np.mean([r['balanced_accuracy'] for r in readers])),test_examples=len(frame),
                    original_test_trials=int(frame.trials.sum()),readers=int(frame.reader.nunique()))
                csv(store,case+'/predictions.csv.gz',frame)
                store.json(case+'/result.json',dict(score=score,per_reader=readers,selected_regions=selection,fit_warnings=fit_warnings,
                    confusion=confusion_matrix(truth,pred,labels=[0,1]).tolist(),prediction_path=case+'/predictions.csv.gz',
                    predictions_sha256=a.file_digest(store.root/case/'predictions.csv.gz')))
                summaries.append(score)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',required=True,type=Path)
    p.add_argument('--free-drive-bytes',required=True,type=int);args=p.parse_args()
    out=args.thesis_root/'Results/word_sentiment_eeg/teco_spatial40'
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    if args.free_drive_bytes<max(0,SPEC['budget']-used):raise OSError('Reserve at most20MB for new outputs; no epoch reads started.')
    import fcntl,tempfile
    lock=Path(tempfile.gettempdir(),'teco-spatial40-'+a.digest(str(out))[:16]+'.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta,baseline,source=old.load_cached(args.thesis_root)
    meta,baseline,counts=cohort(meta,baseline)
    archive=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz'
    montage=pd.read_csv(archive/'teco_channel_regions.csv')
    if montage.channel.duplicated().any() or set(montage.region)!=set(a.REGIONS):raise ValueError('Unexpected TeCo montage/regions')
    store=ImmutableStore(out,SPEC['budget'])
    code=['scripts/teco_spatial40.py','scripts/teco_average40.py','scripts/analyze_trial_archive.py','scripts/zuco_recovered_followups.py','scripts/audit_zuco_overlap.py']
    store.json('protocol.json',dict(spec=SPEC,source=source,code={f:a.file_digest(ROOT/f) for f in code},
        montage_sha256=a.file_digest(archive/'teco_channel_regions.csv'),
        versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']}))
    csv(store,'reader_cohort.csv',counts)
    print('FIXED COHORT:',meta.subject.nunique(),'readers,',len(meta),'trials',flush=True)
    post,pre=cached_channel_means(archive,meta,montage,store)
    features=spatial_features(post,pre,baseline,montage)
    rows=evaluate(meta,features,store)
    csv(store,'scores.csv',pd.DataFrame(rows))
    from audit_zuco_overlap import audit
    try:overlap=audit(args.thesis_root,store)
    except Exception as error:
        overlap=dict(verified_rows=[],unverified=[dict(reason=type(error).__name__+': '+str(error))],scope='Historical audit failed; no overlap conclusion. New TeCo scores are separate.')
    store.json('zuco_overlap_audit.json',overlap)
    if overlap['verified_rows']:csv(store,'zuco_overlap_verified.csv',pd.DataFrame(overlap['verified_rows']))
    print('ZUCO METADATA AUDIT:',len(overlap['verified_rows']),'verified fold rows;',len(overlap['unverified']),'unverified cases',flush=True)
    paths=['protocol.json','reader_cohort.csv','scores.csv','zuco_overlap_audit.json']+[str(p.relative_to(out)) for p in out.glob('size*/repeat*/*/result.json')]
    store.json('complete.json',dict(status='completed',cases=len(rows),checksums={f:a.file_digest(out/f) for f in paths}))
    print('TECO SPATIAL COMPLETE:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
