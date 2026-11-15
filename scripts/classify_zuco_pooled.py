#!/usr/bin/env python3
"""One pooled SR+NR+TSR binary classifier; existing archive, GPU, small outputs.

Loads validated existing feature-cache metadata (not feature preparation), then
stages only selected saved EEG trials on Colab's temporary disk. Nothing is
re-epoched/cleaned/relabelled. Known-class averages are offline diagnostics.
"""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import analyze_trial_archive as a
import classify_cnn as c
from word_sentiment_eeg.common import digest, file_digest, write_json
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

TASKS=('SR','NR','TSR')
SPEC=dict(version=1,tasks=list(TASKS),target='positive versus negative',policy='lexical/contextual-word agreement',
          models=['CNN single','CNN max40','logistic single','logistic max40','dummy single','dummy max40','task_only single','task_only max40'],
          splits=['sentence','word','reader'],folds=3,averaging=[1,40],remainder='retain undersized final group',
          averaging_groups=['reader','task','class'],cnn='existing independent EEGNet-inspired 8/2/16',
          cnn_input_ms=[0,792],logistic_input_ms=[224,304],logistic_C=1,epochs=25,patience=4,batch=128,
          checkpoint_every=5,budget_bytes=100_000_000,seed=2026,
          warning='No additional paper regression cleaning. No unseen reader+word+sentence joint holdout. Test labels construct averages.')


def read(p):return json.loads(Path(p).read_text())


def inventory(archive,source,thesis):
    if read(source/'manifest.json').get('status')!='completed':raise ValueError('Original classification feature preparation must be completed')
    records,_=a.load_completed(archive)
    records=sorted([r for r in records if r['dataset']=='zuco'],key=lambda r:(TASKS.index(r['task']),r['subject']))
    if len(records)!=36 or any(len({r['subject'] for r in records if r['task']==t})!=12 for t in TASKS):raise ValueError('Expected36ZuCo reader/task recordings')
    if len({r['subject'] for r in records})!=12:raise ValueError('Reader identities must join across tasks')
    tables=[];identities=[];known=read(source/'label_sources.json')
    protocol=read(archive/'protocol.json')
    for i,r in enumerate(records,1):
        print(f'METADATA {i}/36: {r["task"]}/{r["subject"]}',flush=True)
        reader=archive/'zuco/readers'/r['task']/r['subject'];checkpoint=read(reader/'checkpoint.json')
        if checkpoint.get('status')!='completed' or checkpoint['epochs']!=r['epochs'] or checkpoint['identity']['protocol_sha256']!=digest(protocol):raise ValueError('Incomplete/changed archive reader')
        if file_digest(reader/'events.csv.gz')!=checkpoint['events_sha256']:raise ValueError('Corrupt archive event table')
        cache=source/'features/zuco'/r['task']/r['subject']
        if not (cache/'manifest.json').is_file():raise ValueError(f'Missing existing metadata cache: {cache}. No feature preparation or EEG extraction will start.')
        saved=read(cache/'manifest.json')
        expected=digest({'spec':a.FEATURE_SPEC,'checkpoint':file_digest(reader/'checkpoint.json')})
        if saved['signature']!=expected or file_digest(cache/'events.csv.gz')!=saved['checksums']['events.csv.gz']:raise ValueError('Changed/corrupt cached metadata')
        meta=pd.read_csv(cache/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
        if len(meta)!=r['epochs'] or meta.event_id.duplicated().any():raise ValueError('Invalid cached trial roster')
        meta,label=a.attach_available_labels(meta,archive,thesis,'zuco',r['task'])
        if not label or label['sha256'] not in {v['sha256'] for v in known}:raise ValueError('Word annotations differ from original completed analysis')
        if not meta.dataset.eq('zuco').all() or not meta.task.eq(r['task']).all() or not meta.subject.eq(r['subject']).all():raise ValueError('Task/reader metadata mismatch')
        meta['_reader_dir']=str(reader);meta['_recording']=r['task']+'/'+r['subject'];tables.append(meta)
        identities.append(dict(record=r,checkpoint=file_digest(reader/'checkpoint.json'),cache_metadata=file_digest(cache/'events.csv.gz'),labels=label))
    full=pd.concat(tables,ignore_index=True)
    if full.event_id.duplicated().any():raise ValueError('Event IDs repeat across tasks')
    # Native sentence indices restart in each task: include task in physical keys.
    duplicate=full.duplicated(['task','subject','sentence_position','onset_sample_sentence_0based'],keep=False)
    duplicate|=full.signal_sha256.duplicated(keep=False)
    ix,labels,_=a.selected(full,'word_agreement','binary')
    keep=~duplicate.iloc[ix].to_numpy();meta=full.iloc[ix[keep]].copy().reset_index(drop=True)
    meta['truth']=labels[keep];meta['_stage_row']=np.arange(len(meta))
    counts=meta.groupby(['task','truth']).size().unstack(fill_value=0)
    if set(counts.index)!=set(TASKS) or (counts.min(axis=1)<10).any():raise ValueError('Every task needs >=10eligible trials per class')
    report=dict(all_cached_word_observations=len(full),duplicate_observations_excluded=int(duplicate.sum()),selected_trials=len(meta),readers=int(meta.subject.nunique()),
                task_class_counts={t:{k:int(v) for k,v in counts.loc[t].items()} for t in TASKS},
                policy='Same word-agreement/quality policy; global duplicate-waveform exclusion across tasks, task-aware physical keys')
    return meta,identities,report


def stage(meta,archive,temporary,identity):
    temporary=Path(temporary);temporary.mkdir(parents=True,exist_ok=True)
    # Never place a waveform copy anywhere in mounted Drive, even another folder.
    if any(v.lower()=='drive' for v in temporary.resolve().parts):raise ValueError('Temporary staging must not be on Drive')
    montage=pd.read_csv(archive/'zuco_channel_regions.csv').channel.astype(str).tolist()
    if len(montage)!=105:raise ValueError('Expected105ZuCo channels')
    shape=(len(meta),105,100);size=int(np.prod(shape))*4
    marker=temporary/'stage.json';signal_path=temporary/'signals.dat'
    if marker.is_file():
        saved=read(marker)
        if saved['identity']==identity and saved['shape']==list(shape) and signal_path.is_file() and signal_path.stat().st_size==size:
            if file_digest(signal_path)!=saved['sha256']:raise ValueError('Corrupt temporary signal cache')
            print('KEEP RUNTIME EEG STAGE: no archive shard reads',flush=True)
            return np.memmap(signal_path,dtype='float32',mode='r',shape=shape)
    if shutil.disk_usage(temporary).free<size+500_000_000:raise OSError(f'Need {size/1e9:.2f}GB plus0.5GB on Colab temporary disk; no extra Drive EEG copy')
    signal=np.memmap(signal_path,dtype='float32',mode='w+',shape=shape);seen=np.zeros(len(meta),bool)
    groups=list(meta.groupby(['task','subject'],sort=False));total_files=sum(g.epoch_file.nunique() for _,g in groups);done=0
    for ri,((task,subject),table) in enumerate(groups,1):
        print(f'LOAD EEG RECORDING {ri}/{len(groups)}: {task}/{subject}; {len(table)}selected trials',flush=True)
        reader=Path(table['_reader_dir'].iloc[0]);checkpoint=read(reader/'checkpoint.json')
        checks={k:v for block in checkpoint['blocks'] for k,v in block['checksums'].items()}
        for j,(relative,rows) in enumerate(table.groupby('epoch_file',sort=False),1):
            path=a.strict_path(reader,relative);payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=checks[relative]:raise ValueError('Corrupt saved epoch shard: '+str(relative))
            with np.load(BytesIO(payload),allow_pickle=False) as z:
                if z['channel_names'].astype(str).tolist()!=montage or not np.array_equal(z['times_ms'],np.arange(-25,101)*8):raise ValueError('Unexpected montage/time grid')
                positions=rows.epoch_row.astype(int).to_numpy();ids=z['event_id'].astype(str)
                if (positions<0).any() or (positions>=len(ids)).any() or ids[positions].tolist()!=rows.event_id.tolist():raise ValueError('Archive row/event mismatch')
                full=z['eeg_uV'][positions]
                if full.shape!=(len(rows),105,126) or not np.isfinite(full).all():raise ValueError('Invalid saved waveforms')
                hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in full]
                if hashes!=rows.signal_sha256.tolist():raise ValueError('Saved EEG differs from metadata cache')
                pos=rows['_stage_row'].to_numpy(int)
                if seen[pos].any():raise ValueError('Repeated staging row')
                signal[pos]=full[:,:,25:125];seen[pos]=True
            done+=1
            if j==1 or j%10==0 or j==table.epoch_file.nunique():print(f'LOAD {task}/{subject}: {j}/{table.epoch_file.nunique()}shards; overall {done}/{total_files} ({100*done/total_files:.1f}%)',flush=True)
    if not seen.all():raise ValueError('Some selected trials lack EEG')
    signal.flush();write_json(marker,dict(identity=identity,shape=list(shape),sha256=file_digest(signal_path)))
    return signal


def purged(meta,y,folds):
    copy=meta.copy();copy['subject']=meta['_recording'].to_numpy()
    return a.purge_overlapping_train(copy,y,folds)


def folds_for(meta,mode):
    y=(meta.truth.to_numpy()=='positive').astype(int)
    column={'sentence':'sentence_key','word':'word_identity','reader':'subject'}[mode]
    groups=meta[column].astype(str).to_numpy()
    strata=meta.task.astype(str).to_numpy()+'|'+meta.truth.astype(str).to_numpy()
    outer,spec=a.grouped_splits(strata,groups)
    # Fold choice only checks coverage, never model accuracy.
    if mode=='word':outer,spec['outer_purged']=purged(meta,y,outer)
    nested=[]
    for train,test in outer:
        inner,inner_spec=a.grouped_splits(strata[train],groups[train])
        if mode=='word':inner,inner_spec['purged']=purged(meta.iloc[train].reset_index(drop=True),y[train],inner)
        fit,valid=inner[0];fit,valid=train[fit],train[valid]
        for left,right in [(fit,valid),(fit,test),(valid,test)]:
            if set(groups[left])&set(groups[right]) or set(meta.iloc[left].event_id)&set(meta.iloc[right].event_id):raise ValueError('Group/event leakage')
        nested.append((np.asarray(train),fit,valid,np.asarray(test)))
    return nested,spec


def memberships(meta,indices,size,seed):
    rng=np.random.default_rng(seed);groups=[];labels=[];rows=[]
    part=meta.iloc[indices]
    for (subject,task,truth),frame in part.groupby(['subject','task','truth'],sort=True):
        ix=rng.permutation(frame.index.to_numpy())
        for start in range(0,len(ix),size):
            members=ix[start:start+size];groups.append(members)
            labels.append(int(truth=='positive'))
            rows.append(dict(event_id=digest(meta.iloc[members].event_id.tolist())[:32],subject=subject,task=task,truth=truth,
                             original_trials=len(members),is_remainder=len(members)<size))
    if not groups:return [],np.empty(0,int),pd.DataFrame(columns=['event_id','subject','task','truth','original_trials','is_remainder'])
    flat=np.concatenate(groups)
    if len(flat)!=len(indices) or len(set(flat))!=len(indices):raise ValueError('Averaging duplicates/drops selected events')
    return groups,np.asarray(labels,int),pd.DataFrame(rows)


def arrays(signal,meta,indices,size,seed):
    groups,y,table=memberships(meta,indices,size,seed)
    x=np.stack([np.asarray(signal[g]).mean(axis=0) for g in groups]).astype(np.float32) if groups else np.empty((0,signal.shape[1],signal.shape[2]),np.float32)
    return x,y,table


def score_predictions(table,p):
    table=table.copy();y=(table.truth.to_numpy()=='positive').astype(int);pred=(p>=.5).astype(int)
    table['probability_positive']=p;table['prediction']=np.where(pred,'positive','negative')
    def one(ix):
        target=y[ix];estimated=pred[ix]
        row=dict(instances=len(ix),original_trials=int(table.iloc[ix].original_trials.sum()),has_both_classes=len(set(target))==2)
        if len(set(target))<2:return dict(row,balanced_accuracy=None,accuracy=float(accuracy_score(target,estimated)),reason='One class in this subgroup')
        return dict(row,balanced_accuracy=float(balanced_accuracy_score(target,estimated)),accuracy=float(accuracy_score(target,estimated)),macro_f1=float(f1_score(target,estimated)),auc=float(roc_auc_score(target,p[ix])),confusion=confusion_matrix(target,estimated,labels=[0,1]).tolist())
    task_rows={t:one(np.flatnonzero(table.task.to_numpy()==t)) if table.task.eq(t).any() else dict(instances=0,balanced_accuracy=None,reason='No test instances') for t in TASKS}
    reader_rows={s:one(np.flatnonzero(table.subject.to_numpy()==s)) for s in sorted(table.subject.unique())}
    valid=[r['balanced_accuracy'] for r in task_rows.values() if r.get('balanced_accuracy') is not None]
    return dict(pooled=one(np.arange(len(table))),per_task=task_rows,per_reader=reader_rows,equal_task_balanced_accuracy=float(np.mean(valid)) if len(valid)==3 else None,
                average_sizes=dict(min=int(table.original_trials.min()),max=int(table.original_trials.max()),mean=float(table.original_trials.mean()),full40=int(table.original_trials.eq(40).sum()),remainders=int(table.is_remainder.sum()))),table


def fit_cnn(x,y,fit,valid,test,store,relative,seed,device='cuda',epochs=25,patience=4):
    signature=digest(dict(fit=fit.tolist(),valid=valid.tolist(),test=test.tolist(),seed=seed,epochs=epochs,patience=patience,shape=list(x.shape)))
    out=store.root/relative;final=out/'result.json'
    if final.is_file():
        saved=read(final)
        if saved['signature']!=signature or file_digest(out/'probability.npy')!=saved['probability_sha256']:raise ValueError('Corrupt/changed CNN fold')
        print('KEEP CNN '+str(relative),flush=True);return np.load(out/'probability.npy',allow_pickle=False)
    c.seed_all(seed);model=c.CompactEEGNet(x.shape[1],x.shape[2]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    mean,sd=c.train_stats(x,fit);counts=np.bincount(y[fit],minlength=2)
    if min(counts)==0 or len(set(y[valid]))<2:raise ValueError('CNN training/validation needs both classes')
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(len(fit)/(2*counts),dtype=torch.float32,device=device))
    best=-1.;stale=0;start=0;history=[];best_weights=None
    frames=sorted(out.glob('frames/epoch_*.json'))
    if frames:
        mark=read(frames[-1]);frame=out/mark['file']
        if mark['signature']!=signature or file_digest(frame)!=mark['sha256']:raise ValueError('Corrupt CNN checkpoint')
        state=torch.load(frame,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for state_value in optimizer.state.values():
            for k,v in state_value.items():
                if isinstance(v,torch.Tensor):state_value[k]=v.to(device)
        start=state['epoch'];best=state['best'];stale=state['stale'];history=state['history'];best_weights=state['best_weights']
        print(f'RESUME CNN {relative}: epoch {start+1}/{epochs}',flush=True)
    for epoch in range(start,epochs):
        if stale>=patience:break
        c.seed_all(seed+epoch);model.train();order=np.random.default_rng(seed+epoch).permutation(fit)
        for local,batch in c.batches(x,order,mean,sd,device,SPEC['batch']):
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(batch),torch.from_numpy(y[local]).long().to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CNN loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        prob=c.predict(model,x,valid,mean,sd,device,SPEC['batch']);ba=float(balanced_accuracy_score(y[valid],prob>=.5))
        if ba>best+1e-6:best=ba;stale=0;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        history.append(dict(epoch=epoch+1,validation_balanced_accuracy=ba))
        print(f'CNN {relative}: epoch {epoch+1}/{epochs}; validation BA {ba:.3f}; best {best:.3f}',flush=True)
        if (epoch+1)%SPEC['checkpoint_every']==0 or stale>=patience or epoch+1==epochs:
            state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,stale=stale,history=history,best_weights=best_weights)
            with tempfile.TemporaryDirectory() as tmp:
                f=Path(tmp)/'frame.pt';torch.save(state,f);name=f'frames/epoch_{epoch+1:03d}.pt';sha=store.file(relative/name,f)
            store.json(relative/f'frames/epoch_{epoch+1:03d}.json',dict(signature=signature,file=name,sha256=sha))
    if best_weights is None:raise ValueError('No trained model checkpoint')
    model.load_state_dict(best_weights);prob=c.predict(model,x,test,mean,sd,device,SPEC['batch'])
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'probability.npy';np.save(f,prob,allow_pickle=False);sha=store.file(relative/'probability.npy',f)
    store.json(relative/'result.json',dict(signature=signature,probability_sha256=sha,best_validation_ba=best,epochs=len(history),history=history))
    del optimizer,model
    if device=='cuda':torch.cuda.empty_cache()
    return prob


def run_case(signal,meta,mode,size,model,folds,store,device='cuda',epochs=25):
    relative=Path('cases')/f'{mode}__max{size}__{model}';mark=store.root/relative/'score.json'
    if mark.is_file():
        saved=read(mark)
        if file_digest(store.root/relative/'predictions.csv.gz')!=saved['predictions_sha256']:raise ValueError('Corrupt saved predictions')
        print('KEEP CASE '+str(relative),flush=True);return saved['summary']
    predictions=[];split_report=[]
    for fi,(train,fit,valid,test) in enumerate(folds):
        print(f'CASE {relative}: fold {fi+1}/{len(folds)}',flush=True)
        # Validation is a separate original-event partition before averaging.
        fit_groups,yfit,mfit=memberships(meta,fit if model=='cnn' else train,size,2026+fi)
        valid_groups,yvalid,mvalid=memberships(meta,valid,size,2036+fi)
        test_groups,ytest,mtest=memberships(meta,test,size,2046+fi)
        row=dict(fold=fi+1,train_events=len(fit if model=='cnn' else train),validation_events=len(valid) if model=='cnn' else 0,test_events=len(test),
                 train_instances=len(yfit),validation_instances=len(yvalid) if model=='cnn' else 0,test_instances=len(ytest),
                 train_task_counts=mfit.task.value_counts().to_dict(),test_task_counts=mtest.task.value_counts().to_dict(),
                 train_event_sha=digest(sorted(meta.iloc[fit if model=='cnn' else train].event_id.tolist())),test_event_sha=digest(sorted(meta.iloc[test].event_id.tolist())))
        if set(mfit.task)!=set(TASKS) or set(mtest.task)!=set(TASKS):raise ValueError('Pooled train/test must include all three tasks')
        if min(np.bincount(yfit,minlength=2))<2 or len(set(ytest))<2:raise ValueError('Insufficient averaged examples')
        if model=='cnn':
            af=np.stack([np.asarray(signal[g]).mean(axis=0) for g in fit_groups]).astype('float32')
            av=np.stack([np.asarray(signal[g]).mean(axis=0) for g in valid_groups]).astype('float32')
            at=np.stack([np.asarray(signal[g]).mean(axis=0) for g in test_groups]).astype('float32')
            x=np.concatenate([af,av,at]);y=np.concatenate([yfit,yvalid,ytest]);n,m=len(af),len(av)
            prob=fit_cnn(x,y,np.arange(n),np.arange(n,n+m),np.arange(n+m,len(x)),store,relative/f'fold_{fi+1}',2026+fi,device,epochs)
            del x,af,av,at
        elif model=='logistic':
            # The short window was informative previously; keep all electrodes.
            window=np.arange(0,800,8);pick=(window>=224)&(window<=304)
            xf=np.stack([np.asarray(signal[g])[:,:,pick].mean(axis=(0,2)) for g in fit_groups])
            xt=np.stack([np.asarray(signal[g])[:,:,pick].mean(axis=(0,2)) for g in test_groups])
            estimator=a.model_for('logistic',2);estimator.fit(xf,np.where(yfit,'positive','negative'))
            prob=estimator.predict_proba(xt)[:,list(estimator.classes_).index('positive')]
        elif model=='dummy':prob=np.full(len(ytest),float(yfit.mean()))
        elif model=='task_only':
            taskprob={t:float(yfit[mfit.task.to_numpy()==t].mean()) for t in TASKS};prob=mtest.task.map(taskprob).to_numpy(float)
        else:raise ValueError('Unknown model')
        mtest['fold']=fi+1;mtest['probability_positive']=prob;predictions.append(mtest);split_report.append(row)
    frame=pd.concat(predictions,ignore_index=True);summary,frame=score_predictions(frame,frame.probability_positive.to_numpy())
    summary=dict(split=mode,model=model,max_trials_per_average=size,**summary)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'predictions.csv.gz';frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0});sha=store.file(relative/'predictions.csv.gz',path)
    store.json(relative/'split_report.json',split_report);store.json(relative/'score.json',dict(summary=summary,predictions_sha256=sha))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--temporary-root',type=Path,default=Path('/content/zuco-pooled-waveforms'));p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab only; no local training/EEG reads')
    if not torch.cuda.is_available():raise SystemExit('Select T4 GPU, then Run all')
    base=args.thesis_root/'Results/word_sentiment_eeg';archive=base/'trial_archive_125hz';source=base/'classification_similarity';out=base/'zuco_pooled_cnn'
    store=ImmutableStore(out,SPEC['budget_bytes'])
    if args.free_drive_bytes<max(0,SPEC['budget_bytes']-store.used):raise OSError('Reserve up to100MB additional free Drive space; no EEG copy is saved there')
    meta,identities,coverage=inventory(archive,source,args.thesis_root)
    protocol=dict(spec=SPEC,inputs=identities,code={f:file_digest(ROOT/f) for f in ['scripts/classify_zuco_pooled.py','scripts/classify_cnn.py','scripts/analyze_trial_archive.py','src/word_sentiment_eeg/paper_small_checkpoints.py']},
                  montage=file_digest(archive/'zuco_channel_regions.csv'),software={k:version(k) for k in ['numpy','pandas','scipy','scikit-learn','torch']})
    store.json('protocol.json',protocol);store.json('input_counts.json',coverage)
    marker=out/'complete.json'
    if marker.is_file():
        for f,sha in read(marker)['checksums'].items():
            if file_digest(out/f)!=sha:raise ValueError('Completed pooled results changed')
        print('KEEP COMPLETED POOLED RUN: saved results verified; no EEG loading/training',flush=True);return
    if args.temporary_root.resolve().is_relative_to(args.thesis_root.resolve()):raise ValueError('Temporary EEG staging must be outside Thesis/Drive')
    identity=digest(dict(protocol=protocol,events=meta.event_id.tolist()));signal=stage(meta,archive,args.temporary_root,identity)
    results=[]
    for mode in SPEC['splits']:
        folds,spec=folds_for(meta,mode);store.json(f'splits/{mode}.json',dict(spec=spec,train_test_event_ids=[dict(train=meta.iloc[tr].event_id.tolist(),validation=meta.iloc[va].event_id.tolist(),fit=meta.iloc[fi].event_id.tolist(),test=meta.iloc[te].event_id.tolist()) for tr,fi,va,te in folds]))
        for size in SPEC['averaging']:
            for model in ['dummy','task_only','logistic','cnn']:
                results.append(run_case(signal,meta,mode,size,model,folds,store))
    store.json('summary.json',results)
    rows=[]
    for r in results:
        rows.append(dict(split=r['split'],model=r['model'],max_trials_per_average=r['max_trials_per_average'],pooled_ba=r['pooled']['balanced_accuracy'],equal_task_ba=r['equal_task_balanced_accuracy'],test_instances=r['pooled']['instances'],**{t+'_ba':r['per_task'][t].get('balanced_accuracy') for t in TASKS}))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'scores.csv';pd.DataFrame(rows).to_csv(path,index=False);store.file('scores.csv',path)
    store.json('complete.json',dict(status='completed',cases=len(results),checksums={str(f.relative_to(out)):file_digest(f) for f in out.rglob('*') if f.is_file() and f.name not in {'complete.json','runtime_status.json'}}))
    del signal
    print('POOLED SR+NR+TSR COMPLETE:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
