#!/usr/bin/env python3
"""Compact EEGNet-inspired CNN on the existing private trial archive.

No source EEG downloads, re-extraction, feature preparation or Drive EEG copy.
One task at a time is staged on the cloud runtime's temporary disk. All folds
use grouped inner validation, train-only scaling, class weights and resumable
epoch checkpoints. This is an independent PyTorch model, not EEGNet author code.
"""
import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,roc_auc_score,confusion_matrix
import analyze_trial_archive as a
from word_sentiment_eeg.common import digest,file_digest,write_json

SEED=2026
SPEC=dict(model='independent EEGNet-inspired 8/2/16',targets=['positive_negative','neutral_nonneutral'],
          splits=['sentence','word','reader'],outer_folds=3,max_epochs=25,patience=4,batch_size=128,
          learning_rate=.001,weight_decay=.0001,dropout=.5,input_ms=[0,800],last_sample_ms=792,
          sample_rate=125,selection='word_agreement; same amplitude/alignment and duplicate exclusions as linear analysis',
          normalization='per-channel train-only mean/SD',neutral_downsampling=False,seed=SEED)


class CompactEEGNet(nn.Module):
    def __init__(self,channels,samples=100,classes=2):
        super().__init__()
        self.body=nn.Sequential(
            nn.Conv2d(1,8,(1,33),padding=(0,16),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(channels,1),groups=8,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(.5),
            nn.Conv2d(16,16,(1,17),padding=(0,8),groups=16,bias=False),
            nn.Conv2d(16,16,1,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(.5),nn.Flatten())
        self.head=nn.Linear(16*(samples//4//4),classes)

    def forward(self,x):return self.head(self.body(x))


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def atomic_torch(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def stage_task(archive,records,thesis,temporary,prepare=True):
    """Checksum each archive shard once; retain only a temporary post-onset mmap."""
    task=records[0]['dataset']+'/'+records[0]['task'];tables=[];identities=[]
    for r in records:
        reader=archive/r['dataset']/'readers'/r['task']/r['subject']
        checkpoint=json.loads((reader/'checkpoint.json').read_text())
        if checkpoint['status']!='completed' or checkpoint['epochs']!=r['epochs']:
            raise ValueError('Incomplete reader archive')
        if checkpoint['identity']['protocol_sha256']!=digest(json.loads((archive/'protocol.json').read_text())):
            raise ValueError('Reader/archive protocol mismatch')
        if file_digest(reader/'events.csv.gz')!=checkpoint['events_sha256']:raise ValueError('Corrupt reader metadata')
        table=pd.read_csv(reader/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
        table=table[table.status.eq('archived')].copy().reset_index(drop=True)
        table,labels=a.attach_available_labels(table,archive,thesis,r['dataset'],r['task'])
        if len(table)!=r['epochs'] or table.event_id.duplicated().any():raise ValueError('Invalid reader roster')
        table['_reader_dir']=str(reader);tables.append(table)
        identities.append(dict(record=r,checkpoint_sha=file_digest(reader/'checkpoint.json'),labels=labels))
    meta=pd.concat(tables,ignore_index=True)
    if meta.event_id.duplicated().any():raise ValueError('Repeated task event identity')
    montage=pd.read_csv(archive/f'{records[0]["dataset"]}_channel_regions.csv').channel.astype(str).tolist()
    fingerprint=digest(dict(identities=identities,montage=montage,post_times=list(range(0,800,8))))
    directory=temporary/task;directory.mkdir(parents=True,exist_ok=True);mark=directory/'stage.json'
    shape=(len(meta),len(montage),100);bytes_needed=int(np.prod(shape))*4
    if not prepare:return None,meta,fingerprint,directory
    if mark.is_file():
        saved=json.loads(mark.read_text())
        if saved['fingerprint']==fingerprint and (directory/'signals.dat').is_file() and (directory/'signals.dat').stat().st_size==bytes_needed:
            if file_digest(directory/'events.csv.gz')!=saved['metadata_sha']:raise ValueError('Temporary metadata corrupt')
            if file_digest(directory/'signals.dat')!=saved['signal_sha']:raise ValueError('Temporary EEG corrupt')
            retained=pd.read_csv(directory/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
            print(f'KEEP RUNTIME STAGE {task}: no archive shard reads',flush=True)
            return np.memmap(directory/'signals.dat',dtype='float32',mode='r',shape=shape),retained,fingerprint,directory
    if shutil.disk_usage(temporary).free<bytes_needed+1_000_000_000:
        raise OSError(f'Need {bytes_needed/1e9:.2f} GB plus 1 GB free on cloud temporary disk; Drive untouched')
    signal=np.memmap(directory/'signals.dat',dtype='float32',mode='w+',shape=shape)
    meta['_row']=np.arange(len(meta));hashes=np.full(len(meta),'',dtype=object);seen=np.zeros(len(meta),bool)
    for ri,(r,table) in enumerate(zip(records,tables),1):
        reader=Path(table['_reader_dir'].iloc[0]);checkpoint=json.loads((reader/'checkpoint.json').read_text())
        print(f'LOAD {task}: reader {ri}/{len(records)} {r["subject"]} (existing trials, temporary disk only)',flush=True)
        for bi,block in enumerate(checkpoint['blocks'],1):
            for relative in block['epoch_files']:
                path=a.strict_path(reader,relative);payload=path.read_bytes()
                import hashlib
                if hashlib.sha256(payload).hexdigest()!=block['checksums'][relative]:raise ValueError('Corrupt archived shard')
                rows=meta[(meta._reader_dir==str(reader))&(meta.epoch_file==relative)].sort_values('epoch_row',key=lambda x:x.astype(int))
                with np.load(BytesIO(payload),allow_pickle=False) as z:
                    eeg=z['eeg_uV'];times=z['times_ms'];ids=z['event_id'].astype(str).tolist()
                    if ids!=rows.event_id.tolist() or rows.epoch_row.astype(int).tolist()!=list(range(len(rows))):
                        raise ValueError('EEG/event identity mismatch')
                    if z['channel_names'].astype(str).tolist()!=montage or not np.array_equal(times,np.arange(-25,101)*8):
                        raise ValueError('Unexpected archive montage/time axis')
                    if eeg.shape!=(len(rows),len(montage),126) or not np.isfinite(eeg).all():raise ValueError('Invalid archived EEG')
                    pos=rows._row.to_numpy(int)
                    if seen[pos].any():raise ValueError('Repeated shard row')
                    signal[pos]=eeg[:,:,(times>=0)&(times<800)];seen[pos]=True
                    hashes[pos]=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in eeg]
            if bi%10==0:print(f'LOAD {task}/{r["subject"]}: {bi}/{len(checkpoint["blocks"])} blocks',flush=True)
    if not seen.all():raise ValueError('Missing EEG rows during staging')
    signal.flush();meta['signal_sha256']=hashes
    meta.drop(columns=['sentence','_reader_dir','_row'],errors='ignore').to_csv(directory/'events.csv.gz',index=False,compression='gzip')
    write_json(mark,dict(fingerprint=fingerprint,shape=shape,metadata_sha=file_digest(directory/'events.csv.gz'),signal_sha=file_digest(directory/'signals.dat')))
    return signal,meta,fingerprint,directory


def population(meta,target):
    ix,labels,_=a.selected(meta,'word_agreement','binary' if target=='positive_negative' else 'three_class')
    duplicate=meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False)|meta.signal_sha256.duplicated(keep=False)
    retain=~duplicate.iloc[ix].to_numpy();ix=ix[retain];labels=labels[retain]
    classes=['negative','positive'] if target=='positive_negative' else ['neutral','nonneutral']
    y=(labels=='positive').astype(int) if target=='positive_negative' else (labels!='neutral').astype(int)
    if min(np.bincount(y,minlength=2))<10:raise ValueError('Fewer than ten trials in a class')
    return ix,y,classes


def nested_folds(meta,y,mode):
    column={'sentence':'sentence_key','word':'word_identity','reader':'subject'}[mode]
    groups=meta[column].astype(str).to_numpy();outer,spec=a.grouped_splits(y,groups)
    if mode=='word':outer,spec['outer_overlap_purges']=a.purge_overlapping_train(meta,y,outer)
    result=[]
    for train,test in outer:
        inner,inner_spec=a.grouped_splits(y[train],groups[train],max_folds=3)
        if mode=='word':inner,inner_spec['inner_overlap_purges']=a.purge_overlapping_train(meta.iloc[train].reset_index(drop=True),y[train],inner)
        fit,valid=inner[0];fit=train[fit];valid=train[valid]
        if any(set(groups[x])&set(groups[z]) for x,z in [(fit,valid),(fit,test),(valid,test)]):
            raise ValueError('Group leakage in nested fold')
        result.append((fit,valid,test))
    return result,spec


def train_stats(X,indices):
    total=np.zeros(X.shape[1],dtype=np.float64);squares=total.copy();n=0
    for start in range(0,len(indices),256):
        batch=np.asarray(X[indices[start:start+256]],dtype=np.float64)
        total+=batch.sum(axis=(0,2));squares+=(batch*batch).sum(axis=(0,2));n+=batch.shape[0]*batch.shape[2]
    mean=total/n;sd=np.sqrt(np.maximum(squares/n-mean**2,1e-12))
    return mean.astype(np.float32),sd.astype(np.float32)


def batches(X,indices,mean,sd,device,batch_size):
    for start in range(0,len(indices),batch_size):
        local=indices[start:start+batch_size]
        value=(np.array(X[local],copy=True)-mean[None,:,None])/sd[None,:,None]
        yield local,torch.from_numpy(value[:,None]).to(device)


def predict(model,X,indices,mean,sd,device,batch_size):
    model.eval();parts=[]
    with torch.no_grad():
        for _,batch in batches(X,indices,mean,sd,device,batch_size):
            parts.append(model(batch).softmax(1)[:,1].cpu().numpy())
    return np.concatenate(parts)


def train_fold(X,y,fit,valid,test,out,*,seed=SEED,max_epochs=25,patience=4,batch_size=128,device='cpu'):
    """Validation selects epoch; test is evaluated only after training finishes."""
    out.mkdir(parents=True,exist_ok=True)
    signature=digest(dict(fit=fit.tolist(),valid=valid.tolist(),test=test.tolist(),seed=seed,
                          epochs=max_epochs,patience=patience,batch=batch_size))
    result_path=out/'result.json'
    if result_path.is_file():
        result=json.loads(result_path.read_text())
        if result['signature']!=signature:raise ValueError('Fold configuration changed')
        if any(file_digest(out/k)!=v for k,v in result['checksums'].items()):raise ValueError('Corrupt completed CNN fold')
        print('KEEP CNN FOLD '+str(out),flush=True)
        return np.load(out/'probability.npy',allow_pickle=False)
    seed_all(seed);model=CompactEEGNet(X.shape[1],X.shape[2]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    mean,sd=train_stats(X,fit);counts=np.bincount(y[fit],minlength=2)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(len(fit)/(2*counts),dtype=torch.float32,device=device))
    begin=0;best=-1.;stale=0;history=[];last=out/'last.pt';marker=out/'last.json'
    if marker.is_file():
        state_meta=json.loads(marker.read_text())
        if state_meta['signature']!=signature or file_digest(last)!=state_meta['sha256']:raise ValueError('Corrupt/incompatible CNN checkpoint')
        state=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        begin=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
        print(f'RESUME CNN epoch {begin+1}/{max_epochs}: {out}',flush=True)
    for epoch in range(begin,max_epochs):
        if stale>=patience:break
        seed_all(seed+epoch);model.train();order=np.random.default_rng(seed+epoch).permutation(fit);loss_total=0.
        for local,batch in batches(X,order,mean,sd,device,batch_size):
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(batch),torch.from_numpy(y[local]).long().to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CNN loss; results not marked completed')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();loss_total+=loss.item()*len(local)
        probability=predict(model,X,valid,mean,sd,device,batch_size)
        score=float(balanced_accuracy_score(y[valid],probability>=.5))
        improved=score>best+1e-6
        if improved:
            best=score;stale=0;atomic_torch(out/'best.pt',dict(model=model.state_dict(),mean=mean,sd=sd,epoch=epoch,validation_ba=score))
        else:stale+=1
        history.append(dict(epoch=epoch+1,train_loss=loss_total/len(fit),validation_balanced_accuracy=score))
        atomic_torch(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history))
        write_json(marker,dict(signature=signature,sha256=file_digest(last)))
        print(f'CNN {out.name}: epoch {epoch+1}/{max_epochs}; validation BA {score:.3f}; best {best:.3f}',flush=True)
    best_state=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_state['model'])
    probability=predict(model,X,test,mean,sd,device,batch_size)
    np.save(out/'probability.npy',probability,allow_pickle=False)
    pd.DataFrame(history).to_csv(out/'learning_curve.csv',index=False)
    write_json(result_path,dict(signature=signature,best_epoch=best_state['epoch']+1,validation_ba=best,
                               fit_trials=len(fit),validation_trials=len(valid),test_trials=len(test),
                               parameters=sum(p.numel() for p in model.parameters()),
                               checksums={k:file_digest(out/k) for k in ['best.pt','probability.npy','learning_curve.csv']}))
    return probability


def run_case(signal,meta,ix,y,classes,mode,out,options):
    table=meta.iloc[ix].reset_index(drop=True);folds,spec=nested_folds(table,y,mode)
    write_json(out/'split_spec.json',spec)
    p=np.full(len(y),np.nan);fold_id=np.full(len(y),-1)
    # y is task-global here so the training function can address the mmap
    # directly without duplicating the entire EEG tensor for selected labels.
    global_y=np.full(len(meta),-1,dtype=int);global_y[ix]=y
    for fi,(fit,valid,test) in enumerate(folds):
        print(f'FIT {out}: fold {fi+1}/{len(folds)}, train={len(fit)}, validation={len(valid)}, test={len(test)}',flush=True)
        part=out/f'fold_{fi+1}';part.mkdir(parents=True,exist_ok=True)
        # Save every split identity, including inner validation, for leakage audits.
        pd.DataFrame([dict(event_id=table.iloc[j].event_id,role=role) for role,rows in [('train',fit),('validation',valid),('test',test)] for j in rows]).to_csv(part/'split_events.csv.gz',index=False,compression='gzip')
        p[test]=train_fold(signal,global_y,ix[fit],ix[valid],ix[test],part,seed=SEED+fi,**options);fold_id[test]=fi
    if np.isnan(p).any() or (fold_id<0).any():raise ValueError('Incomplete out-of-fold coverage')
    pred=(p>=.5).astype(int)
    score=dict(balanced_accuracy=float(balanced_accuracy_score(y,pred)),accuracy=float(accuracy_score(y,pred)),
               macro_f1=float(f1_score(y,pred,average='macro')),roc_auc=float(roc_auc_score(y,p)),
               trials=len(y),class_counts={c:int((y==i).sum()) for i,c in enumerate(classes)},
               confusion=confusion_matrix(y,pred,labels=[0,1]).tolist(),classes=classes)
    pd.DataFrame(dict(event_id=table.event_id,subject=table.subject,truth=np.asarray(classes)[y],prediction=np.asarray(classes)[pred],
                      probability_class_1=p,fold=fold_id)).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    write_json(out/'score.json',score);return score


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--temporary-root',type=Path,default=Path('/content/word_cnn_temporary'))
    parser.add_argument('--allow-cpu',action='store_true');args=parser.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud only: no local training or archive download')
    if not torch.cuda.is_available() and not args.allow_cpu:raise SystemExit('Select a GPU runtime, then rerun. No analysis started.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    archive=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz';records,archive_protocol=a.load_completed(archive)
    out=args.thesis_root/'Results/word_sentiment_eeg/cnn_eegnet';out.mkdir(parents=True,exist_ok=True)
    if args.temporary_root.resolve().is_relative_to(args.thesis_root.resolve()):raise ValueError('Temporary EEG staging must not be on Drive')
    args.temporary_root.mkdir(parents=True,exist_ok=True)
    protocol=dict(spec=SPEC,code=file_digest(Path(__file__)),helper=file_digest(ROOT/'scripts/analyze_trial_archive.py'),
                  archive=file_digest(archive/'manifest.json'),archive_protocol=archive_protocol,torch=torch.__version__,numpy=np.__version__,device=device)
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('CNN code/data/runtime changed; refusing mixed results')
    write_json(out/'protocol.json',protocol);scores=[];skips=[];by_task=defaultdict(list)
    for r in records:by_task[(r['dataset'],r['task'])].append(r)
    write_json(out/'manifest.json',dict(status='running'))
    options=dict(max_epochs=SPEC['max_epochs'],patience=SPEC['patience'],batch_size=SPEC['batch_size'],device=device)
    for ti,((dataset,task),roster) in enumerate(sorted(by_task.items()),1):
        print(f'TASK {ti}/{len(by_task)}: {dataset}/{task}',flush=True)
        _,meta,identity,tempdir=stage_task(archive,roster,args.thesis_root,args.temporary_root,prepare=False)
        taskdir=out/dataset/task;taskdir.mkdir(parents=True,exist_ok=True)
        if (taskdir/'source.json').is_file() and json.loads((taskdir/'source.json').read_text())!={'fingerprint':identity}:
            raise ValueError('Changed CNN labels or archive checkpoint')
        write_json(taskdir/'source.json',dict(fingerprint=identity))
        completed=taskdir/'task_completed.json'
        if completed.is_file():
            saved=json.loads(completed.read_text())
            if saved['fingerprint']!=identity or any(file_digest(taskdir/k)!=v for k,v in saved['checksums'].items()):
                raise ValueError('Corrupt/changed completed CNN task')
            scores.extend(saved['scores']);print(f'KEEP COMPLETED TASK {dataset}/{task}: no EEG reads',flush=True)
            continue
        signal,meta,identity,tempdir=stage_task(archive,roster,args.thesis_root,args.temporary_root)
        task_scores=[]
        for target in SPEC['targets']:
            try:ix,y,classes=population(meta,target)
            except ValueError as error:skips.append(dict(dataset=dataset,task=task,target=target,reason=str(error)));continue
            for mode in SPEC['splits']:
                case=taskdir/target/mode;case.mkdir(parents=True,exist_ok=True)
                try:nested_folds(meta.iloc[ix].reset_index(drop=True),y,mode)
                except ValueError as error:skips.append(dict(dataset=dataset,task=task,target=target,split=mode,reason=str(error)));continue
                score=run_case(signal,meta,ix,y,classes,mode,case,options)
                row=dict(dataset=dataset,task=task,target=target,split=mode,**score)
                scores.append(row);task_scores.append(row)
                pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
                a.check_budget(out,150_000_000)
        del signal
        shutil.rmtree(tempdir) # only this code's cloud temporary task, never Drive
        if len(task_scores)==6:
            write_json(completed,dict(fingerprint=identity,scores=task_scores,
                checksums={str(p.relative_to(taskdir)):file_digest(p) for p in taskdir.rglob('*') if p.is_file() and p.name!='task_completed.json'}))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    write_json(out/'skipped.json',skips)
    write_json(out/'manifest.json',dict(status='completed' if len(scores)==24 and not skips else 'incomplete',
                                      expected_cases=24,completed_cases=len(scores),skipped_cases=len(skips)))
    if len(scores)!=24 or skips:raise SystemExit('CNN incomplete: saved fits retained; inspect skipped.json')
    print('CNN COMPLETE:',out,flush=True)


if __name__=='__main__':main()
