#!/usr/bin/env python3
"""One fixed CPU experiment: cleaned agent-subset SR, unseen-reader logistic sweep."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.common import digest,file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore
from zuco_recovered_followups import grouped_folds,averaged

SIZES=[1,*range(10,101,10)]
SPEC=dict(version=1,dataset='ZuCo SR',source='completed agent-labelled paper subset',representation='cleaned',
 window_ms=[224,304],channels=105,target='positive versus negative',holdout='reader',folds=3,
 model='logistic C1 balanced liblinear, training-only imputation/scaling',sizes=SIZES,
 grouping_repeats=5,seed=2026,remainders='discard',common_readers='at least100selected events per class',
 warning='Known test labels form averages; descriptive grouping variability is not a confidence interval; readers unseen but words/sentences may overlap',budget=20_000_000)

def read(path):return json.loads(Path(path).read_text())


def load_features(parent,subset,store,work):
    """Recover exactly the completed subset rows, never relabel or rematch them."""
    complete=read(subset/'complete.json')
    if complete.get('status')!='completed':raise ValueError('Wait for the agent subset to finish')
    roster=subset/'selection/trial_selection.csv.gz'
    if file_digest(roster)!=complete['checksums']['selection/trial_selection.csv.gz']:raise ValueError('Subset selection changed')
    table=pd.read_csv(roster,keep_default_na=False,dtype={'event_id':str})
    table=table[table.subset_status.eq('retained')].copy().reset_index(drop=True)
    if table.empty or table.event_id.duplicated().any():raise ValueError('Empty/duplicate subset roster')
    # The audit retains the parent SST sentiment; restore the agent target
    # exactly as the original subset's in-memory selected table did.
    table['sentiment']=table.lexical_class
    if not table.sentiment.isin(['negative','positive']).all() or not table.sentiment.eq(table.contextual_class).all():raise ValueError('Invalid retained agent labels')
    cpu=read(parent/'cpu_preparation_manifest.json')
    if cpu['status']!='completed' or cpu['protocol_sha256']!=file_digest(parent/'protocol.json'):raise ValueError('Parent preparation incomplete/changed')
    subjects=sorted(table.subject.astype(str).unique());identities=[]
    for subject in subjects:
        folder=parent/'readers'/subject;state=read(folder/'prepared.json')
        if state['status']!='prepared' or file_digest(folder/'prepared.json')!=cpu['reader_checkpoint_sha256'][subject]:raise ValueError('Parent reader incomplete/changed')
        rawmeta=folder/'accepted_trial_metadata.csv.gz'
        if file_digest(rawmeta)!=state['checksums'][rawmeta.name]:raise ValueError('Parent metadata changed')
        identities.append(dict(subject=subject,prepared=file_digest(folder/'prepared.json'),metadata=file_digest(rawmeta),waveforms=state['checksums']['prepared_trials.npz']))
    identity=digest(dict(spec=SPEC,selection=file_digest(roster),parent=identities))
    marker=store.root/'features/manifest.json'
    if marker.exists():
        saved=read(marker)
        if saved['identity']!=identity:raise ValueError('Different source selection; refusing mixed feature cache')
        for f,sha in saved['checksums'].items():
            if file_digest(store.root/'features'/f)!=sha:raise ValueError('Corrupt sweep feature cache')
        with np.load(store.root/'features/cleaned_224_304.npz',allow_pickle=False) as z:
            x=z['features'];ids=z['event_id'].astype(str)
        if ids.tolist()!=table.event_id.tolist():raise ValueError('Cached feature row mismatch')
        print('KEEP SMALL FEATURE CACHE: no EEG archive reads',flush=True)
        return x,table,identity
    features=np.zeros((len(table),105),np.float32);times=np.arange(-75,126)*8;pick=(times>=224)&(times<=304)
    for i,subject in enumerate(subjects,1):
        rows=np.flatnonzero(table.subject.astype(str).to_numpy()==subject);part=table.iloc[rows]
        folder=parent/'readers'/subject;state=read(folder/'prepared.json')
        meta=pd.read_csv(folder/'accepted_trial_metadata.csv.gz',keep_default_na=False,dtype={'event_id':str})
        indices=part.parent_row.to_numpy(int)
        if (indices<0).any() or (indices>=len(meta)).any() or meta.iloc[indices].event_id.tolist()!=part.event_id.tolist():raise ValueError('Subset/parent row mismatch')
        print(f'LOAD SAVED READER {i}/{len(subjects)} {subject}: {len(rows)} selected trials; no extraction',flush=True)
        path=work/'reader.npz';shutil.copyfile(folder/'prepared_trials.npz',path)
        if file_digest(path)!=state['checksums']['prepared_trials.npz']:raise ValueError('Corrupt saved parent waveforms')
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['times_ms'],times) or z['event_ids'][indices].astype(str).tolist()!=part.event_id.tolist():raise ValueError('Waveform identity/time mismatch')
            if not np.array_equal(z['labels'][indices],part.parent_sentiment.map({'negative':0,'positive':1}).to_numpy()):raise ValueError('Parent label mismatch')
            signal=z['signal']
            if signal.shape!=(2,len(meta),105,201):raise ValueError('Unexpected saved EEG schema')
            value=signal[1,indices][:,:,pick]
            if not np.isfinite(value).all():raise ValueError('Nonfinite selected EEG')
            features[rows]=value.mean(axis=2);del signal,value
        path.unlink() # Only the runtime-local copy is removed.
    file=work/'cleaned_224_304.npz';np.savez_compressed(file,features=features,event_id=table.event_id.to_numpy(dtype=str))
    sha=store.file('features/cleaned_224_304.npz',file)
    store.json('features/manifest.json',dict(identity=identity,checksums={'cleaned_224_304.npz':sha},source=identities,trials=len(table)))
    return features,table,identity


def common_cohort(x,meta):
    counts=pd.crosstab(meta.subject,meta.sentiment).reindex(columns=['negative','positive'],fill_value=0)
    subjects=counts.index[counts.min(axis=1)>=100].tolist()
    if len(subjects)<3:raise ValueError('Fewer than three readers have100trials in each class; do not change the cohort across curve points')
    keep=meta.subject.isin(subjects).to_numpy();m=meta.loc[keep].reset_index(drop=True)
    return x[keep],m,dict(all_reader_counts=counts.to_dict('index'),included_readers=subjects,
        excluded_readers=sorted(set(counts.index)-set(subjects)),events_before=len(meta),events_used=len(m),
        note='Same reader/event pool at every size; per-reader/class remainder is discarded after shuffling')


def evaluate(x,meta,size,repeat,folds):
    y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy(int)
    predictions=[];audits=[]
    # Size1 is identical across grouping repeats; compute it once.
    for fi,(train,test) in enumerate(folds):
        seed=2026+repeat*1000+fi
        a,ya,_,ma=averaged(x,y,meta,train,size,seed)
        b,yb,readers,mb=averaged(x,y,meta,test,size,seed+100)
        if set(meta.iloc[train].subject)&set(meta.iloc[test].subject) or set(ma)&set(mb):raise ValueError('Train/test leakage')
        if min(np.bincount(ya,minlength=2))<2 or len(set(yb))<2:raise ValueError('Too few complete averages')
        estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
            LogisticRegression(C=1,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=2026))
        estimator.fit(a,ya);p=estimator.predict_proba(b)[:,1];pred=estimator.predict(b)
        dummy=DummyClassifier(strategy='prior').fit(a,ya).predict(b)
        frame=pd.DataFrame(dict(size=size,repeat=repeat,fold=fi+1,reader=readers,truth=yb,prediction=pred,probability_positive=p,dummy_prediction=dummy))
        # Record all original event IDs contributing to each disjoint example.
        frame['member_event_ids']=[json.dumps(meta.iloc[mb[j:j+size]].event_id.tolist()) for j in range(0,len(mb),size)]
        if len(frame)!=len(mb)//size:raise ValueError('Averaging membership mismatch')
        predictions.append(frame)
        audits.append(dict(fold=fi+1,train_examples=len(ya),test_examples=len(yb),train_original_used=len(ma),test_original_used=len(mb),
            train_original_discarded=len(train)-len(ma),test_original_discarded=len(test)-len(mb),
            train_event_sha=digest(sorted(meta.iloc[ma].event_id.tolist())),test_event_sha=digest(sorted(meta.iloc[mb].event_id.tolist())),
            train_readers=sorted(meta.iloc[train].subject.unique()),test_readers=sorted(meta.iloc[test].subject.unique())))
    frame=pd.concat(predictions,ignore_index=True)
    if len(set(frame.reader))!=meta.subject.nunique():raise ValueError('A reader disappeared from this size')
    score=dict(size=size,repeat=repeat,balanced_accuracy=float(balanced_accuracy_score(frame.truth,frame.prediction)),
        dummy_balanced_accuracy=float(balanced_accuracy_score(frame.truth,frame.dummy_prediction)),
        accuracy=float(accuracy_score(frame.truth,frame.prediction)),auc=float(roc_auc_score(frame.truth,frame.probability_positive)),
        test_examples=len(frame),test_original_used=int(len(frame)*size),test_readers=int(frame.reader.nunique()),folds=audits)
    return score,frame


def run_sweep(x,meta,store,work):
    x,meta,cohort=common_cohort(x,meta);store.json('cohort.json',cohort)
    y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy(int);folds=grouped_folds(y,meta.subject.to_numpy())
    store.json('splits.json',[dict(train=meta.iloc[tr].event_id.tolist(),test=meta.iloc[te].event_id.tolist()) for tr,te in folds])
    scores=[]
    for size in SIZES:
        for repeat in range(1 if size==1 else SPEC['grouping_repeats']):
            relative=Path('cases')/f'size{size:03d}_grouping{repeat:02d}';marker=store.root/relative/'score.json'
            if marker.exists():
                saved=read(marker)
                if file_digest(store.root/relative/'predictions.csv.gz')!=saved['predictions_sha256']:raise ValueError('Changed saved sweep predictions')
                score=saved['score'];print(f'KEEP {size}trials/average grouping{repeat+1}',flush=True)
            else:
                print(f'FIT {size}trials/average grouping{repeat+1}/{1 if size==1 else SPEC["grouping_repeats"]}: three held-out-reader folds',flush=True)
                score,predictions=evaluate(x,meta,size,repeat,folds)
                path=work/'predictions.csv.gz';predictions.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
                sha=store.file(relative/'predictions.csv.gz',path);store.json(relative/'score.json',dict(score=score,predictions_sha256=sha))
            scores.append(score)
    frame=pd.DataFrame([{k:s[k] for k in sorted(s) if k!='folds'} for s in scores]);path=work/'repeat_scores.csv';frame.to_csv(path,index=False);store.file('repeat_scores.csv',path)
    summary=frame.groupby('size',sort=True).agg(balanced_accuracy=('balanced_accuracy','mean'),grouping_sd=('balanced_accuracy','std'),
        grouping_min=('balanced_accuracy','min'),grouping_max=('balanced_accuracy','max'),dummy_balanced_accuracy=('dummy_balanced_accuracy','mean'),
        test_examples=('test_examples','first'),test_original_used=('test_original_used','first'),test_readers=('test_readers','first'),groupings=('repeat','count')).reset_index()
    summary['grouping_sd']=summary.grouping_sd.fillna(0);path=work/'scores.csv';summary.to_csv(path,index=False);store.file('scores.csv',path)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(summary['size'],100*summary.balanced_accuracy,'o-',label='Logistic EEG')
    axes[0].fill_between(summary['size'],100*summary.grouping_min,100*summary.grouping_max,alpha=.15,label='Range across regroupings (not CI)')
    axes[0].plot(summary['size'],100*summary.dummy_balanced_accuracy,'--',color='gray',label='Dummy')
    axes[0].axhline(50,color='black',ls=':',lw=1);axes[0].set(xlabel='Trials per same-class average (exact)',ylabel='Balanced accuracy (%)',title='Unseen readers; cleaned EEG,224–304ms');axes[0].legend(fontsize=8)
    axes[1].plot(summary['size'],summary.test_examples,'o-',color='#148f77');axes[1].set(xlabel='Trials per same-class average (exact)',ylabel='Out-of-fold test examples',title='Larger averages leave fewer examples')
    fig.tight_layout();path=work/'averaging_curve.png';fig.savefig(path,dpi=160);plt.close(fig);store.file('averaging_curve.png',path)
    store.json('complete.json',dict(status='completed',sizes=SIZES,checksums={str(f.relative_to(store.root)):file_digest(f) for f in store.root.rglob('*') if f.is_file() and f.name not in {'complete.json','runtime_status.json'}}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; no local EEG loading/training')
    base=args.thesis_root/'Results/word_sentiment_eeg';parent=base/'zuco_paper_figures_recovered';subset=base/'zuco_agent_paper_subset';out=base/'zuco_averaging_sweep'
    store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve up to20MB free for this sweep')
    protocol=dict(spec=SPEC,source_subset=file_digest(subset/'complete.json'),source_parent=file_digest(parent/'cpu_preparation_manifest.json'),
        code={s:file_digest(ROOT/s) for s in ['scripts/zuco_averaging_sweep.py','scripts/zuco_recovered_followups.py','src/word_sentiment_eeg/paper_small_checkpoints.py']},software={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']})
    store.json('protocol.json',protocol)
    if (out/'complete.json').exists():
        for f,sha in read(out/'complete.json')['checksums'].items():
            if file_digest(out/f)!=sha:raise ValueError('Completed sweep output changed')
        print('KEEP COMPLETED SWEEP: no EEG loading or training',flush=True);return
    with tempfile.TemporaryDirectory(prefix='zuco-sweep-') as tmp:
        work=Path(tmp);x,meta,identity=load_features(parent,subset,store,work)
        store.json('feature_identity.json',dict(identity=identity));run_sweep(x,meta,store,work)
    print('AVERAGING SWEEP COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
