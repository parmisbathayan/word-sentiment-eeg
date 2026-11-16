#!/usr/bin/env python3
"""Extend the exact original63.9%case, verifying it before the averaging sweep."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import zuco_averaging_sweep as b
from zuco_recovered_followups import grouped_folds,averaged,metrics,SEED
from word_sentiment_eeg.common import digest,file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

SPEC=dict(version=1,dataset='ZuCo SR',selection='original complete agent subset; no>=100/class reader prefilter',
 representation='cleaned',window=[224,304],model='logistic C1 balanced liblinear',split='original three reader folds',sizes=b.SIZES,
 groupings=5,seed=SEED,remainders='discard per reader/class; readers can contribute no average at large sizes',
 reference='original40-trial first grouping, checked against saved metrics and original-event digests',budget=15_000_000)

def evaluate(x,meta,size,repeat,folds):
    y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy(int);frames=[];audit=[]
    for fi,(train,test) in enumerate(folds):
        seed=SEED+1000*repeat+fi
        a,ya,_,ma=averaged(x,y,meta,train,size,seed);z,yz,readers,mz=averaged(x,y,meta,test,size,seed+100)
        if set(meta.iloc[train].subject)&set(meta.iloc[test].subject) or set(ma)&set(mz):raise ValueError('Partition leakage')
        if not len(ya) or not len(yz) or len(set(ya))<2 or len(set(yz))<2:raise ValueError('Insufficient complete averages in an original fold; do not silently change folds')
        model=b.make_pipeline(b.SimpleImputer(strategy='median',keep_empty_features=True),b.StandardScaler(),b.LogisticRegression(C=1,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=SEED))
        model.fit(a,ya);prediction=model.predict(z);prob=model.predict_proba(z)[:,1]
        dummy=b.DummyClassifier(strategy='prior').fit(a,ya).predict(z)
        frame=pd.DataFrame(dict(fold=fi+1,reader=readers,truth=yz,prediction=prediction,probability_positive=prob,dummy_prediction=dummy))
        frame['member_event_ids']=[json.dumps(meta.iloc[mz[j:j+size]].event_id.tolist()) for j in range(0,len(mz),size)]
        frames.append(frame)
        audit.append(dict(fold=fi+1,train_instances=len(ya),test_instances=len(yz),
            train_event_digest=digest(sorted(meta.iloc[ma].event_id.tolist())),test_event_digest=digest(sorted(meta.iloc[mz].event_id.tolist())),
            test_readers_original=sorted(meta.iloc[test].subject.unique()),test_readers_contributing=sorted(set(readers))))
    frame=pd.concat(frames,ignore_index=True)
    result=dict(size=size,repeat=repeat,**metrics(frame.truth.to_numpy(),frame.prediction.to_numpy(),frame.probability_positive.to_numpy()),
        dummy_balanced_accuracy=float(b.balanced_accuracy_score(frame.truth,frame.dummy_prediction)),test_readers=int(frame.reader.nunique()),
        missing_readers=sorted(set(meta.subject)-set(frame.reader)),folds=audit)
    return result,frame


def reference_check(result,saved):
    if saved['status']!='completed' or saved['representation']!='cleaned' or saved['window']!='published_224_304' or saved['split']!='reader' or saved['model']!='logistic' or saved['average_trials']!=40:raise ValueError('Unexpected saved reference setup')
    for key in ['n','accuracy','balanced_accuracy','auc','macro_f1']:
        if not np.isclose(result[key],saved['pooled'][key],rtol=0,atol=1e-12):raise ValueError('Original40-trial case did not reproduce: '+key)
    if result['confusion_matrix']!=saved['pooled']['confusion_matrix']:raise ValueError('Original confusion matrix differs')
    if len(result['folds'])!=len(saved['folds']):raise ValueError('Original fold count differs')
    for a,z in zip(result['folds'],saved['folds']):
        for key in ['fold','train_instances','test_instances','train_event_digest','test_event_digest']:
            if a[key]!=z[key]:raise ValueError('Original fold/membership differs: '+key)
    return dict(status='verified',balanced_accuracy=result['balanced_accuracy'],test_examples=result['n'],
        note='Original one-grouping40-trial case, not the mean of five regroupings; no seed selection')


def run(x,meta,reference,store,work):
    folds=grouped_folds(meta.sentiment.eq('positive').to_numpy(int),meta.subject.to_numpy())
    counts=pd.crosstab(meta.subject,meta.sentiment).to_dict('index')
    store.json('cohort.json',dict(readers=list(dict.fromkeys(meta.subject)),events=len(meta),counts=counts,policy=SPEC['selection']))
    store.json('splits.json',[dict(train=meta.iloc[tr].event_id.tolist(),test=meta.iloc[te].event_id.tolist()) for tr,te in folds])
    # Always confirm the exact historical case before producing a new curve.
    original,_=evaluate(x,meta,40,0,folds);verified=reference_check(original,reference);store.json('reference_verification.json',verified)
    print(f'ORIGINAL CASE VERIFIED: {100*original["balanced_accuracy"]:.2f}% BA, {original["n"]}test averages',flush=True)
    scores=[]
    for size in b.SIZES:
        for repeat in range(1 if size==1 else SPEC['groupings']):
            rel=Path('cases')/f'size{size:03d}_grouping{repeat:02d}';mark=store.root/rel/'score.json'
            if mark.exists():
                saved=b.read(mark)
                if file_digest(store.root/rel/'predictions.csv.gz')!=saved['sha256']:raise ValueError('Changed predictions')
                score=saved['score'];print(f'KEEP size{size},grouping{repeat+1}',flush=True)
            else:
                print(f'FIT size{size},grouping{repeat+1}: original reader folds',flush=True)
                score,pred=evaluate(x,meta,size,repeat,folds);path=work/'predictions.csv.gz';pred.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
                sha=store.file(rel/'predictions.csv.gz',path);store.json(rel/'score.json',dict(score=score,sha256=sha))
            scores.append(score)
    frame=pd.DataFrame([{k:s[k] for k in ['size','repeat','balanced_accuracy','dummy_balanced_accuracy','n','test_readers']} for s in scores])
    path=work/'repeat_scores.csv';frame.to_csv(path,index=False);store.file('repeat_scores.csv',path)
    summary=frame.groupby('size').agg(balanced_accuracy=('balanced_accuracy','mean'),minimum=('balanced_accuracy','min'),maximum=('balanced_accuracy','max'),
        dummy=('dummy_balanced_accuracy','mean'),test_examples=('n','first'),test_readers=('test_readers','first')).reset_index()
    path=work/'scores.csv';summary.to_csv(path,index=False);store.file('scores.csv',path)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].plot(summary['size'],100*summary.balanced_accuracy,'o-',label='Mean across regroupings')
    axes[0].fill_between(summary['size'],100*summary.minimum,100*summary.maximum,alpha=.15,label='Grouping range, not CI')
    axes[0].scatter([40],[100*original['balanced_accuracy']],marker='D',color='#d97706',zorder=5,label='Verified original40-trial grouping')
    axes[0].plot(summary['size'],100*summary.dummy,'--',color='gray',label='Dummy');axes[0].set(xlabel='Exact trials per average',ylabel='Balanced accuracy (%)',title='Original agent subset; unseen readers');axes[0].legend(fontsize=7)
    axes[1].plot(summary['size'],summary.test_examples,'o-',color='#148f77');axes[1].set(xlabel='Exact trials per average',ylabel='Out-of-fold test examples',title='Original cohort; contributing readers can decrease')
    fig.tight_layout();path=work/'averaging_curve.png';fig.savefig(path,dpi=160);plt.close(fig);store.file('averaging_curve.png',path)
    store.json('complete.json',dict(status='completed',checksums={str(f.relative_to(store.root)):file_digest(f) for f in store.root.rglob('*') if f.is_file() and f.name not in {'complete.json','runtime_status.json'}}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only')
    base=args.thesis_root/'Results/word_sentiment_eeg';out=base/'zuco_original_cohort_sweep';subset=base/'zuco_agent_paper_subset';parent=base/'zuco_paper_figures_recovered';cache=base/'zuco_averaging_sweep'
    if not (cache/'features/manifest.json').exists():raise ValueError('The completed averaging sweep feature cache is required; no EEG reads will start')
    relative='classification/cleaned__published_224_304__reader__avg40__logistic.json';complete=b.read(subset/'complete.json')
    if file_digest(subset/relative)!=complete['checksums'][relative]:raise ValueError('Saved historical classification changed')
    store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve up to15MB for this new comparison')
    protocol=dict(spec=SPEC,cache=file_digest(cache/'features/manifest.json'),reference=file_digest(subset/relative),source=file_digest(subset/'complete.json'),
        code={f:file_digest(ROOT/f) for f in ['scripts/zuco_original_cohort_sweep.py','scripts/zuco_averaging_sweep.py','scripts/zuco_recovered_followups.py']},software={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']})
    store.json('protocol.json',protocol)
    if (out/'complete.json').exists():
        for f,sha in b.read(out/'complete.json')['checksums'].items():
            if file_digest(out/f)!=sha:raise ValueError('Completed comparison changed')
        print('KEEP COMPLETED ORIGINAL-COHORT SWEEP',flush=True);return
    with tempfile.TemporaryDirectory(prefix='original-cohort-') as tmp:
        work=Path(tmp);x,meta,_=b.load_features(parent,subset,SimpleNamespace(root=cache),work)
        subjects=b.read(parent/'cpu_preparation_manifest.json')['subjects'];order=[]
        for subject in subjects:
            part=meta[meta.subject.eq(subject)].sort_values('parent_row',kind='stable');order.extend(part.index.tolist())
        if sorted(order)!=list(range(len(meta))):raise ValueError('Original reader order is incomplete')
        x=x[order];meta=meta.iloc[order].reset_index(drop=True)
        run(x,meta,b.read(subset/relative),store,work)
    print('ORIGINAL-COHORT SWEEP COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
