#!/usr/bin/env python3
"""Bounded, nested TeCo input/model sweep on frozen saved positive/negative trials."""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import warnings
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_spatial40 as s
import teco_average40 as old
import analyze_trial_archive as a
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

SEED=20261118
SIZES=[1,5,10,20,40]
MODES=['reader','sentence','word','mixed']
MODELS=['lr_0.01','lr_0.1','lr_1','lr_10','lda','svm','trees','mlp']
WINDOWS=[(0,200),(100,200),(200,300),(224,304),(300,500),(500,800),(0,500),(0,800)]
SPEC=dict(version=1,seed=SEED,sizes=SIZES,modes=MODES,models=MODELS,windows=WINDOWS,
    data='Frozen original TeCo task1 positive/negative first-stored-fixation roster; saved40Hz/125Hz epochs',
    outer='3 folds; each test fold excluded from input/model selection',inner='2 folds within outer training events',
    selection='screen every EEG input using C1 logistic; top3 inputs advanced to8 model settings; winner by inner pooled BA',
    averaging='exact same-class within reader and partition, discard remainder; sizes are separate prediction targets',
    purge='remove training events within1000ms of validation/test events in same reader/sentence, including inner folds',
    control='pre-onset, word/fixation properties and dummy evaluated on identical memberships',
    residualization='ridge nuisance regression fitted on training examples only',
    limitations=['Outer folds were used in earlier experiments; this is new within-run holdout, not fresh data',
    'Mixed split permits familiar readers/words/sentences, but never same event or overlapping epoch across partitions',
    'Known labels construct test averages; not individual-word deployment accuracy',
    'No neutral class, new labeling, regression deconvolution, source re-extraction or significance search',
    'Checkpointed sweep candidates are development scores; final scores evaluate selection procedures, not test-ranked winners',
    'Epoch purging does not remove all filtering dependence or correlation between related observations'],budget=50_000_000)


def read(path):return json.loads(Path(path).read_text())


def compressed(store,path,**arrays):
    b=BytesIO();np.savez_compressed(b,**arrays);store.bytes(path,b.getvalue())


def feature_set(eeg,times,montage):
    """Return reusable trial vectors, not a second saved full EEG archive."""
    picks=[np.flatnonzero(montage.region.to_numpy()==r) for r in a.REGIONS]
    regional=np.stack([eeg[:,p].mean(axis=1) for p in picks],axis=1)
    out={}
    for start,end in WINDOWS:
        mask=(times>=start)&(times<=end if (start,end)==(224,304) else times<end)
        out[f'channels_{start}_{end}']=eeg[:,:,mask].mean(axis=2)
    bins=[(i,i+100) for i in range(0,800,100)]
    out['regional_bins']=np.concatenate([regional[:,:,(times>=lo)&(times<hi)].mean(axis=2) for lo,hi in bins],axis=1)
    out['regional_early_bins']=out['regional_bins'][:,:18]
    out['regional_late_bins']=out['regional_bins'][:,18:]
    for r,p in zip(a.REGIONS,picks):out['region_'+r]=out['channels_224_304'][:,p]
    mask=(times>=0)&(times<800);v=regional[:,:,mask]
    out['regional_shape']=np.concatenate([v.mean(2),v.std(2),np.ptp(v,axis=2),np.sqrt(np.mean(v*v,axis=2))],axis=1)
    # Log band energy on the short fixed epoch: descriptive features, not precise spectral estimates.
    v=(v-v.mean(axis=2,keepdims=True))*np.hanning(v.shape[2])
    power=np.abs(np.fft.rfft(v,axis=2))**2;freq=np.fft.rfftfreq(v.shape[2],1/125)
    out['regional_log_band_energy']=np.concatenate([np.log1p(power[:,:,(freq>=lo)&(freq<hi)].mean(2)) for lo,hi in [(2,4),(4,8),(8,13),(13,30)]],axis=1)
    out['gfp_bins']=np.stack([eeg[:,:,(times>=lo)&(times<hi)].std(axis=1).mean(axis=1) for lo,hi in bins],axis=1)
    out['pre_control']=eeg[:,:,(times>=-200)&(times<0)].mean(axis=2)
    return {k:np.asarray(v,dtype=np.float32) for k,v in out.items()}


def load_features(thesis,meta,montage,store):
    archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz';frames=[]
    subjects=sorted(meta.subject.unique());times=np.arange(-25,101)*8
    for ri,subject in enumerate(subjects,1):
        rows=meta[meta.subject==subject];folder=archive/'teco/readers/task1'/subject
        prefix=f'features/{subject}';marker=store.root/prefix/'complete.json';cache=store.root/prefix/'vectors.npz'
        identity=dict(checkpoint=a.file_digest(folder/'checkpoint.json'),events=rows.event_id.tolist(),montage=montage.to_dict('list'),spec=SPEC)
        if marker.is_file():
            saved=read(marker)
            if saved['identity']!=json.loads(json.dumps(identity)) or a.file_digest(cache)!=saved['sha256']:raise ValueError('Changed/corrupt sweep feature cache')
            with np.load(cache,allow_pickle=False) as f:
                if f['event_id'].tolist()!=rows.event_id.tolist():raise ValueError('Changed sweep row order')
                features={k:f[k].copy() for k in f.files if k!='event_id'}
            print(f'KEEP FEATURES reader {ri}/{len(subjects)}: {subject}; no epoch reads',flush=True)
        else:
            print(f'PREPARE FEATURES reader {ri}/{len(subjects)}: {subject}, {len(rows)} selected saved epochs',flush=True)
            state=read(folder/'checkpoint.json');checks={k:v for b in state['blocks'] for k,v in b['checksums'].items()}
            eeg=np.empty((len(rows),len(montage),len(times)),np.float32);positions={ix:j for j,ix in enumerate(rows.index)};seen=set()
            files=list(rows.groupby('epoch_file',sort=False))
            for fi,(relative,g) in enumerate(files,1):
                payload=a.strict_path(folder,relative).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=checks[relative]:raise ValueError('Corrupt source shard')
                with np.load(BytesIO(payload),allow_pickle=False) as f:
                    indices=g.epoch_row.astype(int).to_numpy()
                    if (indices<0).any() or (indices>=len(f['event_id'])).any():raise ValueError('Epoch index outside shard')
                    if f['event_id'][indices].astype(str).tolist()!=g.event_id.tolist() or f['channel_names'].astype(str).tolist()!=montage.channel.astype(str).tolist() or not np.array_equal(f['times_ms'],times):raise ValueError('EEG identity/montage/time mismatch')
                    v=f['eeg_uV'][indices]
                    hashes=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in v]
                    if hashes!=g.signal_sha256.tolist() or not np.isfinite(v).all():raise ValueError('EEG differs from frozen validated cache')
                    dst=[positions[ix] for ix in g.index]
                    if seen.intersection(dst):raise ValueError('Repeated event')
                    seen.update(dst);eeg[dst]=v
                if fi==1 or fi%20==0 or fi==len(files):print(f'  reader {ri}/{len(subjects)} {subject}: {fi}/{len(files)} shards',flush=True)
            if len(seen)!=len(rows):raise ValueError('Incomplete feature rows')
            features=feature_set(eeg,times,montage);del eeg
            compressed(store,prefix+'/vectors.npz',event_id=rows.event_id.to_numpy(dtype=str),**features)
            store.json(prefix+'/complete.json',dict(identity=identity,sha256=a.file_digest(cache)))
            print(f'SAVED FEATURES reader {ri}/{len(subjects)}: {subject}',flush=True)
        if any(len(v)!=len(rows) or not np.isfinite(v).all() for v in features.values()):raise ValueError('Invalid compact features')
        frames.append((rows.index.to_numpy(),features))
    result={}
    for ix,features in frames:
        for k,v in features.items():
            if k not in result:result[k]=np.empty((len(meta),v.shape[1]),np.float32)
            result[k][ix]=v
    return result


class NuisanceResidual(BaseEstimator,TransformerMixin):
    """Last six columns are nuisance covariates, never direct classifier inputs."""
    def fit(self,X,y=None):
        from sklearn.linear_model import Ridge
        self.regressor_=Ridge(alpha=10).fit(X[:,-6:],X[:,:-6]);return self
    def transform(self,X):return X[:,:-6]-self.regressor_.predict(X[:,-6:])


def estimator(name,residual=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.dummy import DummyClassifier
    if name.startswith('lr_'):model=LogisticRegression(C=float(name[3:]),class_weight='balanced',solver='liblinear',max_iter=1500,random_state=SEED)
    elif name=='lda':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5])
    elif name=='svm':model=SVC(C=1,kernel='rbf',class_weight='balanced')
    elif name=='trees':model=ExtraTreesClassifier(n_estimators=64,max_depth=8,min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=SEED)
    elif name=='mlp':model=MLPClassifier(hidden_layer_sizes=(32,),alpha=1,max_iter=250,early_stopping=False,random_state=SEED)
    elif name=='dummy':model=DummyClassifier(strategy='prior')
    else:raise ValueError(name)
    steps=[SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
    if residual:steps.append(NuisanceResidual())
    return make_pipeline(*steps,model)


def folds(meta,y,indices,mode,n,seed):
    from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
    if mode=='mixed':
        # Stratify within reader/class where possible. No held-out identity requirement.
        strata=meta.subject.iloc[indices].astype(str).to_numpy()+'_'+y[indices].astype(str)
        splitter=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
        pairs=list(splitter.split(np.zeros(len(indices)),strata))
    else:
        col={'reader':'subject','sentence':'sentence_key','word':'word_identity'}[mode]
        groups=meta[col].iloc[indices].astype(str).to_numpy()
        pairs=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(indices)),y[indices],groups))
        if any(set(groups[tr])&set(groups[te]) for tr,te in pairs):raise AssertionError('Group leakage')
    result=[]
    for tr,te in pairs:
        tr,te=indices[tr],indices[te]
        # Conservatively purge complete epoch spans, same rule for every input.
        clean,removed=a.purge_overlapping_train(meta,y,[(tr,te)])
        tr=clean[0][0]
        if set(y[tr])!={0,1} or set(y[te])!={0,1}:raise ValueError('Class coverage missing after split/purge')
        result.append((tr,te,removed[0]))
    return result


def memberships(meta,y,indices,size,seed):
    # Exact groups only; unlike deployment, grouping uses known class labels.
    return s.memberships(meta,y,indices,size,seed)


def score_one(meta,y,X,groups,name,residual=False,detailed=True):
    train,test=groups
    yt=y[train[:,0]];ye=y[test[:,0]]
    if set(yt)!={0,1} or set(ye)!={0,1} or min(np.bincount(ye,minlength=2))<2:raise ValueError('Insufficient exact-group class coverage')
    model=estimator(name,residual)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(s.averaged(X,train),yt)
    pred=model.predict(s.averaged(X,test))
    return dict(balanced_accuracy=float(balanced_accuracy_score(ye,pred)),**(dict(truth=ye.astype(int).tolist(),prediction=pred.astype(int).tolist(),
        readers=meta.subject.iloc[test[:,0]].astype(str).tolist()) if detailed else {}),test_examples=len(ye),test_trials=int(test.size),train_examples=len(yt),
        confusion=confusion_matrix(ye,pred,labels=[0,1]).tolist(),warnings=sorted(set(str(w.message) for w in caught)))


def save_case(store,path,fn):
    path=path+'/result.json';marker=store.root/path
    if marker.is_file():
        result=read(marker);payload={k:v for k,v in result.items() if k!='checksum'}
        if result['checksum']!=a.digest(payload):raise ValueError('Corrupt fit checkpoint')
        return payload
    try:payload=dict(status='completed',**fn())
    except ValueError as e:payload=dict(status='unavailable',reason=str(e))
    store.json(path,dict(**payload,checksum=a.digest(payload)))
    return payload


def summarize(results):
    if not results or any(r['status']!='completed' for r in results):return None
    matrix=np.sum([r['confusion'] for r in results],axis=0)
    return float(np.mean(np.diag(matrix)/matrix.sum(axis=1)))


def evaluate(meta,features,store,modes=MODES,sizes=SIZES,models=MODELS,outer_n=3,inner_n=2):
    y=meta.truth.eq('positive').to_numpy(dtype=int);all_indices=np.arange(len(meta));summary=[];selection=[]
    inputs=sorted(k for k in features if k not in ['nuisance','pre_control'])
    for mode in modes:
        outer=folds(meta,y,all_indices,mode,outer_n,SEED)
        for fi,(train,test,purged) in enumerate(outer):
            inner=folds(meta,y,train,mode,inner_n,SEED+fi+100)
            for size in sizes:
                prefix=f'fits/{mode}/fold{fi}/size{size}'
                print(f'SWEEP {mode}: outer fold {fi+1}/{len(outer)}, average={size}; train={len(train)}, test={len(test)}, purged={purged}',flush=True)
                def group(tr,te,seed):return memberships(meta,y,tr,size,seed),memberships(meta,y,te,size,seed+100)
                try:inner_groups=[group(tr,te,SEED+fi*100+j) for j,(tr,te,_) in enumerate(inner)]
                except ValueError as e:
                    selection.append(dict(mode=mode,fold=fi,size=size,status='unavailable',reason=str(e)));continue
                for j,((tr,te,pr),(gt,ge)) in enumerate(zip(inner,inner_groups)):
                    store.json(prefix+f'/inner{j}/membership.json',dict(train=gt.tolist(),validation=ge.tolist(),purged=pr))
                dev=[]
                def development(inp,model,stage):
                    cases=[save_case(store,prefix+f'/dev/{inp}/{model}/inner{j}',lambda gs=gs:score_one(meta,y,features[inp],gs,model,inp.startswith('residual_'),detailed=False)) for j,gs in enumerate(inner_groups)]
                    value=summarize(cases)
                    row=dict(mode=mode,fold=fi,size=size,input=inp,model=model,stage=stage,validation_ba=value,status='completed' if value is not None else 'unavailable')
                    dev.append(row);return value
                for ci,inp in enumerate(inputs):
                    development(inp,'lr_1','input_screen')
                    if ci%5==0 or ci==len(inputs)-1:print(f'  INPUT SCREEN {ci+1}/{len(inputs)}; {mode} fold{fi+1} avg{size}',flush=True)
                ranked=sorted([r for r in dev if r['validation_ba'] is not None],key=lambda r:(-r['validation_ba'],r['input']))
                if not ranked:
                    selection.append(dict(mode=mode,fold=fi,size=size,status='unavailable',reason='No valid development cases'));continue
                for inp in [r['input'] for r in ranked[:3]]:
                    for name in models:
                        if name!='lr_1':development(inp,name,'model_search')
                    print('  MODEL SEARCH finished:',inp,flush=True)
                ranked=sorted([r for r in dev if r['validation_ba'] is not None],key=lambda r:(-r['validation_ba'],r['input'],r['model']))
                best=ranked[0];selection.extend(dev)
                try:gt,ge=group(train,test,SEED+1000+fi)
                except ValueError as e:
                    selection.append(dict(mode=mode,fold=fi,size=size,status='unavailable',reason=str(e)));continue
                store.json(prefix+'/outer_membership.json',dict(train=gt.tolist(),test=ge.tolist(),purged=purged,
                    train_event_sha256=a.digest(meta.event_id.iloc[gt.ravel()].tolist()),test_event_sha256=a.digest(meta.event_id.iloc[ge.ravel()].tolist()),
                    train_readers=sorted(meta.subject.iloc[gt.ravel()].unique()),test_readers=sorted(meta.subject.iloc[ge.ravel()].unique())))
                for variant,inp,name in [('selected_eeg',best['input'],best['model']),('fixed_reference','channels_224_304','lr_1'),('pre_control','pre_control','lr_1'),('word_eye_control','nuisance','lr_1'),('dummy','channels_224_304','dummy')]:
                    result=save_case(store,prefix+'/test/'+variant,lambda:score_one(meta,y,features[inp],(gt,ge),name,inp.startswith('residual_')))
                    summary.append(dict(mode=mode,fold=fi,size=size,variant=variant,input=inp,model=name,**result))
                store.json(prefix+'/selection.json',dict(chosen=best,selection_used_outer_test=False))
                print('SAVED OUTER TEST:',mode,'fold',fi+1,'average',size,'selected',best['input'],best['model'],flush=True)
    return summary,selection


def report(rows,selection,store):
    old.csv(store,'development_scores.csv',pd.DataFrame(selection));store.json('outer_results.json',rows)
    combined=[]
    for (mode,size,variant),g in pd.DataFrame(rows).groupby(['mode','size','variant']):
        complete=g[g.status.eq('completed')];rec=dict(mode=mode,size=int(size),variant=variant,completed_folds=len(complete),expected_folds=3)
        if len(complete)==3:
            truth=np.concatenate(complete.truth);pred=np.concatenate(complete.prediction)
            rec.update(balanced_accuracy=float(balanced_accuracy_score(truth,pred)),test_examples=len(truth),test_trials=int(complete.test_trials.sum()),
                fold_min=float(complete.balanced_accuracy.min()),fold_max=float(complete.balanced_accuracy.max()))
        combined.append(rec)
    old.csv(store,'summary.csv',pd.DataFrame(combined))
    # No broad search for p-values. Curve and heatmap are descriptive only.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(combined);fig,axes=plt.subplots(1,4,figsize=(17,4),sharey=True)
    for ax,mode in zip(axes,MODES):
        for variant in ['selected_eeg','fixed_reference','pre_control','word_eye_control','dummy']:
            g=frame[(frame['mode']==mode)&frame.variant.eq(variant)&frame.completed_folds.eq(3)]
            if len(g):ax.plot(g['size'],g.balanced_accuracy*100,'o-',label=variant)
        ax.axhline(50,color='grey',ls=':');ax.set(title=mode,xlabel='Exact trials / average',ylim=(0,100))
    axes[0].set_ylabel('Outer-test balanced accuracy (%)');axes[-1].legend(fontsize=7)
    fig.suptitle('TeCo task1: nested input/model selection; known labels form averages');fig.tight_layout()
    b=BytesIO();fig.savefig(b,format='png',dpi=130);plt.close(fig);store.bytes('sweep.png',b.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    out=args.thesis_root/'Results/word_sentiment_eeg/teco_classification_sweep';store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve at most50MB new output space; no epoch reads started')
    import fcntl,tempfile
    lock=Path(tempfile.gettempdir(),'teco-sweep-'+a.digest(str(out))[:16]+'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta,baseline,source=old.load_cached(args.thesis_root);meta,baseline,counts=s.cohort(meta,baseline)
    archive=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz';montage=pd.read_csv(archive/'teco_channel_regions.csv')
    if montage.channel.duplicated().any() or set(montage.region)!=set(a.REGIONS):raise ValueError('Invalid montage')
    store.json('protocol.json',dict(spec=SPEC,source=source,code_sha256={f:a.file_digest(ROOT/f) for f in ['scripts/teco_classification_sweep.py','scripts/teco_average40.py','scripts/teco_spatial40.py','scripts/analyze_trial_archive.py','scripts/zuco_recovered_followups.py']},versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']}))
    if (out/'complete.json').is_file():
        done=read(out/'complete.json')
        for name,sha in done['checksums'].items():
            if a.file_digest(out/name)!=sha:raise ValueError('Completed sweep checksum failed')
        print('KEEP COMPLETE SWEEP: all saved outputs verified; nothing refitted',flush=True);return
    old.csv(store,'cohort.csv',counts);old.csv(store,'events.csv.gz',meta)
    features=load_features(args.thesis_root,meta,montage,store);features['nuisance']=baseline['nuisance'].astype(np.float32)
    for key in ['channels_224_304','regional_bins']:features['residual_'+key]=np.concatenate([features[key],features['nuisance']],axis=1)
    print('FEATURES READY:',len(meta),'trials;',meta.subject.nunique(),'readers;',len(features)-2,'EEG inputs; 4 evaluation modes',flush=True)
    rows,selection=evaluate(meta,features,store)
    if not rows:raise ValueError('No usable outer evaluation cases; see per-case checkpoints')
    report(rows,selection,store)
    paths=['protocol.json','cohort.csv','events.csv.gz','summary.csv','development_scores.csv','outer_results.json','sweep.png']+[str(p.relative_to(out)) for p in out.rglob('result.json')]+[str(p.relative_to(out)) for p in out.glob('features/*/complete.json')]
    store.json('complete.json',dict(status='completed',outer_rows=len(rows),checksums={f:a.file_digest(out/f) for f in paths}))
    print('TECO SWEEP COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
