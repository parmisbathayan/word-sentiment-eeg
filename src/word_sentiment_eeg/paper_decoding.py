"""Checkpointed paper-style decoding components, not a complete reproduction.

Requires already selected/paired-cleaned and membership-audited trial averages.
No raw EEG reads or guessed label/timing choices. Full published grid searches
are planned before execution and must fit an explicitly supplied fit budget.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .common import digest,file_digest,write_json

NCA_COMPONENTS=(10,20,40,80,160,320,640,1280)
REGULARIZATION=tuple(float(x) for x in np.logspace(-7,2,10))


def decoding_plan(mode,repeats=100,inner_folds=10,components=None):
    if mode not in {'reported_fixed','published_grid'}:raise ValueError('Unknown named decoding mode')
    if repeats<1 or inner_folds<2:raise ValueError('Invalid repeat/fold counts')
    dims=tuple(components or ([320] if mode=='reported_fixed' else NCA_COMPONENTS))
    if any(not isinstance(k,int) or k<1 for k in dims):raise ValueError('Invalid NCA dimensions')
    fits_per_repeat=2 if mode=='reported_fixed' else len(dims)*110*inner_folds+2
    return dict(mode=mode,repeats=repeats,inner_folds=inner_folds,nca_components=list(dims),
                classifier_fits=repeats*fits_per_repeat,
                note='reported_fixed uses C=1, SVM gamma=.03 and the reported median NCA size, not the optimized Table 1 configuration')


def split_averages(meta,stratification,repeats=100,test_fraction=.05,seed=2026):
    from sklearn.model_selection import StratifiedShuffleSplit
    if stratification not in {'sentiment','joint_reader_sentiment'}:
        raise ValueError('Choose explicit stratification; joint feasibility must not silently change the test fraction')
    y=meta.sentiment.to_numpy(dtype=str)
    strata=y if stratification=='sentiment' else (meta.subject.astype(str)+'/'+meta.sentiment.astype(str)).to_numpy()
    ntest=int(np.ceil(len(meta)*test_fraction));nstrata=len(set(strata))
    if ntest<nstrata:
        raise ValueError(f'{ntest} test averages cannot cover {nstrata} reader/class strata at {test_fraction:.1%}; resolve paper protocol ambiguity explicitly')
    splitter=StratifiedShuffleSplit(n_splits=repeats,test_size=test_fraction,random_state=seed)
    return list(splitter.split(np.zeros((len(meta),1)),strata))


def estimator(model,mode,components,seed=2026,inner_folds=10,nca_max_iter=50):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV,StratifiedKFold
    classifier=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=seed) if model=='logistic' else SVC(C=1.,gamma=.03,kernel='rbf')
    if model not in {'logistic','svm_rbf'}:raise ValueError('Unknown paper classifier')
    pipe=Pipeline([('scaler',StandardScaler()),
                   ('nca',NeighborhoodComponentsAnalysis(n_components=components[0],random_state=seed,max_iter=nca_max_iter)),
                   ('classifier',classifier)])
    if mode=='reported_fixed':
        if len(components)!=1:raise ValueError('Fixed mode requires one explicitly recorded component count')
        return pipe
    grid={'nca__n_components':list(components),'classifier__C':list(REGULARIZATION)}
    if model=='svm_rbf':grid['classifier__gamma']=list(REGULARIZATION)
    # The entire scaler/NCA pipeline is refit within each inner training fold.
    # This is costly but does not leak inner validation labels into NCA.
    return GridSearchCV(pipe,grid,scoring='accuracy',cv=StratifiedKFold(inner_folds,shuffle=True,random_state=seed),
                        n_jobs=1,error_score='raise',refit=True,return_train_score=False)


def run_decoding(features,meta,membership,out,*,stratification,mode='reported_fixed',repeats=100,
                 test_fraction=.05,components=None,inner_folds=10,nca_max_iter=50,max_classifier_fits=10000):
    """Run after input gates; writes small prediction/score checkpoints only.

    Event membership must be disjoint globally. This does not guarantee that
    raw EEG intervals of different fixation events are nonoverlapping.
    """
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score
    from sklearn.exceptions import ConvergenceWarning
    plan=decoding_plan(mode,repeats,inner_folds,components)
    if plan['classifier_fits']>max_classifier_fits:
        raise ValueError(f'Planned {plan["classifier_fits"]:,} classifier fits exceeds declared budget {max_classifier_fits:,}; no training started')
    X=np.asarray(features,dtype=np.float32)
    if X.ndim!=2 or len(X)!=len(meta) or not np.isfinite(X).all():raise ValueError('Invalid decoding feature/event join')
    if max(plan['nca_components'])>X.shape[1]:raise ValueError('NCA components exceed EEG feature dimension')
    if meta.average_id.duplicated().any() or membership.event_id.duplicated().any():raise ValueError('Duplicated average/event membership')
    if set(membership.average_id)!=set(meta.average_id):raise ValueError('Average membership roster mismatch')
    if not meta.sentiment.isin(['negative','positive']).all():raise ValueError('Paper binary classes must be negative/positive')
    if len(set(meta.sentiment))!=2:raise ValueError('Both sentiment classes are required')
    lookup=meta.set_index('average_id')
    for identifier,group in membership.groupby('average_id'):
        target=lookup.loc[identifier]
        if set(group.subject)!={target.subject} or set(group.sentiment)!={target.sentiment}:
            raise ValueError('An average mixes readers/classes or its membership labels disagree')
    folds=split_averages(meta,stratification,repeats,test_fraction)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(**plan,test_fraction=test_fraction,stratification=stratification,nca_max_iter=nca_max_iter,
                  seed=2026,features_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                  average_metadata_sha256=digest(meta.to_dict('records')),membership_sha256=digest(membership.to_dict('records')),
                  code_sha256=file_digest(Path(__file__)),
                  raw_interval_overlap_not_prevented=True)
    protocol_path=out/'protocol.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Decoding protocol/data changed; no mixed resume')
    write_json(protocol_path,protocol)
    members={k:set(v.event_id) for k,v in membership.groupby('average_id')}
    y=meta.sentiment.to_numpy(dtype=str);scores=[]
    for ri,(train,test) in enumerate(folds):
        shared_transform=None;shared_warnings=[]
        train_events=set().union(*(members[i] for i in meta.iloc[train].average_id))
        test_events=set().union(*(members[i] for i in meta.iloc[test].average_id))
        if train_events&test_events:raise ValueError('Same original event in train and test averages')
        if mode=='published_grid' and min((y[train]==c).sum() for c in set(y))<inner_folds:
            raise ValueError('Inadequate class coverage for the published 10-fold inner procedure')
        for model_name in ['logistic','svm_rbf']:
            folder=out/f'repeat_{ri:03d}'/model_name;folder.mkdir(parents=True,exist_ok=True)
            mark=folder/'result.json';pred_path=folder/'predictions.csv.gz'
            if mark.is_file():
                saved=json.loads(mark.read_text())
                if file_digest(pred_path)!=saved['predictions_sha256']:raise ValueError('Corrupt saved predictions')
                scores.append(saved['score']);print(f'KEEP decoding repeat {ri+1}/{repeats} {model_name}',flush=True);continue
            print(f'FIT decoding repeat {ri+1}/{repeats} {model_name}',flush=True)
            fitted=estimator(model_name,mode,plan['nca_components'],2026+ri,inner_folds,nca_max_iter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if mode=='reported_fixed':
                    if shared_transform is None:
                        preprocessor=fitted[:-1]
                        shared_transform=(preprocessor.fit_transform(X[train],y[train]),preprocessor.transform(X[test]))
                        shared_warnings=list(caught)
                    classifier=fitted.named_steps['classifier']
                    classifier.fit(shared_transform[0],y[train])
                    pred=classifier.predict(shared_transform[1])
                else:
                    fitted.fit(X[train],y[train]);pred=fitted.predict(X[test])
            all_warnings=[*shared_warnings,*caught]
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in all_warnings)
            score=dict(repeat=ri,model=model_name,train_averages=len(train),test_averages=len(test),
                       accuracy=float(accuracy_score(y[test],pred)),balanced_accuracy=float(balanced_accuracy_score(y[test],pred)),
                       precision=float(precision_score(y[test],pred,pos_label='positive',zero_division=0)),
                       recall=float(recall_score(y[test],pred,pos_label='positive',zero_division=0)),
                       f1=float(f1_score(y[test],pred,pos_label='positive',zero_division=0)),numerically_converged=converged)
            predictions=pd.DataFrame(dict(average_id=meta.iloc[test].average_id.to_numpy(),truth=y[test],prediction=pred))
            predictions.to_csv(pred_path,index=False,compression='gzip')
            write_json(mark,dict(score=score,train_average_ids=meta.iloc[train].average_id.tolist(),
                                 best_parameters=getattr(fitted,'best_params_',{}),warnings=sorted(set(str(w.message) for w in all_warnings)),
                                 predictions_sha256=file_digest(pred_path)))
            scores.append(score)
    frame=pd.DataFrame(scores);frame.to_csv(out/'scores.csv',index=False)
    write_json(out/'manifest.json',dict(status='completed',scope='paper-style decoding component; upstream reproduction gates remain separate',
                                       numerical_review_required=bool((~frame.numerically_converged).any()),cases=len(scores),
                                       finished_at_utc=datetime.now(timezone.utc).isoformat()))
    return frame
