"""Checkpointed published-grid decoding with documented proportional split rounding.

This reconstructs a classification protocol, not the authors' unknown trial IDs,
seeds, length/sentence averaging algorithm, CI formula or upstream MATLAB code.
"""
from pathlib import Path
import tempfile,time,warnings,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score
from sklearn.exceptions import ConvergenceWarning
from .common import digest,file_digest
from .paper_decoding import NCA_COMPONENTS,REGULARIZATION


def read(path):
    import json
    return json.loads(Path(path).read_text())


def apportion(quota,total,lower,upper,rng):
    quota=np.asarray(quota,float);lo=np.broadcast_to(lower,quota.shape).astype(int);hi=np.broadcast_to(upper,quota.shape).astype(int)
    if (lo>hi).any() or not lo.sum()<=total<=hi.sum():raise ValueError('Infeasible documented integer allocation')
    out=np.clip(np.floor(quota).astype(int),lo,hi)
    tie=rng.random(len(out))*1e-10
    while out.sum()<total:
        score=np.where(out<hi,quota-out+tie,-np.inf);out[np.argmax(score)]+=1
    while out.sum()>total:
        score=np.where(out>lo,out-quota+tie,-np.inf);out[np.argmax(score)]-=1
    return out


def random_split(meta,seed,test_fraction=.05):
    """Round reader and class margins, without requiring all24joint cells in test.

    Each reader remains in train/test when n_test permits. Reader-class cells
    may be absent from test:~17test examples cannot cover24joint cells.
    This is an explicit reconstruction choice, not a recovered author algorithm.
    """
    rng=np.random.default_rng(seed);n=len(meta);ntest=int(np.ceil(n*test_fraction));subjects=sorted(meta.subject.astype(str).unique())
    counts=np.array([[int(((meta.subject.astype(str)==s)&meta.sentiment.eq(c)).sum()) for c in ['negative','positive']] for s in subjects])
    totals=counts.sum(axis=1)
    if (totals<2).any() or ntest<len(subjects):raise ValueError('Not enough test averages to keep each reader represented; retain95/5 and report coverage failure')
    nr=apportion(totals*ntest/n,ntest,np.ones(len(subjects),int),totals-1,rng)
    negtarget=int(np.floor(counts[:,0].sum()*ntest/n+.5));lower=np.maximum(0,nr-counts[:,1]);upper=np.minimum(nr,counts[:,0])
    neg=apportion(nr*counts[:,0]/totals,negtarget,lower,upper,rng)
    test=[];rows=[]
    for i,s in enumerate(subjects):
        for j,c in enumerate(['negative','positive']):
            ix=np.flatnonzero((meta.subject.astype(str).to_numpy()==s)&meta.sentiment.eq(c).to_numpy());take=int(neg[i] if j==0 else nr[i]-neg[i])
            test.extend(rng.choice(ix,size=take,replace=False).tolist());rows.append(dict(reader=s,sentiment=c,available=len(ix),test=take,ideal=float(len(ix)*ntest/n)))
    test=np.array(sorted(test),int);train=np.setdiff1d(np.arange(n),test)
    if len(test)!=ntest or set(train)&set(test) or len(set(meta.iloc[test].sentiment))!=2 or len(set(meta.iloc[train].sentiment))!=2:raise ValueError('Bad random split')
    if set(meta.iloc[train].subject)!=set(subjects) or set(meta.iloc[test].subject)!=set(subjects):raise ValueError('Reader coverage changed')
    return train,test,dict(seed=seed,test_fraction_requested=test_fraction,test_count=len(test),test_fraction_actual=len(test)/n,allocation=rows,
        method='bounded proportional reader/class marginal rounding; no held-out identity groups')


def averages(features,meta,size,seed=2026):
    """Length-sorted within reader/class, distinct sentences per average, keep tails.

    'Similar length' and sentence constraints are explicit interpretations of the
    published prose. They are not verified author bin boundaries or memberships.
    """
    rng=np.random.default_rng(seed);values=[];rows=[];members=[]
    if meta.event_id.duplicated().any() or len(features)!=len(meta):raise ValueError('Invalid input event roster')
    for (subject,label),g in meta.groupby(['subject','sentiment'],sort=True):
        shuffled=rng.permutation(g.index.to_numpy());queue=list(shuffled[np.argsort(meta.iloc[shuffled].word_length.to_numpy(float),kind='stable')])
        while queue:
            block=[];later=[];sentences=set()
            for i in queue:
                sentence=str(meta.iloc[i].sentence_key)
                if len(block)<size and sentence not in sentences:block.append(i);sentences.add(sentence)
                else:later.append(i)
            if not block:raise ValueError('Averaging made no progress')
            group=f'average_{len(rows):06d}';part=meta.iloc[block]
            values.append(np.mean(features[block],axis=0,dtype=np.float64).astype('float32'))
            rows.append(dict(average_id=group,subject=str(subject),sentiment=str(label),actual_trials=len(block),is_remainder=len(block)<size,
                min_word_length=float(part.word_length.min()),max_word_length=float(part.word_length.max())))
            for i in block:members.append(dict(average_id=group,event_id=str(meta.iloc[i].event_id),subject=str(subject),sentiment=str(label),sentence_key=str(meta.iloc[i].sentence_key),word=str(meta.iloc[i].word)))
            queue=later
    membership=pd.DataFrame(members)
    if len(membership)!=len(meta) or membership.event_id.duplicated().any():raise ValueError('Dropped/repeated average membership')
    return np.stack(values),pd.DataFrame(rows),membership


def fit_classifier(name,c,gamma=None):
    if name=='logistic':return LogisticRegression(C=float(c),solver='lbfgs',max_iter=2000,random_state=2026)
    if name=='svm_rbf':return SVC(C=float(c),gamma=float(gamma),kernel='rbf')
    raise ValueError('Unknown classifier')


def nca_transform(a,y,z,k,seed,max_iter=100):
    # Same scikit-learn NCA implementation as the published protocol. No fixed
    # component shortcut and no GPU surrogate. Preserve default auto init/tol.
    model=NeighborhoodComponentsAnalysis(n_components=k,random_state=seed,max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');aa=model.fit_transform(a,y);zz=model.transform(z)
    if not np.isfinite(aa).all() or not np.isfinite(zz).all():raise ValueError('Nonfinite NCA projection')
    return aa,zz,dict(iterations=int(model.n_iter_),converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),warnings=sorted(set(str(w.message) for w in caught)))


def nested_grid(x,y,train,store,relative,seed,*,dims=NCA_COMPONENTS,cs=REGULARIZATION,gammas=REGULARIZATION,inner_folds=10,nca_max_iter=100):
    if min(np.bincount(y[train],minlength=2))<inner_folds:raise ValueError('Inadequate class counts for published inner10folds')
    splitter=StratifiedKFold(inner_folds,shuffle=True,random_state=seed);blocks={};all_converged=True
    for fi,(fit,valid) in enumerate(splitter.split(np.zeros(len(train)),y[train])):
        fit,valid=train[fit],train[valid];scaler=None;a=z=None
        for k in dims:
            rel=relative/f'inner{fi:02d}_nca{k:04d}.json';path=store.root/rel
            identity=digest(dict(fit=fit.tolist(),valid=valid.tolist(),components=k,cs=list(cs),gammas=list(gammas),seed=seed,max_iter=nca_max_iter))
            if path.exists():
                block=read(path)
                if block['identity']!=identity:raise ValueError('Different saved grid block')
                print(f'KEEP GRID {relative}: fold{fi+1}/{inner_folds},NCA{k}',flush=True)
            else:
                if scaler is None:
                    scaler=StandardScaler();a=scaler.fit_transform(x[fit]);z=scaler.transform(x[valid])
                print(f'FIT NCA {relative}: fold{fi+1}/{inner_folds},components{k}; one transform reused across all classifier parameters',flush=True);start=time.monotonic()
                aa,zz,convergence=nca_transform(a,y[fit],z,k,seed+fi,nca_max_iter);lr=[];svm=[];classifier_ok=True
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    for c in cs:
                        m=fit_classifier('logistic',c);m.fit(aa,y[fit]);lr.append(float(accuracy_score(y[valid],m.predict(zz))))
                        row=[]
                        for gamma in gammas:
                            m=fit_classifier('svm_rbf',c,gamma);m.fit(aa,y[fit]);row.append(float(accuracy_score(y[valid],m.predict(zz))))
                        svm.append(row)
                classifier_ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                block=dict(identity=identity,logistic=lr,svm_rbf=svm,nca=convergence,classifiers_converged=classifier_ok,seconds=time.monotonic()-start)
                store.json(rel,block);print(f'GRID BLOCK SAVED: {block["seconds"]:.1f}s; NCA converged={convergence["converged"]}',flush=True)
                del aa,zz
            all_converged &=block['nca']['converged'] and block['classifiers_converged'];blocks[fi,k]=block
    best={}
    for name in ['logistic','svm_rbf']:
        score=np.stack([np.mean([np.asarray(blocks[fi,k][name]) for fi in range(inner_folds)],axis=0) for k in dims])
        ix=np.unravel_index(int(np.argmax(score)),score.shape)
        best[name]=dict(components=int(dims[ix[0]]),C=float(cs[ix[1]]),gamma=float(gammas[ix[2]]) if name=='svm_rbf' else None,validation_accuracy=float(score[ix]),all_grid_blocks_converged=bool(all_converged))
    return best


def run_repeat(x,meta,membership,store,relative,repeat,*,dims=NCA_COMPONENTS,cs=REGULARIZATION,gammas=REGULARIZATION,inner_folds=10,nca_max_iter=100):
    signature=digest(dict(shape=list(x.shape),features_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest(),metadata=meta.to_dict('records'),membership=membership.to_dict('records'),repeat=repeat,dims=list(dims),cs=list(cs),gammas=list(gammas),inner_folds=inner_folds,nca_max_iter=nca_max_iter))
    done=store.root/relative/'result.json'
    if done.exists():
        saved=read(done)
        if saved['signature']!=signature or file_digest(store.root/relative/'predictions.csv.gz')!=saved['predictions_sha256']:raise ValueError('Completed repeat changed')
        print(f'KEEP REPEAT {relative}',flush=True);return saved
    train,test,allocation=random_split(meta,2026+repeat);y=meta.sentiment.eq('positive').to_numpy(int)
    train_ids=set(meta.iloc[train].average_id);test_ids=set(meta.iloc[test].average_id)
    tr=membership[membership.average_id.isin(train_ids)];te=membership[membership.average_id.isin(test_ids)]
    if set(tr.event_id)&set(te.event_id):raise ValueError('An original event crossed train/test')
    store.json(relative/'split.json',dict(signature=signature,allocation=allocation,train=meta.iloc[train].average_id.tolist(),test=meta.iloc[test].average_id.tolist(),
        readers_shared=sorted(set(meta.iloc[train].subject)&set(meta.iloc[test].subject)),words_shared=len(set(tr.word)&set(te.word)),sentences_shared=len(set(tr.sentence_key)&set(te.sentence_key)),
        original_event_overlap=0,distinct_event_raw_interval_overlap_not_purged=True))
    best=nested_grid(x,y,train,store,relative/'grid',2026+repeat,dims=dims,cs=cs,gammas=gammas,inner_folds=inner_folds,nca_max_iter=nca_max_iter)
    scaler=StandardScaler();a=scaler.fit_transform(x[train]);z=scaler.transform(x[test]);scores=[];predictions=[];transforms={}
    for name in ['logistic','svm_rbf']:
        selection=best[name];k=selection['components']
        if k not in transforms:transforms[k]=nca_transform(a,y[train],z,k,2026+repeat,nca_max_iter)
        aa,zz,convergence=transforms[k];m=fit_classifier(name,selection['C'],selection['gamma'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');m.fit(aa,y[train]);pred=m.predict(zz)
        classifier_converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        scores.append(dict(model=name,repeat=repeat,accuracy=float(accuracy_score(y[test],pred)),balanced_accuracy=float(balanced_accuracy_score(y[test],pred)),
            f1=float(f1_score(y[test],pred,zero_division=0)),precision=float(precision_score(y[test],pred,zero_division=0)),recall=float(recall_score(y[test],pred,zero_division=0)),
            train_examples=len(train),test_examples=len(test),best_parameters=selection,refit_nca=convergence,classifier_converged=classifier_converged))
        predictions.append(pd.DataFrame(dict(model=name,average_id=meta.iloc[test].average_id.to_numpy(),truth=y[test],prediction=pred)))
    dummy=DummyClassifier(strategy='prior').fit(a,y[train]).predict(z)
    scores.append(dict(model='dummy',repeat=repeat,accuracy=float(accuracy_score(y[test],dummy)),balanced_accuracy=float(balanced_accuracy_score(y[test],dummy)),test_examples=len(test)))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'predictions.csv.gz';pd.concat(predictions).to_csv(path,index=False,compression={'method':'gzip','mtime':0});sha=store.file(relative/'predictions.csv.gz',path)
    result=dict(signature=signature,scores=scores,predictions_sha256=sha,allocation=allocation);store.json(relative/'result.json',result);return result


def summary(rows):
    frame=pd.DataFrame(rows);result=[]
    for (size,model),g in frame.groupby(['size','model'],sort=True):
        vals=g.accuracy.to_numpy(float);n=len(vals)
        review=0
        for row in g.to_dict('records'):
            best=row.get('best_parameters');refit=row.get('refit_nca')
            review+=int((isinstance(best,dict) and not best['all_grid_blocks_converged']) or (isinstance(refit,dict) and not refit['converged']) or row.get('classifier_converged') is False)
        half=float(t.ppf(.975,n-1)*vals.std(ddof=1)/np.sqrt(n)) if n>1 else None
        result.append(dict(size=int(size),model=model,completed_splits=n,numerical_review_splits=review,mean_accuracy=float(vals.mean()),mean_balanced_accuracy=float(g.balanced_accuracy.mean()),
            split_mean_t_lower=max(0,float(vals.mean()-half)) if half is not None else None,split_mean_t_upper=min(1,float(vals.mean()+half)) if half is not None else None,
            minimum=float(vals.min()),maximum=float(vals.max()),note='Descriptive t interval over correlated random splits; not independent-reader uncertainty; author CI formula unavailable'))
    return pd.DataFrame(result)
