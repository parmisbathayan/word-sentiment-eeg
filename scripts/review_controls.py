"""Occurrence-independent controls and EEG prediction of numeric text properties."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, median_absolute_error, r2_score
import teco_other_targets as t
import linguistic_targets as l

SEED = 20261127
FOUR = ['word_length', 'relative_word_position', 'sentence_length', 'is_control']


def identities(meta):
    """Same textual word occurrence shared across readers; not a word type."""
    columns = ['dataset', 'task', 'sentence_key', 'word_index']
    if not set(columns) <= set(meta):
        raise ValueError('Source word occurrence identity missing')
    return meta[columns].astype(str).agg('/'.join, axis=1)


def metadata_four(meta):
    flag = meta.is_control.astype(str).str.lower()
    control = np.where(flag.isin(['true','1']), 1., np.where(flag.isin(['false','0']), 0., np.nan))
    return np.column_stack([pd.to_numeric(meta[c], errors='coerce') for c in FOUR[:3]] + [control])


def collapse(meta, y, classes):
    """Keep one row per occurrence only if both features and target agree."""
    meta = meta.reset_index(drop=True).copy()
    x = metadata_four(meta); key = identities(meta); keep = []; conflicts = []
    for occurrence, ix in meta.groupby(key, sort=True).groups.items():
        ix = np.asarray(list(ix)); a = x[ix]
        invariant = np.all((a == a[0]) | (np.isnan(a) & np.isnan(a[0])))
        if not invariant or len(set(y[ix])) != 1:
            conflicts.append(dict(occurrence=occurrence, readers=meta.subject.iloc[ix].nunique(),
                                  feature_conflict=not bool(invariant), label_conflict=len(set(y[ix])) != 1))
        else:
            keep.append(int(ix[0]))
    rows = meta.iloc[keep].reset_index(drop=True)
    rows['occurrence_key'] = key.iloc[keep].to_numpy()
    return rows, np.asarray(y)[keep], x[keep], conflicts


def standalone_folds(meta, y, mode):
    if mode == 'occurrence_random':
        raw = StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)
    else:
        group = meta.sentence_key if mode == 'sentence' else meta.word_identity
        raw = StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, group)
    result = []
    for train, test in raw:
        if set(meta.occurrence_key.iloc[train]) & set(meta.occurrence_key.iloc[test]):
            raise AssertionError('Source occurrence leaked across partitions')
        if mode != 'occurrence_random':
            col = 'sentence_key' if mode == 'sentence' else 'word_identity'
            if set(meta[col].iloc[train]) & set(meta[col].iloc[test]):raise AssertionError('Named holdout leaked')
        result.append((train, test))
    return result


def independent_groups(y, indices, size, seed):
    rng = np.random.default_rng(seed); result = []; cap = 40 if size == 'max40' else int(size)
    for c in sorted(set(y)):
        ix = rng.permutation(np.asarray(indices)[y[indices] == c])
        for start in range(0, len(ix), cap):
            group = ix[start:start+cap]
            if len(group) == cap or (size == 'max40' and len(group)):result.append(group)
    return result


def size_summary(groups):
    sizes = np.array([len(g) for g in groups], dtype=int)
    if not len(sizes):return dict(examples=0)
    return dict(examples=len(sizes), mean=float(sizes.mean()), median=float(np.median(sizes)),
                minimum=int(sizes.min()), maximum=int(sizes.max()), full40=int((sizes == 40).sum()),
                events=int(sizes.sum()))


def score_control(x, y, train, test, classes, model):
    from sklearn.dummy import DummyClassifier
    yt = np.array([y[g[0]] for g in train]); ye = np.array([y[g[0]] for g in test])
    required = set(range(len(classes)))
    if set(yt) != required or set(ye) != required or np.bincount(ye, minlength=len(classes)).min() < 2:
        raise ValueError('Require every training class and at least two test averages per class')
    estimator = t.estimator('lr_1', len(classes)) if model == 'metadata_four' else DummyClassifier(strategy='prior')
    estimator.fit(l.fast_means(x, train), yt)
    pred = estimator.predict(l.fast_means(x, test))
    return dict(truth=ye.tolist(), prediction=np.asarray(pred).tolist(),
                train_groups=size_summary(train), test_groups=size_summary(test))


def examples(meta, y, classes, target):
    rows = []
    for i, label in enumerate(classes):
        ix = np.flatnonzero(np.asarray(y) == i)
        if len(ix):
            r = meta.iloc[ix[0]]
            rows.append(dict(target=target, label=label, word=str(r.get('word', r.get('word_identity',''))),
                sentence=str(r.get('sentence','')), reader=str(r.get('subject','')), event_id=str(r.event_id),
                sentence_key=str(r.sentence_key), word_index=str(r.get('word_index','')),
                word_length=str(r.get('word_length','')), relative_word_position=str(r.get('relative_word_position','')),
                sentence_length=str(r.get('sentence_length','')), duration_ms=str(r.get('first_fixation_duration_ms',''))))
    return rows


def run_standalone(meta, y, classes, target, store, sizes=None):
    rows, target_y, x, conflicts = collapse(meta, y, classes)
    info = dict(target=target, original_reader_events=len(meta), unique_occurrences=len(rows),
                excluded_conflicting_occurrences=len(conflicts),
                grain='one distinct word occurrence in its sentence, no reader replicas',
                comparison='independent occurrence benchmark, not the historical reader holdout or same averaging membership',
                reader_holdout='not defined after removing reader-level replicas')
    store.json(f'controls/{target}/definition.json', info)
    if conflicts:t.old.csv(store,f'controls/{target}/conflicts.csv.gz',pd.DataFrame(conflicts))
    flags = pd.DataFrame(dict(label=[classes[i] for i in target_y], control=x[:,3])).groupby(['label','control'],dropna=False).size().reset_index(name='occurrences')
    t.old.csv(store,f'controls/{target}/control_flag_counts.csv',flags)
    if len(set(target_y)) != len(classes):return [], [dict(target=target,reason='Target not invariant at occurrence level; see conflict report')]
    result = []; missing = []
    for mode in ['occurrence_random','sentence','word']:
        # Cannot hold out the identities being predicted in a closed-set classifier.
        if mode == 'word' and target.startswith('word_identity'):continue
        try:folds = standalone_folds(rows, target_y, mode)
        except ValueError as e:missing.append(dict(target=target,mode=mode,reason=str(e)));continue
        for size in sizes or [1,10,40,'max40']:
            print(f'CORRECTED CONTROL {target}: {mode}, average={size}; {len(rows)}unique occurrences',flush=True)
            for model in ['metadata_four','dummy']:
                cases = []
                for fi,(train,test) in enumerate(folds):
                    gt = independent_groups(target_y,train,size,SEED+fi)
                    ge = independent_groups(target_y,test,size,SEED+100+fi)
                    case = t.case(store,f'controls/{target}/{mode}/size{size}/{model}/fold{fi}',
                                  lambda:score_control(x,target_y,gt,ge,classes,model))
                    cases.append(case)
                if any(c['status'] != 'completed' for c in cases):
                    missing.append(dict(target=target,mode=mode,size=str(size),model=model,
                                        reason='Incomplete fold coverage; '+str([c.get('reason') for c in cases if c['status']!='completed'])))
                    continue
                truth = np.concatenate([c['truth'] for c in cases]);pred = np.concatenate([c['prediction'] for c in cases])
                result.append(dict(target=target,mode=mode,size=str(size),model=model,unit='unique source occurrences',
                    balanced_accuracy=float(balanced_accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),
                    chance=1/len(classes),test_examples=len(truth),unique_occurrences=len(rows)))
    return result, missing


def audit_partition(meta, y, train, test, family, target, mode, fold, size, gt=None, ge=None):
    if '_audit_occurrence' not in meta:meta['_audit_occurrence']=identities(meta)
    if meta.attrs.get('_audit_target')!=target:
        x=metadata_four(meta)
        meta['_audit_signature']=[t.a.digest(dict(x=[None if np.isnan(v) else float(v) for v in row],label=int(label))) for row,label in zip(x,y)]
        meta.attrs['_audit_target']=target
    key = meta['_audit_occurrence']; a=set(key.iloc[train]); b=key.iloc[test]
    overlap=b.isin(a)
    # Exact four-feature/label pair repetition is a separate diagnostic from occurrence repetition.
    signature=meta['_audit_signature'].tolist()
    seen={signature[i] for i in train}
    return dict(family=family,target=target,mode=mode,fold=fold,size=str(size),
        train_events=len(train),test_events=len(test),train_source_occurrences=len(a),
        test_source_occurrences=b.nunique(),test_events_with_seen_source=int(overlap.sum()),
        fraction_test_events_with_seen_source=float(overlap.mean()) if len(test) else 0.,
        test_events_with_seen_four_feature_label_pair=sum(signature[i] in seen for i in test),
        train_groups=json.dumps(size_summary(gt)) if gt is not None else '',
        test_groups=json.dumps(size_summary(ge)) if ge is not None else '')


def numeric_folds(meta, y, mode):
    # Explicit strata are for balanced split construction, not new target labels.
    strata = np.minimum(np.searchsorted(np.unique(np.quantile(y,[.25,.5,.75])),y),3)
    return l.safe_splits(meta,strata,np.arange(len(meta)),mode,3,t.SEED)


def regression_score(x, y, train, test, name):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.dummy import DummyRegressor
    if len(train)>10_000:train=np.random.default_rng(SEED).choice(train,10_000,replace=False)
    model = (Ridge(alpha=1) if name=='ridge1' else Ridge(alpha=100) if name=='ridge100' else
             ExtraTreesRegressor(n_estimators=64,max_depth=12,min_samples_leaf=5,n_jobs=1,random_state=SEED) if name=='extra_trees' else DummyRegressor(strategy='median'))
    estimator=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model)
    estimator.fit(x[train],y[train]);pred=estimator.predict(x[test])
    return dict(truth=y[test].tolist(),prediction=pred.tolist(),actual_training_events=len(train),test_events=len(test))


def run_numeric(meta, features, store):
    result=[];missing=[]
    for target in FOUR[:3]:
        values=pd.to_numeric(meta[target],errors='coerce').to_numpy(float);keep=np.isfinite(values)
        table=meta.loc[keep].reset_index(drop=True);y=values[keep]
        for mode in ['familiar','sentence','word','reader']:
            try:folds=numeric_folds(table,y,mode)
            except ValueError as e:missing.append(dict(target=target,mode=mode,reason=str(e)));continue
            print(f'EEG NUMERIC TARGET {target}: {mode}; individual trials; {len(y)}events',flush=True)
            for name,input_name in [('ridge1','regional_bins'),('ridge100','regional_bins'),('extra_trees','regional_bins'),('pre_ridge1','pre_control'),('dummy','regional_bins')]:
                x=features[input_name][keep];cases=[]
                for fi,(train,test,_) in enumerate(folds):
                    model='ridge1' if name=='pre_ridge1' else name
                    cases.append(t.case(store,f'numeric/{target}/{mode}/{name}/fold{fi}',lambda:regression_score(x,y,train,test,model)))
                if any(c['status']!='completed' for c in cases):missing.append(dict(target=target,mode=mode,model=name,reason='Incomplete numeric folds'));continue
                truth=np.concatenate([c['truth'] for c in cases]);pred=np.concatenate([c['prediction'] for c in cases])
                result.append(dict(target=target,mode=mode,size='1',model=name,
                    mean_absolute_error=float(mean_absolute_error(truth,pred)),median_absolute_error=float(median_absolute_error(truth,pred)),
                    r2=float(r2_score(truth,pred)),correlation=float(np.corrcoef(truth,pred)[0,1]) if np.std(pred)>1e-12 and np.std(truth)>1e-12 else None,
                    test_events=len(truth),units='fraction0–1' if target=='relative_word_position' else 'characters' if target=='word_length' else 'words',
                    interpretation='regression, not balanced accuracy; readers may share word identities unless word-held-out'))
    return result,missing
