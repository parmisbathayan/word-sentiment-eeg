"""Auditable matching and disjoint event-membership averages for the paper route.

These are generic tested components, not a claim of recovering the authors'
undisclosed timing bins or averaging code. Strata must be supplied explicitly.
"""
import numpy as np
import pandas as pd


def validate_events(meta):
    required={'event_id','subject','sentiment','sentence_key'}
    if not required.issubset(meta.columns):raise ValueError('Missing trial identity/label columns')
    if meta.empty:raise ValueError('No eligible events; report insufficient coverage before averaging')
    if meta.event_id.isna().any() or meta.event_id.astype(str).duplicated().any():
        raise ValueError('Trial event IDs must be unique and present')
    if not meta.sentiment.isin(['negative','positive']).all():
        raise ValueError('Apply explicit sentiment exclusions before binary matching/averaging')
    if meta[['subject','sentence_key']].isna().any().any():raise ValueError('Missing reader/sentence identity')


def matched_trials(meta,strata_columns,seed=2026):
    """Equal positive/negative counts per reader and caller-defined stratum.

    No guessed duration or neighboring-fixation bins. Their exact definitions
    must be recorded by the caller, after verifying the source methods.
    Returns input row indices and a complete, reconciling selection audit.
    """
    validate_events(meta)
    if not strata_columns or 'subject' not in strata_columns:
        raise ValueError('Explicit matching strata must include reader identity')
    if 'sentiment' in strata_columns:raise ValueError('Sentiment is balanced within strata, not part of the strata')
    if meta[list(strata_columns)].isna().any().any():raise ValueError('Undefined matching strata')
    rng=np.random.default_rng(seed);selected=[];summary=[]
    indexed=meta.reset_index(drop=True)
    for key,table in indexed.groupby(list(strata_columns),sort=True,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        groups={c:table.index[table.sentiment.eq(c)].to_numpy() for c in ['negative','positive']}
        count=min(map(len,groups.values()))
        for c,indices in groups.items():
            chosen=np.sort(rng.choice(indices,size=count,replace=False)) if count else np.array([],dtype=int)
            selected.extend(chosen)
            summary.append(dict(zip(strata_columns,key),sentiment=c,input_trials=len(indices),
                                retained_trials=count,excluded_by_matching=len(indices)-count))
    selected=np.sort(np.asarray(selected,dtype=int))
    audit=indexed[['event_id','subject','sentiment']].copy()
    audit['retained']=np.isin(np.arange(len(indexed)),selected)
    audit['outcome']=np.where(audit.retained,'retained','excluded_by_matching')
    return selected,audit,pd.DataFrame(summary)


def average_trials(features,meta,size,strata_columns,seed=2026):
    """Within-reader/class averages with explicit strata and event membership.

    Multi-trial averages use at most one event from each sentence text. Unfilled
    remainders remain explicit and are never advertised as full-size averages.
    Every event appears once. Exact event disjointness does not itself prevent
    temporal EEG overlap between adjacent events: that requires a separate
    source-interval or sentence-level split audit.
    """
    validate_events(meta)
    if not isinstance(size,int) or size<1:raise ValueError('Average size must be a positive integer')
    if len(features)!=len(meta) or not np.isfinite(features).all():raise ValueError('Invalid feature/event join')
    if not {'subject','sentiment'}.issubset(strata_columns):
        raise ValueError('Averages must preserve reader and sentiment; add explicit length strata')
    if meta[list(strata_columns)].isna().any().any():raise ValueError('Undefined averaging strata')
    rng=np.random.default_rng(seed);averages=[];records=[];members=[]
    indexed=meta.reset_index(drop=True)
    for key,group in indexed.groupby(list(strata_columns),sort=True,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        remaining=list(rng.permutation(group.index.to_numpy()))
        while remaining:
            chunk=[];deferred=[];sentences=set()
            for row in remaining:
                sentence=indexed.iloc[row].sentence_key
                if len(chunk)<size and sentence not in sentences:
                    chunk.append(row);sentences.add(sentence)
                else:deferred.append(row)
            if not chunk:raise RuntimeError('Averaging made no progress')
            average_id=f'average_{len(averages):06d}'
            averages.append(np.mean(features[chunk],axis=0,dtype=np.float64).astype(np.float32))
            records.append(dict(zip(strata_columns,key),average_id=average_id,
                                requested_trials=size,actual_trials=len(chunk),is_remainder=len(chunk)<size))
            for row in chunk:members.append(dict(average_id=average_id,event_id=indexed.iloc[row].event_id,
                                                 subject=indexed.iloc[row].subject,sentiment=indexed.iloc[row].sentiment,
                                                 sentence_key=indexed.iloc[row].sentence_key))
            remaining=deferred
    shape=(0,)+features.shape[1:]
    output=np.stack(averages) if averages else np.empty(shape,dtype=np.float32)
    membership=pd.DataFrame(members)
    if len(membership)!=len(meta) or membership.event_id.duplicated().any():
        raise RuntimeError('Missing or duplicated event membership')
    return output,pd.DataFrame(records),membership
