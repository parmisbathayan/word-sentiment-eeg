"""Task-1 word-associated fixation adaptation; no invented nuisance timeline.

Only native reading EEG is used. All stored fixation segments are candidates.
This is NOT the full regression-cleaned paper reproduction.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_npz, digest, file_digest, write_json
from .teco import orient_native, NATIVE_HZ
from .teco_erp import load_labels
from .erp import locate

TIMES=np.arange(-75,126)*8
CLASSES=('negative','positive')


def montage():
    path=Path(__file__).resolve().parents[2]/'assets/teco_montage.txt'
    m=pd.read_csv(path,sep=r'\s+',header=None,names=['index','x','y','z','labels'])
    # Native TeCo x=right, y=anterior; shared projection expects X=anterior,Y=left.
    return m.assign(X=m.y,Y=-m.x,Z=m.z)


def segments(word):
    value=word.get('rawEEG')
    if isinstance(value,(list,tuple)):return list(value)
    if isinstance(value,np.ndarray) and value.dtype==object:return list(value.reshape(-1))
    return [value]


def balance(table):
    """Exact word length and pooled target-duration tertiles, within reader.

    Full previous/next fixation timings are unavailable in these native pickles;
    this is explicitly different from the paper's neighbor-timing matching.
    """
    table=table.copy();table['selection_status']=table.status
    eligible=table[table.status.eq('eligible')]
    if eligible.empty:return table,[],{}
    edges=np.unique(np.quantile(eligible.duration_ms,[1/3,2/3]))
    table['duration_stratum']=np.searchsorted(edges,table.duration_ms.to_numpy(float),side='right')
    chosen=[]
    for _,group in table[table.status.eq('eligible')].groupby(['word_length','duration_stratum'],sort=True):
        a=[group[group.sentiment.eq(c)].sort_values('event_id').index.to_numpy() for c in CLASSES]
        n=min(map(len,a));chosen.extend(np.r_[a[0][:n],a[1][:n]].tolist())
    table.loc[table.status.eq('eligible'),'selection_status']='excluded_by_matching'
    table.loc[chosen,'selection_status']='accepted'
    return table,chosen,{'target_duration_tertile_edges_ms':edges.tolist(),'matching':'exact word length and target-fixation duration tertile; no neighbor-timing matching'}


def extract(data,reader,lookup,progress=None,max_epoch_bytes=1_200_000_000):
    from scipy.signal import butter,sosfiltfilt,resample_poly
    expected={(k[0],k[1]) for k in lookup};rows=[];epochs=[];seen=set();sentences=set()
    for k,(trial_key,trial) in enumerate(data.items(),1):
        if not isinstance(trial,dict):raise ValueError('Non-dictionary native trial')
        sid=int(trial['sentenceId']);key=hashlib.sha256(str(trial.get('persian_sentence','')).encode()).hexdigest()[:24]
        if sid in sentences:raise ValueError('Duplicate native sentence ID')
        sentences.add(sid)
        if (sid,key) not in expected:raise ValueError(f'{reader}/{sid}: sentence has no exact private label match')
        raw=orient_native(trial.get('raw_data'));filtered=None
        durations=np.asarray(trial.get('allFixations',{}).get('duration',[]),float).reshape(-1)
        sentence_rows=[]
        for wi,word in sorted(trial.get('word',{}).items(),key=lambda pair:int(pair[0])):
            token=str(word.get('content','')).strip();label=lookup.get((sid,key,int(wi),token))
            if label is None:raise ValueError(f'{reader}/{sid}/{wi}: no exact private word label')
            values=segments(word)
            try:positions=np.asarray(word.get('fixPositions',[]),float).reshape(-1)
            except (ValueError,TypeError):positions=np.array([])
            try:count=float(np.asarray(word.get('nFixations',0)).reshape(-1)[0])
            except (ValueError,TypeError,IndexError):count=np.nan
            for fi,value in enumerate(values):
                eid=digest([reader,sid,key,int(wi),fi])[:32]
                if eid in seen:raise ValueError('Duplicate fixation identity')
                seen.add(eid)
                row=dict(event_id=eid,subject=reader,sentence_id=sid,sentence_key=key,word_index=int(wi),fixation_index=fi,
                    word=token,word_length=len(token),lexical_class=label.lexical_class,contextual_class=label.contextual_class,
                    sentence_context_class=label.sentence_context_class,is_control=str(label.is_control),sentiment=label.contextual_class,
                    duration_ms=np.nan,onset_sample=-1,epoch_index=-1,status='eligible')
                segment=orient_native(value)
                if not np.isfinite(count) or count<=0:row['status']='no_stored_fixation'
                elif count!=int(count) or int(count)!=len(values):row['status']='fixation_count_segment_disagreement'
                elif len(positions)!=len(values) or not np.isfinite(positions).all() or not np.equal(positions,np.floor(positions)).all():row['status']='invalid_fixation_order_indices'
                elif str(label.alignment_warning):row['status']='annotation_alignment_warning'
                elif label.lexical_class!=label.contextual_class or label.contextual_class not in CLASSES:row['status']='neutral_uncertain_or_context_disagreement'
                elif not 1<=len(token)<=20:row['status']='word_length_outside_1_20'
                elif raw is None or not np.isfinite(raw).all():row['status']='invalid_sentence_reading_eeg'
                elif segment is None or segment.shape[1]<3 or not np.isfinite(segment).all():row['status']='invalid_stored_segment'
                else:
                    duration=segment.shape[1]*1000/NATIVE_HZ;row['duration_ms']=duration
                    index=int(positions[fi])-1
                    if index<0 or index>=len(durations) or not np.isfinite(durations[index]) or abs(float(durations[index])-duration)>5:
                        row['status']='fixation_duration_or_order_disagreement'
                    elif duration<100:row['status']='fixation_shorter_than_100ms'
                    else:
                        onset,method=locate(raw,segment);row['match_method']=method
                        if onset is None:row['status']='onset_'+method
                        else:
                            row['onset_sample']=onset
                            if filtered is None:
                                # Whole sentence, never individual short fixation segments.
                                referenced=raw.astype(np.float64);referenced-=referenced.mean(axis=0,keepdims=True)
                                filtered=resample_poly(sosfiltfilt(butter(4,[1,50],btype='bandpass',fs=NATIVE_HZ,output='sos'),referenced,axis=1),5,48,axis=1).astype(np.float32)
                            center=int(np.floor(onset*125/NATIVE_HZ+.5));ix=center+np.arange(-75,126)
                            if ix[0]<0 or ix[-1]>=filtered.shape[1]:row['status']='epoch_outside_sentence'
                            else:
                                epoch=filtered[:,ix].copy()
                                if not np.isfinite(epoch).all():row['status']='nonfinite_epoch'
                                elif abs(epoch).max()>90:row['status']='amplitude_over_90_assumed_uV'
                                else:
                                    if (len(epochs)+1)*epoch.nbytes>max_epoch_bytes:raise OSError('Reader epoch cap reached; earlier reader checkpoints retained')
                                    row['epoch_index']=len(epochs);epochs.append(epoch)
                rows.append(row);sentence_rows.append(len(rows)-1)
        # The same physical onset cannot be counted as two word events.
        by_onset={}
        for i in sentence_rows:
            if rows[i]['status']=='eligible':by_onset.setdefault(rows[i]['onset_sample'],[]).append(i)
        for indices in by_onset.values():
            if len(indices)>1:
                for i in indices:rows[i]['status']='ambiguous_duplicate_onset'
        if progress:progress(k,len(data))
    table,chosen,policy=balance(pd.DataFrame(rows))
    if not chosen:raise ValueError(f'{reader}: no matched positive/negative fixation trials; inspect source/label policy')
    meta=table.loc[chosen].sort_values('event_id').reset_index(drop=True)
    signal=np.stack([epochs[int(i)] for i in meta.epoch_index]).astype(np.float32)
    meta['archive_trial_index']=np.arange(len(meta));counts=np.array([sum(meta.sentiment.eq(c)) for c in CLASSES])
    if min(counts)<2:raise ValueError(f'{reader}: insufficient matched classes {counts.tolist()}')
    orders=[meta[meta.sentiment.eq(c)].sort_values(['word_length','duration_ms','event_id']).index.to_numpy() for c in CLASSES]
    return signal,meta,table,orders,policy


def verify(folder):
    state=json.loads((folder/'prepared.json').read_text())
    if state.get('status')!='prepared':raise ValueError('Reader preparation incomplete')
    for name,sha in state['checksums'].items():
        if file_digest(folder/name)!=sha:raise ValueError('Changed reader checkpoint: '+name)
    return state


def prepare(data,reader,lookup,folder,root,budget,progress=None,source_info=None):
    from .paper_stage_archive import disk_bytes
    signal,meta,audit,orders,policy=extract(data,reader,lookup,progress,max_epoch_bytes=budget)
    # Include a conservative allowance for simultaneous atomic writes and metadata.
    if disk_bytes(root)+signal.nbytes+30_000_000>budget:raise OSError('CPU trial archive cap exceeded; earlier readers preserved')
    folder.mkdir(parents=True,exist_ok=True);m=montage()
    atomic_npz(folder/'prepared_trials.npz',signal=signal,labels=meta.sentiment.map({'negative':0,'positive':1}).to_numpy(np.int8),
        event_ids=meta.event_id.to_numpy(dtype=str),negative_pair_index=orders[0],positive_pair_index=orders[1],
        times_ms=TIMES,channel_names=m.labels.to_numpy(dtype=str),sample_rate_hz=np.array(125),units=np.array('assumed microvolts'))
    meta.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False,compression='gzip');audit.to_csv(folder/'fixation_audit.csv.gz',index=False,compression='gzip')
    counts=np.array([sum(meta.sentiment.eq(c)) for c in CLASSES]);sums=[];squares=[]
    v=np.concatenate([signal,signal.std(axis=1)[:,None,:]],axis=1).astype(np.float64)
    for c in CLASSES:
        x=v[meta.sentiment.eq(c)];sums.append(x.sum(axis=0));squares.append((x*x).sum(axis=0))
    diff=signal[orders[1]]-signal[orders[0]]
    atomic_npz(folder/'statistics.npz',counts=counts,sums=np.array(sums),squares=np.array(squares),
        pair_sum=diff.sum(axis=0,dtype=np.float64),pair_squares=(diff.astype(np.float64)**2).sum(axis=0),pairs=np.array(len(diff)),times_ms=TIMES)
    write_json(folder/'matching_policy.json',policy)
    if source_info is not None:write_json(folder/'source.json',source_info)
    write_json(folder/'prepared.json',dict(status='prepared',subject=reader,counts=counts.tolist(),pairs=len(diff),
        checksums={p.name:file_digest(p) for p in folder.iterdir() if p.is_file() and p.name!='prepared.json'}))
    verify(folder)
