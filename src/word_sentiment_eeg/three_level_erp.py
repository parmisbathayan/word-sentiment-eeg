"""Versioned 40-Hz first-stored-fixation reconstruction; old means stay intact."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .common import atomic_npz, digest, file_digest, sentence_key, write_json
from .erp import iter_sentences, locate
from .teco import first_fixation, orient_native

CLASSES=('negative','neutral','positive')
EXPERIMENTS=('all_three','lexical_only','contextual_only','sentence_only')
LAYERS=('lexical_class','contextual_word_class','sentence_context_class')
REGIONS=('anterior','central','posterior','occipital','left_lateral','right_lateral')
WINDOWS=((0,200),(200,300),(300,500),(500,800))
VERSION='three_level_first_stored_40hz_v1'


def load_annotations(path,dataset):
    frame=pd.read_csv(path,keep_default_na=False)
    if 'contextual_word_class' not in frame and 'contextual_class' in frame:
        frame['contextual_word_class']=frame.contextual_class
    required={'sentence_key','word_index','word','grammatical_class','annotation_status','alignment_warning',*LAYERS}
    required.add('tasks' if dataset=='zuco' else 'sentence_id_teco')
    if not required<=set(frame):
        raise ValueError(f'Missing three-level annotation columns: {sorted(required-set(frame))}')
    for layer in LAYERS:
        if not set(frame[layer])<=set(CLASSES)|{'uncertain','mixed'}:
            raise ValueError(f'Unknown label in {layer}')
    if not set(frame.grammatical_class)<= {'content','function','uncertain'}:
        raise ValueError('Unknown grammatical class')
    if not frame.annotation_status.eq('agent_reviewed_not_human_gold').all():
        raise ValueError('All word rows must be explicitly agent reviewed')
    frame['word_index']=frame.word_index.astype(int)
    keys=['sentence_key','word_index']
    if dataset=='teco':
        frame['sentence_id_teco']=frame.sentence_id_teco.astype(int)
        keys=['sentence_id_teco',*keys,'word']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate three-level annotation keys')
    return frame


def selection(label,experiment):
    if label is None:
        return None,'excluded_text_alignment'
    if str(label.alignment_warning):
        return None,'excluded_annotation_alignment'
    if label.grammatical_class!='content':
        return None,'excluded_function_or_uncertain_grammar'
    values=[getattr(label,f) for f in LAYERS]
    if experiment=='all_three':
        return (values[0],'eligible') if values[0] in CLASSES and len(set(values))==1 else (None,'excluded_three_level_disagreement_or_uncertain')
    layer={'lexical_only':0,'contextual_only':1,'sentence_only':2}[experiment]
    return (values[layer],'eligible') if values[layer] in CLASSES else (None,'excluded_selected_layer_uncertain_or_mixed')


def preprocess40(raw,hz):
    from scipy.signal import butter,sosfiltfilt
    value=np.asarray(raw,dtype=np.float64).copy()
    value-=value.mean(axis=0,keepdims=True)
    return sosfiltfilt(butter(4,40,fs=hz,output='sos'),value,axis=1).astype(np.float32)


def epoch40(filtered,onset,hz):
    pre,post=int(.2*hz),int(.8*hz)
    if onset<pre or onset+post>=filtered.shape[1]:
        return None,'sentence_boundary'
    value=filtered[:,onset-pre:onset+post+1].copy()
    value-=value[:,:pre].mean(axis=1,keepdims=True)
    if not np.isfinite(value).all():
        return None,'nonfinite_epoch'
    if np.max(np.abs(value))>100:
        return None,'amplitude_over_100_native_assumed_uV'
    return value,'accepted'


def source_sentences(path,dataset):
    if dataset=='zuco':
        for position,words,raw,fixations in iter_sentences(path):
            yield position,sentence_key(words),None,raw,[(i,w,fs[0] if fs else None,'available' if fs else 'missing_first_fixation') for i,(w,fs) in enumerate(zip(words,fixations))]
    else:
        with Path(path).open('rb') as handle:
            data=pickle.load(handle)
        for position,(_,trial) in enumerate(data.items()):
            text=str(trial.get('persian_sentence',''))
            key=hashlib.sha256(text.encode()).hexdigest()[:24]
            words=[]
            for index,word in sorted(trial.get('word',{}).items(),key=lambda kv:int(kv[0])):
                first,status=first_fixation(word)
                words.append((int(index),str(word['content']),first,status))
            yield position,key,int(trial['sentenceId']),orient_native(trial.get('raw_data')),words


def process_reader(path,dataset,task,subject,annotations,out,fingerprint,channels,regions,experiments,save_features=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    hz=500 if dataset=='zuco' else 1200
    times=np.arange(-int(.2*hz),int(.8*hz)+1)*1000/hz
    frame=annotations[annotations.tasks.map(lambda v:task in v.split(';'))] if dataset=='zuco' else annotations
    keys=['sentence_key','word_index'] if dataset=='zuco' else ['sentence_id_teco','sentence_key','word_index','word']
    lookup={tuple(getattr(r,k) for k in keys):r for r in frame.itertuples()}
    sums=np.zeros((len(experiments),3,len(channels),len(times)),np.float64)
    counts=np.zeros((len(experiments),3),np.int64);candidates=np.zeros_like(counts)
    rows=[];features=[];seen=set();sentences=0;seen_sentences=set()
    for position,key,sid,raw,words in source_sentences(path,dataset):
        sentences+=1;filtered=None
        if dataset=='teco':
            if (sid,key) in seen_sentences:raise ValueError('Duplicate native sentence ID/text')
            seen_sentences.add((sid,key))
            if not ((frame.sentence_id_teco==sid)&(frame.sentence_key==key)).any():
                raise ValueError('Native TeCo sentence lacks exact annotation')
        valid_raw=raw is not None and raw.ndim==2 and raw.shape[0]==len(channels) and raw.shape[1]>=3
        raw_status='missing_or_invalid_sentence_eeg' if not valid_raw else ('nonfinite_sentence_eeg' if not np.isfinite(raw).all() else 'available')
        for index,word,segment,first_status in words:
            lookup_key=(key,index) if dataset=='zuco' else (sid,key,index,word)
            label=lookup.get(lookup_key)
            if label is None or label.word!=word:
                if dataset=='teco':
                    raise ValueError(f'{task}/{subject}: source word lacks exact annotation ({position},{index})')
                # ZuCo exports include known text/tokenization discrepancies.
                # Preserve each source event with an explicit exclusion; do not
                # guess its class or abandon all other words from this reader.
                label=None
            event_id=digest([dataset,task,subject,position,key,index])[:32]
            if event_id in seen:
                raise ValueError('Duplicate word event')
            seen.add(event_id)
            chosen=[selection(label,e) for e in experiments]
            row=dict(event_id=event_id,task=task,subject=subject,sentence_position=position,sentence_key=key,
                     word_index=index,word=word,sentence_id_teco='' if sid is None else sid,
                     is_control=str(getattr(label,'is_control',False)),grammatical_class=getattr(label,'grammatical_class','unmatched'),
                     lexical_class=getattr(label,'lexical_class','unmatched'),contextual_word_class=getattr(label,'contextual_word_class','unmatched'),
                     sentence_context_class=getattr(label,'sentence_context_class','unmatched'),onset_sample_sentence_0based='',match_method='',word_segment_samples='')
            for ei,(cls,status) in enumerate(chosen):
                if cls is not None:candidates[ei,CLASSES.index(cls)]+=1
            value=None;status='not_selected';onset=None
            if any(c is not None for c,_ in chosen):
                status=raw_status
                if status=='available':
                    if segment is None:
                        status=first_status if first_status!='available' else 'missing_or_invalid_first_fixation'
                    else:
                        row['word_segment_samples']=segment.shape[1]
                        onset,status=locate(raw,segment)
                        row['match_method']=status
                        if onset is not None:
                            row['onset_sample_sentence_0based']=int(onset)
                            if filtered is None:
                                try:filtered=preprocess40(raw,hz)
                                except ValueError:status='sentence_too_short_for_filter'
                            if filtered is not None:value,status=epoch40(filtered,onset,hz)
            for ei,experiment in enumerate(experiments):
                cls,selection_status=chosen[ei]
                row[f'{experiment}_class']=cls or ''
                row[f'{experiment}_status']=status if cls is not None else selection_status
                if cls is not None and value is not None:
                    ci=CLASSES.index(cls);sums[ei,ci]+=value;counts[ei,ci]+=1
            if value is not None and save_features:
                feature={'event_id':event_id,'task':task,'subject':subject,'sentence_key':key,'word_index':index,'word':word}
                for region in REGIONS:
                    picks=np.flatnonzero(np.array(regions)==region)
                    for a,b in WINDOWS:
                        feature[f'{region}_{a}_{b}_ms']=float(value[picks][:,(times>=a)&(times<b)].mean()) if len(picks) else ''
                features.append(feature)
            rows.append(row)
        if sentences%10==0:print(f'{task}/{subject}: {sentences} sentences; accepted {counts.tolist()}',flush=True)
    if not rows:raise ValueError('No source words processed')
    if dataset=='teco' and seen_sentences!=set(zip(frame.sentence_id_teco,frame.sentence_key)):
        raise ValueError('Incomplete native TeCo sentence roster')
    means=np.divide(sums,counts[:,:,None,None],out=np.zeros_like(sums),where=counts[:,:,None,None]>0).astype(np.float32)
    atomic_npz(out/'class_means.npz',means=means,counts=counts,candidates=candidates,classes=np.array(CLASSES),
               experiments=np.array(experiments),channels=np.array(channels),times_ms=times)
    table=pd.DataFrame(rows);temp=out/'word_onsets.tmp.csv.gz';table.to_csv(temp,index=False,compression='gzip');temp.replace(out/'word_onsets.csv.gz')
    names=['class_means.npz','word_onsets.csv.gz']
    if save_features:
        pd.DataFrame(features).to_csv(out/'event_window_features.csv.gz',index=False,compression='gzip');names.append('event_window_features.csv.gz')
    item=dict(dataset=dataset,task=task,subject=subject,status='completed',fingerprint=fingerprint,
              source_sha256=file_digest(path),source_bytes=Path(path).stat().st_size,
              source_sentence_count=sentences,source_word_count=len(rows),
              experiments={e:{'candidates':dict(zip(CLASSES,map(int,candidates[i]))),
                              'accepted':dict(zip(CLASSES,map(int,counts[i]))),
                              'statuses':dict(Counter(table[f'{e}_status']))} for i,e in enumerate(experiments)},
              checksums={n:file_digest(out/n) for n in names},output_dir=str(out))
    write_json(out/'manifest.json',item)
    return item


def resume(out,fingerprint,source=None):
    out=Path(out)
    if not (out/'manifest.json').exists():return None
    item=json.loads((out/'manifest.json').read_text())
    if item.get('fingerprint')!=fingerprint:raise ValueError('Checkpoint fingerprint differs; use a new run tag')
    if source is not None and Path(source).exists() and file_digest(source)!=item['source_sha256']:
        raise ValueError('Source content changed; use a new run tag')
    if item.get('status')!='completed':return None
    if any(not (out/n).exists() or file_digest(out/n)!=sha for n,sha in item['checksums'].items()):return None
    table=pd.read_csv(out/'word_onsets.csv.gz',keep_default_na=False)
    if len(table)!=item['source_word_count'] or table.event_id.duplicated().any():return None
    with np.load(out/'class_means.npz',allow_pickle=False) as data:
        if not np.isfinite(data['means']).all():return None
        if data['means'].shape!=(len(item['experiments']),3,len(data['channels']),len(data['times_ms'])):return None
        for ei,e in enumerate(data['experiments']):
            spec=item['experiments'][str(e)]
            if Counter(table[f'{e}_status'])!=Counter(spec['statuses']):return None
            if not np.array_equal(data['counts'][ei],[spec['accepted'][c] for c in CLASSES]):return None
            if not np.array_equal(data['candidates'][ei],[spec['candidates'][c] for c in CLASSES]):return None
            for c in CLASSES:
                if int((table[f'{e}_class']==c).sum())!=spec['candidates'][c]:return None
                accepted=((table[f'{e}_status']=='accepted')&(table[f'{e}_class']==c)).sum()
                if accepted!=spec['accepted'][c]:return None
        pre=data['times_ms']<0
        if np.abs(data['means'][:,:,:,pre].mean(axis=-1)).max()>1e-4:return None
    return item
