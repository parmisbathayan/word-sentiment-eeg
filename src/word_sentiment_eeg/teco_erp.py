"""Compact native TeCo FRPs. Private source text and EEG stay in the runtime."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_npz, digest, file_digest, write_json
from .erp import locate
from .teco import first_fixation, orient_native, NATIVE_HZ, N_CHANNELS

READERS = ('armin','arya','eisa','elahe','farzam','javad','mehrnaz','nastaran','razie','sajad','sharzad','yaser')
CLASSES = ('negative','neutral','positive')
GROUPS = CLASSES + ('neutral_in_neutral_sentence','neutral_in_source_neutral_sentence')
PRE, POST = 240, 960
TIMES_MS = np.arange(-PRE, POST+1)*1000/NATIVE_HZ
VERSION = 'teco_native_frp_v1'


def load_labels(path):
    frame=pd.read_csv(path,keep_default_na=False)
    required={'sentence_id_teco','sentence_id_zuco','sentence_key','word_index','word','sentence',
              'lexical_class','contextual_class','sentence_context_class','alignment_warning','annotation_status','is_control',
              'original_sentence_label'}
    if not required<=set(frame):
        raise ValueError(f'Private TeCo labels lack {sorted(required-set(frame))}')
    frame['sentence_id_teco']=frame.sentence_id_teco.astype(int)
    frame['word_index']=frame.word_index.astype(int)
    original=pd.to_numeric(frame.original_sentence_label,errors='raise')
    if not set(original)<= {-1,0,1}:
        raise ValueError('Original source sentence labels must use -1/0/1')
    frame['original_sentence_label']=original.astype(int)
    allowed=set(CLASSES)|{'uncertain'}
    if not set(frame.lexical_class)<=allowed or not set(frame.contextual_class)<=allowed:
        raise ValueError('Unexpected Persian word class')
    if not set(frame.sentence_context_class)<=allowed|{'mixed'}:
        raise ValueError('Unexpected Persian sentence-context class')
    if not frame.annotation_status.eq('agent_reviewed_not_human_gold').all():
        raise ValueError('Every private word annotation must be explicitly agent reviewed')
    if not frame.is_control.astype(str).str.lower().isin({'true','false','1','0'}).all():
        raise ValueError('Control flags must be explicit booleans')
    # Token variants may exist between readers. Resolve by exact token and sentence,
    # never by compressed derived feature position or a positional fallback.
    keys=['sentence_id_teco','sentence_key','word_index','word']
    if frame.duplicated(keys).any():
        raise ValueError('Private TeCo annotation keys are not unique')
    return frame, {tuple(getattr(r,k) for k in keys):r for r in frame.itertuples()}


def is_control(value):
    return str(value).strip().lower() in {'true','1','yes','control'}


def groups_for_label(row):
    if is_control(row.is_control):
        return (), 'control_excluded'
    if str(row.alignment_warning):
        return (), 'annotation_alignment_warning'
    if row.annotation_status!='agent_reviewed_not_human_gold':
        return (), 'annotation_not_reviewed'
    if row.lexical_class not in CLASSES or row.lexical_class!=row.contextual_class:
        return (), 'uncertain_or_context_disagreement'
    groups=(row.contextual_class,)
    if row.contextual_class=='neutral' and row.sentence_context_class=='neutral':
        groups+=('neutral_in_neutral_sentence',)
    if row.contextual_class=='neutral' and int(row.original_sentence_label)==0:
        groups+=('neutral_in_source_neutral_sentence',)
    return groups, 'eligible'


def preprocess(raw):
    from scipy.signal import butter, sosfiltfilt
    signal=raw.astype(np.float64)
    signal-=signal.mean(axis=0,keepdims=True)
    return sosfiltfilt(butter(4,30,fs=NATIVE_HZ,output='sos'),signal,axis=1).astype(np.float32)


def epoch(filtered,onset):
    if onset<PRE or onset+POST>=filtered.shape[1]:
        return None,'sentence_boundary'
    signal=filtered[:,onset-PRE:onset+POST+1].copy()
    signal-=signal[:,:PRE].mean(axis=1,keepdims=True)
    if not np.isfinite(signal).all():
        return None,'nonfinite_epoch'
    if np.abs(signal).max()>100:
        return None,'amplitude_over_100_assumed_uV'
    return signal,'accepted'


def extract_reader(data,reader,lookup,channels,output,fingerprint,source_info):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    sums={g:np.zeros((N_CHANNELS,len(TIMES_MS)),dtype=np.float64) for g in GROUPS}
    counts=Counter(); rows=[]; seen=set(); trial_ids=set(); empty_word_trials=0
    expected_sentences={(k[0],k[1]) for k in lookup}
    for position,(trial_key,trial) in enumerate(data.items()):
        if not isinstance(trial,dict):
            raise ValueError(f'{reader}: non-dictionary native trial')
        sid=int(trial['sentenceId'])
        if sid in trial_ids:
            raise ValueError(f'{reader}: duplicate native sentence ID {sid}')
        trial_ids.add(sid)
        sentence=str(trial.get('persian_sentence',''))
        key=hashlib.sha256(sentence.encode()).hexdigest()[:24]
        if (sid,key) not in expected_sentences:
            raise ValueError(f'{reader}: native sentence {sid} has no exact private annotation match')
        raw=orient_native(trial.get('raw_data'))
        raw_status='available'
        if raw is None or raw.shape[1]<3:
            raw_status='missing_reading_raw_data'
        elif not np.isfinite(raw).all():
            raw_status='nonfinite_reading_raw_data'
        filtered=None
        if not trial.get('word',{}):
            empty_word_trials+=1
        for word_index,word in sorted(trial.get('word',{}).items(),key=lambda kv:int(kv[0])):
            index=int(word_index)
            token=str(word.get('content','')).strip()
            identity=digest(['TeCo',reader,sid,key,index])[:32]
            if identity in seen:
                raise ValueError(f'{reader}: duplicate native word event')
            seen.add(identity)
            label=lookup.get((sid,key,index,token))
            if label is None:
                raise ValueError(f'{reader}: no exact private word label for native sentence {sid}, word {index}')
            groups,status=groups_for_label(label)
            row={'event_id':identity,'dataset':'TeCo','task':'task1','subject':reader,
                 'sentence_position':position,'raw_trial_key':str(trial_key),
                 'sentence_id_teco':sid,'sentence_key':key,'word_index':index,'word':token,
                 'sentence_id_zuco':label.sentence_id_zuco if label is not None else '',
                 'lexical_class':label.lexical_class if label is not None else '',
                 'contextual_class':label.contextual_class if label is not None else '',
                 'sentence_context_class':label.sentence_context_class if label is not None else '',
                 'original_sentence_label':int(label.original_sentence_label) if label is not None else '',
                 'is_control':bool(label is not None and is_control(label.is_control)),
                 'onset_sample_sentence_0based':'','onset_ms':'','match_method':'',
                 'first_fixation_samples':'','accepted_groups':'','class_epoch_index':'','status':status}
            if groups:
                if raw_status!='available':
                    status=raw_status
                else:
                    segment,status=first_fixation(word)
                    if segment is not None:
                        row['first_fixation_samples']=segment.shape[1]
                        onset,method=locate(raw,segment)
                        row['match_method']=method
                        if onset is None:
                            status=method
                        else:
                            row['onset_sample_sentence_0based']=onset
                            row['onset_ms']=onset*1000/NATIVE_HZ
                            if filtered is None:
                                try:
                                    filtered=preprocess(raw)
                                except ValueError:
                                    raw_status='sentence_too_short_for_filter'
                            if filtered is None:
                                status=raw_status
                            else:
                                value,status=epoch(filtered,onset)
                                if value is not None:
                                    row['class_epoch_index']=counts[groups[0]]
                                    row['accepted_groups']='|'.join(groups)
                                    for group in groups:
                                        sums[group]+=value; counts[group]+=1
            row['status']=status
            rows.append(row)
        if (position+1)%10==0:
            print(f'TECO {reader}: {position+1}/{len(data)} sentences; {sum(counts[g] for g in CLASSES)} accepted words',flush=True)
    if trial_ids!={sid for sid,key in expected_sentences}:
        raise ValueError(f'{reader}: incomplete native sentence roster')
    if not rows:
        raise ValueError(f'{reader}: no native words available')
    means=np.stack([sums[g]/counts[g] if counts[g] else np.zeros_like(sums[g]) for g in GROUPS]).astype(np.float32)
    atomic_npz(output/'class_means.npz',means=means,counts=np.array([counts[g] for g in GROUPS]),
               classes=np.array(GROUPS),channels=np.array(channels),times_ms=TIMES_MS)
    table=pd.DataFrame(rows)
    temporary=output/'word_onsets.tmp.csv.gz'
    table.to_csv(temporary,index=False,compression='gzip')
    temporary.replace(output/'word_onsets.csv.gz')
    manifest={'dataset':'TeCo','task':'task1','subject':reader,'fingerprint':fingerprint,
              'status':'completed','source':source_info,'source_sentence_count':len(trial_ids),
              'source_word_count':len(rows),'accepted_unique_word_events':int((table.status=='accepted').sum()),
              'native_sentences_without_word_entries':empty_word_trials,
              'counts':{g:int(counts[g]) for g in GROUPS},'exclusions':dict(Counter(table.status)),
              'output_dir':str(output),'individual_trial_arrays_saved':False,
              'checksums':{n:file_digest(output/n) for n in ('class_means.npz','word_onsets.csv.gz')}}
    write_json(output/'manifest.json',manifest)
    return manifest


def inspect_reader(output,reader,fingerprint,channels,source_metadata):
    """Deep validate a compact checkpoint before allowing reuse."""
    output=Path(output)
    if not (output/'manifest.json').exists():
        return None
    try:
        item=json.loads((output/'manifest.json').read_text())
        if item['fingerprint']!=fingerprint or item['subject']!=reader or item['status']!='completed':
            return None
        if any(item['source'].get(k)!=v for k,v in source_metadata.items()):
            return None
        for name,expected in item['checksums'].items():
            if file_digest(output/name)!=expected:
                return None
        with np.load(output/'class_means.npz',allow_pickle=False) as arrays:
            means=arrays['means']; counts=arrays['counts']
            if means.shape!=(len(GROUPS),N_CHANNELS,len(TIMES_MS)) or not np.isfinite(means).all():
                return None
            if not np.array_equal(arrays['classes'],GROUPS) or not np.array_equal(arrays['channels'],channels) or not np.array_equal(arrays['times_ms'],TIMES_MS):
                return None
            if not np.array_equal(counts,[item['counts'][g] for g in GROUPS]):
                return None
            if np.max(np.abs(means[:,:,:PRE].mean(axis=2)))>1e-4:
                return None
        table=pd.read_csv(output/'word_onsets.csv.gz',keep_default_na=False)
        if len(table)!=item['source_word_count'] or table.event_id.duplicated().any() or not (table.subject==reader).all():
            return None
        identities=[digest(['TeCo',reader,int(r.sentence_id_teco),r.sentence_key,int(r.word_index)])[:32]
                    for r in table.itertuples()]
        if table.event_id.tolist()!=identities:
            return None
        accepted=table[table.status=='accepted']
        if len(accepted)!=sum(item['counts'][g] for g in CLASSES):
            return None
        for g in GROUPS:
            count=accepted.accepted_groups.map(lambda v:g in v.split('|')).sum()
            if count!=item['counts'][g]:
                return None
        for group in CLASSES:
            subset=accepted[accepted.contextual_class==group]
            if subset.lexical_class.tolist()!=[group]*len(subset):
                return None
            if subset.class_epoch_index.astype(int).tolist()!=list(range(len(subset))):
                return None
        strict=accepted[accepted.accepted_groups.map(lambda v:'neutral_in_neutral_sentence' in v.split('|'))]
        if not (strict.sentence_context_class=='neutral').all() or not (strict.contextual_class=='neutral').all():
            return None
        source_neutral=accepted[accepted.accepted_groups.map(lambda v:'neutral_in_source_neutral_sentence' in v.split('|'))]
        if not source_neutral.original_sentence_label.astype(int).eq(0).all() or not source_neutral.contextual_class.eq('neutral').all():
            return None
        for row in accepted.itertuples():
            expected={row.contextual_class}
            if row.contextual_class=='neutral' and row.sentence_context_class=='neutral':
                expected.add('neutral_in_neutral_sentence')
            if row.contextual_class=='neutral' and int(row.original_sentence_label)==0:
                expected.add('neutral_in_source_neutral_sentence')
            if is_control(row.is_control) or set(row.accepted_groups.split('|'))!=expected:
                return None
        return item
    except (OSError,ValueError,KeyError,EOFError,TypeError):
        return None
