"""Native Persian TeCo schema, traced from the user's existing research scripts.

Reading EEG: trial['raw_data'], 126 channels, documented 1200 Hz.
Fixations: trial['word'][position]['rawEEG'], a list of separate waveforms.
CSV sentence_id is the English source ID, not the native TeCo sentenceId.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from .erp import locate

N_CHANNELS=126
NATIVE_HZ=1200


def normalize_sentence(text):
    return re.sub(r'\s+','',str(text)).replace('\u200c','')


def canonical_reader(name):
    value=str(name).lower().removesuffix('_trt_total.pickle').removesuffix('_with_total_task1')
    return {'amin':'armin','razieh':'razie','shahrzad':'sharzad'}.get(value,value)


def sentence_labels(path):
    frame=pd.read_csv(path,keep_default_na=False)
    required={'sentence_id','sentence','sentiment_label','control'}
    if not required<=set(frame):
        raise ValueError(f'TeCo sentence CSV lacks {sorted(required-set(frame))}')
    keys=frame.sentence.map(normalize_sentence)
    if keys.duplicated().any():
        raise ValueError('TeCo label CSV has ambiguous normalized Persian sentences')
    return {normalize_sentence(r.sentence):r for r in frame.itertuples()}


def find_raw_files(data_root):
    """Known layouts, in priority order; choose at most one file per reader.

    Alternate copies are reported, not counted as extra subjects. Selection is
    explicit, not a claim that alternate files are byte-identical.
    """
    root=Path(data_root)
    patterns=('teco_pickles/*/task1.pickle','pickle_files_acc3/*/task1.pickle',
              'pickle_files_acc5/*/task1.pickle','TeCo/task1_with_total/*/task1.pickle',
              'task1_with_total/*/task1.pickle')
    selected={};alternates=[]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            reader=canonical_reader(path.parent.name)
            if reader in selected:
                if path!=selected[reader]:
                    alternates.append({'reader':reader,'selected':str(selected[reader]),'alternate':str(path)})
            else:
                selected[reader]=path
    return selected,alternates


def orient_native(value):
    try:
        array=np.asarray(value)
    except (TypeError,ValueError):
        return None
    if array.ndim!=2 or not np.issubdtype(array.dtype,np.number):
        return None
    if array.shape[0]==N_CHANNELS:
        return array
    if array.shape[1]==N_CHANNELS:
        return array.T
    return None


def first_fixation(word):
    """Preserve missing first fixations; never concatenate or promote a later one."""
    try:
        count=np.asarray(word.get('nFixations',0),dtype=float).reshape(-1)
    except (TypeError,ValueError):
        return None,'invalid_fixation_count'
    if count.size!=1 or not np.isfinite(count[0]) or count[0]<=0:
        return None,'no_fixation'
    value=word.get('rawEEG')
    if isinstance(value,(list,tuple)):
        first=value[0] if value else None
    elif isinstance(value,np.ndarray) and value.dtype==object:
        first=value.flat[0] if value.size else None
    else:
        first=value
    segment=orient_native(first)
    if segment is None or segment.shape[1]<3 or not np.isfinite(segment).all():
        return None,'missing_or_invalid_first_fixation'
    return segment,'available'


def sample_onsets(data,labels=None,max_trials=4,words_per_trial=5):
    """Bounded onsets-only verification; this is not a full ERP extraction."""
    checks=[];coverage={'trials':len(data),'valid_reading_arrays':0,'matched_sentence_labels':0,
                        'control_trials':0,'controls_with_reading_arrays':0}
    for position,(trial_key,trial) in enumerate(data.items()):
        if not isinstance(trial,dict):
            continue
        raw=orient_native(trial.get('raw_data'))
        valid=raw is not None and raw.shape[1]>=3
        coverage['valid_reading_arrays']+=int(valid)
        label=labels.get(normalize_sentence(trial.get('persian_sentence',''))) if labels else None
        coverage['matched_sentence_labels']+=int(label is not None)
        control=bool(label is not None and str(label.control).upper()=='CONTROL')
        coverage['control_trials']+=int(control)
        coverage['controls_with_reading_arrays']+=int(control and valid)
        if position>=max_trials:
            continue
        sampled=0
        for index,word in sorted(trial.get('word',{}).items(),key=lambda kv:int(kv[0])):
            segment,status=first_fixation(word)
            if status=='no_fixation':
                continue
            start=None
            if not valid:
                status='missing_reading_raw_data'
            elif not np.isfinite(raw).all():
                status='nonfinite_reading_raw_data'
            elif segment is not None:
                start,status=locate(raw,segment)
            checks.append({'raw_trial_key':str(trial_key),'sentence_id_teco':int(trial['sentenceId']),
                           'sentence_id_zuco':int(label.sentence_id) if label is not None else None,
                           'word_index_source':str(index),'status':status,
                           'onset_sample_reading_0based':start,
                           'onset_ms_at_documented_1200_hz':start*1000/NATIVE_HZ if start is not None else None,
                           'reading_shape':list(raw.shape) if valid else None,
                           'first_segment_shape':list(segment.shape) if segment is not None else None,
                           'is_control':control})
            sampled+=1
            if sampled>=words_per_trial:
                break
    return coverage,checks
