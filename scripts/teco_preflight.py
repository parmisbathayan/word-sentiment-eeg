"""Inspect the user's Persian TeCo files in Colab; never infer missing word onsets.

This reads existing trusted research pickles on mounted Drive. It writes a small
private structure report and exact token inventory to Drive, not the public repo.
It deliberately does not label Persian words from their English translations or
turn time-collapsed TRT_total features into ERP waveforms.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


TEXT_KEYS=('content','word','text','persian_word','token')


def describe(value,depth=0):
    """Bounded shape/key summary; never serialize EEG samples."""
    if isinstance(value,np.ndarray):
        result={'type':'ndarray','shape':list(value.shape),'dtype':str(value.dtype)}
        if value.dtype==object and depth<4:
            result['first_items']=[describe(v,depth+1) for v in value.flat[:2]]
        return result
    if isinstance(value,pd.DataFrame):
        return {'type':'DataFrame','shape':list(value.shape),'columns':list(map(str,value.columns)),
                'dtypes':{str(k):str(v) for k,v in value.dtypes.items()}}
    if isinstance(value,dict):
        result={'type':'dict','size':len(value),'keys':[str(k) for k in list(value)[:80]]}
        if depth<4:
            important=('word','rawEEG','rawData','EEG','eeg','fixations','fixation','sfreq','sampling_rate','sample_rate')
            selected=[k for k in important if k in value]
            selected.extend(k for k in list(value)[:5] if k not in selected)
            result['first_items']={str(k):describe(value[k],depth+1) for k in selected[:10]}
        return result
    if isinstance(value,(list,tuple)):
        return {'type':type(value).__name__,'length':len(value),
                'first_items':[describe(v,depth+1) for v in value[:2]] if depth<4 else []}
    if isinstance(value,str):
        return {'type':'str','characters':len(value)}
    if isinstance(value,(int,float,np.number,bool)):
        return {'type':type(value).__name__,'value':float(value)}
    return {'type':type(value).__name__}


def trial_words(trial):
    words=trial.get('word',{})
    if not isinstance(words,dict):
        return [],'word_container_not_dict'
    indices=sorted(words,key=int)
    if not indices:
        return [],'empty_word_list'
    values=[words[k] for k in indices]
    for field in TEXT_KEYS:
        if all(isinstance(w,dict) and isinstance(w.get(field),str) and w[field].strip() for w in values):
            return [(str(i),w[field].strip()) for i,w in zip(indices,values)],f'explicit_word_field:{field}'
    split=str(trial.get('persian_sentence','')).split()
    if len(split)==len(indices):
        return list(zip(map(str,indices),split)),'whitespace_count_matches_needs_verification'
    return [],f'unresolved_token_alignment:{len(indices)}_word_entries_{len(split)}_text_tokens'


def preflight(thesis,inspect_raw=False):
    thesis=Path(thesis)
    out=thesis/'Results/word_sentiment_eeg/teco_preflight'
    out.mkdir(parents=True,exist_ok=True)
    folder=thesis/'Data/TRT_Total'
    paths=sorted(folder.glob('*_trt_total.pickle'))
    report={'status':'preflight_only_not_erp','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'feature_folder':str(folder),'feature_files':[],'raw_candidates':[],
            'requirements_for_erp':['Continuous or sentence EEG with verified sample rate and units',
                                    'First-fixation onsets aligned to that EEG and exact Persian tokens',
                                    'Agent review of Persian word sentiment in Persian context',
                                    'Original channel order and montage; interpolation is not a substitute for timing'],
            'limitations':['TRT_total averages have no time axis and cannot provide ERP waveforms',
                           'Existing sentence EEG arrays do not establish word fixation onsets',
                           'Prior TeCo intermediate named 500 Hz was inferred as 600 Hz; verify raw metadata before epoching',
                           'English word labels are not transferred to Persian tokens']}
    tokens={};alignment=Counter()
    for path in paths:
        print(f'TECO FEATURES {path.name}',flush=True)
        with path.open('rb') as handle:
            data=pickle.load(handle)
        if not isinstance(data,dict):
            raise ValueError(f'{path.name}: expected trial dictionary')
        info={'file':str(path),'bytes':path.stat().st_size,'structure':describe(data)}
        for key,trial in data.items():
            if not isinstance(trial,dict):
                continue
            words,method=trial_words(trial);alignment[method]+=1
            sentence=str(trial.get('persian_sentence',''))
            sid=str(trial.get('sentenceId',key))
            stable=hashlib.sha256(sentence.encode()).hexdigest()[:24]
            token_list=tuple(words)
            identity=(sid,stable,token_list,method)
            if identity not in tokens:
                tokens[identity]={'sentence_id':sid,'sentence_key':stable,'sentence':sentence,
                                  'words':words,'token_alignment':method,'readers':set()}
            tokens[identity]['readers'].add(path.name.removesuffix('_trt_total.pickle'))
        report['feature_files'].append(info)
        del data;gc.collect()
    rows=[]
    for item in tokens.values():
        for index,word in item['words']:
            rows.append({'sentence_id':item['sentence_id'],'sentence_key':item['sentence_key'],
                         'word_index_source':index,'word':word,'sentence':item['sentence'],
                         'token_alignment':item['token_alignment'],'reader_count':len(item['readers']),
                         'annotation_status':'pending_Persian_agent_review'})
    pd.DataFrame(rows,columns=['sentence_id','sentence_key','word_index_source','word','sentence',
                              'token_alignment','reader_count','annotation_status']).to_csv(out/'word_inventory.csv',index=False)
    rawroot=thesis/'Data/teco_pickles'
    rawpaths=sorted(rawroot.glob('*/task1.pickle')) if rawroot.exists() else []
    report['raw_candidates']=[{'file':str(p),'bytes':p.stat().st_size} for p in rawpaths]
    report['feature_reader_files']=len(paths)
    report['word_inventory_rows']=len(rows)
    report['sentence_token_versions']=len(tokens)
    report['token_alignment_methods']=dict(alignment)
    report['raw_inspection']='not_requested' if not inspect_raw else 'no_candidates'
    # Preserve the small feature inventory before opening a large raw pickle.
    (out/'structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if inspect_raw and rawpaths:
        path=rawpaths[0]
        print(f'TECO RAW STRUCTURE {path} ({path.stat().st_size/1e9:.2f} GB existing file; no download)',flush=True)
        report['raw_inspection']='started_not_yet_completed'
        (out/'structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        try:
            with path.open('rb') as handle:
                raw=pickle.load(handle)
            report['sample_raw_structure']={'file':str(path),'structure':describe(raw)}
            report['raw_inspection']='sample_inspected_onsets_not_yet_verified'
            del raw;gc.collect()
        except Exception as error:
            report['sample_raw_error']=f'{type(error).__name__}: {error}'
            report['raw_inspection']='failed_see_error'
    report['next_step']='Inspect this structure report before implementing word-onset extraction. No TeCo ERP has been calculated.'
    (out/'structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','feature_reader_files','word_inventory_rows','token_alignment_methods','next_step')},ensure_ascii=False,indent=2),flush=True)
    print('TeCo preflight saved:',out,flush=True)
    if not paths:
        raise FileNotFoundError(f'No feature pickles found in {folder}; diagnostic report was saved. Check the mounted account.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--inspect-raw',action='store_true',help='Load one existing multi-GB TeCo pickle in cloud memory for a shape/key summary')
    args=parser.parse_args()
    preflight(args.thesis_root,args.inspect_raw)
