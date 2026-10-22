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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.teco import (canonical_reader,find_raw_files,normalize_sentence,
                                    sample_onsets,sentence_labels)


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
            important=('word','rawEEG','rawET','raw_data','allFixations','rawData','EEG','sfreq','sampling_rate')
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
    feature_readers=[canonical_reader(p.name) for p in paths]
    if len(feature_readers)!=len(set(feature_readers)):
        raise ValueError('Multiple TRT feature files map to the same TeCo reader; choose one source per person')
    labels_path=thesis/'Data/teco_sentiment_labels_task1.csv'
    labels=sentence_labels(labels_path)
    report={'status':'preflight_only_not_erp','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'feature_folder':str(folder),'feature_files':[],'raw_candidates':[],
            'implementation_basis':['Original Thesis-Load & Explore data.ipynb, cells 7,13,29,63,65-68',
                                    'Original Thesis-Data Structures.ipynb, cells 3,8-11',
                                    'Existing brainshaping/teco.py and EEG-spatial/eeg_spatial/teco.py',
                                    'Corrected GLIM TeCo trial-order and resampling audit'],
            'native_schema':{'reading_signal':'trial.raw_data, 126 x samples',
                             'word_fixations':'trial.word[position].rawEEG, one array per fixation',
                             'native_sampling_hz_documented':1200,
                             'sampling_basis':'Original notebook constants, not an embedded acquisition header',
                             'label_join':'Normalized Persian sentence text; native sentenceId differs from CSV sentence_id (ZuCo source ID)',
                             'fixPositions':'Fixation-order indices; never treated as EEG sample onsets',
                             'answer_data':'Separate answer-period EEG; not substituted for reading EEG'},
            'requirements_for_erp':['Continuous or sentence EEG with verified sample rate and units',
                                    'First-fixation onsets aligned to that EEG and exact Persian tokens',
                                    'Agent review of Persian word sentiment in Persian context',
                                    'Original channel order and montage; interpolation is not a substitute for timing'],
            'limitations':['TRT_total averages have no time axis and cannot provide ERP waveforms; native rawEEG segments do',
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
        info={'file':str(path),'reader':canonical_reader(path.name),'bytes':path.stat().st_size,'structure':describe(data),
              'sentence_label_matches':0,'TRT_vector_shapes':{}}
        shapes=Counter()
        for key,trial in data.items():
            if not isinstance(trial,dict):
                continue
            words,method=trial_words(trial);alignment[method]+=1
            sentence=str(trial.get('persian_sentence',''))
            sid=str(trial.get('sentenceId',key))
            label=labels.get(normalize_sentence(sentence))
            info['sentence_label_matches']+=int(label is not None)
            for word in trial.get('word',{}).values():
                shapes[str(np.asarray(word.get('TRT_total',[])).shape)]+=1
            stable=hashlib.sha256(sentence.encode()).hexdigest()[:24]
            token_list=tuple(words)
            identity=(sid,stable,token_list,method)
            if identity not in tokens:
                tokens[identity]={'sentence_id':sid,'sentence_key':stable,'sentence':sentence,
                                  'words':words,'token_alignment':method,'readers':set(),
                                  'sentence_id_zuco':int(label.sentence_id) if label is not None else None,
                                  'sentence_sentiment_original':int(label.sentiment_label) if label is not None else None,
                                  'control':str(label.control) if label is not None else '',
                                  'sentence_label_match':label is not None}
            tokens[identity]['readers'].add(canonical_reader(path.name))
        info['TRT_vector_shapes']=dict(shapes)
        report['feature_files'].append(info)
        del data;gc.collect()
    rows=[]
    for item in tokens.values():
        for index,word in item['words']:
            rows.append({'sentence_id_teco':item['sentence_id'],'sentence_id_zuco':item['sentence_id_zuco'],
                         'sentence_key':item['sentence_key'],
                         'word_index_source':index,'word':word,'sentence':item['sentence'],
                         'token_alignment':item['token_alignment'],'reader_count':len(item['readers']),
                         'sentence_sentiment_original':item['sentence_sentiment_original'],
                         'control':item['control'],'sentence_label_match':item['sentence_label_match'],
                         'annotation_status':'pending_Persian_agent_review'})
    pd.DataFrame(rows,columns=['sentence_id_teco','sentence_id_zuco','sentence_key','word_index_source','word','sentence',
                              'token_alignment','reader_count','sentence_sentiment_original','control',
                              'sentence_label_match','annotation_status']).to_csv(out/'word_inventory.csv',index=False)
    rawfiles,alternates=find_raw_files(thesis/'Data')
    report['raw_candidates']=[{'reader':reader,'file':str(p),'bytes':p.stat().st_size} for reader,p in sorted(rawfiles.items())]
    report['alternate_raw_copies_not_counted_twice']=alternates
    report['feature_reader_files']=len(paths)
    report['word_inventory_rows']=len(rows)
    report['sentence_token_versions']=len(tokens)
    report['token_alignment_methods']=dict(alignment)
    report['raw_inspection']='not_requested' if not inspect_raw else 'no_candidates'
    # Preserve the small feature inventory before opening a large raw pickle.
    (out/'structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if inspect_raw and rawfiles:
        reader,path=sorted(rawfiles.items())[0]
        print(f'TECO RAW STRUCTURE {path} ({path.stat().st_size/1e9:.2f} GB existing file; no download)',flush=True)
        report['raw_inspection']='started_not_yet_completed'
        (out/'structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        try:
            with path.open('rb') as handle:
                raw=pickle.load(handle)
            report['sample_raw_structure']={'file':str(path),'structure':describe(raw)}
            coverage,checks=sample_onsets(raw,labels)
            report['sample_native_coverage']=coverage
            report['sample_onset_checks']=checks
            report['sample_match_counts']=dict(Counter(c['status'] for c in checks))
            report['raw_inspection']='sample_checked_not_full_reader_or_erp_audit'
            pd.DataFrame(checks).to_csv(out/'sample_word_onsets.csv',index=False)
            print('Native TeCo sample matching:',report['sample_match_counts'],flush=True)
            del raw;gc.collect()
        except Exception as error:
            report['sample_raw_error']=f'{type(error).__name__}: {error}'
            report['raw_inspection']='failed_see_error'
    report['next_step']='Use native 126-channel raw_data/rawEEG and verified sentence joins for TeCo epochs, after Persian word-label review and full onsets validation. No TeCo ERP has been calculated.'
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
