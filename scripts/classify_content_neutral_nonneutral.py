#!/usr/bin/env python3
"""Content-word sensitivity analysis using existing neutral/nonneutral features.

Only grammar eligibility changes. Sentiment labels, EEG features, quality rules,
model settings and holdout types are inherited. No feature preparation or EEG
reads. Original analyses and the running CNN have separate protocols/outputs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import classify_neutral_nonneutral as n
import analyze_trial_archive as a

TECO_GRAMMAR_SHA256='2b0cb972609f1878e17b4483023f13191fbaa422536bcffbdc57dcd39adcf347'


def grammar_source(thesis,dataset):
    if dataset=='zuco':return a.ROOT/'annotations/word_labels_three_level.csv'
    if dataset!='teco':raise ValueError('Unknown grammar dataset')
    path=thesis/'gpt uploaded files/word_sentiment_eeg/teco/word_grammar_labels.csv'
    if not path.is_file():raise ValueError('Private TeCo grammar file missing: '+str(path))
    if a.file_digest(path)!=TECO_GRAMMAR_SHA256:raise ValueError('TeCo grammar source differs from the reviewed file')
    return path


def grammar_overlay(meta,path,dataset,task):
    """Join only the independent grammar column; never replace sentiment labels."""
    keys=['sentence_key','word_index','word']
    if dataset=='teco':keys=['sentence_id_teco',*keys]
    frame=pd.read_csv(path,keep_default_na=False,dtype=str)
    if dataset=='zuco':frame=frame[frame.tasks.map(lambda value:task in value.split(';'))].copy()
    if not set(keys+['grammatical_class']).issubset(frame):raise ValueError('Missing grammar identity columns')
    if frame.duplicated(keys).any():raise ValueError('Nonunique grammar identities')
    left=meta[keys].astype(str).copy();left['_order']=np.arange(len(meta))
    merged=left.merge(frame[keys+['grammatical_class']],on=keys,how='left',validate='many_to_one',sort=False)
    if merged._order.tolist()!=list(range(len(meta))):raise ValueError('Grammar join changed event order')
    tags=merged.grammatical_class.fillna('unknown').to_numpy(dtype=str)
    tags=np.where(np.isin(tags,['content','function','uncertain']),tags,'unknown')
    result=meta.copy();result['grammatical_class']=tags
    return result


def content_selection(meta):
    """Count exclusions from the original eligible population, before filtering."""
    duplicate=meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'],keep=False)
    duplicate|=meta.signal_sha256.duplicated(keep=False)
    base=meta.loc[~duplicate].reset_index(drop=True)
    ix,y=n.population(base);eligible=base.iloc[ix]
    tags=eligible.grammatical_class.astype(str).to_numpy();content=tags=='content'
    counts=dict(original_eligible_trials=len(eligible),function_excluded=int((tags=='function').sum()),
                uncertain_or_missing_excluded=int((~np.isin(tags,['content','function'])).sum()),
                content_retained=int(content.sum()),neutral_retained=int(((y=='negative')&content).sum()),
                nonneutral_retained=int(((y=='positive')&content).sum()),
                retained_source_word_occurrences=int(eligible.loc[content,['sentence_key','word_index']].drop_duplicates().shape[0]))
    if counts['original_eligible_trials']!=counts['function_excluded']+counts['uncertain_or_missing_excluded']+counts['content_retained']:
        raise ValueError('Grammar exclusions do not reconcile')
    keep=(~duplicate)&meta.grammatical_class.eq('content')
    return np.flatnonzero(keep.to_numpy()),counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--thesis-root',type=Path,required=True);args=parser.parse_args()
    if sys.platform=='darwin':raise SystemExit('Run in Colab; no local training or feature downloads')
    source=args.thesis_root/'Results/word_sentiment_eeg/classification_similarity'
    archive=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz'
    out=source/'neutral_nonneutral_content';out.mkdir(parents=True,exist_ok=True)
    if json.loads((source/'manifest.json').read_text()).get('status')!='completed':raise ValueError('Original classification must be complete')
    records,_=a.load_completed(archive)
    paths={dataset:grammar_source(args.thesis_root,dataset) for dataset in ['zuco','teco']}
    for r in records:
        if not (source/'features'/r['dataset']/r['task']/r['subject']/'manifest.json').is_file():
            raise ValueError('Missing feature cache; no EEG will be read')
    protocol=dict(code=a.file_digest(Path(__file__)),helper=a.file_digest(Path(a.__file__)),
                  binary_helper=a.file_digest(Path(n.__file__)),source_protocol=a.file_digest(source/'protocol.json'),
                  caches=[a.file_digest(source/'features'/r['dataset']/r['task']/r['subject']/'manifest.json') for r in records],
                  grammar={d:dict(path=str(p),sha256=a.file_digest(p)) for d,p in paths.items()},
                  policy='word_agreement AND grammatical_class=content, both classes, any sentence sentiment',neutral_downsampling=False)
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('Content analysis inputs/settings changed; refusing mixed results')
    a.write_json(out/'protocol.json',protocol);a.write_json(out/'manifest.json',dict(status='running'))
    by_task=defaultdict(list);label_sources=[]
    for i,r in enumerate(records,1):
        print(f'LOAD SAVED FEATURES {i}/{len(records)}: {r["dataset"]}/{r["task"]}/{r["subject"]}',flush=True)
        meta,post,pre,nuisance,label=n.cached_record(archive,r,source,args.thesis_root)
        if label and label not in label_sources:label_sources.append(label)
        meta=grammar_overlay(meta,paths[r['dataset']],r['dataset'],r['task'])
        by_task[(r['dataset'],r['task'])].append((meta,post,pre,nuisance))
    if label_sources!=json.loads((source/'label_sources.json').read_text()):raise ValueError('Original sentiment annotation sources changed')
    scores=[];skips=[];coverage=[]
    for (dataset,task),parts in sorted(by_task.items()):
        meta=pd.concat([p[0] for p in parts],ignore_index=True)
        if meta.event_id.duplicated().any():raise ValueError('Repeated task event identity')
        arrays=[np.concatenate([p[j] for p in parts]) for j in [1,2,3]]
        keep,counts=content_selection(meta);coverage.append(dict(dataset=dataset,task=task,**counts))
        pd.DataFrame(coverage).to_csv(out/'selection_counts.csv',index=False)
        print(f'CONTENT {dataset}/{task}: {counts}',flush=True)
        if min(counts['neutral_retained'],counts['nonneutral_retained'])<10:
            skips.append(dict(dataset=dataset,task=task,reason='Fewer than ten content trials in a class'));continue
        rows,missing=n.task_analysis(meta.iloc[keep].reset_index(drop=True),*[X[keep] for X in arrays],out/dataset/task)
        scores.extend(dict(dataset=dataset,task=task,policy='content_only',**r) for r in rows)
        skips.extend(dict(dataset=dataset,task=task,**r) for r in missing)
        pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);a.check_budget(out,60_000_000)
    a.write_json(out/'skipped.json',skips)
    a.write_json(out/'manifest.json',dict(status='completed' if len(scores)==44 and not skips else 'incomplete',expected_cases=44,completed_cases=len(scores),skipped_cases=len(skips)))
    if len(scores)!=44 or skips:raise SystemExit('Some content comparisons lack coverage; inspect counts/skipped.json. Saved fits retained.')
    print('CONTENT NEUTRAL/NONNEUTRAL COMPLETE:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
