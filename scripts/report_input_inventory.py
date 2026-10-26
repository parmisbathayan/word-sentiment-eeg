"""Inventory new label policies against saved word rows, without loading EEG.

Available = a row exists in a saved word_onsets.csv.gz, NOT usable fixation or
accepted EEG. Historical 30 Hz acceptance is reported separately only.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

CLASSES = ('positive','negative','neutral')
POLICIES = ('all_three','lexical_only','contextual_only','sentence_only')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def integer(value):
    text = str(value)
    number = float(text)
    if not number.is_integer():
        raise ValueError('Noninteger identity: '+text)
    return int(number)


def key(row):
    return (str(row['sentence_key']),integer(row['word_index']))


def selected(row, policy):
    # Recompute from explicit labels; never reuse old selection_eligible flags.
    if row['grammatical_class'] != 'content' or row.get('alignment_warning',''):
        return None
    if row.get('annotation_status')!='agent_reviewed_not_human_gold':
        return None
    lexical=row['lexical_class']; contextual=row.get('contextual_word_class',row.get('contextual_class',''));sentence=row['sentence_context_class']
    if policy=='all_three':
        return lexical if lexical in CLASSES and lexical==contextual==sentence else None
    label={'lexical_only':lexical,'contextual_only':contextual,'sentence_only':sentence}[policy]
    return label if label in CLASSES else None


def expected_roster(run, manifest, dataset):
    if dataset=='teco':
        roster={'task1':set(manifest['expected_readers'])}
        records=[('task1',x['subject']) for x in manifest['readers']]
    else:
        records=[(x['task'],x['subject']) for x in manifest['files']]
        provenance=run/'provenance.json'
        expected=json.loads(provenance.read_text()).get('expected_sources',[]) if provenance.exists() else []
        if expected:
            pairs=[(x['task'],x['subject']) for x in expected]
            if len(pairs)!=len(set(pairs)):
                raise ValueError('Duplicate expected source identity')
        else:
            pairs=list(records)
            pairs += [tuple(x.split('/',1)) for x in manifest.get('pending_new_readers',[])]
            pairs += [(x['task'],x['subject']) for x in manifest.get('failures',[])]
            pairs=sorted(set(pairs))
            if manifest.get('expected_files',len(pairs))!=len(pairs):
                raise ValueError('Cannot establish full expected reader roster; need provenance expected_sources')
        roster=defaultdict(set)
        for task,reader in pairs:
            roster[task].add(reader)
    if len(records)!=len(set(records)):
        raise ValueError('Duplicate reader/task manifest record')
    if any(reader not in roster.get(task,set()) for task,reader in records):
        raise ValueError('Observed recording outside expected roster')
    return roster,records


def inventory(run_dir,dataset,annotations):
    run=Path(run_dir);annotations=Path(annotations)
    manifest=json.loads((run/'manifest.json').read_text())
    roster,records=expected_roster(run,manifest,dataset)
    labels=read_csv(annotations)
    lookup={key(x):x for x in labels}
    if len(lookup)!=len(labels):
        raise ValueError('Duplicate annotation identity')
    required={'sentence_key','word_index','word','lexical_class','contextual_class','sentence_context_class','grammatical_class','annotation_status'}
    if not labels or not required.issubset(labels[0]):
        raise ValueError('Missing explicit three-level annotation columns')
    source=defaultdict(set)
    for label in labels:
        if label['lexical_class'] not in set(CLASSES)|{'uncertain'} or label['contextual_class'] not in set(CLASSES)|{'uncertain'}:
            raise ValueError('Unknown word sentiment class')
        if label.get('contextual_word_class',label['contextual_class']) != label['contextual_class']:
            raise ValueError('Contextual word alias differs from contextual_class')
        if label['sentence_context_class'] not in set(CLASSES)|{'mixed','uncertain'}:
            raise ValueError('Unknown sentence sentiment class')
        if label['grammatical_class'] not in {'content','function','uncertain'}:
            raise ValueError('Unknown grammatical class')
        if label['annotation_status']!='agent_reviewed_not_human_gold':
            raise ValueError('Missing explicit reviewed annotation')
        tasks=label['tasks'].split(';') if dataset=='zuco' else ['task1']
        if dataset=='zuco' and not label.get('tasks'):
            raise ValueError('Missing annotation task membership')
        for task in tasks:
            for policy in POLICIES:
                cls=selected(label,policy)
                if cls:
                    source[(task,policy,cls)].add(key(label))
    observed_positions=defaultdict(set);pairs=defaultdict(set);presentations=Counter()
    reader_positions=defaultdict(set);reader_presentations=Counter();reader_sentences=defaultdict(set)
    onset_count=Counter();statuses=Counter();old=Counter();all_ids=set();source_hashes={};observed_readers=defaultdict(set)
    for task,reader in records:
        folder=run/'readers'/task/reader if dataset=='zuco' else run/'readers'/reader
        path=folder/'word_onsets.csv.gz';events=read_csv(path)
        source_hashes[str(path.relative_to(run))]=digest(path)
        observed_readers[task].add(reader)
        identities=set()
        for event in events:
            if event.get('task')!=task or event.get('subject')!=reader:
                raise ValueError('Saved onset task/reader identity mismatch')
            identity=event.get('event_id')
            if not identity or identity in all_ids:
                raise ValueError('Missing or duplicate event ID across input files')
            all_ids.add(identity)
            position_column='raw_sentence_position' if dataset=='zuco' else 'sentence_position'
            position=integer(event[position_column])
            presentation=(position,integer(event['word_index']))
            if presentation in identities:
                raise ValueError('Duplicate word presentation')
            identities.add(presentation)
            source_key=key(event)
            label=lookup.get(source_key)
            if label is None or label['word']!=event['word']:
                raise ValueError('Unknown or nonexact source word alignment: '+str(source_key))
            if dataset=='zuco' and task not in label['tasks'].split(';'):
                raise ValueError('Event is outside annotation task membership')
            if dataset=='teco' and integer(label['sentence_id_teco'])!=integer(event['sentence_id_teco']):
                raise ValueError('TeCo sentence ID mismatch')
            status=event.get('status','unknown')
            statuses[(task,reader,status)]+=1
            if status=='accepted':
                old_class=event.get('contextual_class','unknown')
                if old_class not in CLASSES:
                    raise ValueError('Historical accepted row missing historical class')
                old[(task,reader,old_class)]+=1
            for policy in POLICIES:
                cls=selected(label,policy)
                if cls is None:
                    continue
                bucket=(task,policy,cls)
                observed_positions[bucket].add(source_key)
                pairs[bucket].add((reader,*source_key))
                presentations[bucket]+=1
                rb=(*bucket,reader)
                reader_positions[rb].add(source_key);reader_presentations[rb]+=1
                reader_sentences[rb].add((position,event['sentence_key']))
                if event.get('onset_sample_sentence_0based','') not in ('','nan','None'):
                    onset_count[bucket]+=1
    summary=[];per_reader=[]
    tasks=sorted(set(roster)|{x[0] for x in source})
    for task in tasks:
        expected=roster.get(task,set())
        for policy in POLICIES:
            for cls in CLASSES:
                bucket=(task,policy,cls)
                summary.append(dict(task=task,policy=policy,**{'class':cls},
                    unique_source_positions=len(source[bucket]),
                    unique_source_sentences=len({x[0] for x in source[bucket]}),
                    available_source_positions=len(observed_positions[bucket]),
                    readers_expected=len(expected),readers_observed=len(observed_readers[task]),
                    readers_with_candidate_input=sum(bool(reader_presentations[(*bucket,r)]) for r in expected),
                    theoretical_unique_word_reader_pairs=len(source[bucket])*len(expected),
                    observed_unique_word_reader_pairs=len(pairs[bucket]),
                    observed_word_presentations=presentations[bucket],
                    observed_presentations_with_old_recovered_onset=onset_count[bucket]))
                for reader in sorted(expected):
                    rb=(*bucket,reader)
                    per_reader.append(dict(task=task,subject=reader,policy=policy,**{'class':cls},
                        onset_table_available=reader in observed_readers[task],
                        available_unique_source_positions=len(reader_positions[rb]),
                        observed_word_presentations=reader_presentations[rb],
                        observed_sentence_presentations=len(reader_sentences[rb])))
    output=run/'analysis/three_level_input_inventory';output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'input_counts.csv',summary,list(summary[0]))
    save_csv(output/'per_reader_input_counts.csv',per_reader,list(per_reader[0]))
    old_rows=[dict(task=t,subject=r,**{'class':c},old30hz_oldpolicy_accepted_presentations=old[(t,r,c)]) for t,r in records for c in CLASSES]
    save_csv(output/'OLD30HZ_OLDPOLICY_accepted_counts.csv',old_rows,list(old_rows[0]) if old_rows else ['task','subject','class','old30hz_oldpolicy_accepted_presentations'])
    status_rows=[dict(task=t,subject=r,historical_status=s,word_presentations=n) for (t,r,s),n in sorted(statuses.items())]
    save_csv(output/'historical_status_inventory.csv',status_rows,list(status_rows[0]) if status_rows else ['task','subject','historical_status','word_presentations'])
    report={'dataset':dataset,'run_status':manifest.get('status','unknown'),'annotations_sha256':digest(annotations),
        'script_sha256':digest(Path(__file__)),'input_onset_sha256':source_hashes,
        'label_policy':'Four independent content-only policies, neutral retained, TeCo control reading included as candidate input; no class balancing.',
        'availability_definition':'An exact word row exists in a saved reader onset table. Does NOT establish a usable fixation, raw EEG, or accepted epoch under new settings.',
        'unique_identity':'sentence_key + word_index; unique word-reader pairs collapse repeated sentence presentations within a reader/task.',
        'presentation_identity':'task + reader + raw sentence position + word_index; TSR repeated source sentences count separately.',
        'historical_acceptance':'OLD30HZ_OLDPOLICY file only; not new 40 Hz acceptance. Old unprocessed/excluded candidates can lack recovered onsets.',
        'source_eeg_read':False,'new_eeg_acceptance_computed':False,'old_files_modified':False,
        'source_word_positions':len(labels),'observed_word_presentations_all_labels':sum(statuses.values()),
        'expected_readers_by_task':{t:sorted(r) for t,r in roster.items()},
        'unknown_alignments':0}
    (output/'inventory_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (output/'README.txt').write_text('INPUT COUNTS ONLY\n\n'+report['availability_definition']+'\n\n'+report['historical_acceptance']+'\n\nUnique source word positions, observed word-reader pairs, and observed presentation rows answer different questions. Missing reader tables do not imply zero underlying data; they mean unavailable in this saved run.\n')
    print('Input inventory saved:',output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--dataset',required=True,choices=['zuco','teco'])
    parser.add_argument('--annotations',required=True,type=Path)
    args=parser.parse_args();inventory(args.run_dir,args.dataset,args.annotations)
