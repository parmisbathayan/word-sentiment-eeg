"""Input counting: repeated items, missing readers, controls and old acceptance."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('inventory',ROOT/'scripts/report_input_inventory.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


for dataset in ['zuco','teco']:
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);run=root/'run';run.mkdir();task='TSR' if dataset=='zuco' else 'task1'
        labels=[]
        for index,cls in enumerate(['positive','negative','neutral','neutral','positive','positive']):
            labels.append(dict(sentence_key='source',word_index=index,word='w'+str(index),tasks=task,
                sentence_id_teco=1,lexical_class=cls,contextual_class='negative' if index==5 else cls,
                sentence_context_class=cls,grammatical_class='function' if index==3 else 'content',
                alignment_warning='',annotation_status='agent_reviewed_not_human_gold',is_control=index==4))
        annotation=root/'labels.csv';write(annotation,labels)
        observed={'A':[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,2)],'B':[(0,0),(0,2)]}
        paths=[]
        for reader,positions in observed.items():
            events=[]
            for presentation,index in positions:
                label=labels[index]
                events.append(dict(event_id=f'{reader}_{presentation}_{index}',task=task,subject=reader,
                    sentence_key='source',word_index=index,word=label['word'],sentence_id_teco=1,
                    raw_sentence_position=presentation,sentence_position=presentation,
                    onset_sample_sentence_0based='' if index==4 else 400,
                    contextual_class=label['contextual_class'],status='control_sentence' if index==4 else 'accepted'))
            folder=run/'readers'/task/reader if dataset=='zuco' else run/'readers'/reader
            path=folder/'word_onsets.csv.gz';write(path,events);paths.append(path)
        if dataset=='zuco':
            manifest={'status':'incomplete','expected_files':3,'files':[{'task':task,'subject':r} for r in ['A','B']], 'pending_new_readers':[task+'/C']}
        else:
            manifest={'status':'incomplete','expected_readers':['A','B','C'],'readers':[{'subject':r} for r in ['A','B']]}
        (run/'manifest.json').write_text(json.dumps(manifest))
        old_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [*paths,run/'manifest.json']}
        out=module.inventory(run,dataset,annotation)
        data=module.read_csv(out/'input_counts.csv');by={(x['policy'],x['class']):x for x in data}
        pos=by['all_three','positive'];neutral=by['all_three','neutral'];negative=by['all_three','negative']
        assert int(pos['unique_source_positions'])==2 # includes previously excluded control
        assert int(pos['theoretical_unique_word_reader_pairs'])==6 # expected3 not observed2
        assert int(pos['observed_unique_word_reader_pairs'])==3
        assert int(pos['observed_word_presentations'])==3
        assert int(pos['observed_presentations_with_old_recovered_onset'])==2
        assert int(neutral['unique_source_positions'])==1 # grammatical function excluded
        assert int(neutral['observed_unique_word_reader_pairs'])==2
        assert int(neutral['observed_word_presentations'])==3 # repeated sentence presentation
        assert int(negative['observed_unique_word_reader_pairs'])==1 # B missing negative
        assert int(by['lexical_only','positive']['unique_source_positions'])==3
        assert int(by['contextual_only','negative']['unique_source_positions'])==2
        per=module.read_csv(out/'per_reader_input_counts.csv')
        assert all(x['onset_table_available']=='False' and x['observed_word_presentations']=='0' for x in per if x['subject']=='C')
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in old_hashes.items())
        # Duplicate word/event rows must not inflate counts.
        bad=module.read_csv(paths[0]);write(paths[0],bad+[bad[0]])
        try:module.inventory(run,dataset,annotation)
        except ValueError as error:assert 'duplicate event' in str(error)
        else:raise AssertionError('Duplicate event accepted')
        # Exact word joins cannot silently disappear from the denominator.
        bad[0]['word']='wrong';write(paths[0],bad)
        try:module.inventory(run,dataset,annotation)
        except ValueError as error:assert 'nonexact source word' in str(error)
        else:raise AssertionError('Unknown token accepted')
print('Input inventory tests passed for ZuCo and TeCo')
