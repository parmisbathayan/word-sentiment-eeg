"""Rebuild context controls while preserving original extraction and event identity.

Default runs genuine MATLAB/filter integration in cloud. --unit-only uses the
same source generator with a mocked reader/filter and needs only numpy/pandas.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_zuco_context_controls as context
from word_sentiment_eeg.common import atomic_npz,file_digest,sentence_key,write_json


def fixture_and_checks(directory,unit_only):
    rawdir=directory/'Data';rawdir.mkdir()
    source=rawdir/'resultsZAB_SR.mat'
    rng=np.random.default_rng(984)
    channels=[f'E{i}' for i in range(105)]
    raws=[];sentences=[];annotation_rows=[];contexts={};ratings={}
    starts=[500,1100,1700]
    for index in range(2):
        raw=rng.normal(size=(105,3000)).astype(np.float32);raw[104]=0;raws.append(raw)
        words=[f'bad{index}',f'plain{index}',f'good{index}']
        key=sentence_key(words)
        contexts[key]=SimpleNamespace(sentence_context_class='neutral' if index==0 else 'negative')
        ratings[key]='neutral'
        sentence={'content':' '.join(words),'rawData':raw,'word':[]}
        for j,(word,cls) in enumerate(zip(words,context.CLASSES)):
            sentence['word'].append({'content':word,'rawEEG':raw[:,starts[j]:starts[j]+70]})
            annotation_rows.append({'sentence_key':key,'word_index':j,'word':word,'tasks':'SR',
                                    'lexical_class':cls,'contextual_class':cls,'alignment_warning':'',
                                    'annotation_status':'agent_reviewed_not_human_gold'})
        sentences.append(sentence)
    if unit_only:
        source.write_bytes(b'fixture source bytes, protected by SHA; mocked reader/filter in unit mode')
    else:
        from scipy.io import savemat
        savemat(source,{'sentenceData':sentences})
    def iter_fixture(_path):
        for i,sentence in enumerate(sentences):
            yield i,[w['content'] for w in sentence['word']],raws[i],[[w['rawEEG']] for w in sentence['word']]
    base=directory/'base/readers/SR/ZAB'
    record={'task':'SR','subject':'ZAB','filename':source.name}
    with patch.object(context.original,'iter_sentences',side_effect=iter_fixture) if unit_only else patch.object(context.original,'iter_sentences',wraps=context.original.iter_sentences):
        with patch.object(context.original,'preprocess',side_effect=lambda value:value.copy()) if unit_only else patch.object(context.original,'preprocess',wraps=context.original.preprocess):
            item=context.original.process_file(source,record,pd.DataFrame(annotation_rows),base,'base-fixture',channels,save_trials=False)
            assert item['epochs_per_class']==dict.fromkeys(context.CLASSES,2)
            original_files={p:file_digest(p) for p in base.rglob('*') if p.is_file()}
            target=directory/'new/readers/SR/ZAB'
            events=pd.read_csv(base/'word_onsets.csv.gz',keep_default_na=False)
            selected=context.select_events(events,contexts,ratings)
            assert len(selected)==6 and selected.event_id.nunique()==4
            assert not selected.duplicated(['event_id','cohort']).any()
            # Source review-neutral applies to SR only, even when identical text
            # appears in another task or an accidental mapping is supplied.
            nr_events=events.copy();nr_events['task']='NR'
            assert not context.select_events(nr_events,contexts,ratings).cohort.eq(context.SOURCE_NEUTRAL).any()
            assert not context.select_events(events,contexts).cohort.eq(context.SOURCE_NEUTRAL).any()
            # No fixation matching should occur during saved-onset reconstruction.
            with patch.object(context.original,'locate',side_effect=AssertionError('Onsets must be reused')):
                with patch.object(context.original,'epoch',wraps=context.original.epoch) as calls:
                    result=context.process_record(source,base,target,item,contexts,'new-fixture',channels,source_ratings=ratings)
                    assert calls.call_count==4 # Six memberships, four physical epochs.
            assert result['selected_events']==4 and result['cohort_memberships']==6
            assert result['counts'][context.SOURCE_NEUTRAL]==2
            assert result['counts']['neutral_in_neutral_sentence']==1
            assert result['counts']['neutral_in_negative_sentence']==1
            with np.load(base/'class_means.npz',allow_pickle=False) as original_means:
                with np.load(target/'cohort_means.npz',allow_pickle=False) as rebuilt:
                    np.testing.assert_array_equal(rebuilt['mean_uV'][:3],original_means['mean_uV'])
                    np.testing.assert_array_equal(rebuilt['counts'][:3],original_means['counts'])
                    # All original neutral events belong to source-neutral here.
                    np.testing.assert_array_equal(rebuilt['mean_uV'][context.GROUPS.index(context.SOURCE_NEUTRAL)],original_means['mean_uV'][1])
            # Resume validates the stored result and never opens source EEG.
            with patch.object(context.original,'iter_sentences',side_effect=AssertionError('Resume must not reopen source')):
                assert context.checkpoint(target,base,'new-fixture',channels,contexts,ratings) is not None
            try:
                context.checkpoint(target,base,'changed-labels',channels,contexts,ratings)
                raise AssertionError('Mismatched analysis fingerprint was reused')
            except ValueError as error:
                assert 'different code/labels' in str(error)
            try:
                context.process_record(source,base,target,item,contexts,'new-fixture',channels,source_hash='wrong',source_ratings=ratings)
                raise AssertionError('Source hash mismatch accepted')
            except ValueError as error:
                assert 'differs from the original' in str(error)
            # Changed membership must not silently reuse an old checkpoint.
            assert context.checkpoint(target,base,'new-fixture',channels,contexts,{}) is None
            # Detect corrupted ZIP/NPZ, then rebuild the new result only.
            (target/'cohort_means.npz').write_bytes(b'incomplete write')
            assert context.checkpoint(target,base,'new-fixture',channels,contexts,ratings) is None
            context.process_record(source,base,target,item,contexts,'new-fixture',channels,source_ratings=ratings)
            # Even if a writer updates hashes, a changed copied original mean is
            # rejected by readback against the immutable base class means.
            with np.load(target/'cohort_means.npz',allow_pickle=False) as data:
                arrays={name:data[name].copy() for name in data.files}
            arrays['mean_uV'][0,0,0]+=1
            atomic_npz(target/'cohort_means.npz',**arrays)
            saved=json.loads((target/'manifest.json').read_text())
            saved['files']['cohort_means.npz']=file_digest(target/'cohort_means.npz')
            write_json(target/'manifest.json',saved)
            assert context.checkpoint(target,base,'new-fixture',channels,contexts,ratings) is None
            context.process_record(source,base,target,item,contexts,'new-fixture',channels,source_ratings=ratings)
            # Duplicate membership is invalid, although a physical event can
            # legitimately appear once in each of two distinct controls.
            selected_path=target/'selected_events.csv.gz'
            selected=pd.read_csv(selected_path,keep_default_na=False)
            selected=pd.concat([selected,selected.iloc[:1]],ignore_index=True)
            selected.to_csv(selected_path,index=False,compression='gzip')
            saved=json.loads((target/'manifest.json').read_text())
            saved['files']['selected_events.csv.gz']=file_digest(selected_path)
            saved['cohort_memberships']=len(selected)
            write_json(target/'manifest.json',saved)
            assert context.checkpoint(target,base,'new-fixture',channels,contexts,ratings) is None
            (target/'manifest.json').write_text('{partial json')
            assert context.checkpoint(target,base,'new-fixture',channels,contexts,ratings) is None
            for path,sha in original_files.items():
                assert file_digest(path)==sha, f'Original result changed: {path}'


unit_only='--unit-only' in sys.argv
with tempfile.TemporaryDirectory() as temp:
    fixture_and_checks(Path(temp),unit_only)
ratings=context.load_source_ratings(ROOT/'annotations/source_sentence_ratings.csv')
assert len(ratings)==400 and sum(v=='neutral' for v in ratings.values())==137
source=json.loads((ROOT/'annotations/source_sentences.json').read_text())
assert set(ratings)=={s['sentence_key'] for s in source if 'SR' in s['tasks']}
print('ZuCo context controls passed: saved-onset reconstruction, original preservation, dual membership, source-rated neutral, corruption/readback/resume; mode='+('unit-only' if unit_only else 'native MATLAB and filter integration'))
