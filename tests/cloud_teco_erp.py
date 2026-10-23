"""Synthetic native TeCo extraction/resume tests; no private text or EEG inputs."""
from pathlib import Path
import hashlib
import json
import tempfile
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import file_digest,write_json
from word_sentiment_eeg.teco_erp import (GROUPS,TIMES_MS,PRE,load_labels,groups_for_label,extract_reader,inspect_reader)
from run_teco_erp import load_montage,summarize,save_state


def main():
    rng=np.random.default_rng(197)
    montage=load_montage(ROOT/'assets/teco_montage.txt')
    channels=montage.channel.tolist()
    assert len(channels)==126 and len(set(montage.region))==6
    with tempfile.TemporaryDirectory() as name:
        folder=Path(name); records=[];trials={}
        # Deliberately synthetic neutral strings, with known first waveform
        # locations. >=5 accepted events in every group supports comparison.
        for sid,group in enumerate(('negative','neutral','positive'),1):
            sentence=f'Synthetic native sentence {sid}'
            key=hashlib.sha256(sentence.encode()).hexdigest()[:24]
            raw=rng.normal(0,2,(126,6500))
            words={}
            for wi in range(7):
                onset=500+wi*700
                words[wi]={'content':f'token{wi}','nFixations':1,'rawEEG':[raw[:,onset:onset+170].copy()]}
                records.append({'sentence_id_teco':sid,'sentence_id_zuco':sid+10,'sentence_key':key,
                                'word_index':wi,'word':f'token{wi}','sentence':sentence,
                                'lexical_class':group,'contextual_class':group,'sentence_context_class':'neutral',
                                'original_sentence_label':0,
                                'alignment_warning':'','annotation_status':'agent_reviewed_not_human_gold',
                                'is_control':False})
            # Missing first fixation must never promote a good second one.
            words[5]['rawEEG']=[None,raw[:,4000:4170].copy()];words[5]['nFixations']=2
            # Word6 is a control-labelled event and must be excluded despite its
            # clear sentiment annotation and available exact signal.
            records[-1]['is_control']=True
            trials[sid]={'sentenceId':sid,'persian_sentence':sentence,'raw_data':raw,'word':words}
        csv=folder/'private_labels.csv';pd.DataFrame(records).to_csv(csv,index=False)
        labels,lookup=load_labels(csv)
        neutral=next(r for r in labels.itertuples() if r.contextual_class=='neutral' and not r.is_control)
        assert groups_for_label(neutral._replace(sentence_context_class='positive'))[0]==('neutral','neutral_in_source_neutral_sentence')
        assert groups_for_label(neutral._replace(original_sentence_label=1))[0]==('neutral','neutral_in_neutral_sentence')
        out=folder/'readers'/'synthetic'
        source={'path':'synthetic.pkl','bytes':100,'mtime_ns':100,'sha256':'synthetic-source-hash'}
        item=extract_reader(trials,'synthetic',lookup,channels,out,'fingerprint',source)
        assert item['counts']=={'negative':5,'neutral':5,'positive':5,'neutral_in_neutral_sentence':5,
                                'neutral_in_source_neutral_sentence':5}
        metadata={k:source[k] for k in ('path','bytes','mtime_ns')}
        assert inspect_reader(out,'synthetic','fingerprint',channels,metadata) is not None
        with np.load(out/'class_means.npz') as arrays:
            assert arrays['means'].shape==(5,126,1201)
            assert np.max(np.abs(arrays['means'][:,:,:PRE].mean(axis=-1)))<1e-5
            assert np.array_equal(arrays['times_ms'],TIMES_MS)
            assert np.array_equal(arrays['means'][1],arrays['means'][3])
            assert np.array_equal(arrays['means'][1],arrays['means'][4])
        table=pd.read_csv(out/'word_onsets.csv.gz',keep_default_na=False)
        assert (table.status=='accepted').sum()==15
        assert (table.status=='missing_or_invalid_first_fixation').sum()==3
        assert (table.status=='control_excluded').sum()==3
        assert not table.event_id.duplicated().any()
        assert set(table.loc[table.status=='accepted','onset_sample_sentence_0based'].astype(int))=={500,1200,1900,2600,3300}
        # Reuse preserves all existing scientific file hashes and does not add
        # events or another copy of the same reader to a grand average.
        original={p.name:file_digest(p) for p in out.iterdir() if p.is_file()}
        assert inspect_reader(out,'synthetic','fingerprint',channels,metadata)
        assert original=={p.name:file_digest(p) for p in out.iterdir() if p.is_file()}
        assert inspect_reader(out,'synthetic','wrong',channels,metadata) is None
        assert inspect_reader(out,'synthetic','fingerprint',channels,{**metadata,'bytes':101}) is None
        rosters=summarize(folder,[item],montage)
        assert rosters['strict_neutral']['readers']==['synthetic']
        assert rosters['source_neutral']['readers']==['synthetic']
        assert (folder/'plots/strict_neutral_regions.png').is_file()
        assert (folder/'plots/source_neutral_regions.png').is_file()
        assert len(pd.read_csv(folder/'subject_window_means.csv'))==60
        assert pd.read_csv(folder/'regional_erp.csv').sem_between_readers.isna().all()
        assert save_state(folder,[item],[],['synthetic'],'fingerprint','reader_saved')['status']=='incomplete'
        assert save_state(folder,[item],[],['synthetic'],'fingerprint','finished')['status']=='completed'
        try:
            save_state(folder,[item,item],[],['synthetic'],'fingerprint','finished')
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate reader accepted')
        # Corruption is rejected even if the old manifest falsely said complete.
        with (out/'class_means.npz').open('ab') as stream:
            stream.write(b'corrupt')
        assert inspect_reader(out,'synthetic','fingerprint',channels,metadata) is None
        # A missing sentence roster must not be treated as a successful reader.
        try:
            extract_reader({1:trials[1]},'incomplete',lookup,channels,folder/'bad','fp',source)
        except ValueError as error:
            assert 'incomplete native sentence roster' in str(error)
        else:
            raise AssertionError('Incomplete source roster accepted')
    print('Native TeCo synthetic extraction, strict baseline, source resume, integrity and reader weighting tests passed')


if __name__=='__main__':
    main()
