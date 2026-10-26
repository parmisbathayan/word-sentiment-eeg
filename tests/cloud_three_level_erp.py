"""Synthetic native signals verify new40Hz selection/extraction/resume, not real results."""
import hashlib
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from word_sentiment_eeg.common import file_digest,sentence_key
from word_sentiment_eeg.three_level_erp import (CLASSES,EXPERIMENTS,load_annotations,selection,
                                               process_reader,resume,preprocess40)
from word_sentiment_eeg.erp import preprocess as old30
from run_three_level_erp import montage,outputs


def must_fail(fn,match):
    try:fn()
    except ValueError as error:assert match in str(error),error
    else:raise AssertionError('Expected failure')


def main():
    rng=np.random.default_rng(836)
    base=dict(alignment_warning='',grammatical_class='content',lexical_class='positive',
              contextual_word_class='negative',sentence_context_class='neutral')
    row=SimpleNamespace(**base)
    assert selection(row,'all_three')[0] is None
    assert [selection(row,e)[0] for e in EXPERIMENTS[1:]]==['positive','negative','neutral']
    assert selection(SimpleNamespace(**{**base,'grammatical_class':'function'}),'lexical_only')[0] is None
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        for dataset,nch,hz in [('zuco',105,500),('teco',126,1200)]:
            labels=[];sentences=[];native={}
            for si,cls in enumerate(CLASSES,1):
                tokens=[f'{cls}_{j}' for j in range(7)]
                text=' '.join(tokens)
                key=sentence_key(tokens) if dataset=='zuco' else hashlib.sha256(text.encode()).hexdigest()[:24]
                raw=rng.normal(0,2,(nch,7000)).astype(np.float32)
                words=[];native_words={}
                for j,word in enumerate(tokens):
                    onset=700+j*750;seg=raw[:,onset:onset+170].copy()
                    rawseg=seg
                    if j==5:
                        rawseg=np.empty(2,dtype=object);rawseg[0]=np.nan;rawseg[1]=seg
                    words.append(dict(content=word,rawEEG=rawseg))
                    native_words[j]=dict(content=word,nFixations=2 if j==5 else 1,rawEEG=[None,seg] if j==5 else [seg])
                    labels.append(dict(sentence_key=key,sentence_id_teco=si,word_index=j,word=word,tasks='SR',
                                       lexical_class=cls,contextual_word_class=cls,sentence_context_class=cls,
                                       grammatical_class='function' if j==6 else 'content',is_control=j==4,
                                       annotation_status='agent_reviewed_not_human_gold',alignment_warning=''))
                sentences.append(dict(content=text,word=words,rawData=raw))
                native[si]=dict(sentenceId=si,persian_sentence=text,raw_data=raw,word=native_words,
                                answer_data=np.ones((300,126))*99999)
            source=folder/f'{dataset}.mat' if dataset=='zuco' else folder/f'{dataset}.pickle'
            if dataset=='zuco':savemat(source,dict(sentenceData=sentences))
            else:
                with source.open('wb') as handle:pickle.dump(native,handle)
            if dataset=='zuco':
                labels[6]['word']='deliberate tokenization mismatch'
            labels_path=folder/f'{dataset}_labels.csv';pd.DataFrame(labels).to_csv(labels_path,index=False)
            ann=load_annotations(labels_path,dataset)
            m=montage(dataset);task='SR' if dataset=='zuco' else 'task1'
            out=folder/dataset/'readers'/task/'synthetic'
            before=file_digest(source)
            item=process_reader(source,dataset,task,'synthetic',ann,out,'fixture',m.channel.tolist(),m.region.tolist(),EXPERIMENTS,True)
            assert file_digest(source)==before
            for e in EXPERIMENTS:
                assert item['experiments'][e]['accepted']==dict.fromkeys(CLASSES,5),item
                assert item['experiments'][e]['candidates']==dict.fromkeys(CLASSES,6),item
            table=pd.read_csv(out/'word_onsets.csv.gz',keep_default_na=False)
            assert (table.all_three_status=='accepted').sum()==15
            assert (table.loc[table.is_control.astype(str).str.lower()=='true','all_three_status']=='accepted').all()
            assert (table.loc[table.word_index==5,'all_three_status']!='accepted').all()
            grammar_statuses=table.loc[table.word_index==6,'all_three_status'].tolist()
            assert grammar_statuses.count('excluded_function_or_uncertain_grammar')==(2 if dataset=='zuco' else 3)
            if dataset=='zuco':assert grammar_statuses.count('excluded_text_alignment')==1
            assert not table.event_id.duplicated().any()
            assert len(pd.read_csv(out/'event_window_features.csv.gz'))==15
            assert resume(out,'fixture',source) is not None
            saved=file_digest(out/'class_means.npz')
            assert resume(out,'fixture',source) is not None and saved==file_digest(out/'class_means.npz')
            with source.open('ab') as handle:handle.write(b'changed source')
            must_fail(lambda:resume(out,'fixture',source),'Source content changed')
            must_fail(lambda:resume(out,'different',None),'fingerprint differs')
            with (out/'class_means.npz').open('ab') as handle:handle.write(b'corrupt')
            assert resume(out,'fixture',None) is None
            outputs(folder/dataset,[item],EXPERIMENTS,m)
            assert (folder/dataset/'plots'/f'{task}_synthetic_all_three_row.png').stat().st_size>1000
            assert not (folder/dataset/'plots'/f'{task}_all_three_group.png').exists(),'One reader is not group SEM evidence'
        raw=rng.normal(0,1,(105,3000)).astype(np.float32)
        assert np.max(np.abs(preprocess40(raw,500)-old30(raw)))>.01,'40Hz reconstruction accidentally equals30Hz'
    print('Three-level40Hz native extraction, four policies, control inclusion, first-slot, resume and plots passed.')


if __name__=='__main__':main()
