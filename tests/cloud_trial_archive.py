"""All-channel 125Hz MATLAB/Persian-pickle trial export and interrupted resume."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from word_sentiment_eeg import trial_archive as ta
from word_sentiment_eeg.common import file_digest,write_json,sentence_key
from run_three_level_erp import montage


def main():
    rng=np.random.default_rng(1891)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for dataset,nch,hz in [('zuco',105,500),('teco',126,1200)]:
            old=root/dataset/'old';old.mkdir(parents=True)
            labels=[];oldrows=[];sentences=[];native={}
            for si in range(2):
                tokens=[f'word{si}_{j}' for j in range(4)];text=' '.join(tokens)
                key=sentence_key(tokens) if dataset=='zuco' else hashlib.sha256(text.encode()).hexdigest()[:24]
                raw=rng.normal(0,3,(nch,6500)).astype(np.float32)
                raw[0,800:1000]+=1000
                words=[];nativewords={}
                for j,word in enumerate(tokens):
                    onset=700+j*1000;segment=raw[:,onset:onset+90].copy()
                    rawseg=segment
                    if j==3:
                        rawseg=np.empty(2,dtype=object);rawseg[0]=np.nan;rawseg[1]=segment
                    words.append(dict(content=word,rawEEG=rawseg))
                    nativewords[j]=dict(content=word,nFixations=2 if j==3 else 1,rawEEG=[None,segment] if j==3 else [segment])
                    labels.append(dict(sentence_key=key,sentence_id_teco=si+1,word_index=j,word=word,tasks='SR',
                                       lexical_class='positive',contextual_word_class='negative' if j==2 else 'positive',
                                       sentence_context_class='neutral',grammatical_class='function' if j==1 else 'content',
                                       is_control=j==1,annotation_status='agent_reviewed_not_human_gold',alignment_warning=''))
                    oldrows.append(dict(event_id=f'{dataset}-{si}-{j}',raw_sentence_position=si,sentence_position=si,
                                        sentence_key=key,word_index=j,word=word,status='accepted' if j==0 else 'control_excluded',
                                        onset_method='exact' if j==0 else '',match_method='exact' if j==0 else '',
                                        onset_sample_sentence_0based=onset if j==0 else '',lexical_class='positive',contextual_class='positive'))
                sentences.append(dict(content=text,rawData=raw,word=words))
                native[si]=dict(sentenceId=si+1,persian_sentence=text,raw_data=raw,word=nativewords,
                                answer_data=np.full((nch,5000),99999.))
            source=root/(dataset+'.mat' if dataset=='zuco' else dataset+'.pickle')
            if dataset=='zuco':savemat(source,dict(sentenceData=sentences))
            else:
                with source.open('wb') as f:pickle.dump(native,f)
            sourcehash=file_digest(source)
            write_json(old/'manifest.json',{'source_sha256':sourcehash,'source':{'sha256':sourcehash}})
            pd.DataFrame(oldrows).to_csv(old/'word_onsets.csv.gz',index=False)
            out=root/dataset/'trials';channels=montage(dataset).channel.tolist();ann=pd.DataFrame(labels)
            oldhashes={p:file_digest(p) for p in old.iterdir()}
            # Commit one source sentence, then emulate a disconnected runtime.
            def interrupt(*args):raise RuntimeError('synthetic interruption')
            try:
                with patch.object(ta,'CHUNK',2):ta.prepare_reader(source,dataset,'SR','reader',old,ann,out,{},channels,interrupt)
            except RuntimeError as error:assert 'synthetic interruption' in str(error)
            else:raise AssertionError('Expected interrupted preparation')
            cp=json.loads((out/'checkpoint.json').read_text());assert cp['status']=='incomplete' and cp['next_sentence_position']==1
            committed={out/p:sha for b in cp['blocks'] for p,sha in b['checksums'].items()}
            with patch.object(ta,'CHUNK',2):state=ta.prepare_reader(source,dataset,'SR','reader',old,ann,out,{},channels)
            assert state['status']=='completed' and state['source_word_rows']==8 and state['epochs']==6,state
            events=pd.read_csv(out/'events.csv.gz',keep_default_na=False)
            assert not events.event_id.duplicated().any()
            assert set(events.loc[events.word_index==3,'status'])=={'missing_or_invalid_first_fixation'}
            assert events.loc[events.word_index==1,'status'].eq('archived').all(),'Control/function words silently excluded'
            assert events.loc[events.word_index==2,'status'].eq('archived').all(),'Label disagreement silently excluded'
            assert events.loc[events.word_index==0,'amplitude_over_100_uV'].astype(str).str.lower().eq('true').all()
            assert events.loc[events.word_index==0,'status'].eq('archived').all(),'High amplitude silently dropped'
            assert len(events.query("onset_method == 'cached_exact'"))==2
            for b in state['blocks']:
                ta.inspect_block(out,b,channels,ta.TIMES_MS)
                for name in b['epoch_files']:
                    with np.load(out/name) as a:assert a['eeg_uV'].shape[1:]==(nch,126)
            for p,sha in committed.items():assert file_digest(p)==sha
            for p,sha in oldhashes.items():assert file_digest(p)==sha
            assert file_digest(source)==sourcehash
            source.unlink()
            # A fully prepared reader never redownloads/reopens its raw source.
            with patch.object(ta,'source_sentences',side_effect=AssertionError('must not re-extract')):
                state=ta.prepare_reader(source,dataset,'SR','reader',old,ann,out,{},channels)
            assert state['epochs']==6
    # Native time zero need not lie on the sentence's resampled grid. Anchor
    # each resampling segment to the fixation instead; an impulse at onset is
    # exactly sample25 at either native rate, with no 4ms rounding shift.
    for hz in [500,1200]:
        onset=1703;signal=np.zeros((2,4500),np.float32);signal[0,onset]=10
        value,base,status=ta.make_epoch(signal,onset,hz)
        assert status=='archived' and value.shape==(2,126)
        assert np.argmax(value[0])==25 and ta.TIMES_MS[25]==0
        t=(np.arange(4500)-onset)/hz
        wave=np.stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*100*t)]).astype(np.float32)
        value,base,status=ta.make_epoch(wave,onset,hz)
        assert np.std(value[1,10:-10])<.02*np.std(value[0,10:-10]),'Anti-alias filter failed'
        for edge in [int(.2*hz),4500-int(.8*hz)-1]:
            v,_,status=ta.make_epoch(wave,edge,hz)
            assert status=='archived' and v.shape==(2,126) and np.isfinite(v).all()
    print('All-channel 125Hz trial archives passed: MATLAB and TeCo pickle; complete channels/time, control/function/disagreement/amplitude preservation, first slot, corruption checks and interrupted resume.')


if __name__=='__main__':main()
