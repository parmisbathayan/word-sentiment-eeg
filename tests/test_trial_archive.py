"""Archive identities, full signals, retained flags and crash-safe resumption."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from word_sentiment_eeg import trial_archive as ta
from word_sentiment_eeg.common import file_digest,write_json


class TrialArchiveTests(unittest.TestCase):
    def setUp(self):
        # Local tests do not install SciPy. The cloud integration exercises the
        # actual polyphase filter, attenuation and native-rate event alignment.
        def interpolate(value,up,down):
            x=np.arange(int(np.ceil(value.shape[1]*up/down)))*down/up
            return np.stack([np.interp(x,np.arange(value.shape[1]),r) for r in value])
        self.resampler=patch.object(ta,'_resample',side_effect=interpolate)
        self.resampler.start();self.addCleanup(self.resampler.stop)

    def test_baseline_and_amplitude_are_reversible_and_retained(self):
        raw=np.arange(6*1500,dtype=np.float32).reshape(6,1500)
        value,base,status=ta.make_epoch(raw,250,500)
        self.assertEqual(status,'archived');self.assertEqual(value.shape,(6,126))
        np.testing.assert_allclose(value+base[:,None],raw[:,150:651:4])
        self.assertGreater(np.abs(value).max(),100)
        self.assertEqual(ta.make_epoch(raw,20,500)[2],'sentence_boundary')

    def test_committed_chunks_resume_and_corruption_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);channels=['A','B'];times=np.arange(-100,401)*2.
            identity={'dataset':'test','task':'SR','subject':'r'}
            w=ta.ArchiveWriter(out,identity,channels,times)
            val=np.zeros((2,len(times)),np.float32)
            for i in range(35):
                row=dict(event_id=str(i),status='archived',epoch_file='',epoch_row='')
                w.add(row,val,np.zeros(2,np.float32))
            w.add(dict(event_id='missing',status='missing_fixation',epoch_file='',epoch_row=''))
            w.flush(10,10,True)
            before={p:file_digest(p) for p in out.rglob('*') if p.is_file() and p.name!='checkpoint.json'}
            self.assertEqual(len(w.state['blocks'][0]['epoch_files']),2)
            resumed=ta.ArchiveWriter(out,identity,channels,times)
            self.assertEqual(len(resumed.seen),36)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                resumed.add(dict(event_id='0',status='missing'))
            final=resumed.finish(10,10)
            self.assertEqual(final['epochs'],35);self.assertEqual(final['source_word_rows'],36)
            for p,sha in before.items():self.assertEqual(file_digest(p),sha)
            one=out/final['blocks'][0]['epoch_files'][0];one.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                ta.ArchiveWriter(out,identity,channels,times)

    def test_missing_cached_position_is_not_silently_accepted(self):
        raw=np.random.default_rng(8).normal(size=(3,800)).astype(np.float32)
        from types import SimpleNamespace
        old=SimpleNamespace(onset_method='exact',onset_sample_sentence_0based=70)
        segment=raw[:,100:130]
        with self.assertRaisesRegex(ValueError,'no longer matches'):ta.cached_onset(raw,segment,old)
        old.onset_sample_sentence_0based=100
        self.assertEqual(ta.cached_onset(raw,segment,old),(100,'cached_exact'))
        shifted=segment+np.array([1,2,3])[:,None];old.onset_method='constant_channel_offset'
        self.assertEqual(ta.cached_onset(raw,shifted,old),(100,'cached_constant_channel_offset'))

    def test_all_labels_controls_and_function_words_remain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.mat';source.write_bytes(b'synthetic source identity')
            old=root/'old';old.mkdir();out=root/'archive';rng=np.random.default_rng(7)
            raw=(rng.normal(size=(6,3000))*90).astype(np.float32)
            tokens=['good','the','not'];key=ta.sentence_key(tokens)
            old_rows=[]
            for i,word in enumerate(tokens):
                old_rows.append(dict(event_id=f'old{i}',raw_sentence_position=0,sentence_key=key,word_index=i,word=word,
                                     contextual_class=['positive','neutral','negative'][i],lexical_class=['positive','neutral','neutral'][i],
                                     status=['accepted','control_excluded','excluded_uncertain_or_context_disagreement'][i],
                                     onset_method='exact' if i==0 else '',onset_sample_sentence_0based=300 if i==0 else ''))
            pd.DataFrame(old_rows).to_csv(old/'word_onsets.csv.gz',index=False)
            write_json(old/'manifest.json',{'source_sha256':file_digest(source)})
            ann=pd.DataFrame([dict(sentence_key=key,word_index=i,word=word,tasks='SR',grammatical_class='function' if i else 'content',
                                   lexical_class=['positive','neutral','neutral'][i],contextual_word_class=['positive','neutral','negative'][i],
                                   sentence_context_class='mixed',is_control=True,annotation_status='agent_reviewed_not_human_gold',alignment_warning='')
                              for i,word in enumerate(tokens)])
            words=[(i,t,raw[:,300+i*500:320+i*500].copy(),'available') for i,t in enumerate(tokens)]
            generator=[(0,key,None,raw,words)]
            calls=[]
            def trace(raw,hz):calls.append(hz);return raw.copy()
            with patch.object(ta,'source_sentences',return_value=iter(generator)),patch.object(ta,'preprocess40',side_effect=trace):
                result=ta.prepare_reader(source,'zuco','SR','R',old,ann,out,{},list('ABCDEF'))
            self.assertEqual(calls,[500]);self.assertEqual(result['epochs'],3)
            events=pd.read_csv(out/'events.csv.gz')
            self.assertTrue(events.is_control.all());self.assertTrue(events.amplitude_over_100_uV.all())
            self.assertEqual(events.historical_accepted_30hz.sum(),1)
            self.assertEqual(events.iloc[0].onset_method,'cached_exact')
            self.assertEqual(set(events.grammatical_class),{'content','function'})
            # Completed resume reads no source sentences and never duplicates events.
            with patch.object(ta,'source_sentences',side_effect=AssertionError('must not re-extract')):
                resumed=ta.prepare_reader(source,'zuco','SR','R',old,ann,out,{},list('ABCDEF'))
            self.assertEqual(resumed['epochs'],3)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source hash'):
                ta.prepare_reader(source,'zuco','SR','R',old,ann,out,{},list('ABCDEF'))

    def test_storage_cap_stops_before_signal_writes(self):
        from prepare_trial_archive import StorageBudget
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);budget=StorageBudget(root,1000)
            writer=ta.ArchiveWriter(root/'reader',dict(dataset='test',task='t',subject='r'),['A','B'],ta.TIMES_MS,budget)
            writer.add(dict(event_id='new',status='archived',epoch_file='',epoch_row=''),
                       np.zeros((2,126),np.float32),np.zeros(2,np.float32))
            with self.assertRaisesRegex(OSError,'storage cap'):
                writer.flush(1,1,True)
            self.assertFalse(list(root.rglob('*.npz')))
            self.assertFalse((root/'reader/checkpoint.json').exists())

    def test_unicode_word_identity(self):
        self.assertEqual(ta.word_identity(' فیلم '),'فیلم')
        self.assertNotEqual(ta.word_identity('فیلم'),ta.word_identity('بازی'))


if __name__=='__main__':unittest.main()
