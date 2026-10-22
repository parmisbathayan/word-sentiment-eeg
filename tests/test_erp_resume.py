import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import run_erp
from word_sentiment_eeg.common import atomic_npz, file_digest, sentence_key, write_json
from word_sentiment_eeg.erp import CLASSES, TIMES_MS, epoch
from word_sentiment_eeg import erp_checkpoint as cp


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.target=Path(self.tmp.name)/'SR/ZAB';self.target.mkdir(parents=True)
        self.channels=[f'C{i}' for i in range(105)]
        self.words=[f'word{i}' for i in range(9)]
        self.raw=np.random.default_rng(81).normal(size=(105,2000)).astype(np.float32)
        records=[];self.values={c:[] for c in CLASSES};self.ids={c:[] for c in CLASSES}
        for j,word in enumerate(self.words):
            cls=CLASSES[j%3];onset=300+j*100
            event=f'{j:024d}'
            value,status=epoch(self.raw,onset);assert status=='accepted'
            self.values[cls].append(value);self.ids[cls].append(event)
            records.append({'event_id':event,'status':'accepted','contextual_class':cls,
                'raw_sentence_position':0,'word_index':j,'word':word,'sentence_key':sentence_key(self.words),
                'onset_sample_sentence_0based':onset,'epoch_index_in_class_file':j//3})
        for cls in CLASSES:
            atomic_npz(self.target/f'{cls}_epochs.npz',eeg_uV=np.stack(self.values[cls]),
                event_id=np.array(self.ids[cls]),channel_names=np.array(self.channels),times_ms=TIMES_MS,sfreq_hz=np.array(500))
        atomic_npz(self.target/'class_means.npz',mean_uV=np.stack([np.stack(self.values[c]).mean(0) for c in CLASSES]),
            counts=np.array([3,3,3]),classes=np.array(CLASSES),times_ms=TIMES_MS,channel_names=np.array(self.channels))
        pd.DataFrame(records).to_csv(self.target/'word_onsets.csv.gz',index=False)
        self.item={'fingerprint':'old-compatible','task':'SR','subject':'ZAB','output_dir':str(self.target),
            'source_sha256':'verified-source','word_rows':9,'epochs_per_class':{c:3 for c in CLASSES},'status':'completed'}
        write_json(self.target/'manifest.json',self.item)
        self.record={'task':'SR','subject':'ZAB'}

    def tearDown(self):
        self.tmp.cleanup()

    def inspect(self):
        return cp.inspect_checkpoint(self.target,self.record,{'old-compatible'},self.channels)

    def source(self,*args):
        return iter([(0,self.words,self.raw,[])])

    def test_complete_legacy_is_kept(self):
        saved=self.inspect()
        self.assertTrue(saved['item']['analysis_ready'])
        self.assertTrue(saved['item']['trial_exports_complete'])

    def test_missing_neutral_does_not_redo_onsets_or_positive_negative(self):
        keep=[self.target/n for n in ['class_means.npz','word_onsets.csv.gz','positive_epochs.npz','negative_epochs.npz']]
        before={p:file_digest(p) for p in keep}
        (self.target/'neutral_epochs.npz').unlink()
        saved=self.inspect()
        self.assertTrue(saved['item']['analysis_ready'])
        self.assertEqual(saved['item']['missing_trial_classes'],['neutral'])
        with patch.object(cp,'iter_sentences',self.source),patch.object(cp,'preprocess',lambda raw:raw),patch.object(run_erp,'locate',side_effect=AssertionError('onsets must not be recomputed')):
            cp.repair_trial_exports(Path('fixture.mat'),saved,self.channels,'verified-source')
        for p in keep:
            self.assertEqual(file_digest(p),before[p])
        self.assertTrue(self.inspect()['item']['trial_exports_complete'])
        np.testing.assert_array_equal(cp.load_class_epochs(self.target,'neutral'),np.stack(self.values['neutral']))

    def test_interrupted_chunk_repair_reuses_completed_chunks(self):
        (self.target/'neutral_epochs.npz').unlink()
        original=cp.save_chunk;calls=[]
        def interrupted(path,*args):
            calls.append(path)
            if len(calls)==2:
                raise OSError('simulated disconnect')
            return original(path,*args)
        with patch.object(cp,'CHUNK_TRIALS',2),patch.object(cp,'iter_sentences',self.source),patch.object(cp,'preprocess',lambda raw:raw):
            saved=self.inspect()
            with patch.object(cp,'save_chunk',interrupted),self.assertRaises(OSError):
                cp.repair_trial_exports(Path('fixture.mat'),saved,self.channels,'verified-source')
            first=calls[0];stamp=first.stat().st_mtime_ns
            resumed=self.inspect()
            self.assertEqual(resumed['item']['missing_trial_chunks'],1)
            cp.repair_trial_exports(Path('fixture.mat'),resumed,self.channels,'verified-source')
            self.assertEqual(first.stat().st_mtime_ns,stamp)
            self.assertTrue(self.inspect()['item']['trial_exports_complete'])

    def test_changed_source_or_settings_rejected(self):
        (self.target/'neutral_epochs.npz').unlink()
        with self.assertRaises(ValueError):
            cp.repair_trial_exports(Path('fixture.mat'),self.inspect(),self.channels,'changed-source')
        with self.assertRaises(ValueError):
            cp.inspect_checkpoint(self.target,self.record,{'different-settings'},self.channels)

    def test_missing_analysis_is_recoverable_only_after_identity_checks(self):
        (self.target/'word_onsets.csv.gz').unlink()
        with self.assertRaises(cp.MissingAnalysisArtifact) as caught:
            self.inspect()
        self.assertEqual(caught.exception.item['source_sha256'],'verified-source')
        with self.assertRaises(ValueError):
            cp.inspect_checkpoint(self.target,self.record,{'different-settings'},self.channels)
        with self.assertRaises(ValueError):
            cp.inspect_checkpoint(self.target,{'task':'SR','subject':'ZDM'},{'old-compatible'},self.channels)

    def test_duplicate_reader_cannot_enter_summary(self):
        with self.assertRaisesRegex(ValueError,'Duplicate reader/task'):
            run_erp.save_run_state(Path(self.tmp.name)/'run',[self.item,self.item],
                                   [self.record],[],'compatible','audit')

    def test_full_drive_stops_before_extraction(self):
        usage=shutil.disk_usage(self.tmp.name)
        with patch.object(run_erp.shutil,'disk_usage',return_value=type(usage)(15*1024**3,15*1024**3,0)):
            with self.assertRaisesRegex(OSError,'Drive uploads may be pending'):
                run_erp.check_output_space(self.tmp.name)
        with patch.object(run_erp.shutil,'disk_usage',return_value=type(usage)(50*1024**3,10*1024**3,40*1024**3)):
            self.assertEqual(run_erp.check_output_space(self.tmp.name).free,40*1024**3)

    def test_global_count_includes_all_saved_readers_before_work(self):
        items=[{**self.item,'subject':f'Z{i}','trial_exports_complete':i==0} for i in range(5)]
        expected=[{'task':'SR','subject':f'Z{i}'} for i in range(6)]
        run=Path(self.tmp.name)/'run'
        state=run_erp.save_run_state(run,items,expected,[],'compatible','resume_audit_complete')
        self.assertEqual(state['processed_files'],5)
        self.assertEqual(state['complete_trial_export_files'],1)
        self.assertEqual(state['pending_new_readers'],['SR/Z5'])
        self.assertEqual(len(state['pending_trial_repairs']),4)

    def test_current_code_recognizes_real_legacy_fingerprint(self):
        current,legacy=run_erp.cache_fingerprints(ROOT/'annotations/word_labels.csv',ROOT/'configs/zuco1_sources.json')
        self.assertEqual(legacy,'a49719ea0d6626bc3c10bd73ea6f7c875b076c1ec4882129c1f8c2ed8f6cfced')
        self.assertNotEqual(current,legacy)


if __name__=='__main__':
    unittest.main()
