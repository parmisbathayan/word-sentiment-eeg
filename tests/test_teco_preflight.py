"""TeCo preflight preserves token identity and never invents ERP timing."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from teco_preflight import trial_words,preflight
from word_sentiment_eeg.teco import first_fixation,find_raw_files,sample_onsets,sentence_labels


class TeCoPreflightTests(unittest.TestCase):
    def test_explicit_tokens_and_unresolved_alignment(self):
        trial={'persian_sentence':'A B','word':{2:{'content':'different'},1:{'content':'token'}}}
        words,method=trial_words(trial)
        self.assertEqual(words,[('1','token'),('2','different')])
        self.assertEqual(method,'explicit_word_field:content')
        words,method=trial_words({'persian_sentence':'A B','word':{0:{},1:{},2:{}}})
        self.assertEqual(words,[])
        self.assertTrue(method.startswith('unresolved_token_alignment'))

    def test_preflight_retains_reader_token_versions_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);feature=root/'Data/TRT_Total';feature.mkdir(parents=True)
            pd.DataFrame([{'sentence_id':301,'sentence':'same sentence','sentiment_label':0,'control':''}]).to_csv(
                root/'Data/teco_sentiment_labels_task1.csv',index=False)
            rawdir=root/'Data/teco_pickles/reader';rawdir.mkdir(parents=True)
            for reader,word in [('A','one'),('B','other')]:
                trial={'sentenceId':7,'persian_sentence':'same sentence',
                       'word':{1:{'content':word,'TRT_total':np.ones(126)}}}
                (feature/f'{reader}_trt_total.pickle').write_bytes(pickle.dumps({0:trial}))
            raw=rawdir/'task1.pickle'
            raw.write_bytes(pickle.dumps({0:{'word':{1:{'rawEEG':np.zeros((126,90))}},'rawData':np.zeros((126,900))}}))
            original=raw.read_bytes()
            report=preflight(root,inspect_raw=True)
            self.assertEqual(raw.read_bytes(),original)
            self.assertEqual(report['feature_reader_files'],2)
            self.assertEqual(report['sentence_token_versions'],2)
            self.assertEqual(report['raw_inspection'],'sample_checked_not_full_reader_or_erp_audit')
            result=root/'Results/word_sentiment_eeg/teco_preflight'
            table=pd.read_csv(result/'word_inventory.csv')
            self.assertEqual(set(table.word),{'one','other'})
            self.assertEqual(set(table.sentence_id_teco),{7})
            self.assertEqual(set(table.sentence_id_zuco),{301})
            self.assertTrue((table.annotation_status=='pending_Persian_agent_review').all())
            saved=json.loads((result/'structure.json').read_text())
            self.assertEqual(saved['status'],'preflight_only_not_erp')
            self.assertFalse(list(result.glob('*.npz')))

    def test_native_first_onset_and_separate_sentence_id_domains(self):
        rng=np.random.default_rng(19)
        raw=rng.normal(size=(126,1500)).astype(np.float32)
        words={0:{'nFixations':2,'rawEEG':[raw[:,350:430],raw[:,900:980]]},
               1:{'nFixations':2,'rawEEG':[None,raw[:,600:680]]}}
        data={42:{'sentenceId':7,'persian_sentence':'some text','raw_data':raw,'word':words},
              99:{'sentenceId':9,'persian_sentence':'answer only','answer_data':raw.T,
                  'word':{0:{'nFixations':1,'rawEEG':[raw[:,300:380]]}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.csv'
            pd.DataFrame([{'sentence_id':301,'sentence':'some text','sentiment_label':1,'control':''},
                          {'sentence_id':101,'sentence':'answer only','sentiment_label':0,'control':'CONTROL'}]).to_csv(path,index=False)
            coverage,checks=sample_onsets(data,sentence_labels(path))
        self.assertEqual(checks[0]['status'],'exact')
        self.assertEqual(checks[0]['onset_sample_reading_0based'],350)
        self.assertEqual(checks[0]['sentence_id_teco'],7)
        self.assertEqual(checks[0]['sentence_id_zuco'],301)
        self.assertEqual(checks[1]['status'],'missing_or_invalid_first_fixation')
        self.assertIsNone(checks[1]['onset_sample_reading_0based'])
        self.assertEqual(checks[2]['status'],'missing_reading_raw_data')
        self.assertEqual(coverage['control_trials'],1)
        self.assertEqual(coverage['controls_with_reading_arrays'],0)
        self.assertIsNone(first_fixation({'nFixations':None,'rawEEG':raw})[0])

    def test_known_layouts_and_reader_aliases_do_not_duplicate_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('teco_pickles/armin_with_total_task1/task1.pickle',
                         'pickle_files_acc5/armin_with_total_task1/task1.pickle',
                         'TeCo/task1_with_total/razie/task1.pickle'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            selected,alternates=find_raw_files(root)
            self.assertEqual(set(selected),{'armin','razie'})
            self.assertEqual(len(alternates),1)


if __name__=='__main__':
    unittest.main()
