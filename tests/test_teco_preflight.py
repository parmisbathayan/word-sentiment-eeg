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
            self.assertEqual(report['raw_inspection'],'sample_inspected_onsets_not_yet_verified')
            result=root/'Results/word_sentiment_eeg/teco_preflight'
            table=pd.read_csv(result/'word_inventory.csv')
            self.assertEqual(set(table.word),{'one','other'})
            self.assertTrue((table.annotation_status=='pending_Persian_agent_review').all())
            saved=json.loads((result/'structure.json').read_text())
            self.assertEqual(saved['status'],'preflight_only_not_erp')
            self.assertFalse(list(result.glob('*.npz')))


if __name__=='__main__':
    unittest.main()
