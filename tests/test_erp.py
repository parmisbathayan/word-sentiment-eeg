import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.erp import eligible, locate, epoch, _v5_fixations


class ERPTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.random.default_rng(27).normal(size=(105,1500)).astype(np.float32)

    def test_onset_and_constant_offset(self):
        start,method=locate(self.raw,self.raw[:,321:402])
        self.assertEqual((start,method),(321,'exact'))
        offset=np.arange(105,dtype=np.float32)[:,None]/20
        self.assertEqual(locate(self.raw,self.raw[:,321:402]+offset),(321,'constant_channel_offset'))

    def test_duplicate_match_is_rejected(self):
        self.raw[:,700:781]=self.raw[:,321:402]
        self.assertEqual(locate(self.raw,self.raw[:,321:402]),(None,'nonunique_match'))

    def test_first_rejected_fixation_is_preserved(self):
        cells=np.empty(2,dtype=object)
        cells[0]=np.nan
        cells[1]=self.raw[:,300:400]
        parsed=_v5_fixations(cells)
        self.assertIsNone(parsed[0])
        self.assertEqual(parsed[1].shape,(105,100))

    def test_epoch_baseline_and_boundaries(self):
        value,status=epoch(self.raw,500)
        self.assertEqual(status,'accepted')
        self.assertEqual(value.shape,(105,501))
        np.testing.assert_allclose(value[:,:100].mean(axis=1),0,atol=1e-6)
        self.assertEqual(epoch(self.raw,99)[1],'sentence_boundary')
        self.raw[5,600]=200
        self.assertEqual(epoch(self.raw,500)[1],'amplitude_over_100_uV')

    def test_conservative_label_policy(self):
        row=SimpleNamespace(lexical_class='positive',contextual_class='positive',alignment_warning='',annotation_status='agent_reviewed_not_human_gold')
        self.assertTrue(eligible(row))
        row.contextual_class='negative'
        self.assertFalse(eligible(row))
        row.contextual_class='uncertain'
        self.assertFalse(eligible(row))
        row.contextual_class='positive';row.alignment_warning='token mismatch'
        self.assertFalse(eligible(row))


if __name__=='__main__':
    unittest.main()
