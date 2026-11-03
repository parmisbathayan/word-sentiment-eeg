from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.paper_fixations import word_regions,map_word,blink_predecessors,preceding_negation,selection_outcomes


class FixationTests(unittest.TestCase):
    def test_spatial_match_reports_shared_boundaries_as_ambiguous(self):
        boxes=word_regions([[10,10,20,20],[30,10,40,20],[10,40,20,50]],'horizontal_midpoints')
        self.assertEqual(map_word(24,15,boxes),(0,'matched'))
        self.assertEqual(map_word(25,15,boxes),(None,'ambiguous_word_bounds'))
        self.assertEqual(map_word(100,15,boxes),(None,'out_of_word_bounds'))
        self.assertEqual(map_word(15,45,boxes),(2,'matched'))

    def test_blink_between_fixations_and_negation_window(self):
        flags=blink_predecessors([1,2,3],[200,200,200],[1.5],[100])
        np.testing.assert_array_equal(flags,[False,True,False])
        self.assertTrue(preceding_negation(['not','a','very','good','film'],3))
        self.assertFalse(preceding_negation(['not','a','very','good','film'],4))

    def test_sequential_outcomes_keep_neutral_only_in_original_analyses(self):
        row=dict(mapping_status='matched',duration_ms=200,blink_predecessor=False,
                 word_label_status='matched',sentiment='positive',preceded_by_negation=False)
        events=pd.DataFrame([row,dict(row,sentiment='neutral'),dict(row,duration_ms=50,blink_predecessor=True),
                             dict(row,mapping_status='out_of_word_bounds',duration_ms=50),dict(row,blink_predecessor=None)])
        self.assertEqual(selection_outcomes(events).tolist(),['eligible_before_matching',
            'neutral_excluded_in_paper_binary_route','blink_predecessor','unavailable_word_mapping','unknown_blink_predecessor'])


if __name__=='__main__':unittest.main()
