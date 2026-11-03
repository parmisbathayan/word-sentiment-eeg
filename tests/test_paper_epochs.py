"""Shared raw/clean rejection, source-block boundaries and feature grid tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.paper_epochs import paired_epochs,summarize_paired_epochs,decoding_features


class PairedEpochTests(unittest.TestCase):
    def fixture(self):
        raw=np.zeros((105,1000));clean=np.ones_like(raw)
        events=pd.DataFrame([dict(event_id=str(i),subject='A',block_id='1',sentiment='positive',onset_seconds=t)
                             for i,t in enumerate([.1,2.,4.,6.])])
        raw[0,250]=91 # raw rejects event at 2 s, even though cleaned is fine
        clean[1,500]=np.nan # clean data invalidates both representations at 4 s
        return raw,clean,events

    def test_outcomes_and_shared_ids_reconcile(self):
        raw,clean,events=self.fixture()
        means,audit=summarize_paired_epochs(raw,clean,events,units='uV',baseline=None)
        self.assertEqual(audit.outcome.tolist(),['epoch_outside_block','raw_amplitude_exceeds_threshold',
                                               'nonfinite_paired_epoch','accepted'])
        self.assertEqual(means['positive']['n'],1)
        np.testing.assert_array_equal(means['positive']['raw'],0)
        np.testing.assert_array_equal(means['positive']['cleaned'],1)
        self.assertEqual(means['positive']['raw'].shape,(105,201))
        features=decoding_features(means['positive']['cleaned'])
        self.assertEqual(features.shape,(6615,))

    def test_units_baseline_and_threshold_order(self):
        raw=np.full((1,500),91e-6);clean=raw.copy()
        events=self.fixture()[2].iloc[[1]].copy()
        outcome=list(paired_epochs(raw,clean,events,units='V',baseline=(-.6,0)))[0]
        self.assertEqual(outcome[0]['outcome'],'raw_amplitude_exceeds_threshold')
        raw[:]=90e-6;clean[:]=10e-6
        _,r,c=list(paired_epochs(raw,clean,events,units='V',baseline=(-.6,0)))[0]
        np.testing.assert_allclose(r,0,atol=1e-12);np.testing.assert_allclose(c,0,atol=1e-12)

    def test_unknown_units_duplicate_and_mixed_blocks_fail(self):
        raw,clean,events=self.fixture()
        with self.assertRaises(ValueError):list(paired_epochs(raw,clean,events,units='unknown',baseline=None))
        events.loc[1,'block_id']='2'
        with self.assertRaises(ValueError):list(paired_epochs(raw,clean,events,units='uV',baseline=None))
        events.loc[1,'block_id']='1';events.loc[1,'event_id']='0'
        with self.assertRaises(ValueError):list(paired_epochs(raw,clean,events,units='uV',baseline=None))


if __name__=='__main__':unittest.main()
