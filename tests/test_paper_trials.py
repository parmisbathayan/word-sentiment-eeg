"""Matching attrition and pseudo-trial membership, using small synthetic arrays."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.paper_trials import matched_trials,average_trials


class PaperTrialsTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for reader in ['A','B']:
            for sentiment,n in [('negative',7),('positive',11)]:
                for i in range(n):
                    rows.append(dict(event_id=f'{reader}/{sentiment}/{i}',subject=reader,sentiment=sentiment,
                                     sentence_key=f'sentence{i%5}',length_stratum=i%2))
        meta=pd.DataFrame(rows)
        return meta,np.arange(len(meta)*3,dtype=np.float32).reshape(-1,3)

    def test_matching_balances_within_every_stratum_and_reconciles(self):
        meta,_=self.fixture()
        ix,audit,summary=matched_trials(meta,['subject','length_stratum'])
        counts=meta.iloc[ix].groupby(['subject','length_stratum','sentiment']).size().unstack()
        np.testing.assert_array_equal(counts.negative,counts.positive)
        self.assertEqual(len(audit),len(meta))
        self.assertEqual(int(summary.retained_trials.sum()),len(ix))
        self.assertEqual(int(summary.excluded_by_matching.sum()),len(meta)-len(ix))
        np.testing.assert_array_equal(ix,matched_trials(meta,['subject','length_stratum'])[0])

    def test_averages_never_share_events_and_preserve_reader_class(self):
        meta,X=self.fixture()
        for size in [1,10,20,40]:
            means,records,members=average_trials(X,meta,size,['subject','sentiment','length_stratum'])
            self.assertEqual(set(members.event_id),set(meta.event_id))
            self.assertFalse(members.event_id.duplicated().any())
            for i,record in records.iterrows():
                group=members[members.average_id.eq(record.average_id)]
                self.assertEqual(group.subject.nunique(),1)
                self.assertEqual(group.sentiment.nunique(),1)
                self.assertEqual(group.sentence_key.nunique(),len(group))
                self.assertEqual(record.actual_trials,len(group))
                self.assertLessEqual(len(group),size)
                ids=meta.index[meta.event_id.isin(group.event_id)]
                np.testing.assert_allclose(means[i],X[ids].mean(axis=0))

    def test_unknown_strata_and_duplicate_input_fail(self):
        meta,X=self.fixture()
        with self.assertRaises(ValueError):matched_trials(meta,['length_stratum'])
        meta.loc[0,'length_stratum']=np.nan
        with self.assertRaises(ValueError):average_trials(X,meta,10,['subject','sentiment','length_stratum'])
        meta,X=self.fixture();meta.loc[0,'event_id']=meta.loc[1,'event_id']
        with self.assertRaises(ValueError):matched_trials(meta,['subject','length_stratum'])


if __name__=='__main__':unittest.main()
