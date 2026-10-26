"""Small tests run with bundled NumPy/pandas; sklearn fitting is cloud-only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_trial_archive as a


def rows():
    return pd.DataFrame(dict(event_id=['a','b','c','d','e'],lexical_class=['positive','positive','neutral','negative','positive'],
                             contextual_word_class=['positive','negative','neutral','negative','positive'],
                             sentence_context_class=['negative','negative','neutral','negative','positive'],
                             grammatical_class=['content','content','function','content','content'],
                             amplitude_over_100_uV=['False','False','False','True','False'],annotation_alignment_warning=['','','','','token mismatch']))


class TrialAnalysisTests(unittest.TestCase):
    def test_explicit_label_populations(self):
        m=rows()
        ix,y,_=a.selected(m,'word_agreement','three_class');self.assertEqual(ix.tolist(),[0,2]);self.assertEqual(y.tolist(),['positive','neutral'])
        ix,y,_=a.selected(m,'all_three_content','three_class');self.assertEqual(len(ix),0)
        self.assertEqual(a.selected(m,'lexical_content','binary')[0].tolist(),[0,1])
        self.assertEqual(a.selected(m,'contextual_content','binary')[1].tolist(),['positive','negative'])
        self.assertEqual(a.selected(m,'sentence_content','binary')[1].tolist(),['negative','negative'])

    def test_window_features_have_no_label_information(self):
        times=np.arange(-25,101)*8
        x=np.stack([np.repeat(times[None],6,axis=0),np.repeat((times+2)[None],6,axis=0)]).astype(np.float32)
        post,pre=a.feature_blocks(x,times,a.REGIONS)
        self.assertEqual(post.shape,(2,48));self.assertEqual(pre.shape,(2,12))
        np.testing.assert_allclose(post[1]-post[0],2)
        self.assertEqual(post[0,0],times[(times>=0)&(times<100)].mean())
        self.assertEqual(pre[0,0],times[(times>=-200)&(times<-100)].mean())

    def test_word_fold_overlap_purge(self):
        m=pd.DataFrame(dict(subject=['r']*13,dataset=['zuco']*13,sentence_position=[0]*13,
                            onset_sample_sentence_0based=[0,200,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,210]))
        y=np.array(['negative','positive']*6+['negative'])
        # Only the first two training events overlap the test epoch.
        result,n=a.purge_overlapping_train(m,y,[(np.arange(12),np.array([12]))])
        self.assertEqual(n,[2]);self.assertEqual(result[0][0].tolist(),list(range(2,12)))

    def test_unicode_label_join_preserves_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'archive';folder=archive/'annotations';folder.mkdir(parents=True)
            meta=pd.DataFrame(dict(event_id=['b','a'],sentence_id_teco=[2,1],sentence_key=['s2','s1'],word_index=[0,0],word=['فیلم','جذاب'],
                                   lexical_class=['unknown']*2,contextual_word_class=['unknown']*2,sentence_context_class=['unknown']*2,
                                   grammatical_class=['unknown']*2,annotation_alignment_warning=['']*2,is_control=['unknown']*2))
            labels=meta.iloc[::-1].copy().drop(columns=['event_id','annotation_alignment_warning'])
            labels['lexical_class']=['positive','neutral'];labels['contextual_word_class']=['negative','neutral'];labels['alignment_warning']=''
            labels.to_csv(folder/'teco_word_labels.csv',index=False)
            joined,source=a.attach_available_labels(meta,archive,root,'teco','task1')
            self.assertEqual(joined.event_id.tolist(),['b','a']);self.assertEqual(joined.lexical_class.tolist(),['neutral','positive'])
            self.assertTrue(source['sha256'])

    def test_similarity_excludes_self_same_sentence_and_cross_reader(self):
        rng=np.random.default_rng(9);m=[];ys=[]
        for reader in ['A','B']:
            for i in range(18):
                cls=a.CLASSES[i%3];m.append(dict(subject=reader,sentence_key=f's{i//3}',event_id=reader+str(i),word_identity=f'word{i%6}',word=f'w{i%6}',word_length=5));ys.append(cls)
        m=pd.DataFrame(m);pairs,repeats=a.similarity_pairs(m,rng.normal(size=(len(m),48)),np.array(ys),max_pairs=5)
        byid=m.set_index('event_id')
        self.assertGreater(len(pairs),0);self.assertGreater(len(repeats),0)
        for row in pairs.itertuples():
            x=byid.loc[row.event_a];y=byid.loc[row.event_b]
            self.assertEqual(x.subject,y.subject);self.assertNotEqual(x.sentence_key,y.sentence_key);self.assertNotEqual(row.event_a,row.event_b)
        for row in repeats.itertuples():
            x=byid.loc[row.event_a];y=byid.loc[row.event_b];z=byid.loc[row.control_event]
            self.assertEqual(x.word_identity,y.word_identity);self.assertNotEqual(x.word_identity,z.word_identity)
            self.assertEqual(x.subject,z.subject);self.assertNotEqual(x.sentence_key,z.sentence_key)

    def test_no_raw_extraction_imports(self):
        import ast
        tree=ast.parse((ROOT/'scripts/analyze_trial_archive.py').read_text())
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('run_erp','word_sentiment_eeg.erp','word_sentiment_eeg.trial_archive')) for n in names))


if __name__=='__main__':unittest.main()
