import io
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from word_sentiment_eeg.paper_labels import SST,canonical,five_class,tree_spans


def fixture():
    # not good . : positive target word, negative sentence/containing phrase.
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w') as z:
        z.writestr('stanfordSentimentTreebank/dictionary.txt','not|0\ngood|1\n.|2\nnot good|3\nnot good .|4\n')
        z.writestr('stanfordSentimentTreebank/sentiment_labels.txt','phrase ids|sentiment values\n0|0.5\n1|0.8\n2|0.5\n3|0.1\n4|0.1\n')
        z.writestr('stanfordSentimentTreebank/SOStr.txt','not|good|.\n')
        z.writestr('stanfordSentimentTreebank/STree.txt','4|4|5|5|0\n')
    target.seek(0);return target


class LabelsTests(unittest.TestCase):
    def test_word_and_sentence_labels_are_not_conflated(self):
        joined=SST(fixture()).join(['not','good.'])
        self.assertEqual(joined['status'],'matched')
        self.assertEqual(joined['sentence_label']['sentiment'],'negative')
        self.assertEqual(joined['words'][1]['word_label']['sentiment'],'positive')
        self.assertEqual(joined['words'][1]['target_span_label'],None)
        self.assertEqual(joined['words'][1]['smallest_containing_phrase_label']['sentiment'],'negative')

    def test_no_fuzzy_match_or_invented_label(self):
        joined=SST(fixture()).join(['not','goood.'])
        self.assertEqual(joined['status'],'sentence_not_found')
        joined=SST(fixture()).join(['notgo','od.'])
        self.assertEqual(joined['words'][0]['status'],'source_word_splits_sst_token')

    def test_boundaries_and_tree_validation(self):
        self.assertEqual([five_class(s) for s in [0,.2,.4,.6,.8,1]],[-2,-1,0,1,2,2])
        self.assertEqual(canonical('“ -LRB- fine -RRB- ”'),'"(fine)"')
        with self.assertRaises(ValueError):tree_spans([1,0,-1],2)


if __name__=='__main__':unittest.main()
