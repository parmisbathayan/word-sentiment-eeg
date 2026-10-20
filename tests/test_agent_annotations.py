import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from word_sentiment_eeg.agent_annotations import attach_agent_annotations
from word_sentiment_eeg.common import sentence_key
from word_sentiment_eeg.evaluate import run_probe


class PreparedAnnotationTests(unittest.TestCase):
    def test_complete_unique_dataset_coverage(self):
        annotations = pd.read_csv(ROOT / 'annotations/word_labels.csv', keep_default_na=False)
        source = json.loads((ROOT / 'annotations/source_sentences.json').read_text())
        self.assertEqual(len(source), 1037)
        self.assertEqual(len(annotations), 20458)
        self.assertFalse(annotations.duplicated(['sentence_key','word_index']).any())
        for sentence in source:
            part = annotations[annotations.sentence_ordinal == sentence['ordinal']].sort_values('word_index')
            self.assertEqual(part.word.tolist(), sentence['words'])
            self.assertTrue(part.contextual_class.isin(['positive','negative','neutral','uncertain']).all())

    def test_scope_and_named_entity_examples(self):
        data = pd.read_csv(ROOT / 'annotations/word_labels.csv', keep_default_na=False)
        def item(ordinal, form):
            return data[(data.sentence_ordinal == ordinal) & (data.word_form == form)].iloc[0]
        # Negated positive evaluation: label layers differ, not automatically neutral.
        self.assertEqual((item(287, 'good').lexical_class, item(287, 'good').contextual_class), ('positive','negative'))
        # Not only does not reverse good; nothing if not affirms sincerity.
        self.assertEqual(item(55, 'good').contextual_class, 'positive')
        self.assertEqual(item(57, 'sincere').contextual_class, 'positive')
        # Proper name Love is not a positive assertion of love.
        self.assertEqual((item(543, 'love').lexical_class, item(543, 'love').contextual_class), ('positive','neutral'))
        # Unclear sarcasm is retained as uncertainty rather than forced polarity.
        self.assertEqual(item(270, 'rewarded').contextual_class, 'uncertain')

    def test_exact_cache_join_and_no_neutral_fallback(self):
        data = pd.read_csv(ROOT / 'annotations/word_labels.csv', keep_default_na=False)
        first = data[data.sentence_ordinal == 0]
        table = pd.DataFrame({'sentence_key': first.sentence_key, 'word_index': first.word_index,
                              'word': first.word, 'task': 'SR', 'sentence_id_original': first.sentence_id_original})
        actual, report = attach_agent_annotations(table, ROOT / 'annotations/word_labels.csv')
        self.assertEqual(actual.contextual_class.tolist(), first.contextual_class.tolist())
        broken = table.copy()
        broken.loc[broken.index[0], 'word'] = 'not_the_original_word'
        with self.assertRaises(ValueError):
            attach_agent_annotations(broken, ROOT / 'annotations/word_labels.csv')

    def test_known_alignment_issue_explicitly_excluded(self):
        table = pd.DataFrame([{'sentence_key':'different-tokenization', 'word_index':0, 'word':'Stella',
                               'task':'TSR', 'sentence_id_original':113637237}])
        actual, report = attach_agent_annotations(table, ROOT / 'annotations/word_labels.csv')
        self.assertEqual(actual.lexical_class.iloc[0], 'unknown')
        self.assertEqual(actual.contextual_class.iloc[0], 'unresolved_alignment')
        self.assertEqual(report['alignment_excluded_eeg_rows'], 1)


if __name__ == '__main__':
    unittest.main()
