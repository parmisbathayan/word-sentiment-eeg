"""Offline tests with synthetic signals. No downloads, environments or training data."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from word_sentiment_eeg.common import atomic_npz, normalize_word, sentence_key
from word_sentiment_eeg.data import load_sources, validate_cache
from word_sentiment_eeg.evaluate import make_splits, metrics, run_probe, transform_fit
from word_sentiment_eeg.labels import annotate, apply_reviews, class_from_valence, load_lexicon
from word_sentiment_eeg.raw_export import orient, v5_cells
from word_sentiment_eeg.report import class_exports

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/default.json").read_text())


def synthetic_table(n_sentences=30, readers=3):
    rows, x = [], []
    rng = np.random.default_rng(120)
    for sentence in range(n_sentences):
        for subject in range(readers):
            for j, label in enumerate(["negative", "neutral", "positive"]):
                rows.append({"row_id": len(rows), "sentence_key": f"s{sentence}",
                    "subject": f"reader{subject}", "word_form": f"word{sentence}_{j}",
                    "word": f"word{sentence}_{j}", "word_index": j,
                    "item_id": f"s{sentence}w{j}", "lexical_class": label,
                    "relative_position": j / 2, "sentence_length": 3,
                    "n_fixations": 1, "trt_ms": 200, "ffd_ms": 200, "gd_ms": 200})
                # A known positive control with the same nuisance values in every class.
                vector = rng.normal(0, 0.1, 12)
                vector[j] += 5
                x.append(vector)
    return pd.DataFrame(rows), np.array(x)


class LabelsTests(unittest.TestCase):
    def test_unknown_is_not_neutral_and_boundaries(self):
        t = CONFIG["label_thresholds"]
        actual = [class_from_valence(v, t) for v in [np.nan, .1, .35, .40, .45, .55, .60, .65, .9]]
        self.assertEqual(actual, ["unknown", "negative", "negative", "ambiguous", "neutral", "neutral", "ambiguous", "positive", "positive"])

    def test_word_normalization_preserves_internal_boundaries(self):
        self.assertEqual(normalize_word('“Wonderful!”'), "wonderful")
        self.assertEqual(normalize_word("isn’t"), "isn't")
        self.assertNotEqual(normalize_word("good-bad"), normalize_word("goodbad"))

    def test_context_never_overwrites_lexical_label(self):
        frame = pd.DataFrame({"sentence_key": ["s", "s"], "word_index": [0, 1], "word": ["not", "wonderful"]})
        labeled = annotate(frame, {"wonderful": {"valence": .9, "arousal": .7, "dominance": .7}}, CONFIG["label_thresholds"])
        self.assertEqual(labeled.lexical_class.tolist(), ["unknown", "positive"])
        self.assertEqual(labeled.contextual_class.tolist(), ["unreviewed", "unreviewed"])
        self.assertIn("nearby_negation", labeled.context_flags.iloc[1])

    def test_reviews_separate_and_traceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reviews.csv"
            pd.DataFrame([{"item_id": "a", "contextual_class": "negative", "reviewer": "professor", "notes": "negated"}]).to_csv(path, index=False)
            table = pd.DataFrame({"item_id": ["a", "a"], "lexical_class": ["positive"] * 2, "contextual_class": ["unreviewed"] * 2})
            actual = apply_reviews(table, path)
            self.assertEqual(actual.lexical_class.tolist(), ["positive"] * 2)
            self.assertEqual(actual.contextual_class.tolist(), ["negative"] * 2)

    def test_wrong_lexicon_scale_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lexicon.tsv"
            path.write_text("Word\tValence\tArousal\tDominance\nbad\t-0.7\t0.5\t0.5\n")
            with self.assertRaises(ValueError):
                load_lexicon(path)


class SplitsTests(unittest.TestCase):
    def test_all_readers_of_sentence_stay_together(self):
        table, _ = synthetic_table()
        for _, train, test, _ in make_splits(table, "sentence", 5, 42):
            self.assertFalse(set(table.iloc[train].sentence_key) & set(table.iloc[test].sentence_key))
            self.assertEqual(len(test), len(table[table.sentence_key.isin(table.iloc[test].sentence_key)]))

    def test_shared_words_are_purged(self):
        table, _ = synthetic_table()
        table.loc[table.word_index == 0, "word_form"] = "shared-negative"
        for _, train, test, purged in make_splits(table, "sentence_word_form", 5, 42):
            self.assertFalse(set(table.iloc[train].word_form) & set(table.iloc[test].word_form))
            self.assertGreater(purged, 0)

    def test_joint_subject_sentence_and_word_form_holdout(self):
        table, _ = synthetic_table(readers=5)
        for _, train, test, _ in make_splits(table, "subject_sentence_word_form", 5, 42):
            for column in ["subject", "sentence_key", "word_form"]:
                self.assertFalse(set(table.iloc[train][column]) & set(table.iloc[test][column]))


class NumericalTests(unittest.TestCase):
    def test_scaling_uses_training_only(self):
        a = np.array([[1., 0., np.nan], [3., 0., 2.], [5., 0., 3.]])
        b = np.array([[1000., 100., 100.]])
        x1, _, info1 = transform_fit(a, b)
        x2, _, info2 = transform_fit(a, b * 100)
        np.testing.assert_equal(x1, x2)
        self.assertEqual(info1, info2)
        self.assertEqual(info1["features_kept"], 2)

    def test_known_signal_and_permutation_control(self):
        table, x = synthetic_table()
        cfg = {**CONFIG, "permutations": 9, "minimum_train_per_class": 5,
               "minimum_test_per_class": 3, "minimum_train_word_forms_per_class": 2}
        with redirect_stdout(io.StringIO()):
            report, predictions = run_probe(table, x, cfg, "sentence_word_form", ["negative", "neutral", "positive"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(predictions), len(table))
        self.assertGreater(report["scores"]["eeg"]["balanced_accuracy"], .95)
        self.assertLess(np.mean(report["word_form_permutation_screen"]["null_balanced_accuracies"]), .65)

    def test_insufficient_classes_do_not_disappear(self):
        table, x = synthetic_table()
        cfg = {**CONFIG, "minimum_train_per_class": 10000}
        report, predictions = run_probe(table, x, cfg, "sentence", ["negative", "neutral", "positive"])
        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["completed_folds"], 0)
        self.assertTrue(predictions.empty)

    def test_metrics_class_order(self):
        m = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [0, 1])
        self.assertAlmostEqual(m["balanced_accuracy"], .75)
        self.assertEqual(m["confusion"], [[1, 1], [0, 2]])


class DataTests(unittest.TestCase):
    def test_bad_offsets_rejected(self):
        data = {"words": np.array(["a"]), "features": np.ones((1, 8, 105)), "offsets": np.array([0, 2]),
                "sentence_id": np.array([1]), "sample_id": np.array(["s"]), "fixations": np.ones(1)}
        with self.assertRaises(ValueError):
            validate_cache(data, "fixture")

    def test_skipped_word_alignment_retained_and_class_arrays_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ZAB.npz"
            features = np.ones((3, 8, 105), np.float32)
            features[1] = np.nan
            atomic_npz(path, words=np.array(["bad", "mystery", "good"]), features=features,
                       offsets=np.array([0, 3]), sentence_id=np.array([0]), sample_id=np.array(["a"]), fixations=np.array([1, 0, 1]))
            with redirect_stdout(io.StringIO()):
                table, x, _ = load_sources([{"dataset": "ZuCo1", "task": "NR", "files": [str(path)]}])
            self.assertEqual(table.word_index.tolist(), [0, 1, 2])
            self.assertEqual(table.eeg_status.tolist(), ["ok", "not_fixated", "ok"])
            lex = {"bad": {"valence": .1}, "good": {"valence": .9}}
            table = annotate(table, lex, CONFIG["label_thresholds"])
            class_exports(table, x, Path(tmp) / "out")
            with np.load(Path(tmp) / "out/eeg_by_class/NR/positive.npz") as export:
                self.assertEqual(export["features"].shape, (1, 8, 105))
                np.testing.assert_equal(export["row_id"], [2])

    def test_raw_segments_are_oriented_without_padding(self):
        segment = np.ones((50, 105))
        self.assertEqual(orient(segment).shape, (105, 50))
        self.assertIsNone(orient(np.ones((10, 10))))
        self.assertEqual(v5_cells(np.ones((105, 51)))[0].shape, (105, 51))

    def test_sentence_identity_not_numeric_id_or_task(self):
        self.assertEqual(sentence_key(["Good", "film"]), sentence_key(["good", "film"]))
        self.assertNotEqual(sentence_key(["good", "film"]), sentence_key(["film", "good"]))


if __name__ == "__main__":
    unittest.main()
