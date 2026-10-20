"""Cloud-only dependency/schema smoke checks plus an end-to-end synthetic run.

The optional NRC source check runs only on GitHub Actions. No EEG is downloaded.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from word_sentiment_eeg.common import atomic_npz, sentence_key
from word_sentiment_eeg.labels import NRC_URL, NRC_V1_MEMBER, load_lexicon
from word_sentiment_eeg.raw_export import iter_raw_words, export_available_raw

ROOT = Path(__file__).resolve().parents[1]


def check_raw_schemas(directory):
    # MATLAB v5: nested fixation cells and a missing rawEEG field.
    words = [{"content": "bad", "rawEEG": np.ones((105, 25))},
             {"content": "good", "rawEEG": np.ones((105, 30)) * 2}]
    path = directory / "resultsZAB_SR.mat"
    savemat(path, {"sentenceData": [{"content": "bad good", "word": words}]})
    parsed = list(iter_raw_words(path))
    assert parsed[0][1] == ["bad", "good"]
    assert parsed[0][2][1][0].shape == (105, 30)
    # HDF5/v7.3 references: preserve word indices and variable fixation lengths.
    path = directory / "resultsZDM_NR.mat"
    with h5py.File(path, "w") as f:
        sentence = f.create_group("sentenceData")
        group = f.create_group("word0")
        refs, rawrefs = [], []
        for i, word in enumerate(["bad", "good"]):
            refs.append(f.create_dataset(f"text{i}", data=np.array([ord(c) for c in word], dtype=np.uint16)).ref)
            rawrefs.append(f.create_dataset(f"raw{i}", data=np.ones((20 + i, 105))).ref)
        group.create_dataset("content", data=np.array(refs, dtype=h5py.ref_dtype).reshape(-1, 1))
        group.create_dataset("rawEEG", data=np.array(rawrefs, dtype=h5py.ref_dtype).reshape(-1, 1))
        sentence.create_dataset("word", data=np.array([group.ref], dtype=h5py.ref_dtype).reshape(1, 1))
    parsed = list(iter_raw_words(path))
    assert parsed[0][2][1][0].shape == (105, 21)
    table = pd.DataFrame([{"task": task, "subject": subject, "sentence_key": sentence_key(["bad", "good"]),
                           "word_index": j, "row_id": i * 2 + j, "item_id": f"{task}{j}", "word": word,
                           "lexical_class": label, "eeg_status": "ok", "needs_context_review": False}
                          for i, (task, subject) in enumerate([("SR", "ZAB"), ("NR", "ZDM")])
                          for j, (word, label) in enumerate([("bad", "negative"), ("good", "positive")])])
    report = export_available_raw(table, directory, directory / "rawout", "testfingerprint")
    assert len(report["files"]) == 2
    for entry in report["files"]:
        assert entry["segments_per_class"] == {"negative": 1, "neutral": 0, "positive": 1}


def smoke(directory, prepared=False):
    config = json.loads((ROOT / "configs/default.json").read_text())
    config.update(tasks=["SR"], folds=3, permutations=3, exclude_context_flagged=False,
                  minimum_train_per_class=5, minimum_test_per_class=3,
                  minimum_train_word_forms_per_class=2, max_eeg_components=8,
                  run_tag="smoke_prepared" if prepared else "smoke_legacy")
    cfg = directory / "smoke.json"
    cfg.write_text(json.dumps(config))
    words, features, offsets = [], [], [0]
    lexicon_lines = ["Word\tValence\tArousal\tDominance"]
    for sid in range(18):
        for j, value in enumerate([.1, .5, .9]):
            token = f"word{sid}class{j}"
            words.append(token)
            lexicon_lines.append(f"{token}\t{value}\t0.5\t0.5")
            feature = np.zeros((8, 105), dtype=np.float32)
            feature[j, 0] = 5
            features.append(feature)
        offsets.append(len(words))
    cache = directory / config["cache_relative_paths"]["SR"]
    atomic_npz(cache / "ZAB.npz", words=np.array(words), features=np.array(features), offsets=np.array(offsets),
               sentence_id=np.arange(18), sample_id=np.array([str(s) for s in range(18)]), fixations=np.ones(len(words)))
    lexicon = directory / "fixture.tsv"
    lexicon.write_text("\n".join(lexicon_lines))
    command = [sys.executable, str(ROOT / "main.py"), "--thesis-root", str(directory), "--config", str(cfg)]
    if prepared:
        annotations = []
        for sid in range(18):
            tokens = words[sid*3:(sid+1)*3]
            for j, label in enumerate(['negative','neutral','positive']):
                annotations.append({'sentence_key':sentence_key(tokens), 'word_index':j, 'word':tokens[j],
                    'lexical_class':label, 'contextual_class':label, 'annotation_flags':'', 'annotation_note':'synthetic fixture',
                    'annotation_status':'synthetic_fixture', 'sentence_id_original':sid, 'tasks':'SR', 'alignment_warning':''})
        annotation_file = directory / 'fixture_annotations.csv'
        pd.DataFrame(annotations).to_csv(annotation_file,index=False)
        command += ['--annotations',str(annotation_file)]
    else:
        command += ['--lexicon',str(lexicon)]
    subprocess.run(command, check=True)
    run = directory / "Results/word_sentiment_eeg" / config['run_tag']
    assert (run / "REPORT.md").exists()
    assert json.loads((run / "manifest.json").read_text())["status"] == "completed"
    assert len(pd.read_csv(run / "all_words.csv.gz")) == len(words)
    for result in (run / "evaluations").glob("*.json"):
        summary = json.loads(result.read_text())
        assert summary["status"] == "completed", summary
    # Identical reruns reuse outputs without retraining.
    subprocess.run(command, check=True)


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    check_raw_schemas(directory)
    smoke(directory)
    smoke(directory, prepared=True)
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get('CHECK_NRC_SOURCE') == '1':
        # Read the real external schema in a cloud runtime, never on the Mac.
        request = urllib.request.Request(NRC_URL, headers={"User-Agent": "word-sentiment-eeg/research"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        names = [n for n in archive.namelist() if n == NRC_V1_MEMBER]
        assert len(names) == 1, names
        path = directory / "NRC-VAD-Lexicon.txt"
        path.write_bytes(archive.read(names[0]))
        actual = load_lexicon(path)
        assert len(actual) > 19000
        print(f"Official NRC v1 schema verified in cloud: {len(actual):,} words")
print("Cloud schema and end-to-end checks passed.")
