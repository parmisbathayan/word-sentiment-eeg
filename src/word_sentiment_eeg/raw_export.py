"""Export original variable-length word fixation EEG; no timing is invented.

These are preprocessed ZuCo word.rawEEG segments, not acquisition-raw EEG and
not fixed post-onset FRP epochs. No padding, interpolation or concatenation
across fixations is used within a segment. Stored segments are grouped by class.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import atomic_npz, digest, file_digest, sentence_key, write_json


def orient(segment, from_hdf5=False):
    segment = np.asarray(segment)
    if segment.ndim != 2 or not np.issubdtype(segment.dtype, np.number):
        return None
    if segment.shape == (105, 105):
        return (segment.T if from_hdf5 else segment).astype(np.float32)
    if segment.shape[0] == 105:
        return segment.astype(np.float32)
    if segment.shape[1] == 105:
        return segment.T.astype(np.float32)
    return None


def hdf_cells(handle, ref):
    import h5py
    if not ref:
        return []
    obj = handle[ref]
    if not isinstance(obj, h5py.Dataset) or obj.attrs.get("MATLAB_empty", 0):
        return []
    if h5py.check_dtype(ref=obj.dtype) is not None:
        return [v for child in np.asarray(obj).reshape(-1) for v in hdf_cells(handle, child)]
    value = orient(np.asarray(obj), from_hdf5=True)
    return [value] if value is not None and value.shape[1] else []


def v5_cells(value):
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [a for child in value.reshape(-1) for a in v5_cells(child)]
    value = orient(value)
    return [value] if value is not None and value.shape[1] else []


def iter_raw_words(path):
    """Yield (sentence_position, words, per_word_segments, raw_field_available)."""
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            data = handle["sentenceData"]
            for i, ref in enumerate(np.asarray(data["word"]).reshape(-1)):
                if not ref or not isinstance(handle[ref], h5py.Group) or "content" not in handle[ref]:
                    continue
                group = handle[ref]
                words = ["".join(chr(int(v)) for v in np.asarray(handle[r]).reshape(-1) if int(v) > 0)
                         for r in np.asarray(group["content"]).reshape(-1)]
                if "rawEEG" not in group:
                    yield i, words, [[] for _ in words], False
                else:
                    refs = np.asarray(group["rawEEG"]).reshape(-1)
                    if len(refs) != len(words):
                        raise ValueError(f"Raw word references and text disagree in {path}, sentence {i}")
                    yield i, words, [hdf_cells(handle, r) for r in refs], True
        return
    from scipy.io import loadmat
    data = loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=["sentenceData"])
    for i, sentence in enumerate(np.atleast_1d(data["sentenceData"])):
        entries = list(np.atleast_1d(getattr(sentence, "word", [])))
        if not entries or not hasattr(entries[0], "content"):
            continue
        words = [str(getattr(w, "content", "")).strip() for w in entries]
        exists = any(hasattr(w, "rawEEG") for w in entries)
        yield i, words, [v5_cells(getattr(w, "rawEEG", None)) for w in entries], exists


def export_available_raw(table, raw_dir, artifact_dir, fingerprint):
    raw_dir, artifact_dir = Path(raw_dir), Path(artifact_dir)
    report = {"representation": "preprocessed word.rawEEG fixation segments; variable length, no baseline epoch",
              "sampling_rate_hz": 500, "dataset": "ZuCo 1.0",
              "amplitude_units": "preserved exactly as stored in source; inspect source preprocessing before amplitude inference",
              "no_download": True, "source_dir": str(raw_dir), "files": [], "missing_tasks": []}
    files = sorted(raw_dir.rglob("results*_*.mat")) if raw_dir.exists() else []
    task_files = {task: [] for task in table.task.unique()}
    for path in files:
        match = re.fullmatch(r"results([A-Za-z0-9]+)_(SR|NR|TSR)\.mat", path.name)
        if match and match.group(2) in task_files:
            task_files[match.group(2)].append((path, match.group(1)))
    for task, candidates in task_files.items():
        if not candidates:
            report["missing_tasks"].append(task)
        for path, subject in candidates:
            subset = table[(table.task == task) & (table.subject == subject)]
            if subset.empty:
                continue
            print(f"Exporting available raw fixation EEG {task}/{subject}", flush=True)
            source_hash = file_digest(path)
            target = artifact_dir / fingerprint[:16] / task / subject
            done = target / "manifest.json"
            if done.exists():
                previous = __import__("json").loads(done.read_text())
                if previous.get("source_sha256") == source_hash:
                    report["files"].append(previous)
                    continue
            lookup = {(r.sentence_key, r.word_index): r for r in subset.itertuples()}
            segments = {c: [] for c in ("negative", "neutral", "positive")}
            metadata = {c: [] for c in segments}
            field_present = False
            n_unmatched, n_invalid = 0, 0
            for sentence_position, words, per_word, has_raw in iter_raw_words(path):
                field_present |= has_raw
                skey = sentence_key(words)
                for j, fixation_segments in enumerate(per_word):
                    row = lookup.get((skey, j))
                    if row is None:
                        n_unmatched += 1
                        continue
                    label = row.lexical_class
                    if label not in segments or row.eeg_status == "not_fixated":
                        continue
                    for fixation_index, segment in enumerate(fixation_segments):
                        if not np.isfinite(segment).all() or segment.shape[1] < 2:
                            n_invalid += 1
                            continue
                        segments[label].append(segment)
                        metadata[label].append({"row_id": row.row_id, "item_id": row.item_id,
                            "word": row.word, "word_index": j, "sentence_key": skey,
                            "raw_sentence_position": sentence_position, "fixation_index_in_stored_cell": fixation_index,
                            "n_samples": segment.shape[1], "context_flagged": row.needs_context_review,
                            "lexical_class": row.lexical_class,
                            "contextual_class": getattr(row, 'contextual_class', 'unreviewed')})
            for label, blocks in segments.items():
                offsets = np.cumsum([0] + [b.shape[1] for b in blocks], dtype=np.int64)
                atomic_npz(target / f"{label}.npz", samples=np.concatenate(blocks, axis=1) if blocks else np.zeros((105, 0), np.float32),
                           offsets=offsets, sfreq_hz=np.array(500), row_id=np.array([r["row_id"] for r in metadata[label]], dtype=np.int64))
                pd.DataFrame(metadata[label], columns=["row_id", "item_id", "word", "word_index", "sentence_key", "raw_sentence_position", "fixation_index_in_stored_cell", "n_samples", "context_flagged", "lexical_class", "contextual_class"]).to_csv(target / f"{label}_segments.csv", index=False)
            item = {"source": str(path), "source_sha256": source_hash, "task": task, "subject": subject,
                    "output_dir": str(target), "raw_field_present": field_present,
                    "status": "exported" if field_present else "rawEEG_unavailable",
                    "segments_per_class": {k: len(v) for k, v in segments.items()},
                    "unmatched_word_rows": n_unmatched, "invalid_segments_excluded": n_invalid}
            write_json(done, item)
            report["files"].append(item)
    report["status"] = "exported_available_only" if report["files"] else "no_raw_files_found"
    return report
