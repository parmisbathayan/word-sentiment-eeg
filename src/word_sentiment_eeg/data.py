"""Read existing ZuCo word caches without copying the source EEG on Drive.

Cache schema follows the earlier thesis project's fusion/word_eeg.py. Every
word retains its original sentence/word index even when EEG is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, file_digest, sentence_key

BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
EXPECTED_SUBJECTS = {"ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"}


def validate_cache(data, path):
    required = {"words", "features", "offsets", "sentence_id", "sample_id", "fixations"}
    if not required.issubset(data):
        raise ValueError(f"{path}: missing {sorted(required - set(data))}")
    n = len(data["words"])
    offsets = data["offsets"]
    if (offsets.ndim != 1 or len(offsets) != len(data["sentence_id"]) + 1
            or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0)
            or not np.issubdtype(offsets.dtype, np.integer)):
        raise ValueError(f"{path}: invalid word offsets; alignment cannot be trusted")
    if data["features"].shape != (n, 8, 105) or data["fixations"].shape != (n,):
        raise ValueError(f"{path}: expected words × 8 bands × 105 channels and one fixation count per word")
    if len(data["sample_id"]) != len(offsets) - 1:
        raise ValueError(f"{path}: trial IDs do not match offsets")
    for key in ("trt_ms", "ffd_ms", "gd_ms"):
        if key in data and data[key].shape != (n,):
            raise ValueError(f"{path}: invalid {key} alignment")


def inspect_sources(thesis_root, config, overrides=None):
    root = Path(thesis_root)
    result = []
    for task in config["tasks"]:
        path = Path((overrides or {}).get(task, root / config["cache_relative_paths"][task]))
        files = sorted(path.glob("Z*.npz")) if path.is_dir() else []
        subjects = {p.stem for p in files}
        result.append({"dataset": "ZuCo1", "task": task, "path": str(path),
                       "measure": config["measure"], "files": [str(p) for p in files],
                       "subjects_found": sorted(subjects),
                       "missing_expected_subjects": sorted(EXPECTED_SUBJECTS - subjects),
                       "status": "found" if files else "missing"})
    return result


def load_sources(inventory):
    frames, blocks, provenance = [], [], []
    next_row = 0
    seen = set()
    for source in inventory:
        for filename in source["files"]:
            path = Path(filename)
            print(f"Reading {source['task']}/{path.name}", flush=True)
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            validate_cache(data, path)
            subject = path.stem
            raw_x = data["features"].astype(np.float32)
            rows = []
            for trial, (a, b) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
                a, b = int(a), int(b)
                words = [str(w) for w in data["words"][a:b]]
                skey = sentence_key(words)
                unique = (source["dataset"], source["task"], subject, skey)
                if unique in seen:
                    raise ValueError(f"Duplicate reader × task × sentence in {path}; resolve before analysis")
                seen.add(unique)
                for j, word in enumerate(words):
                    idx = a + j
                    rows.append({
                        "row_id": next_row + idx, "dataset": source["dataset"], "task": source["task"],
                        "subject": subject, "sentence_key": skey,
                        "sentence_id_original": int(data["sentence_id"][trial]),
                        "sample_id_original": str(data["sample_id"][trial]),
                        "item_id": digest([source["dataset"], source["task"], skey, j])[:24],
                        "word_index": j, "word": word, "sentence": " ".join(words),
                        "sentence_text_source": "reconstructed_from_original_word_list",
                        "sentence_length": len(words), "relative_position": j / max(1, len(words) - 1),
                        "n_fixations": float(data["fixations"][idx]),
                        **{key: float(data[key][idx]) if key in data else np.nan
                           for key in ("trt_ms", "ffd_ms", "gd_ms")},
                        "source_file": str(path), "source_word_row": idx,
                        "source_sentence_row": trial,
                        "original_sentence_label": int(data["label"][trial]) if "label" in data else 99,
                    })
            frame = pd.DataFrame(rows)
            if len(frame) != len(raw_x):
                raise ValueError(f"{path}: lost words while reading offsets")
            frames.append(frame)
            blocks.append(raw_x)
            next_row += len(raw_x)
            provenance.append({"path": str(path), "sha256": file_digest(path),
                               "bytes": path.stat().st_size, "words": len(raw_x)})
    if not frames:
        raise FileNotFoundError("No word caches found. See data_inventory.json for the expected Drive paths.")
    table = pd.concat(frames, ignore_index=True)
    x = np.concatenate(blocks)
    # A skip is never imputed into a valid recording. Partial feature loss is
    # retained for audit but excluded from the predefined complete-EEG analysis.
    table["eeg_status"] = np.select([
        table.n_fixations.le(0).to_numpy() | ~np.isfinite(table.n_fixations.to_numpy()),
        ~np.isfinite(x).any(axis=(1, 2)),
        ~np.isfinite(x).all(axis=(1, 2)),
    ], ["not_fixated", "missing_eeg", "partial_eeg"], default="ok")
    assert np.array_equal(table.row_id, np.arange(len(x)))
    return table, x, provenance
