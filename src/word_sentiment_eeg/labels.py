"""Lexical labels and context review flags are separate annotation layers.

NRC VAD v1 is fetched only by an explicit cloud-only function. Unmatched words
never become neutral. Context flags are screening heuristics, not annotations.
"""
import csv
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .common import file_digest, normalize_word, write_json

NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
NRC_HOME = "https://saifmohammad.com/WebPages/nrc-vad.html"
NRC_V1_MEMBER = "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
NEGATORS = {"not", "no", "nor", "neither", "never", "without", "hardly", "barely", "scarcely"}
CONTRAST = {"but", "however", "although", "though", "yet", "despite"}


def ensure_lexicon(directory):
    directory = Path(directory)
    target = directory / "NRC-VAD-Lexicon.txt"
    if target.exists():
        return target
    try:
        import google.colab  # noqa: F401; available in the Colab subprocess too
    except ImportError as error:
        raise RuntimeError("Lexicon downloads are cloud-only. Run Colab or supply --lexicon.") from error
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "NRC-VAD-Lexicon.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        req = urllib.request.Request(NRC_URL, headers={"User-Agent": "word-sentiment-eeg/research"})
        with urllib.request.urlopen(req, timeout=90) as response, open(tmp, "wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            raise ValueError("NRC download is not a ZIP; original server may be unavailable.")
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        # The official archive also contains BipolarScale/NRC-VAD-Lexicon.txt.
        # Select the documented [0,1] file explicitly; never take an arbitrary basename match.
        candidates = [n for n in z.namelist() if n == NRC_V1_MEMBER]
        if len(candidates) != 1:
            raise ValueError(f"Expected one v1 English lexicon; found {candidates}")
        target.write_bytes(z.read(candidates[0]))
        # Retain original notices/README alongside the source data, on Drive only.
        for name in z.namelist():
            if "readme" in Path(name).name.lower() and "__MACOSX" not in name:
                (directory / ("NRC-" + Path(name).name)).write_bytes(z.read(name))
    write_json(directory / "source.json", {
        "name": "NRC VAD v1 (2018)", "url": NRC_URL, "homepage": NRC_HOME,
        "sha256": file_digest(target), "archive_sha256": file_digest(archive),
        "terms": "Non-commercial research/education; no redistribution. Cite Mohammad (2018)."
    })
    return target


def load_lexicon(path):
    # The official v1 English file is headerless. Also accept a named-column
    # copy without dropping its first real word (pandas' default header inference).
    columns = ["word", "valence", "arousal", "dominance"]
    table = pd.read_csv(path, sep="\t", encoding="utf-8-sig", header=None, names=columns)
    if len(table) and str(table.iloc[0]["word"]).strip().lower() == "word":
        if [str(v).strip().lower() for v in table.iloc[0]] == columns:
            table = table.iloc[1:].copy()
    for c in ("valence", "arousal", "dominance"):
        table[c] = pd.to_numeric(table[c], errors="raise")
        if not table[c].between(0, 1).all():
            raise ValueError("Expected v1 scores in [0,1]; do not silently use v2 scores or a different scale.")
    table["word_form"] = table.word.map(normalize_word)
    if table.word_form.duplicated().any():
        raise ValueError("Ambiguous normalized lexicon entries; inspect instead of averaging.")
    return table.set_index("word_form")[["valence", "arousal", "dominance"]].to_dict("index")


def class_from_valence(value, thresholds):
    if value is None or not np.isfinite(value):
        return "unknown"
    if value <= thresholds["negative_max"]:
        return "negative"
    if value >= thresholds["positive_min"]:
        return "positive"
    if thresholds["neutral_min"] <= value <= thresholds["neutral_max"]:
        return "neutral"
    return "ambiguous"


def annotate(table, lexicon, thresholds):
    result = table.copy()
    result["word_form"] = result.word.map(normalize_word)
    for c in ("valence", "arousal", "dominance"):
        result[c] = [lexicon.get(w, {}).get(c, np.nan) for w in result.word_form]
    result["lexical_class"] = [class_from_valence(v, thresholds) for v in result.valence]
    result["label_source"] = "NRC_VAD_v1_exact_normalized_lookup"
    result["label_status"] = "automatic_lexical_proxy"
    result["contextual_class"] = "unreviewed"
    result["context_notes"] = ""
    result["context_reviewer"] = ""
    # Compute shared flags once per sentence; never repeat annotation by reader.
    flags = {}
    for sid, words in result.groupby("sentence_key", sort=False):
        words = words.drop_duplicates("word_index").sort_values("word_index")
        tokens = words.word_form.tolist()
        classes = set(words.lexical_class)
        mixed = {"negative", "positive"}.issubset(classes)
        contrast = bool(set(tokens) & CONTRAST)
        for j, row in enumerate(words.itertuples()):
            prefix = tokens[max(0, j - 3):j]
            neg = any(w in NEGATORS or w.endswith("n't") for w in prefix)
            # This does not claim negation scope, sarcasm, or contextual polarity.
            reasons = [name for name, present in (
                ("nearby_negation", neg), ("contrast_in_sentence", contrast),
                ("mixed_lexical_polarities", mixed)) if present]
            flags[(sid, row.word_index)] = ";".join(reasons)
    result["context_flags"] = [flags[(s, w)] for s, w in zip(result.sentence_key, result.word_index)]
    result["needs_context_review"] = result.context_flags.ne("")
    return result


def apply_reviews(table, path):
    """Optional human contextual labels; deliberately never overwrite lexical labels."""
    if path is None or not Path(path).exists():
        return table
    reviews = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {"item_id", "contextual_class", "reviewer", "notes"}
    if not required.issubset(reviews):
        raise ValueError(f"Review CSV needs {sorted(required)}")
    reviews = reviews[reviews.contextual_class.ne("") & reviews.contextual_class.ne("unreviewed")]
    allowed = {"negative", "neutral", "positive", "uncertain", "exclude"}
    if reviews.item_id.duplicated().any() or not set(reviews.contextual_class) <= allowed:
        raise ValueError("Review IDs must be unique and contextual labels valid.")
    if not set(reviews.item_id) <= set(table.item_id):
        raise ValueError("Review CSV contains IDs absent from this dataset.")
    if reviews.reviewer.str.strip().eq("").any():
        raise ValueError("Reviewed labels must have reviewer provenance.")
    result = table.copy()
    by_id = reviews.set_index("item_id")
    selected = result.item_id.isin(by_id.index)
    result.loc[selected, "contextual_class"] = result.loc[selected, "item_id"].map(by_id.contextual_class)
    result.loc[selected, "context_notes"] = result.loc[selected, "item_id"].map(by_id.notes)
    result.loc[selected, "context_reviewer"] = result.loc[selected, "item_id"].map(by_id.reviewer)
    return result
