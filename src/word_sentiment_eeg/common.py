"""Small deterministic I/O helpers; no network or installation side effects."""
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_word(word):
    # Preserve internal apostrophes/hyphens. Do not collapse two distinct tokens.
    word = unicodedata.normalize("NFKC", str(word)).casefold().replace("’", "'")
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", word)


def sentence_key(words):
    return digest([unicodedata.normalize("NFKC", str(w)).casefold().strip() for w in words])[:24]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
