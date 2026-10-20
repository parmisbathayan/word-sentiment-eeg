"""Fixation-onset recovery and baseline-corrected word FRPs (ZuCo 1.0).

Onsets are relative to the first sample of sentenceData.rawData (zero based).
Only the FIRST stored fixation is a candidate; losing it never promotes a later
fixation. All matches are checked over the entire segment and all 105 channels.
"""
import numpy as np
from .raw_export import orient

CLASSES = ('negative', 'neutral', 'positive')
SFREQ = 500
PRE, POST = 100, 400  # -200..+800 ms inclusive, 501 samples
TIMES_MS = np.arange(-PRE, POST + 1) * 1000 / SFREQ
ROI = ('E55', 'Cz', 'E31', 'E80', 'E79', 'E54', 'E106', 'E7', 'E87', 'E37', 'E78', 'E61')


def eligible(row):
    """Conservative classes: both layers agree and neither needs adjudication."""
    return (row.lexical_class in CLASSES and row.lexical_class == row.contextual_class
            and not str(row.alignment_warning) and row.annotation_status == 'agent_reviewed_not_human_gold')


def locate(raw, segment, tolerance=1e-4):
    if segment is None or segment.shape[1] < 3 or not np.isfinite(segment).all():
        return None, 'missing_or_invalid_fixation'
    length, total = segment.shape[1], raw.shape[1]
    if length > total:
        return None, 'longer_than_sentence'
    spread = segment.std(axis=1)
    channels = np.argsort(spread)[-8:]
    if spread[channels[-1]] < 1e-8:
        return None, 'flat_segment'
    threshold = tolerance * max(float(np.abs(segment[channels]).max()), 1.)
    n = total - length + 1
    # Three samples cheaply shortlist positions; full-segment validation follows.
    for method in ('exact', 'constant_channel_offset'):
        r, s = (raw, segment) if method == 'exact' else (np.diff(raw, axis=1), np.diff(segment, axis=1))
        cost = sum(np.abs(r[channels, j:j+n] - s[channels, j:j+1]).max(axis=0) for j in range(min(3, s.shape[1])))
        candidates = np.flatnonzero(cost <= threshold * min(3, s.shape[1]))
        matches = []
        for start in candidates:
            difference = raw[:, start:start+length] - segment
            if method != 'exact':
                difference = difference - np.median(difference, axis=1, keepdims=True)
            if np.max(np.abs(difference)) <= threshold:
                matches.append(int(start))
        if len(matches) == 1:
            return matches[0], method
        if len(matches) > 1:
            return None, 'nonunique_match'
    return None, 'unmatched'


def preprocess(raw):
    from scipy.signal import butter, sosfiltfilt
    # Source data are already preprocessed by ZuCo. Apply low-pass to the whole
    # sentence before epoching; never filter each short word segment separately.
    value = raw.astype(np.float64)
    value -= value.mean(axis=0, keepdims=True)
    return sosfiltfilt(butter(4, 30, fs=SFREQ, output='sos'), value, axis=1).astype(np.float32)


def epoch(filtered, onset):
    if onset < PRE or onset + POST >= filtered.shape[1]:
        return None, 'sentence_boundary'
    value = filtered[:, onset-PRE:onset+POST+1].copy()
    value -= value[:, :PRE].mean(axis=1, keepdims=True)
    if not np.isfinite(value).all():
        return None, 'nonfinite_epoch'
    if np.abs(value).max() > 100:
        return None, 'amplitude_over_100_uV'
    return value, 'accepted'


def _v5_fixations(value):
    # Preserve rejected NaN cells as None so first fixation cannot be replaced.
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [v for item in value.reshape(-1) for v in _v5_fixations(item)]
    return [orient(value)]


def _hdf_fixations(handle, ref):
    import h5py
    if not ref:
        return [None]
    obj = handle[ref]
    if not isinstance(obj, h5py.Dataset) or obj.attrs.get('MATLAB_empty', 0):
        return [None]
    if h5py.check_dtype(ref=obj.dtype) is not None:
        return [v for child in np.asarray(obj).reshape(-1) for v in _hdf_fixations(handle, child)]
    return [orient(np.asarray(obj), from_hdf5=True)]


def iter_sentences(path):
    """Stream v7.3 sentences; v5 is necessarily loaded as one subject file."""
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as handle:
            data = handle['sentenceData']
            wordrefs = np.asarray(data['word']).reshape(-1)
            rawrefs = np.asarray(data['rawData']).reshape(-1) if 'rawData' in data else [None]*len(wordrefs)
            for pos, (wr, rr) in enumerate(zip(wordrefs, rawrefs)):
                if not wr or not isinstance(handle[wr], h5py.Group):
                    continue
                group = handle[wr]
                if 'content' not in group:
                    continue
                words = [''.join(chr(int(c)) for c in np.asarray(handle[r]).reshape(-1) if int(c)>0)
                         for r in np.asarray(group['content']).reshape(-1)]
                segments = [_hdf_fixations(handle, r) for r in np.asarray(group['rawEEG']).reshape(-1)] if 'rawEEG' in group else [[None] for _ in words]
                if len(words) != len(segments):
                    raise ValueError(f'Word/segment count mismatch in {path}, sentence {pos}')
                raw = orient(np.asarray(handle[rr]), from_hdf5=True) if rr else None
                yield pos, words, raw, segments
        return
    from scipy.io import loadmat
    data = loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=['sentenceData'])
    for pos, sent in enumerate(np.atleast_1d(data['sentenceData'])):
        entries = list(np.atleast_1d(getattr(sent, 'word', [])))
        if not entries or not hasattr(entries[0], 'content'):
            continue
        words = [str(w.content).strip() for w in entries]
        yield pos, words, orient(getattr(sent, 'rawData', None)), [_v5_fixations(getattr(w, 'rawEEG', None)) for w in entries]
