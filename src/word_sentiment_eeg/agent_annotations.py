"""Join prepared agent annotations to original cache words with exact checks."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import file_digest, normalize_word


def attach_agent_annotations(table, path):
    annotations = pd.read_csv(path, keep_default_na=False)
    required = {'sentence_key', 'word_index', 'word', 'lexical_class', 'contextual_class',
                'annotation_flags', 'annotation_note', 'annotation_status', 'sentence_id_original',
                'tasks', 'alignment_warning'}
    if not required <= set(annotations):
        raise ValueError(f'Prepared annotation table is missing {sorted(required - set(annotations))}')
    if annotations.duplicated(['sentence_key', 'word_index']).any():
        raise ValueError('Duplicate sentence/word annotation keys')
    allowed = {'positive', 'negative', 'neutral', 'uncertain'}
    for field in ['lexical_class', 'contextual_class']:
        if not set(annotations[field]) <= allowed:
            raise ValueError(f'Unexpected prepared labels in {field}')
    lookup = {(r.sentence_key, int(r.word_index)): r for r in annotations.itertuples()}
    # A known source encoding/tokenization issue is explicit, never a positional
    # fallback: retain cache rows for audit but exclude them from EEG class exports.
    warnings = {int(r.sentence_id_original): r.alignment_warning for r in annotations.itertuples()
                if r.alignment_warning}
    result = table.copy()
    records, missing = [], []
    for row in table.itertuples():
        item = lookup.get((row.sentence_key, int(row.word_index)))
        if item is None or str(item.word) != str(row.word) or row.task not in item.tasks.split(';'):
            warning = warnings.get(int(row.sentence_id_original))
            if warning:
                records.append({'lexical_class': 'unknown', 'contextual_class': 'unresolved_alignment',
                    'annotation_flags': 'alignment_mismatch', 'annotation_note': warning,
                    'annotation_status': 'excluded_unresolved_alignment'})
                continue
            missing.append((row.task, getattr(row, 'subject', 'unspecified'), int(row.sentence_id_original), int(row.word_index), row.word))
            records.append({})
        else:
            records.append({field: getattr(item, field) for field in ['lexical_class', 'contextual_class',
                            'annotation_flags', 'annotation_note', 'annotation_status']})
    if missing:
        raise ValueError(f'{len(missing)} EEG rows lack exact prepared labels; no dictionary/neutral fallback. First rows: {missing[:8]}')
    joined = pd.DataFrame(records, index=result.index)
    for field in joined:
        result[field] = joined[field]
    result['label_source'] = 'Codex_single_agent_sentence_and_token_review_v2'
    result['label_status'] = result.annotation_status
    result['context_flags'] = result.annotation_flags
    result['context_notes'] = result.annotation_note
    result['context_reviewer'] = 'Codex (agent; not human consensus)'
    result['needs_context_review'] = result.contextual_class.eq('uncertain') | result.annotation_status.eq('excluded_unresolved_alignment')
    result['word_form'] = table.word.map(normalize_word)
    for field in ['valence', 'arousal', 'dominance']:
        result[field] = np.nan  # agent labels are categorical; never invent NRC ratings
    result['lexical_context_agree'] = result.lexical_class.eq(result.contextual_class)
    return result, {'source': str(Path(path)), 'sha256': file_digest(path),
                    'annotation_rows': len(annotations), 'matched_eeg_rows': int((result.annotation_status != 'excluded_unresolved_alignment').sum()),
                    'annotation_sentences': int(annotations.sentence_key.nunique()),
                    'alignment_excluded_eeg_rows': int((result.annotation_status == 'excluded_unresolved_alignment').sum())}
