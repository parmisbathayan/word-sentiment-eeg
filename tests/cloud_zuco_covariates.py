"""Confound tables must preserve identities, eligibility, counts and unknowns."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('covariates', ROOT / 'scripts/summarize_zuco_covariates.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as temp:
    run = Path(temp); (run / 'audit').mkdir()
    (run / 'audit/integrity.json').write_text('{"status":"passed"}')
    words = ['bad', 'plain', 'verygood', 'negated', 'lost']
    labels = pd.DataFrame([{'sentence_key': 'fixture', 'word_index': i, 'word': word,
                           'lexical_class': ['negative', 'neutral', 'positive', 'positive', 'negative'][i],
                           'contextual_class': ['negative', 'neutral', 'positive', 'negative', 'negative'][i],
                           'alignment_warning': '', 'annotation_status': 'agent_reviewed_not_human_gold', 'tasks': 'SR'}
                          for i, word in enumerate(words)])
    labels.to_csv(run / 'word_labels.csv', index=False)
    records = []
    for reader in ('R1', 'R2'):
        output = run / 'readers/SR' / reader; output.mkdir(parents=True)
        rows = [{'event_id': reader + str(i), 'task': 'SR', 'subject': reader,
                 'raw_sentence_position': 0, 'sentence_key': 'fixture', 'word_index': i, 'word': word,
                 'lexical_class': labels.iloc[i].lexical_class, 'contextual_class': labels.iloc[i].contextual_class,
                 'status': ['accepted', 'accepted', 'accepted', 'excluded_uncertain_or_context_disagreement',
                            'excluded_missing_or_invalid_fixation'][i],
                 'first_fixation_duration_ms': [100, 200, 300, 50, None][i],
                 'next_located_fixation_ms': [150, 600, None, 100, None][i]}
                for i, word in enumerate(words)]
        pd.DataFrame(rows).to_csv(output / 'word_onsets.csv.gz', index=False, compression='gzip')
        records.append({'task': 'SR', 'subject': reader, 'word_rows': 5,
                        'epochs_per_class': {'negative': 1, 'neutral': 1, 'positive': 1}})
    (run / 'manifest.json').write_text(json.dumps({'status': 'completed', 'files': records}))
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    readers, summary = module.summarize(run)
    neg = readers[(readers.subject == 'R1') & (readers['class'] == 'negative')].iloc[0]
    assert neg.annotation_eligible_observations == 2 and neg.accepted_observations == 1
    assert neg.acceptance_percent == 50
    assert neg.word_letters_median == 3 and neg.word_position_fraction_median == 0
    assert neg.next_located_fixation_before_200ms_percent_all_accepted == 100
    positive = readers[(readers.subject == 'R1') & (readers['class'] == 'positive')].iloc[0]
    assert positive.annotation_eligible_observations == 1 # Reversed word excluded.
    assert positive.next_located_fixation_unknown_observations == 1
    assert positive.next_located_fixation_before_800ms_percent_all_accepted == 0
    assert pd.isna(positive.next_located_fixation_interval_ms_median)
    assert summary.loc[summary['class'] == 'negative', 'distinct_accepted_source_words'].iloc[0] == 1
    assert summary.loc[summary['class'] == 'negative', 'accepted_observations'].iloc[0] == 2
    for p, sha in originals.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
    events = run / 'readers/SR/R1/word_onsets.csv.gz'
    corrupt = pd.read_csv(events); corrupt.loc[1, 'event_id'] = corrupt.loc[0, 'event_id']
    corrupt.to_csv(events, index=False, compression='gzip')
    try:
        module.summarize(run)
        raise AssertionError('Duplicate event was accepted')
    except ValueError as error:
        assert 'duplicate ID' in str(error)
    print('ZuCo covariates: count eligibility, duplicates, missing timing, source units, preservation passed')
