"""Class balance and observed reading covariates from saved ZuCo onset tables.

This is a descriptive confound/overlap check, not covariate-adjusted EEG.
No source EEG is loaded and no existing result is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from word_sentiment_eeg.erp import CLASSES


def quantiles(values, prefix):
    values = pd.to_numeric(values, errors='coerce').dropna()
    result = {f'{prefix}_observed': len(values)}
    for name, q in (('q25', .25), ('median', .5), ('q75', .75)):
        result[f'{prefix}_{name}'] = float(values.quantile(q)) if len(values) else None
    return result


def summarize(run_dir):
    run = Path(run_dir)
    if json.loads((run / 'audit/integrity.json').read_text())['status'] != 'passed':
        raise ValueError('The saved-result integrity audit must pass first')
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('Need a completed original run')
    items = manifest['files']
    keys = [(x['task'], x['subject']) for x in items]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate recording')
    labels_path = run / 'word_labels.csv'
    labels = pd.read_csv(labels_path, keep_default_na=False)
    label_keys = ['sentence_key', 'word_index']
    if labels.duplicated(label_keys).any():
        raise ValueError('Duplicate source word labels')
    labels = labels[['sentence_key', 'word_index', 'word', 'lexical_class', 'contextual_class',
                     'alignment_warning', 'annotation_status', 'tasks']].rename(columns={
                     'word': 'annotation_word', 'lexical_class': 'annotation_lexical_class',
                     'contextual_class': 'annotation_contextual_class'})
    rows = []; exclusions = []; identities = set(); unique_items = {}; source_hashes = {}
    for item in items:
        task, reader = item['task'], item['subject']
        path = run / 'readers' / task / reader / 'word_onsets.csv.gz'
        source_hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        events = pd.read_csv(path, keep_default_na=False, dtype={'sentence_key': str, 'event_id': str})
        if len(events) != item['word_rows'] or events.event_id.duplicated().any():
            raise ValueError(f'{task}/{reader}: wrong onset row count or duplicate ID')
        if identities.intersection(set(events.event_id)):
            raise ValueError('Duplicate event ID across readers')
        identities.update(events.event_id)
        if not events.task.eq(task).all() or not events.subject.eq(reader).all():
            raise ValueError('Onset identity mismatch')
        sentence_groups = events.groupby(['raw_sentence_position', 'sentence_key'], sort=False)
        sentence_sizes = sentence_groups.word_index.transform('max') + 1
        if not np.array_equal(sentence_sizes.to_numpy(), sentence_groups.word_index.transform('size').to_numpy()):
            raise ValueError('Incomplete or repeated per-sentence word indices')
        events['sentence_word_count'] = sentence_sizes
        events['word_position_fraction'] = np.where(sentence_sizes > 1, events.word_index / (sentence_sizes - 1), 0.)
        events['word_letters'] = events.word.astype(str).map(lambda word: sum(char.isalpha() for char in word))
        for col in ('first_fixation_duration_ms', 'next_located_fixation_ms'):
            events[col] = pd.to_numeric(events[col], errors='coerce')
        known = events.next_located_fixation_ms.dropna()
        if not np.isfinite(known).all() or (known <= 0).any():
            raise ValueError('Invalid next-located-fixation interval')
        events = events.merge(labels, on=label_keys, how='left', validate='many_to_one')
        eligible = (events.annotation_word.eq(events.word) &
                    events.annotation_lexical_class.isin(CLASSES) &
                    events.annotation_lexical_class.eq(events.annotation_contextual_class) &
                    events.alignment_warning.eq('') &
                    events.annotation_status.eq('agent_reviewed_not_human_gold') &
                    events.tasks.fillna('').map(lambda value: task in value.split(';')))
        if (events.status.eq('accepted') & ~eligible).any():
            raise ValueError('Accepted event violates the source annotation policy')
        for cls in CLASSES:
            targets = events[eligible & events.contextual_class.eq(cls)]
            accepted = targets[targets.status.eq('accepted')]
            n = len(accepted)
            if n != item['epochs_per_class'][cls]:
                raise ValueError('Accepted event count differs from original manifest')
            ids = unique_items.setdefault((task, cls), set())
            ids.update(zip(accepted.sentence_key, accepted.word_index))
            row = {'task': task, 'subject': reader, 'class': cls,
                   'annotation_eligible_observations': len(targets), 'accepted_observations': n,
                   'acceptance_percent': 100 * n / len(targets) if len(targets) else None,
                   'distinct_accepted_source_words': len(set(zip(accepted.sentence_key, accepted.word_index)))}
            for column, prefix in (('word_letters', 'word_letters'), ('word_index', 'word_index_0based'),
                                   ('word_position_fraction', 'word_position_fraction'),
                                   ('sentence_word_count', 'sentence_words'),
                                   ('first_fixation_duration_ms', 'first_fixation_ms'),
                                   ('next_located_fixation_ms', 'next_located_fixation_interval_ms')):
                row.update(quantiles(accepted[column], prefix))
            row['next_located_fixation_unknown_observations'] = int(accepted.next_located_fixation_ms.isna().sum())
            for threshold in (200, 500, 800):
                count = int(accepted.next_located_fixation_ms.lt(threshold).sum())
                row[f'next_located_fixation_before_{threshold}ms_observations'] = count
                row[f'next_located_fixation_before_{threshold}ms_percent_all_accepted'] = 100 * count / n if n else None
            rows.append(row)
            for status, count in targets.status.value_counts().items():
                exclusions.append({'task': task, 'subject': reader, 'class': cls, 'status': status,
                                   'observations': int(count), 'denominator_annotation_eligible': len(targets)})
        print(f'COVARIATES {task}/{reader}: accepted counts verified', flush=True)
    per_reader = pd.DataFrame(rows)
    summaries = []
    descriptive = [c for c in per_reader.columns if c.endswith('_median') or c.endswith('_percent_all_accepted')]
    for (task, cls), part in per_reader.groupby(['task', 'class'], sort=False):
        row = {'task': task, 'class': cls, 'readers': len(part),
               'accepted_observations': int(part.accepted_observations.sum()),
               'annotation_eligible_observations': int(part.annotation_eligible_observations.sum()),
               'distinct_accepted_source_words': len(unique_items[(task, cls)]),
               'median_reader_acceptance_percent': float(part.acceptance_percent.median()),
               'min_reader_accepted_observations': int(part.accepted_observations.min()),
               'max_reader_accepted_observations': int(part.accepted_observations.max()),
               'next_located_fixation_unknown_observations': int(part.next_located_fixation_unknown_observations.sum())}
        for col in descriptive:
            row['median_across_readers_' + col] = float(part[col].median()) if part[col].notna().any() else None
        for threshold in (200, 500, 800):
            row[f'next_located_fixation_before_{threshold}ms_observations'] = int(part[f'next_located_fixation_before_{threshold}ms_observations'].sum())
        summaries.append(row)
    task_summary = pd.DataFrame(summaries)
    out = run / 'analysis/covariate_balance'; out.mkdir(parents=True, exist_ok=True)
    per_reader.to_csv(out / 'reader_class_covariates.csv', index=False)
    task_summary.to_csv(out / 'task_class_covariates.csv', index=False)
    pd.DataFrame(exclusions).to_csv(out / 'eligible_word_exclusion_counts.csv', index=False)
    protocol = {'status': 'completed', 'descriptive_only': True,
                'source_onset_sha256': source_hashes,
                'source_label_sha256': hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                'aggregation': 'Medians across reader-specific summaries; repeated trials are not independent subjects',
                'word_length': 'Unicode alphabetic code points only; apostrophes, punctuation and numbers excluded',
                'word_position': 'Zero-based source token index; fraction index/(sentence_word_count-1), one-word sentences assigned zero',
                'fixation_duration': 'Stored first-fixation milliseconds; original export (segment sample length-1)*2',
                'next_fixation': 'Time to next successfully located stored fixation in this sentence; not guaranteed to be the actual next eye movement',
                'overlap_fractions': 'Denominator all accepted events, including unknown next interval. These are observed lower bounds; unlocated fixations and cross-sentence responses are missing.',
                'not_available': ['Independent word-frequency lexicon', 'Part of speech', 'Validated content/function-word status',
                                  'EEG voltage per individual word in this compact run', 'Unlocated/cross-sentence fixation timing'],
                'not_performed': ['Covariate-adjusted EEG model', 'Matched trial reconstruction', 'Sentiment-specific causal inference'],
                'original_extraction_files_modified': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    report = ['# Class balance and reading covariates', '',
              'These tables check whether word classes also differ in observable properties of the stimuli and reading. '
              'They describe potential confounds; they do not remove them from the EEG averages.', '',
              '|Task|Class|Accepted observations|Distinct source words|Median reader acceptance %|Median reader word length|Median reader first fixation ms|',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in task_summary.to_dict('records'):
        def fmt(value):
            return 'missing' if value is None or pd.isna(value) else f'{value:.3g}'
        report.append(f'|{r["task"]}|{r["class"]}|{r["accepted_observations"]}|{r["distinct_accepted_source_words"]}|'
                      f'{fmt(r["median_reader_acceptance_percent"])}|{fmt(r["median_across_readers_word_letters_median"])}|'
                      f'{fmt(r["median_across_readers_first_fixation_ms_median"])}|')
    report += ['', '“Observations” include the same source word read by multiple readers; they are not that many unique words. '
               'The class-level summary gives medians of reader summaries rather than treating every trial as an independent participant.', '',
               'The CSVs also show sentence position and length, quartiles, exclusion counts, unknown next-fixation counts and how often '
               'another successfully located fixation begins before 200, 500 or 800 ms. The latter are lower bounds on overlap opportunities. '
               'Missing next-fixation timing does not mean the eye remained still, and finding no next fixation before a threshold does not prove an uncontaminated epoch.', '',
               'Differences here motivate matched word/epoch subsets or a word-level regression with reader and item effects. '
               'Equalizing class counts alone would not match word length, frequency, part of speech, sentence context or fixation behavior. '
               'Frequency, POS and content/function-word status remain unmeasured; this script does not infer them from spelling.', '',
               'All outputs are confined to `analysis/covariate_balance/`; original EEG/class means/onset files are not changed.', '']
    (out / 'COVARIATE_INTERPRETATION.md').write_text('\n'.join(report))
    print(f'COVARIATE SUMMARY COMPLETE: {out}', flush=True)
    return per_reader, task_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir)
