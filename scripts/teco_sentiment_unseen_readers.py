#!/usr/bin/env python3
"""Reader-held-out sentiment follow-up using the completed TeCo feature caches."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import pandas as pd
import teco_other_targets as t

PLAN = {name: dict(modes=['reader'], sizes=[1, 10, 40, 'max40']) for name in
        ['sentiment_binary', 'neutral_nonneutral', 'sentiment_three_class']}
BUDGET = 50_000_000
SPEC = dict(version=1, parent_protocol=t.SPEC, plan=PLAN, budget=BUDGET,
            comparison='Same frozen labels, cached inputs, seed, nested selection and controls as teco_other_targets; only change sentiment holdout to reader',
            require_reference='Completed original teco_other_targets; identical source identities and per-target event membership/labels',
            fixed_binary='Require saved advanced all-channel224–304ms reference; never silently substitute regional features',
            no_extraction=True, no_new_feature_cache=True,
            interpretation='Exploratory; known class labels form averages; no new independent confirmation or significance test')


def verified_reference(thesis):
    root = thesis / 'Results/word_sentiment_eeg/teco_other_targets'
    state = t.read(root / 'complete.json')
    if state.get('status') != 'completed':
        raise ValueError('Original teco_other_targets has no completed reference')
    required = ['protocol.json'] + [f'targets/{name}/{file}' for name in PLAN
                                   for file in ['definition.json', 'events.csv.gz']]
    for path in required:
        expected = state['checksums'].get(path)
        if not expected or t.a.file_digest(root / path) != expected:
            raise ValueError('Missing/changed reference file: ' + path)
    return root, state, t.read(root / 'protocol.json')


def verify_target(reference, name, table, y, features, info):
    before = t.read(reference / f'targets/{name}/definition.json')
    if t.a.digest(before) != t.a.digest(info):
        raise ValueError('Target definition differs from completed reference: ' + name)
    original = pd.read_csv(reference / f'targets/{name}/events.csv.gz', keep_default_na=False,
                           dtype={'event_id': str, 'subject': str, 'sentence_key': str, 'word_identity': str})
    columns = ['event_id', 'subject', 'sentence_key', 'word_identity',
               'sentence_position', 'onset_sample_sentence_0based']
    current = table[columns].copy()
    current['target_code'] = y
    if len(current) != len(original) or current.event_id.duplicated().any():
        raise ValueError('Changed target event roster: ' + name)
    for col in [*columns, 'target_code']:
        # CSV numeric columns may infer integer vs float; compare values, not dtype.
        left, right = current[col].reset_index(drop=True), original[col].reset_index(drop=True)
        if col in ['sentence_position', 'onset_sample_sentence_0based', 'target_code']:
            equal = (pd.to_numeric(left, errors='raise').eq(pd.to_numeric(right, errors='raise')) |
                     (left.isna() & right.isna())).all()
        else:
            equal = left.astype(str).eq(right.astype(str)).all()
        if not equal:
            raise ValueError(f'Changed target membership/labels: {name}/{col}')
    if name == 'sentiment_binary' and 'advanced_channels_224_304' not in features:
        raise ValueError('Completed binary advanced feature cache required; no extraction or fallback is allowed')
    return current


def run(thesis, free_drive_bytes):
    out = thesis / 'Results/word_sentiment_eeg/teco_sentiment_unseen_readers'
    store = t.ImmutableStore(out, BUDGET)
    if free_drive_bytes < max(0, BUDGET - store.used):
        raise OSError('Reserve at most50MB for new results; no extraction will run')
    reference, state, protocol = verified_reference(thesis)
    meta, features, source = t.load_all(thesis)
    if t.a.digest(source) != t.a.digest(protocol['source']):
        raise ValueError('Cached source differs from completed teco_other_targets reference')
    files = ['scripts/teco_sentiment_unseen_readers.py', 'scripts/teco_other_targets.py',
             'scripts/teco_average40.py', 'scripts/teco_classification_sweep.py',
             'scripts/analyze_trial_archive.py']
    store.json('protocol.json', dict(spec=SPEC, source=source,
               reference_complete_sha256=t.a.file_digest(reference / 'complete.json'),
               reference_protocol_sha256=state['checksums']['protocol.json'],
               code={f: t.a.file_digest(ROOT / f) for f in files},
               versions={n: version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn']}))
    if (out / 'complete.json').is_file():
        for path, sha in t.read(out / 'complete.json')['checksums'].items():
            if t.a.file_digest(out / path) != sha:
                raise ValueError('Corrupt completed output: ' + path)
        print('KEEP COMPLETE UNSEEN-READER SENTIMENT: verified, no refits', flush=True)
        return
    rows, development, skipped = [], [], []
    for ti, (name, plan) in enumerate(PLAN.items(), 1):
        print(f'BEGIN SENTIMENT TARGET {ti}/3: {name}; unseen readers; sizes1,10,exact40,Max40', flush=True)
        table, y, vectors, classes, info = t.prepare_target(thesis, meta, features, name)
        export = verify_target(reference, name, table, y, vectors, info)
        store.json(f'targets/{name}/definition.json', info)
        t.old.csv(store, f'targets/{name}/events.csv.gz', export)
        print(f'VERIFIED ORIGINAL ROSTER: {name}; {len(table)}trials; {table.subject.nunique()}readers; {info["class_counts"]}', flush=True)
        result, dev, unavailable = t.evaluate(table, y, vectors, classes, plan, store, name)
        rows.extend(result); development.extend(dev); skipped.extend(unavailable)
    if not rows:
        raise ValueError('No usable outer tests; see live unavailable-condition logs')
    summary = t.summarize(rows)
    t.old.csv(store, 'summary.csv', summary)
    t.old.csv(store, 'development.csv.gz', pd.DataFrame(development))
    t.old.csv(store, 'unavailable.csv', pd.DataFrame(skipped, columns=['target', 'mode', 'fold', 'size', 'reason']))
    t.zipped_json(store, 'outer_results.json.gz', rows)
    choices = [{k: r.get(k) for k in ['target', 'mode', 'fold', 'size', 'input', 'model', 'status']}
               for r in rows if r['variant'] == 'selected_eeg']
    t.old.csv(store, 'selected_inputs_models.csv', pd.DataFrame(choices))
    t.figures(store, summary, rows)
    paths = ['protocol.json', 'summary.csv', 'development.csv.gz', 'unavailable.csv',
             'outer_results.json.gz', 'selected_inputs_models.csv', 'overview.png']
    paths += [str(p.relative_to(out)) for p in out.rglob('result.json.gz')]
    paths += [str(p.relative_to(out)) for p in out.glob('targets/*/definition.json')]
    paths += [str(p.relative_to(out)) for p in out.glob('targets/*/events.csv.gz')]
    store.json('complete.json', dict(status='completed',
               checksums={f: t.a.file_digest(out / f) for f in paths}, outer_cases=len(rows),
               planned_conditions=12, complete_conditions=int(summary.groupby(['target','size']).completed_folds.min().eq(3).sum())))
    print('TECO UNSEEN-READER SENTIMENT COMPLETE:', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thesis-root', type=Path, required=True)
    p.add_argument('--free-drive-bytes', type=int, required=True)
    args = p.parse_args()
    import fcntl, tempfile
    path = args.thesis_root / 'Results/word_sentiment_eeg/teco_sentiment_unseen_readers'
    lock = Path(tempfile.gettempdir(), 'teco-sentiment-readers-' + t.a.digest(str(path))[:16] + '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(args.thesis_root, args.free_drive_bytes)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
