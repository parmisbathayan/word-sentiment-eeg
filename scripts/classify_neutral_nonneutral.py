#!/usr/bin/env python3
"""Neutral versus nonneutral follow-up; requires existing feature caches.

No EEG reads or feature preparation. Original analyses remain unchanged.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import analyze_trial_archive as a

LEGACY_STORAGE_ONLY_CODE='1f0ba76d81f74b38d24b3bb9dfbbd2f498cb55c73d7231ea59aa25af16312ebe'


def reconcile_protocol(path, protocol):
    if path.is_file():
        saved=json.loads(path.read_text())
        if saved!=protocol:
            # The first release had a 25 MB guard, which could stop after the
            # third dataset/task. Only that exact released code may migrate;
            # every data/helper/target field must remain identical.
            old=dict(saved); old_code=old.pop('code',None)
            new=dict(protocol); new.pop('code',None)
            if old_code!=LEGACY_STORAGE_ONLY_CODE or old!=new:
                raise ValueError('Follow-up protocol changed; refusing to mix saved cases')
            a.write_json(path.parent/'storage_limit_migration.json',dict(
                old_code=old_code,new_code=protocol['code'],old_output_limit_mb=25,new_output_limit_mb=60,
                reason='Storage-only correction; features, selection, splits and models unchanged'))
            print('RESUME: storage-limit fix recognized; completed cases are retained and checksum-validated.',flush=True)
    a.write_json(path,protocol)


def population(meta):
    ix, original, _ = a.selected(meta, 'word_agreement', 'three_class')
    # Reuse the original tested binary evaluator's internal negative/positive
    # coding. All exported labels explicitly name the actual scientific target.
    encoded = np.where(original == 'neutral', 'negative', 'positive')
    return ix, encoded


def label_outputs(score, predictions, readers, diagnostic):
    score = dict(score)
    score['classes'] = ['neutral', 'nonneutral']
    score['class_counts'] = {
        'neutral': score['class_counts']['negative'],
        'nonneutral': score['class_counts']['positive'],
    }
    predictions = predictions.copy()
    for column in ['truth', 'prediction']:
        predictions[column] = predictions[column].map({'negative': 'neutral', 'positive': 'nonneutral'})
    predictions = predictions.rename(columns={'probability_positive': 'probability_nonneutral'})
    return score, predictions, readers, diagnostic


def cached_record(archive, record, source, thesis):
    cache = source/'features'/record['dataset']/record['task']/record['subject']
    if not (cache/'manifest.json').is_file():
        raise ValueError(f'Completed feature cache missing: {cache}. This follow-up never prepares features.')
    # reader_features validates checkpoint signatures, cache hashes, event order
    # and label joins. With a committed cache it never opens EEG shards.
    return a.reader_features(archive, record, source, thesis)


def task_analysis(meta, post, pre, nuisance, out):
    out.mkdir(parents=True, exist_ok=True)
    duplicate = meta.duplicated(['subject','sentence_position','onset_sample_sentence_0based'], keep=False)
    duplicate |= meta.signal_sha256.duplicated(keep=False)
    keep = ~duplicate.to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    post, pre, nuisance = post[keep], pre[keep], nuisance[keep]
    ix, y = population(meta)
    table = meta.iloc[ix].reset_index(drop=True)
    counts = {c: int((y == c).sum()) for c in ['negative','positive']}
    if min(counts.values()) < 10:
        return [], [dict(reason='Fewer than 10 trials per class', encoded_class_counts=counts)]
    scores, skipped, specs = [], [], []
    for mode, column in [('sentence','sentence_key'), ('word','word_identity'), ('reader','subject')]:
        try:
            folds, spec = a.grouped_splits(y, table[column].to_numpy(dtype=str))
            if mode == 'word':
                folds, spec['purged_training_trials_per_fold'] = a.purge_overlapping_train(table, y, folds)
        except ValueError as error:
            skipped.append(dict(split=mode, reason=str(error)))
            continue
        specs.append(dict(split=mode, **spec))
        variants = [('post','logistic'), ('post','shrinkage_lda'), ('post','dummy')]
        if mode == 'sentence':
            variants += [('pre','logistic'), ('nuisance','logistic')]
        for feature, model in variants:
            case = out/'cases'/f'{mode}__{feature}__{model}'
            case.mkdir(parents=True, exist_ok=True)
            mark = case/'result.json'
            if mark.is_file():
                saved = json.loads(mark.read_text())
                if a.file_digest(case/'predictions.csv.gz') != saved['predictions_sha256']:
                    raise ValueError('Saved prediction checksum mismatch')
                score = saved['score']
                print(f'KEEP neutral/nonneutral {out.name}/{case.name}', flush=True)
            else:
                print(f'FIT neutral/nonneutral {out.name}/{case.name}: {len(ix)} trials', flush=True)
                X = {'post':post, 'pre':pre, 'nuisance':nuisance}[feature][ix]
                score, predictions, readers, confusion, diagnostic = a.evaluate(
                    table, X, y, ('negative','positive'), folds, model)
                score, predictions, readers, diagnostic = label_outputs(score, predictions, readers, diagnostic)
                predictions.to_csv(case/'predictions.csv.gz', index=False, compression='gzip')
                a.write_json(mark, dict(score=score, per_reader=readers, confusion=confusion.tolist(),
                                       same_word_diagnostic=diagnostic,
                                       predictions_sha256=a.file_digest(case/'predictions.csv.gz')))
            scores.append(dict(target='neutral_nonneutral', split=mode, feature=feature, model=model, **score))
    pd.DataFrame(scores).to_csv(out/'scores.csv', index=False)
    a.write_json(out/'split_specs.json', specs)
    a.write_json(out/'skipped.json', skipped)
    draw_confusions(out)
    return scores, skipped


def draw_confusions(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(3,3,figsize=(12,10))
    for ri, split in enumerate(['sentence','word','reader']):
        for ci, model in enumerate(['logistic','shrinkage_lda','dummy']):
            ax = axes[ri,ci]
            mark = out/'cases'/f'{split}__post__{model}'/'result.json'
            if not mark.is_file():
                ax.text(.5,.5,'Insufficient coverage',ha='center'); ax.axis('off'); continue
            saved = json.loads(mark.read_text())
            cm = np.asarray(saved['confusion'],float)
            cm /= cm.sum(axis=1,keepdims=True)
            ConfusionMatrixDisplay(cm,display_labels=['neutral','nonneutral']).plot(
                ax=ax,colorbar=False,values_format='.2f',cmap='Blues')
            ax.set_title(f'{split} / {model}\nBalanced accuracy {saved["score"]["balanced_accuracy"]:.3f}',fontsize=9)
    fig.suptitle(f'{out.parent.name}/{out.name}: neutral vs nonneutral (positive + negative)')
    fig.tight_layout(); fig.savefig(out/'confusions.png',dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root',type=Path,required=True)
    args = parser.parse_args()
    source = args.thesis_root/'Results/word_sentiment_eeg/classification_similarity'
    out = source/'neutral_nonneutral'
    archive = args.thesis_root/'Results/word_sentiment_eeg/trial_archive_125hz'
    if json.loads((source/'manifest.json').read_text()).get('status') != 'completed':
        raise ValueError('Complete the original classification notebook first')
    records, _ = a.load_completed(archive)
    # Preflight every reader before permitting the cache reader to run.
    for record in records:
        if not (source/'features'/record['dataset']/record['task']/record['subject']/'manifest.json').is_file():
            raise ValueError('A feature cache is missing; no EEG will be read')
    protocol = dict(code=a.file_digest(Path(__file__)), helper=a.file_digest(Path(a.__file__)),
                    source_protocol=a.file_digest(source/'protocol.json'),
                    caches=[a.file_digest(source/'features'/r['dataset']/r['task']/r['subject']/'manifest.json') for r in records],
                    target='neutral vs positive+negative', policy='word_agreement', neutral_downsampling=False)
    out.mkdir(parents=True,exist_ok=True)
    reconcile_protocol(out/'protocol.json',protocol)
    by_task = defaultdict(list); labels = []
    for i, record in enumerate(records,1):
        print(f'LOAD SAVED FEATURES {i}/{len(records)}: {record["dataset"]}/{record["task"]}/{record["subject"]}',flush=True)
        meta,post,pre,nuisance,label = cached_record(archive,record,source,args.thesis_root)
        by_task[(record['dataset'],record['task'])].append((meta,post,pre,nuisance))
        if label and label not in labels: labels.append(label)
    if json.loads((source/'label_sources.json').read_text()) != labels:
        raise ValueError('Annotation sources differ from the completed original analysis')
    scores, skips = [], []
    for (dataset,task), parts in sorted(by_task.items()):
        meta = pd.concat([p[0] for p in parts],ignore_index=True)
        if meta.event_id.duplicated().any(): raise ValueError('Duplicate event IDs')
        arrays = [np.concatenate([p[j] for p in parts]) for j in [1,2,3]]
        rows, missing = task_analysis(meta,*arrays,out/dataset/task)
        scores.extend(dict(dataset=dataset,task=task,**r) for r in rows)
        skips.extend(dict(dataset=dataset,task=task,**r) for r in missing)
        a.check_budget(out,60_000_000)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    a.write_json(out/'skipped.json',skips)
    a.write_json(out/'manifest.json',dict(status='completed',tasks_complete=['/'.join(k) for k in sorted(by_task)],cases=len(scores)))
    print('NEUTRAL/NONNEUTRAL COMPLETE:',out,flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
