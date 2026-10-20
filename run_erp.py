"""Colab-first word EEG/onset/class ERP runner. No classifiers or new labeling."""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import atomic_npz, digest, file_digest, sentence_key, write_json
from word_sentiment_eeg.erp import CLASSES, TIMES_MS, ROI, SFREQ, eligible, epoch, iter_sentences, locate, preprocess


def cloud_download(record, target):
    if sys.platform == 'darwin':
        raise RuntimeError('EEG downloads on the Mac are disabled. Run this notebook in Colab.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < record['bytes'] * 1.2:
        raise OSError('Colab has insufficient temporary disk space for the next source file')
    partial = target.with_suffix('.part')
    for attempt in range(4):
        try:
            logging.info('Cloud download: %s (%.2f GB); deleted after extraction', record['filename'], record['bytes']/1e9)
            request = urllib.request.Request(record['url'], headers={'User-Agent':'word-sentiment-eeg research'})
            with urllib.request.urlopen(request, timeout=180) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, length=4*1024*1024)
            if partial.stat().st_size != record['bytes']:
                raise IOError(f"Source size mismatch: expected {record['bytes']}, got {partial.stat().st_size}")
            os.replace(partial, target)
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(min(15 * 2**attempt, 60))


def process_file(path, record, annotations, output, fingerprint, channels):
    task, subject = record['task'], record['subject']
    output.mkdir(parents=True, exist_ok=True)
    lookup = {(r.sentence_key, int(r.word_index)):r for r in annotations.itertuples() if task in r.tasks.split(';')}
    blocks, rowids = {c:[] for c in CLASSES}, {c:[] for c in CLASSES}
    audit = []
    n_sentences = 0
    for position, words, raw, per_word in iter_sentences(path):
        n_sentences += 1
        skey = sentence_key(words)
        usable_raw = raw is not None and raw.shape[1] > 20 and np.isfinite(raw).all()
        filtered = preprocess(raw) if usable_raw else None
        # Locate every fixation, including neutral/excluded words, to describe
        # later-fixation overlap. NEVER substitute a later valid fixation for first.
        matches = [[locate(raw, segment) if usable_raw else (None, 'missing_sentence_eeg')
                    for segment in fixation_segments] for fixation_segments in per_word]
        all_onsets = sorted({t for group in matches for t, _ in group if t is not None})
        for j, word in enumerate(words):
            label = lookup.get((skey, j))
            onset, method = matches[j][0] if matches[j] else (None, 'missing_first_fixation')
            event_id = digest([task, subject, position, skey, j])[:24]
            entry = {'event_id':event_id, 'task':task, 'subject':subject, 'raw_sentence_position':position,
                     'sentence_key':skey, 'word_index':j, 'word':word, 'sentence':' '.join(words),
                     'lexical_class':label.lexical_class if label else 'unmatched',
                     'contextual_class':label.contextual_class if label else 'unmatched',
                     'onset_sample_sentence_0based':onset, 'onset_ms_sentence':None if onset is None else onset*1000/SFREQ,
                     'onset_method':method, 'epoch_index_in_class_file':None,
                     'first_fixation_duration_ms':None, 'next_located_fixation_ms':None}
            if onset is not None:
                segment = per_word[j][0]
                entry['first_fixation_duration_ms'] = (segment.shape[1]-1)*1000/SFREQ
                later = [v for v in all_onsets if v > onset]
                entry['next_located_fixation_ms'] = (later[0]-onset)*1000/SFREQ if later else None
            if label is None or label.word != word:
                entry['status'] = 'excluded_text_alignment'
            elif not eligible(label):
                entry['status'] = 'excluded_uncertain_or_context_disagreement'
            elif onset is None:
                entry['status'] = 'excluded_' + method
            else:
                value, status = epoch(filtered, onset)
                entry['status'] = status
                if value is not None:
                    cls = label.contextual_class
                    entry['epoch_index_in_class_file'] = len(blocks[cls])
                    blocks[cls].append(value)
                    rowids[cls].append(event_id)
            audit.append(entry)
        if n_sentences % 50 == 0:
            logging.info('%s/%s: %d sentences, accepted %s', task, subject, n_sentences, {c:len(b) for c,b in blocks.items()})
    if not n_sentences:
        raise ValueError('Source contains no readable sentence/word records')
    means, counts = [], []
    for cls in CLASSES:
        values = np.stack(blocks[cls]) if blocks[cls] else np.empty((0,105,len(TIMES_MS)), np.float32)
        blocks[cls].clear()  # release individual arrays before compression
        means.append(values.mean(axis=0) if len(values) else np.full((105,len(TIMES_MS)), np.nan, np.float32))
        counts.append(len(values))
        atomic_npz(output / f'{cls}_epochs.npz', eeg_uV=values, event_id=np.array(rowids[cls], dtype='U24'),
                   times_ms=TIMES_MS, channel_names=np.array(channels), sfreq_hz=np.array(SFREQ))
        del values
    atomic_npz(output / 'class_means.npz', mean_uV=np.stack(means), counts=np.array(counts), classes=np.array(CLASSES),
               times_ms=TIMES_MS, channel_names=np.array(channels))
    frame = pd.DataFrame(audit)
    frame.to_csv(output / 'word_onsets.csv.gz', index=False, compression='gzip')
    item = {'task':task,'subject':subject,'fingerprint':fingerprint,'output_dir':str(output),
            'source_filename':path.name,'source_sha256':file_digest(path), 'source_bytes':path.stat().st_size,
            'sentences':n_sentences,'word_rows':len(frame),'epochs_per_class':dict(zip(CLASSES,counts)),
            'status_counts':{str(k):int(v) for k,v in frame.status.value_counts().items()},
            'status':'completed' if sum(counts) else 'no_usable_epochs'}
    write_json(output / 'manifest.json', item)
    return item


def summarize(items, run, channels, expected, failures, fingerprint):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roi_idx = [channels.index(c) for c in ROI]
    rows, roi_rows, coverage = [], [], []
    plotdir = run / 'plots'
    plotdir.mkdir(exist_ok=True)
    for task in sorted({r['task'] for r in expected}):
        subjects = [i for i in items if i['task'] == task]
        matched = [i for i in subjects if min(i['epochs_per_class'].values()) >= 5]
        coverage.append(f"- {task}: {len(subjects)} processed files; {len(matched)} readers with ≥5 epochs in every class.")
        for item in subjects:
            for cls, count in item['epochs_per_class'].items():
                rows.append({'task':task,'subject':item['subject'],'class':cls,'accepted_epochs':count,
                             'included_in_comparable_grand_average':item in matched})
        if not matched:
            continue
        array = np.stack([np.load(Path(i['output_dir'])/'class_means.npz', allow_pickle=False)['mean_uV'] for i in matched])
        grand = array.mean(axis=0)
        atomic_npz(run/f'{task}_grand_average.npz', mean_uV=grand, classes=np.array(CLASSES),
                   subject_means_uV=array, subjects=np.array([i['subject'] for i in matched]),
                   times_ms=TIMES_MS, channel_names=np.array(channels))
        fig, ax = plt.subplots(figsize=(9,5))
        for ci, cls in enumerate(CLASSES):
            curves = array[:,ci,roi_idx,:].mean(axis=1)
            mean = curves.mean(axis=0)
            sem = curves.std(axis=0, ddof=1)/np.sqrt(len(curves)) if len(curves)>1 else np.full(len(TIMES_MS),np.nan)
            line, = ax.plot(TIMES_MS,mean,label=cls)
            if len(curves)>1:
                ax.fill_between(TIMES_MS,mean-sem,mean+sem,color=line.get_color(),alpha=.15)
            for ti,t in enumerate(TIMES_MS):
                roi_rows.append({'task':task,'class':cls,'time_ms':t,'mean_uV':float(mean[ti]),
                                 'sem_across_readers_uV':float(sem[ti]),'n_readers':len(curves)})
        ax.axvline(0,color='black',lw=.8); ax.axhline(0,color='grey',lw=.6)
        ax.set(xlabel='Time from first fixation onset (ms)',ylabel='Amplitude (µV)',
               title=f'{task}: centro-parietal FRP; equal reader weights (n={len(matched)})')
        ax.legend(); fig.tight_layout(); fig.savefig(plotdir/f'{task}_class_erp.png',dpi=170); plt.close(fig)
    pd.DataFrame(rows).to_csv(run/'class_counts.csv',index=False)
    pd.DataFrame(roi_rows).to_csv(run/'class_erp_roi.csv',index=False)
    all_processed = len(items) == len(expected) and not failures
    any_epochs = any(sum(i['epochs_per_class'].values()) for i in items)
    state = 'completed' if all_processed and any_epochs else 'incomplete'
    report = ['# Word sentiment: onsets and fixation-related ERPs', '', f'Run status: **{state}**.', '',
              'Uses the prepared single-agent labels. Uncertain words, lexical/context disagreement and alignment problems are automatically excluded.',
              'All three classes are saved separately for each reader and task. Onsets are first-fixation samples relative to sentence EEG (zero based), not absolute recording or sentence stimulus times.', '',
              'Epoch: −200 to +800 ms (501 samples at 500 Hz). Average reference; 4th-order 30 Hz zero-phase low-pass on the full sentence; −200..0 ms baseline; reject post-baseline |amplitude| >100 µV.',
              'Only the original first fixation is used; a rejected first fixation never becomes a later fixation. Every onset is verified by full-segment, all-channel matching.', '',
              '## Coverage', *coverage, '',
              'Grand averages use the same readers in all classes (≥5 epochs per class), with equal reader weights. Shading is ±1 reader SEM, not a significance test. Classes have unequal word counts.',
              'These are descriptive fixation-related potentials (FRPs). Adjacent fixations overlap in time; this run does not deconvolve overlap or establish a sentiment-specific neural response.', '',
              '## Files', '- `class_counts.csv`: accepted epochs and readers.',
              '- `class_erp_roi.csv`, `plots/*_class_erp.png`: class curves.',
              '- `*_grand_average.npz`: all 105 channels, per-reader and grand class means.',
              '- Each cached task/reader folder: `negative_epochs.npz`, `neutral_epochs.npz`, `positive_epochs.npz`, `word_onsets.csv.gz`, `class_means.npz`, `manifest.json`.',
              'The word-onset table links each accepted event_id to its class epoch row and records every automatic exclusion.', '',
              f'Cached epochs: `{items[0]["output_dir"] if items else "no usable files"}`', '',
              '## Missing / failed files', *[f'- {f["task"]}/{f["subject"]}: {f["reason"]}' for f in failures]]
    (run/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(run/'manifest.json', {'status':state,'fingerprint':fingerprint,'expected_files':len(expected),
                                   'processed_files':len(items),'files':items,'failures':failures})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thesis-root', type=Path, required=True)
    p.add_argument('--run-tag', default='v3_word_erp')
    p.add_argument('--raw-dir', type=Path)
    p.add_argument('--allow-cloud-download', action='store_true')
    p.add_argument('--tasks', nargs='+', choices=['SR','NR','TSR'], default=['SR','NR','TSR'])
    p.add_argument('--subjects', nargs='+')
    p.add_argument('--annotations', type=Path, default=ROOT/'annotations/word_labels.csv')
    p.add_argument('--source-manifest', type=Path, default=ROOT/'configs/zuco1_sources.json')
    p.add_argument('--tmp-dir', type=Path, default=Path('/content/word-eeg-source-tmp'))
    args = p.parse_args()
    if Path(args.run_tag).name != args.run_tag:
        p.error('run-tag must be one folder name')
    run = args.thesis_root/'Results/word_sentiment_eeg'/args.run_tag
    run.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(run/'run.log')])
    annotations = pd.read_csv(args.annotations,keep_default_na=False)
    if annotations.duplicated(['sentence_key','word_index']).any():
        raise ValueError('Annotation keys are not unique')
    expected = [r for r in json.loads(args.source_manifest.read_text())['files'] if r['task'] in args.tasks and (not args.subjects or r['subject'] in args.subjects)]
    if not expected:
        raise ValueError('No source files selected')
    channels = pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.tolist()
    assert len(channels)==105 and len(set(channels))==105
    fingerprint = digest({'labels':file_digest(args.annotations),'code':{str(f.relative_to(ROOT)):file_digest(f) for f in [Path(__file__),ROOT/'src/word_sentiment_eeg/erp.py',ROOT/'src/word_sentiment_eeg/raw_export.py',ROOT/'assets/zuco_chanlocs.csv']},'source_manifest':file_digest(args.source_manifest)})
    write_json(run/'provenance.json',{'fingerprint':fingerprint,'annotation_sha256':file_digest(args.annotations),
               'label_method':'single agent review, not human gold','sfreq_hz':SFREQ,'epoch_ms':[-200,800],
               'onset_source':'match first word.rawEEG to sentenceData.rawData; exact or constant per-channel offset',
               'source_scripts':['https://osf.io/download/369ur/','https://osf.io/download/gan9v/','https://osf.io/download/mksxg/'],
               'duration_units':'source MATLAB fixation durations are samples; exported milliseconds=(segment length-1)*2',
               'excluded_policy':'uncertain in either layer, lexical/context disagreement, any alignment warning',
               'roi_channels':ROI,'expected_sources':expected})
    cache = args.thesis_root/'CachedArtifacts/word_sentiment_eeg'/args.run_tag/fingerprint[:16]
    source_roots = [args.raw_dir] if args.raw_dir else [args.thesis_root/'Data/zuco_og_raw',args.thesis_root/'Data/zuco_nr_raw',args.thesis_root/'Data/zuco_tsr_raw']
    existing = {f.name:f for root in source_roots if root.exists() for f in root.rglob('results*_*.mat')}
    items, failures = [], []
    for record in expected:
        target = cache/record['task']/record['subject']
        done = target/'manifest.json'
        if done.exists():
            item = json.loads(done.read_text())
            if item.get('fingerprint') == fingerprint and all((target/name).exists() for name in ['class_means.npz','word_onsets.csv.gz',*[f'{c}_epochs.npz' for c in CLASSES]]):
                logging.info('Reuse %s/%s',record['task'],record['subject']); items.append(item)
                summarize(items,run,channels,expected,failures,fingerprint)
                continue
        path = existing.get(record['filename'])
        downloaded = False
        try:
            if path is None:
                if not args.allow_cloud_download:
                    raise FileNotFoundError('Source MATLAB file absent; use the Colab runner for cloud download')
                path = args.tmp_dir/record['filename']
                cloud_download(record,path)
                downloaded = True
            logging.info('Extract %s/%s',record['task'],record['subject'])
            items.append(process_file(path,record,annotations,target,fingerprint,channels))
        except Exception as error:
            logging.exception('Failed %s/%s; other files continue',record['task'],record['subject'])
            failures.append({'task':record['task'],'subject':record['subject'],'reason':str(error)})
        finally:
            if downloaded:
                path.unlink(missing_ok=True)
        summarize(items,run,channels,expected,failures,fingerprint)
    logging.info('Results: %s',run)
    if failures:
        raise SystemExit('Some files failed. Completed readers are saved. Run all again to retry the missing files.')


if __name__=='__main__':
    main()
