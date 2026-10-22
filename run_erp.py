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
import tempfile
import fcntl
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import atomic_npz, digest, file_digest, sentence_key, write_json
from word_sentiment_eeg.erp import CLASSES, TIMES_MS, ROI, SFREQ, eligible, epoch, iter_sentences, locate, preprocess
from word_sentiment_eeg.erp_checkpoint import EpochChunks, MissingAnalysisArtifact, inspect_checkpoint, repair_trial_exports

# Recognize legacy checkpoints without tying scientific identity to logging/I/O.
LEGACY_RUNNER_SHA256 = 'b6d73b12a6a29e7ea7f972bfe63ac77dd247e53b6e739f7df1b3b5a663c2f572'


def check_output_space(root):
    """DriveFS may acknowledge cached writes before cloud uploads complete."""
    usage=shutil.disk_usage(root)
    if usage.free < 1024**3:
        raise OSError(
            f'Output storage has only {usage.free / 1024**3:.2f} GiB free. '
            'Stopping before more EEG writes. Free storage or choose a smaller '
            'output policy before resuming. Do not reset the runtime while '
            'Drive uploads may be pending. The 1 GiB check is minimum headroom, '
            'not an estimate of space needed for all trial exports.')
    return usage


def cache_fingerprints(annotations, source_manifest):
    science={str(f.relative_to(ROOT)):file_digest(f) for f in
             [ROOT/'src/word_sentiment_eeg/erp.py',ROOT/'src/word_sentiment_eeg/raw_export.py',ROOT/'assets/zuco_chanlocs.csv']}
    base={'labels':file_digest(annotations),'source_manifest':file_digest(source_manifest)}
    current=digest({**base,'epoch_protocol':'ZuCo1-first-fixation-v3-1','code':science})
    legacy=digest({**base,'code':{**science,'run_erp.py':LEGACY_RUNNER_SHA256}})
    return current,legacy


def save_run_state(run,items,expected,failures,fingerprint,stage):
    ready={(i['task'],i['subject']) for i in items}
    if len(ready)!=len(items):
        raise ValueError('Duplicate reader/task in run state; refusing double-counted results')
    complete=[i for i in items if i.get('trial_exports_complete',False)]
    done=len(items)==len(expected) and len(complete)==len(expected) and not failures
    state={'status':'completed' if done else 'incomplete','stage':stage,
           'updated_at_utc':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint,
           'expected_files':len(expected),'processed_files':len(items),
           'analysis_ready_files':len(items),'complete_trial_export_files':len(complete),
           'pending_new_readers':[f"{r['task']}/{r['subject']}" for r in expected if (r['task'],r['subject']) not in ready],
           'pending_trial_repairs':[f"{i['task']}/{i['subject']}" for i in items if not i.get('trial_exports_complete')],
           'files':items,'failures':failures}
    write_json(run/'manifest.json',state)
    return state


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
    writer=EpochChunks(output,channels)
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
                    entry['epoch_index_in_class_file'] = writer.add(cls,value,event_id)
            audit.append(entry)
        if n_sentences % 50 == 0:
            logging.info('%s/%s: %d sentences, accepted %s', task, subject, n_sentences, writer.counts)
    if not n_sentences:
        raise ValueError('Source contains no readable sentence/word records')
    means,counts=writer.finish()
    logging.info('Save %s/%s onset table and class averages',task,subject)
    atomic_npz(output / 'class_means.npz', mean_uV=means, counts=np.array(counts), classes=np.array(CLASSES),
               times_ms=TIMES_MS, channel_names=np.array(channels))
    frame = pd.DataFrame(audit)
    frame.to_csv(output / 'word_onsets.csv.gz', index=False, compression='gzip')
    item = {'task':task,'subject':subject,'fingerprint':fingerprint,'output_dir':str(output),
            'source_filename':path.name,'source_sha256':file_digest(path), 'source_bytes':path.stat().st_size,
            'sentences':n_sentences,'word_rows':len(frame),'epochs_per_class':dict(zip(CLASSES,counts)),
            'status_counts':{str(k):int(v) for k,v in frame.status.value_counts().items()},
            'status':'completed' if sum(counts) else 'no_usable_epochs',
            'analysis_ready':True,'trial_exports_complete':True,'missing_trial_classes':[],
            'missing_trial_chunks':0,'epoch_storage':'bounded_chunks_v2'}
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
    all_processed = len(items) == len(expected) and all(i.get('trial_exports_complete') for i in items) and not failures
    any_epochs = any(sum(i['epochs_per_class'].values()) for i in items)
    state = 'completed' if all_processed and any_epochs else 'incomplete'
    report = ['# Word sentiment: onsets and fixation-related ERPs', '', f'Run status: **{state}**.', '',
              'Uses the prepared single-agent labels. Uncertain words, lexical/context disagreement and alignment problems are automatically excluded.',
              'All three classes are saved separately for each reader and task. Onsets are first-fixation samples relative to sentence EEG (zero based), not absolute recording or sentence stimulus times.', '',
              'Epoch: −200 to +800 ms (501 samples at 500 Hz). Average reference; 4th-order 30 Hz zero-phase low-pass on the full sentence; −200..0 ms baseline; reject post-baseline |amplitude| >100 µV.',
              'Only the original first fixation is used; a rejected first fixation never becomes a later fixation. Every onset is verified by full-segment, all-channel matching.', '',
              '## Coverage', *coverage, '',
              f"Analyses/onset tables ready: {len(items)}/{len(expected)}. Complete trial EEG exports: {sum(i.get('trial_exports_complete',False) for i in items)}/{len(expected)}.",
              'Saved analyses remain counted during restarts. Missing trial exports are repaired from stored onsets, without repeating label assignment or onset matching.', '',
              'Grand averages use the same readers in all classes (≥5 epochs per class), with equal reader weights. Shading is ±1 reader SEM, not a significance test. Classes have unequal word counts.',
              'These are descriptive fixation-related potentials (FRPs). Adjacent fixations overlap in time; this run does not deconvolve overlap or establish a sentiment-specific neural response.', '',
              '## Files', '- `class_counts.csv`: accepted epochs and readers.',
              '- `class_erp_roi.csv`, `plots/*_class_erp.png`: class curves.',
              '- `*_grand_average.npz`: all 105 channels, per-reader and grand class means.',
              '- Each cached task/reader folder: `epochs_index.json`, per-class epoch chunks (or preserved legacy `*_epochs.npz` files), `word_onsets.csv.gz`, `class_means.npz`, `manifest.json`.',
              'The epoch index lists each class file and its global start/stop row. The word-onset table links each accepted event_id to that global class row and records every automatic exclusion.', '',
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
    # Check the mounted destination, not Colab's unrelated temporary disk.
    # Do this before opening log/report files or changing existing run metadata.
    check_output_space(args.thesis_root)
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
    # Prevent two extraction subprocesses from writing the same run in one runtime.
    lock_path=Path(tempfile.gettempdir())/('word-erp-'+digest([str(args.thesis_root.resolve()),args.run_tag])[:16]+'.lock')
    lock=lock_path.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another extraction process already owns this run in this runtime. Do not launch a second one.')
    fingerprint,legacy=cache_fingerprints(args.annotations,args.source_manifest)
    write_json(run/'provenance.json',{'fingerprint':fingerprint,'compatible_legacy_fingerprint':legacy,
               'runner_sha256':file_digest(Path(__file__)), 'annotation_sha256':file_digest(args.annotations),
               'label_method':'single agent review, not human gold','sfreq_hz':SFREQ,'epoch_ms':[-200,800],
               'onset_source':'match first word.rawEEG to sentenceData.rawData; exact or constant per-channel offset',
               'duration_units':'source MATLAB fixation durations are samples; exported milliseconds=(segment length-1)*2',
               'excluded_policy':'uncertain in either layer, lexical/context disagreement, any alignment warning',
               'epoch_storage':'bounded_chunks_v2; legacy single-file exports reused',
               'mean_accumulation':'float64 streaming sums; stored as float32; legacy means preserved',
               'roi_channels':ROI,'expected_sources':expected})
    cache_root=args.thesis_root/'CachedArtifacts/word_sentiment_eeg'/args.run_tag
    legacy_root=cache_root/legacy[:16]
    cache=legacy_root if legacy_root.exists() else cache_root/fingerprint[:16]
    source_roots=[args.raw_dir] if args.raw_dir else [args.thesis_root/'Data/zuco_og_raw',args.thesis_root/'Data/zuco_nr_raw',args.thesis_root/'Data/zuco_tsr_raw']
    existing={f.name:f for root in source_roots if root.exists() for f in root.rglob('results*_*.mat')}
    items,failures,plans=[],[],[]
    recoveries={}
    logging.info('RESUME AUDIT: checking all %d reader checkpoints before extraction',len(expected))
    for audit_index,record in enumerate(expected,1):
        target=cache/record['task']/record['subject']
        # If a current-format folder exists separately, inspect it first.
        current_target=cache_root/fingerprint[:16]/record['task']/record['subject']
        if (current_target/'manifest.json').exists():
            target=current_target
        recovery_target=cache_root/fingerprint[:16]/'recovered'/record['task']/record['subject']
        # A recovered analysis is the sole active checkpoint for this reader.
        # The original incomplete folder remains untouched for inspection.
        if (recovery_target/'manifest.json').exists():
            target=recovery_target
        try:
            logging.info('CHECK %d/%d saved checkpoint %s/%s',audit_index,len(expected),record['task'],record['subject'])
            cp=inspect_checkpoint(target,record,{fingerprint,legacy},channels)
        except MissingAnalysisArtifact as error:
            recoveries[(record['task'],record['subject'])]={
                'reader':record['task']+'/'+record['subject'],
                'original_dir':str(target),'output_dir':str(recovery_target),
                'source_sha256':error.item['source_sha256'],'reason':str(error)}
            plans.append((record,recovery_target,None))
            logging.warning('REBUILD %s/%s: %s; preserve original files and verify source hash before rebuilding',
                            record['task'],record['subject'],error)
            continue
        except Exception as error:
            logging.exception('Checkpoint validation failed %s/%s; preserving existing files',record['task'],record['subject'])
            failures.append({'task':record['task'],'subject':record['subject'],'reason':'checkpoint validation: '+str(error)})
            continue
        if cp:
            items.append(cp['item'])
            logging.info('KEEP %s/%s: onsets/averages already computed; trial export %s',
                         record['task'],record['subject'], 'complete' if cp['item']['trial_exports_complete']
                         else 'needs repair for '+','.join(cp['item']['missing_trial_classes']))
            if not cp['item']['trial_exports_complete']:
                plans.append((record,target,cp))
        else:
            plans.append((record,target,None))
    full=sum(i['trial_exports_complete'] for i in items)
    new=sum(cp is None and (r['task'],r['subject']) not in recoveries for r,_,cp in plans)
    logging.info('RESUME PLAN: %d/%d analyses already saved; %d full trial exports; %d need export-only repair; %d new readers; %d missing analyses to rebuild',
                 len(items),len(expected),full,len(items)-full,new,len(recoveries))
    write_json(run/'resume_plan.json',{'fingerprint':fingerprint,'analysis_ready':len(items),
        'full_trial_exports':full,'saved_reader_ids':[i['task']+'/'+i['subject'] for i in items],
        'new_readers':[r['task']+'/'+r['subject'] for r,_,cp in plans if cp is None and (r['task'],r['subject']) not in recoveries],
        'analysis_rebuilds':list(recoveries.values()),
        'trial_repairs':[{'reader':i['task']+'/'+i['subject'],'classes':i['missing_trial_classes'],
                         'chunks':i['missing_trial_chunks']} for i in items if not i['trial_exports_complete']],
        'validation_failures':failures})
    write_json(run/'recovery_plan.json',{'readers':list(recoveries.values()),
               'policy':'Keep valid readers. Rebuild compatible checkpoints missing analysis files in recovered/; preserve originals.'})
    save_run_state(run,items,expected,failures,fingerprint,'resume_audit_complete')
    # Advance new readers first. Missing trial exports are repaired afterwards,
    # using old onsets; saved readers never go through the onset matcher again.
    plans.sort(key=lambda plan: plan[2] is not None)
    for record,target,cp in plans:
        path=existing.get(record['filename']);downloaded=False
        recovery=recoveries.get((record['task'],record['subject']))
        action='repair_trial_exports' if cp else ('rebuild_missing_analysis' if recovery else 'extract_new_reader')
        stage=f"{action} {record['task']}/{record['subject']}"
        save_run_state(run,items,expected,failures,fingerprint,stage)
        logging.info('STAGE %s | analyses saved %d/%d',stage,len(items),len(expected))
        try:
            check_output_space(args.thesis_root)
            if path is None:
                if not args.allow_cloud_download:
                    raise FileNotFoundError('Source MATLAB file absent; use the Colab runner for cloud download')
                path=args.tmp_dir/record['filename']
                cloud_download(record,path);downloaded=True
            if cp:
                logging.info('Validate original source hash before export-only repair')
                updated=repair_trial_exports(path,cp,channels,file_digest(path))
                items=[updated if (i['task'],i['subject'])==(record['task'],record['subject']) else i for i in items]
            else:
                if recovery and file_digest(path)!=recovery['source_sha256']:
                    raise ValueError('Source changed since original checkpoint; refusing incompatible rebuild')
                item=process_file(path,record,annotations,target,fingerprint,channels)
                verified=inspect_checkpoint(target,record,{fingerprint,legacy},channels)
                if not verified or not verified['item']['trial_exports_complete']:
                    raise IOError('New reader export did not pass checkpoint verification')
                items.append(verified['item'])
            logging.info('SAVED: %d/%d analyses; %d complete trial exports',len(items),len(expected),sum(i['trial_exports_complete'] for i in items))
        except Exception as error:
            logging.exception('Failed %s/%s; saved analyses are retained',record['task'],record['subject'])
            failures.append({'task':record['task'],'subject':record['subject'],'reason':str(error)})
        finally:
            if downloaded:
                path.unlink(missing_ok=True)
        save_run_state(run,items,expected,failures,fingerprint,'reader_checkpoint_saved')
    logging.info('Build final class figures/averages once; no per-reader repeated plotting')
    summarize(items,run,channels,expected,failures,fingerprint)
    final=save_run_state(run,items,expected,failures,fingerprint,'finished')
    logging.info('Results: %s',run)
    lock.close()
    if final['status']!='completed':
        logging.error('INCOMPLETE: %d/%d analyses; %d/%d full trial exports. Failures:',
                      len(items),len(expected),sum(i['trial_exports_complete'] for i in items),len(expected))
        for failure in failures:
            logging.error('  %s/%s: %s',failure['task'],failure['subject'],failure['reason'])
        raise SystemExit('Run incomplete. Saved analyses are retained; see resume_plan.json/manifest.json for missing readers or trial exports.')


if __name__=='__main__':
    main()
