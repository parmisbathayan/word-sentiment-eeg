"""Save word onsets and class ERPs without archiving individual trial arrays."""
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import run_erp as full
from word_sentiment_eeg.common import digest, file_digest, write_json
from word_sentiment_eeg.erp_checkpoint import MissingAnalysisArtifact, inspect_checkpoint


def copy_small(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary=target.with_name(target.name+'.copying')
    shutil.copyfile(source,temporary)
    if file_digest(source)!=file_digest(temporary):
        raise OSError(f'Copy verification failed: {target}')
    os.replace(temporary,target)


def portable_bundle(run):
    """One small file for later inspection; never include old EEG trial caches."""
    files=[p for p in run.rglob('*') if p.is_file() and
           (p.name in {'class_means.npz','word_onsets.csv.gz','manifest.json',
                       'REPORT.md','class_counts.csv','class_erp_roi.csv',
                       'provenance.json','word_labels.csv','zuco_chanlocs.csv'}
            or p.name.endswith('_grand_average.npz') or p.name.endswith('_class_erp.png'))]
    temporary=run/'compact_results.tmp.zip'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(files):
            archive.write(path,path.relative_to(run))
    os.replace(temporary,run/'compact_results.zip')
    logging.info('Portable results bundle: %.1f MB', (run/'compact_results.zip').stat().st_size/1e6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--run-tag',default='v4_word_erp_compact')
    p.add_argument('--reuse-run-tag',default='v3_word_erp')
    p.add_argument('--annotations',type=Path,default=full.ROOT/'annotations/word_labels.csv')
    p.add_argument('--source-manifest',type=Path,default=full.ROOT/'configs/zuco1_sources.json')
    p.add_argument('--raw-dir',type=Path)
    p.add_argument('--allow-cloud-download',action='store_true')
    p.add_argument('--tmp-dir',type=Path,default=Path('/content/word-eeg-source-tmp'))
    args=p.parse_args()
    if any(Path(tag).name!=tag for tag in (args.run_tag,args.reuse_run_tag)):
        p.error('Run tags must be single folder names')
    full.check_output_space(args.thesis_root)
    run=args.thesis_root/'Results/word_sentiment_eeg'/args.run_tag
    run.mkdir(parents=True,exist_ok=True)
    lock= (Path(tempfile.gettempdir())/('compact-erp-'+digest(str(run.resolve()))[:16]+'.lock')).open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another compact extraction is already running for this output folder.')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(run/'run.log')])
    labels=full.pd.read_csv(args.annotations,keep_default_na=False)
    if labels.duplicated(['sentence_key','word_index']).any():
        raise ValueError('Annotation keys are not unique')
    records=json.loads(args.source_manifest.read_text())['files']
    if not records or len({(r['task'],r['subject']) for r in records})!=len(records):
        raise ValueError('Source manifest must contain unique reader/task records')
    channels=full.pd.read_csv(full.ROOT/'assets/zuco_chanlocs.csv').labels.tolist()
    fingerprint,legacy=full.cache_fingerprints(args.annotations,args.source_manifest)
    old=args.thesis_root/'CachedArtifacts/word_sentiment_eeg'/args.reuse_run_tag
    roots=[args.raw_dir] if args.raw_dir else [args.thesis_root/'Data'/n for n in ('zuco_og_raw','zuco_nr_raw','zuco_tsr_raw')]
    existing={f.name:f for root in roots if root.exists() for f in root.rglob('results*_*.mat')}
    write_json(run/'provenance.json',{'fingerprint':fingerprint,'compatible_legacy_fingerprint':legacy,
        'output_policy':'onsets_and_class_averages','individual_trial_arrays_saved':False,
        'runner_sha256':file_digest(Path(__file__)),'annotation_sha256':file_digest(args.annotations),
        'source_inventory':records,'epoch_ms':[-200,800],'sfreq_hz':500,
        'regeneration':'Use saved onset samples, source SHA-256, and the unchanged epoch/preprocessing code.',
        'original_caches_modified':False})
    copy_small(args.annotations,run/'word_labels.csv')
    copy_small(full.ROOT/'assets/zuco_chanlocs.csv',run/'zuco_chanlocs.csv')
    items,failures=[],[]
    for record in records:
        task,subject=record['task'],record['subject']
        target=run/'readers'/task/subject
        downloaded=False;path=None
        logging.info('CHECK %s/%s | %d/%d analyses ready',task,subject,len(items),len(records))
        try:
            full.check_output_space(args.thesis_root)
            candidates=[target,old/fingerprint[:16]/'recovered'/task/subject,
                        old/fingerprint[:16]/task/subject,old/legacy[:16]/task/subject]
            saved=None;source_hash=None
            for candidate in candidates:
                try:
                    saved=inspect_checkpoint(candidate,record,{fingerprint,legacy},channels,check_trials=False)
                except MissingAnalysisArtifact as error:
                    source_hash=error.item['source_sha256']
                    logging.info('Incomplete analysis %s: %s',candidate,error)
                    continue
                if saved:
                    break
            if saved:
                origin=Path(saved['item']['output_dir'])
                item=dict(saved['item'])
                if origin!=target:
                    for name in ('class_means.npz','word_onsets.csv.gz'):
                        copy_small(origin/name,target/name)
                    logging.info('KEEP %s/%s: copied verified onsets/averages; no trial arrays copied',task,subject)
                else:
                    logging.info('KEEP %s/%s: compact checkpoint already saved',task,subject)
                item.update(output_dir=str(target),epoch_storage='averages_only',
                            trial_exports_complete=False,trial_export_validation='not_requested',
                            missing_trial_classes=[],missing_trial_chunks=0)
                if origin!=target:
                    item['reused_from']=str(origin)
                write_json(target/'manifest.json',item)
            else:
                path=existing.get(record['filename'])
                if path is None:
                    if not args.allow_cloud_download:
                        raise FileNotFoundError('Source absent; enable cloud download in Colab')
                    path=args.tmp_dir/record['filename']
                    full.cloud_download(record,path);downloaded=True
                if source_hash and file_digest(path)!=source_hash:
                    raise ValueError('Source changed since the original checkpoint; refusing rebuild')
                logging.info('EXTRACT %s/%s: calculate epochs in memory; save onsets/averages only',task,subject)
                item=full.process_file(path,record,labels,target,fingerprint,channels,save_trials=False)
            verified=inspect_checkpoint(target,record,{fingerprint,legacy},channels,check_trials=False)
            if not verified:
                raise OSError('Compact checkpoint was not readable after saving')
            items.append(verified['item'])
            logging.info('SAVED %d/%d unique reader/task analyses',len(items),len(records))
        except Exception as error:
            logging.exception('FAILED %s/%s; existing results retained',task,subject)
            failures.append({'task':task,'subject':subject,'reason':str(error)})
        finally:
            if downloaded:
                path.unlink(missing_ok=True)
        full.save_run_state(run,items,records,failures,fingerprint,'reader_saved',require_trials=False)
    full.summarize(items,run,channels,records,failures,fingerprint,require_trials=False)
    state=full.save_run_state(run,items,records,failures,fingerprint,'finished',require_trials=False)
    portable_bundle(run)
    logging.info('Results: %s | %s | %d/%d analyses',run,state['status'],len(items),len(records))
    if state['status']!='completed':
        raise SystemExit('Compact run incomplete. See the failures above; valid reader results were retained.')


if __name__=='__main__':
    main()
