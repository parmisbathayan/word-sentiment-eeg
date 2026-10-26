#!/usr/bin/env python3
"""Independent Colab-first full trial-data preparation for ZuCo + TeCo."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from word_sentiment_eeg.common import digest,file_digest,write_json
from word_sentiment_eeg.trial_archive import VERSION,CHUNK,prepare_reader
from word_sentiment_eeg.teco import find_raw_files
from run_three_level_erp import montage

RUNS={'zuco':'v4_word_erp_compact','teco':'teco_v1_word_erp_compact'}


def inventory(thesis,datasets):
    records=[]
    native,alternates=find_raw_files(thesis/'Data') if 'teco' in datasets else ({},[])
    sources=json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files']
    for dataset in datasets:
        run=thesis/'Results/word_sentiment_eeg'/RUNS[dataset]
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest.get('status')!='completed':raise ValueError('Historical run not complete: '+str(run))
        items=manifest['files'] if dataset=='zuco' else manifest['readers']
        expected=36 if dataset=='zuco' else 12
        pairs=[(i['task'],i['subject']) for i in items]
        if len(items)!=expected or len(set(pairs))!=expected:raise ValueError('Incomplete/duplicate historical reader roster')
        for i in items:
            task,subject=i['task'],i['subject']
            old=run/'readers'/task/subject if dataset=='zuco' else run/'readers'/subject
            if not (old/'word_onsets.csv.gz').is_file() or not (old/'manifest.json').is_file():raise FileNotFoundError(old)
            if dataset=='zuco':
                source=next(r for r in sources if (r['task'],r['subject'])==(task,subject))
                folder={'SR':'zuco_og_raw','NR':'zuco_nr_raw','TSR':'zuco_tsr_raw'}[task]
                path=thesis/'Data'/folder/source['filename'];n=int(i['word_rows']);size=int(i['source_bytes'])
            else:
                if subject not in native:raise FileNotFoundError(f'Native TeCo input missing for {subject}')
                path=native[subject];source=None;n=int(i['source_word_count']);size=int(i['source']['bytes'])
            channels,hz=(105,500) if dataset=='zuco' else (126,1200)
            events=pd.read_csv(old/'word_onsets.csv.gz',keep_default_na=False)
            if len(events)!=n:raise ValueError('Historical manifest/onset row mismatch')
            known=events.onset_sample_sentence_0based.astype(str).ne('') & ~events.status.eq('sentence_boundary')
            not_previously_searched=events.status.isin(['control_excluded','uncertain_or_context_disagreement','annotation_alignment_warning','annotation_not_reviewed']) if dataset=='teco' else pd.Series(False,index=events.index)
            candidates=int((known|not_previously_searched).sum())
            records.append(dict(dataset=dataset,task=task,subject=subject,path=str(path),historical_folder=str(old),
                                source_record=source,word_rows=n,candidate_epoch_upper_bound=candidates,source_bytes=size,
                                epoch_bytes=channels*(int(hz)+1)*4+channels*4))
    return records,alternates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',choices=RUNS,default=['zuco','teco'])
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--allow-cloud-download',action='store_true')
    parser.add_argument('--free-drive-bytes',type=int,help='Verified Google Drive account free bytes, supplied by Colab preflight')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if len(set(args.datasets))!=len(args.datasets):parser.error('Duplicate dataset')
    run=args.thesis_root/'Results/word_sentiment_eeg/trial_archive_40hz'
    records,alternates=inventory(args.thesis_root,args.datasets)
    plan={'version':VERSION,'reader_recordings':len(records),'datasets':args.datasets,
          'upper_bound_uncompressed_bytes':sum(r['candidate_epoch_upper_bound']*r['epoch_bytes'] for r in records),
          'note':'Upper bound uses saved recoverable onsets plus all previously unsearched TeCo label/control exclusions; same-window boundary failures cannot produce epochs. Actual output can be smaller; compression is not assumed.',
          'existing_output_bytes':sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) if run.exists() else 0,
          'source_files_missing_from_drive':[r['path'] for r in records if not Path(r['path']).exists()],
          'largest_source_bytes':max(r['source_bytes'] for r in records),'records':records}
    print(json.dumps({k:v for k,v in plan.items() if k!='records'},indent=2),flush=True)
    # Plan before source reads. No archive is marked complete by this step.
    run.mkdir(parents=True,exist_ok=True);write_json(run/'preflight.json',plan)
    write_json(run/'alternate_teco_copies.json',alternates)
    if args.plan_only:return
    if args.free_drive_bytes is None:raise SystemExit('Run the notebook Drive-quota preflight first. Full native trial EEG needs tens of GB.')
    # Conservative: budget for all words, without assuming compression or dropping classes.
    committed_upper=0
    for r in records:
        cp=run/r['dataset']/'readers'/r['task']/r['subject']/'checkpoint.json'
        if cp.is_file():
            state=json.loads(cp.read_text())
            committed_upper+=sum(b['epoch_count'] for b in state['blocks'])*r['epoch_bytes']
    remaining=max(0,plan['upper_bound_uncompressed_bytes']-committed_upper)
    if args.free_drive_bytes<remaining+2*1024**3:
        raise SystemExit(f'Insufficient verified Drive quota for conservative full-trial budget: need up to {(remaining+2*1024**3)/1024**3:.1f} GiB, free {args.free_drive_bytes/1024**3:.1f} GiB. No source extraction started. Do not delete original EEG or historical results.')
    if not Path('/content').is_dir():raise SystemExit('Full source preparation is intended for Colab, not this Mac.')
    lockpath=Path(tempfile.gettempdir())/('trial-archive-'+digest(str(run))[:16]+'.lock')
    lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    label_paths={'zuco':ROOT/'annotations/word_labels_three_level.csv',
                 'teco':args.thesis_root/'gpt uploaded files/word_sentiment_eeg/teco/word_labels_three_level.csv'}
    frames={};label_sha={}
    for d in args.datasets:
        path=label_paths[d]
        # Three-level labels can be joined later without re-reading EEG. Do not
        # block full signal preservation on a missing private revised CSV.
        frames[d]=pd.read_csv(path,keep_default_na=False) if path.is_file() else None
        label_sha[d]=file_digest(path) if path.is_file() else None
    protocol=dict(version=VERSION,lowpass_hz=40,filter='Butterworth order4 zero-phase, whole sentence',
                  epoch_ms=[-200,800],baseline_ms='[-200,0)',reference='average of all retained native EEG channels',
                  storage='all channels, native sampling, float32 EEG + per-channel baseline, bounded lossless NPZ chunks',
                  sample_rates_hz={'zuco':500,'teco':1200},channels={'zuco':105,'teco':126},chunk_trials=CHUNK,
                  source_population='all stored word occurrences; no label/grammar/control/negation selection',
                  fixation='first stored fixation only; no later substitution',
                  amplitude='>100 native assumed µV flagged, retained; filtering/rejection at classification time',
                  upstream_blink_screen='unknown; no new blink exclusion',duration='short first fixations flagged, not dropped',
                  annotations_sha256=label_sha,annotation_status='agent annotations, not human consensus gold',
                  source_units='ZuCo stored µV; TeCo native µV assumed and not independently calibrated',
                  onset_reuse='historical same-source SHA256 plus full segment validation; otherwise recover first stored segment',
                  code_sha256={n:file_digest(ROOT/n) for n in ['prepare_trial_archive.py','src/word_sentiment_eeg/trial_archive.py','src/word_sentiment_eeg/three_level_erp.py','src/word_sentiment_eeg/erp.py','src/word_sentiment_eeg/teco.py','src/word_sentiment_eeg/raw_export.py']})
    prior=run/'protocol.json'
    if prior.exists() and json.loads(prior.read_text())!=protocol:raise SystemExit('Archive settings/labels/code changed. Preserve existing archive; reconcile before resuming. Do not silently mix chunks.')
    write_json(prior,protocol)
    for d in args.datasets:
        montage(d).to_csv(run/f'{d}_channel_regions.csv',index=False)
        if label_paths[d].is_file():
            dest=run/'annotations'/f'{d}_word_labels.csv';dest.parent.mkdir(exist_ok=True)
            if not dest.exists():shutil.copyfile(label_paths[d],dest)
    items=[];failures=[]
    def state(stage,active=None):
        write_json(run/'manifest.json',{'status':'completed' if stage=='finished' and len(items)==len(records) and not failures else 'incomplete',
                   'stage':stage,'version':VERSION,'expected_reader_recordings':len(records),'completed_reader_recordings':len(items),
                   'readers':items,'failures':failures,'active':active,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    def progress(dataset,task,subject,position,words):
        state('preparing',dict(dataset=dataset,task=task,subject=subject,last_source_sentence=position,word_rows_seen=words))
    state('starting')
    for r in records:
        d,task,subject=r['dataset'],r['task'],r['subject'];path=Path(r['path']);temporary=False
        out=run/d/'readers'/task/subject
        try:
            cp=out/'checkpoint.json'
            already_complete=cp.is_file() and json.loads(cp.read_text()).get('status')=='completed'
            if not path.is_file() and not already_complete:
                if d!='zuco' or not args.allow_cloud_download:raise FileNotFoundError(path)
                if not Path('/content').is_dir():raise RuntimeError('Raw-source downloads are cloud-only')
                from run_erp import cloud_download
                path=Path('/content/word-trial-archive-source')/r['source_record']['filename']
                cloud_download(r['source_record'],path);temporary=True
            result=prepare_reader(path,d,task,subject,Path(r['historical_folder']),frames[d],out,protocol,montage(d).channel.tolist(),progress)
            items.append(dict(dataset=d,task=task,subject=subject,output_dir=str(out),epochs=result['epochs'],
                              source_word_rows=result['source_word_rows'],outcomes=result['outcomes'],source_sha256=result['identity']['source_sha256']))
        except Exception as error:
            failures.append(dict(dataset=d,task=task,subject=subject,error=str(error)))
            print(f'FAILED {d}/{task}/{subject}: {error}',flush=True)
            state('stopped_on_error');raise
        finally:
            if temporary:path.unlink(missing_ok=True)
        state('preparing')
    counts=pd.DataFrame(items);counts.to_csv(run/'reader_coverage.csv',index=False)
    state('finished')
    (run/'README.md').write_text('''# Reusable full word-trial archive at 40 Hz

This is data preparation only. No classifier or correlation result has run.
All finite, in-bounds first-stored-fixation epochs are archived with all native
channels and time samples; no averaging across trials, channels or time bins.
ZuCo: 105 × 501 samples at 500 Hz. TeCo task1: 126 × 1201 at 1200 Hz.
Average reference and whole-sentence 40-Hz low-pass precede epoching.
Epoch −200…800 ms; baseline −200…0 excluding zero. Per-channel baseline is
stored separately, allowing reversal of baseline subtraction at this filter.

Every source word has metadata, including unsuccessful events. Sentiment
classes, function words, controls, uncertain and negated words are retained.
Missing revised labels can be joined by exact source identity later, without
repeating EEG preparation. Amplitude >100 µV and <100-ms duration are flags,
not archive exclusions. Later analysis must choose and report a fixed policy.
No blink/predecessor screen or overlap deconvolution is claimed.

Each reader has events.csv.gz plus blocks/block-N/events.csv.gz and bounded
epochs-N.npz files. event_id, epoch_file and epoch_row give a one-to-one join.
NPZ keys: eeg_uV [trial,channel,time], baseline_uV [trial,channel], event_id,
time_ms (stored as times_ms) and channel_names. checkpoint.json lists hashes,
counts and the next source sentence. Committed chunks are never appended twice.
Rerunning verifies committed chunks and resumes remaining source sentences.
A completed manifest requires every expected reader, not merely an ended cell.

Historical 30-Hz averages and original source files stay unchanged. These new
40-Hz signals are a different preprocessing condition, not the old trial data.
They support many later time-domain and within-band spectral features, but a
new epoch window, frequencies removed by upstream/40-Hz filtering, later
fixations or continuous-data deconvolution can still require original sources.
This archive is not a promise that all future EEG work can avoid source reads.
''')
    print(f'ARCHIVE COMPLETE: {run}; {sum(i["epochs"] for i in items)} full individual trials',flush=True)


if __name__=='__main__':main()
