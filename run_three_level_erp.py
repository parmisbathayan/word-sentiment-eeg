#!/usr/bin/env python3
"""Colab-first 40-Hz reconstruction; content words, four explicit label policies."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import digest,file_digest,write_json
from word_sentiment_eeg.three_level_erp import (CLASSES,EXPERIMENTS,REGIONS,VERSION,
                                               load_annotations,process_reader,resume)
from word_sentiment_eeg.teco import find_raw_files


def montage(dataset):
    if dataset=='teco':
        from run_teco_erp import load_montage
        frame=load_montage(ROOT/'assets/teco_montage.txt')
    else:
        frame=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').rename(columns={'labels':'channel'})
        frame['region']=np.select([frame.X>=3,frame.X<=-6,frame.X<=-3,frame.Y>5,frame.Y< -5],
                                 ['anterior','occipital','posterior','left_lateral','right_lateral'],default='central')
    return frame


def outputs(run,items,experiments,montage_frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];status_rows=[];coverage=[]
    for item in items:
        for experiment,spec in item['experiments'].items():
            for c in CLASSES:
                rows.append(dict(task=item['task'],subject=item['subject'],experiment=experiment,class_label=c,
                                 candidate_word_events=spec['candidates'][c],accepted_trials=spec['accepted'][c]))
            for status,n in spec['statuses'].items():
                status_rows.append(dict(task=item['task'],subject=item['subject'],experiment=experiment,status=status,count=n))
    pd.DataFrame(rows).to_csv(run/'candidate_and_accepted_counts.csv',index=False)
    pd.DataFrame(status_rows).to_csv(run/'terminal_status_counts.csv',index=False)
    plots=run/'plots';plots.mkdir(exist_ok=True)
    picks=[np.arange(len(montage_frame))]+[np.flatnonzero(montage_frame.region.to_numpy()==r) for r in REGIONS]
    titles=['Whole-head signed mean',*REGIONS]
    for task in sorted({i['task'] for i in items}):
        for ei,experiment in enumerate(experiments):
            task_items=[i for i in items if i['task']==task]
            roster=[i for i in task_items if all(i['experiments'][experiment]['accepted'][c]>=5 for c in CLASSES)]
            coverage.append(dict(task=task,experiment=experiment,readers=[i['subject'] for i in roster],minimum_per_class=5))
            # Never fill absent class means with zero and call them data.
            if len(roster)>=2:
                values=[]
                for i in roster:
                    with np.load(Path(i['output_dir'])/'class_means.npz') as data:
                        values.append(data['means'][ei]);times=data['times_ms']
                values=np.stack(values)
                fig,axes=plt.subplots(2,4,figsize=(20,8),sharex=True)
                for p,(title,indices) in enumerate(zip(titles,picks)):
                    ax=axes.flat[p]
                    if not len(indices):ax.set_visible(False);continue
                    curves=values[:,:,indices,:].mean(axis=2)
                    for ci,c in enumerate(CLASSES):
                        mean=curves[:,ci].mean(axis=0);sem=curves[:,ci].std(axis=0,ddof=1)/np.sqrt(len(roster))
                        line=ax.plot(times,mean,label=c)[0];ax.fill_between(times,mean-sem,mean+sem,color=line.get_color(),alpha=.18)
                    ax.set_title(title);ax.axvline(0,color='gray',lw=.6);ax.set_xlabel('First stored fixation onset (ms)');ax.set_ylabel('Voltage (assumed µV)')
                axes.flat[0].legend();axes.flat[7].axis('off')
                fig.suptitle(f'{task} / {experiment}: equal-reader mean ±1 SEM, n={len(roster)}; 40 Hz')
                fig.tight_layout();fig.savefig(plots/f'{task}_{experiment}_group.png',dpi=130);plt.close(fig)
            for item in task_items:
                with np.load(Path(item['output_dir'])/'class_means.npz') as data:
                    values=data['means'][ei];times=data['times_ms']
                fig,axes=plt.subplots(1,7,figsize=(30,3.5),sharex=True)
                for ax,title,indices in zip(axes,titles,picks):
                    for ci,c in enumerate(CLASSES):
                        n=item['experiments'][experiment]['accepted'][c]
                        if n and len(indices):ax.plot(times,values[ci,indices].mean(axis=0),label=f'{c} n={n}')
                    ax.set_title(title);ax.axvline(0,color='gray',lw=.5)
                axes[0].legend(fontsize=7);fig.suptitle(f'{task}/{item["subject"]} / {experiment}: 40 Hz')
                fig.tight_layout();fig.savefig(plots/f'{task}_{item["subject"]}_{experiment}_row.png',dpi=110);plt.close(fig)
    write_json(run/'comparison_rosters.json',coverage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['zuco','teco'],required=True)
    p.add_argument('--thesis-root',type=Path,required=True)
    p.add_argument('--annotations',type=Path)
    p.add_argument('--run-tag')
    p.add_argument('--experiments',nargs='+',choices=EXPERIMENTS,default=['all_three'])
    p.add_argument('--tasks',nargs='+',default=['SR','NR','TSR'])
    p.add_argument('--subjects',nargs='+')
    p.add_argument('--allow-cloud-download',action='store_true')
    p.add_argument('--save-event-features',action='store_true')
    args=p.parse_args()
    if len(set(args.experiments))!=len(args.experiments):p.error('Duplicate experiment')
    annotations=args.annotations or (ROOT/'annotations/word_labels_three_level.csv' if args.dataset=='zuco' else args.thesis_root/'gpt uploaded files/word_sentiment_eeg/teco/word_labels_three_level.csv')
    labels=load_annotations(annotations,args.dataset)
    tag=args.run_tag or f'{args.dataset}_v2_three_level_40hz'
    if Path(tag).name!=tag or tag in {'.','..'}:p.error('run-tag must be a single folder name')
    run=args.thesis_root/'Results/word_sentiment_eeg'/tag
    run.mkdir(parents=True,exist_ok=True)
    m=montage(args.dataset);channels=m.channel.tolist();regions=m.region.tolist()
    protocol=dict(version=VERSION,dataset=args.dataset,experiments=args.experiments,
                  annotations_sha256=file_digest(annotations),lowpass_hz=40,filter='Butterworth4 zero-phase whole sentence',
                  reference='all native channels average',epoch_ms=[-200,800],baseline_ms='[-200,0)',
                  amplitude_limit_native_assumed_uV=100,grammatical_policy='content only',controls='reading included; answer_data never used',
                  fixation='first stored slot; never promote later fixation',minimum_duration_filter='not added',blink_filter='not added',
                  class_downsampling=False,features=args.save_event_features,tasks=args.tasks if args.dataset=='zuco' else ['task1'],
                  subjects=args.subjects,channel_regions=list(zip(channels,regions)),
                  code={n:file_digest(ROOT/n) for n in ['run_three_level_erp.py','src/word_sentiment_eeg/three_level_erp.py','src/word_sentiment_eeg/erp.py','src/word_sentiment_eeg/teco.py']})
    fingerprint=digest(protocol)
    existing=run/'protocol.json'
    if existing.exists() and json.loads(existing.read_text())!=protocol:
        raise SystemExit('Configuration/labels/code changed: use a new --run-tag; completed runs are preserved.')
    write_json(existing,protocol);m.to_csv(run/'channel_regions.csv',index=False)
    data=args.thesis_root/'Data'
    if args.dataset=='zuco':
        records=json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files']
        records=[r for r in records if r['task'] in args.tasks and (not args.subjects or r['subject'] in args.subjects)]
        for r in records:
            folder={'SR':'zuco_og_raw','NR':'zuco_nr_raw','TSR':'zuco_tsr_raw'}[r['task']]
            r['path']=data/folder/r['filename']
    else:
        native,alternates=find_raw_files(data)
        records=[dict(subject=s,task='task1',path=path) for s,path in sorted(native.items()) if not args.subjects or s in args.subjects]
        write_json(run/'alternate_source_copies.json',alternates)
        if not args.subjects and len(records)!=12:raise SystemExit(f'Expected 12 native task1 readers, found {len(records)}; no other TeCo task is silently substituted.')
    if not records:raise SystemExit('No source readers selected')
    items=[];failures=[]
    def state(stage):
        write_json(run/'manifest.json',dict(dataset=args.dataset,status='completed' if stage=='finished' and len(items)==len(records) and not failures else 'incomplete',
                   stage=stage,fingerprint=fingerprint,expected_readings=len(records),readers=items,failures=failures))
        write_json(run/'progress.json',dict(stage=stage,readers_complete=len(items),readers_expected=len(records),failures=len(failures)))
    state('starting')
    for record in records:
        task,subject=record['task'],record['subject'];path=record['path'];temporary=False
        out=run/'readers'/task/subject
        try:
            item=resume(out,fingerprint,path)
            if item is None:
                if not path.exists():
                    if args.dataset!='zuco' or not args.allow_cloud_download:raise FileNotFoundError(path)
                    # Reuses the existing cloud-only downloader; never writes raw EEG to Drive.
                    from run_erp import cloud_download
                    path=Path('/content/three_level_source')/record['filename']
                    cloud_download(record,path);temporary=True
                item=process_reader(path,args.dataset,task,subject,labels,out,fingerprint,channels,regions,args.experiments,args.save_event_features)
            else:print(f'KEEP {task}/{subject}: validated 40-Hz checkpoint',flush=True)
            items.append(item)
        except Exception as error:
            failures.append(dict(task=task,subject=subject,error=str(error)));print(f'FAILED {task}/{subject}: {error}',flush=True)
        finally:
            if temporary:path.unlink(missing_ok=True)
        state(f'{task}/{subject}')
    outputs(run,items,args.experiments,m) if items else None
    state('finished')
    write_json(run/'upstream_qc_unknown.json',dict(blink_predecessor_exclusion_count=None,outside_word_exclusion_count=None,under100ms_exclusion_count=None,
               reason='Word-level starting population; upstream missing-event counts not measured, not zero.'))
    (run/'REPORT.md').write_text(f'''# Three-level content-word FRPs at 40 Hz

Dataset: {args.dataset}; experiments: {args.experiments}; completed reader/task recordings: {len(items)}/{len(records)}.
TeCo is task1 only in this adapter. All control reading periods are eligible; answer-period EEG is never used.
Main rule requires lexical, contextual-word and sentence labels all agree. Optional single-layer rules are independent of the other labels.
Function/uncertain grammatical categories are excluded; no neutral downsampling. See protocol.json for all methods.
New source EEG epochs: average reference, whole-sentence zero-phase 40-Hz low-pass, −200…800 ms inclusive, baseline −200…0 exclusive, absolute amplitude <=100 native assumed µV.
First stored fixation only; no promotion of later valid fixations. No new blanket negation/blink/<100-ms exclusion has been invented.
Counts and terminal exclusions are saved separately for each experiment; these are overlapping experiment populations and must not be summed as distinct trials.
Saved reader-word onsets, all-channel class means, optional event-window region features; no huge trial-EEG archive.
Group plots require the same readers with >=5 epochs in each class and at least two readers; otherwise report coverage only. Individual plots omit absent classes.
These overlapping FRPs do not include deconvolution. Scalp regions are descriptive sensor summaries. Whole-head signed means approach zero under average referencing.
Old 30-Hz runs remain separate. Classification and repeated-word split/correlation analyses are future work, not results of this run.
''')
    if failures:raise SystemExit('Incomplete; valid checkpoints retained. See manifest.json.')
    print(f'Results: {run}',flush=True)


if __name__=='__main__':main()
