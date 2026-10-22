"""Audit saved compact results and plot individual readers; never extract EEG."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.common import digest, file_digest, write_json
from word_sentiment_eeg.erp import CLASSES, TIMES_MS, ROI, eligible


def require(condition,message):
    if not condition:
        raise ValueError(message)


def timing_summary(run):
    path=run/'run.log'
    if not path.exists():
        return {}
    events=[]
    for line in path.read_text().splitlines():
        match=re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) (.*)',line)
        if match:
            events.append((datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S,%f'),match[2]))
    # Describe the latest complete pass only, rather than adding old reruns.
    starts=[i for i,(_,m) in enumerate(events) if m.startswith('CHECK ') and '| 0/' in m]
    if starts:
        events=events[starts[-1]:]
    if not events:
        return {}
    download_start=extract_start=None
    downloads=extraction=0.;reused=extracted=attempts=0
    for stamp,message in events:
        if message.startswith('CHECK '):
            download_start=extract_start=None
        elif message.startswith('Cloud download:'):
            attempts+=1
            if download_start is None:
                download_start=stamp
        elif message.startswith('EXTRACT '):
            extract_start=stamp;extracted+=1
            if download_start is not None:
                downloads+=(stamp-download_start).total_seconds()
        elif message.startswith('KEEP '):
            reused+=1
        elif message.startswith('SAVED ') and extract_start is not None:
            extraction+=(stamp-extract_start).total_seconds();extract_start=None
    return {'total_seconds':(events[-1][0]-events[0][0]).total_seconds(),
            'reused_readers':reused,'extracted_readers':extracted,'download_attempts':attempts,
            'download_and_retry_seconds':downloads,'extraction_and_save_seconds':extraction,
            'note':'Approximate phases from logged transitions, including I/O; not a CPU profiler.'}


def audit(run,make_plots=True):
    began=time.monotonic()
    manifest=json.loads((run/'manifest.json').read_text())
    provenance=json.loads((run/'provenance.json').read_text())
    expected={(r['task'],r['subject']) for r in provenance['source_inventory']}
    items=manifest['files'];pairs=[(i['task'],i['subject']) for i in items]
    require(manifest['status']=='completed' and not manifest['failures'],'Run is not recorded as complete')
    require(len(pairs)==len(set(pairs)) and set(pairs)==expected,'Missing/duplicate reader-task records')
    require(file_digest(run/'word_labels.csv')==provenance['annotation_sha256'],'Annotation checksum mismatch')
    channels=pd.read_csv(run/'zuco_chanlocs.csv').labels.tolist()
    require(len(channels)==105 and len(set(channels))==105,'Invalid channel montage')
    roi=[channels.index(c) for c in ROI]
    label_frame=pd.read_csv(run/'word_labels.csv',keep_default_na=False)
    labels={(r.sentence_key,int(r.word_index)):r for r in label_frame.itertuples()}
    require(len(labels)==len(label_frame),'Duplicate annotation keys')
    count_table=pd.read_csv(run/'class_counts.csv')
    require(not count_table.duplicated(['task','subject','class']).any(),'Duplicate count-table keys')
    required=[run/n for n in ('manifest.json','provenance.json','word_labels.csv','zuco_chanlocs.csv',
                              'class_counts.csv','class_erp_roi.csv','REPORT.md')]
    means_by_pair={};curves={};rows=[];seen=set();accepted_total=word_rows=0
    max_baseline=0.
    for item in items:
        task,subject=item['task'],item['subject'];pair=(task,subject)
        target=run/'readers'/task/subject
        print(f'AUDIT {task}/{subject}',flush=True)
        local=json.loads((target/'manifest.json').read_text())
        for key in ('task','subject','fingerprint','source_sha256','word_rows','epochs_per_class'):
            require(local[key]==item[key],f'{pair}: reader manifest differs for {key}')
        require(local['fingerprint'] in {provenance['fingerprint'],provenance['compatible_legacy_fingerprint']},f'{pair}: incompatible fingerprint')
        with np.load(target/'class_means.npz',allow_pickle=False) as data:
            means=data['mean_uV'].copy();counts=data['counts'].copy()
            require(means.shape==(3,105,501),f'{pair}: mean shape')
            require(np.array_equal(data['classes'],CLASSES),f'{pair}: class order')
            require(np.array_equal(data['channel_names'],channels),f'{pair}: channel order')
            require(np.array_equal(data['times_ms'],TIMES_MS),f'{pair}: time axis')
        require(np.array_equal(counts,[item['epochs_per_class'][c] for c in CLASSES]),f'{pair}: mean counts')
        require(np.isfinite(means).all(),f'{pair}: nonfinite means')
        baseline=float(np.max(np.abs(means[:,:,:100].mean(axis=-1))))
        require(baseline<1e-4,f'{pair}: baseline mean is inconsistent with preprocessing')
        max_baseline=max(max_baseline,baseline)
        events=pd.read_csv(target/'word_onsets.csv.gz',keep_default_na=False,dtype={'sentence_key':str,'event_id':str})
        require(len(events)==item['word_rows'],f'{pair}: onset row count')
        require((events.task==task).all() and (events.subject==subject).all(),f'{pair}: onset identity')
        require(not events.event_id.duplicated().any(),f'{pair}: duplicate event IDs')
        ids=set(events.event_id)
        require(not seen.intersection(ids),f'{pair}: repeated event IDs across recordings')
        seen.update(ids);word_rows+=len(events)
        for row in events.itertuples():
            require(row.event_id==digest([task,subject,int(row.raw_sentence_position),row.sentence_key,int(row.word_index)])[:24],
                    f'{pair}: event identity does not match source positions')
            if row.status=='accepted':
                label=labels.get((row.sentence_key,int(row.word_index)))
                require(label is not None and eligible(label) and label.word==row.word
                        and label.contextual_class==row.contextual_class and task in label.tasks.split(';'),
                        f'{pair}: accepted event disagrees with annotation policy')
        accepted=events[events.status=='accepted'];accepted_total+=len(accepted)
        for ci,cls in enumerate(CLASSES):
            cls_rows=accepted[accepted.contextual_class==cls]
            require(len(cls_rows)==int(counts[ci]),f'{pair}: accepted count {cls}')
            require(np.array_equal(cls_rows.epoch_index_in_class_file.astype(float),np.arange(len(cls_rows))),
                    f'{pair}: repeated/missing trial order {cls}')
            tab=count_table[(count_table.task==task)&(count_table.subject==subject)&(count_table['class']==cls)]
            require(len(tab)==1 and int(tab.iloc[0].accepted_epochs)==int(counts[ci]),f'{pair}: count CSV mismatch')
        means_by_pair[pair]=means
        curves[pair]=means[:,roi,:].mean(axis=1)
        for ci,cls in enumerate(CLASSES):
            for lo,hi in ((300,500),(500,800)):
                value=float(curves[pair][ci,(TIMES_MS>=lo)&(TIMES_MS<hi)].mean())
                rows.append({'task':task,'subject':subject,'class':cls,'window_start_ms':lo,
                             'window_stop_ms_exclusive':hi,'roi_mean_uV':value,
                             'accepted_epochs':int(counts[ci])})
        required.extend(target/n for n in ('manifest.json','class_means.npz','word_onsets.csv.gz'))
    require(len(count_table)==len(items)*3,'Unexpected count-table rows')
    maximum_grand_error=0.
    for task in sorted({t for t,_ in pairs}):
        eligible_subjects=[i['subject'] for i in items if i['task']==task and min(i['epochs_per_class'].values())>=5]
        if not eligible_subjects:
            continue
        path=run/f'{task}_grand_average.npz'
        with np.load(path,allow_pickle=False) as data:
            subjects=data['subjects'].astype(str).tolist()
            require(len(subjects)==len(set(subjects)) and subjects==eligible_subjects,f'{task}: wrong grand-average readers')
            require(np.array_equal(data['times_ms'],TIMES_MS) and np.array_equal(data['channel_names'],channels)
                    and np.array_equal(data['classes'],CLASSES),f'{task}: grand-average axes')
            stacked=np.stack([means_by_pair[(task,s)] for s in subjects])
            require(np.array_equal(data['subject_means_uV'],stacked),f'{task}: grand file has different subject means')
            err=float(np.max(np.abs(data['mean_uV']-stacked.mean(axis=0))))
            require(err<1e-6,f'{task}: grand average does not reproduce')
            maximum_grand_error=max(maximum_grand_error,err)
        required.extend([path,run/'plots'/f'{task}_class_erp.png'])
    # Read and hash every archived scientific file; this checks CRCs and ensures
    # the portable copy is the same result, not an older partial run.
    bundle=run/'compact_results.zip'
    with zipfile.ZipFile(bundle) as archive:
        require(len(archive.namelist())==len(set(archive.namelist())),'Duplicate ZIP paths')
        for path in required:
            relative=str(path.relative_to(run))
            with archive.open(relative) as stream:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(1024*1024),b''):
                    h.update(block)
            require(h.hexdigest()==file_digest(path),f'Portable ZIP differs from current file: {relative}')
    output=run/'audit';output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'subject_window_means.csv',index=False)
    report={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'reader_task_records':len(pairs),'unique_readers':len({s for _,s in pairs}),
            'word_rows':word_rows,'accepted_epochs':accepted_total,'duplicate_event_ids':0,
            'required_files_compared_with_zip':len(required),
            'maximum_baseline_residual_uV':max_baseline,'maximum_grand_average_error_uV':maximum_grand_error,
            'checks':['reader coverage and identity','annotation checksum and accepted-event labels',
                      'all onset IDs/counts/order','full class-mean arrays and axes',
                      'grand averages reproduced from each reader','portable ZIP file hashes and CRCs'],
            'not_checked':['original multi-GB EEG files not reopened','independent human label validity',
                           'physical amplitude units and montage accuracy against acquisition files'],
            'timing':timing_summary(run)}
    if make_plots:
        plot_subjects(run,items,curves)
    report['audit_elapsed_seconds']=round(time.monotonic()-began,2)
    write_json(output/'integrity.json',report)
    write_json(output/'file_checksums.json',{str(p.relative_to(run)):file_digest(p) for p in required})
    (run/'START_HERE.md').write_text(
        '# Current word-sentiment EEG results — ZuCo 1.0\n\n'
        'This folder is the complete active result. Earlier v3 reports and trial caches are not needed to use it.\n\n'
        '- `REPORT.md`: extraction protocol and coverage.\n'
        '- `plots/*_class_erp.png`: task grand averages.\n'
        '- `plots/subject_grids/`: all 12 readers separately for each task.\n'
        '- `plots/subjects/`: one three-task figure per reader.\n'
        '- `readers/<task>/<reader>/class_means.npz`: 3 classes × 105 channels × 501 time samples, counts and axes.\n'
        '- `readers/<task>/<reader>/word_onsets.csv.gz`: all word occurrences, labels, first-fixation onset and exclusion status.\n'
        '- `readers/<task>/<reader>/manifest.json`: source checksum, settings and accepted counts.\n'
        '- `*_grand_average.npz`: reader means and task grand averages over all channels.\n'
        '- `class_counts.csv`, `class_erp_roi.csv`: counts and plotted group curves.\n'
        '- `audit/integrity.json`: saved-file audit results and explicit limits.\n'
        '- `audit/file_checksums.json`: hashes of the original result files.\n'
        '- `audit/subject_window_means.csv`: descriptive per-reader window measurements; no significance tests.\n'
        '- `word_labels.csv`, `zuco_chanlocs.csv`, `provenance.json`: labels, channel mapping, source inventory and settings.\n'
        '- `compact_results.zip`: portable copy of the extraction result; subject plots and audit added later remain beside it.\n\n'
        'Compact storage omits individual trial EEG arrays. New plots/ROIs/windows of the saved averages need no extraction. '
        'Trial-level matching, changed exclusions/labels, or individual-word voltage analyses require rebuilding relevant trials from source.\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def plot_subjects(run,items,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('tab:blue','tab:orange','tab:green')
    tasks=[t for t in ('SR','NR','TSR') if any(i['task']==t for i in items)]
    subjects=sorted({i['subject'] for i in items})
    counts={(i['task'],i['subject']):i['epochs_per_class'] for i in items}
    grid_dir=run/'plots/subject_grids';grid_dir.mkdir(parents=True,exist_ok=True)
    individual_dir=run/'plots/subjects';individual_dir.mkdir(parents=True,exist_ok=True)
    def panel(ax,task,subject):
        values=curves[(task,subject)]
        for cls,color,value in zip(CLASSES,colors,values):
            ax.plot(TIMES_MS,value,label=cls,color=color,lw=1.15)
        ax.axvline(0,color='black',lw=.65);ax.axhline(0,color='grey',lw=.55)
        n=counts[(task,subject)]
        ax.set_title(f"{subject} · {task}\nN−={n['negative']}, N0={n['neutral']}, N+={n['positive']}",fontsize=9)
        ax.set_xlim(-200,800)
    for task in tasks:
        selected=[s for s in subjects if (task,s) in curves]
        nrows=(len(selected)+2)//3
        fig,axes=plt.subplots(nrows,3,figsize=(15,3*nrows),sharex=True,sharey=True,squeeze=False)
        for ax,subject in zip(axes.flat,selected):
            panel(ax,task,subject)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f'{task}: individual-reader class FRPs — no individual-trial uncertainty bands',fontsize=14)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper right',ncol=3)
        fig.supxlabel('Time from first fixation onset (ms)');fig.supylabel('ROI amplitude (µV as stored)')
        fig.tight_layout(rect=(.02,.02,1,.95));fig.savefig(grid_dir/f'{task}_subjects.png',dpi=140);plt.close(fig)
    for subject in subjects:
        fig,axes=plt.subplots(1,len(tasks),figsize=(5*len(tasks),4),sharex=True,sharey=True,squeeze=False)
        for ax,task in zip(axes.flat,tasks):
            if (task,subject) in curves:
                panel(ax,task,subject)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper right',ncol=3)
        fig.suptitle(f'{subject}: class averages for this reader; same ROI and preprocessing',fontsize=12)
        fig.supxlabel('Time from first fixation onset (ms)');fig.supylabel('ROI amplitude (µV as stored)')
        fig.tight_layout(rect=(.02,.02,1,.90));fig.savefig(individual_dir/f'{subject}.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    audit(args.run_dir,not args.no_plots)
