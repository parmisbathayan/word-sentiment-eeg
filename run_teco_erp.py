"""Native Persian TeCo compact FRPs; run in Colab against existing private Drive files."""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import digest,file_digest,write_json,atomic_npz
from word_sentiment_eeg.teco import find_raw_files,NATIVE_HZ
from word_sentiment_eeg.teco_erp import (READERS,CLASSES,GROUPS,PRE,POST,TIMES_MS,VERSION,
                                        load_labels,extract_reader,inspect_reader)


def load_montage(path):
    frame=pd.read_csv(path,sep=r'\s+',header=None,names=['index','X','Y','Z','channel'])
    if len(frame)!=126 or frame.channel.duplicated().any() or frame['index'].tolist()!=list(range(1,127)):
        raise ValueError('Expected original ordered 126-channel native TeCo montage')
    if not np.isfinite(frame[['X','Y','Z']]).all().all():
        raise ValueError('Nonfinite native montage coordinates')
    # Native coordinates: X points right, Y anterior. Fixed disjoint geometric
    # bins are scalp summaries, not claims of intracranial source localization.
    frame['region']=np.select([frame.Y>=.5,frame.Y<=-.8,frame.Y<=-.4,frame.X<=-.65,frame.X>=.65],
                              ['anterior','occipital','posterior','left_lateral','right_lateral'],default='central')
    return frame


def source_metadata(path):
    stat=path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def save_state(run,items,failures,expected,fingerprint,stage):
    seen=[r['subject'] for r in items]
    if len(seen)!=len(set(seen)):
        raise ValueError('Duplicate reader would inflate grand averages')
    complete=set(seen)==set(expected) and not failures and stage=='finished'
    state={'dataset':'TeCo','status':'completed' if complete else 'incomplete',
           'stage':stage,'fingerprint':fingerprint,'expected_readers':list(expected),
           'completed_readers':len(items),'readers':items,'failures':failures,
           'missing_readers':sorted(set(expected)-set(seen)),
           'updated_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(run/'manifest.json',state)
    write_json(run/'progress.json',{'status':state['status'],'stage':stage,'readers_complete':len(items),
                                  'readers_expected':len(expected),'updated_at_utc':state['updated_at_utc']})
    return state


def summarize(run,items,montage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run); plots=run/'plots';plots.mkdir(exist_ok=True)
    # Regenerate only this runner's figures; a previously eligible comparison
    # must not leave a stale plot when a source change removes its last reader.
    for name in ['pooled_neutral_regions.png','strict_neutral_regions.png','source_neutral_regions.png',
                 'sensor_regions.png',*[f'{reader}_regions.png' for reader in READERS]]:
        (plots/name).unlink(missing_ok=True)
    for name in ('pooled_neutral_grand_average.npz','strict_neutral_grand_average.npz','source_neutral_grand_average.npz'):
        (run/name).unlink(missing_ok=True)
    channels=montage.channel.tolist();regions=list(dict.fromkeys(montage.region))
    montage.to_csv(run/'channel_regions.csv',index=False)
    means={}
    for item in items:
        with np.load(Path(item['output_dir'])/'class_means.npz',allow_pickle=False) as arrays:
            means[item['subject']]=arrays['means'].copy()
    counts=pd.DataFrame([{'subject':i['subject'],'group':g,'accepted_epochs':i['counts'][g]} for i in items for g in GROUPS])
    counts.to_csv(run/'class_counts.csv',index=False)
    csvrows=[];windows=[];rosters={}
    colors={'negative':'#c43b45','neutral':'#777777','positive':'#267d50',
            'neutral_in_neutral_sentence':'#2765a4','neutral_in_source_neutral_sentence':'#a35c18'}
    comparisons={'pooled_neutral':CLASSES,'strict_neutral':('negative','neutral_in_neutral_sentence','positive'),
                 'source_neutral':('negative','neutral_in_source_neutral_sentence','positive')}
    for comparison,groups in comparisons.items():
        readers=sorted(i['subject'] for i in items if all(i['counts'][g]>=5 for g in groups))
        rosters[comparison]={'minimum_epochs_each_group':5,'groups':list(groups),'readers':readers}
        indices=[GROUPS.index(g) for g in groups]
        if not readers:
            logging.warning('%s: no readers have at least five epochs in every comparison group',comparison)
            continue
        data=np.stack([means[r][indices] for r in readers])
        average=data.mean(axis=0)
        atomic_npz(run/f'{comparison}_grand_average.npz',means=average,reader_means=data,
                   readers=np.array(readers),classes=np.array(groups),channels=np.array(channels),times_ms=TIMES_MS)
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        for ax,region in zip(axes.flat,regions):
            picks=np.flatnonzero(montage.region.to_numpy()==region)
            roi=data[:,:,picks].mean(axis=2)
            for gi,group in enumerate(groups):
                curve=roi[:,gi].mean(axis=0)
                sem=roi[:,gi].std(axis=0,ddof=1)/np.sqrt(len(readers)) if len(readers)>1 else np.full_like(curve,np.nan)
                ax.plot(TIMES_MS,curve,label=group,color=colors[group])
                if len(readers)>1:
                    ax.fill_between(TIMES_MS,curve-sem,curve+sem,alpha=.15,color=colors[group])
                csvrows.extend({'comparison':comparison,'region':region,'group':group,'time_ms':float(t),
                                'mean_assumed_uV':float(v),'sem_between_readers':float(s),'n_readers':len(readers)} for t,v,s in zip(TIMES_MS,curve,sem))
            ax.axvline(0,color='k',lw=.5);ax.axhline(0,color='k',lw=.5)
            ax.set_title(f'{region} ({len(picks)} channels)')
            ax.set_xlabel('First-fixation onset (ms)');ax.set_ylabel('Amplitude (assumed µV)')
        axes.flat[0].legend(fontsize=8)
        uncertainty='Shading: between-reader SEM' if len(readers)>1 else 'One reader: between-reader SEM unavailable; no uncertainty band'
        fig.suptitle(f'TeCo task1 · {comparison} · {len(readers)} equally weighted readers\n{uncertainty}; overlapping fixation responses not deconvolved')
        fig.tight_layout();fig.savefig(plots/f'{comparison}_regions.png',dpi=140);plt.close(fig)
    for reader,values in means.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        item=next(i for i in items if i['subject']==reader)
        for ax,region in zip(axes.flat,regions):
            picks=np.flatnonzero(montage.region.to_numpy()==region)
            for gi,group in enumerate(GROUPS):
                if not item['counts'][group]:
                    continue
                roi=values[gi,picks].mean(axis=0)
                ax.plot(TIMES_MS,roi,color=colors[group],label=f'{group} (n={item["counts"][group]})')
                for start,stop in ((300,500),(500,800)):
                    mask=(TIMES_MS>=start)&(TIMES_MS<stop)
                    windows.append({'subject':reader,'group':group,'region':region,'window_start_ms':start,
                                    'window_stop_ms_exclusive':stop,'amplitude_assumed_uV':float(roi[mask].mean()),
                                    'accepted_epochs':item['counts'][group]})
            ax.set_title(region);ax.axvline(0,color='k',lw=.5);ax.axhline(0,color='k',lw=.5)
            ax.set_xlabel('First-fixation onset (ms)');ax.set_ylabel('Amplitude (assumed µV)')
        axes.flat[0].legend(fontsize=7);fig.suptitle(f'TeCo {reader}: descriptive class averages, no trial-level error bars')
        fig.tight_layout();fig.savefig(plots/f'{reader}_regions.png',dpi=120);plt.close(fig)
    pd.DataFrame(csvrows).to_csv(run/'regional_erp.csv',index=False)
    pd.DataFrame(windows).to_csv(run/'subject_window_means.csv',index=False)
    write_json(run/'comparison_rosters.json',rosters)
    fig,ax=plt.subplots(figsize=(8,8))
    for region in regions:
        subset=montage[montage.region==region]
        ax.scatter(subset.X,subset.Y,label=region)
        for row in subset.itertuples():
            ax.text(row.X,row.Y,row.channel,fontsize=5)
    ax.set_aspect('equal');ax.set_xlabel('Native X: right');ax.set_ylabel('Native Y: anterior');ax.legend(fontsize=8)
    ax.set_title('Native TeCo 126-channel scalp assignments');fig.tight_layout();fig.savefig(plots/'sensor_regions.png',dpi=160);plt.close(fig)
    return rosters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--raw-dir',type=Path)
    parser.add_argument('--run-tag',default='teco_v1_word_erp_compact')
    parser.add_argument('--montage',type=Path,default=ROOT/'assets/teco_montage.txt')
    args=parser.parse_args()
    if Path(args.run_tag).name!=args.run_tag:
        parser.error('Run tag must be one folder name')
    annotations=args.annotations or args.thesis_root/'gpt uploaded files/word_sentiment_eeg/teco/word_labels.csv'
    if not annotations.is_file():
        raise SystemExit(f'Private Persian labels not present: {annotations}. Do not substitute English labels.')
    labels,lookup=load_labels(annotations)
    sources,alternates=find_raw_files(args.raw_dir or args.thesis_root/'Data')
    missing=sorted(set(READERS)-set(sources));extras=sorted(set(sources)-set(READERS))
    if missing or extras:
        raise SystemExit(f'Native TeCo reader roster mismatch. Missing={missing}; unexpected={extras}. No extraction started.')
    montage=load_montage(args.montage);channels=montage.channel.tolist()
    run=args.thesis_root/'Results/word_sentiment_eeg'/args.run_tag
    run.mkdir(parents=True,exist_ok=True)
    lock=(Path(tempfile.gettempdir())/('teco-erp-'+digest(str(run.resolve()))[:16]+'.lock')).open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another extraction is running for this TeCo result folder.')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(run/'run.log')])
    if shutil.disk_usage(run).free<300*1024**2:
        raise SystemExit('Free at least 300 MiB in the mounted result account before TeCo extraction.')
    code_paths=[Path(__file__),ROOT/'src/word_sentiment_eeg/teco_erp.py',ROOT/'src/word_sentiment_eeg/teco.py',ROOT/'src/word_sentiment_eeg/erp.py']
    fingerprint=digest({'version':VERSION,'code':{p.name:file_digest(p) for p in code_paths},
                        'annotation_sha256':file_digest(annotations),'montage_sha256':file_digest(args.montage)})
    provenance={'dataset':'TeCo','version':VERSION,'fingerprint':fingerprint,'annotation_sha256':file_digest(annotations),
                'source_inventory':[{'reader':r,**source_metadata(sources[r])} for r in READERS],
                'alternate_copies_not_counted':alternates,'sampling_hz':NATIVE_HZ,'epoch_ms':[-200,800],
                'sampling_rate_basis':'Original TeCo notebook constants; not independently recovered acquisition header',
                'amplitude_units':'Native scale retained; microvolt convention assumed, not independently calibrated',
                'preprocessing':'Whole native sentence: average reference, fourth-order 30 Hz zero-phase lowpass; epoch -200..800 inclusive; baseline -200..0 exclusive; reject absolute amplitude >100 assumed uV',
                'onsets':'First stored fixation waveform, full-segment/all126-channel exact or constant-channel-offset match in reading raw_data; never answer_data/fixPositions/later fixation',
                'word_policy':'Agent reviewed, lexical/contextual agree, no alignment warning; controls excluded',
                'neutral_baseline':'Pooled neutral words; separate agent strict-neutral-sentence subset; separate original source-label0 neutral-sentence subset (overall neutral sentiment need not be affect-free). These subsets do not independently remove sentence-context confounding',
                'source_reuse':'Source path/size/mtime checked at resume; full source SHA256 computed on extraction; source bytes are not rehashed when metadata is unchanged',
                'individual_trial_arrays_saved':False,'public_text_published':False,'groups':list(GROUPS)}
    previous=run/'provenance.json'
    if previous.exists() and json.loads(previous.read_text()).get('fingerprint')!=fingerprint:
        raise SystemExit('Code, labels, or montage changed. Use a new --run-tag to preserve existing results cleanly.')
    write_json(previous,provenance)
    shutil.copyfile(annotations,run/'word_labels.csv')
    shutil.copyfile(args.montage,run/'teco_montage.txt')
    items=[];failures=[]
    for reader in READERS:
        path=sources[reader];metadata=source_metadata(path);out=run/'readers'/reader
        data=None
        logging.info('CHECK %s | %d/%d readers ready',reader,len(items),len(READERS))
        try:
            item=inspect_reader(out,reader,fingerprint,channels,metadata)
            if item is None:
                if shutil.disk_usage(run).free<300*1024**2:
                    raise OSError('Mounted output free space below 300 MiB safety margin')
                logging.info('EXTRACT %s: existing %.2f GB pickle, native 1200 Hz; no dataset download',reader,path.stat().st_size/1e9)
                source_info={**metadata,'sha256':file_digest(path)}
                previous_reader=out/'manifest.json'
                if previous_reader.exists():
                    previous_source=json.loads(previous_reader.read_text()).get('source',{}).get('sha256')
                    if previous_source and previous_source!=source_info['sha256']:
                        raise ValueError('Source EEG content changed; use a new run tag instead of mixing versions')
                with path.open('rb') as handle:
                    data=pickle.load(handle)
                item=extract_reader(data,reader,lookup,channels,out,fingerprint,source_info)
                data=None;gc.collect()
                if inspect_reader(out,reader,fingerprint,channels,metadata) is None:
                    raise OSError('Saved reader failed integrity validation')
            else:
                logging.info('KEEP %s: validated means/onsets, no repeated extraction',reader)
            items.append(item)
            logging.info('SAVED %d/%d unique TeCo readers',len(items),len(READERS))
        except Exception as error:
            logging.exception('FAILED %s; other valid readers retained',reader)
            failures.append({'subject':reader,'reason':str(error)})
        finally:
            data=None;gc.collect()
        save_state(run,items,failures,READERS,fingerprint,'reader_saved')
    rosters=summarize(run,items,montage) if items else {}
    state=save_state(run,items,failures,READERS,fingerprint,'finished')
    report=f'''# TeCo compact word FRPs\n\nRun status: {state['status']}; {len(items)}/12 readers validated.\n\nNative 126 channels at documented 1200 Hz, -200..800 ms first-fixation epochs.\nAll five group means retain all channels. The source-neutral subset uses original sentence label0 (overall neutral); the separate strict subset requires agent-reviewed affect-free neutral sentence context. Controls and uncertain/context-disagreement words are excluded.\nPrivate word_labels.csv records Persian labels; word_onsets.csv.gz records each reader's token, recovered onset, class and accepted/excluded status.\nAverages use equal reader weights with >=5 epochs in each relevant comparison group.\nComparison rosters: {json.dumps(rosters,ensure_ascii=False)}\n\nThese are descriptive overlapping fixation-related potentials, not deconvolved responses or significant sentiment effects.\nNative amplitude units and sampling rate follow existing project conventions and remain independently uncalibrated.\nLabels are single-agent judgments, not human consensus gold. Strict neutral baseline remains confounded with sentence context if valenced word contexts differ.\n\nFiles: readers/<reader>/class_means.npz and word_onsets.csv.gz, source SHA256/manifests, group NPZs, regional_erp.csv, subject_window_means.csv, comparison_rosters.json and plots/.\nNo original EEG copy or individual-trial archive is saved here. Restarting reuses deeply validated checkpoints; it never adds repeated readers.\n'''
    (run/'REPORT.md').write_text(report)
    logging.info('Results: %s | %s',run,state['status'])
    if state['status']!='completed':
        raise SystemExit('TeCo run incomplete; consult manifest.json. Valid readers retained for resume.')


if __name__=='__main__':
    main()
