"""Descriptive scalp-region comparisons from saved all-channel class averages.

No source EEG loading or trial selection occurs. Regions are geometric scalp
partitions, not localized cortical sources. Their boundaries are fixed before
looking at the class differences; all channels are assigned exactly once.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.erp import CLASSES, TIMES_MS
from word_sentiment_eeg.common import write_json

REGIONS=('anterior','central','posterior','occipital','left_lateral','right_lateral')
WINDOWS=((0,200),(200,300),(300,500),(500,800))


def assign_regions(montage):
    """ZuCo EEGLAB coordinates: +X anterior, +Y left, approximately cm.

    Thresholds refer to this stored HydroCel montage only. Posterior includes
    lateral sites in its X band. This is an exploratory six-way partition.
    """
    if len(montage)!=105 or montage.labels.duplicated().any():
        raise ValueError('Expected the saved 105-channel ZuCo montage')
    if not np.isfinite(montage[['X','Y','Z']].to_numpy()).all():
        raise ValueError('Nonfinite montage coordinates')
    if not (7<float(montage.Z.max())<11 and 7<float(montage.X.max())<11):
        raise ValueError('Montage scale differs from the expected HydroCel coordinates')
    groups=[]
    for row in montage.itertuples():
        if row.X>=3:
            name='anterior'
        elif row.X<=-6:
            name='occipital'
        elif row.X<=-3:
            name='posterior'
        elif row.Y>5:
            name='left_lateral'
        elif row.Y<-5:
            name='right_lateral'
        else:
            name='central'
        groups.append(name)
    result=montage[['index0','labels','X','Y','Z']].copy()
    result['region']=groups
    if set(groups)!=set(REGIONS):
        raise ValueError('Empty scalp region; inspect the montage before plotting')
    return result


def review_regions(run,make_plots=True):
    run=Path(run)
    audit=json.loads((run/'audit/integrity.json').read_text())
    if audit['status']!='passed':
        raise ValueError('Saved-file audit must pass before regional comparisons')
    montage=pd.read_csv(run/'zuco_chanlocs.csv')
    mapping=assign_regions(montage)
    tables=run/'audit/regions';tables.mkdir(parents=True,exist_ok=True)
    plotdir=run/'plots/regions';plotdir.mkdir(parents=True,exist_ok=True)
    mapping.to_csv(tables/'channel_regions.csv',index=False)
    subject_rows=[];curve_rows=[];window_rows=[];data_for_plots={}
    manifest=json.loads((run/'manifest.json').read_text())
    accepted={(i['task'],i['subject']):i['epochs_per_class'] for i in manifest['files']}
    for task in ('SR','NR','TSR'):
        path=run/f'{task}_grand_average.npz'
        if not path.exists():
            continue
        with np.load(path,allow_pickle=False) as data:
            values=data['subject_means_uV'].copy()
            names=data['channel_names'].astype(str).tolist()
            readers=data['subjects'].astype(str).tolist()
            if names!=mapping.labels.tolist() or not np.array_equal(data['times_ms'],TIMES_MS):
                raise ValueError(f'{task}: incompatible saved channel/time axes')
            if not np.array_equal(data['classes'],CLASSES):
                raise ValueError(f'{task}: unexpected class order')
        regional={}
        for region in REGIONS:
            indices=np.flatnonzero(mapping.region.eq(region).to_numpy())
            # Readers remain the unit of averaging and uncertainty. Do not
            # pool trials or compute SEM from electrode samples.
            subject=values[:,:,indices,:].mean(axis=2)
            mean=subject.mean(axis=0)
            sem=subject.std(axis=0,ddof=1)/np.sqrt(len(readers)) if len(readers)>1 else np.full_like(mean,np.nan)
            regional[region]=(subject,mean,sem)
            for ci,cls in enumerate(CLASSES):
                for ti,ms in enumerate(TIMES_MS):
                    curve_rows.append({'task':task,'region':region,'class':cls,'time_ms':ms,
                                       'mean_uV':mean[ci,ti],'sem_across_readers_uV':sem[ci,ti],
                                       'readers':len(readers),'channels':len(indices)})
                for si,reader in enumerate(readers):
                    for lo,hi in WINDOWS:
                        mask=(TIMES_MS>=lo)&(TIMES_MS<hi)
                        subject_rows.append({'task':task,'subject':reader,'region':region,'class':cls,
                                             'window_start_ms':lo,'window_stop_ms_exclusive':hi,
                                             'mean_uV':float(subject[si,ci,mask].mean()),
                                             'accepted_epochs':accepted[(task,reader)][cls]})
            for lo,hi in WINDOWS:
                mask=(TIMES_MS>=lo)&(TIMES_MS<hi)
                for a,b in (('negative','neutral'),('positive','neutral'),('negative','positive')):
                    per_reader=(subject[:,CLASSES.index(a),:]-subject[:,CLASSES.index(b),:])[:,mask].mean(axis=1)
                    window_rows.append({'task':task,'region':region,'contrast':f'{a}_minus_{b}',
                                        'window_start_ms':lo,'window_stop_ms_exclusive':hi,
                                        'mean_difference_uV':float(per_reader.mean()),
                                        'sem_across_readers_uV':float(per_reader.std(ddof=1)/np.sqrt(len(readers))) if len(readers)>1 else None,
                                        'readers':len(readers)})
        data_for_plots[task]=(readers,regional)
        print(f'REGIONS {task}: {len(readers)} readers, six fixed scalp regions',flush=True)
    pd.DataFrame(curve_rows).to_csv(tables/'class_curves.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(tables/'subject_window_means.csv',index=False)
    pd.DataFrame(window_rows).to_csv(tables/'paired_window_contrasts.csv',index=False)
    write_json(tables/'protocol.json',{
        'status':'descriptive','regions':{r:mapping.loc[mapping.region==r,'labels'].tolist() for r in REGIONS},
        'coordinate_source':'Saved ZuCo EEG.chanlocs; X anterior, Y left; approximately cm',
        'rules_in_order':['X >= 3: anterior','X <= -6: occipital','-6 < X <= -3: posterior',
                          '-3 < X < 3 and Y > 5: left lateral','-3 < X < 3 and Y < -5: right lateral',
                          'remaining sites: central'],
        'limitations':['Scalp regions are not localized cortical generators',
                       'Exploratory region/time comparisons; no significance tests or selected best region',
                       'Same accepted trials, labels and reader set as original saved averages',
                       'Neutral words remain pooled across sentence contexts in this existing result'],
        'windows_ms_half_open':WINDOWS,'channel_count':len(mapping)})
    if make_plots:
        plot_regions(mapping,data_for_plots,plotdir)
    return mapping,data_for_plots


def plot_regions(mapping,task_data,plotdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('tab:blue','tab:orange','tab:green')
    fig,ax=plt.subplots(figsize=(7,7))
    for ri,region in enumerate(REGIONS):
        part=mapping[mapping.region==region]
        ax.scatter(-part.Y,part.X,label=f'{region} (n={len(part)})',color=f'C{ri}',s=40)
        for row in part.itertuples():
            ax.annotate(row.labels,(-row.Y,row.X),fontsize=5,ha='center',xytext=(0,4),textcoords='offset points')
    ax.set(xlabel='Rightward scalp coordinate (−Y)',ylabel='Anterior scalp coordinate (X)',
           title='Fixed geometric scalp regions — sensor positions, not source locations')
    ax.axis('equal');ax.legend(fontsize=8,loc='lower left');fig.tight_layout()
    fig.savefig(plotdir/'channel_regions.png',dpi=150);plt.close(fig)
    def panel(ax,values,sem=None):
        for ci,(cls,color) in enumerate(zip(CLASSES,colors)):
            ax.plot(TIMES_MS,values[ci],color=color,label=cls,lw=1)
            if sem is not None:
                ax.fill_between(TIMES_MS,values[ci]-sem[ci],values[ci]+sem[ci],color=color,alpha=.15)
        ax.axvline(0,color='black',lw=.6);ax.axhline(0,color='grey',lw=.6);ax.set_xlim(-200,800)
    for task,(readers,regions) in task_data.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        for ax,region in zip(axes.flat,REGIONS):
            _,mean,sem=regions[region];panel(ax,mean,sem)
            ax.set_title(f'{region} · {(mapping.region==region).sum()} channels')
        fig.suptitle(f'{task}: fixed scalp regions; equal reader averages ±1 SEM ({len(readers)} readers)')
        fig.supxlabel('Time from first fixation onset (ms)');fig.supylabel('Amplitude (µV as stored)')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper right',ncol=3)
        fig.tight_layout(rect=(.02,.02,1,.94));fig.savefig(plotdir/f'{task}_regions.png',dpi=140);plt.close(fig)
        target=plotdir/'subjects';target.mkdir(exist_ok=True)
        for si,reader in enumerate(readers):
            fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
            for ax,region in zip(axes.flat,REGIONS):
                panel(ax,regions[region][0][si]);ax.set_title(region)
            fig.suptitle(f'{task} · {reader}: fixed scalp regions; descriptive class means')
            fig.supxlabel('Time from first fixation onset (ms)');fig.supylabel('Amplitude (µV as stored)')
            fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper right',ncol=3)
            fig.tight_layout(rect=(.02,.02,1,.94));fig.savefig(target/f'{task}_{reader}.png',dpi=110);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    review_regions(args.run_dir,not args.no_plots)
