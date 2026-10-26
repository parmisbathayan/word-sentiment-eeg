"""One saved-result gallery: whole-scalp, original ZuCo ROI, GFP and regions.

Reads compact class averages only. It never imports a downloader, opens raw EEG,
changes an epoch, or writes into extraction checkpoints. GFP here is calculated
from each reader's class-average waveform, not single-trial GFP.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from word_sentiment_eeg.common import file_digest,write_json
from word_sentiment_eeg.erp import CLASSES,ROI,TIMES_MS as ZUCO_TIMES
from word_sentiment_eeg.teco_erp import GROUPS as TECO_GROUPS,TIMES_MS as TECO_TIMES,READERS
from review_regions import REGIONS,assign_regions
from run_teco_erp import load_montage

COLORS={'negative':'#c43b45','neutral':'#777777','positive':'#267d50',
        'neutral_in_neutral_sentence':'#2765a4','neutral_in_source_neutral_sentence':'#a35c18'}
LABELS={'negative':'Negative','neutral':'Neutral (all contexts)','positive':'Positive',
        'neutral_in_neutral_sentence':'Neutral word / strict neutral context',
        'neutral_in_source_neutral_sentence':'Neutral word / source-rated neutral review'}
TECO_COMPARISONS={'pooled_neutral':CLASSES,
                 'strict_neutral':('negative','neutral_in_neutral_sentence','positive'),
                 'source_neutral':('negative','neutral_in_source_neutral_sentence','positive')}

@dataclass
class Record:
    task:str
    subject:str
    means:np.ndarray
    counts:np.ndarray

@dataclass
class Saved:
    dataset:str
    groups:tuple
    times:np.ndarray
    channels:list
    montage:pd.DataFrame
    records:list
    comparisons:dict
    hashes:dict
    units:str


def read_arrays(path,group_names,channels,times,mean_key,count_values):
    with np.load(path,allow_pickle=False) as saved:
        channel_key='channel_names' if mean_key=='mean_uV' else 'channels'
        if (tuple(saved['classes'].astype(str))!=tuple(group_names)
                or saved[channel_key].astype(str).tolist()!=channels
                or not np.array_equal(saved['times_ms'],times)):
            raise ValueError(f'Class/channel/time axes differ: {path}')
        values=saved[mean_key].astype(np.float64)
        counts=saved['counts'].copy()
    if (values.shape!=(len(group_names),len(channels),len(times))
            or counts.shape!=(len(group_names),) or not np.array_equal(counts,count_values)
            or not np.isfinite(counts).all() or (counts<0).any()
            or not np.equal(counts,np.floor(counts)).all()
            or not np.isfinite(values[counts>0]).all()):
        raise ValueError(f'Invalid compact averages/counts: {path}')
    return values,counts.astype(int)


def load_saved(run,dataset):
    run=Path(run);manifest_path=run/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='completed' or manifest.get('failures'):
        raise ValueError('The extraction is not recorded as complete; no complete-run gallery will be written')
    hashes={'manifest.json':file_digest(manifest_path)}
    records=[];seen=set()
    if dataset=='zuco':
        groups=CLASSES;times=ZUCO_TIMES;units='µV as stored'
        audit=run/'audit/integrity.json'
        if not audit.is_file() or json.loads(audit.read_text())['status']!='passed':
            raise ValueError('Run review.ipynb once to audit ZuCo saved results before plotting')
        hashes['audit/integrity.json']=file_digest(audit)
        checksums=json.loads((run/'audit/file_checksums.json').read_text())
        montage=assign_regions(pd.read_csv(run/'zuco_chanlocs.csv'))
        channels=montage.labels.tolist();items=manifest['files'];comparisons={'pooled_neutral':CLASSES}
    elif dataset=='teco':
        groups=TECO_GROUPS;times=TECO_TIMES;units='Assumed µV (native scale)'
        montage=load_montage(run/'teco_montage.txt').rename(columns={'channel':'labels'})
        montage['index0']=np.arange(len(montage));channels=montage.labels.tolist()
        items=manifest['readers'];comparisons=TECO_COMPARISONS
        if (set(manifest['expected_readers'])!=set(READERS) or manifest.get('missing_readers')
                or len(items)!=len(READERS)):
            raise ValueError('TeCo requires all 12 completed canonical readers')
    else:
        raise ValueError('Unknown dataset')
    montage_name='zuco_chanlocs.csv' if dataset=='zuco' else 'teco_montage.txt'
    hashes[montage_name]=file_digest(run/montage_name)
    for item in items:
        task=item['task'];subject=item['subject'];identity=(task,subject)
        if identity in seen:
            raise ValueError('Duplicate recording would inflate the reader average')
        seen.add(identity)
        relative=Path('readers')/task/subject if dataset=='zuco' else Path('readers')/subject
        path=run/relative/'class_means.npz';local=json.loads((path.parent/'manifest.json').read_text())
        if (local['task'],local['subject'],local['fingerprint'])!=(task,subject,item['fingerprint']):
            raise ValueError('Reader identity/fingerprint differs from extraction manifest')
        counts_dict=item['epochs_per_class'] if dataset=='zuco' else item['counts']
        if counts_dict!=(local['epochs_per_class'] if dataset=='zuco' else local['counts']):
            raise ValueError('Reader count manifest differs')
        sha=file_digest(path)
        expected=checksums[str(relative/'class_means.npz')] if dataset=='zuco' else item['checksums']['class_means.npz']
        if sha!=expected or (dataset=='teco' and local['checksums']['class_means.npz']!=expected):
            raise ValueError(f'Saved average checksum failed: {task}/{subject}')
        hashes[str(relative/'class_means.npz')]=sha
        means,counts=read_arrays(path,groups,channels,times,'mean_uV' if dataset=='zuco' else 'means',[counts_dict[g] for g in groups])
        records.append(Record(task,subject,means,counts))
    if dataset=='zuco' and (len(records)!=36 or {r.task for r in records}!={'SR','NR','TSR'} or any(sum(r.task==task for r in records)!=12 for task in ('SR','NR','TSR'))):
        raise ValueError('ZuCo requires the completed 36 reader/task records')
    if dataset=='teco' and ({r.subject for r in records}!=set(READERS) or {r.task for r in records}!={'task1'}):
        raise ValueError('TeCo recording roster differs')
    return Saved(dataset,tuple(groups),times,channels,montage,records,comparisons,hashes,units)


def summaries(values,montage,channels,include_original_roi):
    """Input [reader,group,channel,time]; output retains reader and group axes."""
    if values.ndim!=4 or values.shape[2]!=len(channels) or not np.isfinite(values).all():
        raise ValueError('Finite all-channel reader averages are required')
    result={'whole_scalp_signed_mean':values.mean(axis=2),
            'whole_scalp_gfp':values.std(axis=2,ddof=0)}
    if include_original_roi:
        result['original_12_channel_roi']=values[:,:,[channels.index(c) for c in ROI],:].mean(axis=2)
    for region in REGIONS:
        indices=np.flatnonzero(montage.region.eq(region).to_numpy())
        if not len(indices):
            raise ValueError('An empty region cannot be plotted')
        result[region]=values[:,:,indices,:].mean(axis=2)
    return result


def matched(records,groups,all_groups):
    indices=[all_groups.index(g) for g in groups]
    eligible=sorted((r for r in records if min(r.counts[indices])>=5),key=lambda r:r.subject)
    if not eligible:
        return [],None
    return [r.subject for r in eligible],np.stack([r.means[indices] for r in eligible])


def mean_and_sem(per_reader,across_readers=True):
    mean=per_reader.mean(axis=0)
    sem=per_reader.std(axis=0,ddof=1)/np.sqrt(len(per_reader)) if across_readers and len(per_reader)>1 else None
    return mean,sem


def panel_layout(n,horizontal=False):
    if horizontal:
        return 1,n,(4.0*n,4.4)
    if n==1:
        return 1,1,(9,5)
    cols=3 if n in (3,6,9) else 2
    rows=(n+cols-1)//cols
    return rows,cols,(5.4*cols,3.45*rows)


def render_panels(metrics,groups,times,units,path,title,keys,channel_counts,group_plot,horizontal=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,cols,size=panel_layout(len(keys),horizontal)
    fig,axes=plt.subplots(rows,cols,figsize=size,squeeze=False,sharex=True)
    signed_keys=[k for k in metrics if k!='whole_scalp_gfp']
    signed_bound=max([float(np.abs(metrics[k].mean(axis=0)).max()) for k in signed_keys]+[.01])
    for k in signed_keys:
        if group_plot and len(metrics[k])>1:
            band=metrics[k].std(axis=0,ddof=1)/np.sqrt(len(metrics[k]))
            signed_bound=max(signed_bound,float((np.abs(metrics[k].mean(axis=0))+band).max()))
    names={'whole_scalp_signed_mean':'Whole scalp · signed mean (near zero after average reference)',
           'whole_scalp_gfp':"Whole scalp · GFP of readers’ class averages",
           'original_12_channel_roi':'Original ZuCo 12-channel ROI'}
    for ax,key in zip(axes.flat,keys):
        per_reader=metrics[key];mean,sem=mean_and_sem(per_reader,group_plot)
        for gi,group in enumerate(groups):
            ax.plot(times,mean[gi],color=COLORS[group],label=LABELS[group],lw=1.3)
            if sem is not None:
                ax.fill_between(times,mean[gi]-sem[gi],mean[gi]+sem[gi],color=COLORS[group],alpha=.14)
        if key=='whole_scalp_gfp':
            ax.set_ylim(bottom=0)
        elif key=='original_12_channel_roi' and len(keys)==1:
            pass  # Restore the original overview's own autoscaled mean/SEM view.
        else:
            ax.set_ylim(-signed_bound*1.1,signed_bound*1.1)
        extra=f' · {channel_counts[key]} channels' if key in REGIONS else ''
        panel_title=names.get(key,key.replace('_',' ').title())+extra
        ax.set_title(textwrap.fill(panel_title,width=42) if horizontal else panel_title,fontsize=9)
        ax.axvline(0,color='black',lw=.6);ax.axhline(0,color='grey',lw=.5)
        ax.set(xlim=(times[0],times[-1]),xlabel='First-fixation onset (ms)',ylabel=units)
    for ax in list(axes.flat)[len(keys):]:ax.set_visible(False)
    uncertainty='Shading: ±1 SEM across readers' if group_plot and len(next(iter(metrics.values())))>1 else 'No between-reader uncertainty band'
    fig.suptitle(title+'\n'+uncertainty+' · descriptive fixation-related averages',fontsize=12)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=min(len(groups),3),fontsize=8)
    fig.tight_layout(rect=(0,.10 if horizontal else .065,1,.80 if horizontal else .91));path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=125);plt.close(fig)


def render_reader_matrix(saved,task,path):
    """One figure: reader rows and measure columns, without re-epoching EEG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    records=sorted((r for r in saved.records if r.task==task),key=lambda r:r.subject)
    if not records or len({r.subject for r in records})!=len(records):
        raise ValueError('A matrix requires a nonempty, unique reader roster')
    keys=['whole_scalp_signed_mean']
    if saved.dataset=='zuco':keys.append('original_12_channel_roi')
    keys+=['whole_scalp_gfp',*REGIONS]
    titles={'whole_scalp_signed_mean':'Whole-head signed mean',
            'original_12_channel_roi':'Original 12-channel ROI',
            'whole_scalp_gfp':'Whole-head GFP'}
    # One common scale for signed panels within each reader row. GFP is a
    # separate nonnegative metric. Scales can differ between readers.
    fig,axes=plt.subplots(len(records),len(keys),squeeze=False,
                          figsize=(3.5*len(keys)+2.2,2.45*len(records)+1.5),sharex=True)
    fig.subplots_adjust(left=.075,right=.992,bottom=.045,top=.955,hspace=.52,wspace=.28)
    short={'negative':'Neg','neutral':'Neu','positive':'Pos',
           'neutral_in_neutral_sentence':'Strict neu',
           'neutral_in_source_neutral_sentence':'Source neu'}
    for ri,record in enumerate(records):
        valid=np.flatnonzero(record.counts>0)
        if len(valid):
            metrics=summaries(record.means[valid][None],saved.montage,saved.channels,saved.dataset=='zuco')
            bound=max([float(np.abs(metrics[k]).max()) for k in keys if k!='whole_scalp_gfp']+[.01])
        else:metrics={};bound=.01
        for ci,key in enumerate(keys):
            ax=axes[ri,ci]
            for gi,index in enumerate(valid):
                group=saved.groups[index]
                ax.plot(saved.times,metrics[key][0,gi],color=COLORS[group],lw=1)
            if not len(valid):ax.text(.5,.5,'No accepted epochs',ha='center',transform=ax.transAxes,fontsize=8)
            ax.axvline(0,color='black',lw=.5);ax.axhline(0,color='grey',lw=.4)
            ax.set_xlim(saved.times[0],saved.times[-1])
            ax.set_ylim(bottom=0) if key=='whole_scalp_gfp' else ax.set_ylim(-bound*1.1,bound*1.1)
            ax.tick_params(labelsize=7)
            if ri==0:ax.set_title(titles.get(key,key.replace('_',' ').title()),fontsize=10,pad=12)
            if ri==len(records)-1:ax.set_xlabel('Fixation onset (ms)',fontsize=8)
        label=record.subject+'\n'+'\n'.join(f'{short[g]}: {int(record.counts[i])}' for i,g in enumerate(saved.groups))
        axes[ri,0].text(-.26,.5,label,transform=axes[ri,0].transAxes,
                        ha='right',va='center',fontsize=8,fontweight='bold',clip_on=False)
    fig.suptitle(f'{saved.dataset.upper()} / {task}: all {len(records)} readers in one matrix',fontsize=15,y=.988)
    handles=[Line2D([0],[0],color=COLORS[g],label=LABELS[g]) for g in saved.groups]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.012),ncol=len(handles),fontsize=9)
    fig.text(.52,.002,'Individual class means; no SEM band. Signed scales shared within rows, not between readers. '
             f'Units: {saved.units}. Whole-head signed mean is near zero after average reference.',ha='center',fontsize=9)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=110)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    return {'kind':'subject_matrix','task':task,'path':str(path.relative_to(path.parent.parent)),
            'pdf_path':str(path.with_suffix('.pdf').relative_to(path.parent.parent)),
            'readers':[r.subject for r in records],'columns':keys,'layout':'reader_by_measure_matrix',
            'reader_count':len(records),'display_width_px':420*len(keys),
            'title':f'{task}: readers × plot types'}


def write_reader_matrices(saved,out,make_plots=True):
    items=[]
    for task in sorted({r.task for r in saved.records}):
        path=Path(out)/'reader_matrices'/f'{task}_reader_matrix.png'
        if make_plots:
            item=render_reader_matrix(saved,task,path)
        else:
            keys=['whole_scalp_signed_mean']+(['original_12_channel_roi'] if saved.dataset=='zuco' else [])+['whole_scalp_gfp',*REGIONS]
            readers=sorted(r.subject for r in saved.records if r.task==task)
            item={'kind':'subject_matrix','task':task,'path':str(path.relative_to(out)),
                  'pdf_path':str(path.with_suffix('.pdf').relative_to(out)),
                  'readers':readers,'columns':keys,'reader_count':len(readers),
                  'layout':'reader_by_measure_matrix','display_width_px':420*len(keys),
                  'title':f'{task}: readers × plot types'}
        items.append(item)
        print(f'MATRIX {task}: {item["reader_count"]} reader rows × {len(item["columns"])} plot columns',flush=True)
    write_json(Path(out)/'reader_matrix_gallery.json',items)
    return items


def plot_collection(run,dataset,make_plots=True):
    run=Path(run);saved=load_saved(run,dataset);out=run/'analysis/all_plots_v1';out.mkdir(parents=True,exist_ok=True)
    gallery=[];curves=[];rosters={};count_rows=[]
    overall=['whole_scalp_signed_mean','whole_scalp_gfp']
    if dataset=='zuco':overall.insert(1,'original_12_channel_roi')
    channel_counts={r:int(saved.montage.region.eq(r).sum()) for r in REGIONS}
    tasks=sorted({r.task for r in saved.records})
    for task in tasks:
        records=[r for r in saved.records if r.task==task]
        for comparison,groups in saved.comparisons.items():
            readers,values=matched(records,groups,saved.groups)
            key=task+'/'+comparison
            rosters[key]={'readers':readers,'groups':list(groups),'minimum_epochs_per_group':5}
            if values is None:
                print(f'SKIP {key}: no readers have >=5 accepted epochs in every group',flush=True);continue
            metrics=summaries(values,saved.montage,saved.channels,dataset=='zuco')
            for measure,per_reader in metrics.items():
                mean,sem=mean_and_sem(per_reader)
                if sem is None:sem=np.full_like(mean,np.nan)
                for gi,group in enumerate(groups):
                    curves.extend({'task':task,'comparison':comparison,'measure':measure,'group':group,'time_ms':float(t),
                                   'mean':float(m),'sem_across_readers':float(s),'readers':len(readers)}
                                  for t,m,s in zip(saved.times,mean[gi],sem[gi]))
            group_views=([('overall',['whole_scalp_signed_mean']),('original_overview',['original_12_channel_roi']),
                          ('gfp',['whole_scalp_gfp']),('regions',list(REGIONS))] if dataset=='zuco'
                         else [('overall',overall),('regions',list(REGIONS))])
            for kind,keys in group_views:
                rel=Path('group')/f'{task}_{comparison}_{kind}.png'
                if make_plots:render_panels(metrics,groups,saved.times,saved.units,out/rel,
                                           f'{dataset.upper()} · {task} · {comparison.replace("_"," ")} · {len(readers)} equally weighted readers',keys,channel_counts,True)
                gallery.append({'kind':'group','task':task,'comparison':comparison,'path':str(rel),'title':f'{task} · {comparison} · {kind}', 'view':kind,
                                'measures':keys,'reader_count':len(readers)})
            print(f'GROUP {key}: overall and six regions; {len(readers)} matched readers',flush=True)
        for record in sorted(records,key=lambda r:r.subject):
            indices=np.flatnonzero(record.counts>0)
            if not len(indices):continue
            groups=tuple(saved.groups[i] for i in indices)
            values=record.means[indices][None]
            metrics=summaries(values,saved.montage,saved.channels,dataset=='zuco')
            rel=Path('subjects')/f'{task}_{record.subject}.png'
            counts=', '.join(f'{LABELS[g]}: {record.counts[saved.groups.index(g)]}' for g in groups)
            if make_plots:render_panels(metrics,groups,saved.times,saved.units,out/rel,
                                       f'{dataset.upper()} · {task} · {record.subject}\nAccepted word epochs — '+counts,overall+list(REGIONS),channel_counts,False,horizontal=True)
            gallery.append({'kind':'subject','task':task,'subject':record.subject,'path':str(rel),'title':f'{task} · {record.subject} · overall + regions','layout':'one_reader_per_row',
                            'columns':overall+list(REGIONS),'display_width_px':420*(len(overall)+len(REGIONS))})
            count_rows.extend({'task':task,'subject':record.subject,'group':g,'accepted_epochs':int(record.counts[i])} for i,g in enumerate(saved.groups))
    gallery.extend(write_reader_matrices(saved,out,make_plots))
    saved.montage.to_csv(out/'channel_regions.csv',index=False)
    if make_plots:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,7))
        for region in REGIONS:
            part=saved.montage[saved.montage.region==region]
            x=-part.Y if dataset=='zuco' else part.X;y=part.X if dataset=='zuco' else part.Y
            ax.scatter(x,y,label=f'{region} ({len(part)})')
            for label,xx,yy in zip(part.labels,x,y):ax.text(xx,yy,label,fontsize=5)
        ax.set_aspect('equal');ax.set(xlabel='Rightward scalp coordinate',ylabel='Anterior scalp coordinate',title=f'{dataset.upper()}: fixed scalp-region assignments')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'sensor_regions.png',dpi=140);plt.close(fig)
    pd.DataFrame(curves).to_csv(out/'group_curves.csv.gz',index=False,compression='gzip')
    pd.DataFrame(count_rows).to_csv(out/'reader_class_counts.csv',index=False)
    write_json(out/'gallery.json',gallery);write_json(out/'comparison_rosters.json',rosters)
    protocol={'status':'completed','dataset':dataset,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':saved.hashes,'source_eeg_read':False,'extraction_files_modified':False,
              'reader_weighting':'Equal reader means; same >=5-per-group matched readers in all panels of a comparison',
              'whole_scalp_signed_mean':'Arithmetic channel mean of each reader/class average. Average reference makes this approximately zero; signed y-scale shared with other signed panels to avoid magnifying numerical residue.',
              'gfp':'Population standard deviation across channels of each reader/class-average waveform, then equal-reader average. Not GFP of single trials or of the grand average; not baseline-corrected again.',
              'gfp_limitations':'Class-average GFP depends on signal and residual noise, and unequal accepted trial counts can bias class comparisons. Descriptive only; no sentiment-specific/significance claim.',
              'sem':'Between-reader sample SD divided by sqrt(n); ±1 SEM, matching the original task overview. Absent for single-reader figures and n=1 groups',
              'subject_layout':'One combined reader-by-measure matrix per task; PNG and PDF. Individual legacy images also retained.',
              'original_zuco_overview':'Original fixed 12-channel ROI, equal reader means and ±1 SEM; not an all-105-channel mean',
              'regions':'Fixed disjoint geometric scalp partitions, not localized brain sources',
              'units':saved.units,'included':'Existing pooled word classes; TeCo additionally strict and source-neutral baselines. New extraction/context-controls are not run by this plotter.',
              'signed_mean_max_abs':max(float(np.abs(r.means[r.counts>0].mean(axis=1)).max()) for r in saved.records if (r.counts>0).any())}
    write_json(out/'protocol.json',protocol)
    (out/'REPORT.md').write_text(f'''# {dataset.upper()} — all saved-result plots\n\n{len(saved.records)} saved reader/task records checked. {len(gallery)} figures listed in gallery.json, plus a sensor map.\n\n- Group figures: whole-scalp signed mean, global field power (GFP), six scalp regions{', and the original ZuCo 12-channel ROI' if dataset=='zuco' else ''}. Each comparison uses the same readers in all panels.\n- Subject matrices: one combined PNG and PDF per task, with readers as rows and whole-head mean, original ROI (ZuCo), GFP and six regions as columns. Legacy separate-reader figures are retained but the notebook displays the combined matrix. Groups with no accepted epochs are omitted; small groups remain descriptive.\n- Whole-scalp signed mean is approximately zero after average reference. A flat curve is expected and does not mean EEG has no activity. Maximum absolute signed channel mean in saved eligible class averages: {protocol['signed_mean_max_abs']:.4g} ({saved.units}).\n- GFP is a different, nonnegative measure: population standard deviation across all channels of each reader's class-average waveform, then averaged across readers. It is not single-trial GFP, signed ERP voltage, a sentiment score or proof of significance. Unequal trial counts/residual noise affect class-average GFP.\n- Group shading is ±1 SEM across readers (sample SD divided by sqrt(reader count)), matching the original task overview. It is not ±1 SD. No uncertainty band is manufactured for one reader.\n- ZuCo includes one standalone original 12-channel overview and one standalone true all-105-channel mean per task.\n- Scalp areas are geometric sensor groups, not localized cortical generators. Overlapping fixation responses have not been deconvolved.\n- Original accepted events, labels and EEG averages are unchanged. Only compact saved means are read; no original EEG, downloads or extraction.\n- group_curves.csv.gz, reader_class_counts.csv, comparison_rosters.json and protocol.json record values, denominators and source hashes. All outputs stay in analysis/all_plots_v1/.\n\nGFP definition: https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#global-field-power-gfp\n''')
    print(f'ALL PLOTS READY: {out}; {len(gallery)} figures plus sensor map',flush=True)
    return saved,out,gallery


def main(dataset):
    parser=argparse.ArgumentParser(description=f'Plot all saved {dataset.upper()} class EEG averages; no extraction')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--reader-matrices-only',action='store_true',help='Reuse group figures; generate only combined reader matrices')
    args=parser.parse_args()
    if args.reader_matrices_only:
        saved=load_saved(args.run_dir,dataset)
        out=args.run_dir/'analysis/all_plots_v1';out.mkdir(parents=True,exist_ok=True)
        write_reader_matrices(saved,out)
    else:plot_collection(args.run_dir,dataset)

if __name__=='__main__':
    raise SystemExit('Use plot_zuco.py or plot_teco.py')
