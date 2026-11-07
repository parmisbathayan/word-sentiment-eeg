#!/usr/bin/env python3
"""Render four main-paper A–F panel assemblies from computed ZuCo statistics.

All curves, exclusions, p-values and contours originate in this run. No published
effect sizes/significance masks are copied. Layout follows the panel inventory;
output captions state method departures and the fixed reference map window.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import atomic_npz,write_json,file_digest
from word_sentiment_eeg.paper_figure_stats import mixed_models,temporal_mask,scalp_geometry,finish_permutations

COLORS={'negative':'#202020','positive':'#cb3335'}


def load_readers(out):
    state=json.loads((out/'preparation_manifest.json').read_text())
    if state['status']!='completed':raise ValueError('Preparation is not complete')
    readers=state['subjects'];values=[];events=[]
    for reader in readers:
        folder=out/'readers'/reader;saved=json.loads((folder/'manifest.json').read_text())
        if saved['status']!='completed' or any(file_digest(folder/k)!=v for k,v in saved['checksums'].items()):raise ValueError('Corrupt reader figures checkpoint')
        with np.load(folder/'statistics.npz',allow_pickle=False) as z:values.append({k:z[k].copy() for k in z.files})
        events.append(pd.read_csv(folder/'fixation_audit.csv.gz',keep_default_na=False))
    return readers,values,pd.concat(events,ignore_index=True)


def analyze_and_render(out):
    out=Path(out);readers,stats,events=load_readers(out)
    if len(readers)<3:raise ValueError('Group inference requires at least three readers; this source pilot prepared one reader only')
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    counts=np.stack([v['counts'] for v in stats]);sums=np.stack([v['sums'] for v in stats]);squares=np.stack([v['squares'] for v in stats])
    models=[]
    map_models=[]
    for r,name in enumerate(['raw','cleaned']):
        models.append(mixed_models(counts,sums[:,r],squares[:,r],checkpoint=analysis/f'{name}_ml_checkpoint.npz',
            progress=lambda a,b:print(f'MIXED MODEL {name}: {a}/{b} channel/time features',flush=True)))
        map_models.append(mixed_models(counts,np.stack([v['map_sums'][r] for v in stats]),np.stack([v['map_squares'][r] for v in stats]),checkpoint=analysis/f'{name}_map_ml.npz'))
    xy,adj=scalp_geometry(pd.read_csv(ROOT/'assets/zuco_chanlocs.csv'))
    pair_sum=np.sum([v['pair_sum'] for v in stats],axis=0);pair_squares=np.sum([v['pair_squares'] for v in stats],axis=0);n=sum(int(v['pairs']) for v in stats)
    if (analysis/'permutation_results.npz').is_file():
        with np.load(analysis/'permutation_results.npz',allow_pickle=False) as z:permutation={k:z[k].copy() for k in z.files}
    else:
        permutation,clusters=finish_permutations(out/'null_work',pair_sum,pair_squares,n,adj,readers)
        atomic_npz(analysis/'permutation_results.npz',**permutation);write_json(analysis/'clusters.json',clusters)
    atomic_npz(analysis/'group_results.npz',counts=counts,sums=sums,squares=squares,xy=xy,adjacency=adj,
               p=np.stack([v['p'] for v in models]),effect=np.stack([v['effect'] for v in models]),
               map_p=np.stack([v['p'] for v in map_models]),
               converged=np.stack([v['converged'] for v in models]),times_ms=stats[0]['times_ms'])
    # Eye velocity is fitted only where complete native gaze epochs exist.
    ec=np.stack([v['eye_counts'] for v in stats]);pick=(ec>=2).all(axis=1)
    if pick.sum()>=3:
        eye_model=mixed_models(ec[pick],np.stack([v['eye_sums'] for v in stats])[pick],np.stack([v['eye_squares'] for v in stats])[pick],checkpoint=analysis/'eye_ml_checkpoint.npz')
        atomic_npz(analysis/'eye_model.npz',p=eye_model['p'],included_readers=pick)
    else:atomic_npz(analysis/'eye_model.npz',p=np.full((2,201),np.nan),included_readers=pick)
    write_json(analysis/'analysis_manifest.json',dict(status='completed',subjects=readers,paired_trials=n,
               nonconverged_model_features=[int((v['converged']!=1).sum()) for v in models],
               model='Normal/identity ML, correlated reader random intercept and sentiment slope; asymptotic Wald tests',
               tct='Independent electrode permutations for every within-reader ranked trial pair',
               cluster='Within-pair sign flips, maximum absolute spatial-temporal cluster mass',
               sensor_neighbors='Distance-pruned Delaunay, >=5 supra-threshold neighbors',
               source='Pfeiffer et al. 2020, doi:10.1016/j.neuroimage.2020.116934'))
    render_all(out)


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    return plt


def panel(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',fontsize=10,pad=8)


def save(fig,out,name,caption):
    plt=style();folder=out/'figures';folder.mkdir(exist_ok=True)
    if (out/'SYNTHETIC_TEST_ONLY').exists():
        fig.text(.5,.5,'SYNTHETIC TEST — NOT RESULTS',ha='center',rotation=25,fontsize=30,color='#999999',alpha=.25)
    fig.text(.02,.015,caption,fontsize=7,color='#555555',va='bottom',wrap=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def topography(ax,values,xy,limit,title,probability=False):
    from scipy.interpolate import griddata
    from matplotlib.patches import Circle
    grid=np.linspace(-1.05,1.05,100);xx,yy=np.meshgrid(grid,grid)
    finite=np.isfinite(values)
    z=griddata(xy[finite],np.asarray(values)[finite],(xx,yy),method='cubic') if finite.sum()>=4 else np.full(xx.shape,np.nan)
    z[xx*xx+yy*yy>1]=np.nan
    im=ax.imshow(z,extent=[grid[0],grid[-1],grid[0],grid[-1]],origin='lower',cmap='YlOrRd' if probability else 'RdYlBu_r',vmin=0 if probability else -limit,vmax=limit)
    ax.add_patch(Circle((0,0),1,fill=False,lw=.7,color='black'));ax.plot([-.08,0,.08],[1,1.13,1],color='black',lw=.7)
    ax.scatter(xy[:,0],xy[:,1],s=1,color='black');ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.15),title=title);ax.axis('off');return im


def figure1(out,readers,stats,events):
    plt=style();fig=plt.figure(figsize=(15,11));grid=fig.add_gridspec(2,2,width_ratios=[1.05,1.1],height_ratios=[.55,1.3],hspace=.35,wspace=.25)
    ax=fig.add_subplot(grid[0,0]);example=json.loads((out/'readers'/readers[0]/'example_sentence.json').read_text())
    from matplotlib.patches import Rectangle,Circle
    bounds=np.asarray(example['bounds']);gaze=example['gaze'];gaze_colors={'positive':'#68a22b','negative':'#d32127','neutral':'#4266c1','unknown':'#999999'}
    ax.set_facecolor('#eeeeee')
    for word,box in zip(example['words'],bounds):
        ax.text((box[0]+box[2])/2,(box[1]+box[3])/2,word,ha='center',va='center',fontsize=6)
    if gaze:
        x=[v['gaze_x'] for v in gaze];y=[v['gaze_y'] for v in gaze]
        ax.plot(x,y,color='#555555',alpha=.4,lw=.65)
        for row in gaze:
            color=gaze_colors.get(row.get('sentiment','unknown'),'#999999')
            ax.add_patch(Circle((row['gaze_x'],row['gaze_y']),max(7,np.sqrt(row['duration_ms'])*1.3),fill=False,lw=.8,ec=color))
            ax.plot(row['gaze_x'],row['gaze_y'],'.',color=color,ms=3)
    ax.set_xlim(bounds[:,0].min()-40,bounds[:,2].max()+40);ax.set_ylim(bounds[:,3].max()+50,bounds[:,1].min()-50)
    ax.set_aspect('equal');panel(ax,'A','Recorded word fixations · example sentence');ax.set_xticks([]);ax.set_yticks([])
    for i,label in enumerate(['positive','negative','neutral']):ax.text(1.02,.9-i*.13,label.title(),color=gaze_colors[label],transform=ax.transAxes,fontsize=8)
    flow=fig.add_subplot(grid[:,1]);flow.axis('off');panel(flow,'B','Fixation-event selection')
    durations=pd.to_numeric(events.duration_ms);quality=events.mapping_status.eq('matched')&(~events.blink_predecessor.astype(str).str.lower().isin(['true','1']))&(durations>=100)
    known=events.label_5.astype(int).between(-2,2);negation=events.preceded_by_negation.astype(str).str.lower().isin(['true','1'])
    before_pool=quality&known&events.sentiment.isin(['positive','negative'])&~negation
    accepted=events.selection_status.eq('accepted');final=events.loc[accepted].sentiment.value_counts()
    labels=events.loc[before_pool].label_5.astype(int).value_counts();n_pool=int(before_pool.sum());n_qual=int(quality.sum())
    before_text='Events before sentiment pooling\n'+'\n'.join(f'{name}: {int(labels.get(k,0)):,}' for k,name in [(2,'Very positive'),(1,'Positive'),(-1,'Negative'),(-2,'Very negative')])+f'\nN = {n_pool:,}'
    after_text=f'After sentiment pooling\nPositive: {int(labels.get(1,0)+labels.get(2,0)):,}\nNegative: {int(labels.get(-1,0)+labels.get(-2,0)):,}\nN = {n_pool:,}'
    texts=[f'Corrected fixation observations\nN = {len(events):,}',f'Word sentiment-based selection\nN = {n_qual:,}',before_text,after_text,
           f'Events included in analysis\nPositive: {int(final.get("positive",0)):,}\nNegative: {int(final.get("negative",0)):,}\nN = {int(accepted.sum()):,}']
    ys=[.9,.73,.49,.27,.07]
    for i,(text,y) in enumerate(zip(texts,ys)):
        flow.text(.27,y,text,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='square,pad=.6',fc='white',ec='#333333'),transform=flow.transAxes)
        if i<4:flow.annotate('',xy=(.27,ys[i+1]+.045),xytext=(.27,y-.045),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.3})
    excluded=[(.83,'Excluded\nUnmapped / out-of-word bounds\nBlink-preceded / <100 ms\nN = '+f'{len(events)-n_qual:,}'),
              (.62,'Excluded\nNeutral / unknown original label\nPreceding negation\nN = '+f'{n_qual-n_pool:,}'),
              (.17,'Excluded\nMatching / missing timing\nBlock boundaries / ±90 µV\nN = '+f'{n_pool-int(accepted.sum()):,}')]
    for y,text in excluded:
        flow.text(.78,y,text,ha='center',va='center',fontsize=7,bbox=dict(boxstyle='square,pad=.5',fc='white',ec='#777777'),transform=flow.transAxes)
        flow.annotate('',xy=(.60,y),xytext=(.48,y),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.2})
    sub=grid[1,0].subgridspec(3,2,hspace=.3,wspace=.16,height_ratios=[1,1,1.2]);times=stats[0]['times_ms']
    totals=sum(v['counts'].sum() for v in stats)
    mean=sum(v['sums'][:,:,:105].sum(axis=1) for v in stats)/totals
    pz=np.concatenate([v['pz_trials'] for v in stats],axis=1);duration=np.concatenate([v['pz_duration_ms'] for v in stats]);order=np.argsort(duration)
    pz_index=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.astype(str).tolist().index('E62')
    limit=max(float(abs(mean).max()),.01);trial_limit=max(float(np.quantile(abs(pz),.99)),.01)
    for r,title in enumerate(['FRP before cleaning','FRP after cleaning']):
        ax=fig.add_subplot(sub[0,r]);ax.plot(times,mean[r].T,color='black',lw=.35,alpha=.5);ax.plot(times,mean[r,pz_index],color='#c52828',lw=1)
        ax.axvline(0,color='black',ls=':',lw=.7);ax.set(xlim=(-600,1000),ylabel='µV' if r==0 else '',xticklabels=[])
        panel(ax,'C' if r==0 else '',title);ax.text(.04,.88,'E62 (Pz region)',color='#c52828',transform=ax.transAxes,fontsize=7)
        ax=fig.add_subplot(sub[1,r]);ax.imshow(mean[r],aspect='auto',origin='upper',extent=[-600,1000,105,1],cmap='RdYlBu_r',vmin=-limit,vmax=limit)
        ax.axvline(0,color='black',ls=':',lw=.7);ax.set(ylabel='Electrodes' if r==0 else '',xticklabels=[])
        ax=fig.add_subplot(sub[2,r]);im=ax.imshow(pz[r,order],aspect='auto',origin='upper',extent=[-600,1000,len(order),1],cmap='RdYlBu_r',vmin=-trial_limit,vmax=trial_limit)
        ax.plot(duration[order],np.arange(1,len(order)+1),color='black',lw=1);ax.axvline(0,color='black',ls=':',lw=.7)
        ax.set(xlim=(-600,1000),ylabel='Trials (duration-sorted)' if r==0 else '',xlabel='Time relative to fixation onset (ms)')
        if r==1:fig.colorbar(im,ax=ax,shrink=.6,label='µV',pad=.01)
    events.selection_status.value_counts().rename_axis('outcome').reset_index(name='fixations').to_csv(out/'analysis/exclusion_counts.csv',index=False)
    save(fig,out,'figure_1_selection_cleaning','Computed reconstruction, following panels A–C of the paper. Circles encode original word valence and recorded fixation duration.\nThe starting population is the released corrected eye export; upstream removals are not counted as new exclusions. Trial rasters preserve only E62, not another full EEG archive.')



def figure_frp(out,group,permutation,representation):
    plt=style();r=representation;times=group['times_ms'];counts=group['counts'];sums=group['sums'][:,r];xy=group['xy']
    pooled=sums.sum(axis=0)/counts.sum(axis=0)[:,None,None]
    reader_means=sums/counts[:,:,None,None]
    positive=pooled[1,:105];negative=pooled[0,:105];difference=positive-negative
    fig=plt.figure(figsize=(15,7.5));grid=fig.add_gridspec(2,3,hspace=.5,wspace=.38,height_ratios=[1,1.2])
    for col,(name,value) in enumerate([('Positive',positive),('Negative',negative),('Positive − negative',difference)]):
        ax=fig.add_subplot(grid[0,col]);limit=max(float(np.abs(value).max()),.01)
        image=ax.imshow(value,aspect='auto',origin='upper',extent=[-600,1000,105,1],cmap='RdYlBu_r',vmin=-limit,vmax=limit)
        ax.axvline(0,color='black',lw=.8,ls='--');ax.set(xlim=(-600,1000),xlabel='Time relative to fixation onset (ms)',ylabel='Electrodes')
        panel(ax,chr(65+col),name);fig.colorbar(image,ax=ax,shrink=.55,label='µV',pad=.02)
    def p_strip(spec,p,significant,label):
        ax=fig.add_subplot(spec[0]);ax.fill_between(times,0,1-p,color='#777777',lw=0)
        ax.axhline(.95,color='#b93333',ls=':',lw=.7);ax.set(xlim=(-600,1000),ylim=(0,1),ylabel=label,xticklabels=[],yticks=[0,1])
        ax.yaxis.label.set_size(7);ax.yaxis.labelpad=1
        ax=fig.add_subplot(spec[1]);ax.fill_between(times,0,1,where=significant,color='#c72228',step='mid')
        ax.axvline(0,color='#888888',lw=.6,ls='--');ax.set(xlim=(-600,1000),ylim=(0,1),xlabel='Time relative to fixation onset (ms)',ylabel='Stat. diff.',yticks=[])
        ax.yaxis.label.set_size(6);ax.yaxis.labelpad=1
    sub=grid[1,0].subgridspec(3,1,height_ratios=[4,1,.45],hspace=.12)
    ax=fig.add_subplot(sub[0]);panel(ax,'D','Global Field Power (GFP)')
    for ci,label in enumerate(['negative','positive']):
        # Trial-weighted mean; uncertainty from independent reader contributions.
        weights=counts[:,ci]/counts[:,ci].sum()
        line=pooled[ci,105];v=reader_means[:,ci,105]
        sem=np.sqrt(np.sum(weights[:,None]**2*(v-line)**2,axis=0)*len(counts)/(len(counts)-1))
        ax.plot(times,line,color=COLORS[label],label=label.upper());ax.fill_between(times,line-sem,line+sem,color=COLORS[label],alpha=.15)
    ax.axvline(0,color='#888888',lw=.6,ls='--');ax.set(xlim=(-600,1000),ylabel='GFP (µV)',xticklabels=[]);ax.legend(fontsize=7)
    p=group['p'][r,105];p_strip([sub[1],sub[2]],p,temporal_mask(p<.05),'1−Pval')
    sub=grid[1,1].subgridspec(3,1,height_ratios=[4,1,.45],hspace=.12)
    ax=fig.add_subplot(sub[0]);panel(ax,'E','Electrode-by-time analysis and TCT')
    image=(group['p'][r,:105]<.05).astype(float)
    ax.imshow(image,aspect='auto',origin='upper',extent=[-600,1000,105,1],cmap='Reds',vmin=0,vmax=1)
    contour=permutation['cluster_mask'][r]
    if contour.any() and not contour.all():ax.contour(times,np.arange(1,106),contour,levels=[.5],colors='black',linewidths=.8)
    ax.axvline(0,color='#777777',lw=.6,ls='--');ax.set(ylabel='Electrodes',xticklabels=[])
    p_strip([sub[1],sub[2]],permutation['tct_p'][r],permutation['tct_significant'][r],'TCT (1−Pval)')
    stop=280 if r==0 else 304;window=(times>=224)&(times<=stop)
    spec=grid[1,2].subgridspec(2,4,height_ratios=[1,2],hspace=.05,wspace=.1)
    title_ax=fig.add_subplot(spec[0,:]);title_ax.axis('off');panel(title_ax,'F',f'Average topographies (224–{stop} ms)')
    values=[positive[:,window].mean(axis=1),negative[:,window].mean(axis=1),difference[:,window].mean(axis=1),1-group['map_p'][r]]
    limit=max(float(np.max(np.abs(values[:3]))),.01)
    for index,(value,label) in enumerate(zip(values,['POS','NEG','POS−NEG','1−Pval'])):
        ax=fig.add_subplot(spec[1,index]);im=topography(ax,value,xy,1 if index==3 else limit,label,probability=index==3)
        if index in [1,3]:fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.9,pad=.08,label='1−Pval' if index==3 else 'µV')
    name='Raw FRPs' if r==0 else 'Regression-cleaned FRPs';fig.suptitle(name+' · ZuCo sentiment reading',fontweight='bold',y=.99)
    save(fig,out,f'figure_{r+2}_'+('raw_frp' if r==0 else 'cleaned_frp'),
         'Computed Python results in the published A–F panel arrangement. Heatmap limits scale to this run. GFP shading uses reader contributions.\nRed: pointwise mixed-model p<.05; black contours: cluster-corrected p<.05. TCT bars require ≥6 samples. Topography windows are predeclared reference windows, not detected effects.')



def figure4(out,stats,events,eye_model):
    plt=style();from scipy.stats import ttest_rel
    fig=plt.figure(figsize=(15,8));grid=fig.add_gridspec(2,3,hspace=.6,wspace=.35);times=stats[0]['times_ms'];means=np.stack([s['eye_means'] for s in stats])
    def stat_strips(sub,p,significant,x=times):
        ax=fig.add_subplot(sub[0]);ax.fill_between(x,0,1-p,color='#777777',lw=0);ax.axhline(.95,color='#b93333',ls=':',lw=.6)
        ax.set(xlim=(-600,1000),ylim=(0,1),ylabel='1−Pval',xticklabels=[],yticks=[0,1])
        ax.yaxis.label.set_size(7);ax.yaxis.labelpad=1
        ax=fig.add_subplot(sub[1]);ax.fill_between(x,0,1,where=significant,color='#c72228',step='mid');ax.axvline(0,color='#888888',lw=.6,ls=':')
        ax.set(xlim=(-600,1000),ylim=(0,1),xlabel='Time relative to fixation onset (ms)',ylabel='Stat. diff.',yticks=[])
        ax.yaxis.label.set_size(6);ax.yaxis.labelpad=1
    for direction,label in enumerate(['Horizontal eye velocity','Vertical eye velocity']):
        sub=grid[0,direction].subgridspec(3,1,height_ratios=[4,1,.4],hspace=.1);ax=fig.add_subplot(sub[0]);panel(ax,chr(65+direction),label)
        for ci,cls in enumerate(['negative','positive']):
            pick=eye_model['included_readers'];v=means[pick,ci,direction]
            if not len(v):continue
            avg=v.mean(axis=0);sem=v.std(axis=0,ddof=1)/np.sqrt(len(v));ax.plot(times,avg,color=COLORS[cls],label=cls.upper())
            ax.fill_between(times,avg-sem,avg+sem,color=COLORS[cls],alpha=.15)
        ax.axvline(0,color='#888888',lw=.6,ls=':');ax.set(xlim=(-600,1000),ylabel='Velocity (pixels/s)',xticklabels=[]);ax.legend(fontsize=7)
        p=eye_model['p'][direction];stat_strips([sub[1],sub[2]],p,temporal_mask(p<.05))
    sub=grid[0,2].subgridspec(4,1,height_ratios=[1.2,2,1,.4],hspace=.1);ax=fig.add_subplot(sub[0]);panel(ax,'C','Fixation onset probability')
    neighbor=np.stack([s['neighbor_means'] for s in stats]);edges=stats[0]['neighbor_edges'];centers=(edges[:-1]+edges[1:])/2
    palette=['#3f8a37','#2442c0','#c02225','#c02225','#2442c0','#3f8a37']
    for offset in range(6):ax.bar(centers,neighbor[:,:,offset].mean(axis=(0,1)),width=np.diff(edges)*.75,color=palette[offset],alpha=.75,label=f'n{[-3,-2,-1,1,2,3][offset]:+}')
    ax.set(xlim=(-600,1000),ylabel='Probability',xticklabels=[]);ax.legend(ncol=3,fontsize=5,frameon=False)
    ax=fig.add_subplot(sub[1]);total=neighbor.sum(axis=2)
    for ci,cls in enumerate(['negative','positive']):
        mean=total[:,ci].mean(axis=0);sem=total[:,ci].std(axis=0,ddof=1)/np.sqrt(len(total))
        ax.bar(centers+(-8 if ci==0 else 8),mean,width=14,yerr=sem,color=COLORS[cls],label=cls.upper(),error_kw={'lw':.5})
    ax.set(xlim=(-600,1000),ylabel='Probability',xticklabels=[]);ax.legend(fontsize=6)
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore',RuntimeWarning);p=ttest_rel(total[:,1],total[:,0],axis=0).pvalue
    stat_strips([sub[2],sub[3]],p,p<.05,centers)
    sub=grid[1,0].subgridspec(1,2,wspace=.24);before=events.eligible_for_matching.astype(str).str.lower().isin(['true','1']);after=events.selection_status.eq('accepted')
    for column,(keep,title) in enumerate([(before,'Before matching'),(after,'After matching')]):
        ax=fig.add_subplot(sub[column]);panel(ax,'D' if column==0 else '',title)
        for ci,cls in enumerate(['negative','positive']):
            x=pd.to_numeric(events.loc[keep&events.sentiment.eq(cls),'word_length']);hist,_=np.histogram(x,bins=np.arange(.5,21.5))
            ax.bar(np.arange(1,21)+(-.18 if ci==0 else .18),hist,width=.35,color=COLORS[cls],label=cls.upper())
        ax.set(xlabel='Number of characters',ylabel='Number of trials' if column==0 else '');ax.legend(fontsize=6)
    accepted=events[after].copy()
    for col,(field,title,ylabel) in enumerate([('duration_ms','Fixation duration','Fixation duration (ms)'),('word_frequency','Word frequency','Average occurrences in SR text')],1):
        ax=fig.add_subplot(grid[1,col]);panel(ax,chr(68+col),title)
        accepted[field]=pd.to_numeric(accepted[field],errors='coerce');values=accepted.groupby(['subject','sentiment'])[field].mean().unstack().dropna()
        for ci,cls in enumerate(['negative','positive']):
            if cls not in values:continue
            ax.bar(ci,values[cls].mean(),yerr=values[cls].std(ddof=1)/np.sqrt(len(values)),color=COLORS[cls],width=.65,capsize=5)
        if {'negative','positive'}.issubset(values) and len(values)>=3:
            test=ttest_rel(values.positive,values.negative);ax.text(.5,.97,f'Reader-paired p={test.pvalue:.3g}',ha='center',va='top',transform=ax.transAxes,fontsize=7)
        ax.set(xticks=[0,1],xticklabels=['NEG','POS'],ylabel=ylabel)
    save(fig,out,'figure_4_controls','Computed controls in the published A–F arrangement. Eye velocity uses recorded pixels/s and complete gaze epochs; shaded ±SEM across readers.\nNeighbor-bin, duration and frequency comparisons use reader-paired tests. Word frequency counts normalized occurrences in the released SR text. All methodological differences are recorded.')



def render_all(out):
    out=Path(out);readers,stats,events=load_readers(out)
    with np.load(out/'analysis/group_results.npz',allow_pickle=False) as z:group={k:z[k].copy() for k in z.files}
    with np.load(out/'analysis/permutation_results.npz',allow_pickle=False) as z:permutation={k:z[k].copy() for k in z.files}
    with np.load(out/'analysis/eye_model.npz',allow_pickle=False) as z:eye={k:z[k].copy() for k in z.files}
    figure1(out,readers,stats,events);figure_frp(out,group,permutation,0);figure_frp(out,group,permutation,1);figure4(out,stats,events,eye)
    files=list((out/'figures').glob('*'))
    write_json(out/'manifest.json',dict(status='completed',scope='ZuCo SR main figures 1-4; independent Python reconstruction',
               subjects=readers,figure_checksums={str(p.relative_to(out)):file_digest(p) for p in files}))
    print('FOUR MAIN FIGURES SAVED:',out/'figures',flush=True)
