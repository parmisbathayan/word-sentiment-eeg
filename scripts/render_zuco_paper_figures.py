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
    for r,name in enumerate(['raw','cleaned']):
        models.append(mixed_models(counts,sums[:,r],squares[:,r],checkpoint=analysis/f'{name}_ml_checkpoint.npz',
            progress=lambda a,b:print(f'MIXED MODEL {name}: {a}/{b} channel/time features',flush=True)))
    xy,adj=scalp_geometry(pd.read_csv(ROOT/'assets/zuco_chanlocs.csv'))
    pair_sum=np.sum([v['pair_sum'] for v in stats],axis=0);pair_squares=np.sum([v['pair_squares'] for v in stats],axis=0);n=sum(int(v['pairs']) for v in stats)
    if (analysis/'permutation_results.npz').is_file():
        with np.load(analysis/'permutation_results.npz',allow_pickle=False) as z:permutation={k:z[k].copy() for k in z.files}
    else:
        permutation,clusters=finish_permutations(out/'null_work',pair_sum,pair_squares,n,adj,readers)
        atomic_npz(analysis/'permutation_results.npz',**permutation);write_json(analysis/'clusters.json',clusters)
    atomic_npz(analysis/'group_results.npz',counts=counts,sums=sums,squares=squares,xy=xy,adjacency=adj,
               p=np.stack([v['p'] for v in models]),effect=np.stack([v['effect'] for v in models]),
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
    fig.text(.02,.015,caption,fontsize=7,color='#555555',va='bottom',wrap=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def topography(ax,values,xy,limit,title):
    from scipy.interpolate import griddata
    from matplotlib.patches import Circle
    grid=np.linspace(-1.05,1.05,100);xx,yy=np.meshgrid(grid,grid)
    z=griddata(xy,values,(xx,yy),method='cubic');z[xx*xx+yy*yy>1]=np.nan
    im=ax.imshow(z,extent=[grid[0],grid[-1],grid[0],grid[-1]],origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.add_patch(Circle((0,0),1,fill=False,lw=.7,color='black'));ax.plot([-.08,0,.08],[1,1.13,1],color='black',lw=.7)
    ax.scatter(xy[:,0],xy[:,1],s=1,color='black');ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.15),title=title);ax.axis('off');return im


def figure1(out,readers,stats,events):
    plt=style();fig=plt.figure(figsize=(14,9));grid=fig.add_gridspec(2,2,width_ratios=[1.2,1],hspace=.4,wspace=.3)
    ax=fig.add_subplot(grid[0,0]);example=json.loads((out/'readers'/readers[0]/'example_sentence.json').read_text())
    from matplotlib.patches import Rectangle
    bounds=np.asarray(example['bounds']);gaze=example['gaze']
    for word,box in zip(example['words'],bounds):
        ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,lw=.6,color='#aaaaaa'))
        ax.text((box[0]+box[2])/2,(box[1]+box[3])/2,word,ha='center',va='center',fontsize=6)
    if gaze:
        x=[v['gaze_x'] for v in gaze];y=[v['gaze_y'] for v in gaze]
        ax.plot(x,y,color='#555555',alpha=.5,lw=.8);ax.scatter(x,y,s=[max(3,v['duration_ms']/12) for v in gaze],color='#e59342',alpha=.5)
    ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');panel(ax,'A','Word boundaries and recorded fixations');ax.set(xlabel='Screen x (pixels)',ylabel='Screen y (pixels)')
    flow=fig.add_subplot(grid[:,1]);flow.axis('off');panel(flow,'B','Observed selection workflow')
    total=len(events);mapped=events.mapping_status.eq('matched').sum();nonneutral=events.sentiment.isin(['negative','positive']).sum()
    before=events.eligible_for_matching.astype(str).str.lower().isin(['true','1']).sum();accepted=events.selection_status.eq('accepted')
    counts=events.loc[accepted].sentiment.value_counts();outcomes=events.selection_status.value_counts()
    blocks=[('Corrected fixation observations',total),('Word-associated fixations',mapped),('Known positive/negative labels',nonneutral),
            ('Eligible before event matching',before),('Accepted raw AND cleaned epochs',int(accepted.sum()))]
    for i,(label,count) in enumerate(blocks):
        y=.87-i*.16;flow.text(.5,y,f'{label}\nN = {int(count):,}',ha='center',va='center',bbox=dict(boxstyle='square,pad=.6',fc='white',ec='#333333'),transform=flow.transAxes)
        if i<4:flow.annotate('',xy=(.5,y-.10),xytext=(.5,y-.035),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.4})
    flow.text(.5,.1,'Positive: '+str(int(counts.get('positive',0)))+'    Negative: '+str(int(counts.get('negative',0))),ha='center',transform=flow.transAxes)
    axes=grid[1,0].subgridspec(2,1,hspace=.45)
    for r,name in enumerate(['Before regression','After regression']):
        ax=fig.add_subplot(axes[r]);value=stats[0]['example'][r]
        ax.plot(stats[0]['times_ms'],value.T,lw=.4,alpha=.45);ax.axvline(0,color='black',lw=.6);ax.set(ylabel='µV',xlabel='Fixation onset (ms)')
        panel(ax,'C' if r==0 else '',name)
    outcomes.rename_axis('outcome').reset_index(name='fixations').to_csv(out/'analysis/exclusion_counts.csv',index=False)
    save(fig,out,'figure_1_selection_cleaning','Python reconstruction. Starting population is the released corrected eye export; upstream removals are not reconstructed as new exclusions. Full counts: analysis/exclusion_counts.csv.')


def figure_frp(out,group,permutation,representation):
    plt=style();r=representation;times=group['times_ms'];counts=group['counts'];sums=group['sums'][:,r];xy=group['xy']
    pooled=sums.sum(axis=0)/counts.sum(axis=0)[:,None,None]
    reader_means=sums/counts[:,:,None,None]
    positive=pooled[1,:105];negative=pooled[0,:105];difference=positive-negative
    fig=plt.figure(figsize=(14,9));grid=fig.add_gridspec(2,3,hspace=.5,wspace=.35,height_ratios=[1,1.25])
    channel_colors=plt.cm.hsv((np.arctan2(xy[:,1],xy[:,0])/(2*np.pi))%1)
    for col,(name,value) in enumerate([('Positive',positive),('Negative',negative),('Positive − negative',difference)]):
        ax=fig.add_subplot(grid[0,col])
        for channel,line in enumerate(value):ax.plot(times,line,color=channel_colors[channel],lw=.55,alpha=.75)
        ax.axvline(0,color='black',lw=.6);ax.axhline(0,color='#aaaaaa',lw=.4);ax.set(xlim=(-600,1000),xlabel='Fixation onset (ms)',ylabel='µV')
        panel(ax,chr(65+col),name+' · 105 electrodes')
    ax=fig.add_subplot(grid[1,0]);panel(ax,'D','Global field power')
    for ci,label in enumerate(['negative','positive']):
        line=pooled[ci,105];sem=reader_means[:,ci,105].std(axis=0,ddof=1)/np.sqrt(len(counts))
        ax.plot(times,line,color=COLORS[label],label=label.title());ax.fill_between(times,line-sem,line+sem,color=COLORS[label],alpha=.15)
    mask=temporal_mask(group['p'][r,105]<.05)
    ax.plot(times[mask],np.full(mask.sum(),ax.get_ylim()[0]),'.',color='#b41675',ms=3,label='ML p<.05, ≥6 samples')
    ax.axvline(0,color='black',lw=.6);ax.set(xlim=(-600,1000),xlabel='Fixation onset (ms)',ylabel='GFP (µV)');ax.legend(fontsize=7)
    sub=grid[1,1].subgridspec(3,1,height_ratios=[2,1,1],hspace=.45)
    ax=fig.add_subplot(sub[0]);panel(ax,'E','Electrode/time and topographic tests')
    image=(group['p'][r,:105]<.05).astype(float)
    ax.imshow(image,aspect='auto',origin='lower',extent=[-600,1000,0,105],cmap='Reds',vmin=0,vmax=1)
    contour=permutation['cluster_mask'][r]
    if contour.any() and not contour.all():ax.contour(times,np.arange(105),contour,levels=[.5],colors='black',linewidths=.8)
    ax.set(ylabel='Electrode',xticklabels=[])
    ax=fig.add_subplot(sub[1]);p=permutation['tct_p'][r];ax.plot(times,-np.log10(p),color='black',lw=.9);ax.axhline(-np.log10(.05),color='red',lw=.7,ls='--')
    ax.set(xlim=(-600,1000),ylabel='−log₁₀ p',xticklabels=[]);ax.set_title('TCT: independent electrode shuffles per paired trial',fontsize=7)
    ax=fig.add_subplot(sub[2]);ax.plot(times,permutation['tct_observed_gfp'][r],color='#415aaa',lw=1)
    significant=permutation['tct_significant'][r];ax.plot(times[significant],np.zeros(significant.sum()),'.',color='red',ms=3)
    ax.set(xlim=(-600,1000),xlabel='Fixation onset (ms)',ylabel='Difference GFP')
    top=grid[1,2].subgridspec(2,2,hspace=.25);window=(times>=224)&(times<=304)
    values=[positive[:,window].mean(axis=1),negative[:,window].mean(axis=1),difference[:,window].mean(axis=1)]
    limit=max(float(np.max(np.abs(values[:2]))),.01)
    axes=[fig.add_subplot(top[0,0]),fig.add_subplot(top[0,1]),fig.add_subplot(top[1,:])]
    for index,(axis,value,name) in enumerate(zip(axes,values,['Positive','Negative','Positive − negative'])):
        im=topography(axis,value,xy,limit if index<2 else max(float(abs(value).max()),.01),name)
        if index==2:fig.colorbar(im,ax=axis,shrink=.65,label='µV')
    axes[0].text(-.15,1.3,'F  224–304 ms reference window',transform=axes[0].transAxes,fontweight='bold',fontsize=9)
    name='Raw FRPs' if r==0 else 'Regression-cleaned FRPs'
    fig.suptitle(name+' · ZuCo sentiment reading',fontweight='bold',y=.99)
    save(fig,out,f'figure_{r+2}_'+('raw_frp' if r==0 else 'cleaned_frp'),
         'Computed Python results. Shading: ±1 SEM across readers. Red heatmap: uncorrected Gaussian mixed-model p<.05; black contours: cluster permutation p<.05.\nTopographies use the predeclared paper window, not a claim of significance in this run. See METHOD_DIFFERENCES.md and model convergence report.')


def figure4(out,stats,events,eye_model):
    plt=style();from scipy.stats import ttest_rel
    fig,axes=plt.subplots(2,3,figsize=(14,8));times=stats[0]['times_ms'];means=np.stack([s['eye_means'] for s in stats])
    for direction,label in enumerate(['Horizontal eye velocity','Vertical eye velocity']):
        ax=axes[0,direction];panel(ax,chr(65+direction),label)
        for ci,cls in enumerate(['negative','positive']):
            pick=eye_model['included_readers'];v=means[pick,ci,direction]
            if not len(v):continue
            avg=v.mean(axis=0);sem=v.std(axis=0,ddof=1)/np.sqrt(len(v));ax.plot(times,avg,color=COLORS[cls],label=cls)
            ax.fill_between(times,avg-sem,avg+sem,color=COLORS[cls],alpha=.15)
        ax.axvline(0,color='black',lw=.6);ax.set(xlim=(-600,1000),xlabel='Fixation onset (ms)',ylabel='pixels/second');ax.legend(fontsize=7)
        mask=temporal_mask(eye_model['p'][direction]<.05);ax.plot(times[mask],np.zeros(mask.sum()),'.',color='red',ms=3)
    ax=axes[0,2];panel(ax,'C','Neighboring fixation onset probabilities')
    neighbor=np.stack([s['neighbor_means'] for s in stats]);edges=stats[0]['neighbor_edges'];centers=(edges[:-1]+edges[1:])/2
    for ci,cls in enumerate(['negative','positive']):
        for offset in range(6):ax.plot(centers,neighbor[:,ci,offset].mean(axis=0),color=COLORS[cls],alpha=.45,lw=.8,label=cls if offset==0 else None)
    ax.set(xlim=(-600,1000),xlabel='Relative onset (ms); ±3 neighboring fixations',ylabel='Probability / 60-ms bin');ax.legend(fontsize=7)
    ax=axes[1,0];panel(ax,'D','Word length: before / after matching')
    before=events.eligible_for_matching.astype(str).str.lower().isin(['true','1']);after=events.selection_status.eq('accepted')
    for cls in ['negative','positive']:
        for keep,style_,name in [(before,'--','before'),(after,'-','accepted')]:
            x=pd.to_numeric(events.loc[keep&events.sentiment.eq(cls),'word_length'])
            hist,edges=np.histogram(x,bins=np.arange(.5,21.5));hist=hist/max(hist.sum(),1)
            ax.step(np.arange(1,21),hist,color=COLORS[cls],ls=style_,label=cls+' '+name,where='mid')
    ax.set(xlabel='Characters',ylabel='Proportion');ax.legend(fontsize=6)
    accepted=events[after].copy()
    for col,(field,title) in enumerate([('duration_ms','Fixation duration (ms)'),('frequency_zipf','English word frequency (Zipf)')],1):
        ax=axes[1,col];panel(ax,chr(68+col),title)
        if field not in accepted:continue
        accepted[field]=pd.to_numeric(accepted[field],errors='coerce')
        values=accepted.groupby(['subject','sentiment'])[field].mean().unstack().dropna()
        for ci,cls in enumerate(['negative','positive']):
            if cls not in values:continue
            x=np.full(len(values),ci)+np.linspace(-.08,.08,len(values));ax.scatter(x,values[cls],color=COLORS[cls],s=18)
            ax.errorbar(ci,values[cls].mean(),yerr=values[cls].std(ddof=1)/np.sqrt(len(values)),fmt='_',color=COLORS[cls],capsize=4,ms=20)
        if {'negative','positive'}.issubset(values) and len(values)>=3:
            test=ttest_rel(values.positive,values.negative);ax.set_title(f'{chr(68+col)}  {title}\nReader-paired p={test.pvalue:.3g}',loc='left',fontweight='bold',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['Negative','Positive'],ylabel=title)
    fig.tight_layout(rect=(0,.06,1,.97))
    save(fig,out,'figure_4_controls','Eye-velocity curves use complete native gaze epochs; mixed-model markers require ≥6 consecutive samples. C shows all six neighbor offsets.\nE–F show reader means ±SEM and reader-paired tests (explicit control-analysis departure). F uses wordfreq Zipf reference values. No published p-values are copied.')


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
