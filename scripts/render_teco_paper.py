"""Paper-style TeCo panels, computed values only; missing controls stay blank."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from render_zuco_paper_figures import style,panel,topography
from word_sentiment_eeg.teco_paper import TIMES,montage
from word_sentiment_eeg.paper_figure_stats import temporal_mask
COLORS=['#222222','#cb3335']


def save(fig,root,name,caption):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.text(.02,.01,caption,fontsize=8,va='bottom',color='#555555')
    for ext in ['png','svg']:fig.savefig(folder/(name+'.'+ext),dpi=160,bbox_inches='tight')
    style().close(fig)


def render(root,readers,stats,audit):
    plt=style();root=Path(root)
    with np.load(root/'analysis/group_results.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    with np.load(root/'analysis/permutation_results.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    counts=g['counts'];sums=g['sums'];C=len(montage());means=sums.sum(axis=0)/counts.sum(axis=0)[:,None,None]
    pos,neg=means[1,:C],means[0,:C];diff=pos-neg
    fig=plt.figure(figsize=(15,8));grid=fig.add_gridspec(2,3,hspace=.5,wspace=.33)
    for j,(title,value) in enumerate([('Positive',pos),('Negative',neg),('Positive − negative',diff)]):
        ax=fig.add_subplot(grid[0,j]);v=max(float(np.max(np.abs([pos,neg]))) if j<2 else float(abs(value).max()),.01)
        im=ax.imshow(value,origin='upper',aspect='auto',extent=[-600,1000,C,1],cmap='RdYlBu_r',vmin=-v,vmax=v)
        fig.colorbar(im,ax=ax,shrink=.8,label='Assumed µV')
        ax.axvline(0,color='black',ls=':',lw=.6);ax.set(xlabel='Fixation onset (ms)',ylabel='Channels');panel(ax,chr(65+j),title+' (µV)')
    ax=fig.add_subplot(grid[1,0]);panel(ax,'D','Global field power: mean of trial GFPs')
    rm=sums/counts[:,:,None,None]
    for c,name in enumerate(['Negative','Positive']):
        # Same reader-contribution uncertainty convention as the ZuCo renderer.
        v=rm[:,c,C];w=counts[:,c]/counts[:,c].sum()
        sem=np.sqrt(np.sum(w[:,None]**2*(v-means[c,C])**2,axis=0)*len(v)/(len(v)-1))
        ax.plot(TIMES,means[c,C],color=COLORS[c],label=name);ax.fill_between(TIMES,means[c,C]-sem,means[c,C]+sem,color=COLORS[c],alpha=.15)
    ax.set(xlabel='Fixation onset (ms)',ylabel='GFP (assumed µV)');ax.legend(fontsize=8)
    sub=grid[1,1].subgridspec(2,1,height_ratios=[4,1]);ax=fig.add_subplot(sub[0]);panel(ax,'E','Pointwise ML tests; corrected cluster contours')
    ax.imshow(g['p'][:C]<.05,origin='upper',aspect='auto',extent=[-600,1000,C,1],cmap='Reds',vmin=0,vmax=1)
    mask=p['cluster_mask']
    if mask.any() and not mask.all():ax.contour(TIMES,np.arange(1,C+1),mask,levels=[.5],colors='black',linewidths=.8)
    ax.set(ylabel='Channels');ax=fig.add_subplot(sub[1]);ax.plot(TIMES,1-p['tct_p'],color='#555555',lw=.8)
    ax.fill_between(TIMES,0,1,where=p['tct_significant'],color='#cb3335',alpha=.4);ax.axhline(.95,color='black',ls=':',lw=.6)
    ax.set(xlabel='Fixation onset (ms)',ylabel='TCT 1−p',ylim=(0,1))
    sub=grid[1,2].subgridspec(1,3);window=(TIMES>=224)&(TIMES<=304)
    values=[pos[:,window].mean(axis=1),neg[:,window].mean(axis=1),diff[:,window].mean(axis=1)];limit=max(float(abs(np.array(values)).max()),.01)
    for j,(name,value) in enumerate(zip(['POS','NEG','POS−NEG'],values)):
        ax=fig.add_subplot(sub[0,j]);topography(ax,value,g['xy'],limit,name)
        if j==0:ax.text(0,1.35,'F  Reference window: 224–304 ms',transform=ax.transAxes,fontsize=9,fontweight='bold')
    fig.suptitle(f'TeCo Task 1 · raw FRP adaptation · {len(readers)} readers · {int(counts.sum()):,} accepted fixations',fontweight='bold')
    save(fig,root,'figure_2_raw_frp','No full blink/saccade/keypress regression available; no cleaned-FRP result is implied. Maps use a fixed reference window.\nRed heatmap: uncorrected pointwise p<.05. Black: cluster-corrected p<.05. TCT requires ≥6 consecutive samples. No baseline subtraction.')
    # Selection report + pooled all-channel traces + duration-sorted individual Pz trials.
    fig,axes=plt.subplots(1,3,figsize=(15,6));panel(axes[0],'A','Selection audit (word-associated events only)');axes[0].axis('off')
    counts_a=audit.selection_status.value_counts();text='\n'.join(f'{name.replace("_"," ")}: {int(n):,}' for name,n in counts_a.items())
    axes[0].text(0,.96,text,va='top',fontsize=8,transform=axes[0].transAxes)
    panel(axes[1],'B','All-channel FRPs (not their signed scalp mean)')
    allmean=sums[:,:,:C].sum(axis=(0,1))/counts.sum();axes[1].plot(TIMES,allmean.T,color='black',lw=.4,alpha=.35)
    pz=montage().labels.str.lower().tolist().index('pz');axes[1].plot(TIMES,allmean[pz],color='#cb3335',label='Pz');axes[1].legend();axes[1].set(xlabel='Fixation onset (ms)',ylabel='Assumed µV')
    rasters=[];dur=[]
    for reader in readers:
        folder=root/'readers'/reader
        with np.load(folder/'prepared_trials.npz',allow_pickle=False) as z:rasters.append(z['signal'][:,pz])
        dur.extend(pd.read_csv(folder/'accepted_trial_metadata.csv.gz').duration_ms.to_list())
    x=np.concatenate(rasters);order=np.argsort(dur);v=max(float(np.quantile(abs(x),.99)),.01)
    axes[2].imshow(x[order],aspect='auto',extent=[-600,1000,len(x),1],origin='upper',cmap='RdYlBu_r',vmin=-v,vmax=v)
    axes[2].plot(np.array(dur)[order],np.arange(1,len(x)+1),color='black',lw=.7);axes[2].set(xlabel='Fixation onset (ms)',ylabel='Accepted trials');panel(axes[2],'C','Pz individual trials, sorted by duration')
    fig.suptitle('TeCo selection and raw fixation response',fontweight='bold')
    save(fig,root,'figure_1_selection','Includes control reading; answer signals excluded. No scanpath/cleaned panel is fabricated. Stored fixations are not the full eye-event timeline.')
    fig,axes=plt.subplots(2,3,figsize=(15,8));accepted=audit[audit.selection_status.eq('accepted')]
    for ax,title in zip(axes[0],['Horizontal eye velocity','Vertical eye velocity','Full neighboring fixation timeline']):
        panel(ax,'',title);ax.axis('off');ax.text(.5,.5,'Unavailable in validated native pickle schema\nNot tested; no zero-valued substitute',ha='center',va='center',transform=ax.transAxes,fontsize=10)
    vocab=audit.drop_duplicates(['sentence_key','word_index']);freq=vocab.word.value_counts()
    for j,(column,title) in enumerate([('word_length','Word length'),('duration_ms','Fixation duration (ms)'),('frequency','Word frequency in available task text')]):
        ax=axes[1,j];panel(ax,chr(68+j),title)
        vals=[]
        for c,name in enumerate(['negative','positive']):
            values=accepted.loc[accepted.sentiment.eq(name)]
            v=values.word.map(freq).to_numpy(float) if column=='frequency' else values[column].to_numpy(float)
            vals.append(v);ax.hist(v,bins=15,alpha=.4,color=COLORS[c],density=True,label=name)
        ax.legend(fontsize=8);ax.set(ylabel='Density')
    save(fig,root,'figure_4_controls','Descriptive control distributions, not proof of equivalence. Full oculomotor controls and regression-cleaned Figure 3 require additional native event sources.')
    summary=dict(total_trials=int(counts.sum()),readers=len(readers),minimum_cluster_p=min((c['p'] for c in json.loads((root/'analysis/clusters.json').read_text())),default=None),
        significant_clusters=sum(c['p']<.05 for c in json.loads((root/'analysis/clusters.json').read_text())),
        tct_significant_times_ms=TIMES[p['tct_significant']].tolist(),raw_only=True)
    (root/'analysis/RESULTS.md').write_text('# TeCo raw-FRP adaptation\n\n'+json.dumps(summary,indent=2)+'\n\nSee METHOD_DIFFERENCES.md before interpretation.\n')
