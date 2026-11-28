#!/usr/bin/env python3
"""One resumable cache-only Colab run: TeCo task1 / ZuCo SR EEG, PCA and VAE RSA."""
import argparse
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import linguistic_targets as labels
import analyze_trial_archive as archive
import cross_dataset_rsa_core as c
from word_sentiment_eeg.common import digest, file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

INPUTS=['full_0_800','window_200_400','regional_summary','normalized_0_800']
TARGETS=['sentiment_binary','neutral_nonneutral','pos','content_function','fixation_200']
PAIRS=[('teco','teco'),('zuco','zuco'),('teco','zuco')]
BUDGET=50_000_000
SPEC=dict(version=1,scope='TeCo task1 / ZuCo SR ONLY',seed=c.SEED,inputs=INPUTS,targets=TARGETS,permutations=9999,regroupings=3,
    averaging='exact40 per reader/task/class, balanced class group counts, max4 groups/class/reader, no remainders; 3 fixed regroupings',
    labels='Frozen original word_agreement binary/three-class prediction rosters; saved verified Stanza POS/content-function; recorded first-fixation duration <200 vs >=200ms',
    preparation='original shared 40Hz/125Hz accepted caches, NO extra paper regression cleaning; 6regions x8bins0–800ms; pre12bins-200–0ms',
    short_window='200–400ms from 100ms bins, NOT exact published224–304ms; no EEG reload/extraction to invent missing resolution',
    representation='joint train-only scalar equalization per dataset then shared median imputation/standardization; amplitudes not native-voltage comparison',
    modes='3 reader-heldout folds, joint encoder per fold; plus all-reader descriptive model trained on all valid cached trials (including evaluation data)',
    validation='Held-out mode: one training-side validation reader/dataset excluded from scaler/PCA/VAE fit; evaluation readers never fit; all-data validation overlaps fitting and is descriptive',
    vae='d→32Tanh→8mu/logvar, symmetric decoder; Adam1e-3, betaKL.05, max60epochs, patience8, equal dataset loss weights; no class labels used for fitting',
    pca='up to8 dimensions; training-only balanced subsample for fit',
    representations=['features','pca','vae_average_embeddings','vae_encode_average','pre_onset'],
    metrics='EEG/pre pattern Pearson; PCA/VAE cosine (shared coordinates, rotation-invariant); Spearman RDM vs0same/1different; no squared correlation',
    inference='One-sided reader-conditional whole-class-label swaps, shared per-reader permutations for all three folds; exclude self/within-reader pairs in square-matrix statistics; primary regroup0 only',
    correction='Holm and BH across ALL available held-out primary-regroup input/target/representation/matrix tests; regroup1/2 sensitivity only, all-data descriptive only',
    controls='pre-onset separate scaler on same training readers; four-property RDM diagnostic/partial-rank comparison; no duplicated-row metadata classification',
    limitations=['Test labels construct averages; not a decoder or individual-word accuracy',
    'Held-out readers may share word/sentence stimuli with training; source-overlap fractions reported',
    'Reader-conditional randomization assumes reader contrast exchangeability/symmetry; fixed stimuli, NOT generalization to random words or languages',
    'Reader/session, eye movement, class-related word properties and dataset differences can produce structure',
    'Feature summary/normalization transformations precede averaging, except explicitly named encode-average VAE',
    'No semantic-aspect annotations, classification reruns, supervised encoder fine-tuning or task2/3 analysis in this notebook'],budget=BUDGET)


def read(path):return json.loads(Path(path).read_text())


def csv_bytes(frame):return frame.to_csv(index=False).encode()


def load_cached(thesis):
    base=thesis/'Results/word_sentiment_eeg';a=base/'trial_archive_125hz';src=base/'classification_similarity'
    if read(src/'manifest.json').get('status')!='completed':raise ValueError('Completed original feature cache required')
    records,_=archive.load_completed(a)
    records=sorted([r for r in records if (r['dataset'],r['task']) in {('teco','task1'),('zuco','SR')}],key=lambda r:(r['dataset'],r['subject']))
    if len(records)!=24:raise ValueError('Expected12 readers each for TeCo task1 and ZuCo SR')
    tables=[];post=[];pre=[];proof=[]
    for i,r in enumerate(records,1):
        print(f'LOAD SAVED FEATURES subject {i}/{len(records)}: {r["dataset"]}/{r["task"]}/{r["subject"]}',flush=True)
        m,x,p,raw,identity=labels.cached_record(a,src,r)
        tables.append(m);post.append(x);pre.append(p);proof.append(dict(dataset=r['dataset'],**identity))
    m=pd.concat(tables,ignore_index=True);x=np.concatenate(post);p=np.concatenate(pre)
    if m.event_id.duplicated().any():raise ValueError('Duplicate cached event ID')
    dup=m.duplicated(['dataset','task','subject','sentence_position','onset_sample_sentence_0based'],keep=False)
    if 'signal_sha256' in m:dup |= m.signal_sha256.duplicated(keep=False)
    keep=~dup.to_numpy();m=m.loc[keep].reset_index(drop=True);x=x[keep];p=p[keep]
    if x.shape!=(len(m),48) or p.shape!=(len(m),12) or not np.isfinite(x).all() or not np.isfinite(p).all():raise ValueError('Unexpected/nonfinite cache features')
    m['source_key']=m.dataset.astype(str)+'/'+m.task.astype(str)+'/'+m.sentence_key.astype(str)+'/'+m.word_index.astype(str)
    m['reader']=m.dataset.astype(str)+'/'+m.subject.astype(str)
    m['word_identity']=m.word.map(labels.normalize)
    return m,x,p,dict(records=proof,removed_duplicate_events=int(dup.sum()),feature_spec=archive.FEATURE_SPEC)


def frozen_labels(thesis,meta,kind):
    values=np.full(len(meta),-1,int);proof=[]
    classes=['negative','positive'] if kind=='sentiment_binary' else ['neutral','nonneutral']
    for dataset,task in [('teco','task1'),('zuco','SR')]:
        target='binary' if kind=='sentiment_binary' else 'three_class'
        folder=thesis/f'Results/word_sentiment_eeg/classification_similarity/{dataset}/{task}/cases/word_agreement__{target}__sentence__post__logistic'
        marker=read(folder/'result.json');path=folder/'predictions.csv.gz'
        if marker['checksums']['predictions.csv.gz']!=file_digest(path):raise ValueError('Frozen sentiment roster changed: '+str(path))
        roster=pd.read_csv(path,keep_default_na=False,dtype={'event_id':str,'subject':str})
        expected={'negative','positive'} if target=='binary' else {'negative','neutral','positive'}
        if roster.event_id.duplicated().any() or set(roster.truth)!=expected or len(roster)!=marker['score']['trials']:raise ValueError('Invalid frozen label roster')
        ix=pd.Index(meta.event_id).get_indexer(roster.event_id);present=ix>=0
        # Original duplicated-waveform exclusions can remove roster events; report them.
        ix=ix[present];rr=roster.loc[present].reset_index(drop=True)
        if not meta.dataset.iloc[ix].eq(dataset).all() or not meta.task.iloc[ix].eq(task).all() or meta.subject.iloc[ix].astype(str).tolist()!=rr.subject.astype(str).tolist():raise ValueError('Frozen sentiment event identity mismatch')
        truth=rr.truth.to_numpy() if target=='binary' else np.where(rr.truth.eq('neutral'),'neutral','nonneutral')
        values[ix]=pd.Categorical(truth,categories=classes).codes
        proof.append(dict(dataset=dataset,sha256=file_digest(path),eligible_events=len(ix),removed_duplicate_events=int((~present).sum())))
    return values,classes,proof


def target_labels(thesis,meta,name):
    if name in ['sentiment_binary','neutral_nonneutral']:return frozen_labels(thesis,meta,name)
    if name=='fixation_200':
        v=pd.to_numeric(meta.first_fixation_duration_ms,errors='coerce').to_numpy(float)
        y=np.where(np.isfinite(v)&(v>0),(v>=200).astype(int),-1)
        return y,['short_<200ms','long_>=200ms'],dict(threshold_ms=200,unit='individual reader first stored fixation; not occurrence-median')
    y=np.full(len(meta),-1,int);proof=[]
    classes=['NOUN','VERB','ADJ'] if name=='pos' else ['content','function']
    for dataset,language in [('teco','fa'),('zuco','en')]:
        root=thesis/f'Results/word_sentiment_eeg/linguistic_targets/labels/{language}'
        marker=read(root/'complete.json');path=root/'words.csv.gz'
        if marker['sha256']!=file_digest(path):raise ValueError('Saved linguistic annotations changed')
        annotations=pd.read_csv(path,keep_default_na=False);ix=np.flatnonzero(meta.dataset.eq(dataset))
        joined=labels.attach(meta.iloc[ix].reset_index(drop=True),annotations)
        target=joined.upos if name=='pos' else joined.grammar_target
        good=joined.nlp_valid.to_numpy()&target.isin(classes).to_numpy()
        y[ix[good]]=pd.Categorical(target[good],categories=classes).codes
        proof.append(dict(dataset=dataset,label_sha256=file_digest(path),eligible_events=int(good.sum()),automatic=True))
    return y,classes,proof


def four_properties(meta):
    flag=np.where(meta.is_control.astype(str).str.lower().isin(['true','1']),1.,np.where(meta.is_control.astype(str).str.lower().isin(['false','0']),0.,np.nan))
    return np.column_stack([pd.to_numeric(meta[k],errors='coerce') for k in ['word_length','relative_word_position','sentence_length']]+[flag])


def torch_bytes(value):
    import torch
    b=io.BytesIO();torch.save(value,b);return b.getvalue()


def overwrite_state(store,path,value):
    """Mutable progress checkpoint, never renamed into Drive Trash; bounded size."""
    p=store.root/path;payload=torch_bytes(value);old=p.stat().st_size if p.exists() else 0
    store.check(max(0,len(payload)-old));p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    store.used+=len(payload)-old
    # Separate SHA makes torn writes explicit; previously completed models remain intact.
    (p.with_suffix('.sha256')).write_text(file_digest(p))


def load_torch(path):
    import torch
    try:return torch.load(path,map_location='cpu',weights_only=False)
    except Exception as error:raise ValueError('Unreadable own VAE checkpoint: '+str(path)) from error


def fit_representations(meta,X,fit,val,out,store,device,epochs,seed,evaluation=None):
    ds=meta.dataset.astype(str).to_numpy()
    balanced=c.balanced_sample(meta,fit,12000,seed);vix=c.balanced_sample(meta,val,3000,seed+1)
    transform=c.SharedTransform().fit(X[balanced],ds[balanced]);z=transform.transform(X,ds)
    pca=c.PCA(n_components=min(8,z.shape[1]-1),svd_solver='full').fit(z[balanced])
    modelpath=store.root/out/'model.pt';marker=store.root/out/'complete.json'
    if marker.exists():
        mark=read(marker)
        if mark['sha256']!=file_digest(modelpath):raise ValueError('Completed VAE model checksum failed')
        saved=load_torch(modelpath);model=c.make_vae(z.shape[1],min(8,z.shape[1]-1)).to(device);model.load_state_dict(saved['weights']);model.eval();history=saved['history']
        print('KEEP COMPLETE VAE:',out,flush=True)
    else:
        statepath=store.root/out/'progress.pt';state=None
        if statepath.exists():
            sha=statepath.with_suffix('.sha256')
            if sha.exists() and sha.read_text().strip()==file_digest(statepath):state=load_torch(statepath)
            else:print('Uncommitted/torn VAE progress ignored; rebuilding this unfinished small model',flush=True)
        model,history=c.train_vae(z[fit],z[vix],ds[fit],ds[vix],seed,epochs,device,state,
                                  lambda state:overwrite_state(store,out/'progress.pt',state))
        store.bytes(out/'model.pt',torch_bytes(dict(weights={k:v.detach().cpu() for k,v in model.state_dict().items()},history=history)))
        store.json(out/'complete.json',dict(sha256=file_digest(modelpath),epochs=len(history)))
    store.json(out/'history.json',history)
    emb=c.embeddings(model,z,device)
    evaluation=vix if evaluation is None else np.asarray(evaluation,int)
    diagnostic=c.latent_diagnostics(model,z[evaluation],ds[evaluation],meta.reader.iloc[evaluation].to_numpy(),device)
    diagnostic.update(fitting_trials=len(fit),validation_trials=len(vix),pca_variance_explained=pca.explained_variance_ratio_.tolist(),
                      dataset_scalar_scales=transform.scales,training_event_digest=digest(meta.event_id.iloc[fit].tolist()),
                      validation_event_digest=digest(meta.event_id.iloc[vix].tolist()),
                      diagnostic_evaluation_trials=len(evaluation),diagnostic_evaluation_readers=int(meta.reader.iloc[evaluation].nunique()))
    store.json(out/'diagnostics.json',diagnostic)
    return z,pca,model,emb,diagnostic


def save_matrix(store,relative,matrices,gm,classes,title,work,mode='all_data_descriptive'):
    """Save representative full-input/regroup0 matrices; all scores are saved."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13.5,8));arrays={}
    for column,((left,right),pack) in enumerate(matrices.items()):
        ax=axes[0,column]
        S=pack['similarity'].copy();ia=pack['left_indices'];ib=pack['right_indices']
        ordera=np.lexsort((gm.reader.iloc[ia].to_numpy(),gm.label.iloc[ia].to_numpy()))
        orderb=np.lexsort((gm.reader.iloc[ib].to_numpy(),gm.label.iloc[ib].to_numpy()))
        S=S[np.ix_(ordera,orderb)]
        if left==right:np.fill_diagonal(S,np.nan)
        image=ax.imshow(S,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set(title=f'{left} × {right}',xlabel='40-trial averages',ylabel='40-trial averages')
        for axis,ids,order in [('y',ia,ordera),('x',ib,orderb)]:
            positions=[];names=[]
            for ccode in sorted(gm.label.unique()):
                pos=np.flatnonzero(gm.label.iloc[ids].to_numpy()[order]==ccode)
                if len(pos):positions.append(float(pos.mean()));names.append(classes[int(ccode)])
            if axis=='x':ax.set_xticks(positions,names,fontsize=7)
            else:ax.set_yticks(positions,names,fontsize=7)
        ideal=pack['ideal_dissimilarity'][np.ix_(ordera,orderb)]
        axes[1,column].imshow(ideal,vmin=0,vmax=1,cmap='gray_r',aspect='auto')
        axes[1,column].set(title='Ideal dissimilarity: 0 same / 1 different',xlabel='40-trial averages',ylabel='40-trial averages')
        arrays[left+'_'+right]=pack['similarity'];arrays[left+'_'+right+'_ideal']=pack['ideal_dissimilarity'];arrays[left+'_'+right+'_mask']=pack['inference_mask']
    fig.colorbar(image,ax=axes[0,:].ravel().tolist(),label='Pattern similarity',fraction=.02,pad=.04)
    scope='All-data descriptive' if mode=='all_data_descriptive' else 'Held-out readers: '+mode
    fig.suptitle(title+'\n'+scope+'; class-sorted; each example averages 40 known same-class trials',fontsize=10)
    path=work/'matrix.png';fig.savefig(path,dpi=125,bbox_inches='tight');plt.close(fig)
    store.file(str(relative)+'.png',path,reuse_committed=True)
    b=io.BytesIO();np.savez_compressed(b,**arrays);store.bytes(str(relative)+'.npz',b.getvalue())


def run(meta,post,pre,targets,store,device='cpu',epochs=60,permutations=9999,regroupings=3,inputs=INPUTS):
    import tempfile
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    folds=c.reader_folds(meta);ds=meta.dataset.astype(str).to_numpy();properties=four_properties(meta)
    store.json('reader_folds.json',meta[['dataset','task','subject']].assign(fold=folds).drop_duplicates().to_dict('records'))
    rows=[];failures=[];nulls={};all_diagnostics=[]
    modes=[('reader_fold'+str(i+1),np.flatnonzero(folds!=i),np.flatnonzero(folds==i)) for i in range(3)]
    modes.append(('all_data_descriptive',np.arange(len(meta)),np.arange(len(meta))))
    with tempfile.TemporaryDirectory(prefix='cross-rsa-') as tmp:
        work=Path(tmp)
        for mi,(mode,tr,te) in enumerate(modes):
            if mode=='all_data_descriptive':fit=tr;_,val=c.validation_readers(meta,tr)
            else:
                fit,val=c.validation_readers(meta,tr)
                if set(meta.reader.iloc[tr])&set(meta.reader.iloc[te]):raise AssertionError('Reader leakage')
            # A property RDM is diagnostic only. It is not an old duplicated-row classifier.
            pi=SimpleImputer(strategy='median',keep_empty_features=True).fit(properties[fit]);ps=StandardScaler().fit(pi.transform(properties[fit]));prop=ps.transform(pi.transform(properties))
            pt=c.SharedTransform().fit(pre[c.balanced_sample(meta,fit,12000,c.SEED)],ds[c.balanced_sample(meta,fit,12000,c.SEED)])
            pre_scaled=pt.transform(pre,ds)
            for ii,inp in enumerate(inputs):
                print(f'REPRESENTATION {mi+1}/{len(modes)} {mode}; input {ii+1}/{len(inputs)} {inp}',flush=True)
                X=c.input_view(post,inp);prefix=Path('models')/mode/inp
                z,pca,model,emb,diagnostic=fit_representations(meta,X,fit,val,prefix,store,device,epochs,c.stable_seed(c.SEED,mode,inp),te)
                all_diagnostics.append(dict(mode=mode,input=inp,**diagnostic))
                for ti,(target,(y,classes)) in enumerate(targets.items()):
                    print(f'  TARGET {ti+1}/{len(targets)} {target}: exact40, {regroupings} fixed regroupings',flush=True)
                    for repeat in range(regroupings):
                        rel=Path('cases')/mode/inp/target/f'grouping{repeat}'
                        marker=store.root/rel/'result.json.gz'
                        if marker.exists():
                            saved=json.loads(gzip.decompress(marker.read_bytes()))
                            if digest({k:v for k,v in saved.items() if k!='checksum'})!=saved['checksum']:raise ValueError('Case checkpoint checksum failed')
                            rows.extend(saved['rows']);failures.extend(saved['unavailable'])
                            for k,v in saved['nulls'].items():nulls.setdefault(k,[]).append(v)
                            print('  KEEP completed RSA case:',rel,flush=True);continue
                        valid=te[y[te]>=0];case_rows=[];missing=[];case_null={}
                        try:groups,gm,excluded=c.exact_groups(meta,y,valid,40,4,c.stable_seed(c.SEED,target,mode,repeat),range(len(classes)))
                        except ValueError as e:
                            failures.append(dict(mode=mode,input=inp,target=target,regrouping=repeat,reason=str(e)));continue
                        # Same memberships across ALL input types and representation variants.
                        membership=Path('groups')/mode/target/f'grouping{repeat}.json.gz'
                        payload=dict(groups=[meta.event_id.iloc[g].tolist() for g in groups],metadata=gm.to_dict('records'),excluded_readers=excluded,classes=classes)
                        store.bytes(membership,gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0))
                        averaged_z=c.average(z,groups)
                        vectors={'features':averaged_z,'pca':pca.transform(averaged_z).astype(np.float32),
                                 'vae_average_embeddings':c.average(emb,groups),
                                 'vae_encode_average':c.embeddings(model,averaged_z,device)}
                        if inp=='full_0_800':vectors['pre_onset']=c.average(pre_scaled,groups)
                        props=c.average(prop,groups)
                        for rep,v in vectors.items():
                            packs={};metric='pearson' if rep in ['features','pre_onset'] else 'cosine'
                            for left,right in PAIRS:
                                setting=dict(mode=mode,input=inp,target=target,representation=rep,regrouping=repeat,matrix=left+'_'+right)
                                try:
                                    inference=repeat==0 and mode!='all_data_descriptive'
                                    result,null,pack=c.rsa_case(v,gm,left,right,metric,permutations if inference else 0,c.stable_seed(c.SEED,target),props)
                                    key='/'.join([inp,target,rep,left+'_'+right])
                                    if inference:case_null[key]=dict(null=null.tolist(),observed=result['rsa_rho'],mode=mode)
                                    case_rows.append(dict(**setting,**result,status='completed',trials_per_average=40,excluded_reader_tasks=len(excluded)))
                                    packs[(left,right)]=pack
                                except ValueError as error:missing.append(dict(**setting,reason=str(error)))
                            if mode in ['all_data_descriptive','reader_fold1'] and inp=='full_0_800' and repeat==0 and len(packs)==3:
                                save_matrix(store,Path('plots')/mode/target/rep,packs,gm,classes,f'{target} | {rep} | {metric}',work,mode)
                        saved=dict(rows=case_rows,rows_sha256=digest(case_rows),unavailable=missing,nulls=case_null)
                        saved['checksum']=digest(saved)
                        store.bytes(rel/'result.json.gz',gzip.compress(json.dumps(saved,sort_keys=True,allow_nan=False).encode(),mtime=0));rows.extend(case_rows);failures.extend(missing)
                        for k,v in case_null.items():nulls.setdefault(k,[]).append(v)
                        print(f'  SAVED {rel}: {len(case_rows)} matrices; {len(missing)} unavailable',flush=True)
    results=pd.DataFrame(rows)
    # Immutable case JSON sorts keys; export the same CSV column order on resume.
    results=results.reindex(sorted(results.columns),axis=1)
    summary=[]
    for key,parts in sorted(nulls.items()):
        inp,target,rep,pair=key.split('/')
        record=dict(input=inp,target=target,representation=rep,matrix=pair,regrouping=0,completed_folds=len(parts),
                    rsa_rho_mean=float(np.mean([p['observed'] for p in parts])),mode='held_out_readers')
        # Require all three reader folds: incomplete coverage never becomes a successful inference.
        if len(parts)==3:
            n=np.mean([p['null'] for p in parts],axis=0);obs=record['rsa_rho_mean']
            record['reader_conditional_p']=float((1+np.sum(n>=obs-1e-12))/(len(n)+1))
        else:record['reader_conditional_p']=None
        summary.append(record)
    frame=pd.DataFrame(summary)
    if len(frame):
        good=frame.reader_conditional_p.notna();frame['holm_p']=np.nan;frame['bh_fdr_q']=np.nan
        for method,column in [('holm','holm_p'),('bh','bh_fdr_q')]:frame.loc[good,column]=c.corrected_p(frame.loc[good,'reader_conditional_p'].to_numpy(),method)
        frame['correction_family_size']=int(good.sum())
    store.bytes('results.csv',csv_bytes(results));store.bytes('heldout_summary.csv',csv_bytes(frame))
    store.bytes('unavailable.csv',csv_bytes(pd.DataFrame(failures,columns=['mode','input','target','representation','regrouping','matrix','reason'])))
    store.json('vae_diagnostics.json',all_diagnostics)
    if len(frame):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(len(targets),3,figsize=(13,max(3,2.7*len(targets))),squeeze=False)
        for ri,target in enumerate(targets):
            for ci,(a,b) in enumerate(PAIRS):
                g=frame[(frame.target==target)&(frame.matrix==a+'_'+b)&(frame.input=='full_0_800')]
                ax=axes[ri,ci]
                if len(g):
                    ax.bar(np.arange(len(g)),g.rsa_rho_mean,color=['#33658a' if x=='features' else '#999999' if x=='pre_onset' else '#2a9d8f' for x in g.representation])
                    ax.set_xticks(np.arange(len(g)),[s.replace('vae_','VAE ').replace('_',' ') for s in g.representation],rotation=25,ha='right',fontsize=7)
                ax.axhline(0,color='black',lw=.6);ax.set_ylim(-1,1);ax.set_title(target+' | '+a+' × '+b,fontsize=9);ax.set_ylabel('Mean held-out RSA rho')
        fig.suptitle('Primary grouping; mean of 3 reader folds. Bars are effects, not significance. See corrected p/q table.',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.97));path=work/'overview.png' if work.exists() else Path(tempfile.gettempdir())/'rsa-overview.png'
        fig.savefig(path,dpi=125);plt.close(fig);store.file('overview.png',path,reuse_committed=True)
    return results,frame


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; no local EEG computation or ML training')
    import torch,fcntl,tempfile
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('Select T4 GPU, then Run all. No analysis started.')
    out=args.thesis_root/'Results/word_sentiment_eeg/cross_dataset_rsa_task1';store=ImmutableStore(out,BUDGET)
    lock=Path(tempfile.gettempdir(),'cross-dataset-rsa.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.free_drive_bytes<max(0,BUDGET-store.used):raise OSError('Reserve at most50MB additional Drive space for this run')
    versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib']}
    code={f:file_digest(ROOT/f) for f in ['scripts/cross_dataset_rsa.py','scripts/cross_dataset_rsa_core.py','scripts/linguistic_targets.py','scripts/teco_other_targets.py','scripts/teco_average40.py','scripts/analyze_trial_archive.py']}
    store.json('protocol.json',dict(spec=SPEC,code=code,software=versions,device=args.device))
    if (out/'complete.json').exists():
        for name,sha in read(out/'complete.json')['checksums'].items():
            if file_digest(out/name)!=sha:raise ValueError('Completed RSA output changed: '+name)
        print('KEEP COMPLETED CROSS-DATASET RSA: verified; no feature load or training',flush=True);return
    meta,post,pre,proof=load_cached(args.thesis_root);store.json('source_identity.json',proof)
    targets={};missing=[];counts=[];examples=[]
    for name in TARGETS:
        try:y,classes,identity=target_labels(args.thesis_root,meta,name)
        except FileNotFoundError as error:
            if name=='sentiment_binary':raise ValueError('Core frozen sentiment roster absent; no substitute labels allowed') from error
            missing.append(dict(target=name,reason='Required saved labels absent: '+str(error)));continue
        targets[name]=(y,classes);store.json('targets/'+name+'/definition.json',dict(classes=classes,proof=identity,counts={d:{classes[i]:int(np.sum((y==i)&meta.dataset.eq(d).to_numpy())) for i in range(len(classes))} for d in sorted(meta.dataset.unique())}))
        for (d,s),g in meta.groupby(['dataset','subject']):
            for i,cl in enumerate(classes):counts.append(dict(target=name,dataset=d,subject=s,label=cl,eligible_trials=int(np.sum(y[g.index]==i))))
        for d in sorted(meta.dataset.unique()):
            for i,cl in enumerate(classes):
                picked=meta.loc[(y==i)&meta.dataset.eq(d)].drop_duplicates('source_key').head(3)
                for r in picked.itertuples():examples.append(dict(target=name,dataset=d,label=cl,word=r.word,source_key=r.source_key))
    store.bytes('class_counts.csv',csv_bytes(pd.DataFrame(counts)));store.json('missing_targets.json',missing)
    store.bytes('label_examples.csv',csv_bytes(pd.DataFrame(examples)))
    results,summary=run(meta,post,pre,targets,store,args.device)
    store.json('complete.json',dict(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat(),scope=SPEC['scope'],
        targets=list(targets),result_rows=len(results),new_output_cap_bytes=BUDGET,
        checksums={str(f.relative_to(out)):file_digest(f) for f in out.rglob('*') if f.is_file() and f.name not in ['complete.json','runtime_status.json']}))
    print('CROSS-DATASET EEG/PCA/VAE RSA COMPLETE:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
