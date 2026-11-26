#!/usr/bin/env python3
"""One Colab run: audit historical experiments, independent controls, numeric EEG targets, confirmed refixations."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_other_targets as t
import linguistic_targets as l
import review_controls as c
import review_fixations as f

BUDGET=50_000_000
SPEC=dict(version=1,seed=c.SEED,budget=BUDGET,
    controls='four metadata variables, one source word occurrence per row; occurrence-random, unseen-sentence and unseen-word splits; fixed LR C1 and dummy',
    grain='Reader-identity occurrence control is undefined. Duration control uses median first-fixation duration across>=3readers per source occurrence, a different target explicitly named occurrence_median.',
    numeric='TeCo task1 individual EEG trials; word length,relative word position,sentence length regression; ridge1/ridge100/ExtraTrees,pre-onset ridge,dummy; familiar/sentence/word/reader splits; cap10000training events perfit',
    fixations='TeCo task1 first and second chronologically confirmed native fixations of the SAME word occurrence; all pairs kept together in reader/sentence/word splits; exact1/10/40/Max40; first and later averages use the same word pairs',
    native='Only paired-fixation step reads existing local Drive native task1.pickle once per unfinished reader; no source download, full EEG copy or saved epoch archive; compact pair features persisted',
    limits='50MB total output; historical files read-only; no feature ablation; no NLP/model downloads; known labels construct averages; exploratory, not confirmatory')


def checked(root,relative,state):
    path=root/relative;sha=state['checksums'].get(relative)
    if not sha or t.a.file_digest(path)!=sha:raise ValueError('Unverified historical file: '+str(path))
    return path


def saved_labels(root,language):
    path=root/f'labels/{language}/words.csv.gz';mark=t.read(root/f'labels/{language}/complete.json')
    if t.a.file_digest(path)!=mark['sha256']:raise ValueError('Saved automatic label cache changed')
    return pd.read_csv(path,keep_default_na=False)


def audit_history(thesis,all_meta,store):
    """Reconstruct exact outer groups, checking saved membership when available."""
    result=[];missing=[];base=thesis/'Results/word_sentiment_eeg'
    for family in ['teco_other_targets','teco_sentiment_unseen_readers','linguistic_targets']:
        root=base/family
        if not (root/'complete.json').is_file():missing.append(dict(family=family,reason='Completed run absent'));continue
        state=t.read(root/'complete.json')
        if state.get('status')!='completed':missing.append(dict(family=family,reason='Run not complete'));continue
        for targetpath in sorted((root/'targets').glob('*/events.csv.gz')):
            relative=str(targetpath.relative_to(root));checked(root,relative,state)
            events=pd.read_csv(targetpath,keep_default_na=False,dtype={'event_id':str});target=targetpath.parent.name
            ix=pd.Index(all_meta.event_id).get_indexer(events.event_id)
            if (ix<0).any():raise ValueError('Historical event missing from current validated caches')
            meta=all_meta.iloc[ix].reset_index(drop=True);y=events.target_code.to_numpy(int)
            # Recover the exact word identities used in the original named holdout.
            if 'word_identity' in events:meta['word_identity']=events.word_identity.to_numpy()
            split=l.safe_splits if family=='linguistic_targets' else t.split
            fold_cache={}
            members=sorted((root/'fits'/target).glob('*/*/*/outer_members.json.gz'))
            for member in members:
                mode,fold,size=member.parts[-4:-1];fi=int(fold.removeprefix('fold'));raw=size.removeprefix('size');size='max40' if raw=='max40' else int(raw)
                old=json.loads(gzip.decompress(member.read_bytes()))
                if mode not in fold_cache:fold_cache[mode]=split(meta,y,np.arange(len(meta)),mode,3,t.SEED)
                tr,te,_=fold_cache[mode][fi]
                gt=t.membership(meta,y,tr,size,t.SEED+fi+2000);ge=t.membership(meta,y,te,size,t.SEED+fi+3000)
                if [g.tolist() for g in gt]!=old['train'] or [g.tolist() for g in ge]!=old['test']:
                    raise ValueError('Historical memberships do not reproduce: '+str(member))
                result.append(c.audit_partition(meta,y,np.concatenate(gt),np.concatenate(ge),family,target,mode,fi+1,size,gt,ge))
    # First three sentiment notebooks have a frozen shared binary roster.
    binary_root=base/'teco_average40'
    if (binary_root/'complete.json').is_file():
        roster,_=t.old.frozen_roster(base/'classification_similarity')
        ix=pd.Index(all_meta.event_id).get_indexer(roster.event_id)
        if (ix<0).any():raise ValueError('Frozen binary events missing')
        meta=all_meta.iloc[ix].reset_index(drop=True);y=roster.truth.eq('positive').to_numpy(int)
        for family in ['teco_average40','teco_spatial40']:
            root=base/family
            pattern='cases/*/memberships.csv.gz' if family=='teco_average40' else 'size*/repeat*/memberships.csv.gz'
            seen=set()
            for path in sorted(root.glob(pattern)):
                members=pd.read_csv(path);signature=t.a.file_digest(path)
                if signature in seen:continue
                seen.add(signature)
                if family=='teco_average40':mode=path.parent.name.split('__')[0];size=path.parent.name.split('__')[1].removeprefix('max')
                else:mode='reader';size=path.parts[-3].removeprefix('size')
                for fi,part in members.groupby('fold'):
                    groups={}
                    for p,g in part.groupby('partition'):
                        eventgroups=[json.loads(v) for v in g.event_ids]
                        groups[p]=[pd.Index(meta.event_id).get_indexer(e) for e in eventgroups]
                        if any((a<0).any() for a in groups[p]):raise ValueError('Unknown historical group event')
                    gt,ge=groups['train'],groups['test']
                    result.append(c.audit_partition(meta,y,np.concatenate(gt),np.concatenate(ge),family,'sentiment_binary',mode,int(fi),size,gt,ge))
        root=base/'teco_classification_sweep'
        for path in sorted((root/'fits').glob('*/*/*/outer_membership.json')):
            mode,fold,size=path.parts[-4:-1];old=t.read(path);gt=[np.asarray(v,int) for v in old['train']];ge=[np.asarray(v,int) for v in old['test']]
            if t.a.digest(meta.event_id.iloc[np.concatenate(gt)].tolist())!=old['train_event_sha256'] or t.a.digest(meta.event_id.iloc[np.concatenate(ge)].tolist())!=old['test_event_sha256']:
                raise ValueError('Sweep event-group checksum mismatch')
            result.append(c.audit_partition(meta,y,np.concatenate(gt),np.concatenate(ge),'teco_classification_sweep','sentiment_binary',mode,int(fold.removeprefix('fold'))+1,size.removeprefix('size'),gt,ge))
    t.old.csv(store,'audit/historical_occurrence_overlap_and_group_sizes.csv',pd.DataFrame(result))
    t.old.csv(store,'audit/history_unavailable.csv',pd.DataFrame(missing,columns=['family','reason']))
    return result,missing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; do not process native EEG on the Mac')
    out=args.thesis_root/'Results/word_sentiment_eeg/experiment_review_followup';store=t.ImmutableStore(out,BUDGET)
    if args.free_drive_bytes<max(0,BUDGET-store.used):raise OSError('Reserve at most50MB; no source data read')
    import fcntl,tempfile
    lock=Path(tempfile.gettempdir(),'experiment-review-'+t.a.digest(str(out))[:16]+'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=args.thesis_root/'Results/word_sentiment_eeg';reference=sources/'linguistic_targets'
    if t.read(reference/'complete.json').get('status')!='completed':raise ValueError('Completed linguistic_targets required')
    files=['scripts/experiment_review_followup.py','scripts/review_controls.py','scripts/review_fixations.py','scripts/teco_other_targets.py','scripts/linguistic_targets.py']
    store.json('protocol.json',dict(spec=SPEC,code={f:t.a.file_digest(ROOT/f) for f in files},versions={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
        references={n:t.a.file_digest(sources/n/'complete.json') for n in ['teco_other_targets','teco_sentiment_unseen_readers','linguistic_targets'] if (sources/n/'complete.json').is_file()}))
    if (out/'complete.json').is_file():
        for name,sha in t.read(out/'complete.json')['checksums'].items():
            if t.a.file_digest(out/name)!=sha:raise ValueError('Changed completed output: '+name)
        print('KEEP COMPLETE EXPERIMENT REVIEW: all outputs verified',flush=True);return
    print('PHASE1/5: validated saved metadata/features and label examples',flush=True)
    tm,tf,tr,tp=l.load_pool(args.thesis_root,'teco');zm,zf,zr,zp=l.load_pool(args.thesis_root,'zuco')
    store.json('source_identity.json',dict(teco=tp,zuco=zp))
    tj=l.attach(tm,saved_labels(reference,'fa'));english=saved_labels(reference,'en')
    mask=zm.task.eq('SR').to_numpy();sj=l.attach(zm.loc[mask].reset_index(drop=True),english)
    sf={k:v[mask] for k,v in zf.items()};sr=zr[zr.task.eq('SR')]
    teco_source,_=l.source_tokens(tr);english_source,_=l.source_tokens(sr)
    pools={};example_rows=[]
    for name in l.TASKS:
        meta,vectors,raw,source=(sj,sf,sr,english_source) if name=='semantic_nouns' else (zm,zf,zr,english_source) if name=='zuco_reading_task' else (tj,tf,tr,teco_source)
        table,y,x,classes,plan,definition=l.target_data(name,meta,vectors,raw,source)
        pools[name]=(table,y,classes);example_rows+=c.examples(table,y,classes,name)
    for name in ['reader_identity','fixation_150','fixation_200','fixation_250','sentiment_binary','neutral_nonneutral','sentiment_three_class']:
        table,y,x,classes,info=t.prepare_target(args.thesis_root,tm,tf,name)
        pools[name]=(table,y,classes);example_rows+=c.examples(table,y,classes,name)
    t.old.csv(store,'label_examples.csv',pd.DataFrame(example_rows))
    print('PHASE2/5: exact historical split/group-size and source-occurrence overlap audit',flush=True)
    audit_history(args.thesis_root,pd.concat([tm,zm],ignore_index=True),store)
    print('PHASE3/5: standalone occurrence-level non-EEG controls',flush=True)
    scores=[];missing=[dict(target='reader_identity',reason='Cannot define reader identity with one reader-free row per source occurrence; historical reader control is not an occurrence-level prediction problem')]
    for name,(table,y,classes) in pools.items():
        if name=='reader_identity' or name.startswith('fixation_'):continue
        sizes=[1,5,10,20,40,'max40'] if name=='sentiment_binary' else [1,10,40,'max40']
        result,absent=c.run_standalone(table,y,classes,name,store,sizes);scores+=result;missing+=absent
    # Reader-specific duration labels cannot be silently collapsed. Export a separate contextual estimand.
    duration=tm.copy();duration['_value']=pd.to_numeric(duration.first_fixation_duration_ms,errors='coerce')
    duration=duration[np.isfinite(duration._value)&duration._value.gt(0)]
    for threshold in [150,200,250]:
        group=duration.groupby(c.identities(duration));keep=group.subject.transform('nunique').ge(3)
        table=duration.loc[keep].copy();median=table.groupby(c.identities(table))._value.transform('median')
        y=np.where(median<threshold,0,1);result,absent=c.run_standalone(table,y,['short_median','long_median'],f'fixation_{threshold}_occurrence_median',store)
        scores+=result;missing+=absent
    t.old.csv(store,'corrected_occurrence_controls.csv',pd.DataFrame(scores));t.old.csv(store,'control_unavailable.csv',pd.DataFrame(missing))
    print('PHASE4/5: EEG regression for word length,relative position,sentence length',flush=True)
    numeric,absent=c.run_numeric(tm,tf,store);t.old.csv(store,'eeg_numeric_predictions.csv',pd.DataFrame(numeric));t.old.csv(store,'numeric_unavailable.csv',pd.DataFrame(absent))
    print('PHASE5/5: first versus second confirmed fixations, same word and sentence',flush=True)
    pm,px,pb,absent=f.prepare_pairs(args.thesis_root,tm,tf,store)
    if pm is None:paired=[]
    else:
        t.old.csv(store,'confirmed_pair_roster.csv.gz',pm)
        paired,extra=f.analyze_pairs(pm,px,pb,store);absent+=extra
        sample=pm.iloc[0]
        t.old.csv(store,'fixation_label_examples.csv',pd.DataFrame([dict(label=label,word=sample.word,sentence=sample.get('sentence',''),reader=sample.subject,pair_id=sample.pair_id,onset=sample[col],duration_ms=sample[dur]) for label,col,dur in [('first','first_onset','first_duration_ms'),('second','later_onset','later_duration_ms')]]))
    t.old.csv(store,'confirmed_fixation_results.csv',pd.DataFrame(paired));t.old.csv(store,'fixation_unavailable.csv',pd.DataFrame(absent))
    files=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['complete.json','runtime_status.json']]
    store.json('complete.json',dict(status='completed',checksums={f:t.a.file_digest(out/f) for f in files},
        native_pair_scope='all12readers' if pm is not None and pm.subject.nunique()==12 else 'incomplete cohort; see fixation_unavailable.csv',
        corrected_control_rows=len(scores),numeric_rows=len(numeric),paired_rows=len(paired)))
    print('EXPERIMENT REVIEW FOLLOW-UP COMPLETE:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
