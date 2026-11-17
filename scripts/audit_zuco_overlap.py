"""Metadata-only audit of direct predictor-sample overlap in historical folds.

Refuses to report overlap counts unless regenerated event memberships match
the saved historical fold hashes. No signal loading, refitting or new selection.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import zuco_recovered_followups as z


def count_overlap(meta,train,test,span_samples):
    # Same offsets for both conditions: common sampled EEG is possible iff
    # centers differ by <= span. Require the SAME reader and source block.
    onsets=pd.to_numeric(meta.onset_seconds).to_numpy(dtype=float)
    if not np.isfinite(onsets).all():raise ValueError('Nonfinite historical onset; overlap counts withheld')
    centers=np.floor(onsets*125+.5).astype(np.int64)
    keys=list(zip(meta.subject.astype(str),meta.block_id.astype(str).str.removesuffix('.0')))
    train_hits=test_hits=pairs=0
    for key in set(keys[i] for i in train)&set(keys[i] for i in test):
        a=np.sort([centers[i] for i in train if keys[i]==key]);b=np.sort([centers[i] for i in test if keys[i]==key])
        counts=np.searchsorted(b,a+span_samples,side='right')-np.searchsorted(b,a-span_samples,side='left')
        pairs+=int(counts.sum());train_hits+=int((counts>0).sum())
        counts=np.searchsorted(a,b+span_samples,side='right')-np.searchsorted(a,b-span_samples,side='left')
        test_hits+=int((counts>0).sum())
    return dict(overlapping_train_events=train_hits,overlapping_test_events=test_hits,
                cross_partition_event_pairs=pairs,train_events=len(train),test_events=len(test))


def audit(thesis,store):
    root=thesis/'Results/word_sentiment_eeg';parent=root/'zuco_paper_figures_recovered'
    rows=[];unverified=[]
    try:
        cpu=z.read(parent/'cpu_preparation_manifest.json');frames=[]
        for subject in cpu['subjects']:
            folder=parent/'readers'/subject;state=z.read(folder/'prepared.json');path=folder/'accepted_trial_metadata.csv.gz'
            if z.file_digest(folder/'prepared.json')!=cpu['reader_checkpoint_sha256'][subject] or z.file_digest(path)!=state['checksums'][path.name]:
                raise ValueError('Historical parent metadata checksum mismatch')
            frame=pd.read_csv(path,keep_default_na=False);frame['subject']=subject;frames.append(frame)
        parent_meta=pd.concat(frames,ignore_index=True)
    except (FileNotFoundError,KeyError,ValueError) as error:
        return dict(verified_rows=[],unverified=[dict(cohort='all',reason=str(error))],scope='No EEG was read')
    cases=[('sst',root/'zuco_recovered_followups',parent_meta)]
    subset=root/'zuco_agent_paper_subset'
    try:
        done=z.read(subset/'complete.json');path=subset/'selection/trial_selection.csv.gz'
        if z.file_digest(path)!=done['checksums']['selection/trial_selection.csv.gz']:raise ValueError('Agent roster checksum mismatch')
        meta=pd.read_csv(path,keep_default_na=False);meta=meta[meta.subset_status.eq('retained')].copy().reset_index(drop=True)
        meta['sentiment']=meta.lexical_class
        cases.append(('agent',subset,meta))
    except (FileNotFoundError,KeyError,ValueError) as error:unverified.append(dict(cohort='agent',reason=str(error)))
    for cohort,out,meta in cases:
        meta=meta.copy();meta['word_key']=meta.word.map(z.clean_word)
        y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy()
        for mode in ['sentence','word','reader']:
            groups=meta[{'sentence':'sentence_key','word':'word_key','reader':'subject'}[mode]].to_numpy()
            folds=z.grouped_folds(y,groups)
            for window,span in [('published_224_304',10),('post_0_800',99)]:
                for size in [1,40]:
                    try:
                        marker=next((p for p in [out/'complete.json',out/'cpu_complete.json',out/'classification_complete.json'] if p.is_file()),None)
                        if marker is None:raise FileNotFoundError('Historical completion marker unavailable')
                        proof=z.read(marker)['checksums'];results=[]
                        for representation in ['raw','cleaned']:
                            relative=f'classification/{representation}__{window}__{mode}__avg{size}__logistic.json'
                            if z.file_digest(out/relative)!=proof[relative]:raise ValueError('Historical classification checksum mismatch')
                            results.append(z.read(out/relative))
                        candidate=[]
                        for fi,(train,test) in enumerate(folds):
                            _,_,_,ma=z.averaged(np.zeros((len(meta),1),np.float32),y,meta,train,size,z.SEED+fi)
                            _,_,_,mb=z.averaged(np.zeros((len(meta),1),np.float32),y,meta,test,size,z.SEED+100+fi)
                            for result in results:
                                saved=result['folds'][fi]
                                if saved['train_event_digest']!=z.digest(sorted(meta.event_id.iloc[ma].tolist())) or saved['test_event_digest']!=z.digest(sorted(meta.event_id.iloc[mb].tolist())):
                                    raise ValueError('Regenerated membership does not match saved fold hashes; counts withheld')
                            candidate.append(dict(cohort=cohort,split=mode,window=window,size=size,fold=fi+1,
                                **count_overlap(meta,ma,mb,span)))
                        rows.extend(candidate)
                    except (FileNotFoundError,KeyError,ValueError) as error:
                        unverified.append(dict(cohort=cohort,split=mode,window=window,size=size,reason=str(error)))
    return dict(verified_rows=rows,unverified=unverified,scope='Direct reuse of the same125Hz predictor samples only; no EEG reads.',
        limitations=['Overlapping stored epochs alone do not prove input-feature overlap.',
            '224–304ms uses11samples; 0–800ms features use100samples0–792ms.',
            'Does not assess filtering influence or full-reader regression-cleaning dependence.',
            'Counts are reported only after exact saved fold membership hashes agree; no significance claims.'])
