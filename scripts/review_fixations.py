"""First versus second confirmed fixation on the SAME word occurrence."""
from io import BytesIO
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
import teco_other_targets as t
import review_controls as c
from word_sentiment_eeg.teco import find_raw_files, orient_native, NATIVE_HZ
from word_sentiment_eeg.teco_paper import segments
from word_sentiment_eeg.three_level_erp import preprocess40
from word_sentiment_eeg.trial_archive import make_epoch
from word_sentiment_eeg.erp import locate


def confirmed_pair(raw, word, durations):
    values=segments(word)
    count=np.asarray(word.get('nFixations',0),float).reshape(-1)
    positions=np.asarray(word.get('fixPositions',[]),float).reshape(-1)
    if len(count)!=1 or not np.isfinite(count[0]) or count[0]!=len(values) or len(values)<2:
        raise ValueError('No consistent repeated fixation count')
    if len(positions)!=len(values) or not np.isfinite(positions).all() or not np.equal(positions,np.floor(positions)).all():
        raise ValueError('Invalid native fixation-order indices')
    if len(set(positions))!=len(positions):raise ValueError('Duplicate native fixation-order index')
    order=np.argsort(positions,kind='stable')
    result=[]
    for rank,j in enumerate(order[:2]):
        segment=orient_native(values[j]);index=int(positions[j])-1
        if segment is None or segment.shape[0]!=raw.shape[0] or segment.shape[1]<3 or not np.isfinite(segment).all():
            raise ValueError('Invalid first/second native fixation waveform')
        duration=(segment.shape[1]-1)*1000/NATIVE_HZ
        if index<0 or index>=len(durations) or not np.isfinite(durations[index]) or abs(float(durations[index])-duration)>5:
            raise ValueError('Waveform duration disagrees with native fixation order/duration')
        onset,method=locate(raw,segment)
        if onset is None:raise ValueError('Native waveform cannot be uniquely localized: '+method)
        result.append(dict(onset=int(onset),duration_ms=float(duration),native_order=int(positions[j]),stored_index=int(j)))
    if result[1]['onset']<=result[0]['onset']:raise ValueError('Native fixation order disagrees with EEG onset order')
    if result[1]['onset']<result[0]['onset']+orient_native(values[order[0]]).shape[1]-2:
        raise ValueError('First and second native fixation intervals overlap')
    return result


def reader_pairs(data, reader_meta, reference_post, regions, progress=None):
    lookup={(int(r.sentence_id_teco),str(r.sentence_key),int(r.word_index)):i for i,r in reader_meta.iterrows()}
    if len(lookup)!=len(reader_meta):raise ValueError('Repeated cached source word occurrence')
    rows=[];post=[];pre=[];audit=[]
    for ti,(_,trial) in enumerate(data.items(),1):
        raw=orient_native(trial.get('raw_data'));filtered=None
        sid=int(trial['sentenceId']);key=hashlib.sha256(str(trial.get('persian_sentence','')).encode()).hexdigest()[:24]
        durations=np.asarray(trial.get('allFixations',{}).get('duration',[]),float).reshape(-1)
        for wi,word in sorted(trial.get('word',{}).items(),key=lambda p:int(p[0])):
            pos=lookup.get((sid,key,int(wi)))
            if pos is None:continue
            source=reader_meta.loc[pos]
            if str(word.get('content','')).strip()!=str(source.word).strip():raise ValueError('Source/cached token changed')
            reason='accepted'
            try:
                if raw is None or raw.shape[0]!=len(regions) or not np.isfinite(raw).all():raise ValueError('Invalid sentence EEG')
                pair=confirmed_pair(raw,word,durations)
                if pair[0]['onset']!=int(source.onset_sample_sentence_0based):raise ValueError('Earliest native fixation is not the cached first stored fixation')
                if filtered is None:filtered=preprocess40(raw,NATIVE_HZ)
                epochs=[]
                for r in pair:
                    epoch,_,status=make_epoch(filtered,r['onset'],NATIVE_HZ)
                    if epoch is None:raise ValueError('Pair epoch unavailable: '+status)
                    epochs.append(epoch)
                x,b=t.a.feature_blocks(np.stack(epochs),np.arange(-25,101)*8,regions)
                if not np.allclose(x[0],reference_post[pos],rtol=1e-4,atol=1e-4):
                    raise RuntimeError('First-fixation features do not reproduce the saved preparation; stop before mixing pipelines')
                d=source.to_dict();d.update(pair_id=t.a.digest([str(source.event_id),'confirmed_first_second'])[:32],
                    first_onset=pair[0]['onset'],later_onset=pair[1]['onset'],first_duration_ms=pair[0]['duration_ms'],
                    later_duration_ms=pair[1]['duration_ms'],first_native_order=pair[0]['native_order'],later_native_order=pair[1]['native_order'])
                rows.append(d);post.append(x);pre.append(b)
            except ValueError as e:reason=str(e)
            audit.append(dict(subject=str(source.subject),event_id=str(source.event_id),sentence_key=key,word_index=int(wi),status=reason))
        if progress:progress(ti,len(data))
    table=pd.DataFrame(rows) if rows else pd.DataFrame(columns=[*reader_meta.columns,'pair_id','first_onset','later_onset','first_duration_ms','later_duration_ms','first_native_order','later_native_order'])
    return table,np.asarray(post,np.float32).reshape(-1,2,48),np.asarray(pre,np.float32).reshape(-1,2,12),audit


def prepare_pairs(thesis, meta, features, store):
    sources,_=find_raw_files(thesis/'Data');root=thesis/'Results/word_sentiment_eeg/trial_archive_125hz'
    regions=pd.read_csv(root/'teco_channel_regions.csv').region.astype(str).tolist()
    tables=[];xs=[];bs=[];missing=[]
    for i,reader in enumerate(sorted(meta.subject.unique()),1):
        prefix=f'fixations/cache/{reader}';marker=store.root/(prefix+'/complete.json')
        rows=meta[meta.subject.eq(reader)].copy();identity=dict(archive_checkpoint=t.a.file_digest(root/f'teco/readers/task1/{reader}/checkpoint.json'),events=rows.event_id.tolist(),version='same_occurrence_confirmed_first_second_v1')
        if marker.is_file():
            state=t.read(marker)
            if state['identity']!=identity:raise ValueError('Changed paired-fixation source identity')
            for p,sha in state['checksums'].items():
                if t.a.file_digest(store.root/p)!=sha:raise ValueError('Corrupt paired-fixation feature checkpoint')
            print(f'KEEP FIXATION FEATURES reader{i}/12: {reader}; no native EEG reread',flush=True)
            table=pd.read_csv(store.root/(prefix+'/pairs.csv.gz'),keep_default_na=False)
            with np.load(store.root/(prefix+'/features.npz'),allow_pickle=False) as f:x=f['post'];b=f['pre']
        else:
            path=sources.get(reader)
            if path is None:
                missing.append(dict(subject=reader,reason='Native task1.pickle required for later fixations; no new download attempted'));continue
            checkpoint=t.read(root/f'teco/readers/task1/{reader}/checkpoint.json')
            if t.a.file_digest(path)!=checkpoint['identity']['source_sha256']:raise ValueError('Native source differs from validated first-fixation archive')
            print(f'CONFIRMED FIXATION PREP reader{i}/12: {reader}; one native read, compact features only',flush=True)
            with path.open('rb') as stream:data=pickle.load(stream)
            table,x,b,audit=reader_pairs(data,rows,features['regional_bins'],regions,
                progress=lambda k,n:print(f'FIXATION SOURCE {reader}: sentence{k}/{n}',flush=True) if k%10==0 or k==n else None)
            del data
            if table.empty:
                missing.append(dict(subject=reader,reason='No confirmed in-bounds first/second pairs'))
            t.old.csv(store,prefix+'/pairs.csv.gz',table)
            payload=BytesIO();np.savez_compressed(payload,post=x,pre=b);store.bytes(prefix+'/features.npz',payload.getvalue())
            t.old.csv(store,prefix+'/audit.csv.gz',pd.DataFrame(audit))
            paths=[prefix+'/pairs.csv.gz',prefix+'/features.npz',prefix+'/audit.csv.gz']
            store.json(prefix+'/complete.json',dict(identity=identity,checksums={p:t.a.file_digest(store.root/p) for p in paths},pairs=len(table)))
        if not table.empty:tables.append(table);xs.append(x);bs.append(b)
    if not tables:return None,None,None,missing
    return pd.concat(tables,ignore_index=True),np.concatenate(xs),np.concatenate(bs),missing


def pair_folds(meta, mode):
    group=meta.subject if mode=='reader' else meta.sentence_key if mode=='sentence' else meta.word_identity
    if group.nunique()<3:raise ValueError('Fewer than three independent groups')
    result=[]
    for train,test in GroupKFold(3).split(np.zeros(len(meta)),groups=group):
        if set(group.iloc[train])&set(group.iloc[test]):raise AssertionError('Pair holdout leaked')
        # Purge an entire TRAIN PAIR if either fixation overlaps either held-out fixation epoch.
        keep=[]
        for i in train:
            same=meta.iloc[test];same=same[same.subject.eq(meta.subject.iloc[i])&same.sentence_key.eq(meta.sentence_key.iloc[i])]
            forbidden=same[['first_onset','later_onset']].to_numpy().ravel()
            current=meta[['first_onset','later_onset']].iloc[i].to_numpy()
            if not len(forbidden) or np.min(np.abs(current[:,None]-forbidden[None,:]))/1.2>1000:keep.append(i)
        result.append((np.asarray(keep,int),np.asarray(test,int)))
    return result


def pair_groups(meta,indices,size,seed):
    rng=np.random.default_rng(seed);cap=40 if size=='max40' else int(size);out=[]
    for reader in sorted(meta.subject.iloc[indices].unique()):
        ix=rng.permutation(indices[meta.subject.iloc[indices].to_numpy()==reader])
        for start in range(0,len(ix),cap):
            g=ix[start:start+cap]
            if len(g)==cap or (size=='max40' and len(g)):out.append(g)
    return out


def score_pairs(x,train,test,model):
    if len(train)<2 or len(test)<2:raise ValueError('Need two paired averages in train and test')
    def values(groups):
        a=np.stack([x[g].mean(axis=0) for g in groups]);return a.reshape(-1,a.shape[-1]),np.tile([0,1],len(groups))
    xt,yt=values(train);xe,ye=values(test);est=t.estimator(model,2);est.fit(xt,yt);pred=est.predict(xe)
    return dict(truth=ye.tolist(),prediction=pred.tolist(),train_groups=c.size_summary(train),test_groups=c.size_summary(test))


def analyze_pairs(meta,post,pre,store):
    if meta.subject.nunique()<6:
        return [],[dict(target='confirmed_fixation_repetition',reason='Fewer than six readers have confirmed pairs; no partial-cohort classifier reported')]
    control=c.metadata_four(meta);control=np.repeat(control[:,None,:],2,axis=1)
    assert np.array_equal(control[:,0],control[:,1],equal_nan=True)
    result=[];missing=[]
    for mode in ['reader','sentence','word']:
        try:folds=pair_folds(meta,mode)
        except ValueError as e:missing.append(dict(target='confirmed_fixation_repetition',mode=mode,reason=str(e)));continue
        for size in [1,10,40,'max40']:
            print(f'PAIRED FIXATION CLASSIFIER {mode}: average={size}; {len(meta)}confirmed word pairs',flush=True)
            for name,x,model in [('fixed_eeg',post,'lr_1'),('lda_eeg',post,'lda'),('pre_control',pre,'lr_1'),('metadata_four',control,'lr_1'),('dummy',post,'dummy')]:
                cases=[]
                for fi,(tr,te) in enumerate(folds):
                    gt=pair_groups(meta,tr,size,c.SEED+fi);ge=pair_groups(meta,te,size,c.SEED+100+fi)
                    cases.append(t.case(store,f'fixations/results/{mode}/size{size}/{name}/fold{fi}',lambda:score_pairs(x,gt,ge,model)))
                if any(r['status']!='completed' for r in cases):missing.append(dict(target='confirmed_fixation_repetition',mode=mode,size=str(size),model=name,reason='Incomplete paired-average coverage'));continue
                truth=np.concatenate([r['truth'] for r in cases]);pred=np.concatenate([r['prediction'] for r in cases]);ba=float(balanced_accuracy_score(truth,pred))
                if name=='metadata_four' and abs(ba-.5)>1e-12:raise AssertionError('Identical paired word properties unexpectedly classified fixation order')
                result.append(dict(target='confirmed_fixation_repetition',mode=mode,size=str(size),model=name,
                    balanced_accuracy=ba,macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),chance=.5,test_examples=len(truth),
                    interpretation='same source word occurrence; first/second pairs kept together; identical word composition per class average'))
    return result,missing
