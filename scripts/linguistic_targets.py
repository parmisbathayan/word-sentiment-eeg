#!/usr/bin/env python3
"""Remaining EEG targets: TeCo linguistic labels and ZuCo reading task, cached EEG only."""
import argparse,gzip,json,re,unicodedata,warnings
from pathlib import Path
from io import BytesIO
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import teco_other_targets as t
import teco_average40 as old
import analyze_trial_archive as a
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score
SEED=20261121;CAP=10000
TASKS=['pos','content_function','word_identity5','word_identity10','repetition','semantic_nouns','zuco_reading_task']
SPEC=dict(version=1,seed=SEED,split_and_model_seed=t.SEED,targets=TASKS,budget=50_000_000,train_cap_examples=CAP,
 linguistic_data='TeCo task1 all accepted cached epochs, not sentiment-filtered; complete source-token metadata for annotations and repetition',
 labels='Stanza1.10.1 fa/en default_fast resources1.10.0, pretokenized original word positions; English WordNet noun sense consensus on ZuCo SR only',
 semantics='ZuCo SR English only: physical_entity.n.01 vs abstraction.n.06 ancestors; all noun senses must agree; ambiguous/unmapped excluded; NOT human concreteness ratings or contextual sense disambiguation',
 repetition='first vs later stored textual presentation per reader/task, using ALL source word records, including unfixated words; not refixation or lifetime familiarity',
 identity='top5/top10 forms by source-occurrence count, >=3source occurrences and>=6readers; vocabulary fixed before EEG fitting; no unseen-word class test',
 tasks='ZuCo SR/NR/TSR; full feature pools; distinguish recording/task conditions, which are confounded with text and session',
 evaluation='3outer/2inner folds; familiar, unseen sentence, unseen reader, plus unseen word where meaningful; exact1,exact10,max40',
 controls='fixed C1 logistic, pre-onset, word-properties-only, dummy; no text/word IDs/POS labels/task IDs as EEG predictors',
 cap='Per fit, deterministic stratified sample of at most10000training examples BEFORE preprocessing; all heldout examples retained; record selected training-group digest',
 limits=['Automated labels, not agent or human consensus; every annotation and exclusion exported',
 'Test class labels construct averages, not single-word deployment accuracy',
 'Model selection by validation only; no exact paper replication or significance claims',
 'No EEG extraction or waveform reads; new Drive outputs<=50MB; NLP models stay on temporary Colab disk'])
CONTENT={'NOUN','PROPN','VERB','ADJ','ADV'}
FUNCTION={'ADP','AUX','CCONJ','DET','PART','PRON','SCONJ'}


def normalize(word):
    return unicodedata.normalize('NFKC',str(word)).casefold().replace('ي','ی').replace('ك','ک').strip()


def cached_record(archive,source,r):
    d,task,subject=r['dataset'],r['task'],r['subject'];reader=archive/d/'readers'/task/subject;cache=source/'features'/d/task/subject
    checkpoint=t.read(reader/'checkpoint.json');manifest=t.read(cache/'manifest.json')
    if checkpoint['status']!='completed' or checkpoint['epochs']!=r['epochs']:raise ValueError('Incomplete reader cache')
    if a.file_digest(reader/'events.csv.gz')!=checkpoint['events_sha256']:raise ValueError('Source metadata checksum failed')
    if checkpoint['identity']['protocol_sha256']!=a.digest(t.read(archive/'protocol.json')):raise ValueError('Archive protocol mismatch')
    signature=a.digest(dict(spec=a.FEATURE_SPEC,checkpoint=a.file_digest(reader/'checkpoint.json')))
    if manifest['signature']!=signature:raise ValueError('Cached feature specification mismatch')
    for f,sha in manifest['checksums'].items():
        if a.file_digest(a.strict_path(cache,f))!=sha:raise ValueError('Corrupt cached vectors')
    meta=pd.read_csv(cache/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
    with np.load(cache/'features.npz',allow_pickle=False) as f:
        if f['event_id'].astype(str).tolist()!=meta.event_id.tolist():raise ValueError('Cached feature event mismatch')
        post=f['post'].copy();pre=f['pre'].copy()
    if len(meta)!=r['epochs'] or not meta.dataset.eq(d).all() or not meta.task.eq(task).all() or not meta.subject.eq(subject).all():raise ValueError('Reader data identity mismatch')
    # Read source EVENT TABLE only. Never open an EEG shard, pickle or MATLAB file.
    raw=pd.read_csv(reader/'events.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
    return meta,post,pre,raw,dict(reader=subject,task=task,cache_sha256=a.file_digest(cache/'manifest.json'),source_metadata_sha256=checkpoint['events_sha256'])


def load_pool(thesis,dataset):
    archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz';source=thesis/'Results/word_sentiment_eeg/classification_similarity'
    if t.read(source/'manifest.json')['status']!='completed':raise ValueError('Existing feature cache preparation incomplete')
    records,_=a.load_completed(archive);records=sorted([r for r in records if r['dataset']==dataset],key=lambda r:(r['task'],r['subject']))
    tables=[];posts=[];pres=[];raws=[];proof=[]
    for i,r in enumerate(records,1):
        print(f'LOAD CACHE {dataset} {i}/{len(records)} {r["task"]}/{r["subject"]}; metadata/vectors only',flush=True)
        m,x,b,raw,identity=cached_record(archive,source,r);tables.append(m);posts.append(x);pres.append(b);raws.append(raw);proof.append(identity)
    meta=pd.concat(tables,ignore_index=True);post=np.concatenate(posts);pre=np.concatenate(pres);raw=pd.concat(raws,ignore_index=True)
    if meta.event_id.duplicated().any():raise ValueError('Repeated event ID')
    duplicate=meta.duplicated(['dataset','task','subject','sentence_position','onset_sample_sentence_0based'],keep=False)|meta.signal_sha256.duplicated(keep=False)
    keep=~duplicate.to_numpy();meta=meta.loc[keep].reset_index(drop=True);meta['word_identity']=meta.word.map(normalize);features=t.regional_inputs(post[keep],pre[keep])
    control=np.where(meta.is_control.astype(str).str.lower().isin(['true','1']),1.,np.where(meta.is_control.astype(str).str.lower().isin(['false','0']),0.,np.nan))
    features['metadata_control']=np.column_stack([pd.to_numeric(meta[c],errors='coerce') for c in ['word_length','relative_word_position','sentence_length']]+[control]).astype(np.float32)
    return meta,features,raw,dict(records=proof,removed_duplicate_events=int(duplicate.sum()))


def source_tokens(raw):
    required={'sentence_key','word_index','word'}
    if not required<=set(raw):raise ValueError('Full source-token metadata absent; do not infer context from fixated words only')
    source=raw[['sentence_key','word_index','word']].copy();source['word_index']=pd.to_numeric(source.word_index,errors='coerce')
    source=source[source.word_index.notna()&source.word.astype(str).str.len().gt(0)]
    source['word_index']=source.word_index.astype(int);source=source[source.word_index>=0]
    source['normalized_word']=source.word.map(normalize)
    conflict=source.groupby(['sentence_key','word_index']).normalized_word.nunique()
    bad=set(conflict[conflict>1].index.get_level_values(0))
    source=source[~source.sentence_key.isin(bad)].drop_duplicates(['sentence_key','word_index']).sort_values(['sentence_key','word_index']).reset_index(drop=True)
    return source,sorted(bad)


def noun_semantics(word,lemma,wn,language='eng'):
    senses=[]
    for form in dict.fromkeys([normalize(lemma),normalize(word)]):
        if form and form!='_':senses.extend(wn.synsets(form.replace(' ','_'),pos=wn.NOUN,lang=language))
    unique={s.name():s for s in senses};labels=set()
    for synset in unique.values():
        ancestors={s.name() for path in synset.hypernym_paths() for s in path}
        physical='physical_entity.n.01' in ancestors;abstract='abstraction.n.06' in ancestors
        labels.add('physical' if physical and not abstract else 'abstract' if abstract and not physical else 'unknown')
    if not unique:return 'unknown','no_mapped_noun_sense'
    if len(labels)!=1 or 'unknown' in labels:return 'unknown','ambiguous_or_unresolved_senses'
    return next(iter(labels)),'all_mapped_noun_senses_agree'


def linguistic_labels(source,store,model_dir,nlp_factory=None,wordnet=None,language='fa'):
    prefix=f'labels/{language}'
    identity=dict(source_sha256=a.digest(source.to_dict('records')),method=SPEC['labels'],semantics=SPEC['semantics'],language=language)
    store.json(prefix+'/protocol.json',identity)
    marker=store.root/prefix/'complete.json';path=store.root/prefix/'words.csv.gz'
    if marker.is_file():
        saved=t.read(marker)
        if saved['identity']!=identity or a.file_digest(path)!=saved['sha256']:raise ValueError('Changed/corrupt annotation checkpoint')
        print('KEEP COMPLETE NLP LABELS; no NLP model download or annotation rerun',flush=True)
        return pd.read_csv(path,keep_default_na=False)
    nlp=None;rows=[];sentences=list(source.groupby('sentence_key',sort=True))
    for si,(key,g) in enumerate(sentences,1):
        relative=prefix+'/sentences/'+a.digest(str(key))[:24]+'.json.gz';p=store.root/relative
        sentence_identity=a.digest(g.to_dict('records'))
        if p.is_file():
            saved=json.loads(gzip.decompress(p.read_bytes()))
            if saved['source_sha256']!=sentence_identity:raise ValueError('Sentence annotation source changed')
            rows.extend(saved['rows']);continue
        if nlp is None:
            if nlp_factory is None:
                import stanza,nltk
                model_dir.mkdir(parents=True,exist_ok=True)
                stanza.download(language,model_dir=str(model_dir/'stanza'),package='default_fast',processors='tokenize,pos,lemma',resources_version='1.10.0',verbose=False)
                nlp=stanza.Pipeline(language,dir=str(model_dir/'stanza'),package='default_fast',processors='tokenize,pos,lemma',tokenize_pretokenized=True,use_gpu=False,download_method=None,resources_version='1.10.0',verbose=False)
                if language=='en':
                    nltk.data.path.insert(0,str(model_dir/'nltk'))
                    if not nltk.download('wordnet',download_dir=str(model_dir/'nltk'),quiet=True,raise_on_error=True):raise ValueError('WordNet download failed')
                    from nltk.corpus import wordnet as wn
                    wordnet=wn
                    wn.synsets('banana',pos=wn.NOUN,lang='eng')
            else:nlp=nlp_factory()
        words=g.word.astype(str).tolist();doc=nlp([words]);tokens=[w for sentence in doc.sentences for w in sentence.words]
        aligned=len(tokens)==len(words) and all(normalize(w.text)==normalize(text) for w,text in zip(tokens,words))
        current=[]
        for j,r in enumerate(g.itertuples()):
            token=tokens[j] if aligned else None;upos=str(token.upos) if token else 'X';lemma=str(token.lemma) if token else ''
            category,reason=noun_semantics(r.word,lemma,wordnet) if upos=='NOUN' and language=='en' else ('unknown','semantic_language_not_covered' if language!='en' else 'not_common_noun' if aligned else 'token_alignment_failed')
            current.append(dict(sentence_key=str(key),word_index=int(r.word_index),word=str(r.word),normalized_word=normalize(r.word),upos=upos,lemma=lemma,
                grammatical_class='content' if upos in CONTENT else 'function' if upos in FUNCTION else 'unknown',semantic_class=category,semantic_reason=reason,
                annotation_status='automatic_stanza_wordnet' if aligned else 'excluded_alignment'))
        t.zipped_json(store,relative,dict(source_sha256=sentence_identity,rows=current));rows.extend(current)
        if si==1 or si%20==0 or si==len(sentences):print(f'ANNOTATE {language} SENTENCE {si}/{len(sentences)}; saved independently',flush=True)
    labels=pd.DataFrame(rows);old.csv(store,prefix+'/words.csv.gz',labels)
    store.json(prefix+'/complete.json',dict(identity=identity,sha256=a.file_digest(path),sentences=len(sentences),words=len(labels),automatic_labels=True))
    return labels


def attach(meta,labels):
    keys=['sentence_key','word_index'];left=meta.copy();left['word_index']=pd.to_numeric(left.word_index).astype(int);left['_row']=np.arange(len(left))
    right=labels.copy();right['word_index']=pd.to_numeric(right.word_index).astype(int)
    if right.duplicated(keys).any():raise ValueError('Ambiguous annotation identity')
    renamed=right.rename(columns={'word':'annotation_word','normalized_word':'annotation_normalized'})
    merged=left.merge(renamed,on=keys,how='left',validate='many_to_one',sort=False,suffixes=('','_nlp')).sort_values('_row').reset_index(drop=True)
    valid=merged.word.map(normalize).eq(merged.annotation_normalized.fillna(''))&merged.annotation_status_nlp.fillna('').eq('automatic_stanza_wordnet') if 'annotation_status_nlp' in merged else merged.word.map(normalize).eq(merged.annotation_normalized.fillna(''))&merged.annotation_status.fillna('').eq('automatic_stanza_wordnet')
    # Never allow an old coarse grammatical_class to masquerade as a new POS label.
    for col in ['upos','lemma','semantic_class','semantic_reason']:
        if col+'_nlp' in merged:merged[col]=merged[col+'_nlp']
    if 'grammatical_class_nlp' in merged:merged['grammar_target']=merged.grammatical_class_nlp
    else:merged['grammar_target']=merged.grammatical_class
    merged['nlp_valid']=valid
    return merged


def repetition_labels(raw):
    result={}
    for _,g in raw.groupby(['dataset','task','subject'],sort=True):
        g=g.copy();g['_sentence']=pd.to_numeric(g.sentence_position,errors='coerce');g['_word']=pd.to_numeric(g.word_index,errors='coerce')
        if g[['_sentence','_word']].isna().any().any():raise ValueError('Repetition requires complete numeric source order')
        g=g.sort_values(['_sentence','_word'],kind='stable');seen=set();positions=set()
        for r in g.itertuples():
            if r.event_id in result:raise ValueError('Repeated source event identity')
            occurrence=(r.sentence_position,r.word_index)
            if occurrence in positions:raise ValueError('Repeated source word occurrence')
            positions.add(occurrence);word=normalize(r.word)
            result[r.event_id]='repeat' if word in seen else 'first';seen.add(word)
    return result


def safe_splits(meta,y,indices,mode,n,seed):
    if mode=='word':
        from sklearn.model_selection import StratifiedGroupKFold
        groups=meta.word_identity.iloc[indices].astype(str).to_numpy();raw=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(np.zeros(len(indices)),y[indices],groups)
        output=[]
        shifted=meta.copy();shifted['sentence_position']=meta.dataset.astype(str)+'/'+meta.task.astype(str)+'/'+meta.sentence_position.astype(str)
        for tr,te in raw:
            tr,te=np.asarray(indices)[tr],np.asarray(indices)[te];before=len(tr);clean,_=a.purge_overlapping_train(shifted,y,[(tr,te)]);tr=clean[0][0]
            if set(y[tr])!=set(y[indices]) or set(y[te])!=set(y[indices]):raise ValueError('Class coverage missing after word split')
            if set(meta.word_identity.iloc[tr])&set(meta.word_identity.iloc[te]):raise AssertionError('Word identity leakage')
            output.append((tr,te,dict(purged_training=before-len(tr),discarded_unfamiliar_test=0)))
        return output
    shifted=meta.copy();shifted['sentence_position']=meta.dataset.astype(str)+'/'+meta.task.astype(str)+'/'+meta.sentence_position.astype(str)
    return t.split(shifted,y,indices,mode,n,seed)


def fast_means(X,groups):
    sizes={len(g) for g in groups}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if sizes=={1}:return X[np.array([g[0] for g in groups])]
        if len(sizes)==1:return np.nanmean(X[np.stack(groups)],axis=1)
        return np.stack([np.nanmean(X[g],axis=0) for g in groups])


def limited_score(meta,y,X,groups,model_name,nclasses,detailed=False):
    from sklearn.model_selection import train_test_split
    train,test=groups;yt=np.array([y[g[0]] for g in train]);ye=np.array([y[g[0]] for g in test]);required=set(range(nclasses))
    if set(yt)!=required or set(ye)!=required or np.bincount(ye,minlength=nclasses).min()<2:raise ValueError('Too few target-class averages')
    selected=np.arange(len(train))
    if len(train)>CAP:selected,_=train_test_split(selected,train_size=CAP,stratify=yt,random_state=SEED)
    fit_groups=[train[i] for i in selected];model=t.estimator(model_name,nclasses)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(fast_means(X,fit_groups),yt[selected])
    pred=model.predict(fast_means(X,test));sizes=[len(g) for g in test]
    result=dict(balanced_accuracy=float(balanced_accuracy_score(ye,pred)),macro_f1=float(f1_score(ye,pred,average='macro',zero_division=0)),confusion=confusion_matrix(ye,pred,labels=range(nclasses)).tolist(),
        test_examples=len(test),test_events=int(sum(sizes)),train_examples=len(train),actual_fit_examples=len(selected),fit_group_indices_sha256=a.digest(selected.tolist()),
        actual_group_min=int(min(sizes)),actual_group_max=int(max(sizes)),actual_group_mean=float(np.mean(sizes)),warnings=sorted(set(str(w.message)[:250] for w in caught)))
    if detailed:result.update(truth=ye.tolist(),prediction=pred.astype(int).tolist())
    return result


def target_data(name,meta,features,raw,source=None):
    if name=='pos':labels=meta.upos.where(meta.nlp_valid&meta.upos.isin(['NOUN','VERB','ADJ']))
    elif name=='content_function':labels=meta.grammar_target.where(meta.nlp_valid&meta.grammar_target.isin(['content','function']))
    elif name=='semantic_nouns':labels=meta.semantic_class.where(meta.nlp_valid&meta.semantic_class.isin(['physical','abstract'])&meta.dataset.eq('zuco')&meta.task.eq('SR'))
    elif name=='repetition':
        mapping=repetition_labels(raw);labels=meta.event_id.map(mapping)
        if labels.isna().any():raise ValueError('Accepted event absent from full source-order metadata')
    elif name.startswith('word_identity'):
        size=int(name.removeprefix('word_identity'));observed=meta.groupby('word_identity').agg(readers=('subject','nunique'),events=('event_id','size'))
        frequencies=source.assign(identity=source.word.map(normalize)).groupby('identity').size().rename('occurrences')
        eligible=observed.join(frequencies).fillna(0);eligible=eligible[(eligible.readers>=6)&(eligible.occurrences>=3)&(eligible.events>=50)]
        eligible=eligible[[bool(re.search(r'\w',str(k))) for k in eligible.index]].sort_values(['occurrences','events'],ascending=False,kind='stable')
        if len(eligible)<size:raise ValueError(f'Only{len(eligible)}words meet predetermined repetition/reader coverage; need{size}')
        vocabulary=eligible.head(size).index.tolist();labels=meta.word_identity.where(meta.word_identity.isin(vocabulary))
    elif name=='zuco_reading_task':labels=meta.task
    else:raise ValueError(name)
    ix=np.flatnonzero(labels.notna().to_numpy());rows=meta.iloc[ix].reset_index(drop=True);values=labels.iloc[ix].astype(str).to_numpy();classes=sorted(set(values));y=pd.Categorical(values,categories=classes).codes.astype(int)
    expected=3 if name in ['pos','zuco_reading_task'] else int(name.removeprefix('word_identity')) if name.startswith('word_identity') else 2
    if len(classes)!=expected or min(np.bincount(y))<20:raise ValueError('Insufficient independent label/class coverage')
    vectors={k:v[ix] for k,v in features.items()}
    modes=['familiar','sentence','reader']
    if name in ['pos','content_function','semantic_nouns','repetition']:modes+=['word']
    return rows,y,vectors,classes,dict(modes=modes,sizes=[1,10,'max40']),dict(trials=len(rows),excluded=len(meta)-len(rows),datasets=sorted(rows.dataset.unique()),tasks=sorted(rows.task.unique()),classes=classes,class_counts={c:int((values==c).sum()) for c in classes},word_vocabulary=classes if name.startswith('word_identity') else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,required=True);p.add_argument('--model-dir',type=Path,default=Path('/content/word_eeg_nlp_models'));args=p.parse_args()
    out=args.thesis_root/'Results/word_sentiment_eeg/linguistic_targets';store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve50MB for results; no NLP models or data read yet')
    import fcntl,tempfile
    lock=Path(tempfile.gettempdir(),'linguistic-'+a.digest(str(out))[:16]+'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=dict(spec=SPEC,code={f:a.file_digest(ROOT/f) for f in ['scripts/linguistic_targets.py','scripts/teco_other_targets.py','scripts/teco_average40.py','scripts/analyze_trial_archive.py']},versions={n:version(n) for n in ['numpy','pandas','scikit-learn','stanza','nltk']})
    store.json('protocol.json',proof)
    if (out/'complete.json').is_file():
        for f,sha in t.read(out/'complete.json')['checksums'].items():
            if a.file_digest(out/f)!=sha:raise ValueError('Corrupt completed result')
        print('KEEP COMPLETED LINGUISTIC STUDY',flush=True);return
    meta,features,raw,source_proof=load_pool(args.thesis_root,'teco');store.json('teco_source.json',source_proof)
    source,bad=source_tokens(raw);store.json('labels/excluded_source_sentences.json',bad)
    labels=linguistic_labels(source,store,args.model_dir);joined=attach(meta,labels)
    rows=[];development=[];skipped=[]
    for i,name in enumerate(TASKS,1):
        print(f'BEGIN TARGET {i}/{len(TASKS)}: {name}',flush=True)
        if name=='semantic_nouns':
            zuco_meta,zuco_features,zuco_raw,zuco_proof=load_pool(args.thesis_root,'zuco');store.json('zuco_source.json',zuco_proof)
            mask=zuco_meta.task.eq('SR').to_numpy();meta=zuco_meta.loc[mask].reset_index(drop=True);features={k:v[mask] for k,v in zuco_features.items()};raw=zuco_raw[zuco_raw.task.eq('SR')]
            source,bad=source_tokens(raw);store.json('labels/excluded_english_source_sentences.json',bad)
            english=linguistic_labels(source,store,args.model_dir,language='en');joined=attach(meta,english)
        elif name=='zuco_reading_task':
            meta,features,raw=zuco_meta,zuco_features,zuco_raw;joined=meta
        try:table,y,vectors,classes,plan,definition=target_data(name,joined,features,raw,source)
        except ValueError as error:skipped.append(dict(target=name,reason=str(error)));continue
        store.json(f'targets/{name}/definition.json',definition)
        ids=table[['event_id','dataset','task','subject','sentence_key','word_identity','sentence_position','onset_sample_sentence_0based']].copy();ids['target_code']=y
        old.csv(store,f'targets/{name}/events.csv.gz',ids)
        result,dev,missing=t.evaluate(table,y,vectors,classes,plan,store,name,splitter=safe_splits,scorer=limited_score)
        rows.extend(result);development.extend(dev);skipped.extend(missing)
    if not rows:raise ValueError('No targets had sufficient coverage; labels and checkpoints retained')
    summary=t.summarize(rows);old.csv(store,'summary.csv',summary);old.csv(store,'development.csv.gz',pd.DataFrame(development));old.csv(store,'unavailable.csv',pd.DataFrame(skipped,columns=['target','mode','fold','size','reason']))
    t.zipped_json(store,'outer_results.json.gz',rows);t.figures(store,summary,rows)
    paths=['protocol.json','teco_source.json','summary.csv','development.csv.gz','unavailable.csv','outer_results.json.gz','overview.png']+[str(f.relative_to(out)) for f in out.rglob('result.json.gz')]+[str(f.relative_to(out)) for f in out.glob('targets/*/definition.json')]
    paths += [str(f.relative_to(out)) for f in out.glob('labels/*/complete.json')] + [str(f.relative_to(out)) for f in out.glob('labels/*/words.csv.gz')]
    if (out/'zuco_source.json').is_file():paths.append('zuco_source.json')
    store.json('complete.json',dict(status='completed',checksums={f:a.file_digest(out/f) for f in paths}))
    print('LINGUISTIC TARGETS COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
