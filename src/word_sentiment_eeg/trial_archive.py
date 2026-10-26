"""Label-independent, resumable full-channel trial archive; no model training.

All-channel 125-Hz, 40-Hz-low-pass epochs are saved, not class means or selected features.
Every source word has a metadata outcome. Finite in-bounds epochs survive
regardless of labels, function-word/control status or amplitude flags.
"""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from .common import atomic_npz, digest, file_digest, sentence_key, write_json
from .erp import locate
from .three_level_erp import source_sentences, preprocess40

VERSION='full_word_trials_40hz_125sps_v1'
OUTPUT_HZ=125
TIMES_MS=np.arange(-25,101,dtype=np.float64)*8
CHUNK=32
LAYERS=('lexical_class','contextual_word_class','sentence_context_class')


def word_identity(word):
    # Unicode-safe for Persian and English. Keep punctuation rather than
    # accidentally conflating distinct source tokens; downstream can choose.
    return unicodedata.normalize('NFKC',str(word)).casefold().strip()


def read_historical(folder, dataset):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    events=pd.read_csv(folder/'word_onsets.csv.gz',keep_default_na=False,dtype={'event_id':str,'sentence_key':str})
    position='raw_sentence_position' if dataset=='zuco' else 'sentence_position'
    if events.duplicated([position,'word_index']).any():raise ValueError('Repeated historical event keys')
    expected=manifest.get('source_sha256') if dataset=='zuco' else manifest['source']['sha256']
    if not expected:raise ValueError('Historical source SHA256 is required for cached onset reuse')
    lookup={(int(getattr(r,position)),int(r.word_index)):r for r in events.itertuples()}
    return manifest,events,lookup,expected


def label_lookup(frame,dataset,task):
    if frame is None:return {}
    f=frame[frame.tasks.map(lambda x:task in str(x).split(';'))] if dataset=='zuco' else frame
    keys=['sentence_key','word_index'] if dataset=='zuco' else ['sentence_id_teco','sentence_key','word_index','word']
    if f.duplicated(keys).any():raise ValueError('Duplicate annotation join keys')
    return {tuple(getattr(r,k) for k in keys):r for r in f.itertuples()}


def cached_onset(raw,segment,old):
    """Trust uniqueness only from same-source saved unique matches, then
    validate the full first stored segment at that position again."""
    method=getattr(old,'onset_method',getattr(old,'match_method','')) if old else ''
    onset=getattr(old,'onset_sample_sentence_0based','') if old else ''
    if str(onset) and method in {'exact','constant_channel_offset'}:
        start=int(float(onset));length=segment.shape[1]
        if start<0 or start+length>raw.shape[1]:raise ValueError('Cached onset outside source sentence')
        delta=raw[:,start:start+length]-segment
        if method=='constant_channel_offset':delta-=np.median(delta,axis=1,keepdims=True)
        picks=np.argsort(segment.std(axis=1))[-8:]
        tolerance=1e-4*max(float(np.abs(segment[picks]).max()),1.)
        if np.abs(delta).max()>tolerance:raise ValueError('Cached onset no longer matches first stored waveform')
        return start,'cached_'+method
    if method in {'unmatched','nonunique_match','flat_segment','longer_than_sentence'}:
        return None,'cached_'+method
    return locate(raw,segment)


def _resample(value,up,down):
    from scipy.signal import resample_poly
    return resample_poly(value,up,down,axis=1,padtype='line')


def make_epoch(filtered,onset,hz):
    # The input is already whole-sentence 40-Hz filtered. Use actual sentence
    # context around the epoch for the additional anti-alias resampling FIR.
    # Starting 200 ms before the target epoch makes onset exactly sample25 at
    # 125Hz, regardless of its position on the native sentence sampling grid.
    from math import gcd
    pre,post=int(.2*hz),int(.8*hz)
    if onset<pre or onset+post>=filtered.shape[1]:return None,None,'sentence_boundary'
    margin=int(.2*hz);start=onset-pre-margin;stop=onset+post+margin+1
    value=filtered[:,max(0,start):min(filtered.shape[1],stop)]
    left=max(0,-start);right=max(0,stop-filtered.shape[1])
    if left or right:value=np.pad(value,((0,0),(left,right)),mode='reflect')
    common=gcd(hz,OUTPUT_HZ)
    sampled=_resample(value,OUTPUT_HZ//common,hz//common)
    value=sampled[:,25:25+len(TIMES_MS)].astype(np.float32)
    if value.shape[1]!=len(TIMES_MS):raise ValueError('Resampled epoch time grid differs')
    baseline=value[:,:25].mean(axis=1,dtype=np.float64).astype(np.float32)
    value-=baseline[:,None]
    if not np.isfinite(value).all():return None,None,'nonfinite_epoch'
    return value,baseline,'archived'


def inspect_block(root,block,channels,times):
    """Read one bounded shard at a time; hashes and event-to-array joins.
    Never materialize the entire trial archive in memory."""
    root=Path(root)
    for rel,sha in block['checksums'].items():
        if not (root/rel).is_file() or file_digest(root/rel)!=sha:raise ValueError('Missing/corrupt committed archive file: '+rel)
    events=pd.read_csv(root/block['events'],keep_default_na=False,dtype={'event_id':str})
    if len(events)!=block['word_rows'] or events.event_id.duplicated().any():raise ValueError('Invalid block event metadata')
    expected=events[events.status=='archived']
    if len(expected)!=block['epoch_count']:raise ValueError('Block trial count mismatch')
    found=[]
    for rel in block['epoch_files']:
        with np.load(root/rel,allow_pickle=False) as a:
            ids=a['event_id'].astype(str);n=len(ids)
            if (a['eeg_uV'].shape!=(n,len(channels),len(times)) or a['eeg_uV'].dtype!=np.float32
                or a['baseline_uV'].shape!=(n,len(channels)) or not np.isfinite(a['eeg_uV']).all()
                or not np.isfinite(a['baseline_uV']).all() or not np.array_equal(a['times_ms'],times)
                or a['channel_names'].astype(str).tolist()!=list(channels)):
                raise ValueError('Invalid full-channel epoch shard '+rel)
            if np.abs(a['eeg_uV'][:,:,times<0].mean(axis=2)).max(initial=0)>1e-3:raise ValueError('Incorrect epoch baseline')
            rows=expected[expected.epoch_file==rel].sort_values('epoch_row',key=lambda v:v.astype(int))
            if rows.epoch_row.astype(int).tolist()!=list(range(n)) or rows.event_id.tolist()!=ids.tolist():raise ValueError('Epoch/event identity mismatch')
            found.extend(ids)
    if len(found)!=len(expected) or len(set(found))!=len(found):raise ValueError('Duplicate/missing block EEG events')
    return events


class ArchiveWriter:
    def __init__(self,out,identity,channels,times,budget=None):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.channels=list(channels);self.times=times;self.identity=identity;self.budget=budget
        self.rows=[];self.values=[];self.baselines=[];self.stored=[]
        self.state={'identity':identity,'blocks':[],'next_sentence_position':0,'source_sentences_seen':0,'status':'incomplete'}
        path=self.out/'checkpoint.json'
        self.seen=set()
        if path.exists():
            self.state=json.loads(path.read_text())
            if self.state['identity']!=identity:raise ValueError('Archive/source/protocol identity differs; refusing overwrite')
            for number,block in enumerate(self.state['blocks'],1):
                if number==1 or number%10==0:print(f'VERIFY {self.out.name}: committed block {number}/{len(self.state["blocks"])}',flush=True)
                events=inspect_block(self.out,block,channels,times)
                if self.seen.intersection(events.event_id):raise ValueError('Duplicate event across committed blocks')
                self.seen.update(events.event_id)
            print(f"KEEP {len(self.state['blocks'])} verified blocks; resume at source sentence {self.state['next_sentence_position']}",flush=True)

    def add(self,row,value=None,baseline=None):
        if row['event_id'] in self.seen:raise ValueError('Duplicate EEG/source event')
        self.seen.add(row['event_id']);self.rows.append(row)
        if value is not None:
            self.stored.append(row);self.values.append(value);self.baselines.append(baseline)

    def flush(self,next_position,sentence_count,force=False):
        if not self.rows:return
        if not force and len(self.values)<CHUNK and len(self.rows)<500:return
        # Deterministic uncommitted paths are safe to rewrite after interruption.
        # Already committed blocks are validated and never rewritten.
        rel=Path('blocks')/f"block-{len(self.state['blocks']):06d}"
        folder=self.out/rel
        previous=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) if folder.exists() else 0
        if self.budget:
            estimate=sum(v.nbytes for v in self.values)+sum(v.nbytes for v in self.baselines)+sum(len(str(r).encode()) for r in self.rows)+1024**2
            self.budget.reserve(estimate)
        folder.mkdir(parents=True,exist_ok=True)
        files=[]
        for start in range(0,len(self.values),CHUNK):
            rows=self.stored[start:start+CHUNK]
            epoch_rel=str(rel/f'epochs-{start//CHUNK:03d}.npz')
            for j,row in enumerate(rows):row.update(epoch_file=epoch_rel,epoch_row=j)
            atomic_npz(self.out/epoch_rel,eeg_uV=np.stack(self.values[start:start+CHUNK]).astype(np.float32),
                       baseline_uV=np.stack(self.baselines[start:start+CHUNK]),event_id=np.array([r['event_id'] for r in rows]),
                       times_ms=self.times,channel_names=np.array(self.channels))
            files.append(epoch_rel)
        events_rel=str(rel/'events.csv.gz');temp=folder/'events.tmp.csv.gz'
        pd.DataFrame(self.rows).to_csv(temp,index=False,compression='gzip');temp.replace(self.out/events_rel)
        block={'events':events_rel,'epoch_files':files,'word_rows':len(self.rows),'epoch_count':len(self.values),
               'next_sentence_position':next_position,'checksums':{n:file_digest(self.out/n) for n in [events_rel,*files]}}
        inspect_block(self.out,block,self.channels,self.times)
        if self.budget:self.budget.used+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())-previous
        self.state['blocks'].append(block);self.state.update(next_sentence_position=next_position,source_sentences_seen=sentence_count)
        write_json(self.out/'checkpoint.json',self.state)
        print(f"COMMIT {self.identity['dataset']}/{self.identity['task']}/{self.identity['subject']}: "
              f"source sentence {next_position}; {sum(b['epoch_count'] for b in self.state['blocks'])} full trials; "
              f"{len(self.state['blocks'])} blocks",flush=True)
        self.rows.clear();self.values.clear();self.baselines.clear();self.stored.clear()

    def finish(self,next_position,sentences):
        self.flush(next_position,sentences,True)
        frames=[pd.read_csv(self.out/b['events'],keep_default_na=False) for b in self.state['blocks']]
        if not frames:raise ValueError('No source word metadata found')
        events=pd.concat(frames,ignore_index=True)
        if events.event_id.duplicated().any():raise ValueError('Repeated event in final inventory')
        self.state.update(status='completed',source_sentences_seen=sentences,next_sentence_position=next_position,
                          source_word_rows=len(events),epochs=int(events.status.eq('archived').sum()),
                          outcomes={str(k):int(v) for k,v in events.status.value_counts().items()})
        temp=self.out/'events.tmp.csv.gz';events.to_csv(temp,index=False,compression='gzip');temp.replace(self.out/'events.csv.gz')
        self.state['events_sha256']=file_digest(self.out/'events.csv.gz')
        write_json(self.out/'checkpoint.json',self.state)
        return self.state


def prepare_reader(path,dataset,task,subject,old_folder,annotations,out,protocol,channels,progress=None,budget=None):
    _,old_events,old_lookup,expected=read_historical(old_folder,dataset)
    print(f'HASH SOURCE {dataset}/{task}/{subject}',flush=True)
    sha=file_digest(path) if Path(path).is_file() else expected
    if sha!=expected:raise ValueError('Source hash differs from historical onset source')
    hz=500 if dataset=='zuco' else 1200;times=TIMES_MS.copy()
    identity={'dataset':dataset,'task':task,'subject':subject,'source_sha256':sha,
              'historical_onsets_sha256':file_digest(Path(old_folder)/'word_onsets.csv.gz'),'protocol_sha256':digest(protocol)}
    writer=ArchiveWriter(out,identity,channels,times,budget)
    if writer.state['status']=='completed':
        if file_digest(Path(out)/'events.csv.gz')!=writer.state['events_sha256']:raise ValueError('Final metadata corrupt')
        return writer.state
    if not Path(path).is_file():raise FileNotFoundError('Incomplete reader still needs source EEG: '+str(path))
    labels=label_lookup(annotations,dataset,task);next_position=writer.state['next_sentence_position'];sentences=writer.state['source_sentences_seen']
    historical_seen=set();available_keys=set(old_lookup)
    for position,key,sid,raw,words in source_sentences(path,dataset):
        for index,word,_,_ in words:
            if (position,index) in old_lookup:historical_seen.add((position,index))
        if position<next_position:continue
        sentences+=1;filtered=None
        valid=raw is not None and raw.ndim==2 and raw.shape[0]==len(channels) and raw.shape[1]>=3
        raw_status='available' if valid and np.isfinite(raw).all() else ('nonfinite_sentence_eeg' if valid else 'missing_sentence_eeg')
        sentence_rows=[]
        for ordinal,(index,word,segment,first_status) in enumerate(words):
            word=str(word).strip()
            old=old_lookup.get((position,index))
            if old is not None and (str(old.word)!=word or str(old.sentence_key)!=key):raise ValueError('Historical/source token alignment changed')
            lookup_key=(key,index) if dataset=='zuco' else (sid,key,index,word)
            label=labels.get(lookup_key)
            if label is not None and str(label.word)!=word:label=None
            def field(name,default='unknown'):
                return getattr(label,name,getattr(old,'contextual_class' if name=='contextual_word_class' else name,default))
            row=dict(event_id=digest([dataset,task,subject,position,key,index])[:32],historical_event_id=getattr(old,'event_id',''),
                     dataset=dataset,task=task,subject=subject,sentence_position=position,sentence_key=key,sentence_id_teco=sid if sid is not None else '',
                     word_index=index,word=word,word_identity=word_identity(word),word_length=len(word),sentence_length=len(words),
                     word_position_ordinal=ordinal,relative_word_position=ordinal/max(1,len(words)-1),sentence=str(field('sentence',' '.join(w[1] for w in words))),
                     lexical_class=field('lexical_class'),contextual_word_class=field('contextual_word_class'),sentence_context_class=field('sentence_context_class'),
                     grammatical_class=field('grammatical_class'),is_control=field('is_control'),original_sentence_label=field('original_sentence_label'),
                     annotation_status=field('annotation_status'),annotation_alignment_warning=field('alignment_warning',''),
                     historical_status=getattr(old,'status','not_in_historical_inventory'),historical_accepted_30hz=bool(old and old.status=='accepted'),
                     onset_sample_sentence_0based='',onset_method='',first_fixation_duration_ms='',next_word_first_fixation_ms='',
                     under_100ms='',blink_predecessor='unknown',epoch_file='',epoch_row='',max_abs_uV='',max_abs_stored_uV='',amplitude_over_100_uV='',resample_reflect_padding=False,status=raw_status)
            value=baseline=None
            if raw_status=='available':
                if segment is None or segment.ndim!=2 or segment.shape[0]!=len(channels) or segment.shape[1]<3 or not np.isfinite(segment).all():
                    row['status']=first_status if segment is None and first_status!='available' else 'missing_or_invalid_first_fixation'
                else:
                    duration=(segment.shape[1]-1)*1000/hz;row.update(first_fixation_duration_ms=duration,under_100ms=duration<100)
                    onset,method=cached_onset(raw,segment,old);row['onset_method']=method
                    if onset is None:row['status']=method
                    else:
                        row['onset_sample_sentence_0based']=int(onset)
                        if filtered is None:
                            try:filtered=preprocess40(raw,hz)
                            except ValueError:filtered=False
                        if filtered is False:row['status']='sentence_too_short_for_filter'
                        else:
                            pre,post=int(.2*hz),int(.8*hz)
                            native_peak=None
                            if onset>=pre and onset+post<filtered.shape[1]:
                                native=filtered[:,onset-pre:onset+post+1]
                                native_peak=float(np.abs(native-native[:,:pre].mean(axis=1,keepdims=True)).max())
                                row['resample_reflect_padding']=onset-2*pre<0 or onset+post+pre>=filtered.shape[1]
                            value,baseline,row['status']=make_epoch(filtered,onset,hz)
                            if value is not None:
                                row.update(max_abs_uV=native_peak,max_abs_stored_uV=float(np.abs(value).max()),amplitude_over_100_uV=native_peak>100)
            sentence_rows.append((row,value,baseline))
        first_onsets=sorted({r['onset_sample_sentence_0based'] for r,_,_ in sentence_rows if r['onset_sample_sentence_0based']!=''})
        for row,value,baseline in sentence_rows:
            onset=row['onset_sample_sentence_0based']
            if onset!='':
                later=next((x for x in first_onsets if x>onset),None)
                if later is not None:row['next_word_first_fixation_ms']=(later-onset)*1000/hz
            writer.add(row,value,baseline)
        next_position=position+1
        writer.flush(next_position,sentences)
        if progress:progress(dataset,task,subject,next_position,len(writer.seen))
        if sentences%10==0:print(f'PROGRESS {dataset}/{task}/{subject}: {sentences} source sentences inspected',flush=True)
    if available_keys!=historical_seen:raise ValueError('Source iteration did not cover every historical word event')
    if len(writer.seen)!=len(old_events):raise ValueError('Source word inventory differs from historical run')
    state=writer.finish(next_position,sentences)
    return state
