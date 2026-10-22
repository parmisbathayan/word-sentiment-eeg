"""Validate existing analysis separately from trial exports; repair from onsets.

No valid class average or epoch file is overwritten during repair. Trial
exports use bounded chunks so missing chunks can be rebuilt independently.
"""
import json
import logging
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .common import atomic_npz, digest, sentence_key, write_json
from .erp import CLASSES, TIMES_MS, SFREQ, epoch, iter_sentences, preprocess

CHUNK_TRIALS = 64  # ~13.5 MB uncompressed per chunk, rather than a 0.5–1 GB file


class MissingAnalysisArtifact(FileNotFoundError):
    """A compatible reader manifest exists but its analysis is incomplete."""

    def __init__(self, target, item, missing):
        self.item = item
        super().__init__(f"Missing analysis artifacts in {target}: {', '.join(missing)}")


def epoch_file_valid(path, event_ids, channels):
    """Check ZIP structure, EEG header and small alignment arrays, not full EEG."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open('eeg_uV.npy') as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = (np.lib.format.read_array_header_1_0(stream) if version==(1,0)
                                    else np.lib.format.read_array_header_2_0(stream))
                if shape != (len(event_ids),105,len(TIMES_MS)) or dtype != np.dtype('float32'):
                    return False
        with np.load(path,allow_pickle=False) as data:
            return (np.array_equal(data['event_id'].astype(str),np.array(event_ids,dtype=str))
                    and np.array_equal(data['times_ms'],TIMES_MS)
                    and np.array_equal(data['channel_names'].astype(str),np.array(channels))
                    and int(data['sfreq_hz'])==SFREQ)
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
        return False


def class_parts(target, cls, ids, channels):
    legacy = target/f'{cls}_epochs.npz'
    if legacy.is_file() and epoch_file_valid(legacy,ids,channels):
        return [{'path':legacy.name,'start':0,'stop':len(ids)}], []
    parts, missing = [], []
    for start in range(0,len(ids),CHUNK_TRIALS):
        stop = min(start+CHUNK_TRIALS,len(ids))
        relative = f'{cls}_epochs/part-{start:06d}-{stop:06d}.npz'
        if epoch_file_valid(target/relative,ids[start:stop],channels):
            parts.append({'path':relative,'start':start,'stop':stop})
        else:
            missing.append({'path':relative,'start':start,'stop':stop})
    return parts, missing


def inspect_checkpoint(target, record, fingerprints, channels):
    """Return a reusable analysis even when some individual trial files are absent."""
    done=target/'manifest.json'
    if not done.exists():
        return None
    item=json.loads(done.read_text())
    if item.get('fingerprint') not in fingerprints:
        raise ValueError('Checkpoint belongs to different extraction settings/labels; refusing silent replacement')
    if (item.get('task'),item.get('subject')) != (record['task'],record['subject']):
        raise ValueError('Checkpoint reader/task identity mismatch')
    missing=[name for name in ('class_means.npz','word_onsets.csv.gz')
             if not (target/name).is_file()]
    if missing:
        raise MissingAnalysisArtifact(target,item,missing)
    with np.load(target/'class_means.npz',allow_pickle=False) as data:
        counts=[item['epochs_per_class'][c] for c in CLASSES]
        if (data['mean_uV'].shape != (3,105,len(TIMES_MS))
            or not np.array_equal(data['counts'],counts)
            or not np.array_equal(data['classes'].astype(str),np.array(CLASSES))
            or not np.array_equal(data['channel_names'].astype(str),np.array(channels))
            or not np.array_equal(data['times_ms'],TIMES_MS)):
            raise ValueError('Saved class average schema/counts do not agree with its checkpoint')
        for ci,count in enumerate(counts):
            if count and not np.isfinite(data['mean_uV'][ci]).all():
                raise ValueError('Nonfinite saved class average')
    events=pd.read_csv(target/'word_onsets.csv.gz',keep_default_na=False,
                       dtype={'event_id':str,'sentence_key':str})
    if len(events)!=item['word_rows']:
        raise ValueError('Saved onset row count does not agree with checkpoint')
    accepted=events[events.status=='accepted']
    if accepted.event_id.duplicated().any():
        raise ValueError('Duplicate accepted event IDs')
    ids={}
    for cls in CLASSES:
        ids[cls]=accepted.loc[accepted.contextual_class==cls,'event_id'].astype(str).tolist()
        if len(ids[cls])!=item['epochs_per_class'][cls]:
            raise ValueError(f'Saved onset counts disagree for {cls}')
    parts,missing={},{}
    for cls in CLASSES:
        logging.info('Validate trial export %s/%s/%s',record['task'],record['subject'],cls)
        parts[cls],missing[cls]=class_parts(target,cls,ids[cls],channels)
    item.update(output_dir=str(target),analysis_ready=True,
                trial_exports_complete=not any(missing.values()),
                missing_trial_classes=[c for c in CLASSES if missing[c]],
                missing_trial_chunks=sum(map(len,missing.values())))
    return {'item':item,'events':events,'ids':ids,'parts':parts,'missing':missing}


class EpochChunks:
    def __init__(self,target,channels):
        self.target=Path(target)
        self.channels=channels
        self.blocks={c:[] for c in CLASSES}
        self.ids={c:[] for c in CLASSES}
        self.counts={c:0 for c in CLASSES}
        self.sums={c:np.zeros((105,len(TIMES_MS)),np.float64) for c in CLASSES}
        self.parts={c:[] for c in CLASSES}

    def add(self,cls,value,event_id):
        index=self.counts[cls]
        self.counts[cls]+=1
        self.blocks[cls].append(value)
        self.ids[cls].append(event_id)
        self.sums[cls]+=value
        if len(self.blocks[cls])>=CHUNK_TRIALS:
            self.flush(cls)
        return index

    def flush(self,cls):
        if not self.blocks[cls]:
            return
        stop=self.counts[cls];start=stop-len(self.blocks[cls])
        relative=f'{cls}_epochs/part-{start:06d}-{stop:06d}.npz'
        save_chunk(self.target/relative,np.stack(self.blocks[cls]),self.ids[cls],self.channels)
        self.parts[cls].append({'path':relative,'start':start,'stop':stop})
        self.blocks[cls].clear();self.ids[cls].clear()

    def finish(self):
        for cls in CLASSES:
            self.flush(cls)
        means=np.stack([(self.sums[c]/self.counts[c]).astype(np.float32) if self.counts[c]
                        else np.full((105,len(TIMES_MS)),np.nan,np.float32) for c in CLASSES])
        write_json(self.target/'epochs_index.json',{'schema_version':2,'classes':self.parts,
                    'counts':self.counts,'chunk_trials':CHUNK_TRIALS})
        return means,[self.counts[c] for c in CLASSES]


def save_chunk(path,values,ids,channels):
    logging.info('Save epoch chunk %s (%d trials)',path.name,len(ids))
    atomic_npz(path,eeg_uV=values,event_id=np.array(ids,dtype='U24'),times_ms=TIMES_MS,
               channel_names=np.array(channels),sfreq_hz=np.array(SFREQ))
    if not epoch_file_valid(path,ids,channels):
        raise IOError(f'Could not verify saved epoch chunk: {path}')


def repair_trial_exports(path,checkpoint,channels,source_sha256):
    """Rebuild missing chunks from cached onsets; never call the onset matcher."""
    item=checkpoint['item'];target=Path(item['output_dir'])
    if source_sha256 != item['source_sha256']:
        raise ValueError('Source changed since saved onsets; refusing incompatible repair')
    events=checkpoint['events']
    accepted=events[events.status=='accepted'].copy()
    locations={}
    buffers={}
    needed=set()
    for cls in CLASSES:
        rows=accepted[accepted.contextual_class==cls].reset_index(drop=True)
        for part in checkpoint['missing'][cls]:
            key=(cls,part['start'],part['stop'])
            buffers[key]={}
            for index in range(part['start'],part['stop']):
                row=rows.iloc[index]
                locations[(int(row.raw_sentence_position),int(row.word_index))]=(key,index,row)
                needed.add(int(row.raw_sentence_position))
    logging.info('Repair %s/%s: reuse saved onsets/averages; rebuild only %s (%d small chunks)',
                 item['task'],item['subject'],item['missing_trial_classes'],len(buffers))
    if not needed:
        return item
    for position,words,raw,_ in iter_sentences(path):
        if position not in needed:
            continue
        filtered=preprocess(raw)
        skey=sentence_key(words)
        for j,word in enumerate(words):
            match=locations.get((position,j))
            if match is None:
                continue
            key,index,row=match
            if skey!=row.sentence_key or word!=row.word:
                raise ValueError('Repair source text does not match saved onset table')
            value,status=epoch(filtered,int(float(row.onset_sample_sentence_0based)))
            if status!='accepted':
                raise ValueError(f'Previously accepted event {row.event_id} changed during repair: {status}')
            buffers[key][index]=value
            cls,start,stop=key
            if len(buffers[key])==stop-start:
                ids=checkpoint['ids'][cls][start:stop]
                relative=f'{cls}_epochs/part-{start:06d}-{stop:06d}.npz'
                save_chunk(target/relative,np.stack([buffers[key][k] for k in range(start,stop)]),ids,channels)
                buffers[key].clear()
                checkpoint['parts'][cls].append({'path':relative,'start':start,'stop':stop})
        needed.remove(position)
        if not needed:
            break
    if needed or any(buffers.values()):
        raise ValueError('Source ended before all missing trial chunks could be rebuilt')
    # Reinspect every class before marking complete. Valid legacy files survive.
    parts={}
    for cls in CLASSES:
        parts[cls],missing=class_parts(target,cls,checkpoint['ids'][cls],channels)
        if missing:
            raise IOError(f'Repair still missing {cls} chunks')
    write_json(target/'epochs_index.json',{'schema_version':2,'classes':parts,
        'counts':item['epochs_per_class'],'chunk_trials':CHUNK_TRIALS})
    item.update(trial_exports_complete=True,missing_trial_classes=[],missing_trial_chunks=0)
    write_json(target/'manifest.json',item)
    return item


def load_class_epochs(target,cls):
    """Convenience loader for small analyses/tests; stream chunks for large data."""
    target=Path(target)
    if (target/'epochs_index.json').exists():
        index=json.loads((target/'epochs_index.json').read_text())
        names=[p['path'] for p in sorted(index['classes'][cls],key=lambda p:p['start'])]
    else:
        names=[f'{cls}_epochs.npz']
    values=[]
    for name in names:
        with np.load(target/name,allow_pickle=False) as data:
            values.append(data['eeg_uV'])
    return np.concatenate(values) if values else np.empty((0,105,len(TIMES_MS)),np.float32)
