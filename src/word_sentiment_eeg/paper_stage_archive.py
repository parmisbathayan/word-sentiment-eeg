"""Lossless, reader-level CPU/GPU handoff for the paper figure workflow."""
import json
from pathlib import Path
import numpy as np
from .common import atomic_npz,write_json,file_digest

LEGACY_ENDPOINT_CODE={
    'scripts/prepare_zuco_paper_figures.py':'06e1e7ca2b4cf7a9838910632b56075a83b7da3faa289e89b71307692e7c0dab',
    'scripts/zuco_paper_stages.py':'eee0118be9573bbb7697b44125b537726d79fcc48783e5befb352add19f08cf7',
    'src/word_sentiment_eeg/paper_stage_archive.py':'9ba5b5200598a29c1c377693cff419cfd34074ccb5bb322e2ccff54b8a49a342'}


def migrate_endpoint_protocol(old,new,out):
    """Only migrate the exact released endpoint fix; valid prepared data unchanged."""
    out=Path(out)
    if {k:v for k,v in old.items() if k!='code'}!={k:v for k,v in new.items() if k!='code'}:
        raise ValueError('Settings, data or environment changed; endpoint-only migration refused')
    if set(old['code'])!=set(new['code']):raise ValueError('Unknown code inventory in old checkpoint')
    for name,sha in old['code'].items():
        expected=LEGACY_ENDPOINT_CODE.get(name,new['code'][name])
        if sha!=expected:raise ValueError('Unknown prior scientific code; endpoint migration refused: '+name)
    if (out/'cpu_preparation_manifest.json').exists() or (out/'preparation_manifest.json').exists():
        raise ValueError('Endpoint recovery is restricted to an incomplete CPU preparation')
    if (out/'null_work').exists() and any((out/'null_work').iterdir()):raise ValueError('Existing GPU work must not be silently migrated')
    preserved=[]
    for mark in sorted((out/'readers').glob('*/prepared.json')):
        reader=mark.parent;state=verify_prepared(reader)
        if (reader/'manifest.json').exists():raise ValueError('GPU reader checkpoint cannot use this CPU-only migration')
        mapping=json.loads((reader/'source_mapping.json').read_text())
        for block in mapping['blocks']:
            for kind in ['raw','corrected']:
                for events in block['duration_endpoint_audit'][kind].values():
                    if events['events'] and not (0<events['minimum_recomputed_ms']<=events['maximum_recomputed_ms']<=10000):
                        raise ValueError('Prepared reader does not satisfy the unchanged historical endpoint rule')
        preserved.append(dict(subject=state['subject'],prepared_sha256=file_digest(mark)))
    write_json(out/'protocol_before_endpoint_fix.json',old)
    write_json(out/'endpoint_recovery.json',dict(reason='Single-sample blink handling only; previously prepared readers have strictly positive endpoint durations',
        preserved_readers=preserved,old_code=old['code'],new_code=new['code']))
    print('ENDPOINT FIX MIGRATION: verified and retained CPU readers '+str([v['subject'] for v in preserved]),flush=True)


def disk_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def save_trials(folder,signal,metadata,negative,positive,storage_root,budget):
    folder=Path(folder);signal=np.asarray(signal)
    if signal.dtype!=np.float32 or signal.ndim!=4 or signal.shape[0]!=2 or signal.shape[2:]!=(105,201) or not np.isfinite(signal).all():
        raise ValueError('Expected finite raw/clean float32 trial arrays')
    if signal.shape[1]!=len(metadata) or metadata.event_id.duplicated().any():raise ValueError('Trial identity mismatch')
    negative=np.asarray(negative,dtype=np.int64);positive=np.asarray(positive,dtype=np.int64)
    if len(negative)!=len(positive) or min(len(negative),len(positive))<2:raise ValueError('Missing ranked trial pairs')
    labels=metadata.sentiment.map({'negative':0,'positive':1}).to_numpy()
    if not np.isin(labels,[0,1]).all() or not (labels[negative]==0).all() or not (labels[positive]==1).all():raise ValueError('Pair labels disagree')
    estimate=signal.nbytes+2_000_000
    if budget is None or storage_root is None or disk_bytes(storage_root)+estimate>budget:
        raise OSError('Prepared-trial storage cap would be exceeded. Earlier reader checkpoints retained; no trial file written.')
    import pandas as pd
    montage=pd.read_csv(Path(__file__).resolve().parents[2]/'assets/zuco_chanlocs.csv').labels.astype(str).to_numpy(dtype=str)
    atomic_npz(folder/'prepared_trials.npz',signal=signal,labels=labels.astype(np.int8),
               event_ids=metadata.event_id.astype(str).to_numpy(dtype=str),negative_pair_index=negative,positive_pair_index=positive,
               channel_names=montage,times_ms=np.arange(-75,126)*8,sample_rate_hz=np.array(125),
               representations=np.array(['raw','cleaned']),units=np.array('microvolts'))
    metadata.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False,compression='gzip')


def mark_prepared(folder,subject,counts,pairs):
    folder=Path(folder)
    files=[p for p in folder.iterdir() if p.is_file() and p.name not in {'prepared.json','manifest.json'} and '.tmp' not in p.name]
    write_json(folder/'prepared.json',dict(status='prepared',subject=subject,accepted_counts=np.asarray(counts).tolist(),pairs=int(pairs),
        storage='lossless float32 raw and cleaned; representation,trial,channel,time',
        checksums={p.name:file_digest(p) for p in files}))


def verify_prepared(folder):
    folder=Path(folder);state=json.loads((folder/'prepared.json').read_text())
    if state.get('status')!='prepared' or not {'prepared_trials.npz','statistics.npz','accepted_trial_metadata.csv.gz'}.issubset(state['checksums']):
        raise ValueError('Incomplete CPU reader checkpoint')
    for name,sha in state['checksums'].items():
        if not (folder/name).is_file() or file_digest(folder/name)!=sha:raise ValueError('CPU checkpoint changed: '+name)
    return state


def finish_reader(folder,null_folder,tct,cluster,use_gpu=True):
    from .paper_figure_stats import permutation_reader
    folder=Path(folder);state=verify_prepared(folder)
    if (folder/'manifest.json').exists():
        done=json.loads((folder/'manifest.json').read_text())
        if done.get('status')!='completed' or any(file_digest(folder/k)!=v for k,v in done['checksums'].items()):raise ValueError('GPU reader checkpoint changed')
        print('KEEP GPU READER '+state['subject'],flush=True);return
    with np.load(folder/'prepared_trials.npz',allow_pickle=False) as data:
        signal=data['signal'];neg=data['negative_pair_index'];pos=data['positive_pair_index']
        raw=signal[0,pos]-signal[0,neg];clean=signal[1,pos]-signal[1,neg]
    observed=np.stack([raw,clean])
    with np.load(folder/'statistics.npz',allow_pickle=False) as stats:
        if not np.array_equal(observed.sum(axis=1,dtype=np.float64),stats['pair_sum']):raise ValueError('Saved trial pairs disagree with prepared statistics')
        if not np.array_equal((observed.astype(np.float64)**2).sum(axis=1),stats['pair_squares']):raise ValueError('Saved squared trial pairs disagree')
    permutation_reader(raw,clean,null_folder,state['subject'],tct,cluster,use_gpu=use_gpu)
    checksums=dict(state['checksums']);checksums['prepared.json']=file_digest(folder/'prepared.json')
    write_json(folder/'manifest.json',dict(status='completed',subject=state['subject'],accepted_counts=state['accepted_counts'],pairs=state['pairs'],checksums=checksums))
