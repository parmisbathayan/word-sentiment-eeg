"""Paired raw/clean FRPs with shared attrition and bounded-memory accumulation.

Inputs must already have validated block, event and word identities. This does
not solve the source sentence mapping or reproduce Automagic preprocessing.
No baseline is silently introduced; the caller must declare that choice.
"""
import numpy as np
import pandas as pd


def paired_epochs(raw, cleaned, events, *, units, baseline, sample_rate=125,
                  limits=(-.6, 1.), amplitude_uv=90):
    """Yield (audit record, raw epoch, clean epoch), one event at a time.

    Call after word selection AND matching. An excursion in raw EEG rejects
    BOTH representations. Unknown/nonfinite clean data also cannot pass.
    Epoch bounds refer to this block only. Input arrays are channels × samples.
    Baseline is either None (explicitly none) or a [start,end) interval in seconds.
    Peak rejection precedes any declared baseline subtraction.
    """
    raw=np.asarray(raw);cleaned=np.asarray(cleaned)
    if raw.ndim!=2 or raw.shape!=cleaned.shape or not all(raw.shape):
        raise ValueError('Expected paired nonempty channel-by-sample arrays')
    if units not in {'uV','V'}:raise ValueError('Declare EEG units as uV or V; do not infer scale')
    if not np.isfinite(sample_rate) or sample_rate<=0 or not np.isfinite(amplitude_uv) or amplitude_uv<=0:
        raise ValueError('Invalid sample rate or raw amplitude threshold')
    required={'event_id','subject','block_id','sentiment','onset_seconds'}
    if not required.issubset(events.columns) or events[list(required)].isna().any().any():
        raise ValueError('Missing validated event identities/onsets')
    if events.event_id.astype(str).duplicated().any():raise ValueError('Duplicate event identities')
    if events[['subject','block_id']].drop_duplicates().shape[0]!=1:
        raise ValueError('Process exactly one reader/block at a time')
    if not events.sentiment.isin(['positive','negative']).all():
        raise ValueError('Apply paper binary selection before paired epoch processing')
    bounds=np.asarray(limits,float)*sample_rate
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1] or not np.allclose(bounds,np.round(bounds)):
        raise ValueError('Epoch limits must lie exactly on the output sample grid')
    lags=np.arange(int(round(bounds[0])),int(round(bounds[1]))+1)
    times=lags/sample_rate
    base=None
    if baseline is not None:
        if len(baseline)!=2 or not np.isfinite(baseline).all() or baseline[0]>=baseline[1]:
            raise ValueError('Invalid explicit baseline interval')
        if baseline[0]<times[0] or baseline[1]>times[-1]:raise ValueError('Baseline outside epoch')
        base=(times>=baseline[0])&(times<baseline[1])
        if not base.any():raise ValueError('No samples in baseline interval')
    scale=1. if units=='uV' else 1e6
    for row in events.to_dict('records'):
        record=dict(row,outcome='accepted',peak_raw_uv=None,onset_sample_0based=None)
        onset=float(row['onset_seconds'])
        if not np.isfinite(onset) or onset<0:
            record['outcome']='invalid_onset';yield record,None,None;continue
        center=int(np.floor(onset*sample_rate+.5));record['onset_sample_0based']=center
        indices=center+lags
        if indices[0]<0 or indices[-1]>=raw.shape[1]:
            record['outcome']='epoch_outside_block';yield record,None,None;continue
        r=np.asarray(raw[:,indices],dtype=np.float64)*scale
        c=np.asarray(cleaned[:,indices],dtype=np.float64)*scale
        if not np.isfinite(r).all() or not np.isfinite(c).all():
            record['outcome']='nonfinite_paired_epoch';yield record,None,None;continue
        record['peak_raw_uv']=float(np.abs(r).max())
        if record['peak_raw_uv']>amplitude_uv:
            record['outcome']='raw_amplitude_exceeds_threshold';yield record,None,None;continue
        if base is not None:
            r-=r[:,base].mean(axis=1,keepdims=True)
            c-=c[:,base].mean(axis=1,keepdims=True)
        yield record,r,c


def summarize_paired_epochs(raw, cleaned, events, **options):
    """Small class means and full sequential audit; never stores trial tensors."""
    sums={};counts={};rows=[]
    for record,r,c in paired_epochs(raw,cleaned,events,**options):
        rows.append(record)
        if r is None:continue
        label=record['sentiment']
        if label not in sums:sums[label]=[np.zeros_like(r),np.zeros_like(c)];counts[label]=0
        sums[label][0]+=r;sums[label][1]+=c;counts[label]+=1
    means={label:dict(raw=pair[0]/counts[label],cleaned=pair[1]/counts[label],n=counts[label])
           for label,pair in sums.items()}
    audit=pd.DataFrame(rows)
    if len(audit)!=len(events):raise RuntimeError('Epoch audit does not reconcile')
    return means,audit


def decoding_features(epoch, *, sample_rate=125, epoch_start=-.6):
    """All-channel 0…500 ms samples, channel-major; 105 × 63 at 125 Hz.

    At 125 Hz the last included sample is 496 ms. This extracts the grid as-is;
    it does not interpolate an artificial 500-ms sample or reduce electrodes.
    """
    epoch=np.asarray(epoch)
    if epoch.ndim!=2 or not np.isfinite(epoch).all():raise ValueError('Invalid accepted epoch')
    if not np.isfinite(sample_rate) or sample_rate<=0 or not np.isfinite(epoch_start):
        raise ValueError('Invalid decoding time grid')
    times=epoch_start+np.arange(epoch.shape[1])/sample_rate
    mask=(times>=-1e-12)&(times<=.5+1e-12)
    if not mask.any() or times[-1]<.5-1/sample_rate:
        raise ValueError('Epoch does not cover the decoding interval')
    if times[0]>1e-12:raise ValueError('Epoch begins after decoding onset')
    return epoch[:,mask].reshape(-1).astype(np.float32)
