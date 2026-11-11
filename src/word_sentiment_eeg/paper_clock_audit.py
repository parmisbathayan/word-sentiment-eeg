"""Candidate clocks independently checked against stored EEG fixation waveforms.

This module does not alter extraction acceptance or claim recovered trials.
Word/sentence identity, order and duration establish event correspondence before
any candidate clock's errors are inspected. Sentiment labels are never used.
"""
import numpy as np


def anchors(codes,samples,eye_events,rate):
    codes=np.asarray([int(str(v).strip()) for v in codes]);use=np.isin(codes,[10,11,12,13]);c=codes[use]
    t=(np.asarray(samples,float)[use]-1)/rate;e=np.asarray(eye_events,float)
    candidates=[]
    if e.ndim!=2 or e.shape[1]!=2:raise ValueError('Invalid trigger table')
    for col in [0,1]:
        mask=np.isin(e[:,col],[10,11,12,13])
        if np.array_equal(e[mask,col],c):candidates.append(e[mask,1-col])
    if len(candidates)!=1 or len(c)<10:raise ValueError('No unique complete shared trigger sequence')
    x=candidates[0]
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(x)<=0) or np.any(np.diff(t)<=0):raise ValueError('Invalid trigger clocks')
    return c,x,t


def fit_candidates(x,y):
    from scipy.stats import theilslopes
    x=np.asarray(x,float);y=np.asarray(y,float);origin=float(np.median(x));xc=x-origin
    slope,intercept=np.polyfit(xc,y,1)
    robust=theilslopes(y,xc,method='joint')
    result={}
    for name,s,b in [('original_affine',slope,intercept),('theil_sen_candidate',robust.slope,robust.intercept)]:
        if not .00098<s<.00102:raise ValueError('Implausible millisecond clock slope')
        residual=(y-(xc*s+b))*1000
        result[name]=dict(eye_origin_ms=origin,slope_seconds_per_eye_ms=float(s),eeg_seconds_at_origin=float(b),
            anchor_max_error_ms=float(abs(residual).max()),anchor_median_error_ms=float(np.median(abs(residual))),
            anchors_over_8ms=int(sum(abs(residual)>8)))
    return result


def errors(clock,eye_ms,eeg_seconds):
    predicted=(np.asarray(eye_ms)-clock['eye_origin_ms'])*clock['slope_seconds_per_eye_ms']+clock['eeg_seconds_at_origin']
    return (predicted-np.asarray(eeg_seconds))*1000


def reading_intervals(codes,eye_ms,eeg_seconds):
    """Explicit 10→11 reading intervals; never assume nominal 50-trial order."""
    rows=[]
    for i,code in enumerate(codes[:-1]):
        if code==10 and codes[i+1]==11:
            rows.append(dict(anchor_index=i,eye_start=float(eye_ms[i]),eye_stop=float(eye_ms[i+1]),
                eeg_start=float(eeg_seconds[i]),eeg_stop=float(eeg_seconds[i+1])))
    return rows


def summarize_validation(clocks,rows):
    output={};eye=[r['eye_ms'] for r in rows];eeg=[r['waveform_eeg_seconds'] for r in rows]
    sentences=len({r['sentence_position'] for r in rows})
    for name,clock in clocks.items():
        e=errors(clock,eye,eeg);ae=abs(e)
        output[name]=dict(checked_fixations=len(rows),checked_sentences=sentences,
            median_error_ms=float(np.median(ae)) if len(e) else None,p99_error_ms=float(np.quantile(ae,.99)) if len(e) else None,
            maximum_error_ms=float(ae.max()) if len(e) else None,
            signed_median_error_ms=float(np.median(e)) if len(e) else None,
            within_2ms=int(sum(ae<=2)),within_4ms=int(sum(ae<=4)),within_8ms=int(sum(ae<=8)),
            # Audit-only criterion, predeclared; not adoption of a revised method.
            candidate_passes_audit=bool(len(e)>=100 and sentences>=10 and np.quantile(ae,.99)<=2 and ae.max()<=4))
    return output
