"""Independent Python building blocks for a Pfeiffer-2020 methods pilot.

Not an Unfold port or numerical-equivalence claim. No labels enter the nuisance
model. The native timelines must be synchronized before using these functions.
"""
import numpy as np
import pandas as pd


def synchronize(eeg_codes, eeg_samples_1based, eye_events, sample_rate, max_error_ms=None):
    """Fit the eye-clock to EEG seconds using an exact common trigger sequence.

    Only sentence on/off triggers synchronize clocks, as described in the paper.
    Trigger 15 is an answer keypress, not a synchronization anchor.
    Ambiguous/missing/reordered trigger sequences stop rather than being guessed.
    """
    common={10,11,12,13}
    eeg_codes=np.asarray([int(str(c).strip()) for c in eeg_codes])
    use=np.isin(eeg_codes,list(common))
    codes=eeg_codes[use]
    times=(np.asarray(eeg_samples_1based,dtype=float)[use]-1)/sample_rate
    eye=np.asarray(eye_events,dtype=float)
    if eye.ndim!=2 or eye.shape[1]!=2:raise ValueError('Expected two-column eye trigger table')
    candidates=[]
    for code_column in [0,1]:
        mask=np.isin(eye[:,code_column],list(common))
        if np.array_equal(eye[mask,code_column],codes):
            candidates.append(eye[mask,1-code_column])
    if len(candidates)!=1 or len(codes)<10:
        raise ValueError('Missing, reordered or ambiguous shared trigger sequence; synchronization not inferred')
    eye_times=candidates[0]
    if np.any(np.diff(times)<=0) or np.any(np.diff(eye_times)<=0):
        raise ValueError('Trigger clocks are not strictly increasing')
    # Center before fitting to avoid conditioning on a large absolute eye clock.
    origin=float(eye_times.mean())
    slope,center=np.polyfit(eye_times-origin,times,1)
    if not .00098<slope<.00102:raise ValueError('Eye timestamps are not a plausible millisecond clock')
    residual=times-((eye_times-origin)*slope+center)
    max_error=float(np.max(np.abs(residual)))
    tolerance=1/sample_rate if max_error_ms is None else float(max_error_ms)/1000
    if not 0<tolerance<=.008:raise ValueError('Timing tolerance must be positive and at most one 125-Hz sample')
    if max_error>tolerance+1e-10:
        error=ValueError(f'Synchronization residual {max_error*1000:.3f} ms exceeds configured {tolerance*1000:.3f} ms tolerance')
        error.diagnostics=dict(eye_origin_ms=origin,slope_seconds_per_eye_ms=float(slope),
                               eeg_seconds_at_origin=float(center),max_residual_ms=max_error*1000,
                               anchors=[dict(code=int(c),eye_ms=float(e),eeg_seconds=float(t),residual_ms=float(r)*1000)
                                        for c,e,t,r in zip(codes,eye_times,times,residual)])
        raise error
    return dict(eye_origin_ms=origin,slope_seconds_per_eye_ms=float(slope),
                eeg_seconds_at_origin=float(center),triggers=len(codes),
                configured_tolerance_ms=tolerance*1000,native_one_sample_target_met=bool(max_error<=1/sample_rate+1e-10),
                max_residual_ms=max_error*1000,rms_residual_ms=float(np.sqrt(np.mean(residual**2)))*1000)


def eye_to_seconds(values,sync):
    return ((np.asarray(values,dtype=float)-sync['eye_origin_ms'])*sync['slope_seconds_per_eye_ms']
            +sync['eeg_seconds_at_origin'])


def event_basis(events):
    """Fit nuisance covariate scaling/spline knots once for an entire reader."""
    from sklearn.preprocessing import SplineTransformer
    required={'fixation','saccade','blink','keypress'}
    if not required.issubset(set(events.event_type)):
        raise ValueError('Requires explicitly observed fixation/saccade/blink/keypress events')
    duration=events.loc[events.event_type=='fixation','duration_ms'].to_numpy(dtype=float)
    amplitude=events.loc[events.event_type=='saccade','amplitude_deg'].to_numpy(dtype=float)
    if not np.isfinite(duration).all() or np.any(duration<=0) or duration.std()<1e-12:
        raise ValueError('Invalid or constant fixation durations')
    if not np.isfinite(amplitude).all() or np.any(amplitude<0) or len(np.unique(amplitude))<8:
        raise ValueError('Invalid/insufficient saccade amplitude variation')
    spline=SplineTransformer(n_knots=8,degree=3,knots='quantile',include_bias=True).fit(amplitude[:,None])
    return dict(duration_mean_ms=float(duration.mean()),duration_sd_ms=float(duration.std()),spline=spline)


def time_expanded_design(events,n_samples,sample_rate=125,limits=(-.6,1.0),basis_reference=None):
    """Sparse FIR nuisance GLM with continuous, fractional-sample event timing.

    Cubic quantile B-splines are a documented Python choice, not assumed equal
    to Unfold's spline implementation. Linear sample weights preserve mapped
    event latencies rather than introducing rounding errors into the timeline.
    """
    from scipy.sparse import coo_matrix
    if not set(events.event_type).issubset({'fixation','saccade','blink','keypress'}):raise ValueError('Unknown nuisance event type')
    onset_values=events.onset_seconds.to_numpy(dtype=float)
    if not np.isfinite(onset_values).all() or np.any(onset_values<0) or np.any(onset_values>=n_samples/sample_rate):
        raise ValueError('Events outside their actual recording block')
    reference=event_basis(events) if basis_reference is None else basis_reference
    predictors=[];names=[];basis_info={}
    for kind in ['fixation','saccade','blink','keypress']:
        table=events[events.event_type==kind]
        onsets=table.onset_seconds.to_numpy(dtype=float)*sample_rate
        cols=[np.ones(len(table))];labels=['intercept']
        if kind=='fixation':
            duration=table.duration_ms.to_numpy(dtype=float)
            if not np.isfinite(duration).all() or np.any(duration<=0):raise ValueError('Invalid fixation durations')
            center=reference['duration_mean_ms'];scale=reference['duration_sd_ms']
            cols.append((duration-center)/scale);labels.append('duration_z')
            basis_info[kind]=dict(duration_mean_ms=center,duration_sd_ms=scale)
        if kind=='saccade':
            amplitude=table.amplitude_deg.to_numpy(dtype=float)
            if not np.isfinite(amplitude).all() or np.any(amplitude<0):raise ValueError('Invalid saccade amplitudes')
            # 8 knots + degree 3 - 1 = 10 basis functions. Drop one alongside
            # the separate event intercept to avoid partition-of-unity aliasing.
            spline=reference['spline']
            basis=spline.transform(amplitude[:,None]) if len(amplitude) else np.empty((0,10))
            cols.extend(basis[:,1:].T);labels += [f'bspline_{i}' for i in range(1,basis.shape[1])]
            basis_info[kind]=dict(basis='cubic quantile B-splines, 10 including intercept',
                                  knots=spline.bsplines_[0].t.tolist())
        for col,label in zip(cols,labels):
            predictors.append((onsets,np.asarray(col)));names.append(kind+'/'+label)
    lags=np.arange(round(limits[0]*sample_rate),round(limits[1]*sample_rate)+1)
    rows=[];columns=[];values=[]
    for pi,(onsets,weights) in enumerate(predictors):
        lower=np.floor(onsets).astype(int);fraction=onsets-lower
        for offset,factors in [(0,1-fraction),(1,fraction)]:
            rr=lower[:,None]+offset+lags[None,:]
            vv=np.broadcast_to((weights*factors)[:,None],rr.shape)
            cc=np.broadcast_to(pi*len(lags)+np.arange(len(lags))[None,:],rr.shape)
            valid=(rr>=0)&(rr<n_samples)&(vv!=0)
            rows.extend(rr[valid]);columns.extend(cc[valid]);values.extend(vv[valid])
    design=coo_matrix((values,(rows,columns)),shape=(n_samples,len(names)*len(lags))).tocsr()
    return design,dict(predictors=names,lags_ms=(lags*1000/sample_rate).tolist(),basis=basis_info,
                       fractional_timing='linear sample weights',nonzero=int(design.nnz))


def fit_nuisance(eeg,design,atol=1e-5,maxiter=2000,progress=None):
    """Fit one continuous channel at a time; refuse non-converged solutions."""
    from scipy.sparse.linalg import lsmr
    if eeg.ndim!=2 or eeg.shape[1]!=design.shape[0] or not np.isfinite(eeg).all():
        raise ValueError('Invalid continuous EEG/design dimensions or values')
    betas=np.empty((eeg.shape[0],design.shape[1]),dtype=np.float32)
    residual=np.empty_like(eeg,dtype=np.float32);diagnostics=[]
    norms=np.sqrt(np.asarray(design.power(2).sum(axis=0)).reshape(-1))
    if np.any(norms==0):raise ValueError('An expanded predictor has no support')
    scaled=design.multiply(1/norms).tocsr()
    for channel in range(eeg.shape[0]):
        result=lsmr(scaled,eeg[channel].astype(float),atol=atol,btol=atol,maxiter=maxiter)
        if result[1] not in [0,1,2,4,5]:
            raise ValueError(f'Nuisance solver did not converge for channel {channel}: stop={result[1]}, iterations={result[2]}')
        betas[channel]=result[0]/norms
        residual[channel]=eeg[channel]-scaled@result[0]
        diagnostics.append(dict(channel_index=channel,stop=int(result[1]),iterations=int(result[2]),
                                residual_norm=float(result[3]),normal_equation_residual=float(result[4]),
                                condition_estimate=float(result[6])))
        if progress:progress(channel+1,eeg.shape[0])
    return residual,betas,diagnostics


def fit_reader_blocks(blocks,sample_rate=125,limits=(-.6,1.0),progress=None):
    """Share nuisance coefficients across one reader's separate recording blocks.

    Stack rows of block-specific designs, not signals on an invented continuous
    clock. No lagged response can spill across a source-file boundary. Covariate
    scaling and spline knots are shared across the reader, not refit per block.
    Caller is responsible for preprocessing and independently validated clocks.
    """
    from scipy.sparse import vstack
    if not blocks:raise ValueError('No recording blocks')
    ids=[b['block_id'] for b in blocks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate recording block identity')
    channels=list(blocks[0]['channel_names'])
    for block in blocks:
        if list(block['channel_names'])!=channels or block['eeg'].shape[0]!=len(channels):
            raise ValueError('Reader blocks have different montage/order')
    reference=event_basis(pd.concat([b['events'] for b in blocks],ignore_index=True))
    designs=[];metadata=[];offset=0
    for block in blocks:
        design,info=time_expanded_design(block['events'],block['eeg'].shape[1],sample_rate,limits,reference)
        designs.append(design)
        metadata.append(dict(block_id=block['block_id'],row_start=offset,row_stop=offset+design.shape[0],
                             samples=design.shape[0],design=info))
        offset+=design.shape[0]
    if any(m['design']['predictors']!=metadata[0]['design']['predictors'] for m in metadata):
        raise ValueError('Block nuisance predictor identities disagree')
    combined=vstack(designs,format='csr')
    eeg=np.concatenate([b['eeg'] for b in blocks],axis=1)
    residual,betas,solvers=fit_nuisance(eeg,combined,progress=progress)
    outputs={m['block_id']:residual[:,m['row_start']:m['row_stop']] for m in metadata}
    return outputs,betas,dict(blocks=metadata,solver=solvers,source_boundary_spill=False,
                              shared_basis=True,shared_coefficients=True)


def audit_epochs(eeg,residual,events,sample_rate=125):
    """All available fixation epochs for engineering QA, without sentiment selection.

    Baseline subtraction is deliberately absent: no unstated baseline policy is
    smuggled into a paper replication. Out-of-word and label selection still need
    original word bounds and SST joins. These are not sentiment-class FRPs.
    """
    lags=np.arange(-75,126);raw=[];clean=[];counts={'fixations':0,'boundary':0,'amplitude':0,'accepted_for_qa':0}
    for onset in events.loc[events.event_type=='fixation','onset_seconds']:
        counts['fixations']+=1
        center=int(np.floor(onset*sample_rate+.5));indices=center+lags
        if indices[0]<0 or indices[-1]>=eeg.shape[1]:counts['boundary']+=1;continue
        epoch=eeg[:,indices]
        if np.max(np.abs(epoch))>90:counts['amplitude']+=1;continue
        raw.append(epoch);clean.append(residual[:,indices]);counts['accepted_for_qa']+=1
    if not raw:raise ValueError('No epochs for engineering QA')
    return np.mean(raw,axis=0),np.mean(clean,axis=0),counts
