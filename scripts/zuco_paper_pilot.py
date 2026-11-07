#!/usr/bin/env python3
"""One original ZuCo SR block: synchronized Python nuisance-GLM pilot.

Engineering validation only. No word sentiment labels, class comparison,
classification or claim of replicating the published results.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from word_sentiment_eeg.paper_python import synchronize,eye_to_seconds,time_expanded_design,fit_nuisance,audit_epochs
from word_sentiment_eeg.common import file_digest,write_json,atomic_npz
from paper_source_audit import fetch_sample


def native_continuous(path):
    import h5py
    with h5py.File(path,'r') as h:
        g=h['EEG'];rate=float(g['srate'][()].squeeze());nchannels=int(g['nbchan'][()].squeeze())
        eeg=g['data'][()].T
        if eeg.shape[0]!=nchannels or nchannels!=105 or rate!=500:
            raise ValueError('Unexpected audited source recording dimensions/rate')
        def reference_values(dataset,chars=False):
            values=[]
            for ref in dataset[()].reshape(-1):
                value=h[ref][()].reshape(-1)
                values.append(''.join(chr(int(v)) for v in value) if chars else float(value[0]))
            return values
        codes=reference_values(g['event/type'],chars=True)
        latencies=reference_values(g['event/latency'])
        names=reference_values(g['chanlocs/labels'],chars=True)
    if not np.isfinite(eeg).all():raise ValueError('Nonfinite source EEG')
    return eeg.astype(np.float32),rate,codes,np.asarray(latencies),names


def normalized_events(eye,codes,latencies,rate,sync,duration_seconds):
    def normalize(value):return re.sub('[^a-z0-9]','',str(value).lower())
    records=[];headers={};outside={}
    for plural,kind in [('fixations','fixation'),('saccades','saccade'),('blinks','blink')]:
        source=getattr(eye['eyeevent'],plural)
        columns=[str(v) for v in np.asarray(source.colheader).reshape(-1)]
        headers[kind]=columns;keys=[normalize(v) for v in columns]
        array=np.asarray(source.data,dtype=float)
        if not array.size:
            outside[kind]=0
            continue
        if array.ndim==1:array=array[None,:]
        def column(aliases):
            found=[i for i,key in enumerate(keys) if key in aliases]
            if len(found)!=1:raise ValueError(f'Unrecognized/ambiguous {kind} event headers: {columns}; expected {aliases}')
            return array[:,found[0]]
        start=column({'latency','start','starttime','sttime','sttimestamp','starttimestamp'})
        duration=column({'duration','dur'})
        onsets=eye_to_seconds(start,sync)
        amplitude=column({'amplitude','amp','ampl','sacamplitude','saccamplitude'}) if kind=='saccade' else np.full(len(start),np.nan)
        outside[kind]=int(((onsets<0)|(onsets>=duration_seconds)).sum())
        for i,onset in enumerate(onsets):
            if not np.isfinite(onset) or not np.isfinite(duration[i]) or duration[i]<=0:
                raise ValueError(f'Invalid {kind} event timing')
            if 0<=onset<duration_seconds:
                records.append(dict(event_type=kind,onset_seconds=float(onset),duration_ms=float(duration[i]),
                                    amplitude_deg=float(amplitude[i]),source_event_index=i))
    for i,(code,latency) in enumerate(zip(codes,latencies)):
        if int(code.strip())==15:
            onset=(latency-1)/rate
            if 0<=onset<duration_seconds:
                records.append(dict(event_type='keypress',onset_seconds=onset,duration_ms=0.,amplitude_deg=np.nan,source_event_index=i))
    return pd.DataFrame(records).sort_values('onset_seconds').reset_index(drop=True),headers,outside


def filter_and_resample(eeg,rate):
    from scipy.signal import firwin,fftconvolve,resample_poly
    # Independent Python single-pass, zero-phase centered FIR. The explicit
    # 0.25-Hz transition heuristic and SciPy coefficients are not EEGLAB parity.
    taps=int(np.ceil(3.3*rate/.25));taps += 1-taps%2
    kernel=firwin(taps,[1.,50.],pass_zero=False,fs=rate,window='hamming')
    pad=(taps-1)//2
    output=[]
    for channel in eeg:
        filtered=fftconvolve(np.pad(channel.astype(float),(pad,pad),mode='reflect'),kernel,mode='valid')
        output.append(resample_poly(filtered,1,4).astype(np.float32))
    return np.stack(output),dict(implementation='SciPy Hamming centered FIR, single pass, reflected boundaries',
                                 cutoffs_hz=[1,50],taps=taps,transition_heuristic_hz=.25,
                                 input_hz=rate,output_hz=125,anti_alias='scipy resample_poly default Kaiser FIR')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--channels',type=int,default=105,help='First N channels for engineering QA; full pilot uses all 105')
    p.add_argument('--engineering-timing-tolerance-ms',type=float,default=None,
                   help='Explicit diagnostic tolerance, at most 8 ms; default is strict one native sample (2 ms)')
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud only: never download EEG to the Mac')
    if not 1<=args.channels<=105:raise ValueError('channels must be 1..105')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((ROOT/'docs/paper_source_audit.json').read_text())
    sources=[next(r for r in audit['samples'] if r['name']==name)
             for name in ['gip_ZAB_SR1_EEG.mat','ZAB_SR1_ET.mat']]
    protocol={'pilot':'ZAB SR block 1, engineering QA only','channels':args.channels,
              'engineering_timing_tolerance_ms':args.engineering_timing_tolerance_ms,
              'runner_sha256':file_digest(Path(__file__)),'library_sha256':file_digest(ROOT/'src/word_sentiment_eeg/paper_python.py'),
              'sources':[{k:s[k] for k in ['name','bytes','download','sha256']} for s in sources],
              'scope':'not the full paper reproduction; no sentiment results',
              'limitations':['Released corpus preprocessing, not a rerun of Automagic 1.9/MARA',
                             'Independent Python FIR, synchronization and spline implementation; MATLAB equivalence unverified',
                             'One block rather than per-reader full-recording deconvolution',
                             'No word-bound mapping, original SST labels, matching, FRP inference or decoding yet',
                             'No implicit epoch baseline correction; all-fixation engineering QA only']}
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('Pilot configuration changed; refusing mixed results')
    write_json(out/'protocol.json',protocol)
    mark=out/'manifest.json'
    if mark.is_file():
        saved=json.loads(mark.read_text())
        if saved.get('status')=='completed':
            if all(file_digest(out/k)==v for k,v in saved['checksums'].items()):
                print('KEEP COMPLETED PILOT:',out,flush=True);return
            raise ValueError('Pilot output checksum mismatch')
    write_json(mark,dict(status='running',stage='source'))
    try:
        with tempfile.TemporaryDirectory(prefix='zuco-python-pilot-') as temp:
            paths=[]
            for source in sources:
                print('Temporary cloud source:',source['name'],flush=True)
                path,sha=fetch_sample(source,Path(temp))
                if sha!=source['sha256']:raise ValueError('Source checksum differs from audited copy')
                paths.append(path)
            eeg,rate,codes,latencies,names=native_continuous(paths[0])
            from scipy.io import loadmat
            eye=loadmat(paths[1],squeeze_me=True,struct_as_record=False)
            sync=synchronize(codes,latencies,eye['event'],rate,args.engineering_timing_tolerance_ms)
            print('SYNC:',json.dumps(sync),flush=True)
            if not sync['native_one_sample_target_met']:
                print('ENGINEERING ONLY: native 2-ms paper target not met; do not call this a faithful reproduction.',flush=True)
            events,headers,outside=normalized_events(eye,codes,latencies,rate,sync,eeg.shape[1]/rate)
            print('EVENTS:',events.event_type.value_counts().to_dict(),flush=True)
            # Input is the corpus's already average-referenced 105-channel data.
            # Do not introduce a new, subset-dependent reference for a small QA run.
            eeg=eeg[:args.channels]
            eeg,filter_info=filter_and_resample(eeg,rate)
            design,design_info=time_expanded_design(events,eeg.shape[1])
            print(f'DESIGN: {design.shape}, {design.nnz} nonzero entries',flush=True)
            write_json(mark,dict(status='running',stage='nuisance_glm'))
            def progress(i,n):print(f'NUISANCE GLM: channel {i}/{n}',flush=True)
            residual,betas,solvers=fit_nuisance(eeg,design,progress=progress)
            raw_mean,clean_mean,counts=audit_epochs(eeg,residual,events)
            atomic_npz(out/'nuisance_betas_and_qa_means.npz',betas=betas,
                       raw_all_fixation_mean_uV=raw_mean,residual_all_fixation_mean_uV=clean_mean,
                       times_ms=np.arange(-75,126)*8,channel_names=np.array(names[:args.channels]))
            events.to_csv(out/'native_event_timeline.csv.gz',index=False,compression='gzip')
            diagnostics=dict(synchronization=sync,event_headers=headers,event_counts=events.event_type.value_counts().to_dict(),
                             events_outside_recording=outside,filter=filter_info,design=design_info,solver=solvers,
                             qa_epoch_counts=counts,source_files_removed_on_exit=True,
                             note='Residual variance reduction is an in-sample fit diagnostic, not validation of physiological specificity')
            write_json(out/'diagnostics.json',diagnostics)
            draw(raw_mean,clean_mean,out)
        names_to_hash=['nuisance_betas_and_qa_means.npz','native_event_timeline.csv.gz','diagnostics.json','engineering_qa.png']
        write_json(mark,dict(status='completed',scope='engineering QA only; not sentiment results',
                            finished_at_utc=datetime.now(timezone.utc).isoformat(),
                            checksums={k:file_digest(out/k) for k in names_to_hash}))
        print('PYTHON ENGINEERING PILOT COMPLETE:',out,flush=True)
    except Exception as error:
        if hasattr(error,'diagnostics'):
            write_json(out/'synchronization_failure.json',error.diagnostics)
        write_json(mark,dict(status='failed',error=f'{type(error).__name__}: {error}'))
        raise


def draw(raw,clean,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # GFP avoids the near-zero all-channel signed mean under average reference.
    times=np.arange(-75,126)*8
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(times,raw.std(axis=0),label='Before nuisance-model subtraction')
    ax.plot(times,clean.std(axis=0),label='After nuisance-model subtraction')
    ax.axvline(0,color='black',linewidth=.8)
    ax.set(xlabel='Fixation onset (ms)',ylabel='GFP of all-fixation QA mean (source µV assumed)',
           title=f'ZAB SR block 1 · {raw.shape[0]} channels · engineering QA, no sentiment groups')
    ax.legend();fig.tight_layout();fig.savefig(out/'engineering_qa.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
