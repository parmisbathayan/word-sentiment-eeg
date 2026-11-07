"""Sentence-labelled released ZuCo data mapped to native continuous EEG.

Never infer sentence identity from nominal block/trial order. A full multichannel
waveform match supplies the recorded location; missing/ambiguous matches exclude
the source sentence. The high-level source supplies explicit text and word bounds.
"""
from pathlib import Path
import numpy as np
from .erp import locate,_v5_fixations,_hdf_fixations
from .raw_export import orient


def released_sentences(path):
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as h:
            data=h['sentenceData'];refs=np.asarray(data['word']).reshape(-1)
            for i,ref in enumerate(refs):
                if not ref or not isinstance(h[ref],h5py.Group):continue
                word=h[ref]
                if 'content' not in word:continue
                words=[''.join(chr(int(c)) for c in np.asarray(h[r]).reshape(-1) if int(c)>0) for r in np.asarray(word['content']).reshape(-1)]
                rawref=np.asarray(data['rawData']).reshape(-1)[i]
                raw=orient(np.asarray(h[rawref]),from_hdf5=True) if rawref else None
                boundref=np.asarray(data['wordbounds']).reshape(-1)[i]
                bounds=np.asarray(h[boundref]).T if boundref else None
                fixrefs=np.asarray(word['rawEEG']).reshape(-1) if 'rawEEG' in word else [None]*len(words)
                segments=[_hdf_fixations(h,r) if r else [None] for r in fixrefs]
                yield dict(position=i,words=words,raw=raw,bounds=bounds,segments=segments)
    else:
        from scipy.io import loadmat
        data=loadmat(path,squeeze_me=True,struct_as_record=False,variable_names=['sentenceData'])
        for i,s in enumerate(np.atleast_1d(data['sentenceData'])):
            entries=list(np.atleast_1d(getattr(s,'word',[])))
            if not entries or not hasattr(entries[0],'content'):continue
            yield dict(position=i,words=[str(w.content).strip() for w in entries],raw=orient(getattr(s,'rawData',None)),
                       bounds=np.asarray(getattr(s,'wordbounds',[])),segments=[_v5_fixations(getattr(w,'rawEEG',None)) for w in entries])


def sentence_location(continuous,sentence):
    raw=sentence['raw']
    if raw is None or not np.isfinite(raw).all():return None,'missing_released_sentence_eeg'
    return locate(continuous,raw)


def native_recording(path):
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as h:
            eeg=h['EEG'];values=eeg['data'][()].T;rate=float(eeg['srate'][()].squeeze())
            def refs(ds,chars=False):
                out=[]
                for ref in ds[()].reshape(-1):
                    v=h[ref][()].reshape(-1);out.append(''.join(chr(int(c)) for c in v) if chars else float(v[0]))
                return out
            codes=refs(eeg['event/type'],True);latencies=refs(eeg['event/latency']);channels=refs(eeg['chanlocs/labels'],True)
    else:
        from scipy.io import loadmat
        eeg=loadmat(path,squeeze_me=True,struct_as_record=False)['EEG'];values=np.asarray(eeg.data);rate=float(eeg.srate)
        events=np.atleast_1d(eeg.event);codes=[str(e.type) for e in events];latencies=[float(e.latency) for e in events]
        channels=[str(c.labels) for c in np.atleast_1d(eeg.chanlocs)]
    if values.ndim!=2 or values.shape[0]!=105 or len(channels)!=105 or rate!=500 or not np.isfinite(values).all():
        raise ValueError('Unexpected released continuous EEG channels/rate/values')
    return values.astype(np.float32),rate,codes,np.asarray(latencies),channels
