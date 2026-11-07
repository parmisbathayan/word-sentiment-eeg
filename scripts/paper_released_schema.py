#!/usr/bin/env python3
"""Cloud-only schema probe of one released, sentence-labelled ZuCo SR reader.

No waveform artifact is retained. This tests whether the released file already
provides eye traces and fixation timing linked to explicit sentence identities.
"""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_erp import cloud_download


def main():
    if sys.platform=='darwin':raise SystemExit('Cloud only; do not download EEG on the Mac')
    source=next(r for r in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if r['task']=='SR' and r['subject']=='ZAB')
    report=dict(source=source,purpose='Released source schema; no sentiment results')
    with tempfile.TemporaryDirectory(prefix='zuco-released-') as temp:
        path=cloud_download(source,Path(temp)/source['filename'])
        import h5py
        if h5py.is_hdf5(path):
            report['format']='HDF5'
            with h5py.File(path,'r') as h:
                data=h['sentenceData'];report['fields']={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in data.items()}
                report['first_sentence_fields']={}
                for name in data:
                    ds=data[name]
                    if not isinstance(ds,h5py.Dataset) or not h5py.check_dtype(ref=ds.dtype):continue
                    refs=ds[()].reshape(-1);ref=next((r for r in refs if r),None)
                    if ref is None:continue
                    obj=h[ref]
                    if isinstance(obj,h5py.Group):
                        report['first_sentence_fields'][name]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in obj.items()}
                    else:report['first_sentence_fields'][name]=dict(shape=list(obj.shape),dtype=str(obj.dtype))
        else:
            report['format']='MAT_v5'
            from scipy.io import loadmat
            s=np.atleast_1d(loadmat(path,squeeze_me=True,struct_as_record=False,variable_names=['sentenceData'])['sentenceData'])
            report['sentences']=len(s);report['fields']=list(s[0]._fieldnames)
            report['first_sentence_fields']={}
            for name in s[0]._fieldnames:
                value=getattr(s[0],name)
                if hasattr(value,'_fieldnames'):report['first_sentence_fields'][name]=value._fieldnames
                else:
                    arr=np.asarray(value);report['first_sentence_fields'][name]=dict(shape=list(arr.shape),dtype=str(arr.dtype))
            for name in ['allFixations','word']:
                value=getattr(s[0],name,None);values=np.atleast_1d(value)
                if len(values) and hasattr(values[0],'_fieldnames'):
                    report[name+'_fields']={k:dict(shape=list(np.asarray(getattr(values[0],k)).shape),dtype=str(np.asarray(getattr(values[0],k)).dtype)) for k in values[0]._fieldnames}
        # Read-only context; only output field names from original corpus code.
        from urllib.request import urlopen
        import re
        code=urlopen('https://osf.io/download/369ur/',timeout=90).read().decode('utf-8',errors='replace')
        report['corpus_output_field_names']=sorted(set(re.findall(r'sentenceData\(i\)\.([A-Za-z0-9_]+)',code)))
    report['temporary_source_removed']=True
    out=ROOT/'artifacts/paper_released_schema.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
