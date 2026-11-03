#!/usr/bin/env python3
"""Inspect public reproduction inputs in a cloud runtime, not on the Mac.

This is a source/schema audit, not a reproduction or a new EEG extraction.
Only a small JSON report is retained. Public source samples are temporary.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

OSF='https://api.osf.io/v2/nodes/q3zws/files/osfstorage/'
FOLDERS={'scripts':'5b7d3977c252f70018e184cb',
         'raw_sr':'5b4f1afe6d4eb300126eb6fb',
         'preprocessed_sr':'5b72869fb081040016e5d599'}


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'word-eeg-reproduction-audit/1'}), timeout=90)


def listing(key):
    url=f'{OSF}{key}/?page[size]=100'
    result=[]
    while url:
        with request(url) as response: page=json.load(response)
        for item in page['data']:
            attrs=item['attributes']
            result.append(dict(id=item['id'],name=attrs['name'],kind=attrs['kind'],
                               bytes=attrs.get('size'),download=item['links'].get('download')))
        url=page.get('links',{}).get('next')
    return result


def summarize(value, depth=0):
    import numpy as np
    if isinstance(value,dict):
        return {k:summarize(v,depth+1) for k,v in value.items() if not k.startswith('__')}
    if hasattr(value,'_fieldnames'):
        return {k:summarize(getattr(value,k),depth+1) for k in value._fieldnames}
    if isinstance(value,np.ndarray):
        result=dict(shape=list(value.shape),dtype=str(value.dtype))
        if value.size and value.dtype==object and depth<4:
            result['first_element_schema']=summarize(value.flat[0],depth+1)
        return result
    if isinstance(value,(float,int,np.number)):
        return {'scalar':float(value)}
    # Avoid putting stimuli or arbitrary source text in a report.
    return {'type':type(value).__name__}


def struct_events(value):
    import numpy as np
    if hasattr(value,'_fieldnames'):
        return [value]
    if isinstance(value,np.ndarray) and value.dtype==object:
        return [v for v in value.flat if hasattr(v,'_fieldnames')]
    return []


def inspect_mat(path):
    import numpy as np
    from scipy.io import loadmat
    data=loadmat(path,squeeze_me=True,struct_as_record=False)
    result={'schema':summarize(data),'event_tables':[]}
    def walk(value, name, depth=0):
        if depth>4:return
        if hasattr(value,'_fieldnames'):
            for field in value._fieldnames:
                child=getattr(value,field)
                if field.lower() in {'event','events','urevent'}:
                    events=struct_events(child)
                    fields=sorted({f for e in events for f in e._fieldnames})
                    counts=Counter(str(getattr(e,'type','unknown')) for e in events)
                    result['event_tables'].append(dict(path=name+'.'+field,count=len(events),fields=fields,type_counts=dict(counts)))
                elif hasattr(child,'_fieldnames'):walk(child,name+'.'+field,depth+1)
    for key,value in data.items():
        if not key.startswith('__'):walk(value,key)
    # Explicitly retain only basic recording scalars, never signal arrays.
    for key,value in data.items():
        if hasattr(value,'_fieldnames'):
            info={}
            for field in ['srate','nbchan','pnts','trials','xmin','xmax']:
                scalar=getattr(value,field,None)
                if scalar is not None and np.isscalar(scalar):info[field]=float(scalar)
            if info:result.setdefault('recordings',{})[key]=info
    return result


def fetch_sample(item, directory):
    if not item.get('download') or not item.get('bytes'):
        raise ValueError('Sample lacks a verified download URL or size')
    if item['bytes']>100_000_000:
        raise ValueError('Refusing sample above the 100 MB per-file audit limit')
    path=directory/item['name'];total=0;sha=hashlib.sha256()
    with request(item['download']) as response,path.open('wb') as target:
        while chunk:=response.read(1024*1024):
            total+=len(chunk)
            if total>100_000_000:raise ValueError('Sample exceeded audit download cap')
            sha.update(chunk);target.write(chunk)
    if total!=item['bytes']:raise ValueError('Sample size differs from inventory')
    return path,sha.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud only: sample EEG must not be downloaded to the Mac')
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'Prerequisite audit only; no reproduction results',
            'paper':'https://doi.org/10.1016/j.neuroimage.2020.116934',
            'inventory':{},'samples':[],
            'exact_study_code':'Not located; dataset preprocessing scripts are not the complete 2020 study code',
            'full_reproduction_implemented':False}
    for name,key in FOLDERS.items():
        print('Inventory',name,flush=True)
        report['inventory'][name]=listing(key)
    samples=[]
    for kind in ['raw_sr','preprocessed_sr']:
        folder=next(x for x in report['inventory'][kind] if x['name']=='ZAB' and x['kind']=='folder')
        children=listing(folder['id'])
        report['inventory'][kind+'_ZAB']=children
        wanted=['ZAB_SR1_EEG.mat','ZAB_SR1_ET.mat'] if kind=='raw_sr' else ['gip_ZAB_SR1_EEG.mat','ZAB_SR1_corrected_ET.mat']
        for name in wanted:
            item=next((x for x in children if x['name']==name),None)
            if item:samples.append((kind,item))
    with tempfile.TemporaryDirectory(prefix='zuco-paper-audit-') as temp:
        for kind,item in samples:
            print('Inspect public sample',kind,item['name'],flush=True)
            entry={'group':kind,**item}
            try:
                path,sha=fetch_sample(item,Path(temp));entry['sha256']=sha
                entry['inspection']=inspect_mat(path)
                path.unlink()
            except Exception as error:
                entry['inspection_error']=f'{type(error).__name__}: {error}'
            report['samples'].append(entry)
    report['temporary_source_files_removed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('AUDIT SAVED:',args.output,flush=True)


if __name__=='__main__':main()
