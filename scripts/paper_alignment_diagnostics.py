#!/usr/bin/env python3
"""Cloud-only source/timing investigation; no sentiment estimates or Drive writes.

Compare direct affine timing with an independent reconstruction of EYE-EEG's
regression direction and rounded endpoint grid. Do not delete difficult anchors
or change the acceptance criterion. Inspect corpus scripts for mapping evidence.
"""
import json
from pathlib import Path
import re
import sys
import tempfile
import numpy as np
from scipy.io import loadmat
from paper_source_audit import fetch_sample,request
from zuco_paper_pilot import native_continuous

ROOT=Path(__file__).resolve().parents[1]


def summarize_errors(errors,rate):
    rounded=np.sign(errors)*np.floor(np.abs(errors)+.5)
    return dict(max_continuous_ms=float(np.max(np.abs(errors)))*1000/rate,
                rms_continuous_ms=float(np.sqrt(np.mean(errors**2)))*1000/rate,
                max_discrete_samples=int(np.max(np.abs(rounded))),
                discrete_distribution={str(int(i)):int((rounded==i).sum()) for i in np.unique(rounded)},
                all_within_one_native_sample=bool(np.max(np.abs(rounded))<=1))


def compare(codes,latencies,eye,rate,allowed):
    codes=np.asarray([int(str(c).strip()) for c in codes]);e=np.asarray(eye['event'],float)
    ix=np.isin(codes,allowed);ex=np.isin(e[:,1],allowed)
    if not np.array_equal(codes[ix],e[ex,1]):return dict(status='trigger_sequence_mismatch')
    x=np.asarray(latencies)[ix];y=e[ex,0]
    if len(x)<3:raise ValueError('Too few anchors')
    x0=x.mean();y0=y.mean()
    direct=np.polyfit(y-y0,x-x0,1);mapped=(y-y0)*direct[0]+direct[1]+x0
    inverse=np.polyfit(x-x0,y-y0,1)
    # EYE-EEG source rounds fitted endpoint eye timestamps, then creates a
    # linear interpolation grid across corresponding EEG sample indices.
    endpoints=np.floor((np.array([x[0],x[-1]])-x0)*inverse[0]+inverse[1]+y0+.5)
    remapped=x[0]+(y-endpoints[0])/(endpoints[1]-endpoints[0])*(x[-1]-x[0])
    # The original toolbox replaces the endpoint timestamps with their fitted
    # rounded values for its final QC. Report BOTH unchanged and replaced views.
    replaced=remapped.copy();replaced[[0,-1]]=x[[0,-1]]
    return dict(status='exact_sequence',anchors=len(x),codes=allowed,
                direct_affine=summarize_errors(mapped-x,rate),
                eyeeg_style_unchanged_anchors=summarize_errors(remapped-x,rate),
                eyeeg_style_replaced_endpoint_qc=summarize_errors(replaced-x,rate),
                note='Independent equation-level diagnostic; not executing MATLAB or asserting full toolbox parity')


def main():
    if sys.platform=='darwin':raise SystemExit('Cloud only; never download EEG/source datasets to the Mac')
    inventory=json.loads((ROOT/'docs/paper_source_audit.json').read_text())
    names=['ZAB_SR1_EEG.mat','gip_ZAB_SR1_EEG.mat','ZAB_SR1_ET.mat','ZAB_SR1_corrected_ET.mat']
    report=dict(purpose='Resolve original timing/source identity, not sentiment results',comparisons={})
    with tempfile.TemporaryDirectory(prefix='paper-alignment-') as directory:
        folder=Path(directory);paths={}
        for name in names:
            source=next(s for s in inventory['samples'] if s['name']==name)
            path,sha=fetch_sample(source,folder)
            if sha!=source['sha256']:raise ValueError('Changed audited source bytes')
            paths[name]=path
        _,rate,codes,latencies,_=native_continuous(paths['gip_ZAB_SR1_EEG.mat'])
        original=loadmat(paths['ZAB_SR1_EEG.mat'],squeeze_me=True,struct_as_record=False)
        raw=original['EEG'];events=np.asarray(raw.event,dtype=object).reshape(-1)
        raw_codes=[str(e.type) for e in events];raw_latencies=np.asarray([float(e.latency) for e in events])
        report['raw_preprocessed_trigger_identity']=dict(raw_count=len(events),preprocessed_count=len(codes),
            same_codes=raw_codes==codes,
            max_latency_difference_samples=float(np.abs(raw_latencies-latencies).max()) if len(raw_latencies)==len(latencies) else None)
        for name in ['ZAB_SR1_ET.mat','ZAB_SR1_corrected_ET.mat']:
            eye=loadmat(paths[name],squeeze_me=True,struct_as_record=False)
            report['comparisons'][name]={}
            for label,allowed in [('sentence',[10,11,12,13]),('all_shared',[10,11,12,13,15,100])]:
                report['comparisons'][name][label]=compare(codes,latencies,eye,rate,allowed)
        report['corpus_script_evidence']={}
        for label,url in [('firstLevel_SR','https://osf.io/download/369ur/'),('SyncET2EEG','https://osf.io/download/74w3x/')]:
            with request(url) as response:code=response.read(1_000_000).decode('utf-8',errors='replace')
            # Short read-only evidence from original corpus scripts, not vendored
            # code. No execution of downloaded source and no stimulus text.
            patterns=r'pop_importeyetracker|startEnd|sentences\s*=|wordbounds|ZAB|==\s*10|==\s*11|==\s*12|==\s*13|50|100'
            excerpts=[dict(line=i,text=line.strip()[:220]) for i,line in enumerate(code.splitlines(),1)
                      if re.search(patterns,line) and not line.lstrip().startswith('%')]
            report['corpus_script_evidence'][label]=dict(url=url,matching_lines=excerpts[:35])
    report['temporary_sources_removed']=True
    target=ROOT/'artifacts/paper_alignment_diagnostics.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
