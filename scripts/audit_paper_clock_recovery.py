#!/usr/bin/env python3
"""Cloud-only independent waveform-clock validation. Does not recover/overwrite EEG.

Candidate clocks use triggers only. Validation pairs are built using sentence
waveform identity, word rectangles, stored fixation order and duration, never
nearest-clock-time matching. Output excludes stimulus text and signal arrays.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT)]
from paper_source_audit import listing,FOLDERS
from prepare_zuco_paper_figures import source_inventory,fetch,event_table,repair_eye_duration_columns
from run_erp import cloud_download
from word_sentiment_eeg.paper_released import native_recording,released_sentences,sentence_location
from word_sentiment_eeg.erp import locate
from word_sentiment_eeg.paper_fixations import map_word
from word_sentiment_eeg.paper_clock_audit import anchors,fit_candidates,reading_intervals,summarize_validation
from word_sentiment_eeg.common import file_digest,write_json


def search_missing():
    """Scan both complete SR folder trees, including misplaced subject files."""
    candidates=[];visited=set();scanned=0
    for kind in ['raw_sr','preprocessed_sr']:
        stack=[(FOLDERS[kind],kind,0)]
        while stack:
            key,path,depth=stack.pop()
            if key in visited:continue
            visited.add(key)
            for item in listing(key):
                scanned+=1
                if item['kind']=='folder':
                    if depth>=4:raise ValueError('Unexpected folder depth; broad source search incomplete')
                    stack.append((item['id'],path+'/'+item['name'],depth+1))
                elif ('ZDN' in item['name'].upper() or 'ZKW' in item['name'].upper()) and 'ET' in item['name'].upper():
                    candidates.append(dict(path=path,**item))
    return dict(scanned_items=scanned,folders=len(visited),scope='Public raw/preprocessed SR trees only; not proof of absence in every release or archive',candidates=candidates)


def validate_block(subject,block,eeg,sentences,eye,corrected,rate,codes,latencies):
    c,x,y=anchors(codes,latencies,eye['event'],rate);clocks=fit_candidates(x,y);intervals=reading_intervals(c,x,y)
    # Corrected selection timestamps must share the original native eye clock.
    cc,cx,cy=anchors(codes,latencies,corrected['event'],rate)
    if not np.array_equal(c,cc) or not np.array_equal(x,cx):raise ValueError('Raw/corrected eye trigger timestamps differ; no common clock established')
    fixes=event_table(corrected,'fixations');duration=fixes.endtime.to_numpy(float)-fixes.latency.to_numpy(float)
    valid_duration=np.isfinite(duration)&(duration>=100)
    counts=Counter();rows=[];sentence_matches=0;used_intervals=set()
    for s in sentences:
        onset,method=sentence_location(eeg,s)
        if onset is None:continue
        sentence_matches+=1;start=onset/rate;stop=(onset+s['raw'].shape[1]-1)/rate
        # Resolve sentence reading interval from waveform location + EEG triggers.
        # No proposed eye-clock map participates in this identity association.
        options=[(i,v) for i,v in enumerate(intervals) if abs(v['eeg_start']-start)<=.100 and abs(v['eeg_stop']-stop)<=.100]
        if len(options)!=1:counts['sentence_interval_unresolved']+=1;continue
        ii,interval=options[0]
        if ii in used_intervals:raise ValueError('Two released sentences claim the same reading interval')
        used_intervals.add(ii)
        pick=(fixes.latency.to_numpy(float)>=interval['eye_start'])&(fixes.endtime.to_numpy(float)<=interval['eye_stop'])&valid_duration
        candidates=fixes.loc[pick].sort_values('latency');by_word={}
        bounds=np.asarray(s['bounds'])
        if bounds.shape!=(len(s['words']),4):counts['invalid_word_bounds']+=1;continue
        for index,fix in candidates.iterrows():
            wi,status=map_word(float(fix.fix_avgpos_x),float(fix.fix_avgpos_y),bounds)
            if wi is not None:by_word.setdefault(wi,[]).append((index,fix))
        for wi,segments in enumerate(s['segments']):
            stored=[];unlocatable=False
            for fi,segment in enumerate(segments):
                if segment is None or not np.isfinite(segment).all():continue
                if segment.shape[1]*1000/rate<100:continue
                local,status=locate(s['raw'],segment)
                if local is None:counts['unmatched_stored_segment']+=1;unlocatable=True;continue
                stored.append((fi,local,segment.shape[1]*1000/rate))
            events=by_word.get(wi,[])
            if unlocatable or len(stored)!=len(events) or not stored:
                counts['word_event_count_or_waveform_unresolved']+=1;continue
            # Pair in actual stored order, corroborated by increasing waveform
            # positions and duration. Never choose the nearest predicted onset.
            if len(stored)>1 and np.any(np.diff([v[1] for v in stored])<=0):counts['unordered_stored_fixations']+=1;continue
            if any(abs(length-(float(f.endtime)-float(f.latency)))>4 for (_,_,length),(_,f) in zip(stored,events)):
                counts['word_duration_sequence_mismatch']+=1;continue
            for (fi,local,length),(index,fix) in zip(stored,events):
                rows.append(dict(sentence_position=s['position'],word_index=wi,stored_fixation_index=fi,
                    corrected_fixation_index=int(index),eye_ms=float(fix.latency),waveform_eeg_seconds=(onset+local)/rate,duration_ms=length))
    # Reused physical EEG onset is never independent evidence for two events.
    keys=[r['waveform_eeg_seconds'] for r in rows];duplicate={v for v,n in Counter(keys).items() if n>1}
    counts['duplicate_waveform_times_excluded']=sum(v in duplicate for v in keys);rows=[r for r in rows if r['waveform_eeg_seconds'] not in duplicate]
    return dict(subject=subject,block=block,source_sentence_matches=sentence_matches,identity_audit=dict(counts),candidate_clocks=clocks,
        validation=summarize_validation(clocks,rows),validation_events=rows,
        conclusion='Audit only. Passing candidate does not alter accepted trials or prove full nuisance-timeline equivalence.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',default='ZAB');p.add_argument('--blocks',nargs='+',default=['3','5','7'])
    p.add_argument('--source-search',action='store_true');p.add_argument('--thesis-root',type=Path);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Run in cloud only; no EEG downloads on Mac')
    target=ROOT/'artifacts/clock_recovery_audit';target.mkdir(parents=True,exist_ok=True)
    if args.source_search:
        report=search_missing();write_json(target/'source_search.json',report);print('SOURCE SEARCH',json.dumps(report),flush=True)
    source=next(s for s in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if s['task']=='SR' and s['subject']==args.subject)
    inventory=source_inventory(args.subject);inventory=[b for b in inventory if b['block_id'] in args.blocks]
    if {b['block_id'] for b in inventory}!=set(args.blocks):raise ValueError('Requested block absent')
    with tempfile.TemporaryDirectory(prefix='clock-validation-') as directory:
        folder=Path(directory);released=None
        if args.thesis_root:
            candidate=args.thesis_root/'Data/zuco_og_raw'/source['filename']
            if candidate.is_file():released=candidate
        if released is None:released=cloud_download(source,folder/source['filename'])
        print('LOAD RELEASED SENTENCE/FIXATION WAVEFORMS',args.subject,flush=True)
        sentences=list(released_sentences(released));print('RELEASED SENTENCES',len(sentences),flush=True)
        for item in inventory:
            print('INDEPENDENT VALIDATION',args.subject,item['block_id'],flush=True)
            if not item['available']:
                write_json(target/f'{args.subject}_{item["block_id"]}.json',dict(status='missing_sources',missing=item['missing_sources']));continue
            with tempfile.TemporaryDirectory(dir=folder,prefix='block-') as bdir:
                paths={k:fetch(item[k],Path(bdir)) for k in ['eeg','eye','corrected']}
                eeg,rate,codes,latencies,channels=native_recording(paths['eeg'])
                eye=loadmat(paths['eye'],squeeze_me=True,struct_as_record=False);corrected=loadmat(paths['corrected'],squeeze_me=True,struct_as_record=False)
                try:report=validate_block(args.subject,item['block_id'],eeg,sentences,eye,corrected,rate,codes,latencies)
                except ValueError as e:report=dict(subject=args.subject,block=item['block_id'],status='unresolved',reason=str(e))
                report['source_checksums']={k:file_digest(v) for k,v in paths.items()}
                write_json(target/f'{args.subject}_{item["block_id"]}.json',report)
                print('AUDIT RESULT',json.dumps({k:v for k,v in report.items() if k!='validation_events'}),flush=True)
                del eeg,eye,corrected
    print('AUDIT COMPLETE: temporary cloud sources deleted; no accepted EEG or Drive results changed',flush=True)


if __name__=='__main__':main()
