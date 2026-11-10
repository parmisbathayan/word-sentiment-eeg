#!/usr/bin/env python3
"""ZuCo SR only: prepare raw/clean paper-figure statistics from released sources.

Sentence identity is validated by multichannel waveform matching, never nominal
trial order. Downloads are temporary in Colab. Keeps sufficient statistics,
event audits, betas and permutation summaries, not a second trial EEG archive.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from run_erp import cloud_download
from paper_source_audit import listing,FOLDERS
from paper_sst_source import acquire_sst
from zuco_paper_pilot import normalized_events,filter_and_resample
from word_sentiment_eeg.common import digest,file_digest,write_json,atomic_npz,sentence_key
from word_sentiment_eeg.paper_released import released_sentences,sentence_location,native_recording
from word_sentiment_eeg.paper_python import synchronize,eye_to_seconds,fit_reader_blocks
from word_sentiment_eeg.paper_fixations import map_word,blink_predecessors,preceding_negation,selection_outcomes
from word_sentiment_eeg.paper_labels import SST
from word_sentiment_eeg.paper_trials import matched_trials
from word_sentiment_eeg.paper_figure_stats import permutation_reader

TIMES=np.arange(-75,126)*8
BIN_EDGES=np.r_[np.arange(-600,1000,60),1000]
DEPARTURES=[
 'Released corpus preprocessing is reused; Automagic/MARA are not rerun.',
 'Independent Python 1-50 Hz FIR, splines and least-squares solver; MATLAB numerical equivalence unverified.',
 'Native timing tolerance is one output-grid sample (8 ms), not the reported 2 ms; every block residual is reported.',
 'Blocks exceeding the 8-ms global clock-fit limit are excluded with observed fixation counts and diagnostics; coverage can differ from the paper.',
 'Sentence text and bounds are located by full-waveform match; unmatched released sentences are not guessed.',
 'Original SST labels with explicit dictionary-only fallback; ambiguous or missing original labels are excluded.',
 'Original word rectangles; blink-predecessor rule is an explicit operational reconstruction.',
 'Matching: reader, exact word length 1-20, tertiles of previous and next fixation-onset gaps; exact original sampling code unavailable.',
 'Python normal/identity random-intercept-and-slope ML with asymptotic Wald p-values; no MATLAB fitglme numerical-parity claim.',
 'Electrode adjacency uses distance-pruned Delaunay geometry, not a recovered FieldTrip neighbor file.',
 'Word frequency counts case-folded, edge-punctuation-stripped occurrences in the released SR text pool; normalization is explicit.',
 'Control-panel neighbor-bin, duration and frequency tests use reader-paired t-tests; EEG/GFP inference uses individual-trial mixed models.',
 'Eye velocity uses recorded pixels/second; no invented visual-angle calibration.',
 'Eye-event duration is recomputed from recorded endtime minus latency; inconsistent stored duration exports are audited.',
 'Topography panels use predeclared reference windows (raw 224-280 ms; cleaned 224-304 ms), including when this run is nonsignificant.',
 'The four main figures are the scope; published decoding Table 1 is a separate analysis, not fabricated from existing linear/CNN scores.'
]


def source_inventory(subject):
    audit=json.loads((ROOT/'docs/paper_source_audit.json').read_text())
    raw_folder=next(x for x in audit['inventory']['raw_sr'] if x['name']==subject and x['kind']=='folder')
    processed_folder=next(x for x in audit['inventory']['preprocessed_sr'] if x['name']==subject and x['kind']=='folder')
    raw=listing(raw_folder['id']);processed=listing(processed_folder['id']);blocks=[]
    for item in processed:
        match=re.match(r'^(?:[A-Za-z]+_)?'+re.escape(subject)+r'_(?:SR|SNR)(\d+)_EEG\.mat$',item['name'])
        if not match:continue
        number=int(match[1]);eye_name=f'{subject}_SR{number}_ET.mat';corrected_name=f'{subject}_SR{number}_corrected_ET.mat'
        eye=next((x for x in raw if x['name']==eye_name),None);corrected=next((x for x in processed if x['name']==corrected_name),None)
        if eye is None or corrected is None:raise ValueError('Missing native eye-event source for '+item['name'])
        blocks.append(dict(block_id=str(number),eeg=item,eye=eye,corrected=corrected))
    ids=[b['block_id'] for b in blocks]
    if len(ids)!=len(set(ids)) or not blocks:raise ValueError('Ambiguous or absent preprocessed native recording blocks')
    return sorted(blocks,key=lambda b:int(b['block_id']))


def fetch(item,folder):
    path=folder/item['name']
    if not path.is_file():cloud_download(dict(filename=item['name'],url=item['download'],bytes=item['bytes']),path)
    return path


def event_table(eye,name):
    value=getattr(eye['eyeevent'],name);columns=[str(v) for v in np.asarray(value.colheader).reshape(-1)]
    values=np.asarray(value.data,float)
    if values.size==0:values=np.empty((0,len(columns)))
    if values.ndim==1:values=values[None,:]
    return pd.DataFrame(values,columns=columns)


def repair_eye_duration_columns(eye):
    """Use recorded interval endpoints, not physically inconsistent duration fields.

    Several released native exports have finite but impossible stored durations.
    The onset/end timestamps remain independently usable. No gaps or onsets are
    guessed. All durations use the same explicit end-minus-start convention.
    """
    audit={}
    for name in ['fixations','saccades','blinks']:
        source=getattr(eye['eyeevent'],name);columns=[str(v) for v in np.asarray(source.colheader).reshape(-1)]
        array=np.asarray(source.data,float)
        if not array.size:audit[name]=dict(events=0);continue
        if array.ndim==1:array=array[None,:]
        start,end,duration=[columns.index(k) for k in ['latency','endtime','duration']]
        physical=array[:,end]-array[:,start]
        if not np.isfinite(physical).all() or np.any(physical<=0) or np.any(physical>10000):
            raise ValueError('Eye-event endpoints do not form plausible millisecond intervals: '+name)
        audit[name]=dict(events=len(array),stored_duration_mismatches=int((np.abs(array[:,duration]-physical)>2).sum()),
                         minimum_recomputed_ms=float(physical.min()),maximum_recomputed_ms=float(physical.max()))
        array=array.copy();array[:,duration]=physical;source.data=array
    return audit


def sentence_metadata(path,sst):
    records=[]
    for value in released_sentences(path):
        value.pop('segments',None) # keep labelled sentence EEG only until it is mapped
        joined=sst.join(value['words'],allow_dictionary_fallback=True)
        value['labels']=joined['words'];value['sst_sentence_status']=joined['status'];value['sentence_key']=sentence_key(value['words'])
        records.append(value)
    return records


def map_block(subject,block_id,raw,rate,codes,latencies,eye,corrected,sentences,used):
    sync=synchronize(codes,latencies,eye['event'],rate,max_error_ms=8.)
    timeline,headers,outside=normalized_events(eye,codes,latencies,rate,sync,raw.shape[1]/rate)
    fixes=event_table(corrected,'fixations');blinks=event_table(eye,'blinks')
    starts=eye_to_seconds(fixes.latency.to_numpy(),sync)
    blink_times=eye_to_seconds(blinks.latency.to_numpy(),sync)
    blink_flags=blink_predecessors(starts,fixes.duration.to_numpy(),blink_times,blinks.duration.to_numpy())
    raw_starts=timeline.loc[timeline.event_type.eq('fixation'),'onset_seconds'].to_numpy()
    maps=[];mapping_audit=[]
    for sentence in sentences:
        if sentence['position'] in used:continue
        onset,status=sentence_location(raw,sentence)
        if onset is not None:
            bounds=np.asarray(sentence['bounds'])
            if bounds.shape!=(len(sentence['words']),4):raise ValueError('Source word/bounds count mismatch')
            maps.append((onset/rate,(onset+sentence['raw'].shape[1]-1)/rate,sentence))
            used.add(sentence['position']);mapping_audit.append(dict(sentence_position=sentence['position'],onset_sample=onset,status=status,block_id=block_id))
    maps.sort(key=lambda value:value[0])
    if any(a[1]>=b[0] for a,b in zip(maps,maps[1:])):raise ValueError('Overlapping released sentence locations')
    rows=[]
    for index,fix in fixes.iterrows():
        onset=starts[index]
        if not 0<=onset<raw.shape[1]/rate:continue
        row=dict(event_id=digest([subject,block_id,int(index)])[:32],subject=subject,block_id=block_id,onset_seconds=float(onset),
                 duration_ms=float(fix.duration),gaze_x=float(fix.fix_avgpos_x),gaze_y=float(fix.fix_avgpos_y),
                 blink_predecessor=bool(blink_flags[index]),mapping_status='unavailable_sentence_mapping',word_label_status='unknown',
                 sentiment='unknown',preceded_by_negation=False,word='',word_index=-1,word_length=0,sentence_key='',label_5=99,
                 previous_gap_ms=np.nan,next_gap_ms=np.nan,neighbor_offsets_ms=[])
        candidates=[entry for entry in maps if entry[0]<=onset<=entry[1]]
        if len(candidates)>1:raise ValueError('A fixation maps to multiple released sentences')
        if candidates:
            _,_,sentence=candidates[0];wi,status=map_word(row['gaze_x'],row['gaze_y'],sentence['bounds']);row['mapping_status']=status
            if wi is not None:
                word=sentence['words'][wi];label=sentence['labels'][wi] if wi<len(sentence['labels']) else {}
                original=label.get('word_label')
                row.update(word=word,word_index=wi,word_length=len(word),sentence_key=sentence['sentence_key'],
                           sentence_position=sentence['position'],preceded_by_negation=preceding_negation(sentence['words'],wi),
                           original_label_method=label.get('word_label_rule'),sst_sentence_status=sentence['sst_sentence_status'])
                if original:row.update(word_label_status='matched',sentiment=original['sentiment'],label_5=int(original['label_5']))
        j=np.searchsorted(raw_starts,onset);near=[k for k in [j-1,j] if 0<=k<len(raw_starts)]
        if near:
            j=min(near,key=lambda k:abs(raw_starts[k]-onset))
            if abs(raw_starts[j]-onset)<=.002:
                offsets=[(raw_starts[j+k]-onset)*1000 if 0<=j+k<len(raw_starts) else np.nan for k in [-3,-2,-1,1,2,3]]
                row['neighbor_offsets_ms']=offsets
                row['previous_gap_ms']=-offsets[2];row['next_gap_ms']=offsets[3]
        rows.append(row)
    table=pd.DataFrame(rows)
    if table.empty:raise ValueError('No corrected fixation events in the native block')
    table['selection_status']=selection_outcomes(table)
    return table,timeline,dict(block_id=block_id,synchronization=sync,raw_events_outside_recording=outside,
                              source_mapping=mapping_audit),maps


def excluded_clock_block(subject,block_id,corrected,error):
    """Retain every observed fixation as an explicit source-level exclusion."""
    fixes=event_table(corrected,'fixations');rows=[]
    for index,fix in fixes.iterrows():
        rows.append(dict(event_id=digest([subject,block_id,int(index)])[:32],subject=subject,block_id=block_id,
            onset_seconds=np.nan,duration_ms=float(fix.duration),gaze_x=float(fix.fix_avgpos_x),gaze_y=float(fix.fix_avgpos_y),
            blink_predecessor=False,mapping_status='unavailable_clock_alignment',word_label_status='unknown',sentiment='unknown',
            preceded_by_negation=False,word='',word_index=-1,word_length=0,sentence_key='',label_5=99,
            previous_gap_ms=np.nan,next_gap_ms=np.nan,neighbor_offsets_ms=[],selection_status='block_failed_clock_alignment'))
    return pd.DataFrame(rows),dict(block_id=block_id,status='excluded_clock_alignment',
        reason=str(error),synchronization=error.diagnostics,corrected_fixations_excluded=len(rows),source_mapping=[])


def matching(table):
    eligible=table.selection_status.eq('eligible_before_matching')
    bad_length=~table.word_length.between(1,20)
    bad_neighbors=~np.isfinite(table[['previous_gap_ms','next_gap_ms']]).all(axis=1)
    table.loc[eligible&bad_length,'selection_status']='word_length_outside_1_20'
    table.loc[table.selection_status.eq('eligible_before_matching')&bad_neighbors,'selection_status']='missing_neighbor_timing'
    eligible=table[table.selection_status.eq('eligible_before_matching')].copy();bins={}
    table['eligible_for_matching']=table.selection_status.eq('eligible_before_matching')
    if eligible.empty:raise ValueError('No eligible original-SST sentiment events')
    for column in ['previous_gap_ms','next_gap_ms']:
        edges=np.unique(np.quantile(eligible[column],[1/3,2/3]));bins[column]=edges.tolist()
        eligible[column+'_stratum']=np.searchsorted(edges,eligible[column],side='right')
    indices,audit,summary=matched_trials(eligible,['subject','word_length','previous_gap_ms_stratum','next_gap_ms_stratum'])
    chosen=set(eligible.iloc[indices].event_id)
    table.loc[table.selection_status.eq('eligible_before_matching'),'selection_status']='excluded_by_matching'
    table.loc[table.event_id.isin(chosen),'selection_status']='matched_before_epoch_rejection'
    return table,bins,summary


def eye_velocity(eye,sync,epoch_times):
    if '_paper_velocity' in eye:
        times,velocity=eye['_paper_velocity']
        return np.stack([np.interp(epoch_times,times,velocity[:,c],left=np.nan,right=np.nan) for c in range(2)])
    data=np.asarray(eye['data'],float)
    if data.ndim!=2 or data.shape[1]<4:raise ValueError('Native gaze time series unavailable')
    times=eye_to_seconds(data[:,0],sync)
    if np.any(np.diff(times)<=0):raise ValueError('Eye sample clock is not strictly increasing')
    valid=np.isfinite(data).all(axis=1)&(data[:,1]>0)&(data[:,2]>0)&(data[:,3]>0)
    velocity=np.gradient(data[:,1:3],times,axis=0)
    valid &=np.r_[False,valid[:-1]]&np.r_[valid[1:],False]
    velocity[~valid]=np.nan
    eye['_paper_velocity']=(times,velocity)
    return np.stack([np.interp(epoch_times,times,velocity[:,c],left=np.nan,right=np.nan) for c in range(2)])


def prepare_reader(subject,source,inventory,sst,out,null_folder,limits,use_gpu,source_paths=None,
                   defer_permutations=False,storage_root=None,preparation_budget_bytes=None):
    from scipy.io import loadmat
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='zuco-paper-reader-') as temporary:
        folder=Path(temporary)
        existing=source_paths.get(source['filename']) if source_paths else None
        path=Path(existing) if existing else cloud_download(source,folder/source['filename'])
        if path.stat().st_size!=source['bytes']:raise ValueError('Released source size changed')
        released_sha=file_digest(path)
        sentences=sentence_metadata(path,sst);blocks=[];tables=[];diagnostics=[];used=set();example=None;example_sentence=None
        frequency=Counter(word.strip('.,!?;:"()[]{}').casefold() for s in sentences for word in s['words'])
        for bi,item in enumerate(inventory,1):
            print(f'SOURCE {subject}: block {bi}/{len(inventory)}',flush=True)
            eeg_path=fetch(item['eeg'],folder);eye_path=fetch(item['eye'],folder);corrected_path=fetch(item['corrected'],folder)
            raw,rate,codes,latencies,channels=native_recording(eeg_path)
            montage=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.astype(str).tolist()
            if channels!=montage:raise ValueError('Native channel order differs from the verified 105-channel montage')
            eye=loadmat(eye_path,squeeze_me=True,struct_as_record=False);corrected=loadmat(corrected_path,squeeze_me=True,struct_as_record=False)
            duration_audit=dict(raw=repair_eye_duration_columns(eye),corrected=repair_eye_duration_columns(corrected))
            try:
                table,timeline,diagnostic,maps=map_block(subject,item['block_id'],raw,rate,codes,latencies,eye,corrected,sentences,used)
            except ValueError as error:
                if not hasattr(error,'diagnostics'):raise
                table,diagnostic=excluded_clock_block(subject,item['block_id'],corrected,error)
                diagnostic['duration_endpoint_audit']=duration_audit
                diagnostic['source_sha256']={p.name:file_digest(p) for p in [eeg_path,eye_path,corrected_path]}
                tables.append(table);diagnostics.append(diagnostic)
                print(f'EXCLUDE SOURCE BLOCK {subject}/{item["block_id"]}: {error}; {len(table)} observed fixations',flush=True)
                for p in [eeg_path,eye_path,corrected_path]:p.unlink(missing_ok=True)
                continue
            diagnostic['status']='included'
            diagnostic['duration_endpoint_audit']=duration_audit
            filtered,filter_info=filter_and_resample(raw,rate)
            diagnostic['filter']=filter_info;diagnostics.append(diagnostic);tables.append(table)
            diagnostic['source_sha256']={p.name:file_digest(p) for p in [eeg_path,eye_path,corrected_path]}
            blocks.append(dict(block_id=item['block_id'],eeg=filtered,events=timeline,channel_names=channels,eye=eye,sync=diagnostic['synchronization']))
            if example_sentence is None and maps:
                start,stop,s=maps[0];example_sentence=dict(words=s['words'],bounds=np.asarray(s['bounds']).tolist(),
                    gaze=table.loc[(table.onset_seconds>=start)&(table.onset_seconds<=stop),['gaze_x','gaze_y','duration_ms','sentiment']].to_dict('records'))
            # Temporary source copies are no longer needed after conversion.
            for p in [eeg_path,eye_path,corrected_path]:p.unlink(missing_ok=True)
        # Sentence identities have been carried forward in event rows. Release
        # the large source sentence arrays before constructing the nuisance GLM.
        del sentences
        if not blocks:raise ValueError('No native blocks passed clock alignment for '+subject)
        table=pd.concat(tables,ignore_index=True)
        if table.event_id.duplicated().any():raise ValueError('Repeated original fixation event')
        write_json(out/'pre_matching_counts.json',dict(selection=dict(Counter(table.selection_status)),mapping=dict(Counter(table.mapping_status)),
            labels=dict(Counter(table.sentiment)),known_neighbors=int(np.isfinite(table[['previous_gap_ms','next_gap_ms']]).all(axis=1).sum()),
            duration_range_ms=[float(table.duration_ms.min()),float(table.duration_ms.max())],mapped_source_sentences=len(used)))
        print('SELECTION AUDIT', (out/'pre_matching_counts.json').read_text(),flush=True)
        table,bins,match_counts=matching(table)
        write_json(out/'source_mapping.json',dict(blocks=diagnostics,matched_source_sentences=len(used),released_sha256=released_sha))
        match_counts.to_csv(out/'matching_counts.csv',index=False)
        residual,betas,model=fit_reader_blocks(blocks,progress=lambda a,b:print(f'DECONVOLUTION {subject}: channel {a}/{b}',flush=True))
        atomic_npz(out/'nuisance_betas.npz',betas=betas)
        epochs=[[],[]];accepted=[];eyes=[];neighbors=[]
        sums=np.zeros((2,2,106,201),np.float64);squares=np.zeros_like(sums);counts=np.zeros(2,int)
        map_sums=np.zeros((2,2,105));map_squares=np.zeros_like(map_sums)
        for block in blocks:
            indices=table.index[(table.block_id==block['block_id'])&table.selection_status.eq('matched_before_epoch_rejection')]
            for index in indices:
                row=table.loc[index];center=int(np.floor(float(row.onset_seconds)*125+.5));ix=center+np.arange(-75,126)
                if ix[0]<0 or ix[-1]>=block['eeg'].shape[1]:table.loc[index,'selection_status']='epoch_outside_block';continue
                pair=np.stack([block['eeg'][:,ix],residual[block['block_id']][:,ix]])
                if not np.isfinite(pair).all():table.loc[index,'selection_status']='nonfinite_paired_epoch';continue
                if np.max(np.abs(pair[0]))>90:table.loc[index,'selection_status']='raw_amplitude_over_90uV';continue
                table.loc[index,'selection_status']='accepted';table.loc[index,'word_frequency']=frequency[row.word.strip('.,!?;:"()[]{}').casefold()]
                c=0 if row.sentiment=='negative' else 1;counts[c]+=1
                value=np.concatenate([pair,np.std(pair,axis=1)[:,None,:]],axis=1)
                sums[:,c]+=value;squares[:,c]+=value.astype(np.float64)**2
                for r,stop in enumerate([280,304]):
                    window=(TIMES>=224)&(TIMES<=stop);topo=pair[r][:,window].mean(axis=1)
                    map_sums[r,c]+=topo;map_squares[r,c]+=topo.astype(np.float64)**2
                for r in range(2):epochs[r].append(pair[r])
                accepted.append(index);eyes.append(eye_velocity(block['eye'],block['sync'],float(row.onset_seconds)+TIMES/1000))
                neighbors.append(np.stack([np.histogram([offset],bins=BIN_EDGES)[0] if np.isfinite(offset) else np.zeros(len(BIN_EDGES)-1) for offset in row.neighbor_offsets_ms]))
                if example is None:example=pair
        if min(counts)<2:raise ValueError(f'Insufficient accepted paired classes for {subject}: {counts.tolist()}')
        accepted_table=table.loc[accepted].reset_index(drop=True)
        # Rank within reader by word length and fixation duration, as described
        # for the paper's trial-paired consistency analysis. Drop remainders.
        orders=[accepted_table[accepted_table.sentiment.eq(c)].sort_values(['word_length','duration_ms','event_id'],kind='stable').index.to_numpy() for c in ['negative','positive']]
        n=min(map(len,orders));neg,pos=[order[:n] for order in orders]
        raw_diff=np.stack(epochs[0])[pos]-np.stack(epochs[0])[neg]
        clean_diff=np.stack(epochs[1])[pos]-np.stack(epochs[1])[neg]
        if defer_permutations:
            from word_sentiment_eeg.paper_stage_archive import save_trials
            save_trials(out,np.stack([np.stack(v) for v in epochs]),accepted_table,neg,pos,
                        storage_root,preparation_budget_bytes)
            diff=np.stack([raw_diff,clean_diff]).astype(np.float32)
            perm=dict(pair_sum=diff.sum(axis=1,dtype=np.float64),
                      pair_squares=(diff.astype(np.float64)**2).sum(axis=1),pairs=n)
        else:
            perm=permutation_reader(raw_diff,clean_diff,null_folder,subject,limits[0],limits[1],use_gpu=use_gpu)
        eye_values=np.stack(eyes);neighbor_values=np.stack(neighbors);eye_means=[];neighbor_means=[]
        eye_sums=np.zeros((2,2,201));eye_squares=np.zeros_like(eye_sums);eye_counts=np.zeros(2,int)
        for c in ['negative','positive']:
            pick=accepted_table.sentiment.eq(c).to_numpy()
            ci=0 if c=='negative' else 1
            usable=pick&np.isfinite(eye_values).all(axis=(1,2));eye_counts[ci]=int(usable.sum())
            eye_sums[ci]=eye_values[usable].sum(axis=0);eye_squares[ci]=(eye_values[usable]**2).sum(axis=0)
            with __import__('warnings').catch_warnings():
                __import__('warnings').simplefilter('ignore',RuntimeWarning);eye_means.append(eye_sums[ci]/max(1,eye_counts[ci]))
            neighbor_means.append(neighbor_values[pick].mean(axis=0))
        pz_index=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.astype(str).tolist().index('E62')
        pz_trials=np.stack([np.stack(v)[:,pz_index] for v in epochs])
        example=sums[:,:,:105].sum(axis=1)/counts.sum()
        atomic_npz(out/'statistics.npz',counts=counts,sums=sums,squares=squares,map_sums=map_sums,map_squares=map_squares,eye_means=np.stack(eye_means),
                   neighbor_means=np.stack(neighbor_means),times_ms=TIMES,neighbor_edges=BIN_EDGES,
                   eye_counts=eye_counts,eye_sums=eye_sums,eye_squares=eye_squares,
                   pair_sum=perm['pair_sum'],pair_squares=perm['pair_squares'],pairs=np.array(n),example=example,
                   pz_trials=pz_trials,pz_duration_ms=accepted_table.duration_ms.to_numpy())
        for column in ['neighbor_offsets_ms']:
            table[column]=table[column].map(lambda value:json.dumps(value,allow_nan=True))
        table.to_csv(out/'fixation_audit.csv.gz',index=False,compression='gzip')
        match_counts.to_csv(out/'matching_counts.csv',index=False)
        write_json(out/'source_mapping.json',dict(blocks=diagnostics,matched_source_sentences=len(used),released_sha256=released_sha))
        write_json(out/'matching_policy.json',dict(tertile_edges_ms=bins,method='reader/exact length/neighbor tertiles'))
        write_json(out/'nuisance_diagnostics.json',model)
        write_json(out/'example_sentence.json',example_sentence)
        if defer_permutations:
            from word_sentiment_eeg.paper_stage_archive import mark_prepared
            mark_prepared(out,subject,counts,n)
            print(f'CPU PREPARED {subject}: trials and statistics verified on Drive; safe to switch runtime',flush=True)
        else:
            write_json(out/'manifest.json',dict(status='completed',subject=subject,accepted_counts=counts.tolist(),pairs=n,
                        checksums={p.name:file_digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--subjects',nargs='+');parser.add_argument('--blocks',nargs='+')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--free-drive-bytes',type=int);parser.add_argument('--tct-permutations',type=int,default=5000)
    parser.add_argument('--cluster-permutations',type=int,default=1000);parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    if sys.platform=='darwin':raise SystemExit('Cloud only: no source downloads or numerical fitting on the Mac')
    out=args.output or args.thesis_root/'Results/word_sentiment_eeg/zuco_paper_figures'
    if args.plots_only:
        from render_zuco_paper_figures import render_all
        render_all(out);return
    if args.tct_permutations<1 or args.cluster_permutations<1:raise ValueError('Permutation counts must be positive')
    sources=[r for r in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if r['task']=='SR']
    if args.subjects:sources=[r for r in sources if r['subject'] in args.subjects]
    if not sources or (args.subjects and set(args.subjects)!={r['subject'] for r in sources}):raise ValueError('Unknown source reader')
    peak=(args.tct_permutations+args.cluster_permutations)*2*105*201*4+350_000_000
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(subjects=[r['subject'] for r in sources],blocks=args.blocks,sources=sources,
                  tct_permutations=args.tct_permutations,cluster_permutations=args.cluster_permutations,
                  lowpass_hz=50,highpass_hz=1,output_hz=125,epoch_ms=[-600,1000],baseline=None,
                  label_source='original SST; no agent sentiment labels',native_timing_tolerance_ms=8,
                  code={name:file_digest(ROOT/name) for name in ['scripts/prepare_zuco_paper_figures.py','scripts/zuco_paper_pilot.py','scripts/paper_sst_source.py','src/word_sentiment_eeg/paper_released.py','src/word_sentiment_eeg/paper_python.py','src/word_sentiment_eeg/paper_labels.py','src/word_sentiment_eeg/paper_fixations.py','src/word_sentiment_eeg/paper_trials.py','src/word_sentiment_eeg/paper_figure_stats.py']},departures=DEPARTURES)
    from importlib.metadata import version
    protocol['software']={name:version(name) for name in ['numpy','scipy','pandas','h5py','scikit-learn']}
    if (out/'protocol.json').is_file() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Figure preparation protocol changed; do not mix outputs')
    if (out/'manifest.json').is_file():
        saved=json.loads((out/'manifest.json').read_text())
        if saved.get('status')=='completed' and all((out/k).is_file() and file_digest(out/k)==v for k,v in saved['figure_checksums'].items()):
            from render_zuco_paper_figures import load_readers
            load_readers(out)
            analysis=out/'analysis';audit=json.loads((analysis/'analysis_manifest.json').read_text())
            if file_digest(analysis/'analysis_manifest.json')!=saved['analysis_manifest_sha256'] or any(file_digest(analysis/k)!=v for k,v in audit['checksums'].items()):
                raise ValueError('Completed analysis checksums changed')
            print('KEEP COMPLETED RUN: verified reader checkpoints and four figure files. No source reads or refitting.',flush=True);return
        raise ValueError('Final figures changed or incomplete; use --plots-only to rebuild from verified statistics')
    # Count already stored workflow files as reusable capacity, so interrupted
    # runs do not incorrectly demand their entire peak allocation a second time.
    existing_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    required=max(0,peak+100_000_000-existing_bytes)
    if args.output is None and (args.free_drive_bytes is None or args.free_drive_bytes<required):
        raise SystemExit(f'Need {required/1e9:.2f} GB additional verified free Drive space (peak allowance {(peak+100_000_000)/1e9:.2f} GB). No extraction started.')
    write_json(out/'protocol.json',protocol)
    (out/'METHOD_DIFFERENCES.md').write_text('# Python reconstruction differences\n\n'+'\n'.join('- '+v for v in DEPARTURES)+'\n')
    with tempfile.TemporaryDirectory(prefix='paper-sst-') as temporary:
        sst_path,provenance=acquire_sst(Path(temporary));sst=SST(sst_path);write_json(out/'original_label_source.json',provenance)
        for i,source in enumerate(sources,1):
            subject=source['subject'];reader=out/'readers'/subject;mark=reader/'manifest.json'
            if mark.is_file():
                saved=json.loads(mark.read_text())
                if saved.get('status')=='completed' and all(file_digest(reader/k)==v for k,v in saved['checksums'].items()):
                    print(f'KEEP READER {i}/{len(sources)} {subject}',flush=True);continue
                raise ValueError('Reader figure checkpoint corrupt')
            print(f'PREPARE READER {i}/{len(sources)} {subject}',flush=True)
            blocks=source_inventory(subject)
            if args.blocks:blocks=[b for b in blocks if b['block_id'] in args.blocks]
            if not blocks:raise ValueError('Requested native blocks unavailable')
            data_folder=args.thesis_root/'Data/zuco_og_raw';local=data_folder/source['filename']
            prepare_reader(subject,source,blocks,sst,reader,out/'null_work',
                           (args.tct_permutations,args.cluster_permutations),not args.cpu,
                           source_paths={source['filename']:local} if local.is_file() else None)
    write_json(out/'preparation_manifest.json',dict(status='completed',subjects=[r['subject'] for r in sources]))
    if not args.prepare_only:
        from render_zuco_paper_figures import analyze_and_render
        analyze_and_render(out)
        # Remove only this workflow's reproducible null-map work files after
        # their small final statistics and figures have been saved successfully.
        shutil.rmtree(out/'null_work',ignore_errors=True)
    print('ZUCO PAPER FIGURES:',out,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
