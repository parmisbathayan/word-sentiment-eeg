"""Rebuild selected ZuCo context controls from verified saved word onsets.

The original compact result remains unchanged. No fixation matching or trial EEG
archives are repeated: only the source sentences needed for the new cohorts are
filtered/epoched. Missing source files may download inside Colab, never on Mac.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import run_erp as original
from word_sentiment_eeg.common import atomic_npz,digest,file_digest,sentence_key,write_json

np,pd=original.np,original.pd
CLASSES=original.CLASSES
SOURCE_NEUTRAL='neutral_in_source_neutral_sentence'
COHORTS=('negative_in_neutral_sentence','neutral_in_neutral_sentence','positive_in_neutral_sentence',
         'neutral_in_negative_sentence','neutral_in_positive_sentence','neutral_in_mixed_sentence',SOURCE_NEUTRAL)
GROUPS=tuple('pooled_'+c for c in CLASSES)+COHORTS


def load_source_ratings(path):
    """Inherited original SR review ratings, separate from agent context labels."""
    if path is None:
        return {}
    frame=pd.read_csv(path,keep_default_na=False)
    required={'sentence_key','task','source_sentence_class','original_label'}
    if not required.issubset(frame.columns) or frame.sentence_key.duplicated().any():
        raise ValueError('Source sentence ratings need unique explicit source keys')
    if not frame.task.eq('SR').all() or not frame.source_sentence_class.isin(CLASSES).all():
        raise ValueError('Original source sentence ratings apply only to SR')
    if not all({'negative':-1,'neutral':0,'positive':1}[row.source_sentence_class]==int(row.original_label)
               for row in frame.itertuples()):
        raise ValueError('Original source rating mapping is inconsistent')
    return dict(zip(frame.sentence_key,frame.source_sentence_class))


def select_events(events,contexts,source_ratings=None):
    selected=[]
    source_ratings=source_ratings or {}
    if events.event_id.duplicated().any():
        raise ValueError('Original event IDs are duplicated')
    for row in events[events.status=='accepted'].itertuples():
        label=contexts.get(row.sentence_key)
        if label is None:
            raise ValueError(f'Sentence context is missing: {row.sentence_key}')
        context=label.sentence_context_class
        if context not in {'negative','neutral','positive','mixed','uncertain'}:
            raise ValueError('Unexpected sentence context')
        target=f'{row.contextual_class}_in_{context}_sentence'
        cohorts=[target] if target in COHORTS else []
        source_class=source_ratings.get(row.sentence_key,'unknown') if row.task=='SR' else 'unknown'
        if row.task=='SR' and row.contextual_class=='neutral' and source_class=='neutral':
            cohorts.append(SOURCE_NEUTRAL)
        for cohort in cohorts:
            selected.append({'event_id':row.event_id,'raw_sentence_position':int(row.raw_sentence_position),
                             'sentence_key':row.sentence_key,'word_index':int(row.word_index),'word':row.word,
                             'contextual_class':row.contextual_class,'sentence_context_class':context,
                             'source_sentence_class':source_class,
                             'cohort':cohort,'onset_sample_sentence_0based':int(float(row.onset_sample_sentence_0based))})
    return pd.DataFrame(selected,columns=['event_id','raw_sentence_position','sentence_key','word_index','word',
                                          'contextual_class','sentence_context_class','source_sentence_class',
                                          'cohort','onset_sample_sentence_0based'])


def process_record(source,base_reader,target,item,contexts,fingerprint,channels,source_hash=None,source_ratings=None):
    if (source_hash or file_digest(source))!=item['source_sha256']:
        raise ValueError('Source EEG differs from the original compact result')
    events=pd.read_csv(base_reader/'word_onsets.csv.gz',keep_default_na=False)
    selected=select_events(events,contexts,source_ratings)
    with np.load(base_reader/'class_means.npz',allow_pickle=False) as saved:
        if (not np.array_equal(saved['channel_names'],channels) or not np.array_equal(saved['classes'],CLASSES)
                or not np.array_equal(saved['times_ms'],original.TIMES_MS)):
            raise ValueError('Original class/channel axes changed')
        core=saved['mean_uV'].copy();corecounts=saved['counts'].copy()
        if core.shape!=(3,105,501) or not np.isfinite(core[corecounts>0]).all() or not np.array_equal(corecounts,[item['epochs_per_class'][c] for c in CLASSES]):
            raise ValueError('Original class means/counts changed')
    sums=np.zeros((len(COHORTS),len(channels),len(original.TIMES_MS)),np.float64)
    counts=np.zeros(len(COHORTS),int)
    grouped={int(k):v for k,v in selected.groupby('raw_sentence_position')}
    consumed=set()
    for position,words,raw,_ in original.iter_sentences(source):
        table=grouped.get(position)
        if table is None:
            continue
        key=sentence_key(words)
        if not (table.sentence_key==key).all() or raw is None:
            raise ValueError('Saved sentence/onset alignment does not match the source')
        filtered=original.preprocess(raw)
        # A physical EEG event can belong to both strict-agent-neutral and
        # source-rated-neutral controls. Epoch once, accumulate once per cohort.
        for event_id,memberships in table.groupby('event_id',sort=False):
            row=next(memberships.itertuples())
            if row.word_index>=len(words) or words[row.word_index]!=row.word:
                raise ValueError('Source token differs from the original accepted event')
            value,status=original.epoch(filtered,row.onset_sample_sentence_0based)
            if status!='accepted':
                raise ValueError(f'Previously accepted epoch no longer passes: {row.event_id}, {status}')
            for cohort in memberships.cohort:
                index=COHORTS.index(cohort)
                sums[index]+=value;counts[index]+=1
            consumed.add(event_id)
        if len(consumed)%1000< len(table):
            logging.info('%s/%s: rebuilt %d/%d unique selected epochs',item['task'],item['subject'],len(consumed),selected.event_id.nunique())
    if len(consumed)!=selected.event_id.nunique():
        raise ValueError('Some saved onset positions were not found in the EEG source')
    means=np.full(sums.shape,np.nan,np.float32)
    for i,count in enumerate(counts):
        if count:
            means[i]=(sums[i]/count).astype(np.float32)
    target.mkdir(parents=True,exist_ok=True)
    atomic_npz(target/'cohort_means.npz',mean_uV=np.concatenate([core,means]),
               counts=np.concatenate([corecounts,counts]),groups=np.array(GROUPS),
               channel_names=np.array(channels),times_ms=original.TIMES_MS)
    temporary=target/'selected_events.tmp.csv.gz'
    selected.to_csv(temporary,index=False,compression='gzip')
    os.replace(temporary,target/'selected_events.csv.gz')
    result={'task':item['task'],'subject':item['subject'],'fingerprint':fingerprint,
            'source_sha256':item['source_sha256'],'original_onsets_sha256':file_digest(base_reader/'word_onsets.csv.gz'),
            'original_means_sha256':file_digest(base_reader/'class_means.npz'),
            'counts':dict(zip(GROUPS,map(int,np.concatenate([corecounts,counts])))),
            'selected_events':int(selected.event_id.nunique()),'cohort_memberships':len(selected),'status':'completed',
            'files':{n:file_digest(target/n) for n in ('cohort_means.npz','selected_events.csv.gz')}}
    write_json(target/'manifest.json',result)
    return result


def checkpoint(target,base_reader,fingerprint,channels,contexts=None,source_ratings=None):
    if not (target/'manifest.json').is_file():
        return None
    try:
        item=json.loads((target/'manifest.json').read_text())
    except (OSError,json.JSONDecodeError):
        return None
    if not isinstance(item,dict) or not item.get('fingerprint'):
        return None
    if item['fingerprint']!=fingerprint:
        raise ValueError('Existing context analysis uses different code/labels; choose a new analysis tag')
    if item['original_onsets_sha256']!=file_digest(base_reader/'word_onsets.csv.gz') or item['original_means_sha256']!=file_digest(base_reader/'class_means.npz'):
        raise ValueError('The original compact inputs changed')
    try:
        original_record=json.loads((base_reader/'manifest.json').read_text())
        if (item['task'],item['subject'],item['source_sha256'])!=(original_record['task'],original_record['subject'],original_record['source_sha256']):
            return None
        if item['status']!='completed':
            return None
        if any(file_digest(target/name)!=checksum for name,checksum in item['files'].items()):
            return None
        with np.load(target/'cohort_means.npz',allow_pickle=False) as saved:
            counts=saved['counts'];means=saved['mean_uV']
            if not np.array_equal(saved['groups'],GROUPS) or not np.array_equal(saved['channel_names'],channels):
                return None
            if means.shape!=(len(GROUPS),105,501) or not np.array_equal(saved['times_ms'],original.TIMES_MS):
                return None
            if not np.array_equal(counts,[item['counts'][g] for g in GROUPS]):
                return None
            if not np.isfinite(means[counts>0]).all():
                return None
            if not np.isnan(means[counts==0]).all():
                return None
            with np.load(base_reader/'class_means.npz',allow_pickle=False) as core:
                if not np.array_equal(means[:3],core['mean_uV'],equal_nan=True) or not np.array_equal(counts[:3],core['counts']):
                    return None
        selected=pd.read_csv(target/'selected_events.csv.gz',keep_default_na=False)
        if (len(selected)!=item['cohort_memberships'] or selected.event_id.nunique()!=item['selected_events']
                or selected.duplicated(['event_id','cohort']).any()):
            return None
        if contexts is not None:
            expected=select_events(pd.read_csv(base_reader/'word_onsets.csv.gz',keep_default_na=False),contexts,source_ratings)
            if not expected.astype(str).equals(selected.astype(str)):
                return None
        for group in COHORTS:
            if int((selected.cohort==group).sum())!=item['counts'][group]:
                return None
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
        return None
    return item


def summarize(run,items,channels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roi=[channels.index(c) for c in original.ROI]
    comparisons={
        'pooled_with_strict_neutral':('pooled_negative','pooled_neutral','pooled_positive','neutral_in_neutral_sentence'),
        'within_neutral_sentences':COHORTS[:3],
        'neutral_word_contexts':('neutral_in_negative_sentence','neutral_in_neutral_sentence','neutral_in_positive_sentence'),
        'pooled_with_source_neutral':('pooled_negative','pooled_neutral','pooled_positive',SOURCE_NEUTRAL),
        'neutral_definitions':('neutral_in_neutral_sentence',SOURCE_NEUTRAL)}
    rows=[];coverage=[]
    plotdir=run/'plots';plotdir.mkdir(exist_ok=True)
    for item in items:
        for group,count in item['counts'].items():
            rows.append({'task':item['task'],'subject':item['subject'],'group':group,'accepted_epochs':count})
    for task in sorted({i['task'] for i in items}):
        candidates=[i for i in items if i['task']==task]
        for name,groups in comparisons.items():
            if SOURCE_NEUTRAL in groups and task!='SR':
                continue
            readers=[i for i in candidates if min(i['counts'][g] for g in groups)>=5]
            coverage.append({'task':task,'comparison':name,'matched_readers':len(readers),
                             'subjects':';'.join(i['subject'] for i in readers),'minimum_epochs_each_group':5})
            if len(readers)<2:
                continue
            stack=[]
            for item in readers:
                with np.load(run/'readers'/task/item['subject']/'cohort_means.npz',allow_pickle=False) as saved:
                    stack.append(saved['mean_uV'][[GROUPS.index(g) for g in groups]])
            stack=np.stack(stack)
            atomic_npz(run/f'{task}_{name}.npz',subject_means_uV=stack,mean_uV=stack.mean(axis=0),groups=np.array(groups),
                       subjects=np.array([i['subject'] for i in readers]),times_ms=original.TIMES_MS,channel_names=np.array(channels))
            fig,ax=plt.subplots(figsize=(10,5))
            for gi,group in enumerate(groups):
                values=stack[:,gi,roi,:].mean(axis=1);mean=values.mean(axis=0)
                sem=values.std(axis=0,ddof=1)/np.sqrt(len(values))
                line,=ax.plot(original.TIMES_MS,mean,label=group.replace('_',' '))
                ax.fill_between(original.TIMES_MS,mean-sem,mean+sem,color=line.get_color(),alpha=.12)
            ax.axvline(0,color='black',lw=.7);ax.axhline(0,color='grey',lw=.7)
            ax.set(xlabel='Time from first fixation onset (ms)',ylabel='Amplitude (µV as stored)',
                   title=f'{task}: {name.replace("_"," ")} · {len(readers)} matched readers')
            ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plotdir/f'{task}_{name}.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(run/'cohort_counts.csv',index=False)
    pd.DataFrame(coverage).to_csv(run/'comparison_coverage.csv',index=False)
    (run/'REPORT.md').write_text(
        '# ZuCo sentence-context controls\n\n'
        'Original word labels, accepted event IDs, filtering and rejection are retained. '
        'Pooled means are reused exactly; selected context cohorts were rebuilt from saved onsets and verified source files.\n\n'
        '- `pooled_with_strict_neutral` adds neutral words from neutral sentences; it does not isolate word sentiment from sentence context.\n'
        '- `within_neutral_sentences` compares all word classes within neutral sentences; coverage may be too sparse.\n'
        '- `neutral_word_contexts` compares neutral words across sentence contexts; lexical items are not matched.\n'
        '- SR-only `pooled_with_source_neutral` adds neutral words in reviews originally rated neutral. This is a separate control from strict affect-free agent context. Original source ratings do not establish affect-free sentences.\n'
        '- SR-only `neutral_definitions` compares the two neutral-context definitions using matched readers when coverage permits. The original 400 SR ratings contain 137 neutral reviews; NR/TSR have no original source sentiment rating.\n'
        '- All plotted groups use the same readers, each with at least five epochs in every displayed group. Fewer than two readers produces coverage only.\n'
        '- Sentence contexts are explicit single-agent annotations, not human gold. Uncertain sentence contexts are not used in these new cohorts.\n'
        '- Figures are descriptive FRPs. Fixation overlap, word properties and stimulus/item variability remain uncontrolled.\n'
        '- A physical event may appear in two different control cohorts. Epochs are calculated once per physical event, then reused once per cohort; no event is duplicated within a cohort. Counts must be interpreted per cohort, not summed as independent events.\n'
        '- `readers/` holds compact averages and selected event/cohort memberships only; individual trial EEG is not archived.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--base-run-tag',default='v4_word_erp_compact')
    parser.add_argument('--analysis-tag',default='neutral_context_v1')
    parser.add_argument('--sentence-labels',type=Path,default=original.ROOT/'annotations/sentence_context_labels.csv')
    parser.add_argument('--source-sentence-ratings',type=Path,default=original.ROOT/'annotations/source_sentence_ratings.csv')
    parser.add_argument('--allow-cloud-download',action='store_true')
    parser.add_argument('--tmp-dir',type=Path,default=Path('/content/word-eeg-context-tmp'))
    args=parser.parse_args()
    if any(Path(t).name!=t for t in (args.base_run_tag,args.analysis_tag)):
        parser.error('Tags must be single folder names')
    base=args.thesis_root/'Results/word_sentiment_eeg'/args.base_run_tag
    state=json.loads((base/'manifest.json').read_text())
    provenance=json.loads((base/'provenance.json').read_text())
    audit=json.loads((base/'audit/integrity.json').read_text())
    checksums=json.loads((base/'audit/file_checksums.json').read_text())
    if state['status']!='completed' or state['failures']:
        raise ValueError('The original run must be complete')
    if audit['status']!='passed':
        raise ValueError('The original saved-file audit must pass first')
    context_frame=pd.read_csv(args.sentence_labels,keep_default_na=False)
    if context_frame.sentence_key.duplicated().any() or not context_frame.annotation_status.eq('agent_reviewed_not_human_gold').all():
        raise ValueError('Sentence contexts must be unique and explicitly agent-reviewed')
    contexts={r.sentence_key:r for r in context_frame.itertuples()}
    source_ratings=load_source_ratings(args.source_sentence_ratings) if args.source_sentence_ratings.exists() else {}
    original_labels=pd.read_csv(base/'word_labels.csv',keep_default_na=False)
    if set(contexts)!=set(original_labels.sentence_key):
        raise ValueError('Sentence-context coverage does not match the original whole dataset')
    expected_sr={r.sentence_key for r in original_labels.itertuples() if 'SR' in r.tasks.split(';')}
    if source_ratings and set(source_ratings)!=expected_sr:
        raise ValueError('Original source-rating coverage must match every SR sentence exactly')
    channels=pd.read_csv(base/'zuco_chanlocs.csv').labels.tolist()
    fingerprint=digest({'original_fingerprint':state['fingerprint'],'sentence_labels':file_digest(args.sentence_labels),
                        'source_sentence_ratings':file_digest(args.source_sentence_ratings) if source_ratings else None,
                        'runner':file_digest(Path(__file__)),'erp':file_digest(original.ROOT/'src/word_sentiment_eeg/erp.py')})
    run=base/'analysis'/args.analysis_tag;run.mkdir(parents=True,exist_ok=True)
    lockpath=Path(tempfile.gettempdir())/('zuco-context-'+digest(str(run.resolve()))[:16]+'.lock')
    lock=lockpath.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'manifest.json').exists() and json.loads((run/'manifest.json').read_text())['fingerprint']!=fingerprint:
        raise ValueError('Existing context run has different code or labels; choose a new analysis tag')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(run/'run.log')])
    original.check_output_space(args.thesis_root)
    (run/'sentence_context_labels.csv').write_bytes(args.sentence_labels.read_bytes())
    if source_ratings:
        (run/'source_sentence_ratings.csv').write_bytes(args.source_sentence_ratings.read_bytes())
    else:
        logging.warning('Original source ratings unavailable; source-neutral SR controls will not be produced')
    sources={(r['task'],r['subject']):r for r in provenance['source_inventory']}
    expected=state['files']
    if len({(i['task'],i['subject']) for i in expected})!=len(expected) or set(sources)!={(i['task'],i['subject']) for i in expected}:
        raise ValueError('Original/source reader roster is not unique or complete')
    roots=[args.thesis_root/'Data'/name for name in ('zuco_og_raw','zuco_nr_raw','zuco_tsr_raw')]
    local={p.name:p for root in roots if root.exists() for p in root.rglob('results*_*.mat')}
    done=[];failures=[]
    for item in expected:
        task,subject=item['task'],item['subject'];reader=base/'readers'/task/subject;target=run/'readers'/task/subject
        temporary=False;source=None
        try:
            original.check_output_space(args.thesis_root)
            for name in ('manifest.json','word_onsets.csv.gz','class_means.npz'):
                path=reader/name
                if file_digest(path)!=checksums[str(path.relative_to(base))]:
                    raise ValueError('Original reader files changed since the saved-file audit')
            saved=checkpoint(target,reader,fingerprint,channels,contexts,source_ratings)
            if saved:
                logging.info('KEEP %s/%s context cohorts',task,subject)
            else:
                record=sources[(task,subject)];source=local.get(record['filename'])
                if source is None:
                    if not args.allow_cloud_download:
                        raise FileNotFoundError('Missing original EEG; allow cloud download inside Colab')
                    source=args.tmp_dir/record['filename'];original.cloud_download(record,source);temporary=True
                logging.info('REBUILD %s/%s selected context cohorts from saved onsets',task,subject)
                saved=process_record(source,reader,target,item,contexts,fingerprint,channels,source_ratings=source_ratings)
                if checkpoint(target,reader,fingerprint,channels,contexts,source_ratings) is None:
                    raise ValueError('New context checkpoint failed readback')
            done.append(saved)
        except Exception as error:
            logging.exception('FAILED %s/%s',task,subject)
            failures.append({'task':task,'subject':subject,'reason':str(error)})
        finally:
            if temporary and source is not None:
                source.unlink(missing_ok=True)
        write_json(run/'manifest.json',{'status':'incomplete','fingerprint':fingerprint,'expected_records':len(expected),'files':done,'failures':failures})
        logging.info('Saved %d/%d unique context records',len(done),len(expected))
    summarize(run,done,channels)
    result={'status':'completed' if len(done)==len(expected) and not failures else 'incomplete',
            'fingerprint':fingerprint,'expected_records':len(expected),'files':done,'failures':failures,
            'finished_at_utc':datetime.now(timezone.utc).isoformat(),'original_results_modified':False}
    write_json(run/'manifest.json',result)
    bundle=run/'context_results.tmp.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix not in {'.zip','.log'}:
                archive.write(path,path.relative_to(run))
    os.replace(bundle,run/'context_results.zip')
    logging.info('Context results: %s (%s)',run,result['status'])
    if result['status']!='completed':
        raise SystemExit('Context analysis incomplete; saved reader checkpoints are retained')


if __name__=='__main__':
    main()
