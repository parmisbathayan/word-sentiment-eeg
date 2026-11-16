#!/usr/bin/env python3
"""Small readable word lists from metadata only; no EEG, features or model fits."""
import argparse,hashlib,json,sys,unicodedata
from pathlib import Path
import pandas as pd


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def norm(word):
    word=unicodedata.normalize('NFKC',str(word)).casefold().strip()
    while word and unicodedata.category(word[0]).startswith('P'):word=word[1:]
    while word and unicodedata.category(word[-1]).startswith('P'):word=word[:-1]
    return word


def occurrence_table(events,source,annotations):
    keys=['sentence_key','word_index','word']
    table=events.groupby(keys+['selected_class'],sort=True,dropna=False).agg(
        accepted_fixation_trials=('event_id','size'),readers=('subject','nunique'),
        reader_ids=('subject',lambda x:';'.join(sorted(set(x))))).reset_index()
    a=annotations[['sentence_key','word_index','word','sentence','lexical_class','contextual_class','annotation_note','annotation_flags','alignment_warning']].rename(columns={'word':'agent_source_word'})
    if a.duplicated(['sentence_key','word_index']).any():raise ValueError('Ambiguous source annotations')
    table=table.merge(a,on=['sentence_key','word_index'],how='left',validate='many_to_one')
    matched=table.agent_source_word.notna()&table.word.map(norm).eq(table.agent_source_word.fillna('').map(norm))
    for col in ['sentence','lexical_class','contextual_class','annotation_note','annotation_flags','alignment_warning']:
        table.loc[~matched,col]='unavailable_or_text_mismatch'
    table['annotation_join_ok']=matched;table['experiment_word_pool']=source;table['word_identity']=table.word.map(norm)
    return table.fillna('')


def build(parent,subset,annotations_path,fixed_cohort_path=None):
    subset_complete=read(subset/'complete.json');parent_complete=read(parent/'cpu_preparation_manifest.json')
    if subset_complete.get('status')!='completed' or parent_complete.get('status')!='completed':raise ValueError('Saved analyses must be complete')
    annsha=sha(annotations_path);saved_label_sha=read(subset/'protocol.json')['code']['annotations/word_labels.csv']
    if annsha!=saved_label_sha:raise ValueError('Annotation file is not the one used in the saved agent experiment')
    annotations=pd.read_csv(annotations_path,keep_default_na=False);annotations['word_index']=pd.to_numeric(annotations.word_index).astype(int)
    sr=annotations[annotations.tasks.astype(str).str.split(';').map(lambda x:'SR' in x)].copy()
    sources={str(annotations_path):annsha,str(subset/'complete.json'):sha(subset/'complete.json')};tables=[]
    for i,subject in enumerate(parent_complete['subjects'],1):
        folder=parent/'readers'/subject;path=folder/'accepted_trial_metadata.csv.gz';state=read(folder/'prepared.json')
        if sha(folder/'prepared.json')!=parent_complete['reader_checkpoint_sha256'][subject] or sha(path)!=state['checksums'][path.name]:raise ValueError('Parent metadata changed')
        print(f'READ METADATA {i}/{len(parent_complete["subjects"])} {subject}',flush=True)
        frame=pd.read_csv(path,keep_default_na=False,dtype={'event_id':str});frame['subject']=subject;frame['word_index']=pd.to_numeric(frame.word_index).astype(int)
        sources[str(path)]=sha(path);tables.append(frame)
    sst=pd.concat(tables,ignore_index=True)
    if sst.event_id.duplicated().any():raise ValueError('Duplicate SST events')
    audit_path=subset/'selection/trial_selection.csv.gz'
    if sha(audit_path)!=subset_complete['checksums']['selection/trial_selection.csv.gz']:raise ValueError('Agent selection audit changed')
    audit=pd.read_csv(audit_path,keep_default_na=False,dtype={'event_id':str});audit['word_index']=pd.to_numeric(audit.word_index).astype(int)
    if audit.event_id.duplicated().any() or set(audit.event_id)!=set(sst.event_id):raise ValueError('SST and subset parent event pools differ')
    joined=sst[['event_id','sentiment']].merge(audit[['event_id','parent_sentiment']],on='event_id',validate='one_to_one')
    if not joined.sentiment.eq(joined.parent_sentiment).all():raise ValueError('Parent SST labels disagree')
    sst['selected_class']=sst.sentiment
    agent=audit[audit.subset_status.eq('retained')].copy();agent['selected_class']=agent.lexical_class
    if not agent.selected_class.isin(['positive','negative']).all() or not agent.selected_class.eq(agent.contextual_class).all():raise ValueError('Retained agent rule mismatch')
    files={
        'all_agent_annotations.csv':annotations,
        'sst_paper_used_words.csv':occurrence_table(sst,'SST paper pool',sr),
        'agent_paper_used_words.csv':occurrence_table(agent,'agent paper pool',sr)}
    keys=['sentence_key','word_index','word']
    comparison=audit.groupby(keys,sort=True).agg(sst_class=('parent_sentiment','first'),
        agent_lexical_class=('lexical_class','first'),agent_contextual_class=('contextual_class','first'),
        sst_fixation_trials=('event_id','size'),sst_readers=('subject','nunique'),
        agent_retained_fixations=('subset_status',lambda x:int(x.eq('retained').sum())),
        outcomes=('subset_status',lambda x:json.dumps(x.value_counts().to_dict(),sort_keys=True))).reset_index()
    sentence=sr[['sentence_key','word_index','sentence','annotation_note']]
    comparison=comparison.merge(sentence,on=['sentence_key','word_index'],how='left',validate='many_to_one').fillna('')
    comparison['used_in_agent_paper_pool']=comparison.agent_retained_fixations>0;files['sst_vs_agent.csv']=comparison
    pools=[('SST paper pool',sst),('agent paper pool',agent)]
    if fixed_cohort_path is not None and fixed_cohort_path.exists():
        cohort=read(fixed_cohort_path);fixed=agent[agent.subject.isin(cohort['included_readers'])].copy()
        files['agent_fixed_sweep_used_words.csv']=occurrence_table(fixed,'agent fixed-cohort sweep pool',sr);pools.append(('agent fixed-cohort sweep pool',fixed));sources[str(fixed_cohort_path)]=sha(fixed_cohort_path)
    summaries=[]
    for source,events in pools:
        events=events.copy();events['word_identity']=events.word.map(norm);events['occurrence_id']=events.sentence_key.astype(str)+'/'+events.word_index.astype(str)
        summary=events.groupby(['word_identity','selected_class'],sort=True).agg(
            spellings=('word',lambda x:';'.join(sorted(set(x)))),source_word_occurrences=('occurrence_id','nunique'),
            accepted_fixation_trials=('event_id','size'),readers=('subject','nunique')).reset_index();summary['pool']=source;summaries.append(summary)
    files['word_types.csv']=pd.concat(summaries,ignore_index=True)
    mapping=[
        ['SST reconstruction, recovered follow-ups, paper-classification notebook','sst_paper_used_words.csv','Full accepted SST parent pool; grouping may change actual fit membership'],
        ['Agent subset statistics/classification, original63.9%case and original-cohort sweep','agent_paper_used_words.csv','Full retained agent matched pool; per-size leftovers can be discarded'],
        ['First fixed≥100/class averaging sweep','agent_fixed_sweep_used_words.csv' if 'agent_fixed_sweep_used_words.csv' in files else 'not available','Reader-filtered pool; not the exact members of each individual regrouping'],
        ['Earlier first-fixation SR/NR/TSR/TeCo experiments','not exported here','Different EEG archive and policy-specific selections; all_agent_annotations.csv is a text-label reference, not their accepted trial roster']]
    files['experiment_index.csv']=pd.DataFrame(mapping,columns=['experiments','word_list','scope'])
    counts=pd.DataFrame([dict(pool=source,accepted_fixation_trials=len(e),readers=e.subject.nunique(),
        source_word_occurrences=len(e.drop_duplicates(['sentence_key','word_index'])),
        word_identities=e.word.map(norm).nunique(),positive=int(e.selected_class.eq('positive').sum()),negative=int(e.selected_class.eq('negative').sum())) for source,e in pools])
    files['counts.csv']=counts
    return files,sources


def publish(out,files,sources,limit=20_000_000):
    payloads={name:frame.to_csv(index=False).encode('utf-8-sig') for name,frame in files.items()}
    notes='''WORD LABEL OVERVIEW\n\nStart with sst_vs_agent.csv to compare the two selections. Filter by word, sentence, selected class or agent_retained_fixations.\n\nall_agent_annotations.csv: every original agent-reviewed source word occurrence, including neutral, disagreement and all listed tasks. This is not an accepted EEG roster.\nsst_paper_used_words.csv: source word occurrences contributing accepted SST-labelled paper EEG; only positive/negative, because neutral was excluded.\nagent_paper_used_words.csv: occurrences contributing retained agent-labelled matched EEG; target is lexical_class, NOT parent_sentiment.\nagent_fixed_sweep_used_words.csv: same agent pool restricted to the first sweep's included readers, when its cohort file is available.\nword_types.csv: compact normalized word spellings; different class labels remain separate rows.\nexperiment_index.csv: which experiments use which pools.\ncounts.csv: source occurrences, word identities and reader-summed accepted fixation instances.\n\nOne source occurrence is one word location in one sentence. One fixation trial is one reader's recorded fixation on a word; rereadings can add trials. Readers are already aggregated in accepted_fixation_trials: do not multiply again. These files describe selected input pools, not the exact train/test membership of each fold/averaging repeat. Remainder dropping and train-window purging can change those memberships.\n\nSST and agent tables describe saved ZuCo SR paper-style analyses only. They are not a list of every SST dictionary entry or an independent full agent EEG extraction. Missing parent EEG cannot be recovered by changing labels. The agent annotations are not human-consensus labels.\n\nAll exports read metadata only; no EEG signals, feature preparation, reclassification or relabeling. Excel-readable UTF-8 CSV files.\n'''
    payloads['READ_ME.txt']=notes.encode()
    identity=hashlib.sha256(json.dumps(dict(sources=sources,script=sha(Path(__file__))),sort_keys=True).encode()).hexdigest();out.mkdir(parents=True,exist_ok=True)
    marker=out/'manifest.json'
    if marker.exists():
        saved=read(marker)
        if saved['identity']!=identity:raise ValueError('Export inputs changed; do not mix word-list versions')
        for name,digest in saved['checksums'].items():
            if sha(out/name)!=digest:raise ValueError('Saved word list changed: '+name)
        print('KEEP VERIFIED WORD LISTS:',out,flush=True);return
    existing=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    additional=sum(max(0,len(data)-(out/name).stat().st_size) if (out/name).exists() else len(data) for name,data in payloads.items())
    if existing+additional+10000>limit:raise OSError('Word-list20MB cap would be exceeded; no export started')
    checks={}
    for name,data in payloads.items():
        path=out/name
        if path.exists() and not data.startswith(path.read_bytes()):raise ValueError('Conflicting word-list file: '+name)
        if not path.exists() or path.stat().st_size!=len(data):
            with path.open('wb') as stream:stream.write(data);stream.flush()
        checks[name]=sha(path)
        if checks[name]!=hashlib.sha256(data).hexdigest():raise OSError('Export readback failed')
    marker.write_text(json.dumps(dict(status='completed',identity=identity,sources=sources,checksums=checks),indent=2)+'\n')
    print('WORD LISTS SAVED:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab only; no local dataset copies')
    base=args.thesis_root/'Results/word_sentiment_eeg';files,sources=build(base/'zuco_paper_figures_recovered',base/'zuco_agent_paper_subset',args.annotations,base/'zuco_averaging_sweep/cohort.json')
    publish(base/'word_label_overview',files,sources)
    print(files['counts.csv'].to_string(index=False));print('Open sst_vs_agent.csv for side-by-side labels; all_agent_annotations.csv for the full source review.',flush=True)

if __name__=='__main__':main()
