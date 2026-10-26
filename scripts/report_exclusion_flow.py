#!/usr/bin/env python3
"""Reconcile saved terminal statuses without re-extracting EEG or inventing QC."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

CLASSES = ('positive', 'negative', 'neutral')
STAGES = ('label_or_control', 'fixation_or_onset', 'epoch_or_filter_boundary',
          'nonfinite_or_amplitude', 'accepted')
LABEL = {'control_excluded', 'annotation_alignment_warning', 'annotation_not_reviewed',
         'uncertain_or_context_disagreement', 'excluded_text_alignment',
         'excluded_uncertain_or_context_disagreement'}
ONSET = {'invalid_fixation_count', 'no_fixation', 'missing_or_invalid_first_fixation',
         'missing_or_invalid_fixation', 'missing_first_fixation', 'missing_sentence_eeg',
         'missing_reading_raw_data', 'longer_than_sentence', 'flat_segment',
         'nonunique_match', 'unmatched'}
BOUNDARY = {'sentence_boundary', 'sentence_too_short_for_filter'}
ARTIFACT = {'nonfinite_reading_raw_data', 'nonfinite_epoch', 'amplitude_over_100_uV',
            'amplitude_over_100_assumed_uV'}


def terminal_stage(status):
    if status in LABEL:
        return 'label_or_control'
    normalized = status.removeprefix('excluded_')
    if normalized in ONSET:
        return 'fixation_or_onset'
    if normalized in BOUNDARY:
        return 'epoch_or_filter_boundary'
    if normalized in ARTIFACT:
        return 'nonfinite_or_amplitude'
    if status == 'accepted':
        return 'accepted'
    raise ValueError(f'Unrecognized terminal status {status!r}; update mapping explicitly')


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(run, dataset):
    manifest_path = run / 'manifest.json'
    state = json.loads(manifest_path.read_text())
    items = state.get('files' if dataset == 'zuco' else 'readers', [])
    if not items:
        raise ValueError('No reader manifests in this saved run')
    roster = set()
    statuses = []
    readers = []
    classes = []
    for item in items:
        task = item.get('task', 'task1')
        reader = item['subject']
        if (task, reader) in roster:
            raise ValueError(f'Duplicate reader/task {task}/{reader}')
        roster.add((task, reader))
        source_n = int(item['word_rows' if dataset == 'zuco' else 'source_word_count'])
        counts = item['status_counts' if dataset == 'zuco' else 'exclusions']
        by_stage = Counter()
        for status, count in counts.items():
            count = int(count)
            if count < 0:
                raise ValueError('Negative status count')
            stage = terminal_stage(status)
            by_stage[stage] += count
            statuses.append(dict(task=task, subject=reader, status=status, stage=stage, count=count))
        if sum(by_stage.values()) != source_n:
            raise ValueError(f'{task}/{reader}: terminal statuses do not reconcile with source count')
        accepted = item['epochs_per_class' if dataset == 'zuco' else 'counts']
        if sum(int(accepted[c]) for c in CLASSES) != by_stage['accepted']:
            raise ValueError(f'{task}/{reader}: class counts disagree with accepted unique events')
        row = dict(task=task, subject=reader, source_word_observations=source_n,
                   source_sentence_presentations=int(item['sentences' if dataset == 'zuco' else 'source_sentence_count']))
        row.update({s:by_stage[s] for s in STAGES})
        readers.append(row)
        for cls in CLASSES:
            classes.append(dict(task=task, subject=reader, class_label=cls, accepted_trials=int(accepted[cls])))
    return state, readers, statuses, classes


def word_tables(run, dataset, reader_rows, status_rows):
    """Optional lightweight saved-table audit; no native EEG is opened."""
    records = []
    for reader in reader_rows:
        task, subject = reader['task'], reader['subject']
        path = run / 'readers' / task / subject / 'word_onsets.csv.gz' if dataset == 'zuco' else run / 'readers' / subject / 'word_onsets.csv.gz'
        if not path.exists():
            raise ValueError(f'Word table unavailable at {path}; omit --include-word-tables for manifest-only reporting')
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != reader['source_word_observations']:
            raise ValueError(f'{task}/{subject}: word table row count mismatch')
        event_ids = [r['event_id'] for r in rows]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f'{task}/{subject}: duplicate event IDs')
        expected = Counter({r['status']:r['count'] for r in status_rows
                            if r['task']==task and r['subject']==subject})
        if Counter(r['status'] for r in rows) != expected:
            raise ValueError(f'{task}/{subject}: word table statuses disagree with manifest')
        if any(r.get('task')!=task or r.get('subject')!=subject for r in rows):
            raise ValueError(f'{task}/{subject}: word table reader/task mismatch')
        for r in rows:
            r['_task'], r['_reader'] = task, subject
        records.extend(rows)
    summaries = []
    for layer in ('lexical_class', 'contextual_class', 'sentence_context_class'):
        if not all(layer in r for r in records):
            continue
        keys = sorted({(r['_task'], r[layer]) for r in records})
        for task, label in keys:
            selected = [r for r in records if r['_task'] == task and r[layer] == label]
            unique = {(r['sentence_key'], r['word_index']) for r in selected}
            sentences = {r['sentence_key'] for r in selected}
            summaries.append(dict(task=task, label_layer=layer, class_label=label,
                                  unique_source_word_positions_observed=len(unique),
                                  unique_sentences_observed=len(sentences),
                                  actual_reader_word_observations=len(selected),
                                  readers_with_observations=len({r['_reader'] for r in selected}),
                                  accepted_trials=sum(r['status']=='accepted' for r in selected)))
    return summaries


def build(run, dataset, include_words=False, make_plot=True):
    run = Path(run)
    state, readers, statuses, classes = collect(run, dataset)
    out = run / 'analysis' / 'exclusion_flow_v1'
    out.mkdir(parents=True, exist_ok=True)
    totals = {key:sum(r[key] for r in readers) for key in ['source_word_observations', 'source_sentence_presentations', *STAGES]}
    remaining = totals['source_word_observations']
    flow = []
    for stage in STAGES[:-1]:
        excluded = totals[stage]
        flow.append(dict(stage=stage, candidates_before=remaining, excluded=excluded, remaining_after=remaining-excluded))
        remaining -= excluded
    if remaining != totals['accepted']:
        raise ValueError('Aggregate flow does not reconcile')
    protocol = run / 'protocol.json'
    if not protocol.exists():
        protocol = run / 'provenance.json'
    report = {'dataset':dataset, 'run_status':state.get('status', 'unknown'),
              'reader_task_recordings':len(readers), 'distinct_readers':len({r['subject'] for r in readers}),
              'totals':totals, 'flow':flow,
              'flow_semantics':'Terminal-status accounting; conceptual grouped attrition, not evidence that each remaining event passed every later check.',
              'unknown_upstream':{'blink_predecessor_removals':None, 'outside_word_removals':None, 'under_100ms_removals':None},
              'unknown_reason':'Not measured by this saved word-level extraction; unavailable is not zero.',
              'main_classes_balanced_by_downsampling':False,
              'source_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
              'saved_protocol':json.loads(protocol.read_text()) if protocol.exists() else None}
    write_csv(out/'per_reader_flow.csv', readers, list(readers[0]))
    write_csv(out/'terminal_status_counts.csv', statuses, ['task','subject','status','stage','count'])
    write_csv(out/'per_reader_class_counts.csv', classes, ['task','subject','class_label','accepted_trials'])
    write_csv(out/'aggregate_flow.csv', flow, ['stage','candidates_before','excluded','remaining_after'])
    if include_words:
        words = word_tables(run, dataset, readers, statuses)
        if words:
            write_csv(out/'source_and_reader_input_counts.csv', words, list(words[0]))
    (out/'flow.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Word-FRP selection and exclusion report', '',
             f"Dataset: {dataset}; saved run status: {report['run_status']}.",
             f"Recordings: {len(readers)}; distinct readers: {report['distinct_readers']}.",
             f"Input: {totals['source_word_observations']:,} actual reader-word observations across {totals['source_sentence_presentations']:,} reader-sentence presentations.", '',
             '| Conceptual step | Candidates before | Excluded terminal statuses | Remaining |',
             '| --- | ---: | ---: | ---: |']
    lines += [f"| {r['stage']} | {r['candidates_before']:,} | {r['excluded']:,} | {r['remaining_after']:,} |" for r in flow]
    lines += ['', f"Accepted unique word events: **{remaining:,}**.", '',
              'These are grouped terminal-status counts. Selection occurs before later quality checks, so an event excluded for its label was not necessarily tested for a blink, duration, onset or amplitude failure. The table does not claim independently measured pass counts at every intermediate stage.', '',
              'Outside-word events are absent from this word-level starting population. Upstream blink-predecessor, outside-word and <100-ms exclusion counts are **unavailable, not zero**. No additional removals for those criteria are invented here.', '',
              'Onset-recovery failures mean missing/invalid first stored fixations or failed/ambiguous waveform placement in sentence EEG. They are not the covariate-balancing trial matching used by Pfeiffer et al. (2020).', '',
              'First stored fixation only; later fixations do not replace an invalid first slot. Main class counts are not balanced by downsampling. TeCo neutral-control subsets overlap pooled neutral and must not be added as distinct trials.', '',
              'See saved protocol in flow.json for this run’s preprocessing. Existing results retain their original filter; this reporting script does not change EEG or imply a 40-Hz reconstruction.', '',
              'Files: aggregate_flow.csv, per_reader_flow.csv, terminal_status_counts.csv, per_reader_class_counts.csv, flow.json; optional source_and_reader_input_counts.csv uses actual saved reader-token availability, not a nominal multiplication by reader count.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.axis('off')
        boxes = [(f"Stored reader-word observations\nN = {totals['source_word_observations']:,}", 0.92)]
        boxes += [(f"After {r['stage'].replace('_',' ')}\nRemaining = {r['remaining_after']:,}; excluded = {r['excluded']:,}",0.75-i*0.17) for i,r in enumerate(flow)]
        for i,(label,y) in enumerate(boxes):
            ax.text(.5,y,label,ha='center',va='center',transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.65',facecolor='#e8f1f7',edgecolor='#36576d'),fontsize=12)
            if i:
                ax.annotate('',xy=(.5,y+.065),xytext=(.5,boxes[i-1][1]-.065),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
        ax.text(.5,.04,'Grouped terminal-status accounting. Upstream blink / outside-word / <100-ms counts unavailable.\nNo main-class downsampling. See REPORT.md for denominator and ordering limitations.',ha='center',va='center',transform=ax.transAxes,fontsize=9)
        ax.set_title(f'{dataset.upper()} saved word-FRP selection',fontsize=16)
        fig.savefig(out/'exclusion_flow.png',dpi=160,bbox_inches='tight')
        plt.close(fig)
    print(f'Validated {remaining:,} accepted events; report: {out}',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--dataset',choices=('zuco','teco'),required=True)
    parser.add_argument('--include-word-tables',action='store_true')
    parser.add_argument('--no-plot',action='store_true')
    args = parser.parse_args()
    build(args.run_dir,args.dataset,args.include_word_tables,not args.no_plot)


if __name__ == '__main__':
    main()
