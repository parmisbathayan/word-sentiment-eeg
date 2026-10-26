"""Verify saved terminal accounting, subset handling and real flow rendering."""
import importlib.util
import csv
import gzip
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exclusion_report',ROOT/'scripts/report_exclusion_flow.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def must_fail(fn, text):
    try:
        fn()
    except ValueError as error:
        assert text in str(error),error
    else:
        raise AssertionError('Expected failed validation')


with tempfile.TemporaryDirectory() as tmp:
    run=Path(tmp)
    statuses={'accepted':6,'excluded_text_alignment':1,
              'excluded_uncertain_or_context_disagreement':2,
              'excluded_missing_or_invalid_fixation':3,'excluded_nonunique_match':1,
              'sentence_boundary':2,'nonfinite_epoch':1,'amplitude_over_100_uV':1}
    item={'task':'SR','subject':'ZAB','word_rows':17,'sentences':4,
          'status_counts':statuses,'epochs_per_class':{'positive':1,'negative':2,'neutral':3}}
    state={'status':'completed','files':[item]}
    (run/'manifest.json').write_text(json.dumps(state))
    before=(run/'manifest.json').read_bytes()
    report=mod.build(run,'zuco',make_plot=True)
    assert report['totals']['accepted']==6
    assert report['totals']['label_or_control']==3
    assert report['totals']['fixation_or_onset']==4
    assert report['flow'][-1]['remaining_after']==6
    assert all(v is None for v in report['unknown_upstream'].values())
    assert (run/'analysis/exclusion_flow_v1/exclusion_flow.png').stat().st_size>1000
    assert (run/'manifest.json').read_bytes()==before
    state['files']=[item,item]
    (run/'manifest.json').write_text(json.dumps(state))
    must_fail(lambda:mod.collect(run,'zuco'),'Duplicate')
    state['files']=[dict(item,word_rows=18)]
    (run/'manifest.json').write_text(json.dumps(state))
    must_fail(lambda:mod.collect(run,'zuco'),'reconcile')
    state['files']=[dict(item,epochs_per_class={'positive':2,'negative':2,'neutral':3})]
    (run/'manifest.json').write_text(json.dumps(state))
    must_fail(lambda:mod.collect(run,'zuco'),'disagree')
    must_fail(lambda:mod.terminal_stage('unknown_future_status'),'Unrecognized')
    teco={'status':'completed','readers':[{'subject':'armin','task':'task1',
          'source_word_count':12,'source_sentence_count':3,
          'exclusions':{'accepted':6,'control_excluded':2,'no_fixation':2,
                        'sentence_too_short_for_filter':1,'amplitude_over_100_assumed_uV':1},
          'counts':{'positive':1,'negative':2,'neutral':3,
                    'neutral_in_neutral_sentence':1,'neutral_in_source_neutral_sentence':2}}]}
    (run/'manifest.json').write_text(json.dumps(teco))
    report=mod.build(run,'teco',make_plot=False)
    assert report['totals']['accepted']==6,'Overlapping neutral subsets double-counted'
    assert report['totals']['source_word_observations']==12
    folder=run/'readers/armin'
    folder.mkdir(parents=True)
    source_statuses=['accepted']*6+['control_excluded']*2+['no_fixation']*2+['sentence_too_short_for_filter','amplitude_over_100_assumed_uV']
    fields=['event_id','task','subject','sentence_key','word_index','lexical_class','contextual_class','sentence_context_class','status']
    def save_words(statuses):
        with gzip.open(folder/'word_onsets.csv.gz','wt',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for i,status in enumerate(statuses):
                writer.writerow(dict(event_id=str(i),task='task1',subject='armin',sentence_key='s'+str(i//4),word_index=i%4,
                                     lexical_class='neutral',contextual_class='neutral',sentence_context_class='neutral',status=status))
    save_words(source_statuses)
    mod.build(run,'teco',include_words=True,make_plot=False)
    with (run/'analysis/exclusion_flow_v1/source_and_reader_input_counts.csv').open() as handle:
        words=list(csv.DictReader(handle))
    assert len(words)==3
    assert all(int(r['actual_reader_word_observations'])==12 and int(r['unique_source_word_positions_observed'])==12 for r in words)
    save_words(['accepted']*12)
    must_fail(lambda:mod.build(run,'teco',include_words=True,make_plot=False),'statuses disagree')
print('Exclusion-flow accounting, corruption guards, source preservation and rendering passed.')
