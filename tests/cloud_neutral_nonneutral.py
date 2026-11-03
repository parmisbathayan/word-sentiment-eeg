"""Cloud-only model and cache-only follow-up verification."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import classify_neutral_nonneutral as n
import analyze_trial_archive as a


def main():
    rng=np.random.default_rng(44)
    rows=[]; features=[]
    for reader in range(6):
        for sentence in range(18):
            label=a.CLASSES[sentence%3]
            rows.append(dict(event_id=f'{reader}/{sentence}',subject=f'R{reader}',dataset='zuco',
                             sentence_key=f'S{sentence}',sentence_position=sentence,
                             word_identity=f'W{sentence}',onset_sample_sentence_0based=1000,
                             lexical_class=label,contextual_word_class=label,sentence_context_class=label,
                             grammatical_class='content',annotation_alignment_warning='',
                             amplitude_over_100_uV=False,signal_sha256=f'unique{reader}/{sentence}'))
            features.append(rng.normal(0,.1,48)+(0 if label=='neutral' else 5))
    meta=pd.DataFrame(rows);X=np.asarray(features,dtype=np.float32)
    ix,y=n.population(meta)
    assert len(ix)==108 and (y=='negative').sum()==36 and (y=='positive').sum()==72
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)/'zuco'/'SR'
        scores,skips=n.task_analysis(meta,X,X[:,:12],X[:,:6],out)
        assert not skips and len(scores)==11
        for row in scores:
            assert row['classes']==['neutral','nonneutral']
            if row['model']!='dummy': assert row['balanced_accuracy']>.95
        case=out/'cases'/'sentence__post__logistic'
        pred=pd.read_csv(case/'predictions.csv.gz')
        assert set(pred.truth)=={'neutral','nonneutral'}
        assert 'probability_nonneutral' in pred and pred.event_id.is_unique
        assert set(pred[pred.truth=='nonneutral'].probability_nonneutral>.5)=={True}
        assert (out/'confusions.png').is_file()
        # Completed cases are actually reused, not refit.
        original=a.evaluate
        def no_fit(*args,**kwargs): raise AssertionError('Refit completed case')
        a.evaluate=no_fit
        try: n.task_analysis(meta,X,X[:,:12],X[:,:6],out)
        finally: a.evaluate=original
        # Missing cache must fail before the EEG-reading function can run.
        original=a.reader_features
        def no_source(*args,**kwargs): raise AssertionError('Attempted feature preparation')
        a.reader_features=no_source
        try:
            try:n.cached_record(Path(folder),dict(dataset='zuco',task='SR',subject='R0'),Path(folder),Path(folder))
            except ValueError:pass
            else:raise AssertionError('Missing feature cache accepted')
        finally:a.reader_features=original
    print('Neutral/nonneutral mappings, three holdouts, models, cache-only guard and resume passed')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
