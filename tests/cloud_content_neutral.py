"""Grammar-only overlay, reconciled counts and cached-feature classifier path."""
from pathlib import Path
import sys
import tempfile
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import classify_content_neutral_nonneutral as c


def main():
    rng=np.random.default_rng(7);rows=[];X=[]
    for reader in range(6):
        for sentence in range(30):
            label=c.a.CLASSES[sentence%3]
            rows.append(dict(event_id=f'{reader}/{sentence}',subject=f'R{reader}',dataset='zuco',sentence_key=f'S{sentence}',
                sentence_position=sentence,word_index='0',word=f'word{sentence}',word_identity=f'W{sentence}',onset_sample_sentence_0based=1000,
                lexical_class=label,contextual_word_class=label,sentence_context_class='positive',grammatical_class='unknown',
                annotation_alignment_warning='',amplitude_over_100_uV=False,signal_sha256=f'unique{reader}/{sentence}'))
            X.append(rng.normal(0,.1,48)+(0 if label=='neutral' else 5))
    meta=pd.DataFrame(rows);X=np.asarray(X,dtype=np.float32)
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory);keys=['sentence_key','word_index','word']
        frame=meta[keys].drop_duplicates().copy();frame['tasks']='SR;NR';frame['lexical_class']='incorrect-unused-label'
        frame['grammatical_class']=['content']*18+['function']*6+['uncertain']*6
        path=folder/'grammar.csv';frame.to_csv(path,index=False)
        annotated=c.grammar_overlay(meta,path,'zuco','SR')
        np.testing.assert_array_equal(annotated.lexical_class,meta.lexical_class)
        keep,counts=c.content_selection(annotated)
        assert counts==dict(original_eligible_trials=180,function_excluded=36,uncertain_or_missing_excluded=36,
                           content_retained=108,neutral_retained=36,nonneutral_retained=72,retained_source_word_occurrences=18)
        assert len(keep)==108 and annotated.iloc[keep].grammatical_class.eq('content').all()
        # All three sentence sentiments are NOT required to agree: neutral
        # content words survive even though every sentence label here is positive.
        scores,skips=c.n.task_analysis(annotated.iloc[keep].reset_index(drop=True),X[keep],X[keep,:12],X[keep,:6],folder/'results')
        assert len(scores)==11 and not skips
        assert all(row['class_counts']=={'neutral':36,'nonneutral':72} for row in scores)
        # A missing grammar identity stays unknown, never silently content.
        frame=frame.iloc[1:];frame.to_csv(path,index=False)
        missing=c.grammar_overlay(meta,path,'zuco','SR')
        assert missing.loc[missing.sentence_key.eq('S0'),'grammatical_class'].eq('unknown').all()
        _,changed=c.content_selection(missing);assert changed['uncertain_or_missing_excluded']==42
        # Same exact-identity join works for the private Persian grammar route.
        meta['sentence_id_teco']=meta.sentence_position.astype(str)
        frame['sentence_id_teco']=frame.sentence_key.str[1:];frame.to_csv(path,index=False)
        persian=c.grammar_overlay(meta,path,'teco','task1')
        assert persian.event_id.tolist()==meta.event_id.tolist()
        pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
        try:c.grammar_overlay(meta,path,'teco','task1')
        except ValueError:pass
        else:raise AssertionError('Nonunique grammar join accepted')
    print('Content-only follow-up passed: grammar-only join, exclusions/counts, unchanged labels, all holdouts and models')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
