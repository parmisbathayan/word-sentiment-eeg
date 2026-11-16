from pathlib import Path
import sys,tempfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import word_label_overview as w


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(__import__('json').dumps(value))

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);parent=tmp/'parent';subset=tmp/'subset';(subset/'selection').mkdir(parents=True)
        annotation=pd.DataFrame(dict(sentence_key=['s1','s2','s3','s4'],word_index=[0]*4,word=['tokenA','convincing','bad','the'],sentence=['sentence1','never convincing','bad result','the subject'],
            lexical_class=['negative','positive','negative','neutral'],contextual_class=['negative','negative','negative','neutral'],annotation_note=['test label differs fromSST','negation','plain','grammar'],annotation_flags=['']*4,alignment_warning=['']*4,tasks=['SR','SR','SR','NR']))
        ann=tmp/'annotations.csv';annotation.to_csv(ann,index=False)
        audit=[];checkpoints={}
        for subject,ids in [('R0',['a','b','c']),('R1',['d','e','f'])]:
            frame=pd.DataFrame(dict(event_id=ids,sentence_key=['s1','s2','s3'],word_index=[0]*3,word=['tokenA','convincing','bad'],sentiment=['positive','positive','negative']))
            path=parent/'readers'/subject/'accepted_trial_metadata.csv.gz';path.parent.mkdir(parents=True);frame.to_csv(path,index=False)
            save(path.parent/'prepared.json',dict(checksums={path.name:w.sha(path)}));checkpoints[subject]=w.sha(path.parent/'prepared.json')
            g=frame.copy();g['subject']=subject;g['parent_sentiment']=g.sentiment;g['lexical_class']=['negative','positive','negative'];g['contextual_class']=['negative','negative','negative']
            g['subset_status']=['retained','agent_neutral_disagreement_or_warning','retained'];audit.append(g)
        pd.concat(audit).to_csv(subset/'selection/trial_selection.csv.gz',index=False)
        save(parent/'cpu_preparation_manifest.json',dict(status='completed',subjects=['R0','R1'],reader_checkpoint_sha256=checkpoints))
        save(subset/'complete.json',dict(status='completed',checksums={'selection/trial_selection.csv.gz':w.sha(subset/'selection/trial_selection.csv.gz')}))
        save(subset/'protocol.json',dict(code={'annotations/word_labels.csv':w.sha(ann)}))
        cohort=tmp/'cohort.json';save(cohort,dict(included_readers=['R0']))
        files,sources=w.build(parent,subset,ann,cohort)
        assert len(files['all_agent_annotations.csv'])==4
        sst=files['sst_paper_used_words.csv'];agent=files['agent_paper_used_words.csv'];comparison=files['sst_vs_agent.csv']
        assert len(sst)==3 and len(agent)==2
        assert sst.accepted_fixation_trials.sum()==6 and agent.accepted_fixation_trials.sum()==4
        assert agent[agent.word.eq('tokenA')].selected_class.tolist()==['negative'] #not parentSSTpositive
        assert comparison[comparison.word.eq('convincing')].agent_retained_fixations.tolist()==[0]
        assert files['agent_fixed_sweep_used_words.csv'].accepted_fixation_trials.sum()==2
        assert agent.readers.eq(2).all() and files['word_types.csv'].readers.max()==2
        out=tmp/'out';w.publish(out,files,sources)
        before={str(p):p.stat().st_mtime_ns for p in out.iterdir()};w.publish(out,files,sources);assert before=={str(p):p.stat().st_mtime_ns for p in out.iterdir()}
        assert pd.read_csv(out/'sst_vs_agent.csv').word.tolist()==comparison.word.tolist()
        try:w.publish(tmp/'tiny',files,sources,limit=1)
        except OSError:pass
        else:raise AssertionError('Budget ignored')
    print('PASS wordoccurrence/type distinction, reader/fixation counts, SST/agent label differences, fixedcohort mapping, export/reuse and budget')
if __name__=='__main__':main()
