"""Synthetic agent intersection/rematching/handoff tests; cloud only."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import zuco_agent_paper_subset as s
from word_sentiment_eeg.common import write_json,file_digest,atomic_npz
if sys.platform=='darwin':raise SystemExit('Cloud only')
rng=np.random.default_rng(421)
def fake_models(counts,sums,squares,**kw):
    shape=sums.shape[2:];return dict(p=np.full(shape,.5),effect=np.zeros(shape),converged=np.ones(shape))
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);run=tmp/'input';out=tmp/'output';work=tmp/'runtime';subjects=['SYNTH_A','SYNTH_B','SYNTH_C'];marks={}
    n=24;y=np.arange(n)%2
    labels=pd.DataFrame(dict(sentence_key=[f's{i//4}' for i in range(n)],word_index=np.arange(n)%4,
        word=[f'word{i}' for i in range(n)],tasks='SR',lexical_class=np.where(y,'positive','negative'),
        contextual_class=np.where(y,'positive','negative'),alignment_warning='',annotation_status='agent_reviewed_not_human_gold',annotation_flags=''))
    labels['agent_eligible']=True
    for subject in subjects:
        folder=run/'readers'/subject;folder.mkdir(parents=True)
        meta=pd.DataFrame(dict(event_id=[subject+str(i) for i in range(n)],subject=subject,sentence_key=labels.sentence_key,
            word_index=labels.word_index,word=labels.word,word_length=6,word_frequency=2,previous_gap_ms=200,next_gap_ms=200,duration_ms=200,
            sentiment=np.where(y,'negative','positive'),selection_status='accepted',block_id=1)) # deliberately opposite SST labels
        joined,kept=s.select(meta,labels)
        assert len(kept)==n and kept.sentiment.tolist()==labels.lexical_class.tolist()
        assert (kept.sentiment!=kept.parent_sentiment).all()
        changed=labels.copy();changed.loc[0,'word']='DIFFERENT';a,b=s.select(meta,changed)
        assert a.loc[0,'subset_status']=='word_text_mismatch' and 0 not in b.parent_row.tolist()
        changed=labels.copy();changed.loc[0,'agent_eligible']=False;a,b=s.select(meta,changed)
        assert a.loc[0,'subset_status']=='agent_neutral_disagreement_or_warning'
        signal=rng.normal(0,.2,(2,n,105,201)).astype(np.float32)
        atomic_npz(folder/'prepared_trials.npz',signal=signal,labels=1-y,event_ids=meta.event_id.to_numpy(str),times_ms=np.arange(-75,126)*8)
        meta.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False)
        write_json(folder/'prepared.json',dict(status='prepared',checksums={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}))
        marks[subject]=file_digest(folder/'prepared.json')
    write_json(run/'protocol.json',dict(synthetic=True));write_json(run/'cpu_preparation_manifest.json',dict(status='completed',subjects=subjects,
        protocol_sha256=file_digest(run/'protocol.json'),reader_checkpoint_sha256=marks));write_json(run/'manifest.json',dict(status='completed'))
    before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    args=['subset','--thesis-root',str(tmp/'thesis'),'--run-dir',str(run),'--output',str(out),'--work-root',str(work),
          '--cpu','--skip-decoding','--tct','4','--cluster','4']
    with patch.object(sys,'argv',args),patch.object(s,'labels_table',return_value=labels),patch.object(s,'mixed_models',fake_models):s.main()
    result=json.loads((out/'analysis/summary.json').read_text())
    assert result['accepted_negative']==36 and result['accepted_positive']==36 and result['pairs']==36
    assert len(list((out/'figures').glob('*.png')))==3
    assert len(json.loads((out/'analysis/window_tests.json').read_text()))==4
    assert not list(out.rglob('prepared_trials.npz')) and not list(out.rglob('subset_signals.dat'))
    # Completed resume does not load EEG or rewrite outputs, even without GPU.
    saved={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    with patch.object(sys,'argv',args),patch.object(s,'labels_table',return_value=labels),patch.object(s,'stage',side_effect=AssertionError('EEG reloaded')):s.main()
    assert saved=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    for p,data in before.items():assert p.read_bytes()==data
    # Protocol changes cannot silently reuse old inference.
    args2=args[:-1]+['5']
    with patch.object(sys,'argv',args2),patch.object(s,'labels_table',return_value=labels):
        try:s.main()
        except ValueError as error:assert 'immutable' in str(error)
        else:raise AssertionError('Changed protocol mixed')
print('PASS exact source join, relabeling, rematching, raw/clean statistics, actual compact permutations, figures, immutable resume, source preservation')
