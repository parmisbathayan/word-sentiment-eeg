"""Fit the fixed battery on synthetic, private-free archive data in the cloud."""
import json
import os,shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import analyze_trial_archive as a
from word_sentiment_eeg.common import atomic_npz,digest,file_digest,write_json


def fixture(archive):
    rng=np.random.default_rng(1956);records=[];protocol={'output_sample_rate_hz':125,'lowpass_hz':40}
    write_json(archive/'protocol.json',protocol)
    times=np.arange(-25,101)*8;channels=[f'C{i}' for i in range(6)]
    for dataset,tasks in [('zuco',['SR','NR','TSR']),('teco',['task1'])]:
        pd.DataFrame({'channel':channels,'region':a.REGIONS}).to_csv(archive/f'{dataset}_channel_regions.csv',index=False)
        for task in tasks:
            for ri in range(12):
                reader=f'R{ri:02}';folder=archive/dataset/'readers'/task/reader;folder.mkdir(parents=True)
                rows=[];signals=[]
                for i in range(18):
                    cls=a.CLASSES[i%3];event=f'{dataset}_{task}_{reader}_{i}'
                    wave=np.sin(np.linspace(0,3*np.pi,126));wave[times<0]=0
                    spatial=np.linspace(-1,1,6)
                    signal=(rng.normal(0,.15,(6,126))+(i%3-1)*spatial[:,None]*wave*4).astype(np.float32)
                    signal-=signal.mean(axis=0,keepdims=True);signal-=signal[:,:25].mean(axis=1,keepdims=True)
                    signals.append(signal)
                    rows.append(dict(event_id=event,subject=reader,task=task,dataset=dataset,sentence_key=f's{i}',sentence_id_teco=i,
                                     sentence_position=i,onset_sample_sentence_0based=1500,word_index=0,word=f'word{i%6}',word_identity=f'word{i%6}',
                                     word_length=5,relative_word_position=0.,sentence_length=1,first_fixation_duration_ms=150,next_word_first_fixation_ms='',
                                     is_control=False,lexical_class=cls,contextual_word_class=cls,sentence_context_class=cls,
                                     grammatical_class='content',annotation_alignment_warning='',amplitude_over_100_uV=False,
                                     status='archived',epoch_file='epochs.npz',epoch_row=i))
                pd.DataFrame(rows).to_csv(folder/'events.csv.gz',index=False,compression='gzip')
                atomic_npz(folder/'epochs.npz',eeg_uV=np.stack(signals),baseline_uV=np.zeros((18,6),np.float32),
                           event_id=np.array([r['event_id'] for r in rows]),times_ms=times,channel_names=np.array(channels))
                cp={'status':'completed','epochs':18,'events_sha256':file_digest(folder/'events.csv.gz'),
                    'identity':{'protocol_sha256':digest(protocol)},'blocks':[{'epoch_files':['epochs.npz'],'checksums':{'epochs.npz':file_digest(folder/'epochs.npz')}}]}
                write_json(folder/'checkpoint.json',cp)
                records.append(dict(dataset=dataset,task=task,subject=reader,epochs=18))
    write_json(archive/'manifest.json',dict(status='completed',completed_reader_recordings=48,expected_reader_recordings=48,readers=records))
    return records


def main():
    # Actual scaler/imputer train-only behavior: held-out large values must not
    # change fitted training parameters.
    m=a.model_for('logistic',2);X=np.array([[0.],[2.],[4.],[6.]])
    m.fit(X,np.array(['negative','negative','positive','positive']))
    assert float(m.named_steps['standardscaler'].mean_[0])==3.
    before=m.named_steps['standardscaler'].mean_.copy();m.predict(np.array([[9999.]]))
    np.testing.assert_array_equal(m.named_steps['standardscaler'].mean_,before)
    y=np.array(['negative','neutral','positive']*12);groups=np.repeat(np.arange(12),3)
    folds,spec=a.grouped_splits(y,groups)
    assert len(folds)==3
    for train,test in folds:assert not set(groups[train])&set(groups[test])
    try:a.grouped_splits(np.array(['negative','positive']),np.array(['one','one']))
    except ValueError:pass
    else:raise AssertionError('Inadequate grouped coverage accepted')
    with tempfile.TemporaryDirectory() as temp:
        thesis=Path(temp);archive=thesis/'Results/word_sentiment_eeg/trial_archive_125hz';archive.mkdir(parents=True)
        records=fixture(archive);out=thesis/'Results/word_sentiment_eeg/classification_similarity'
        source_hashes={p:file_digest(p) for p in archive.rglob('*') if p.is_file()}
        command=[sys.executable,str(ROOT/'scripts/analyze_trial_archive.py'),'--thesis-root',str(thesis)]
        subprocess.run(command,check=True)
        result=json.loads((out/'manifest.json').read_text());assert result['status']=='completed' and len(result['tasks_complete'])==4
        scores=pd.read_csv(out/'scores.csv')
        main_scores=scores[(scores.policy=='word_agreement')&(scores.feature=='post')&(scores.model=='logistic')]
        assert len(main_scores)==24 and main_scores.balanced_accuracy.min()>.8,main_scores
        coverage=pd.read_csv(out/'input_counts.csv');assert 'ALL_READERS' in set(coverage.subject)
        for dataset,task in [('zuco','SR'),('zuco','NR'),('zuco','TSR'),('teco','task1')]:
            folder=out/dataset/task
            for name in ['binary_confusions','three_class_confusions','classification_overview','regions_and_controls',
                         'similarity_overview','sampled_trial_matrix','reader_reproducibility','reader_score_matrix','label_policy_comparison']:
                assert (folder/'plots'/f'{name}.png').stat().st_size>1000,name
            cases=list((folder/'cases').glob('*/result.json'));assert len(cases)==36
            for path in (folder/'cases').glob('*/predictions.csv.gz'):
                pred=pd.read_csv(path);assert not pred.event_id.duplicated().any();assert pred.fold.ge(0).all()
            assert len(pd.read_csv(folder/'same_word_seen_diagnostic.csv'))>0
            assert len(pd.read_csv(folder/'repeated_word_pairs.csv.gz'))>0
        for path,sha in source_hashes.items():assert file_digest(path)==sha,'Source archive modified'
        if os.environ.get('PLOT_PREVIEW_DIR'):
            previews=Path(os.environ['PLOT_PREVIEW_DIR']);previews.mkdir(parents=True,exist_ok=True)
            for name in ['binary_confusions','similarity_overview','reader_score_matrix','label_policy_comparison']:
                shutil.copyfile(out/'zuco/SR/plots'/f'{name}.png',previews/f'analysis-{name}.png')
        features=next((out/'features').rglob('features.npz'));first_sha=file_digest(features)
        # Reuse does not reopen any archived waveform. Metadata/provenance are
        # still verified against the completed reader checkpoint.
        first=records[0];reader=archive/first['dataset']/'readers'/first['task']/first['subject']
        (reader/'epochs.npz').unlink()
        meta,post,pre,nuisance,_=a.reader_features(archive,first,out,thesis)
        assert len(meta)==18 and post.shape==(18,48) and pre.shape==(18,12) and nuisance.shape==(18,6)
        assert file_digest(features)==first_sha
        # Malformed label attachment or a duplicate event is rejected earlier
        # than model fitting (covered by local tests plus full source hashes).
    print('Classification/similarity cloud integration passed: grouped folds, train-only scaling, binary/3-class models, all label policies, controls, regional results, correlation plots, cache reuse and source preservation.')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
