"""Full output/manifest publication smoke test with synthetic saved CPU trials.

ML optimizer parity is tested separately on real 201-feature fits. Here stubbed
p-values avoid fitting 40k features merely to test file publication and figures.
No scientific results or private data are produced.
"""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import analyze_paper_small_storage as driver
import render_zuco_paper_figures as zuco_renderer
from word_sentiment_eeg.common import atomic_npz,file_digest,write_json

def fake_models(counts,sums,squares,**kwargs):
    shape=sums.shape[2:]
    return dict(p=np.full(shape,.5),effect=np.zeros(shape),se=np.ones(shape),converged=np.ones(shape))

rng=np.random.default_rng(17);times=np.arange(-75,126)*8;edges=np.r_[np.arange(-600,1000,60),1000]
with tempfile.TemporaryDirectory() as tmp:
    for dataset,C in [('zuco',105),('teco',126)]:
        root=Path(tmp)/dataset;root.mkdir();subjects=['TEST_A','TEST_B','TEST_C'];marks={}
        for subject in subjects:
            folder=root/'readers'/subject;folder.mkdir(parents=True);neg=np.arange(6);pos=np.arange(6,12)
            x=rng.normal(0,.2,(2,12,C,201) if dataset=='zuco' else (12,C,201)).astype(np.float32)
            d=x[:,pos]-x[:,neg] if dataset=='zuco' else x[pos]-x[neg]
            atomic_npz(folder/'prepared_trials.npz',signal=x,labels=np.r_[np.zeros(6),np.ones(6)].astype(np.int8),negative_pair_index=neg,positive_pair_index=pos)
            frame=pd.DataFrame(dict(event_id=[subject+str(i) for i in range(12)],subject=subject,word=['SYNTHETIC']*12,sentence_key=np.arange(12).astype(str),word_index=0,
                duration_ms=np.arange(12)*10+150,sentiment=['negative']*6+['positive']*6,selection_status='accepted',mapping_status='matched',blink_predecessor=False,
                preceded_by_negation=False,label_5=[-1]*6+[1]*6,eligible_for_matching=True,word_length=4,word_frequency=1,status='eligible'))
            frame.to_csv(folder/'fixation_audit.csv.gz',index=False,compression='gzip');frame.to_csv(folder/'accepted_trial_metadata.csv.gz',index=False,compression='gzip')
            v=np.concatenate([x,x.std(axis=2)[:,:,None,:]],axis=2) if dataset=='zuco' else np.concatenate([x,x.std(axis=1)[:,None,:]],axis=1)
            sums=np.stack([v[:,ix].sum(axis=1,dtype=float) for ix in [neg,pos]],axis=1) if dataset=='zuco' else np.stack([v[ix].sum(axis=0,dtype=float) for ix in [neg,pos]])
            squares=np.stack([(v[:,ix].astype(float)**2).sum(axis=1) for ix in [neg,pos]],axis=1) if dataset=='zuco' else np.stack([(v[ix].astype(float)**2).sum(axis=0) for ix in [neg,pos]])
            st=dict(counts=np.array([6,6]),sums=sums,squares=squares,pair_sum=d.sum(axis=1 if dataset=='zuco' else 0,dtype=float),pair_squares=(d.astype(float)**2).sum(axis=1 if dataset=='zuco' else 0),pairs=np.array(6),times_ms=times)
            if dataset=='zuco':
                ms=np.zeros((2,2,105));mq=ms.copy()
                for r,stop in enumerate([280,304]):
                    maps=x[r,:,:,((times>=224)&(times<=stop))] if False else x[r][:,:,(times>=224)&(times<=stop)].mean(axis=2)
                    for ci,ix in enumerate([neg,pos]):ms[r,ci]=maps[ix].sum(axis=0);mq[r,ci]=(maps[ix].astype(float)**2).sum(axis=0)
                st.update(map_sums=ms,map_squares=mq,eye_counts=np.array([6,6]),eye_sums=np.zeros((2,2,201)),eye_squares=np.ones((2,2,201)),eye_means=np.zeros((2,2,201)),neighbor_means=np.ones((2,6,len(edges)-1))*.01,neighbor_edges=edges,pz_trials=x[:,:,10],pz_duration_ms=frame.duration_ms.to_numpy())
                write_json(folder/'example_sentence.json',dict(words=['SYNTHETIC'],bounds=[[0,0,100,30]],gaze=[dict(gaze_x=50,gaze_y=15,duration_ms=200,sentiment='positive')]))
                write_json(folder/'source_mapping.json',dict(blocks=[dict(block_id='1',status='included',synchronization={'max_residual_ms':1.},source_mapping=[{'position':0}])]))
            atomic_npz(folder/'statistics.npz',**st)
            write_json(folder/'prepared.json',dict(status='prepared',subject=subject,accepted_counts=[6,6],counts=[6,6],pairs=6,checksums={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}));marks[subject]=file_digest(folder/'prepared.json')
        protocol=dict(code={'src/word_sentiment_eeg/paper_figure_stats.py':file_digest(ROOT/'src/word_sentiment_eeg/paper_figure_stats.py')},subjects=subjects,budget_bytes=500_000_000,tct_permutations=4,cluster_permutations=4,tct=4,cluster=4)
        write_json(root/'protocol.json',protocol);state=dict(status='completed',protocol_sha256=file_digest(root/'protocol.json'),subjects=subjects,reader_checkpoint_sha256=marks,readers=marks);write_json(root/'cpu_preparation_manifest.json',state)
        before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
        argv=['driver','--thesis-root',str(Path(tmp)),'--run-dir',str(root),'--work-root',str(Path(tmp)/'local'),'--dataset',dataset,'--free-drive-bytes','300000000','--cpu']
        with patch.object(sys,'argv',argv),patch.object(zuco_renderer,'mixed_models',fake_models),patch.object(driver,'mixed_models',fake_models):driver.main()
        assert driver.verified_complete(root,dataset)
        for p,b in before.items():assert p.read_bytes()==b
        complete={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        with patch.object(sys,'argv',argv):driver.main()
        assert complete=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        assert len(list((root/'figures').glob('*.png')))==(4 if dataset=='zuco' else 3)
        assert not (root/'null_work').exists()
        print('PASS full synthetic figures + final-manifest verification + completed resume:',dataset,flush=True)
