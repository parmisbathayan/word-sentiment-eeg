"""Gaussian sufficient-statistic identity, permutation recovery and figure QA."""
from pathlib import Path
import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from word_sentiment_eeg.paper_figure_stats import gaussian_profile,mixed_models,permutation_reader,finish_permutations,temporal_mask
from word_sentiment_eeg.paper_released import sentence_location
from prepare_zuco_paper_figures import repair_eye_duration_columns,excluded_clock_block


def figure_previews():
    """Small artificial fixture for layout QA; never publish as scientific output."""
    from word_sentiment_eeg.paper_figure_stats import scalp_geometry
    from render_zuco_paper_figures import render_all
    from word_sentiment_eeg.common import atomic_npz,write_json,file_digest
    rng=np.random.default_rng(88);out=ROOT/'artifacts/paper_figure_previews';out.mkdir(parents=True,exist_ok=True)
    (out/'analysis').mkdir(exist_ok=True);(out/'SYNTHETIC_TEST_ONLY').touch()
    times=np.arange(-75,126)*8;edges=np.r_[np.arange(-600,1000,60),1000]
    readers=['TEST_A','TEST_B','TEST_C','TEST_D'];stats=[];frames=[]
    spatial=np.linspace(-1,1,105)[:,None];temporal=np.exp(-((times-260)/100)**2)[None,:]
    for subject in readers:
        counts=np.array([20,22]);mean=rng.normal(0,.02,(2,2,106,201));mean[:,:,:105]+=spatial[None,None]*temporal[None,None]*.3
        mean[:,:,105]=.2+np.abs(mean[:,:,105]);sums=mean*counts[None,:,None,None]
        stats.append(dict(counts=counts,sums=sums,times_ms=times,neighbor_edges=edges,
                          eye_means=rng.normal(0,10,(2,2,201)),neighbor_means=rng.uniform(0,.02,(2,6,len(edges)-1)),
                          pz_trials=rng.normal(0,1,(2,42,201)),pz_duration_ms=np.arange(42)*8+100))
        rows=[]
        for i in range(54):
            cls='negative' if i<20 else ('positive' if i<42 else 'neutral')
            rows.append(dict(subject=subject,duration_ms=100+8*i,mapping_status='matched',blink_predecessor=False,
                             label_5=-1 if cls=='negative' else (1 if cls=='positive' else 0),preceded_by_negation=False,
                             sentiment=cls,selection_status='accepted' if i<42 else 'neutral',eligible_for_matching=i<42,
                             word_length=1+i%14,word_frequency=1+i%10))
        frames.append(pd.DataFrame(rows))
    folder=out/'readers'/readers[0];folder.mkdir(parents=True,exist_ok=True)
    (folder/'example_sentence.json').write_text(json.dumps(dict(words=['A','positive','example'],bounds=[[0,0,20,30],[30,0,110,30],[120,0,200,30]],
        gaze=[dict(gaze_x=60,gaze_y=15,duration_ms=200,sentiment='positive')])))
    xy,adj=scalp_geometry(pd.read_csv(ROOT/'assets/zuco_chanlocs.csv'))
    p=rng.uniform(.1,1,(2,106,201));p[:,:8,100:110]=.01
    group=dict(times_ms=times,counts=np.stack([v['counts'] for v in stats]),sums=np.stack([v['sums'] for v in stats]),
               xy=xy,p=p,map_p=rng.uniform(.1,1,(2,105)))
    mask=np.zeros((2,105,201),bool);mask[:,:8,100:110]=True
    permutation=dict(cluster_mask=mask,tct_p=rng.uniform(.2,1,(2,201)),tct_significant=np.zeros((2,201),bool))
    for subject,stat,frame in zip(readers,stats,frames):
        folder=out/'readers'/subject;folder.mkdir(parents=True,exist_ok=True)
        atomic_npz(folder/'statistics.npz',**stat);frame.to_csv(folder/'fixation_audit.csv.gz',index=False)
        write_json(folder/'manifest.json',dict(status='completed',checksums={name:file_digest(folder/name) for name in ['statistics.npz','fixation_audit.csv.gz']}))
    write_json(out/'preparation_manifest.json',dict(status='completed',subjects=readers))
    analysis=out/'analysis';atomic_npz(analysis/'group_results.npz',**group);atomic_npz(analysis/'permutation_results.npz',**permutation)
    atomic_npz(analysis/'eye_model.npz',included_readers=np.ones(4,bool),p=np.ones((2,201))*.5);write_json(analysis/'clusters.json',[])
    write_json(analysis/'analysis_manifest.json',dict(status='completed',checksums={name:file_digest(analysis/name) for name in ['group_results.npz','permutation_results.npz','eye_model.npz','clusters.json']}))
    render_all(out)
    assert json.loads((out/'manifest.json').read_text())['status']=='completed'
    assert len(list((out/'figures').glob('*.png')))==4
    print('Four synthetic layout previews rendered; NOT scientific results')


def main():
    from types import SimpleNamespace
    event=SimpleNamespace(colheader=['latency','endtime','duration'],data=np.array([[100.,225.,1.4e14]]))
    eye={'eyeevent':SimpleNamespace(fixations=event,saccades=SimpleNamespace(colheader=[],data=np.empty((0,0))),blinks=SimpleNamespace(colheader=[],data=np.empty((0,0))))}
    audit=repair_eye_duration_columns(eye)
    assert event.data[0,2]==125 and audit['fixations']['stored_duration_mismatches']==1
    eye['eyeevent'].fixations=SimpleNamespace(colheader=['duration','fix_avgpos_x','fix_avgpos_y'],data=np.array([[125.,100.,200.]]))
    error=ValueError('clock exceeds limit');error.diagnostics={'max_residual_ms':10.6}
    excluded,diagnostic=excluded_clock_block('TEST','3',eye,error)
    assert len(excluded)==1 and excluded.selection_status.eq('block_failed_clock_alignment').all()
    assert diagnostic['corrected_fixations_excluded']==1
    rng=np.random.default_rng(75)
    counts=np.array([[12,17],[14,10],[10,19],[18,13]],int)
    values=[[rng.normal(i*.3+c*2,.8,n) for c,n in enumerate(row)] for i,row in enumerate(counts)]
    sums=np.array([[v.sum() for v in row] for row in values]);squares=np.array([[(v*v).sum() for v in row] for row in values])
    parameters=np.array([-.2,-.1,.2,.3]);value,beta,cov=gaussian_profile(parameters,counts,sums,squares,True)
    sigma2=np.exp(2*parameters[0]);sd=np.exp(parameters[1:3]);rho=np.tanh(parameters[3]);B=np.array([[sd[0]**2,rho*sd.prod()],[rho*sd.prod(),sd[1]**2]])
    design=[];covariances=[];ys=[]
    for row,observed in zip(counts,values):
        Z=np.repeat(np.eye(2),row,axis=0);design.append(Z);covariances.append(np.eye(sum(row))*sigma2+Z@B@Z.T);ys.extend(np.concatenate(observed))
    X=np.concatenate(design);V=block_diag(*covariances);inverse=np.linalg.inv(V);dense_cov=np.linalg.inv(X.T@inverse@X);dense_beta=dense_cov@X.T@inverse@ys
    residual=np.array(ys)-X@dense_beta;dense=.5*(np.linalg.slogdet(V)[1]+residual@inverse@residual)
    # Omitted determinant(counts) is constant w.r.t. model parameters.
    np.testing.assert_allclose(value+.5*np.log(counts).sum(),dense,atol=1e-8)
    np.testing.assert_allclose(beta,dense_beta,atol=1e-9);np.testing.assert_allclose(cov,dense_cov,atol=1e-9)
    models=mixed_models(counts,sums[:,:,None],squares[:,:,None]);assert models['effect'][0]>1 and models['p'][0]<.01
    raw=rng.normal(size=(4,1000)).astype('float32');sentence=dict(raw=raw[:,200:500].copy())
    assert sentence_location(raw,sentence)==(200,'exact')
    doubled=np.concatenate([raw,raw],axis=1);assert sentence_location(doubled,sentence)[0] is None
    assert temporal_mask(np.array([0,1,1,1,1,1,1,0],bool)).sum()==6
    N,C,T=20,16,12;pattern=np.r_[np.ones(8),-np.ones(8)][:,None]*np.r_[np.zeros(3),np.ones(6),np.zeros(3)][None,:]
    differences=rng.normal(0,.1,(N,C,T))+pattern
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        first=permutation_reader(differences,differences*.5,folder,'A',31,31,chunk=8,use_gpu=False)
        again=permutation_reader(differences,differences*.5,folder,'A',31,31,chunk=8,use_gpu=False)
        try:
            permutation_reader(differences+1,differences*.5,folder,'A',31,31,chunk=8,use_gpu=False)
        except ValueError as error:assert 'inputs changed' in str(error)
        else:raise AssertionError('Changed reader inputs reused old permutations')
        for mark in folder.glob('*.json'):assert json.loads(mark.read_text())['subjects']==['A']
        second=permutation_reader(differences,differences*.5,folder,'B',31,31,chunk=8,use_gpu=False)
        result,clusters=finish_permutations(folder,first['pair_sum']+second['pair_sum'],first['pair_squares']+second['pair_squares'],40,np.ones((C,C),bool)^np.eye(C,dtype=bool),['A','B'])
        assert result['tct_p'][:,3:9].max()<.05 and result['cluster_mask'].any()
        assert any(x['p']<.05 for x in clusters)
        np.testing.assert_array_equal(first['pair_sum'],again['pair_sum'])
    print('Paper figure numerical checks passed: full Gaussian likelihood, mapping, trial permutation signal, resume and duration rule')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
    figure_previews()
