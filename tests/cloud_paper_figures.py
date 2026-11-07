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
from prepare_zuco_paper_figures import repair_eye_duration_columns


def main():
    from types import SimpleNamespace
    event=SimpleNamespace(colheader=['latency','endtime','duration'],data=np.array([[100.,225.,1.4e14]]))
    eye={'eyeevent':SimpleNamespace(fixations=event,saccades=SimpleNamespace(colheader=[],data=np.empty((0,0))),blinks=SimpleNamespace(colheader=[],data=np.empty((0,0))))}
    audit=repair_eye_duration_columns(eye)
    assert event.data[0,2]==125 and audit['fixations']['stored_duration_mismatches']==1
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
        for mark in folder.glob('*.json'):assert json.loads(mark.read_text())['subjects']==['A']
        second=permutation_reader(differences,differences*.5,folder,'B',31,31,chunk=8,use_gpu=False)
        result,clusters=finish_permutations(folder,first['pair_sum']+second['pair_sum'],first['pair_squares']+second['pair_squares'],40,np.ones((C,C),bool)^np.eye(C,dtype=bool),['A','B'])
        assert result['tct_p'][:,3:9].max()<.05 and result['cluster_mask'].any()
        assert any(x['p']<.05 for x in clusters)
        np.testing.assert_array_equal(first['pair_sum'],again['pair_sum'])
    print('Paper figure numerical checks passed: full Gaussian likelihood, mapping, trial permutation signal, resume and duration rule')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
