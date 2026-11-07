"""Independent Python Gaussian mixed models and paper-style permutation tests.

Normal/identity mixed models use per-reader, per-condition sufficient statistics
of ALL trials, not just 12 reader means. Covariance parameters are fit by ML.
Permutation maps can be accumulated one reader at a time without a second EEG
archive. Never substitute reader permutations for trial permutations silently.
"""
import numpy as np


def gaussian_profile(parameters,counts,sums,squares,return_fit=False):
    """Marginal normal likelihood with correlated random condition intercepts.

    This is a reparameterization of a reader random intercept + sentiment slope.
    Includes within-reader/condition residual sums of squares, preserving the
    full individual-trial Gaussian likelihood. Constants independent of variance
    parameters are omitted. Inputs are readers × 2 conditions for one feature.
    """
    sigma2=np.exp(2*parameters[0]);sd=np.exp(np.asarray(parameters[1:3]));rho=np.tanh(parameters[3])
    B=np.array([[sd[0]**2,rho*sd[0]*sd[1]],[rho*sd[0]*sd[1],sd[1]**2]])
    K=np.broadcast_to(B,(len(counts),2,2)).copy();K[:,0,0]+=sigma2/counts[:,0];K[:,1,1]+=sigma2/counts[:,1]
    sign,logdet=np.linalg.slogdet(K)
    if np.any(sign<=0):return np.inf
    inverse=np.linalg.inv(K);means=sums/counts
    covariance=np.linalg.inv(inverse.sum(axis=0));beta=covariance@np.einsum('sij,sj->i',inverse,means)
    residual=means-beta
    within=np.maximum(squares-sums*sums/counts,0).sum()
    value=.5*((counts.sum()-2*len(counts))*np.log(sigma2)+logdet.sum()+within/sigma2+
              np.einsum('si,sij,sj->',residual,inverse,residual))
    return (value,beta,covariance) if return_fit else float(value)


def mixed_models(counts,sums,squares,progress=None,checkpoint=None):
    """Fit every channel/time feature; failed optimizations have NaN p-values."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    shape=sums.shape[2:];Y=sums.reshape(len(counts),2,-1);Q=squares.reshape(len(counts),2,-1)
    if np.any(counts<2) or len(counts)<3:raise ValueError('Mixed model requires >=3 readers with >=2 trials in each class')
    result={k:np.full(Y.shape[-1],np.nan) for k in ['effect','se','p','converged']}
    begin=0
    if checkpoint is not None:
        from pathlib import Path
        from .common import digest,atomic_npz
        import hashlib
        checkpoint=Path(checkpoint)
        signature=digest([hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in [counts,sums,squares]])
        if checkpoint.is_file():
            with np.load(checkpoint,allow_pickle=False) as saved:
                if str(saved['signature'])!=signature:raise ValueError('Mixed-model sufficient statistics changed')
                begin=int(saved['next_feature'])
                for k in result:result[k]=saved[k].copy()
    N=counts.sum()
    for j in range(begin,Y.shape[-1]):
        sy=Y[:,:,j];sq=Q[:,:,j]
        center=sy.sum()/N;scale=np.sqrt(max((sq.sum()-sy.sum()**2/N)/N,1e-12))
        s=(sy-counts*center)/scale;q=(sq-2*center*sy+counts*center**2)/scale**2
        within=max(float(np.maximum(q-s*s/counts,0).sum()/(N-2*len(counts))),1e-6)
        means=s/counts;between=np.maximum(means.var(axis=0)-within*np.mean(1/counts,axis=0),1e-5)
        start=[.5*np.log(within),.5*np.log(between[0]),.5*np.log(between[1]),0.]
        fit=minimize(gaussian_profile,start,args=(counts,s,q),method='L-BFGS-B',bounds=[(-12,6)]*3+[(-8,8)],options={'maxiter':150})
        result['converged'][j]=float(fit.success)
        if fit.success and np.isfinite(fit.fun):
            _,beta,covariance=gaussian_profile(fit.x,counts,s,q,True)
            effect=(beta[1]-beta[0])*scale
            se=np.sqrt(max(covariance[0,0]+covariance[1,1]-2*covariance[0,1],0))*scale
            result['effect'][j]=effect;result['se'][j]=se
            if se>0:result['p'][j]=2*norm.sf(abs(effect/se))
        if progress and (j%200==0 or j+1==Y.shape[-1]):progress(j+1,Y.shape[-1])
        if checkpoint is not None and ((j+1)%200==0 or j+1==Y.shape[-1]):
            atomic_npz(checkpoint,**result,signature=np.array(signature),next_feature=np.array(j+1))
    return {k:v.reshape(shape) for k,v in result.items()}


def temporal_mask(values,minimum=6):
    values=np.asarray(values,dtype=bool);out=np.zeros_like(values)
    flat=values.reshape(-1,values.shape[-1]);target=out.reshape(flat.shape)
    for i,row in enumerate(flat):
        change=np.diff(np.r_[False,row,False].astype(int));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)
        for a,b in zip(starts,ends):
            if b-a>=minimum:target[i,a:b]=True
    return out


def scalp_geometry(montage):
    """Azimuthal projection and distance-pruned Delaunay sensor adjacency."""
    from scipy.spatial import Delaunay
    xyz=montage[['X','Y','Z']].to_numpy(float);norms=np.linalg.norm(xyz,axis=1)
    theta=np.arctan2(-xyz[:,1],xyz[:,0]);radius=np.arccos(np.clip(xyz[:,2]/norms,-1,1))/(np.pi/2)
    xy=np.c_[radius*np.sin(theta),radius*np.cos(theta)];xy/=np.max(np.linalg.norm(xy,axis=1))
    tri=Delaunay(xy);edges=set()
    for triangle in tri.simplices:
        for a,b in [(triangle[0],triangle[1]),(triangle[0],triangle[2]),(triangle[1],triangle[2])]:edges.add(tuple(sorted((int(a),int(b)))))
    typical=np.median([np.linalg.norm(xy[a]-xy[b]) for a,b in edges]);adj=np.zeros((len(xy),len(xy)),bool)
    for a,b in edges:
        if np.linalg.norm(xy[a]-xy[b])<=2.5*typical:adj[a,b]=adj[b,a]=True
    return xy,adj


def t_stat(total,squares,n):
    variance=np.maximum((squares-total*total/n)/(n-1),1e-12)
    return (total/n)/np.sqrt(variance/n)


def cluster_masses(stat,adj,threshold,minimum_neighbors=5):
    from scipy.sparse import csr_matrix,eye,diags,kron
    from scipy.sparse.csgraph import connected_components
    C,T=stat.shape;time=diags([np.ones(T-1),np.ones(T-1)],[-1,1],shape=(T,T),format='csr')
    graph=kron(csr_matrix(adj),eye(T,format='csr'))+kron(eye(C,format='csr'),time)
    clusters=[]
    for sign in [1,-1]:
        mask=stat*sign>threshold
        mask &=adj.astype(int)@mask.astype(int)>=minimum_neighbors
        indices=np.flatnonzero(mask.reshape(-1))
        if not len(indices):continue
        _,labels=connected_components(graph[indices][:,indices],directed=False)
        for label in np.unique(labels):
            members=indices[labels==label];clusters.append((float(np.abs(stat.reshape(-1)[members]).sum()),members))
    return clusters


def permutation_reader(raw_diff,clean_diff,folder,subject,total_tct=5000,total_cluster=1000,chunk=25,use_gpu=True):
    """Add one reader's trial permutations to resumable sum-map chunks.

    Each TCT shuffle permutes electrode positions independently per paired trial.
    Each cluster shuffle flips signs independently per within-reader trial pair.
    Same permutations are used for raw and cleaned representations. A chunk's
    included-reader list prevents repeated additions after interruption.
    """
    from pathlib import Path
    from .common import write_json,file_digest,atomic_npz,digest
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    diff=np.stack([raw_diff,clean_diff]).astype(np.float32)
    R,N,C,T=diff.shape
    if N<2 or not np.isfinite(diff).all():raise ValueError('Invalid paired differences')
    device=None;torch=None;tensor=None
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():device='cuda';tensor=torch.as_tensor(diff,device=device)
        except ImportError:pass
    for kind,iterations in [('tct',total_tct),('cluster',total_cluster)]:
        for begin in range(0,iterations,chunk):
            end=min(begin+chunk,iterations);mark=folder/f'{kind}_{begin:05d}.json';old_path=None
            included=[];pairs=0
            if mark.is_file():
                saved=__import__('json').loads(mark.read_text())
                old_path=folder/saved['file']
                if file_digest(old_path)!=saved['sha256']:raise ValueError('Permutation checkpoint corrupt')
                included=saved['subjects'];pairs=saved['pairs']
                if subject in included:continue
                with np.load(old_path,allow_pickle=False) as z:maps=z['maps'].copy()
            else:maps=np.zeros((end-begin,R,C,T),np.float32)
            for j,p in enumerate(range(begin,end)):
                seed=int(digest([2026,subject,kind,p])[:8],16);rng=np.random.default_rng(seed)
                if kind=='cluster':
                    signs=rng.choice([-1.,1.],N).astype(np.float32)
                    maps[j]+=np.einsum('n,rnct->rct',signs,diff,optimize=True)
                elif device:
                    index=torch.as_tensor(np.argsort(rng.random((N,C)),axis=1),device=device)
                    shuffled=torch.gather(tensor,2,index[None,:,:,None].expand(R,N,C,T))
                    maps[j]+=shuffled.sum(dim=1).cpu().numpy()
                else:
                    index=np.argsort(rng.random((N,C)),axis=1)
                    maps[j]+=np.take_along_axis(diff,index[None,:,:,None],axis=2).sum(axis=1)
            # Immutable generation + atomic pointer: a disconnect between data
            # and metadata writes leaves the prior committed chunk recoverable.
            path=folder/f'{kind}_{begin:05d}__{len(included)+1:02d}.npz'
            atomic_npz(path,maps=maps)
            write_json(mark,dict(subjects=included+[subject],pairs=pairs+N,file=path.name,sha256=file_digest(path)))
            if old_path and old_path!=path:old_path.unlink(missing_ok=True)
            if begin%250==0 or end==iterations:print(f'PERMUTATIONS {subject} {kind}: {end}/{iterations}',flush=True)
    return dict(pair_sum=diff.sum(axis=1,dtype=np.float64),pair_squares=(diff.astype(np.float64)**2).sum(axis=1),pairs=N)


def finish_permutations(folder,pair_sum,pair_squares,n,adj,subjects):
    from pathlib import Path
    import json
    from scipy.stats import t
    folder=Path(folder);threshold=t.ppf(.975,n-1)
    observed=t_stat(pair_sum,pair_squares,n);obs_gfp=np.std(pair_sum/n,axis=1)
    null_max=[[],[]];tct_exceed=np.zeros_like(obs_gfp,dtype=int);n_tct=0
    for kind in ['cluster','tct']:
        for mark in sorted(folder.glob(kind+'_*.json')):
            saved=json.loads(mark.read_text());path=folder/saved['file']
            from .common import file_digest
            if set(saved['subjects'])!=set(subjects) or saved['pairs']!=n or file_digest(path)!=saved['sha256']:
                raise ValueError('Permutation chunks do not cover the same reader/trial roster')
            with np.load(path,allow_pickle=False) as z:maps=z['maps']
            for value in maps:
                if kind=='tct':tct_exceed+=np.std(value/n,axis=1)>=obs_gfp;n_tct+=1
                else:
                    stat=t_stat(value,pair_squares,n)
                    for r in range(2):null_max[r].append(max([m for m,_ in cluster_masses(stat[r],adj,threshold)],default=0.))
    tct_p=(tct_exceed+1)/(n_tct+1);cluster_mask=np.zeros_like(observed,dtype=bool);cluster_report=[]
    for r in range(2):
        for mass,members in cluster_masses(observed[r],adj,threshold):
            p=(1+np.sum(np.asarray(null_max[r])>=mass))/(len(null_max[r])+1)
            cluster_report.append(dict(representation=r,mass=mass,p=float(p),samples=len(members)))
            if p<.05:cluster_mask[r].reshape(-1)[members]=True
    return dict(tct_p=tct_p,tct_significant=temporal_mask(tct_p<.05),tct_observed_gfp=obs_gfp,
                cluster_mask=cluster_mask,t_values=observed),cluster_report
