"""Single-representation version of the existing paper permutation design."""
import json
from pathlib import Path
import numpy as np
from .common import atomic_npz,digest,file_digest,write_json
from .paper_figure_stats import t_stat,cluster_masses,temporal_mask


def add_reader(diff,folder,subject,tct=5000,cluster=1000,use_gpu=True,chunk=25):
    import hashlib
    diff=np.asarray(diff,dtype=np.float32);N,C,T=diff.shape
    if N<2 or not np.isfinite(diff).all():raise ValueError('Invalid paired trial differences')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(diff.tobytes()).hexdigest();tensor=None
    if use_gpu:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime for the analysis stage; CPU trials are retained')
        tensor=torch.as_tensor(diff,device='cuda')
    for kind,total in [('tct',tct),('cluster',cluster)]:
        for begin in range(0,total,chunk):
            end=min(total,begin+chunk);mark=folder/f'{kind}_{begin:05d}.json';old=None
            if mark.exists():
                state=json.loads(mark.read_text());old=folder/state['file']
                if file_digest(old)!=state['sha256']:raise ValueError('Changed permutation checkpoint')
                if subject in state['subjects']:
                    if state['signatures'][subject]!=signature:raise ValueError('Changed reader trial pairs')
                    continue
                with np.load(old,allow_pickle=False) as z:maps=z['maps'].copy()
            else:state=dict(subjects=[],signatures={},pairs=0);maps=np.zeros((end-begin,C,T),np.float32)
            for j,p in enumerate(range(begin,end)):
                rng=np.random.default_rng(int(digest([2026,subject,kind,p])[:8],16))
                if kind=='cluster':maps[j]+=np.einsum('n,nct->ct',rng.choice([-1.,1.],N).astype(np.float32),diff,optimize=True)
                else:
                    ix=np.argsort(rng.random((N,C)),axis=1)
                    if tensor is not None:
                        shuffled=tensor.gather(1,torch.as_tensor(ix,device='cuda')[:,:,None].expand(N,C,T));maps[j]+=shuffled.sum(dim=0).cpu().numpy()
                    else:maps[j]+=np.take_along_axis(diff,ix[:,:,None],axis=1).sum(axis=0)
            path=folder/f'{kind}_{begin:05d}__{len(state["subjects"])+1:02d}.npz'
            atomic_npz(path,maps=maps)
            write_json(mark,dict(subjects=state['subjects']+[subject],signatures=dict(state['signatures'],**{subject:signature}),
                pairs=state['pairs']+N,file=path.name,sha256=file_digest(path)))
            if old is not None and old!=path:old.unlink(missing_ok=True)
            if begin%250==0 or end==total:print(f'PERMUTATIONS {subject} {kind}: {end}/{total}',flush=True)


def finish(folder,total,squares,n,adj,subjects,tct,cluster,times):
    from scipy.stats import t
    threshold=t.ppf(.975,n-1);observed=t_stat(total,squares,n);gfp=(total/n).std(axis=0)
    exceed=np.zeros_like(gfp,dtype=int);null=[];seen={'cluster':0,'tct':0}
    for kind in ['cluster','tct']:
        for mark in sorted(Path(folder).glob(kind+'_*.json')):
            s=json.loads(mark.read_text());path=mark.parent/s['file']
            if set(s['subjects'])!=set(subjects) or s['pairs']!=n or file_digest(path)!=s['sha256']:raise ValueError('Permutation coverage disagrees with prepared trials')
            with np.load(path,allow_pickle=False) as z:maps=z['maps']
            for v in maps:
                seen[kind]+=1
                if kind=='tct':exceed+=(v/n).std(axis=0)>=gfp
                else:null.append(max([mass for mass,_ in cluster_masses(t_stat(v,squares,n),adj,threshold)],default=0))
    if seen!={'tct':tct,'cluster':cluster}:raise ValueError('Permutation count incomplete')
    report=[];mask=np.zeros_like(observed,dtype=bool)
    for mass,indices in cluster_masses(observed,adj,threshold):
        p=float((1+sum(np.asarray(null)>=mass))/(len(null)+1));channels,bins=np.unravel_index(indices,observed.shape)
        report.append(dict(mass=mass,p=p,channel_time_samples=len(indices),channels=np.unique(channels).tolist(),
            start_ms=int(times[bins].min()),end_ms=int(times[bins].max())))
        if p<.05:mask.reshape(-1)[indices]=True
    p=(exceed+1)/(tct+1)
    return dict(tct_p=p,tct_significant=temporal_mask(p<.05),cluster_mask=mask,t_values=observed),report
