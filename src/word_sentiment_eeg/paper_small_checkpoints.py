"""Write-once inference checkpoints: no Drive rename/unlink generations.

Null maps live only in RAM. Completed all-reader batches persist exact exceedance
counts or cluster maxima, sufficient for the original permutation tests. EEG
trials, features, RNG seeds, float32 sum order and inference rules are unchanged.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from . import common
from .common import digest,file_digest
from .paper_figure_stats import t_stat,cluster_masses,temporal_mask


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()


class ImmutableStore:
    """Writes are new files or repairs of uncommitted partial writes only.

    Completion records are separate immutable checksums. No os.replace/unlink
    on Drive. No scientific data are removed and no Trash is emptied.
    """
    def __init__(self,root,budget):
        self.root=Path(root);self.budget=int(budget);self.root.mkdir(parents=True,exist_ok=True)
        self.used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
    def check(self,extra):
        if self.used+extra>self.budget:raise OSError('Analysis storage cap reached; CPU archives and committed batches retained')
    def bytes(self,relative,payload):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            old=path.read_bytes()
            if old==payload:return path
            if not payload.startswith(old):raise ValueError('Changed/corrupt immutable checkpoint: '+str(relative))
        else:old=b''
        self.check(len(payload)-len(old))
        # Same-file repair of a torn prefix, never replace/rename/delete.
        with path.open('wb') as stream:stream.write(payload);stream.flush()
        self.used+=len(payload)-len(old)
        if path.read_bytes()!=payload:raise OSError('Checkpoint readback failed: '+str(relative))
        return path
    def json(self,relative,value):
        path=self.root/relative
        if path.exists():
            try:
                if json.loads(path.read_text())==value:return path
            except (json.JSONDecodeError,UnicodeDecodeError):pass
        return self.bytes(relative,json_bytes(value))
    def file(self,relative,source,reuse_committed=False):
        source=Path(source);path=self.root/relative;sha=file_digest(source);size=source.stat().st_size
        marker=Path('analysis_state/published')/(digest(str(relative))+'.json')
        mpath=self.root/marker
        if mpath.exists():
            record=json.loads(mpath.read_text())
            if record['path']!=str(relative) or not path.is_file() or file_digest(path)!=record['sha256']:raise ValueError('Published analysis output changed: '+str(relative))
            if not reuse_committed and record['sha256']!=sha:raise ValueError('Recomputed output differs from its committed value: '+str(relative))
            return record['sha256']
        path.parent.mkdir(parents=True,exist_ok=True)
        oldsize=path.stat().st_size if path.exists() else 0
        if path.exists() and oldsize==size and file_digest(path)==sha:pass
        else:
            if oldsize>=size and oldsize:raise ValueError('Uncommitted existing output conflicts: '+str(relative))
            self.check(size-oldsize+1000)
            with source.open('rb') as src,path.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            self.used+=size-oldsize
            if file_digest(path)!=sha:raise OSError('Output readback failed: '+str(relative))
        self.json(marker,dict(path=str(relative),sha256=sha,bytes=size))
        return sha


def valid_chunk(path,identity,kind,begin,end,representations,times):
    try:value=json.loads(path.read_text())
    except (json.JSONDecodeError,UnicodeDecodeError):return None
    payload={k:v for k,v in value.items() if k!='checksum'}
    if value.get('checksum')!=digest(payload):raise ValueError('Permutation batch checksum mismatch')
    if any(payload.get(k)!=v for k,v in dict(identity=identity,kind=kind,begin=begin,end=end).items()):raise ValueError('Permutation batch identity mismatch')
    x=np.asarray(payload['values'])
    expected=(representations,times) if kind=='tct' else (representations,end-begin)
    if x.shape!=expected or not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid permutation batch values')
    if kind=='tct' and ((x>end-begin).any() or not np.equal(x,np.floor(x)).all()):raise ValueError('Invalid TCT exceedance counts')
    return payload


def permutations(differences,subjects,pair_sum,pair_squares,n,adj,times,store,identity,tct=5000,cluster=1000,chunk=25,use_gpu=True,max_new_chunks=None):
    """Chunk-first computation; each persisted batch includes every reader."""
    from scipy.stats import t
    if len(differences)!=len(subjects) or len(set(subjects))!=len(subjects):raise ValueError('Reader roster mismatch')
    R,C,T=pair_sum.shape;threshold=t.ppf(.975,n-1);observed=t_stat(pair_sum,pair_squares,n);gfp=(pair_sum/n).std(axis=1)
    observed_sum=np.zeros_like(pair_sum);observed_squares=np.zeros_like(pair_squares);pairs=0
    for diff in differences:
        if diff.ndim!=4 or diff.shape[0]!=R or diff.shape[2:]!=(C,T) or diff.dtype!=np.float32 or not np.isfinite(diff).all():raise ValueError('Invalid stored trial differences')
        observed_sum+=diff.sum(axis=1,dtype=np.float64);observed_squares+=(diff.astype(np.float64)**2).sum(axis=1);pairs+=diff.shape[1]
    if pairs!=n or not np.array_equal(observed_sum,pair_sum) or not np.array_equal(observed_squares,pair_squares):raise ValueError('Saved trials disagree with prepared pair statistics')
    # No GPU is needed on a completed resume. Load tensors lazily.
    tensors=None;torch=None;exceed=np.zeros((R,T),np.int64);maxima=[[] for _ in range(R)];completed=0;new=0
    total_batches=sum((v+chunk-1)//chunk for v in [tct,cluster])
    for kind,iterations in [('tct',tct),('cluster',cluster)]:
        for begin in range(0,iterations,chunk):
            end=min(begin+chunk,iterations);relative=Path('analysis_state/permutation_batches')/f'{kind}_{begin:05d}_{end:05d}.json';path=store.root/relative
            state=valid_chunk(path,identity,kind,begin,end,R,T) if path.exists() else None
            if state is None:
                if tensors is None and use_gpu:
                    import torch
                    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime; saved CPU trials and completed batches are retained')
                    tensors=[torch.as_tensor(d,device='cuda') for d in differences]
                maps=np.zeros((end-begin,R,C,T),np.float32)
                for si,(subject,diff) in enumerate(zip(subjects,differences)):
                    N=diff.shape[1]
                    for j,p in enumerate(range(begin,end)):
                        rng=np.random.default_rng(int(digest([2026,subject,kind,p])[:8],16))
                        if kind=='cluster':maps[j]+=np.einsum('n,rnct->rct',rng.choice([-1.,1.],N).astype(np.float32),diff,optimize=True)
                        else:
                            ix=np.argsort(rng.random((N,C)),axis=1)
                            if tensors is not None:
                                index=torch.as_tensor(ix,device='cuda');shuffled=torch.gather(tensors[si],2,index[None,:,:,None].expand(R,N,C,T));maps[j]+=shuffled.sum(dim=1).cpu().numpy()
                            else:maps[j]+=np.take_along_axis(diff,ix[None,:,:,None],axis=2).sum(axis=1)
                if kind=='tct':values=sum((value/n).std(axis=1)>=gfp for value in maps).astype(np.int64)
                else:
                    values=np.array([[max([m for m,_ in cluster_masses(t_stat(v,pair_squares,n)[r],adj,threshold)],default=0.) for v in maps] for r in range(R)])
                state=dict(identity=identity,kind=kind,begin=begin,end=end,values=values.tolist());store.json(relative,dict(state,checksum=digest(state)))
                new+=1
            if kind=='tct':exceed+=np.asarray(state['values'],np.int64)
            else:
                for r in range(R):maxima[r].extend(state['values'][r])
            completed+=1
            print(f'PERMUTATION BATCH {completed}/{total_batches}: {kind} {end}/{iterations}; all {len(subjects)} readers; saved compact checkpoint',flush=True)
            if max_new_chunks is not None and new>=max_new_chunks:return None,None
    p=(exceed+1)/(tct+1);mask=np.zeros_like(observed,dtype=bool);report=[]
    for r in range(R):
        for mass,members in cluster_masses(observed[r],adj,threshold):
            cp=float((1+sum(np.asarray(maxima[r])>=mass))/(cluster+1));channels,bins=np.unravel_index(members,(C,T))
            report.append(dict(representation=r,mass=mass,p=cp,samples=len(members),channels=np.unique(channels).tolist(),start_ms=int(times[bins].min()),end_ms=int(times[bins].max())))
            if cp<.05:mask[r].reshape(-1)[members]=True
    return dict(tct_p=p,tct_significant=temporal_mask(p<.05),tct_observed_gfp=gfp,cluster_mask=mask,t_values=observed),report


@contextmanager
def model_checkpoints(store,local_analysis):
    """Persist small immutable ML frames; restore the furthest verified frame.

    Atomic rewrite/replacement happens only on the local runtime disk. Old Drive
    frames remain bounded (~two hundred small files), instead of being trashed.
    """
    local_analysis=Path(local_analysis);local_analysis.mkdir(parents=True,exist_ok=True)
    frames=store.root/'analysis_state/model_frames'
    if frames.exists():
        for directory in frames.iterdir():
            if not directory.is_dir():continue
            candidates=sorted(directory.glob('*.npz'),reverse=True)
            for frame in candidates:
                expected=frame.stem.split('_')[-1]
                if file_digest(frame)!=expected:continue  # uncommitted interrupted transfer
                target=local_analysis/(directory.name+'.npz');shutil.copyfile(frame,target);break
    original=common.atomic_npz
    def save(path,**arrays):
        original(path,**arrays)
        if Path(path).parent.resolve()==local_analysis.resolve() and {'signature','next_feature'}<=set(arrays):
            sha=file_digest(path);relative=Path('analysis_state/model_frames')/Path(path).stem/f'{int(arrays["next_feature"]):07d}_{sha}.npz'
            store.file(relative,path)
    common.atomic_npz=save
    try:yield
    finally:common.atomic_npz=original
