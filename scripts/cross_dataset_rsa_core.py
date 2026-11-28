"""Small, testable RSA and shared-VAE machinery. No data/network side effects."""
import io
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 20261128


def stable_seed(*parts):
    import hashlib
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def reader_folds(meta, folds=3):
    """One joint fold per dataset; evaluation readers never fit their encoder."""
    assignment = np.full(len(meta), -1, int)
    for dataset, rows in meta.groupby('dataset', sort=True):
        subjects = sorted(rows.subject.astype(str).unique())
        if len(subjects) < folds * 2:
            raise ValueError(f'{dataset}: need at least {folds*2} readers')
        np.random.default_rng(stable_seed(SEED, dataset)).shuffle(subjects)
        mapping = {s: i % folds for i, s in enumerate(subjects)}
        assignment[rows.index] = rows.subject.astype(str).map(mapping)
    return assignment


def balanced_sample(meta, indices, cap, seed):
    """Balance datasets without sentiment/class labels; cap each dataset."""
    indices = np.asarray(indices, int)
    pools = [indices[meta.dataset.iloc[indices].eq(d).to_numpy()]
             for d in sorted(meta.dataset.unique())]
    n = min(cap, *(len(p) for p in pools))
    if n < 10:
        raise ValueError('Insufficient representation-training examples')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(p, n, replace=False) for p in pools])


def validation_readers(meta, training):
    validation = []
    for dataset, rows in meta.iloc[training].groupby('dataset', sort=True):
        subjects = sorted(rows.subject.astype(str).unique())
        if len(subjects) < 3:
            raise ValueError('Need >=3 fitting-side readers per dataset for validation')
        subject = subjects[stable_seed(dataset, SEED) % len(subjects)]
        validation.extend(rows.index[rows.subject.astype(str).eq(subject)].tolist())
    validation = np.array(sorted(validation), int)
    return np.setdiff1d(training, validation), validation


class SharedTransform:
    """Train-only dataset scalar equalization, then one common imputer/scaler.

    Dataset scaling changes amplitude interpretation. We compare patterns, not
    absolute native voltages; TeCo's assumed units are not independently verified.
    """
    def fit(self, X, datasets):
        self.scales = {}
        for d in sorted(set(datasets)):
            v = np.asarray(X[np.asarray(datasets) == d], float)
            scale = float(np.nanmedian(np.sqrt(np.nanmean(v*v, axis=1))))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'Invalid training amplitude scale: {d}')
            self.scales[d] = scale
        v = self.equalize(X, datasets)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(v)
        self.scaler = StandardScaler().fit(self.imputer.transform(v))
        return self

    def equalize(self, X, datasets):
        return np.asarray(X, float) / np.array([self.scales[str(d)] for d in datasets])[:, None]

    def transform(self, X, datasets):
        return self.scaler.transform(self.imputer.transform(self.equalize(X, datasets))).astype(np.float32)


def input_view(post, name):
    v = np.asarray(post).reshape(-1, 6, 8)
    if name == 'full_0_800': return v.reshape(-1, 48)
    if name == 'window_200_400': return v[:, :, 2:4].reshape(-1, 12)
    if name == 'regional_summary':
        return np.concatenate([v.mean(2), v.std(2), np.ptp(v, axis=2)], axis=1)
    if name == 'normalized_0_800':
        x = v.reshape(-1, 48)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    raise ValueError('Unknown common input: ' + name)


def exact_groups(meta, y, indices, size=40, cap=4, seed=SEED, required_classes=None):
    """Balanced, disjoint exact-size groups within reader/task/class.

    A reader/task contributes only if every class has >=size events. All classes
    contribute the same number of groups for that reader/task. No Max40 fallback.
    """
    classes = sorted(set(y[np.asarray(indices, int)])) if required_classes is None else list(required_classes)
    if len(classes) < 2: raise ValueError('Fewer than two classes')
    rng = np.random.default_rng(seed)
    groups, rows, excluded = [], [], []
    part = meta.iloc[indices]
    for (d, task, subject), g in part.groupby(['dataset','task','subject'], sort=True):
        pools = [g.index[y[g.index] == c].to_numpy().copy() for c in classes]
        n = min(cap, *(len(p)//size for p in pools))
        if n == 0:
            excluded.append(dict(dataset=d, task=task, subject=subject,
                                 class_counts={str(c): len(p) for c,p in zip(classes,pools)},
                                 reason=f'At least one class has fewer than {size} eligible trials'))
            continue
        for c, p in zip(classes, pools):
            rng.shuffle(p)
            for j in range(n):
                group = p[j*size:(j+1)*size]
                groups.append(group)
                rows.append(dict(dataset=d, task=task, subject=subject,
                                 reader=f'{d}/{subject}', label=int(c), size=size,
                                 source_keys=sorted(set(meta.source_key.iloc[group].astype(str)))))
    if not groups: raise ValueError('No complete balanced exact-size groups')
    members = np.concatenate(groups)
    if len(set(members)) != len(members): raise AssertionError('Repeated source EEG event in averages')
    return groups, pd.DataFrame(rows), excluded


def average(X, groups):
    return np.stack([np.nanmean(X[g], axis=0) for g in groups]).astype(np.float32)


def similarity(A, B, metric='pearson'):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if metric == 'pearson':
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    elif metric != 'cosine': raise ValueError(metric)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    valid = (na > 1e-12)[:, None] & (nb > 1e-12)[None, :]
    values = (A @ B.T) / np.maximum(na[:, None]*nb[None, :], 1e-30)
    return np.where(valid, np.clip(values, -1, 1), np.nan)


def rank_model_score(ranks, binary):
    """Spearman with a 0/1 model; average ranks of a binary vector are affine."""
    ranks = np.asarray(ranks, float); binary = np.asarray(binary, float)
    x = ranks-ranks.mean(); y = binary-binary.mean()
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denom) if denom > 1e-12 else None


def blocked_labels(group_meta, labels, iteration, seed=SEED):
    """One whole class-label permutation per reader, never per matrix cell.

    Reader-conditional random-effects label-swap test; NOT a stimulus-random
    population test. Known labels formed groups. Stimuli can recur across readers.
    """
    result = np.asarray(labels, int).copy()
    for reader, g in group_meta.groupby('reader', sort=True):
        classes = sorted(set(result[g.index]))
        rng = np.random.default_rng(stable_seed(seed, iteration, reader))
        mapping = dict(zip(classes, rng.permutation(classes)))
        result[g.index] = [mapping[int(v)] for v in labels[g.index]]
    return result


@lru_cache(maxsize=32)
def permutation_labels(readers, labels, n, seed):
    """Reuse one reader-block null across inputs/matrix panels; no cell shuffles."""
    labels=np.asarray(labels,int);out=np.tile(labels,(n,1))
    readers=np.asarray(readers)
    for reader in sorted(set(readers)):
        ix=np.flatnonzero(readers==reader);classes=sorted(set(labels[ix]))
        rng=np.random.default_rng(stable_seed(seed,str(reader)))
        for iteration in range(n):
            mapping=dict(zip(classes,rng.permutation(classes)))
            out[iteration,ix]=[mapping[int(v)] for v in labels[ix]]
    return out


def rsa_case(vectors, gm, left, right, metric='pearson', permutations=999,
             seed=SEED, properties=None):
    ia = np.flatnonzero(gm.dataset.eq(left)); ib = np.flatnonzero(gm.dataset.eq(right))
    if len(ia)<4 or len(ib)<4: raise ValueError('Too few complete averages for a matrix')
    S = similarity(vectors[ia], vectors[ib], metric)
    mask = np.isfinite(S)
    if left == right:
        mask &= np.triu(np.ones_like(mask), k=1)
        mask &= gm.reader.iloc[ia].to_numpy()[:,None] != gm.reader.iloc[ib].to_numpy()[None,:]
    if mask.sum() < 6: raise ValueError('Too few between-reader matrix entries')
    labels = gm.label.to_numpy(int)
    model = labels[ia,None] != labels[None,ib]
    ranks = rankdata((1-S)[mask])
    observed = rank_model_score(ranks, model[mask])
    if observed is None: raise ValueError('Constant observed RDM or class model')
    # Partial rank correlation is a diagnostic, not proof of causal sentiment.
    partial = None; property_model = None; property_rdm_correlation = None
    if properties is not None:
        p = np.asarray(properties, float)
        distances = np.sqrt(np.maximum(0, ((p[ia,None,:]-p[None,ib,:])**2).sum(2)))[mask]
        pr = rankdata(distances); property_model = rank_model_score(pr, model[mask])
        if np.std(pr)>1e-12:
            property_rdm_correlation = float(np.corrcoef(pr, ranks)[0,1]) if np.std(ranks)>0 else None
            design = np.column_stack([np.ones(len(pr)), pr])
            er = ranks-design@np.linalg.lstsq(design, ranks, rcond=None)[0]
            mr = model[mask].astype(float); mr -= design@np.linalg.lstsq(design, mr, rcond=None)[0]
            if np.linalg.norm(er)>1e-10 and np.linalg.norm(mr)>1e-10:
                partial = float(er@mr/(np.linalg.norm(er)*np.linalg.norm(mr)))
    null = []
    permuted=permutation_labels(tuple(gm.reader),tuple(labels),permutations,seed)
    li,ri=np.where(mask);li=ia[li];ri=ib[ri]
    centered=ranks-ranks.mean();norm=float(np.linalg.norm(centered));n=len(ranks)
    for begin in range(0,permutations,128):
        batch=permuted[begin:begin+128]
        model_batch=(batch[:,li]!=batch[:,ri]).astype(float)
        counts=model_batch.sum(1);denom=norm*np.sqrt(counts-counts*counts/n)
        if (denom<=1e-12).any():raise ValueError('Degenerate permutation model')
        null.extend((model_batch@centered/denom).tolist())
    same, different = S[mask & ~model], S[mask & model]
    shared = []
    for i,j in zip(*np.where(mask)):
        shared.append(bool(set(gm.source_keys.iloc[ia[i]]) & set(gm.source_keys.iloc[ib[j]])))
    # Reliability: correlations between disjoint same-class averages of one reader.
    rel=[]
    for _,g in gm[gm.dataset.isin([left,right])].groupby(['reader','label']):
        if len(g)>1:
            sim=similarity(vectors[g.index],vectors[g.index],metric)
            vals=sim[np.triu_indices(len(g),1)]; rel.extend(vals[np.isfinite(vals)].tolist())
    stats=dict(rsa_rho=observed, mean_same_class_similarity=float(same.mean()),
               mean_different_class_similarity=float(different.mean()),
               same_minus_different=float(same.mean()-different.mean()),
               partial_rho_word_properties=partial, property_model_rho=property_model,
               eeg_property_rdm_rho=property_rdm_correlation,
               reader_conditional_p=float((1+np.sum(np.asarray(null)>=observed-1e-12))/(len(null)+1)) if null else None,
               evaluated_samples_left=len(ia), evaluated_samples_right=len(ib),
               matrix_entries_used=int(mask.sum()), readers_left=int(gm.reader.iloc[ia].nunique()),
               readers_right=int(gm.reader.iloc[ib].nunique()),
               fraction_pairs_sharing_source_occurrence=float(np.mean(shared)),
               split_average_reliability=float(np.mean(rel)) if rel else None,
               reliability_pairs=len(rel), metric=metric)
    return stats, np.asarray(null), dict(similarity=S.astype(np.float32), ideal_dissimilarity=model.astype(np.uint8),
                                       inference_mask=mask, left_indices=ia, right_indices=ib)


def corrected_p(values, method):
    """Holm FWER and BH-FDR across the declared held-out result family."""
    values=np.asarray(values,float);order=np.argsort(values);p=values[order];n=len(p)
    if method=='holm': out=np.maximum.accumulate(p*(n-np.arange(n)))
    elif method=='bh': out=np.minimum.accumulate((p*n/(np.arange(n)+1))[::-1])[::-1]
    else: raise ValueError(method)
    result=np.empty(n);result[order]=np.minimum(out,1);return result


def make_vae(dim, latent=8):
    import torch
    from torch import nn
    class SmallVAE(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=nn.Sequential(nn.Linear(dim,32),nn.Tanh())
            self.mu=nn.Linear(32,latent);self.logvar=nn.Linear(32,latent)
            self.decoder=nn.Sequential(nn.Linear(latent,32),nn.Tanh(),nn.Linear(32,dim))
        def encode(self,x):
            h=self.encoder(x);return self.mu(h),self.logvar(h).clamp(-8,8)
        def forward(self,x):
            mu,lv=self.encode(x);z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
            return self.decoder(z),mu,lv
    return SmallVAE()


def train_vae(X, validation, datasets, validation_datasets, seed, epochs=60,
              device='cpu', resume=None, save_epoch=None):
    """Equal dataset counts, balanced minibatches, fixed validation MSE, early stop.

    Callback persists one small overwrite-in-place state file, preventing Drive
    Trash generations; completed models are separately sealed by checksums.
    """
    import random, torch
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    latent=min(8, X.shape[1]-1);model=make_vae(X.shape[1],latent).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    best=None;best_loss=float('inf');bad=0;history=[];start=0
    if resume:
        model.load_state_dict(resume['current']);optimizer.load_state_dict(resume['optimizer'])
        best=resume['best'];best_loss=resume['best_loss'];bad=resume['bad'];history=resume['history'];start=resume['epoch']+1
        torch.set_rng_state(resume['rng'].cpu())
        if device=='cuda' and resume.get('cuda_rng') is not None:torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
    x=torch.as_tensor(X,dtype=torch.float32,device=device)
    vx=torch.as_tensor(validation,dtype=torch.float32,device=device)
    ds=np.asarray(datasets);weights=np.array([len(ds)/(len(set(ds))*np.sum(ds==d)) for d in ds],np.float32)
    weights=torch.as_tensor(weights,device=device)
    for epoch in range(start,epochs):
        if bad>=8:break
        model.train();order=np.random.default_rng(stable_seed(seed,epoch)).permutation(len(X));losses=[]
        for begin in range(0,len(order),256):
            batch=x[order[begin:begin+256]];optimizer.zero_grad()
            reconstruction,mu,lv=model(batch)
            w=weights[order[begin:begin+256]]
            mse=(((reconstruction-batch)**2).mean(1)*w).mean();kl=(-.5*(1+lv-mu.square()-lv.exp()).mean(1)*w).mean()
            loss=mse+.05*kl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite VAE loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            mu,lv=model.encode(vx);recon=model.decoder(mu);vmse=float(((recon-vx)**2).mean().cpu())
            vkl=float((-.5*(1+lv-mu.square()-lv.exp())).mean().cpu());objective=vmse+.05*vkl
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_mse=vmse,validation_kl=vkl,validation_objective=objective))
        if objective<best_loss-1e-5:
            best_loss=objective;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        print(f'  VAE epoch {epoch+1}/{epochs}: validation MSE={vmse:.4f}; KL={vkl:.4f}; checkpoint saved',flush=True)
        if save_epoch:
            save_epoch(dict(current=model.state_dict(),optimizer=optimizer.state_dict(),best=best,best_loss=best_loss,
                            bad=bad,history=history,epoch=epoch,rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None))
    if best is None:raise RuntimeError('No successful VAE epoch')
    model.load_state_dict(best);model.eval()
    return model,history


def embeddings(model,X,device='cpu'):
    import torch
    result=[]
    with torch.no_grad():
        for i in range(0,len(X),2048):
            mu,_=model.encode(torch.as_tensor(X[i:i+2048],dtype=torch.float32,device=device));result.append(mu.cpu().numpy())
    return np.concatenate(result)


def latent_diagnostics(model, X, datasets, subjects, device='cpu'):
    import torch
    z=embeddings(model,X,device)
    with torch.no_grad():
        reconstruction=model.decoder(torch.as_tensor(z,dtype=torch.float32,device=device)).cpu().numpy()
    total=float(np.sum((z-z.mean(0))**2))
    def fraction(groups):
        between=sum(len(g)*np.sum((z[g].mean(0)-z.mean(0))**2) for g in
                    [np.flatnonzero(np.asarray(groups)==s) for s in sorted(set(groups))])
        return float(between/total) if total>1e-12 else None
    return dict(reconstruction_mse=float(np.mean((X-reconstruction)**2)),
                zero_prediction_mse=float(np.mean(X**2)),active_latent_dimensions=int((z.std(0)>.01).sum()),
                latent_std=z.std(0).tolist(),dataset_between_group_variance_fraction=fraction(datasets),
                reader_between_group_variance_fraction=fraction(subjects),
                note='Descriptive variance partition, not held-out identity classification; dataset/reader fractions overlap.')
