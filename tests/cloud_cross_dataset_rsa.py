"""Cloud-only scientific checks: shared coordinates, folds, RSA null and resume."""
from pathlib import Path
import sys,tempfile,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import cross_dataset_rsa_core as c
import cross_dataset_rsa as run
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore

rng=np.random.default_rng(53)
rows=[];x=[];y=[]
for d in ['teco','zuco']:
    for s in range(9):
        for label in range(2):
            for j in range(100):
                rows.append(dict(event_id=f'{d}/{s}/{label}/{j}',dataset=d,task='task1' if d=='teco' else 'SR',subject=str(s),
                    reader=f'{d}/{s}',source_key=f'{d}/{label}/{j}',word_length=3+label,relative_word_position=j/100,sentence_length=20,is_control=False))
                pattern=np.tile(np.array([-1.,1.]),24)*(2*label-1)
                x.append(pattern+rng.normal(0,.3,48));y.append(label)
m=pd.DataFrame(rows);X=np.array(x,np.float32);y=np.array(y,int)
folds=c.reader_folds(m)
for f in range(3):
    tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
    assert not set(m.reader.iloc[tr])&set(m.reader.iloc[te])
    fit,val=c.validation_readers(m,tr)
    assert not set(m.reader.iloc[fit])&set(m.reader.iloc[val])
    assert not set(m.reader.iloc[te])&set(m.reader.iloc[val])
    selected=c.balanced_sample(m,fit,12000,7)
    assert m.dataset.iloc[selected].value_counts().nunique()==1
groups,gm,excluded=c.exact_groups(m,y,np.arange(len(m)),size=40,cap=4)
assert not excluded and all(len(g)==40 for g in groups)
assert len(set(np.concatenate(groups)))==len(np.concatenate(groups))
assert all(len(g)==2 for _,g in gm.groupby(['reader','label']))
try:c.exact_groups(m,y,np.arange(len(m)),size=40,required_classes=[0,1,2])
except ValueError:pass
else:raise AssertionError('Missing declared class silently dropped')
v=c.average(X,groups)
for a,b in run.PAIRS:
    r,null,pack=c.rsa_case(v,gm,a,b,permutations=127)
    assert r['rsa_rho']>.7 and r['same_minus_different']>1
    assert r['reader_conditional_p']<.05 and len(null)==127
    if a==b:
        mask=pack['inference_mask'];ia=pack['left_indices'];ib=pack['right_indices']
        assert not np.diag(mask).any()
        assert not (mask & (gm.reader.iloc[ia].to_numpy()[:,None]==gm.reader.iloc[ib].to_numpy()[None,:])).any()
# Null is deterministic and swaps all groups of a class together per reader.
p=c.permutation_labels(tuple(gm.reader),tuple(gm.label),17,7)
for reader,g in gm.groupby('reader'):
    for label,h in g.groupby('label'):assert all(len(set(row[h.index]))==1 for row in p)
assert np.array_equal(p,c.permutation_labels(tuple(gm.reader),tuple(gm.label),17,7))
assert np.allclose(c.corrected_p([.01,.04,.2],'holm'),[.03,.08,.2])
assert np.allclose(c.corrected_p([.01,.04,.2],'bh'),[.03,.06,.2])
tr=np.flatnonzero(folds!=0);te=np.flatnonzero(folds==0)
transform=c.SharedTransform().fit(X[tr],m.dataset.iloc[tr].to_numpy());before=transform.transform(X[tr],m.dataset.iloc[tr].to_numpy())
altered=X.copy();altered[te]*=1e6
assert np.array_equal(before,transform.transform(altered[tr],m.dataset.iloc[tr].to_numpy()))
assert c.input_view(X,'full_0_800').shape[1]==48
assert c.input_view(X,'window_200_400').shape[1]==12
assert c.input_view(X,'regional_summary').shape[1]==18
assert np.allclose(np.linalg.norm(c.input_view(X,'normalized_0_800'),axis=1),1)
# Tiny actual VAE training and resumable model/optimizer checkpoints, on cloud CPU.
with tempfile.TemporaryDirectory() as tmp:
    store=ImmutableStore(Path(tmp),50_000_000)
    tiny=X[:200];validation=X[200:240];ds=np.array(['teco','zuco']*100);vd=np.array(['teco','zuco']*20)
    checkpoints=[]
    def save(state):
        run.overwrite_state(store,Path('progress.pt'),state);checkpoints.append(len(state['history']))
    model,history=c.train_vae(tiny,validation,ds,vd,77,epochs=2,save_epoch=save)
    z=c.embeddings(model,tiny);assert z.shape==(200,8) and np.isfinite(z).all()
    assert np.array_equal(z,c.embeddings(model,tiny))
    state=run.load_torch(Path(tmp)/'progress.pt');model2,history2=c.train_vae(tiny,validation,ds,vd,77,epochs=3,resume=state)
    assert len(history2)==3 and history2[:2]==history
    diagnostic=c.latent_diagnostics(model2,tiny,ds,np.array(['a','b']*100))
    assert diagnostic['reconstruction_mse']>=0
    # End-to-end driver: all folds+all-data, two label pools, groups, plots, checksums, resume.
    results,summary=run.run(m,X,X[:,:12],{'sentiment_binary':(y,['negative','positive'])},store,
                           device='cpu',epochs=2,permutations=7,regroupings=1,inputs=['full_0_800'])
    assert len(summary)==15 and summary.completed_folds.eq(3).all()
    assert summary.holm_p.notna().all() and len(results)==60
    assert len(list((Path(tmp)/'plots').glob('*/*.png')))==5
    old=(Path(tmp)/'results.csv').read_bytes()
    again,s2=run.run(m,X,X[:,:12],{'sentiment_binary':(y,['negative','positive'])},store,
                    device='cpu',epochs=2,permutations=7,regroupings=1,inputs=['full_0_800'])
    assert (Path(tmp)/'results.csv').read_bytes()==old
print('PASS: reader isolation, exact40, declared-class coverage, reader-block RSA permutations, correction, VAE training/resume, plots and complete end-to-end resume.')
