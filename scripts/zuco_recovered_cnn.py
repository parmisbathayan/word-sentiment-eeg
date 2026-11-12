"""Secondary CNN; grouped inner validation, fixed settings, compact checkpoints."""
import io
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from zuco_recovered_followups import SEED,grouped_folds,metrics,read

CNN_SPEC=dict(model='independent compact EEGNet-inspired 8/2/16',epochs=25,patience=4,
    checkpoint_every=5,batch=128,learning_rate=.001,weight_decay=.0001,dropout=.5,
    ms=[0,800],samples=100,classes=['negative','positive'],splits=['sentence','word','reader'],
    preprocessing='inner-fit channel mean/SD; full-reader source preprocessing remains transductive',seed=SEED)

class SmallNet(nn.Module):
    def __init__(self,channels=105,samples=100):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,8,(1,33),padding=(0,16),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(channels,1),groups=8,bias=False),nn.BatchNorm2d(16),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(.5),
            nn.Conv2d(16,16,(1,17),padding=(0,8),groups=16,bias=False),nn.Conv2d(16,16,1,bias=False),
            nn.BatchNorm2d(16),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.5),nn.Flatten(),
            nn.Linear(16*(samples//4//4),2))
    def forward(self,x):return self.net(x[:,None])

def seed_all(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def predict(model,data,ix,batch=128):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(ix),batch):out.append(model(data[ix[start:start+batch]]).softmax(1)[:,1].cpu().numpy())
    return np.concatenate(out)

def fit_fold(data,y,meta,mode,train,test,store,relative,fi,device='cuda'):
    group=meta[{'sentence':'sentence_key','word':'word_key','reader':'subject'}[mode]].astype(str).to_numpy()
    inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+fi).split(np.zeros(len(train)),y[train],group[train]))
    a,b=inner[0];fit,valid=train[a],train[b]
    if set(group[fit])&set(group[valid]) or set(group[train])&set(group[test]) or min(len(set(y[fit])),len(set(y[valid])),len(set(y[test])))<2:raise ValueError('Invalid CNN group split')
    seed_all(SEED+fi);model=SmallNet(data.shape[1],data.shape[2]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CNN_SPEC['learning_rate'],weight_decay=CNN_SPEC['weight_decay'])
    weights=np.bincount(y[fit],minlength=2);lossfn=nn.CrossEntropyLoss(weight=torch.tensor(len(fit)/(2*weights),device=device,dtype=torch.float32))
    mean=np.mean(data[fit],axis=(0,2),keepdims=True,dtype=np.float64).astype(np.float32)
    std=np.std(data[fit],axis=(0,2),keepdims=True,dtype=np.float64).astype(np.float32);std=np.maximum(std,1e-6)
    scaled=torch.as_tensor((data-mean)/std,device=device);target=torch.as_tensor(y,device=device)
    best=-1.;best_weights=None;stale=0;start=0;history=[]
    frames=sorted((store.root/relative).glob('epoch_*.pt')) if (store.root/relative).exists() else []
    from word_sentiment_eeg.common import digest,file_digest
    committed=[]
    for frame in frames:
        marker=store.root/'analysis_state/published'/(digest(str(frame.relative_to(store.root)))+'.json')
        if marker.exists():
            if file_digest(frame)!=read(marker)['sha256']:raise ValueError('CNN frame corrupted')
            committed.append(frame)
    if committed:
        state=torch.load(committed[-1],map_location=device,weights_only=False)
        if state['fold_identity']!=digest([fit.tolist(),valid.tolist(),test.tolist(),mode,CNN_SPEC]):raise ValueError('CNN frame split changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['epoch']
        best=state['best'];best_weights=state['best_weights'];stale=state['stale'];history=state['history']
        torch.set_rng_state(state['rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        print(f'RESUME CNN {relative}: epoch {start}',flush=True)
    for epoch in range(start,CNN_SPEC['epochs']):
        if stale>=CNN_SPEC['patience']:break
        model.train();order=np.random.default_rng(SEED+fi*100+epoch).permutation(fit)
        for j in range(0,len(order),CNN_SPEC['batch']):
            ix=order[j:j+CNN_SPEC['batch']];optimizer.zero_grad(set_to_none=True)
            loss=lossfn(model(scaled[ix]),target[ix]);loss.backward();optimizer.step()
        scores=predict(model,scaled,valid)
        ba=float(np.mean([(scores[y[valid]==c]>=.5).mean() if c else (scores[y[valid]==c]<.5).mean() for c in [0,1]]))
        history.append(dict(epoch=epoch+1,validation_balanced_accuracy=ba))
        if ba>best+1e-6:
            best=ba;stale=0;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        print(f'CNN {relative} epoch {epoch+1}/25 validation BA={ba:.3f}',flush=True)
        if (epoch+1)%CNN_SPEC['checkpoint_every']==0 or stale>=CNN_SPEC['patience'] or epoch+1==CNN_SPEC['epochs']:
            state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                epoch=epoch+1,best=best,best_weights=best_weights,stale=stale,history=history,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                fold_identity=digest([fit.tolist(),valid.tolist(),test.tolist(),mode,CNN_SPEC]))
            payload=io.BytesIO();torch.save(state,payload)
            local=Path('/tmp')/f'zuco-cnn-{fi}.pt';local.write_bytes(payload.getvalue())
            store.file(relative/f'epoch_{epoch+1:03d}.pt',local);local.unlink()
    model.load_state_dict(best_weights);scores=predict(model,scaled,test);pred=(scores>=.5).astype(int)
    result=dict(status='completed',fold=fi+1,fit_trials=len(fit),validation_trials=len(valid),test_trials=len(test),
        best_validation_balanced_accuracy=best,epochs=len(history),history=history,
        group_overlap=0,train_event_digest=digest(meta.iloc[train].event_id.tolist()),test_event_digest=digest(meta.iloc[test].event_id.tolist()),
        **metrics(y[test],pred,scores))
    del model,optimizer,scaled,target
    if device=='cuda':torch.cuda.empty_cache()
    return result

def run_cnn(x,meta,store,work):
    if not torch.cuda.is_available():raise RuntimeError('Select T4 GPU for CNN; diagnostics/linear models use CPU')
    store.json('cnn/settings.json',dict(**CNN_SPEC,torch_version=torch.__version__,cuda_version=torch.version.cuda))
    y=meta.sentiment.map({'negative':0,'positive':1}).to_numpy();results=[];times=np.arange(-75,126)*8
    for r,rep in enumerate(['raw','cleaned']):
        data=x[r][:,:,(times>=0)&(times<800)].copy()
        for mode in CNN_SPEC['splits']:
            groups=meta[{'sentence':'sentence_key','word':'word_key','reader':'subject'}[mode]].to_numpy()
            for fi,(train,test) in enumerate(grouped_folds(y,groups)):
                relative=Path('cnn')/rep/mode/f'fold_{fi+1}';p=store.root/relative/'result.json'
                if p.exists():result=read(p);print('KEEP CNN '+str(relative),flush=True)
                else:
                    result=fit_fold(data,y,meta,mode,train,test,store,relative,fi)
                    store.json(relative/'result.json',result)
                results.append(dict(representation=rep,split=mode,**result))
        del data
    store.json('cnn/summary.json',results)
