#!/usr/bin/env python3
"""Published classification recipe on saved SST SR trials; explicit reconstruction choices.

CPU/scikit-learn, full published grid, no grouped identity holdouts. No source
extraction or preprocessing. This cannot recover the authors' exact trial set,
seeds or undocumented allocation/length-averaging/CI implementation.
"""
import argparse
from importlib.metadata import version
from pathlib import Path
import shutil,sys,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.common import digest,file_digest
from word_sentiment_eeg.paper_small_checkpoints import ImmutableStore
from word_sentiment_eeg import paper_protocol_decoding as d

SPEC=dict(version=1,dataset='ZuCo SR',labels='saved SST word labels, not agent labels',representation='cleaned',
 features='105channels x63samples0–496ms in published0–500ms interval; channel-major flattening',sizes=[40,20,10,1],
 repeats=100,test_fraction=.05,inner_folds=10,nca_components=list(d.NCA_COMPONENTS),C=list(d.REGULARIZATION),gamma=list(d.REGULARIZATION),
 nca_max_iter=100,nca='scikit-learn NCA, default auto initialization/tol; train-only in inner/outer fits',
 averaging='reader/class, sort length with random ties, distinct sentence texts per group, keep all remainders',
 split='bounded proportional reader/class marginal rounding; no reader/word/sentence holdout; unique original events',seed=2026,
 known_differences=['different retained EEG trials and Python preprocessing','undisclosed original averaging and split rounding','unknown original solver seeds/tolerances and CI formula',
 'full-recording regression cleaning precedes splitting','different fixation events can contain overlapping EEG intervals; not purged to preserve random-split recipe'],budget=500_000_000)


def read(p):return d.read(p)


def csv(store,relative,frame,work):
    path=work/Path(relative).name
    frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0} if path.suffix=='.gz' else None)
    return store.file(relative,path)


def features(parent,store,work):
    cpu=read(parent/'cpu_preparation_manifest.json')
    if cpu['status']!='completed' or read(parent/'manifest.json')['status']!='completed' or cpu['protocol_sha256']!=file_digest(parent/'protocol.json'):raise ValueError('Saved parent reproduction incomplete or changed')
    montage=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.astype(str).tolist();tables=[];xs=[]
    for i,subject in enumerate(cpu['subjects'],1):
        folder=parent/'readers'/subject;state=read(folder/'prepared.json')
        if state['status']!='prepared' or file_digest(folder/'prepared.json')!=cpu['reader_checkpoint_sha256'][subject]:raise ValueError('Parent reader changed')
        if file_digest(folder/'accepted_trial_metadata.csv.gz')!=state['checksums']['accepted_trial_metadata.csv.gz']:raise ValueError('Parent metadata changed')
        meta=pd.read_csv(folder/'accepted_trial_metadata.csv.gz',keep_default_na=False,dtype={'event_id':str});meta['subject']=subject
        if not meta.sentiment.isin(['negative','positive']).all() or meta.event_id.duplicated().any():raise ValueError('Invalid original SST labels/IDs')
        for key in ['word_length','sentence_key','word']:
            if key not in meta:raise ValueError('Missing original averaging metadata: '+key)
        meta['word_length']=pd.to_numeric(meta.word_length,errors='raise')
        if not np.isfinite(meta.word_length).all() or meta.sentence_key.eq('').any():raise ValueError('Undefined source lengths/sentences')
        rel=Path('features')/subject;mark=store.root/rel/'manifest.json';identity=digest(dict(prepared=file_digest(folder/'prepared.json'),metadata=state['checksums']['accepted_trial_metadata.csv.gz'],waveforms=state['checksums']['prepared_trials.npz'],spec=SPEC['features']))
        if mark.exists():
            saved=read(mark)
            if saved['identity']!=identity or file_digest(store.root/rel/'features.npz')!=saved['sha256']:raise ValueError('Changed feature cache')
            print(f'KEEP FEATURES {i}/{len(cpu["subjects"])} {subject}: no EEG reads',flush=True)
            with np.load(store.root/rel/'features.npz',allow_pickle=False) as z:x=z['features'];ids=z['event_id'].astype(str)
            if ids.tolist()!=meta.event_id.tolist():raise ValueError('Cached event order changed')
        else:
            print(f'LOAD SAVED EEG {i}/{len(cpu["subjects"])} {subject}: {len(meta)}trials; no extraction',flush=True)
            local=work/'reader.npz';source=folder/'prepared_trials.npz'
            if shutil.disk_usage(work).free<source.stat().st_size+500_000_000:raise OSError('Insufficient Colab temporary disk')
            shutil.copyfile(source,local)
            if file_digest(local)!=state['checksums']['prepared_trials.npz']:raise ValueError('Saved EEG checksum mismatch')
            with np.load(local,allow_pickle=False) as z:
                if z['event_ids'].astype(str).tolist()!=meta.event_id.tolist() or not np.array_equal(z['labels'],meta.sentiment.map({'negative':0,'positive':1}).to_numpy()):raise ValueError('Saved labels/rows differ')
                if z['channel_names'].astype(str).tolist()!=montage or z['representations'].astype(str).tolist()!=['raw','cleaned'] or not np.array_equal(z['times_ms'],np.arange(-75,126)*8):raise ValueError('Unexpected EEG/time/montage')
                signal=z['signal']
                if signal.shape!=(2,len(meta),105,201):raise ValueError('Unexpected waveform shape')
                x=np.ascontiguousarray(signal[1,:,:,75:138]).reshape(len(meta),6615)
                if not np.isfinite(x).all():raise ValueError('Nonfinite selected EEG')
                del signal
            local.unlink() # runtime copy only
            file=work/'features.npz';np.savez_compressed(file,features=x,event_id=meta.event_id.to_numpy(dtype=str));sha=store.file(rel/'features.npz',file)
            store.json(rel/'manifest.json',dict(identity=identity,sha256=sha,rows=len(meta),columns=6615));print(f'FEATURES SAVED {subject}: future runtimes reuse them',flush=True)
        if x.shape!=(len(meta),6615):raise ValueError('Invalid feature shape')
        tables.append(meta);xs.append(x)
    meta=pd.concat(tables,ignore_index=True)
    if meta.event_id.duplicated().any():raise ValueError('Repeated event across readers')
    return np.concatenate(xs),meta


def size_features(x,meta,size,store,work):
    rel=Path('averages')/f'size{size:02d}';mark=store.root/rel/'manifest.json';identity=digest(dict(size=size,spec=SPEC,events=meta.event_id.tolist()))
    if mark.exists():
        saved=read(mark)
        if saved['identity']!=identity:raise ValueError('Changed averages')
        for f,sha in saved['checksums'].items():
            if file_digest(store.root/rel/f)!=sha:raise ValueError('Corrupt saved averages')
        a=pd.read_csv(store.root/rel/'metadata.csv.gz',keep_default_na=False);m=pd.read_csv(store.root/rel/'membership.csv.gz',keep_default_na=False)
        if size==1:z=x
        else:
            with np.load(store.root/rel/'features.npz',allow_pickle=False) as f:z=f['features']
        return z,a,m
    if size==1:
        z=x;a=meta[['subject','sentiment']].copy();a['average_id']=[f'average_{i:06d}' for i in range(len(a))];a['actual_trials']=1;a['is_remainder']=False
        a['min_word_length']=meta.word_length;a['max_word_length']=meta.word_length
        m=meta[['subject','sentiment','event_id','sentence_key','word']].copy();m['average_id']=a.average_id
    else:z,a,m=d.averages(x,meta,size)
    checks={}
    for name,frame in [('metadata.csv.gz',a),('membership.csv.gz',m)]:checks[name]=csv(store,rel/name,frame,work)
    if size!=1:
        path=work/'features.npz';np.savez_compressed(path,features=z);checks['features.npz']=store.file(rel/'features.npz',path)
    store.json(rel/'manifest.json',dict(identity=identity,checksums=checks,examples=len(a),original_events=len(m),remainders=int(a.is_remainder.sum()),
        requested_size=size,actual_min=int(a.actual_trials.min()),actual_max=int(a.actual_trials.max()),features_reference='reader feature caches in original order' if size==1 else 'features.npz'))
    return z,a,m


def report(out,store=None,work=None):
    rows=[]
    for path in sorted(out.glob('decoding/size*/repeat*/result.json')):
        value=read(path);size=int(path.parents[1].name.removeprefix('size'))
        for score in value['scores']:rows.append(dict(size=size,**score))
    if not rows:print('No completed random split yet. Saved feature/grid blocks are reusable.',flush=True);return pd.DataFrame()
    summary=d.summary(rows);print(summary.to_string(index=False),flush=True)
    if store is not None:
        csv(store,'summary.csv',summary,work)
        flat=pd.DataFrame([{k:s.get(k) for k in ['size','model','repeat','accuracy','balanced_accuracy','f1','precision','recall','test_examples']} for s in rows]);csv(store,'scores.csv',flat,work)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        for name in ['logistic','svm_rbf','dummy']:
            g=summary[summary.model.eq(name)].sort_values('size');ax.plot(g['size'],100*g.mean_accuracy,'o-',label=name)
        ax.plot([1,10,20,40],[50,53,55,60],'--',color='#148f77',label='Paper logistic accuracy')
        ax.plot([1,10,20,40],[50,51,54,57],'--',color='#a76b2d',label='Paper SVM accuracy')
        ax.set(xlabel='Maximum trials per average (remainders retained)',ylabel='Mean test accuracy (%)',title='Random95/5splits: Python protocol reconstruction');ax.legend(fontsize=8);fig.tight_layout()
        path=work/'classification_comparison.png';fig.savefig(path,dpi=160);plt.close(fig);store.file('classification_comparison.png',path)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--free-drive-bytes',type=int,default=0);p.add_argument('--phase',choices=['prepare','decode','report'],required=True);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Colab/cloud only; no local EEG/dependency downloads or training')
    base=args.thesis_root/'Results/word_sentiment_eeg';parent=base/'zuco_paper_figures_recovered';out=base/'zuco_paper_classification'
    if args.phase=='report':report(out);return
    store=ImmutableStore(out,SPEC['budget'])
    if args.free_drive_bytes<max(0,SPEC['budget']-store.used):raise OSError('Reserve up to500MB for all persistent features, averages and grid checkpoints')
    protocol=dict(spec=SPEC,parent_cpu=file_digest(parent/'cpu_preparation_manifest.json'),parent_manifest=file_digest(parent/'manifest.json'),
        code={f:file_digest(ROOT/f) for f in ['scripts/zuco_paper_classification.py','src/word_sentiment_eeg/paper_protocol_decoding.py','src/word_sentiment_eeg/paper_decoding.py','src/word_sentiment_eeg/paper_small_checkpoints.py','requirements-paper-classification.txt']},
        software={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']})
    store.json('protocol.json',protocol)
    if (out/'complete.json').exists():
        for f,sha in read(out/'complete.json')['checksums'].items():
            if file_digest(out/f)!=sha:raise ValueError('Completed classification outputs changed')
        print('KEEP COMPLETE: no EEG loading or fitting',flush=True);return
    print('FULL PLAN: 100random splits ×10inner folds ×8NCA dimensions; LR10C and SVM10C×10gamma; all4averaging sizes.',flush=True)
    print('3,520,800 classifier fits including final refits; NCA shared across parameters gives up to32,800NCA fits. This is a long CPU job, not a short sweep.',flush=True)
    with tempfile.TemporaryDirectory(prefix='paper-classification-') as tmp:
        work=Path(tmp);x,meta=features(parent,store,work)
        for size in SPEC['sizes']:
            z,a,m=size_features(x,meta,size,store,work)
            store.json(f'averages/size{size:02d}/split_allocation_example.json',d.random_split(a,2026)[2])
            if args.phase=='decode':
                for repeat in range(SPEC['repeats']):
                    print(f'CLASSIFY size{size}: random split{repeat+1}/100',flush=True)
                    d.run_repeat(z,a,m,store,Path('decoding')/f'size{size:02d}'/f'repeat{repeat:03d}',repeat,nca_max_iter=SPEC['nca_max_iter'])
            del z
        if args.phase=='prepare':store.json('preparation_complete.json',dict(status='completed',source_trials=len(meta),sizes=SPEC['sizes']));print('PREPARATION COMPLETE: saved features/averages ready; no EEG reload needed',flush=True);return
        report(out,store,work)
        store.json('complete.json',dict(status='completed',checksums={str(f.relative_to(out)):file_digest(f) for f in out.rglob('*') if f.is_file() and f.name not in {'complete.json','runtime_status.json'}}))
    print('PAPER CLASSIFICATION RECONSTRUCTION COMPLETE:',out,flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
