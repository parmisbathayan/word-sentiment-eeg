"""Cloud-only SciPy/HDF5 reader and full ERP-run integration checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import sentence_key
from word_sentiment_eeg.erp import iter_sentences


with tempfile.TemporaryDirectory() as tmp:
    directory=Path(tmp)
    rng=np.random.default_rng(412)
    raw=rng.normal(size=(105,5000)).astype(np.float32)
    raw[104]=0
    sentences=[];annotations=[]
    for s in range(6):
        tokens=[f'bad{s}',f'plain{s}',f'good{s}',f'ambig{s}',f'negated{s}',f'lostfirst{s}']
        words=[]
        for j,token in enumerate(tokens):
            start=500+j*600
            segment=raw[:,start:start+70]
            if j==5:
                cells=np.empty(2,dtype=object);cells[0]=np.nan;cells[1]=segment
                segment=cells
            words.append({'content':token,'rawEEG':segment})
            cls=['negative','neutral','positive','uncertain','positive','positive'][j]
            annotations.append({'sentence_key':sentence_key(tokens),'word_index':j,'word':token,'tasks':'SR;NR',
                'lexical_class':cls,'contextual_class':'negative' if j==4 else cls,'alignment_warning':'',
                'annotation_status':'agent_reviewed_not_human_gold'})
        sentences.append({'content':' '.join(tokens),'word':words,'rawData':raw})
    rawdir=directory/'Data/zuco_og_raw';rawdir.mkdir(parents=True)
    source=rawdir/'resultsZAB_SR.mat'
    savemat(source,{'sentenceData':sentences})
    ann=directory/'labels.csv';pd.DataFrame(annotations).to_csv(ann,index=False)
    # Additional v7.3 reference schema, including invalid first cell.
    hdf=rawdir/'resultsZDM_NR.mat'
    with h5py.File(hdf,'w') as f:
        sd=f.create_group('sentenceData');wg=f.create_group('w0')
        tokens=[str(w['content']) for w in sentences[0]['word']]
        texts=[];segments=[]
        for j,token in enumerate(tokens):
            texts.append(f.create_dataset(f'text{j}',data=np.array([ord(c) for c in token],np.uint16)).ref)
            actual=f.create_dataset(f'seg{j}',data=raw[:,500+j*600:570+j*600].T)
            if j==5:
                bad=f.create_dataset('invalid',data=np.array([[np.nan]]))
                cell=f.create_dataset('cell',data=np.array([bad.ref,actual.ref],dtype=h5py.ref_dtype))
                segments.append(cell.ref)
            else:
                segments.append(actual.ref)
        wg.create_dataset('content',data=np.array(texts,dtype=h5py.ref_dtype))
        wg.create_dataset('rawEEG',data=np.array(segments,dtype=h5py.ref_dtype))
        sd.create_dataset('word',data=np.array([wg.ref],dtype=h5py.ref_dtype))
        rd=f.create_dataset('raw',data=raw.T)
        sd.create_dataset('rawData',data=np.array([rd.ref],dtype=h5py.ref_dtype))
    parsed=list(iter_sentences(hdf))
    assert parsed[0][3][5][0] is None
    assert parsed[0][3][5][1].shape==(105,70)
    records=[{'task':'SR','subject':'ZAB','filename':source.name,'bytes':source.stat().st_size,'url':'fixture'},
             {'task':'NR','subject':'ZDM','filename':hdf.name,'bytes':hdf.stat().st_size,'url':'fixture'}]
    manifest=directory/'sources.json';manifest.write_text(json.dumps({'files':records}))
    command=[sys.executable,str(ROOT/'run_erp.py'),'--thesis-root',str(directory),'--annotations',str(ann),'--source-manifest',str(manifest)]
    subprocess.run(command,check=True)
    run=directory/'Results/word_sentiment_eeg/v3_word_erp'
    result=json.loads((run/'manifest.json').read_text())
    assert result['status']=='completed',result
    sr=result['files'][0]
    assert sr['epochs_per_class']=={'negative':6,'neutral':6,'positive':6},sr
    events=pd.read_csv(Path(sr['output_dir'])/'word_onsets.csv.gz')
    assert sum(events.status=='accepted')==18
    assert sum(events.status=='excluded_uncertain_or_context_disagreement')==12
    assert sum(events.status=='excluded_missing_or_invalid_fixation')==6
    assert set(events.loc[events.status=='accepted','onset_sample_sentence_0based'])=={500,1100,1700}
    assert (run/'plots/SR_class_erp.png').exists()
    means=np.load(run/'SR_grand_average.npz',allow_pickle=False)
    assert means['mean_uV'].shape==(3,105,501)
    # Same first fixation in a different MATLAB format produces identical epochs.
    nr=result['files'][1]
    a=np.load(Path(sr['output_dir'])/'positive_epochs.npz')['eeg_uV'][0]
    b=np.load(Path(nr['output_dir'])/'positive_epochs.npz')['eeg_uV'][0]
    np.testing.assert_allclose(a,b)
    subprocess.run(command,check=True)  # checkpoint reuse, no download
print('ERP v5/v7.3 extraction, first-fixation exclusions, onset samples, averaging and resume verified.')
