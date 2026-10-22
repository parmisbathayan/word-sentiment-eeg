"""Cloud-only SciPy/HDF5 reader and full ERP-run integration checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import sentence_key
from word_sentiment_eeg.erp import iter_sentences
from word_sentiment_eeg.erp_checkpoint import load_class_epochs
from word_sentiment_eeg.common import file_digest, write_json
sys.path.insert(0,str(ROOT))
from run_erp import cache_fingerprints


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
    source_bytes=source.read_bytes()
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
    hdf_bytes=hdf.read_bytes()
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
    a=load_class_epochs(sr['output_dir'],'positive')[0]
    b=load_class_epochs(nr['output_dir'],'positive')[0]
    np.testing.assert_allclose(a,b)
    # Restart skips every saved reader: no raw MATLAB source is required.
    for path in rawdir.glob('*.mat'):
        path.unlink()
    resumed=subprocess.run(command,check=True,capture_output=True,text=True)
    assert '2/2 analyses already saved' in resumed.stdout,resumed.stdout
    assert 'extract_new_reader' not in resumed.stdout,resumed.stdout
    assert json.loads((run/'manifest.json').read_text())['processed_files']==2
    # Recognize the old cache namespace/fingerprint rather than starting a new
    # extraction when orchestration code changes.
    current,legacy=cache_fingerprints(ann,manifest)
    cache=directory/'CachedArtifacts/word_sentiment_eeg/v3_word_erp'
    shutil.move(str(cache/current[:16]),str(cache/legacy[:16]))
    for entry in result['files']:
        target=cache/legacy[:16]/entry['task']/entry['subject']
        saved=json.loads((target/'manifest.json').read_text())
        saved['fingerprint']=legacy
        write_json(target/'manifest.json',saved)
    migrated=subprocess.run(command,check=True,capture_output=True,text=True)
    assert '2/2 analyses already saved' in migrated.stdout
    assert 'extract_new_reader' not in migrated.stdout
    # Rebuild only a missing neutral export using cached onset samples. The
    # original source bytes are restored, since repair verifies source identity.
    target=cache/legacy[:16]/'SR/ZAB'
    neutral_before=load_class_epochs(target,'neutral').copy()
    preserved=[target/'class_means.npz',target/'word_onsets.csv.gz',
               *list((target/'positive_epochs').glob('*.npz')),
               *list((target/'negative_epochs').glob('*.npz'))]
    before={str(p):file_digest(p) for p in preserved}
    missing=next((target/'neutral_epochs').glob('*.npz'))
    missing.unlink()
    source.write_bytes(source_bytes)
    repaired=subprocess.run(command,check=True,capture_output=True,text=True)
    assert 'repair_trial_exports SR/ZAB' in repaired.stdout,repaired.stdout
    assert 'extract_new_reader' not in repaired.stdout,repaired.stdout
    assert '2/2 analyses already saved' in repaired.stdout
    for path,sha in before.items():
        assert file_digest(path)==sha,path
    np.testing.assert_array_equal(load_class_epochs(target,'neutral'),neutral_before)
    final=json.loads((run/'manifest.json').read_text())
    assert final['status']=='completed' and final['complete_trial_export_files']==2,final
    # Reproduce the real failure: compatible manifests but missing averages or
    # onset tables. Preserve every surviving old file, rebuild each reader once
    # in a separate deterministic location, then reuse that location on rerun.
    old_nr=cache/legacy[:16]/'NR/ZDM'
    (target/'class_means.npz').unlink()
    (old_nr/'word_onsets.csv.gz').unlink()
    survivors={p:file_digest(p) for folder in (target,old_nr)
               for p in folder.rglob('*') if p.is_file()}
    hdf.write_bytes(hdf_bytes)
    recovered=subprocess.run(command,check=True,capture_output=True,text=True)
    assert recovered.stdout.count('STAGE rebuild_missing_analysis')==2,recovered.stdout
    recovered_state=json.loads((run/'manifest.json').read_text())
    assert recovered_state['status']=='completed',recovered_state
    assert recovered_state['processed_files']==2 and recovered_state['complete_trial_export_files']==2
    assert len({(i['task'],i['subject']) for i in recovered_state['files']})==2
    for p,sha in survivors.items():
        assert file_digest(p)==sha,p
    for entry in recovered_state['files']:
        assert '/recovered/' in entry['output_dir'],entry
    np.testing.assert_array_equal(load_class_epochs(recovered_state['files'][0]['output_dir'],'neutral'),neutral_before)
    recovered_hashes={p:file_digest(p) for entry in recovered_state['files']
                      for p in Path(entry['output_dir']).rglob('*') if p.is_file()}
    source.unlink();hdf.unlink()
    again=subprocess.run(command,check=True,capture_output=True,text=True)
    assert '2/2 analyses already saved' in again.stdout,again.stdout
    assert 'STAGE ' not in again.stdout,again.stdout
    for p,sha in recovered_hashes.items():
        assert file_digest(p)==sha,p
    # Compact output reuses full-run analyses without requiring source files or
    # copying any large trial data. It is independently resumable.
    compact_command=[sys.executable,str(ROOT/'run_compact_erp.py'),'--thesis-root',str(directory),
                     '--annotations',str(ann),'--source-manifest',str(manifest)]
    compact=subprocess.run(compact_command,check=True,capture_output=True,text=True)
    compact_run=directory/'Results/word_sentiment_eeg/v4_word_erp_compact'
    compact_state=json.loads((compact_run/'manifest.json').read_text())
    assert compact_state['status']=='completed' and compact_state['processed_files']==2,compact_state
    assert compact_state['output_policy']=='onsets_and_class_averages'
    assert not list(compact_run.rglob('*_epochs*'))
    assert (compact_run/'compact_results.zip').exists()
    compact_again=subprocess.run(compact_command,check=True,capture_output=True,text=True)
    assert 'EXTRACT ' not in compact_again.stdout,compact_again.stdout
    # Audit actual saved bytes, annotation/event identity, unique observations,
    # and ZIP consistency without reopening source EEG or altering results.
    sys.path.insert(0,str(ROOT/'scripts'))
    from audit_subject_erps import audit
    audit_originals={p:file_digest(p) for p in compact_run.rglob('*') if p.is_file()}
    checked=audit(compact_run,make_plots=False)
    assert checked['status']=='passed' and checked['reader_task_records']==2,checked
    selected=pd.read_csv(compact_run/'audit/word_selection.csv')
    assert selected.accepted_eeg_observations.sum()==checked['accepted_epochs']
    assert (selected.loc[selected.accepted_eeg_observations>0,'label_eligible']).all()
    assert not selected.loc[selected.word=='negated0','label_eligible'].any()
    from review_regions import review_regions, REGIONS
    mapping,regional=review_regions(compact_run,make_plots=True)
    assert len(mapping)==105 and set(mapping.region)==set(REGIONS)
    with np.load(compact_run/'SR_grand_average.npz') as saved:
        channels=mapping.index[mapping.region=='anterior'].tolist()
        expected=saved['subject_means_uV'][:,:,channels,:].mean(axis=2)
        np.testing.assert_allclose(regional['SR'][1]['anterior'][0],expected)
    assert (compact_run/'plots/regions/SR_regions.png').stat().st_size>1000
    assert (compact_run/'plots/regions/subjects/SR_ZAB.png').stat().st_size>1000
    region_table=pd.read_csv(compact_run/'audit/regions/subject_window_means.csv')
    # The fixture's NR reader has <5 accepted epochs/class and must remain
    # excluded from comparable group curves, exactly as in the original run.
    assert set(regional)=={'SR'}
    assert len(region_table)==len(regional)*6*3*4
    for p,sha in audit_originals.items():
        assert file_digest(p)==sha,p
    onset_file=compact_run/'readers/SR/ZAB/word_onsets.csv.gz'
    onset_bytes=onset_file.read_bytes()
    corrupt=pd.read_csv(onset_file)
    corrupt.loc[1,'event_id']=corrupt.loc[0,'event_id']
    corrupt.to_csv(onset_file,index=False,compression='gzip')
    try:
        audit(compact_run,make_plots=False)
    except ValueError as error:
        assert 'duplicate event IDs' in str(error),error
    else:
        raise AssertionError('Audit accepted duplicate event IDs')
    onset_file.write_bytes(onset_bytes)
    for p,sha in recovered_hashes.items():
        assert file_digest(p)==sha,p
    # Fresh compact extraction uses exactly the same accepted events and ERP
    # means as full extraction, but writes no individual EEG trial archives.
    source.write_bytes(source_bytes);hdf.write_bytes(hdf_bytes)
    fresh_command=compact_command+['--run-tag','fresh_compact','--reuse-run-tag','no_old_cache']
    fresh=subprocess.run(fresh_command,check=True,capture_output=True,text=True)
    fresh_run=directory/'Results/word_sentiment_eeg/fresh_compact'
    fresh_state=json.loads((fresh_run/'manifest.json').read_text())
    assert fresh_state['status']=='completed',fresh_state
    assert not list(fresh_run.rglob('*_epochs*'))
    for old_item,new_item in zip(recovered_state['files'],fresh_state['files']):
        with np.load(Path(old_item['output_dir'])/'class_means.npz') as old_mean, np.load(Path(new_item['output_dir'])/'class_means.npz') as new_mean:
            np.testing.assert_array_equal(old_mean['mean_uV'],new_mean['mean_uV'])
        pd.testing.assert_frame_equal(pd.read_csv(Path(old_item['output_dir'])/'word_onsets.csv.gz'),
                                      pd.read_csv(Path(new_item['output_dir'])/'word_onsets.csv.gz'))
print('ERP v5/v7.3 extraction, first-fixation exclusions, onset samples, averaging and resume verified.')
