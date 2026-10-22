"""Cloud-only old/new waveform comparison on a small real ZuCo sample.

The source file is temporary. Only the small audit JSON is retained. This is
separate from, and never changes, the user's Drive checkpoints or labels.
"""
import importlib.util
from itertools import islice
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import run_erp as current
from word_sentiment_eeg import erp_checkpoint as checkpoint
from word_sentiment_eeg.common import file_digest, write_json
from word_sentiment_eeg.erp import CLASSES, iter_sentences


def main():
    if sys.platform=='darwin':
        raise SystemExit('This verification downloads EEG. Run only in the cloud.')
    records=json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files']
    record=next(r for r in records if (r['task'],r['subject'])==('NR','ZJS'))
    annotations=pd.read_csv(ROOT/'annotations/word_labels.csv',keep_default_na=False)
    channels=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv').labels.tolist()
    reference_url='https://raw.githubusercontent.com/parmisbathayan/word-sentiment-eeg/c0d40db/run_erp.py'
    with urllib.request.urlopen(reference_url,timeout=30) as response:
        reference_source=response.read()
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        reference_path=folder/'legacy_runner.py'
        reference_path.write_bytes(reference_source)
        spec=importlib.util.spec_from_file_location('legacy_erp_reference',reference_path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        path=folder/record['filename']
        print('Download one real source into cloud temporary storage:',record['filename'],flush=True)
        current.cloud_download(record,path)
        reader=iter_sentences(path)
        try:
            sample=list(islice(reader,10))
        finally:
            reader.close()
        old_target,new_target=folder/'old',folder/'new'
        print('Compare the original and resume-safe runners on 10 real sentences',flush=True)
        with patch.object(old,'iter_sentences',lambda _:iter(sample)):
            a=old.process_file(path,record,annotations,old_target,'real-audit',channels)
        with patch.object(current,'iter_sentences',lambda _:iter(sample)):
            b=current.process_file(path,record,annotations,new_target,'real-audit',channels)
        assert a['epochs_per_class']==b['epochs_per_class']
        events_a=pd.read_csv(old_target/'word_onsets.csv.gz',keep_default_na=False)
        events_b=pd.read_csv(new_target/'word_onsets.csv.gz',keep_default_na=False)
        pd.testing.assert_frame_equal(events_a,events_b)
        counts={}
        for cls in CLASSES:
            original=checkpoint.load_class_epochs(old_target,cls)
            updated=checkpoint.load_class_epochs(new_target,cls)
            np.testing.assert_array_equal(original,updated)
            counts[cls]=len(original)
        with np.load(old_target/'class_means.npz') as aa,np.load(new_target/'class_means.npz') as bb:
            mask=np.isfinite(aa['mean_uV']) & np.isfinite(bb['mean_uV'])
            delta=float(np.abs(aa['mean_uV'][mask]-bb['mean_uV'][mask]).max()) if mask.any() else 0.
            np.testing.assert_allclose(aa['mean_uV'],bb['mean_uV'],rtol=2e-6,atol=1e-5,equal_nan=True)
        # Repair one real neutral chunk. Preserve means, onsets and other classes.
        neutral_before=checkpoint.load_class_epochs(new_target,'neutral')
        existing=[new_target/'class_means.npz',new_target/'word_onsets.csv.gz',
                  *list((new_target/'positive_epochs').glob('*.npz')),
                  *list((new_target/'negative_epochs').glob('*.npz'))]
        before={str(f):file_digest(f) for f in existing}
        missing=next((new_target/'neutral_epochs').glob('*.npz'))
        missing.unlink()
        cp=checkpoint.inspect_checkpoint(new_target,record,{'real-audit'},channels)
        with patch.object(checkpoint,'iter_sentences',lambda _:iter(sample)):
            checkpoint.repair_trial_exports(path,cp,channels,a['source_sha256'])
        np.testing.assert_array_equal(checkpoint.load_class_epochs(new_target,'neutral'),neutral_before)
        for name,sha in before.items():
            assert file_digest(name)==sha,name
        result={'source':record['filename'],'source_url':record['url'],'source_sha256':a['source_sha256'],
                'scope':'10 real readable NR/ZJS sentences, not all readers',
                'old_runner_reference':reference_url,'sentences':len(sample),'word_rows':len(events_a),
                'accepted_epochs':counts,'trial_waveforms_bitwise_identical':True,
                'onsets_labels_exclusions_identical':True,'max_mean_rounding_difference_uV':delta,
                'missing_neutral_chunk_repaired_identically':True,'existing_files_unchanged':True,
                'user_drive_modified':False,'status':'passed'}
        write_json(ROOT/'real_erp_compatibility.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
