"""Lossless CPU/GPU handoff, restart, identity and storage safety checks."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import atomic_npz
from word_sentiment_eeg.paper_stage_archive import save_trials,mark_prepared,verify_prepared,finish_reader


def main():
    rng=np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);reader=root/'readers/A';reader.mkdir(parents=True)
        signal=rng.normal(size=(2,8,105,201)).astype('float32')
        meta=pd.DataFrame(dict(event_id=[f'event{i}' for i in range(8)],sentiment=['negative']*4+['positive']*4,
            word=['bad']*4+['good']*4,sentence_key=[str(i) for i in range(8)],word_index=np.arange(8),subject='A'))
        neg=np.arange(4);pos=np.arange(4,8)
        save_trials(reader,signal,meta,neg,pos,root,20_000_000)
        with np.load(reader/'prepared_trials.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(signal,z['signal']);np.testing.assert_array_equal(meta.event_id,z['event_ids'])
            assert len(z['channel_names'])==105 and int(z['sample_rate_hz'])==125
            assert z['times_ms'][0]==-600 and z['times_ms'][-1]==1000
        d=signal[:,pos]-signal[:,neg]
        atomic_npz(reader/'statistics.npz',pair_sum=d.sum(axis=1,dtype=np.float64),pair_squares=(d.astype(np.float64)**2).sum(axis=1))
        mark_prepared(reader,'A',[4,4],4);verify_prepared(reader)
        finish_reader(reader,root/'null_work',7,5,use_gpu=False)
        snapshots={p.name:p.read_bytes() for p in (root/'null_work').iterdir()}
        # Simulates a fresh process/runtime: only persistent files enter this call.
        finish_reader(reader,root/'null_work',7,5,use_gpu=False)
        assert snapshots=={p.name:p.read_bytes() for p in (root/'null_work').iterdir()}
        for p in (root/'null_work').glob('*.json'):assert json.loads(p.read_text())['subjects']==['A']
        assert json.loads((reader/'manifest.json').read_text())['status']=='completed'
        original=(reader/'accepted_trial_metadata.csv.gz').read_bytes()
        (reader/'accepted_trial_metadata.csv.gz').write_bytes(original+b'corrupt')
        try:verify_prepared(reader)
        except ValueError:pass
        else:raise AssertionError('Changed metadata accepted')
        (reader/'accepted_trial_metadata.csv.gz').write_bytes(original)
        (reader/'blocked').mkdir()
        try:save_trials(reader/'blocked',signal,meta,neg,pos,root,1)
        except OSError:pass
        else:raise AssertionError('Storage cap ignored')
        assert not (reader/'blocked/prepared_trials.npz').exists()
    print('CPU/GPU handoff passed: lossless signals, full identities, runtime restart, no duplicate permutations, corruption detection and storage cap')


if __name__=='__main__':main()


def endpoint_recovery_tests():
    import copy
    from types import SimpleNamespace
    sys.path.insert(0,str(ROOT/'scripts'))
    from prepare_zuco_paper_figures import repair_eye_duration_columns
    from word_sentiment_eeg.common import write_json,file_digest
    from word_sentiment_eeg.paper_stage_archive import migrate_endpoint_protocol,LEGACY_ENDPOINT_CODE
    def eye_record(duration=2,end=100):
        event=SimpleNamespace(colheader=['latency','endtime','duration'],data=np.array([[100.,float(end),float(duration)]]))
        return {'data':np.c_[np.arange(0.,20.,2.)], 'eyeevent':SimpleNamespace(blinks=event,
            fixations=SimpleNamespace(colheader=[],data=np.array([])),saccades=SimpleNamespace(colheader=[],data=np.array([])))}
    eye=eye_record();audit=repair_eye_duration_columns(eye)
    assert eye['eyeevent'].blinks.data[0,2]==2 and audit['blinks']['single_sample_blinks_retained']==1
    for eye in [eye_record(duration=3),eye_record(end=98),eye_record(duration=1e14)]:
        try:repair_eye_duration_columns(eye)
        except ValueError:pass
        else:raise AssertionError('Invalid blink accepted')
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);reader=root/'readers/ZAB';reader.mkdir(parents=True)
        signal=np.ones((2,4,105,201),np.float32)
        meta=pd.DataFrame(dict(event_id=['a','b','c','d'],sentiment=['negative']*2+['positive']*2))
        save_trials(reader,signal,meta,[0,1],[2,3],root,10_000_000)
        atomic_npz(reader/'statistics.npz',counts=np.array([2,2]))
        events={'blinks':dict(events=2,minimum_recomputed_ms=20,maximum_recomputed_ms=100)}
        write_json(reader/'source_mapping.json',dict(blocks=[dict(duration_endpoint_audit=dict(raw=events,corrected=events))]))
        mark_prepared(reader,'ZAB',[2,2],2);before={p.name:file_digest(p) for p in reader.iterdir()}
        old=dict(workflow='cpu_gpu_stages_v1',software={'numpy':'same'},subjects=['ZAB','ZDM'],code=dict(LEGACY_ENDPOINT_CODE,unchanged='abc'))
        new=copy.deepcopy(old)
        for key in LEGACY_ENDPOINT_CODE:new['code'][key]='new-implementation'
        migrate_endpoint_protocol(old,new,root)
        assert before=={p.name:file_digest(p) for p in reader.iterdir()}
        assert json.loads((root/'endpoint_recovery.json').read_text())['preserved_readers'][0]['subject']=='ZAB'
        for changed in ['method','environment','unknown_code','corrupted_reader']:
            invalid=copy.deepcopy(old)
            if changed=='method':invalid['subjects']=['ZAB']
            elif changed=='environment':invalid['software']['numpy']='different'
            elif changed=='unknown_code':invalid['code']['unchanged']='changed'
            else:(reader/'statistics.npz').write_bytes(b'corrupt')
            try:migrate_endpoint_protocol(invalid,new,root)
            except ValueError:pass
            else:raise AssertionError('Unsafe checkpoint migration accepted: '+changed)
    print('Blink endpoint recovery and exact legacy migration checks passed; prepared reader bytes preserved')


if __name__=='__main__':endpoint_recovery_tests()


def source_inventory_tests():
    from prepare_zuco_paper_figures import resolve_source_inventory
    from word_sentiment_eeg.paper_stage_archive import migrate_endpoint_protocol,LEGACY_BLINK_FIXED_CODE
    import copy
    def file(name):return dict(id=name,name=name,download='unused',bytes=1)
    raw=[file('ZDN_SR'+str(i)+'_ET.mat') for i in [1,2,3,5,6]]+[file('ZDN_SR8_corrected_ET.mat')]
    processed=[file('gip_ZDN_SNR'+str(i)+'_EEG.mat') for i in range(1,9)]
    processed += [file('ZDN_SR'+str(i)+'_corrected_ET.mat') for i in [1,2,3,6,7,8]]
    blocks=resolve_source_inventory('ZDN',raw,processed)
    assert [v['block_id'] for v in blocks if v['available']]==['1','2','3','6']
    assert {v['block_id']:v['missing_sources'] for v in blocks if not v['available']}=={
        '4':['raw_eye_events','corrected_eye_events'],'5':['corrected_eye_events'],'7':['raw_eye_events'],'8':['raw_eye_events']}
    alias=resolve_source_inventory('ZDN',[file('ZDN_SNR1_ET.mat'),file('ZDN_SNR1_corrected_ET.mat')],[file('oip_ZDN_SNR1_EEG.mat')])
    assert alias[0]['available'] and alias[0]['corrected']['name']=='ZDN_SNR1_corrected_ET.mat'
    for bad_raw,bad_processed in [([file('ZDN_SR1_ET.mat'),file('ZDN_SNR1_ET.mat')],processed),
                                  (raw,processed+[file('oip_ZDN_SR1_EEG.mat')])]:
        try:resolve_source_inventory('ZDN',bad_raw,bad_processed)
        except ValueError:pass
        else:raise AssertionError('Ambiguous source accepted')
    with tempfile.TemporaryDirectory() as folder:
        old=dict(subjects=['ZAB','ZDM','ZDN'],software=dict(numpy='same'),code=dict(LEGACY_BLINK_FIXED_CODE,unchanged='abc'))
        new=copy.deepcopy(old)
        for key in LEGACY_BLINK_FIXED_CODE:new['code'][key]='source-fix'
        migrate_endpoint_protocol(old,new,Path(folder))
        assert (Path(folder)/'source_inventory_recovery.json').exists()
    print('All source matching, missing-pair, alias, ambiguity and blink-fixed protocol migration checks passed')


if __name__=='__main__':source_inventory_tests()


def mismatched_trigger_exclusion_test():
    from prepare_zuco_paper_figures import map_block,excluded_clock_block
    from types import SimpleNamespace
    corrected={'eyeevent':SimpleNamespace(fixations=SimpleNamespace(colheader=['duration','fix_avgpos_x','fix_avgpos_y'],data=np.array([[200.,300.,400.]])))}
    try:
        map_block('ZDN','6',np.zeros((105,500)),500,['10','11'],np.array([1,100]),{'event':np.array([[10,100],[12,200]])},corrected,[],set())
    except ValueError as error:
        assert error.diagnostics['status']=='unverified_trigger_sequence'
        events,diagnostic=excluded_clock_block('ZDN','6',corrected,error)
        assert events.selection_status.eq('block_failed_clock_alignment').all()
        assert diagnostic['synchronization']['max_residual_ms'] is None
    else:raise AssertionError('Mismatched trigger sequence was inferred')
    print('Unverified trigger source is explicitly excluded; no fabricated clock')


if __name__=='__main__':mismatched_trigger_exclusion_test()
