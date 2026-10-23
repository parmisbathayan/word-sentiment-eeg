"""Saved-plot integration: reference cancellation, GFP, rosters, immutability."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import plot_saved_erps as plots
from word_sentiment_eeg.common import file_digest,write_json


def check_metrics():
    # Opposite reader topographies cancel in the grand average, but GFP of
    # each reader's class-average signal remains positive. No trial weighting.
    values=np.array([[[[-1.,-2.],[1.,2.]]],[[[1.,2.],[-1.,-2.]]]])
    montage=pd.DataFrame({'region':['a','b']})
    old=plots.REGIONS
    plots.REGIONS=('a','b')
    try: metrics=plots.summaries(values,montage,['a','b'],False)
    finally:plots.REGIONS=old
    np.testing.assert_array_equal(metrics['whole_scalp_signed_mean'],0)
    np.testing.assert_array_equal(metrics['whole_scalp_gfp'].mean(axis=0),[[1.,2.]])
    assert values.mean(axis=0).std(axis=1).max()==0
    shifted=values+10
    plots.REGIONS=('a','b')
    try:other=plots.summaries(shifted,montage,['a','b'],False)
    finally:plots.REGIONS=old
    np.testing.assert_array_equal(metrics['whole_scalp_gfp'],other['whole_scalp_gfp'])
    rec=[plots.Record('task','A',values[0],np.array([5])),plots.Record('task','B',values[1],np.array([100]))]
    readers,stack=plots.matched(rec,('neutral',),('neutral',))
    assert readers==['A','B'] and stack.mean(axis=0).max()==0
    rec[0].counts[0]=4
    assert plots.matched(rec,('neutral',),('neutral',))[0]==['B']


def fixture(run,dataset):
    groups=plots.CLASSES if dataset=='zuco' else plots.TECO_GROUPS
    times=plots.ZUCO_TIMES if dataset=='zuco' else plots.TECO_TIMES
    montage=pd.read_csv(ROOT/'assets/zuco_chanlocs.csv') if dataset=='zuco' else plots.load_montage(ROOT/'assets/teco_montage.txt')
    names=montage.labels.tolist() if dataset=='zuco' else montage.channel.tolist()
    if dataset=='zuco':
        montage.to_csv(run/'zuco_chanlocs.csv',index=False);(run/'audit').mkdir()
        write_json(run/'audit/integrity.json',{'status':'passed'})
    else:(run/'teco_montage.txt').write_bytes((ROOT/'assets/teco_montage.txt').read_bytes())
    readers=[f'R{i:02}' for i in range(12)] if dataset=='zuco' else list(plots.READERS)
    tasks=('SR','NR','TSR') if dataset=='zuco' else ('task1',)
    records=[];checksums={}
    space=np.arange(len(names),dtype=float);space-=space.mean();space/=space.std()
    trace=np.sin(np.arange(len(times))*.02)
    base=space[:,None]*trace[None,:]
    for task in tasks:
        for i,reader in enumerate(readers):
            rel=Path('readers')/task/reader if dataset=='zuco' else Path('readers')/reader
            out=run/rel;out.mkdir(parents=True)
            counts={g:5+i for g in groups}
            if dataset=='teco' and reader=='elahe':counts['neutral_in_neutral_sentence']=1
            means=np.stack([base*(j+1)*(1 if i%2 else -1) for j in range(len(groups))]).astype(np.float32)
            keys={'classes':np.array(groups),'times_ms':times,'counts':np.array(list(counts.values()))}
            keys.update({'mean_uV':means,'channel_names':np.array(names)} if dataset=='zuco' else {'means':means,'channels':np.array(names)})
            np.savez_compressed(out/'class_means.npz',**keys)
            sha=file_digest(out/'class_means.npz');checksums[str(rel/'class_means.npz')]=sha
            item={'task':task,'subject':reader,'fingerprint':'synthetic','status':'completed'}
            item.update({'epochs_per_class':counts} if dataset=='zuco' else {'counts':counts,'checksums':{'class_means.npz':sha}})
            write_json(out/'manifest.json',item);records.append(item)
    if dataset=='zuco':write_json(run/'audit/file_checksums.json',checksums)
    manifest={'status':'completed','failures':[],'files':records} if dataset=='zuco' else {'status':'completed','failures':[],'expected_readers':readers,'missing_readers':[],'readers':records}
    write_json(run/'manifest.json',manifest)


def main():
    check_metrics()
    with tempfile.TemporaryDirectory() as temp:
        for dataset in ('zuco','teco'):
            run=Path(temp)/dataset;run.mkdir();fixture(run,dataset)
            before={p:file_digest(p) for p in run.rglob('*') if p.is_file()}
            saved,out,gallery=plots.plot_collection(run,dataset,make_plots=False)
            assert len(gallery)==(42 if dataset=='zuco' else 18)
            assert len({x['path'] for x in gallery})==len(gallery)
            rosters=json.loads((out/'comparison_rosters.json').read_text())
            if dataset=='teco':
                assert len(rosters['task1/strict_neutral']['readers'])==11
                assert len(rosters['task1/pooled_neutral']['readers'])==12
            for path,sha in before.items():assert file_digest(path)==sha
            if '--unit-only' not in sys.argv:
                # Exercise real matplotlib once for overview, region and subject
                # layouts; avoid producing dozens of redundant synthetic PNGs.
                record=saved.records[0];groups=saved.groups
                metrics=plots.summaries(record.means[None],saved.montage,saved.channels,dataset=='zuco')
                count={r:int(saved.montage.region.eq(r).sum()) for r in plots.REGIONS}
                for name,keys in [('overall',list(metrics)[:3 if dataset=='zuco' else 2]),('regions',list(plots.REGIONS)),('combined',list(metrics))]:
                    path=out/(name+'.png')
                    plots.render_panels(metrics,groups,saved.times,saved.units,path,'Synthetic '+dataset,keys,count,False)
                    assert path.stat().st_size>1000
            # Never silently plot corrupt or repeated reader checkpoints.
            victim=next(p for p in before if p.name=='class_means.npz')
            victim.write_bytes(b'corrupt file')
            try:plots.load_saved(run,dataset)
            except ValueError as error:assert 'checksum' in str(error)
            else:raise AssertionError('Corrupt saved means were accepted')
    print('All-plot checks passed: true full-head mean, per-reader GFP, equal weighting, strict-neutral roster, source preservation and corrupt-checkpoint rejection')

if __name__=='__main__':main()
