"""Saved-plot integration: reference cancellation, GFP, rosters, immutability."""
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import struct

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import plot_saved_erps as plots
import notebook_plot_display as display_plots
from word_sentiment_eeg.common import file_digest,write_json


def check_metrics():
    mean,sem=plots.mean_and_sem(np.array([[1.],[3.],[5.]]))
    np.testing.assert_allclose(mean,[3.])
    np.testing.assert_allclose(sem,[2/np.sqrt(3)])
    assert plots.mean_and_sem(np.array([[1.]]))[1] is None
    assert plots.panel_layout(9,horizontal=True)[:2]==(1,9)
    assert plots.panel_layout(1)[:2]==(1,1)
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
            assert len(gallery)==(48 if dataset=='zuco' else 19)
            assert len({x['path'] for x in gallery})==len(gallery)
            rows=[x for x in gallery if x['kind']=='subject']
            assert all(x['layout']=='one_reader_per_row' for x in rows)
            assert all(len(x['columns'])==8 for x in rows)
            if dataset=='zuco':
                assert not any(x.get('view')=='original_overview' for x in gallery)
                assert all('original_12_channel_roi' not in x.get('columns',[]) for x in gallery)
                assert len([x for x in gallery if x.get('view')=='overall'])==3
                assert all(x['measures']==['whole_scalp_signed_mean'] for x in gallery if x.get('view')=='overall')
            # A task cell shows ONE combined matrix, not twelve reader images.
            reader_task=rows[0]['task']
            for item in rows:
                path=out/item['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic-png-for-html-test')
            matrices=[x for x in gallery if x['kind']=='subject_matrix']
            assert len(matrices)==(3 if dataset=='zuco' else 1)
            for item in matrices:
                assert item['readers']==sorted(r.subject for r in saved.records if r.task==item['task'])
                assert item['reader_count']==12 and len(item['columns'])==8
                path=out/item['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'one-combined-matrix')
            rendered=display_plots.task_reader_rows_html(gallery,out,reader_task)
            assert rendered.count('<img ')==1 and '<figure ' not in rendered
            assert 'overflow:auto' in rendered and 'max-width:none' in rendered
            assert 'base64,' in rendered
            assert 'b25lLWNvbWJpbmVkLW1hdHJpeA==' in rendered
            try:display_plots.task_reader_rows_html(rows,out,reader_task)
            except ValueError as error:assert 'Combined reader matrix missing' in str(error)
            else:raise AssertionError('Legacy separate-reader images used instead of matrix')
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
                    plots.render_panels(metrics,groups,saved.times,saved.units,path,'Synthetic '+dataset,keys,count,False,horizontal=(name=='combined'))
                    assert path.stat().st_size>1000
                    if name=='combined':
                        width,height=struct.unpack('>II',path.read_bytes()[16:24])
                        assert width/height>len(keys)*.75
                path=out/'single-overall.png'
                plots.render_panels(metrics,groups,saved.times,saved.units,path,'Group mean ±1 SEM',
                                    ['whole_scalp_signed_mean'],count,True)
                assert path.stat().st_size>1000
                matrix_path=out/'reader_matrices'/f'{record.task}_reader_matrix.png'
                matrix=plots.render_reader_matrix(saved,record.task,matrix_path)
                assert matrix['reader_count']==12
                width,height=struct.unpack('>II',matrix_path.read_bytes()[16:24])
                assert width>3000 and height>2500
                assert matrix_path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
                if os.environ.get('PLOT_PREVIEW_DIR'):
                    preview=Path(os.environ['PLOT_PREVIEW_DIR']);preview.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(matrix_path,preview/f'{dataset}-reader-matrix.png')
                if dataset=='zuco':
                    two=np.stack([r.means for r in saved.records if r.task==record.task][:2])
                    group_metrics=plots.summaries(two,saved.montage,saved.channels,True)
                    plots.render_panels(group_metrics,groups,saved.times,saved.units,out/'original-overview.png',
                                        'Original ZuCo ROI · equal reader mean ±1 SEM',
                                        ['original_12_channel_roi'],count,True)
                    if os.environ.get('PLOT_PREVIEW_DIR'):
                        preview=Path(os.environ['PLOT_PREVIEW_DIR']);preview.mkdir(parents=True,exist_ok=True)
                        for filename in ('combined.png','original-overview.png','single-overall.png'):
                            shutil.copyfile(out/filename,preview/filename)
            # Never silently plot corrupt or repeated reader checkpoints.
            victim=next(p for p in before if p.name=='class_means.npz')
            victim.write_bytes(b'corrupt file')
            try:plots.load_saved(run,dataset)
            except ValueError as error:assert 'checksum' in str(error)
            else:raise AssertionError('Corrupt saved means were accepted')
    for dataset,tasks in [('zuco',('SR','NR','TSR')),('teco',('task1',))]:
        book=json.loads((ROOT/'notebooks'/f'{dataset}_plots.ipynb').read_text())
        for task in tasks:
            task_cells=[c for c in book['cells'] if c['metadata'].get('id')=='reader-rows-'+task]
            assert len(task_cells)==1 and f"'{task}'" in ''.join(task_cells[0]['source'])
    print('All-plot checks passed: one combined reader matrix per task, PNG/PDF, group overviews, SEM, GFP, roster and source preservation')

if __name__=='__main__':main()
