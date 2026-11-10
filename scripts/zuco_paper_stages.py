#!/usr/bin/env python3
"""CPU preparation -> verified Drive trials -> GPU permutations and figures."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from word_sentiment_eeg.common import file_digest,write_json
from word_sentiment_eeg.paper_stage_archive import disk_bytes,verify_prepared,finish_reader
from prepare_zuco_paper_figures import prepare_reader,source_inventory,DEPARTURES
from paper_sst_source import acquire_sst
from word_sentiment_eeg.paper_labels import SST


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thesis-root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--phase',choices=['prepare','analyze'],required=True)
    p.add_argument('--free-drive-bytes',type=int);p.add_argument('--budget-bytes',type=int,default=4_000_000_000)
    p.add_argument('--subjects',nargs='+');p.add_argument('--blocks',nargs='+')
    p.add_argument('--tct-permutations',type=int,default=5000);p.add_argument('--cluster-permutations',type=int,default=1000)
    p.add_argument('--cpu',action='store_true',help='Use CPU for permutation validation only')
    p.add_argument('--skip-inference',action='store_true',help='Cloud engineering tests only: stop after GPU reader checkpoints')
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('Run in Colab/cloud only')
    out=args.output or args.thesis_root/'Results/word_sentiment_eeg/zuco_paper_figures_staged'
    sources=[s for s in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if s['task']=='SR']
    if args.subjects:sources=[s for s in sources if s['subject'] in args.subjects]
    if not sources or (args.subjects and set(args.subjects)!={s['subject'] for s in sources}):raise ValueError('Unknown reader')
    if min(args.tct_permutations,args.cluster_permutations)<1 or not 0<args.budget_bytes<=10_000_000_000:raise ValueError('Invalid storage/permutation budget')
    null_bytes=(args.tct_permutations+args.cluster_permutations)*2*105*201*4
    prep_limit=args.budget_bytes-null_bytes-250_000_000
    if prep_limit<100_000_000:raise ValueError('Budget too small for trial storage plus permutation reserve')
    names=['scripts/zuco_paper_stages.py','scripts/prepare_zuco_paper_figures.py','scripts/zuco_paper_pilot.py',
           'scripts/paper_sst_source.py','src/word_sentiment_eeg/paper_stage_archive.py','src/word_sentiment_eeg/paper_figure_stats.py',
           'src/word_sentiment_eeg/paper_python.py','src/word_sentiment_eeg/paper_released.py','src/word_sentiment_eeg/paper_labels.py',
           'src/word_sentiment_eeg/paper_fixations.py','src/word_sentiment_eeg/paper_trials.py']
    protocol=dict(workflow='cpu_gpu_stages_v1',subjects=[s['subject'] for s in sources],sources=sources,blocks=args.blocks,
        budget_bytes=args.budget_bytes,tct_permutations=args.tct_permutations,cluster_permutations=args.cluster_permutations,
        code={name:file_digest(ROOT/name) for name in names},software={name:version(name) for name in ['numpy','scipy','pandas','h5py','scikit-learn']},
        storage='Lossless float32: raw/clean, accepted trial, 105 channels, 201 samples; original trial IDs and ranked pair indices')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Staged protocol or numerical environment changed; outputs must not be mixed')
    if args.phase=='analyze' and not (out/'protocol.json').exists():raise ValueError('Run CPU preparation first')
    used=disk_bytes(out);required=max(0,args.budget_bytes-used)
    if args.output is None and (args.free_drive_bytes is None or args.free_drive_bytes<required):raise OSError(f'Need {required/1e9:.2f} GB additional verified free space for the {args.budget_bytes/1e9:.1f} GB capped staged workflow. No extraction started.')
    write_json(out/'protocol.json',protocol)
    (out/'METHOD_DIFFERENCES.md').write_text('# Python reconstruction differences\n\n'+'\n'.join('- '+v for v in DEPARTURES)+'\n')
    missing=[]
    for i,s in enumerate(sources,1):
        reader=out/'readers'/s['subject']
        if (reader/'prepared.json').exists():verify_prepared(reader);print(f'KEEP CPU READER {i}/{len(sources)} {s["subject"]}: saved trials verified',flush=True)
        else:missing.append(s)
    if args.phase=='prepare':
        if missing:
            with tempfile.TemporaryDirectory(prefix='sst-staged-') as temporary:
                path,provenance=acquire_sst(Path(temporary));sst=SST(path);write_json(out/'original_label_source.json',provenance)
                for s in missing:
                    i=[v['subject'] for v in sources].index(s['subject'])+1;reader=out/'readers'/s['subject']
                    print(f'CPU PREPARE READER {i}/{len(sources)} {s["subject"]}',flush=True)
                    inventory=source_inventory(s['subject'])
                    if args.blocks:inventory=[b for b in inventory if b['block_id'] in args.blocks]
                    if not inventory:raise ValueError('No requested source blocks')
                    local=args.thesis_root/'Data/zuco_og_raw'/s['filename']
                    prepare_reader(s['subject'],s,inventory,sst,reader,out/'null_work',
                        (args.tct_permutations,args.cluster_permutations),False,
                        source_paths={s['filename']:local} if local.is_file() else None,
                        defer_permutations=True,storage_root=out,preparation_budget_bytes=prep_limit)
        write_json(out/'cpu_preparation_manifest.json',dict(status='completed',subjects=[s['subject'] for s in sources],
            protocol_sha256=file_digest(out/'protocol.json'),reader_checkpoint_sha256={s['subject']:file_digest(out/'readers'/s['subject']/'prepared.json') for s in sources}))
        print('CPU PREPARATION COMPLETE: all saved trials verified. You can now switch to GPU and run the analysis notebook.',flush=True);return
    if missing:raise ValueError('CPU preparation incomplete; missing readers: '+str([s['subject'] for s in missing]))
    state=json.loads((out/'cpu_preparation_manifest.json').read_text())
    if state['status']!='completed' or state['subjects']!=protocol['subjects'] or state['protocol_sha256']!=file_digest(out/'protocol.json'):raise ValueError('CPU handoff identity changed')
    for s in sources:
        if file_digest(out/'readers'/s['subject']/'prepared.json')!=state['reader_checkpoint_sha256'][s['subject']]:raise ValueError('Prepared reader identity changed')
    fixed=used-(disk_bytes(out/'null_work') if (out/'null_work').exists() else 0)
    if fixed+null_bytes+200_000_000>args.budget_bytes:raise OSError('GPU permutation reserve would exceed total workflow cap; CPU data retained')
    for i,s in enumerate(sources,1):
        print(f'GPU ANALYZE READER {i}/{len(sources)} {s["subject"]}: load saved accepted EEG; NO source download or cleaning',flush=True)
        finish_reader(out/'readers'/s['subject'],out/'null_work',args.tct_permutations,args.cluster_permutations,use_gpu=not args.cpu)
    write_json(out/'preparation_manifest.json',dict(status='completed',subjects=protocol['subjects']))
    if not args.skip_inference:
        from render_zuco_paper_figures import analyze_and_render
        analyze_and_render(out);shutil.rmtree(out/'null_work',ignore_errors=True)
    print(f'GPU STAGE COMPLETE: {out}; retained trial archive can be used for subsequent training',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
