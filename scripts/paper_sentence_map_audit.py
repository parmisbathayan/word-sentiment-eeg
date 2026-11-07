#!/usr/bin/env python3
"""Cloud waveform validation of released sentence identities against native EEG."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from run_erp import cloud_download
from paper_source_audit import fetch_sample
from word_sentiment_eeg.paper_released import released_sentences,sentence_location,native_recording


def main():
    if sys.platform=='darwin':raise SystemExit('Cloud only; do not download EEG on the Mac')
    source=next(r for r in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if r['task']=='SR' and r['subject']=='ZAB')
    native=next(r for r in json.loads((ROOT/'docs/paper_source_audit.json').read_text())['samples'] if r['name']=='gip_ZAB_SR1_EEG.mat')
    report=dict(purpose='Whole-sentence waveform identity audit; no guessed trial ordering',source=source['filename'],native=native['name'],sentences=[])
    with tempfile.TemporaryDirectory(prefix='paper-map-') as temp:
        folder=Path(temp);path=cloud_download(source,folder/source['filename']);native_path,_=fetch_sample(native,folder)
        raw,rate,*_=native_recording(native_path)
        # The first 55 source entries cover the nominal 50-trial block plus a
        # boundary check. The audit does not force the number that must match.
        for sentence in released_sentences(path):
            if sentence['position']>=55:break
            onset,status=sentence_location(raw,sentence)
            report['sentences'].append(dict(source_sentence=sentence['position'],words=len(sentence['words']),
                                            onset_sample_0based=onset,status=status))
            print('MAP',sentence['position'],onset,status,flush=True)
    report['temporary_sources_removed']=True
    target=ROOT/'artifacts/paper_sentence_map_audit.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
