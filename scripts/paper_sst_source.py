"""Cloud-only acquisition of byte-verified original SST data, never a model.

Mirror bytes are accepted only if the original ZIP matches the checksum recorded
by the Stanford NLP dataset repository's dataset_infos.json revision below.
"""
import hashlib
from io import BytesIO
from pathlib import Path
import sys
import urllib.request
import zipfile

ORIGINAL_URL='https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
MIRROR_URL='https://data.deepai.org/sst5.zip'
EXPECTED_SHA256='3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db'
CHECKSUM_SOURCE='https://huggingface.co/datasets/stanfordnlp/sst/blob/fb1011170967fbcecfa3327a706b89e3d24cc8f5/dataset_infos.json'


def verified_payload(payload,expected=EXPECTED_SHA256):
    if hashlib.sha256(payload).hexdigest()==expected:return payload
    with zipfile.ZipFile(BytesIO(payload)) as outer:
        matches=[p for p in outer.namelist() if p.rsplit('/',1)[-1]=='stanfordSentimentTreebank.zip']
        if len(matches)!=1:raise ValueError('Mirror does not contain one original SST ZIP')
        with outer.open(matches[0]) as source:original=source.read(10_000_001)
    if len(original)>10_000_000 or hashlib.sha256(original).hexdigest()!=expected:
        raise ValueError('Original SST ZIP checksum mismatch; refusing replacement data')
    return original


def acquire_sst(directory):
    if sys.platform=='darwin':raise RuntimeError('Cloud only: no SST dataset downloads on the Mac')
    path=Path(directory)/'stanfordSentimentTreebank.zip'
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_SHA256:
        return path,dict(source='validated existing cloud cache',sha256=EXPECTED_SHA256,bytes=path.stat().st_size,
                         checksum_source=CHECKSUM_SOURCE)
    errors=[]
    for url in [ORIGINAL_URL,MIRROR_URL]:
        try:
            print('Read original SST source:',url,flush=True)
            request=urllib.request.Request(url,headers={'User-Agent':'word-eeg-reproduction-audit/1'})
            with urllib.request.urlopen(request,timeout=20) as response:payload=response.read(20_000_001)
            if len(payload)>20_000_000:raise ValueError('SST transport exceeded 20 MB cap')
            original=verified_payload(payload)
            path.write_bytes(original)
            return path,dict(source=url,original_url=ORIGINAL_URL,sha256=EXPECTED_SHA256,
                             bytes=len(original),transport_bytes=len(payload),checksum_source=CHECKSUM_SOURCE,
                             unsuccessful_sources=errors)
        except Exception as error:
            errors.append(dict(url=url,error=f'{type(error).__name__}: {error}'))
    raise RuntimeError('No byte-verified original SST source was accessible: '+str(errors))
