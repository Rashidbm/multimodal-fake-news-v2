"""Check reused DGM4 images against the author's pinned ZIP central directories."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

import requests

from . import remote_genimage as remote
from .development import OUT,DATASETS
from .preprocess import read_rows

REVISION='9efd3f1ca14cb1dac1692cdce0beea6844b968a6'


class RangeFile(io.RawIOBase):
    def __init__(self,path,size):self.path=path;self.size=size;self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,size=-1):
        size=self.size-self.pos if size<0 else min(size,self.size-self.pos)
        value=remote.range_bytes(self.path,self.pos,size,REVISION)
        self.pos+=len(value);return value


def main():
    # This changes only this verifier process, never the training or GenImage source.
    remote.REPO='rshaojimmy/DGM4'
    root=f'https://huggingface.co/api/datasets/{remote.REPO}/tree/{REVISION}/'
    archives={}
    for folder in ('origin','manipulation'):
        response=requests.get(root+folder+'?limit=100',timeout=30);response.raise_for_status()
        archives.update({r['path']:r['size'] for r in response.json() if r['type']=='file'})
    rows=read_rows(DATASETS['dgm4']);groups=defaultdict(list)
    for row in rows:
        relative=str(row['image_path']).split('/DGM4/',1)[1]
        parts=relative.split('/');groups['/'.join(parts[:2])+'.zip'].append((row,relative))
    previous=json.loads((OUT/'dgm4_official_image_sample_verification.json').read_text())
    samples={r['source_path'] for r in previous['results']}
    def check(item):
        archive,items=item;matches=[];errors=[];exact=[]
        with zipfile.ZipFile(RangeFile(archive,archives[archive])) as z:
            names=z.namelist();lookup={}
            for name in names:
                components=name.strip('/').split('/')
                # Preserve the source folder and nested numeric path.
                for start in range(len(components)):
                    suffix='/'.join(components[start:]);lookup.setdefault(suffix,[]).append(name)
            for row,relative in items:
                candidates=lookup.get(relative,lookup.get('/'.join(relative.split('/')[1:]),[]))
                if len(candidates)!=1:
                    errors.append(dict(sample_id=row['sample_id'],error='Archive member missing or ambiguous',relative=relative));continue
                member=candidates[0];info=z.getinfo(member);raw=Path(row['image_path']).read_bytes()
                ok=len(raw)==info.file_size and (zlib.crc32(raw)&0xffffffff)==info.CRC
                if not ok:errors.append(dict(sample_id=row['sample_id'],error='Local size/CRC differs from official archive'))
                else:matches.append(row['sample_id'])
                if relative in samples:
                    official=z.read(member)
                    exact.append(dict(source_path=relative,member=member,local_sha256=hashlib.sha256(raw).hexdigest(),
                                      official_sha256=hashlib.sha256(official).hexdigest(),match=raw==official))
        print(archive,'CRC matches',len(matches),'errors',len(errors),'sample byte matches',sum(r['match'] for r in exact),flush=True)
        return dict(archive=archive,checked=len(items),size_and_crc_matches=len(matches),errors=errors,byte_samples=exact)
    with ThreadPoolExecutor(3) as pool:results=list(pool.map(check,groups.items()))
    report=dict(official_repo=remote.REPO,official_revision=REVISION,checked=len(rows),
                size_and_crc_matches=sum(r['size_and_crc_matches'] for r in results),
                byte_samples_checked=sum(len(r['byte_samples']) for r in results),
                exact_byte_matches=sum(v['match'] for r in results for v in r['byte_samples']),results=results,
                interpretation='All selected local files checked against author ZIP size/CRC; a fixed 16-image sample also compared byte-for-byte. CRC is not a cryptographic identity guarantee.')
    (OUT/'dgm4_official_archive_verification.json').write_text(json.dumps(report,indent=2))
    if report['size_and_crc_matches']!=len(rows) or report['exact_byte_matches']!=report['byte_samples_checked']:
        raise ValueError('DGM4 official archive verification failed')


if __name__=='__main__':main()
