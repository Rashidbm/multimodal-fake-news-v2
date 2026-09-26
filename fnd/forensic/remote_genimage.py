"""Read original split-ZIP members by HTTP ranges without downloading GenImage in full."""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
import zipfile
import zlib

import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError

REPO = 'jzousz/GenImage'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/raw/GenImage_subset'
GENERATORS = {'adm':'ADM', 'biggan':'BigGAN', 'midjourney':'Midjourney',
              'vqdm':'VQDM', 'glide':'glide', 'sdv1_4':'stable_diffusion_v_1_4',
              'sdv1_5':'stable_diffusion_v_1_5', 'wukong':'wukong'}
_sessions=threading.local()
_redirects={}


def get_json(url):
    r = requests.get(url, timeout=60); r.raise_for_status(); return r.json()


def range_bytes(path, start, size, revision, attempts=5):
    """Require an exact Range response; never accidentally stream a multi-GB part."""
    if size == 0: return b''
    if not hasattr(_sessions,'session'):_sessions.session=requests.Session()
    for attempt in range(attempts):
        try:
            url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}'
            # Distinct URLs avoid intermediary caches incorrectly reusing a different range.
            url += f'?download=true&range_start={start}&range_size={size}'
            cached=_redirects.get((revision,path))
            if cached and time.monotonic()-cached[1]<1200:url=cached[0]
            with _sessions.session.get(url, headers={'Range':f'bytes={start}-{start+size-1}'},
                              stream=True, timeout=(20,120)) as r:
                if r.status_code in (401,403):_redirects.pop((revision,path),None)
                r.raise_for_status()
                expected = f'bytes {start}-{start+size-1}/'
                if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(expected):
                    raise ValueError(f'Range not honored: status={r.status_code} range={r.headers.get("Content-Range")}')
                data = r.raw.read(size+1)
                if len(data) != size: raise ValueError(f'Short range {len(data)} != {size}')
                _redirects[(revision,path)]=(r.url,time.monotonic())
                return data
        except (requests.RequestException, UrllibHTTPError, ValueError) as exc:
            if attempt+1 == attempts: raise RuntimeError(f'Range failed for {path} at {start}: {type(exc).__name__}') from exc
            time.sleep(min(2**attempt, 10))


def read_parts(parts, disk, offset, size, revision):
    chunks = []
    while size:
        part = parts[disk]
        available = min(size, part['size']-offset)
        if available <= 0: raise ValueError('Invalid split archive offset')
        chunks.append(range_bytes(part['path'], offset, available, revision))
        size -= available; disk += 1; offset = 0
    return b''.join(chunks)


def build_index(generator, folder, revision):
    dest = OUT/'indexes'/f'{generator}.json.gz'
    if dest.exists():
        with gzip.open(dest, 'rt') as f: result=json.load(f)
        if result['revision'] != revision: raise ValueError('Pinned source changed')
        return result
    files = get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{folder}?limit=1000')
    zip_parts = [p for p in files if p['type']=='file' and p['path'].rsplit('.',1)[-1].startswith('z')]
    zip_parts.sort(key=lambda p: 100000 if p['path'].endswith('.zip') else int(p['path'].rsplit('.z',1)[-1]))
    if not zip_parts or not zip_parts[-1]['path'].endswith('.zip'): raise ValueError('Incomplete archive listing')
    parts = [{'path':p['path'],'size':p['size']} for p in zip_parts]
    last = parts[-1]
    tail_size=min(last['size'], 65536)
    tail=range_bytes(last['path'],last['size']-tail_size,tail_size,revision)
    end=tail.rfind(b'PK\x05\x06')
    if end < 0: raise ValueError('Missing ZIP end record')
    e=struct.unpack_from('<4s4H2LH',tail,end)
    _,disk,cd_disk,per_disk,total,cd_size,cd_offset,comment=e
    z64=tail.rfind(b'PK\x06\x06',0,end)
    if z64 >= 0:
        v=struct.unpack_from('<4sQ2H2I4Q',tail,z64)
        disk,cd_disk,per_disk,total,cd_size,cd_offset=v[4:]
    if disk != len(parts)-1: raise ValueError('ZIP part count mismatch')
    central=read_parts(parts,cd_disk,cd_offset,cd_size,revision)
    members=[]; pos=0
    while pos < len(central):
        c=struct.unpack_from(zipfile.structCentralDir,central,pos)
        if c[0] != zipfile.stringCentralDir: raise ValueError('Invalid central entry')
        n,x,k=c[12:15]
        name=central[pos+46:pos+46+n].decode('utf-8' if c[5]&0x800 else 'cp437')
        extra=central[pos+46+n:pos+46+n+x]
        pos += 46+n+x+k
        if name.endswith('/') or Path(name).suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}: continue
        p=name.replace('\\','/').split('/')
        split=next((s for s in ['train','val'] if s in p),None)
        label=0 if 'nature' in p else 1 if 'ai' in p else None
        if split is None or label is None: raise ValueError(f'Unknown source label: {name}')
        uncompressed,compressed,offset,member_disk=c[11],c[10],c[18],c[15]
        ep=0
        while ep+4 <= len(extra):
            tag,length=struct.unpack_from('<HH',extra,ep); payload=extra[ep+4:ep+4+length];ep+=4+length
            if tag==1:
                q=0; values=[uncompressed,compressed,offset,member_disk]
                for j, sentinel in enumerate([0xffffffff,0xffffffff,0xffffffff,0xffff]):
                    if values[j]==sentinel:
                        fmt='<I' if j==3 else '<Q';values[j]=struct.unpack_from(fmt,payload,q)[0];q+=4 if j==3 else 8
                uncompressed,compressed,offset,member_disk=values
        members.append(dict(name=name,split=split,label=label,disk=member_disk,offset=offset,
                            compressed=compressed,size=uncompressed,crc=c[9],method=c[6]))
    result=dict(generator=generator,repo=REPO,revision=revision,parts=parts,members=members,
                central_sha256=hashlib.sha256(central).hexdigest(),total_zip_entries=total)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest,'wt') as f: json.dump(result,f)
    from collections import Counter
    print(generator,len(members),dict(Counter((x['split'],x['label']) for x in members)),flush=True)
    return result


def extract_member(index, member):
    parts, revision=index['parts'],index['revision']
    # ZIP filenames repeat in the local header; central name length is normally identical.
    expected_header=30+len(member['name'].encode('utf-8'))+512
    amount=min(expected_header+member['compressed'],sum(p['size'] for p in parts[member['disk']:])-member['offset'])
    blob=read_parts(parts,member['disk'],member['offset'],amount,revision)
    header=struct.unpack_from('<4s5H3I2H',blob)
    if header[0]!=b'PK\x03\x04': raise ValueError('Invalid local header')
    start=30+header[-2]+header[-1]
    if len(blob)<start+member['compressed']:
        blob=read_parts(parts,member['disk'],member['offset'],start+member['compressed'],revision)
    compressed=blob[start:start+member['compressed']]
    if member['method']==8: raw=zlib.decompress(compressed,-15)
    elif member['method']==0: raw=compressed
    else: raise ValueError('Unsupported compression')
    if len(raw)!=member['size'] or zlib.crc32(raw)&0xffffffff!=member['crc']:
        raise ValueError('Archive CRC/length mismatch')
    return raw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generators',nargs='+',default=list(GENERATORS))
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'source.json'
    if lock.exists(): source=json.loads(lock.read_text())
    else:
        source=dict(repo=REPO,revision=get_json(f'https://huggingface.co/api/datasets/{REPO}')['sha'],
                    official_project='https://github.com/GenImage-Dataset/GenImage',
                    source_kind='Third-party mirror of original multipart split archives')
        lock.write_text(json.dumps(source,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(build_index,g,GENERATORS[g],source['revision']) for g in args.generators]
        for future in concurrent.futures.as_completed(futures): future.result()


if __name__=='__main__': main()
