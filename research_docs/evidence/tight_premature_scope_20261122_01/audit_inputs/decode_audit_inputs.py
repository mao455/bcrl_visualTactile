#!/usr/bin/env python3
"""Standard-library-only decode and verify; emits exact raw arrays and JSON.

Usage: python decode_audit_inputs.py DIR --verify-only
       python decode_audit_inputs.py DIR --out decoded

The raw .bin files are exact C-order dataset bytes. The accompanying manifest
keeps original NumPy dtype.str and shape, which suffice for optional rebuilding:
  np.frombuffer(raw, dtype=np.dtype(a['dtype'])).reshape(a['shape'])
For HDF5, create the recorded group and each transition/<dataset key>, attach
metadata_json from metadata_selected.json, and result_json from result.json.
NumPy/h5py are optional reconstruction dependencies, not needed for this decoder.
"""
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if not a.verify_only and a.out is None:p.error('--out or --verify-only required')
    records=arrays=missing=totalbytes=0;cohorts={};seen=set()
    manifests=json.loads((a.directory/'shards.json').read_text(encoding='utf-8'))
    expected_cohorts=json.loads((a.directory/'registration.json').read_text(encoding='utf-8'))['expected_trajectories']
    assert {v['file'] for v in manifests}=={v.name for v in a.directory.glob('*.jsonl.gz')},'Missing or unexpected shard'
    for entry in manifests:
        shard=a.directory/entry['file'];digest=hashlib.sha256()
        with shard.open('rb') as rawfile:
            for block in iter(lambda:rawfile.read(4*1024*1024),b''):digest.update(block)
        assert shard.stat().st_size==entry['bytes'] and digest.hexdigest()==entry['sha256'],shard
        shard_identities=[]
        with gzip.open(shard,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);key=(r['cohort'],r['identity']);assert key not in seen,key;seen.add(key)
                assert r['cohort']==entry['cohort'];shard_identities.append(r['identity'])
                records+=1;cohorts[r['cohort']]=cohorts.get(r['cohort'],0)+1
                dest=a.out/r['cohort']/r['identity'] if a.out is not None and not a.verify_only else None
                if dest is not None:dest.mkdir(parents=True,exist_ok=True)
                for name,v in r['datasets'].items():
                    if v is None:missing+=1;continue
                    raw=base64.b64decode(v['base64'],validate=True)
                    assert len(raw)==v['nbytes']==math.prod(v['shape'])*int(v['dtype'][2:]),name
                    assert hashlib.sha256(raw).hexdigest()==v['raw_sha256'],name
                    arrays+=1;totalbytes+=len(raw)
                    if dest is not None:
                        target=dest/'transition'/(name+'.bin');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                result=r['result_json'].encode('utf-8');assert hashlib.sha256(result).hexdigest()==r['result_json_sha256']
                metadata=r['metadata_selected_json'].encode('utf-8');assert hashlib.sha256(metadata).hexdigest()==r['metadata_selected_json_sha256']
                if dest is not None:
                    (dest/'metadata_selected.json').write_text(r['metadata_selected_json'],encoding='utf-8')
                    (dest/'result.json').write_bytes(result)
                    manifest={k:v for k,v in r.items() if k not in ('datasets','result_json','metadata_selected_json')}
                    manifest['datasets']={k:({kk:vv for kk,vv in v.items() if kk!='base64'} if v is not None else None) for k,v in r['datasets'].items()}
                    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        assert len(shard_identities)==entry['trajectories'] and shard_identities==entry['identities'],shard
    assert cohorts==expected_cohorts,(cohorts,expected_cohorts)
    result=dict(passed=True,records=records,arrays=arrays,missing_optional_fields=missing,raw_array_bytes=totalbytes,cohorts=cohorts,
                shard_file_sha256_verified=len(manifests),shard_record_counts_and_identity_order_verified=True,expected_cohort_counts_verified=True,
                metadata_selected_json_sha256_verified=records,result_json_sha256_verified=records)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
