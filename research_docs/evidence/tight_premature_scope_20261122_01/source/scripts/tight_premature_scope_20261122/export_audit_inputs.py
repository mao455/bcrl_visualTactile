"""Lossless numerical audit-input export; no policy or analysis is executed.

Only the frozen analyze() input datasets are read. Original HDF5 files, visual
payloads, weights, caches, and original evidence remain unchanged.
"""
from __future__ import annotations

import ast
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import h5py
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
SCOPE=PROJECT/'outputs/diagnostics/tight_premature_scope_20261122_01'
OUT=SCOPE/'audit_inputs'
PUBLIC=PROJECT/'research_docs/evidence/tight_premature_scope_20261122_01/audit_inputs'
ANALYZE=PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py'
FROZEN_ANALYZE_SHA='72787cec47b3c2a233da6b7e13bdddea987b50ac69bfdbd05b6eee910c72f237'
MAX_ESTIMATED_SHARD_BYTES=40*1024*1024


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


DECODER = r'''#!/usr/bin/env python3
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
'''


def attr_text(value):
    return value.decode('utf-8') if isinstance(value,bytes) else str(value)


def main():
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    assert sha(ANALYZE)==FROZEN_ANALYZE_SHA,'Frozen analyze() changed'
    tree=ast.parse(ANALYZE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
    fields=sorted({n.args[0].value for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='arr'
                   and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)}|
                  {'step_id','dagger/control_source','dagger/expert_label_valid','dagger/segment_id'})
    assert len(fields)==17,fields
    registrations=[SCOPE/'train_dev/source_registration.json',SCOPE/'terminal/source_registration.json']
    source_lists=[SCOPE/'train_dev/source_sha256.json',SCOPE/'terminal/source_sha256.json']
    jobs=[]
    for reg in registrations:jobs.extend(read(reg)['jobs'])
    assert Counter(j['cohort'] for j in jobs)=={'train':180,'development':96,'terminal':96}
    prior_hashes={}
    for src in source_lists:
        for v in read(src):
            if v['path'] in prior_hashes:assert prior_hashes[v['path']]['sha256']==v['sha256']
            prior_hashes[v['path']]=dict(v,prior_manifest=str(src.relative_to(PROJECT)),prior_manifest_sha256=sha(src))
    registration=dict(version='lossless_numerical_audit_inputs_v1',registered_at=datetime.now(timezone.utc).isoformat(),
        purpose='Mechanical export only; no analysis, new threshold, terminal selection, inference, simulation, optimizer, or collection',
        frozen_analyze_sha256=FROZEN_ANALYZE_SHA,dataset_fields=fields,field_discovery='literal arr() calls inside frozen analyze AST plus step_id and three directly consumed dagger fields',
        metadata_fields=['config','joint_order','fk_urdf_sha256','fk_root','trajectory_kind'],
        encoding='one UTF-8 JSON object per trajectory per line; each present array stores exact C-order bytes as base64, original dtype.str/shape, byte count and raw SHA256',
        missing='null means field absent; never substitute zeros or false',source_hashes='reference previously verified source manifests; do not rehash original large HDF5',
        max_estimated_compressed_shard_bytes=MAX_ESTIMATED_SHARD_BYTES,expected_trajectories=dict(Counter(j['cohort'] for j in jobs)))
    write(OUT/'registration.json',registration)
    records=[];shards=[];missing=Counter();array_count=raw_bytes=0
    for cohort in ('train','development','terminal'):
        cohort_jobs=[j for j in jobs if j['cohort']==cohort];compressed_budget=0;part=0;handle=None;partrecords=[];path=None
        def close_shard():
            if handle is None:return
            handle.close()
            assert path.stat().st_size<50*1024*1024,path
            shards.append(dict(file=path.name,cohort=cohort,trajectories=len(partrecords),identities=list(partrecords),bytes=path.stat().st_size,sha256=sha(path)))
        for j in cohort_jobs:
            source=prior_hashes[j['path']]
            with h5py.File(j['path'],'r') as f:
                group=f[j['group']];tr=group['transition'];meta_raw=attr_text(f.attrs['metadata_json']);meta=json.loads(meta_raw)
                selected={k:meta[k] for k in registration['metadata_fields'] if k in meta}
                metadata_selected_json=json.dumps(selected,ensure_ascii=False,separators=(',',':'))
                if j.get('result_path'):
                    result_bytes=Path(j['result_path']).read_bytes();result_raw=result_bytes.decode('utf-8');result_source=dict(prior_hashes[j['result_path']])
                    assert sha_bytes(result_bytes)==result_source['sha256']
                else:
                    result_raw=attr_text(f.attrs['result_json']);result_source=dict(kind='HDF5 root result_json attr',source_path=j['path'])
                arrays={}
                for name in fields:
                    if name not in tr:arrays[name]=None;missing[name]+=1;continue
                    a=np.asarray(tr[name][:]);assert not a.dtype.hasobject and a.dtype.fields is None
                    raw=a.tobytes(order='C');encoded=base64.b64encode(raw).decode('ascii')
                    assert base64.b64decode(encoded,validate=True)==raw
                    arrays[name]=dict(dtype=a.dtype.str,shape=list(a.shape),order='C',nbytes=len(raw),raw_sha256=sha_bytes(raw),base64=encoded)
                    array_count+=1;raw_bytes+=len(raw)
            record=dict(version=registration['version'],cohort=cohort,identity=j['identity'],policy=j.get('policy'),group=j['group'],
                source_numeric=dict(source),source_result=result_source,source_job=j,metadata_selected_json=metadata_selected_json,
                metadata_selected_json_sha256=sha_bytes(metadata_selected_json.encode('utf-8')),original_metadata_json_sha256=sha_bytes(meta_raw.encode('utf-8')),
                result_json=result_raw,result_json_sha256=sha_bytes(result_raw.encode('utf-8')),datasets=arrays)
            line=(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
            estimate=len(gzip.compress(line,compresslevel=9,mtime=0))
            if handle is None or compressed_budget+estimate>MAX_ESTIMATED_SHARD_BYTES:
                close_shard();path=OUT/f'{cohort}.part{part:03d}.jsonl.gz';handle=gzip.GzipFile(filename=str(path),mode='wb',compresslevel=9,mtime=0)
                part+=1;compressed_budget=0;partrecords=[]
            handle.write(line);compressed_budget+=estimate;partrecords.append(j['identity'])
            records.append(dict(cohort=cohort,identity=j['identity'],numeric_source_sha256=source['sha256'],group=j['group'],
                arrays={k:({kk:vv for kk,vv in v.items() if kk!='base64'} if v is not None else None) for k,v in arrays.items()}))
        close_shard()
        print(json.dumps(dict(cohort=cohort,trajectories=len(cohort_jobs),shards=part)),flush=True)
    write(OUT/'shards.json',shards);write(OUT/'array_sha256_manifest.json',records)
    (OUT/'decode_audit_inputs.py').write_text(DECODER)
    (OUT/'README.md').write_text('''# 无损数值审计输入\n\n训练180、开发96、终测96共372轨迹，仅机械复制冻结analyze实际消费的17个数值dataset（缺失可选字段为null），不含视觉、原始HDF5、大权重或缓存。每行保留实际group、source SHA引用、metadata所需config/joint_order/FK身份、完整实际result JSON，以及数组dtype/shape/C-order原始字节base64和逐数组SHA。没有新增分析、阈值或选种。\n\n标准库验证：\n\n```bash\npython decode_audit_inputs.py . --verify-only\npython decode_audit_inputs.py . --out /tmp/tight_decoded\n```\n\n后者生成每轨迹的transition/<字段>.bin、manifest.json、metadata_selected.json和result.json；无需NumPy或h5py即可解码并校验每个原始数组字节。原始大HDF5的SHA引用前次已核验的source manifests，本导出不重新扫描原始视觉载荷。\n\n若需重建最小HDF5，使用NumPy/h5py：读取manifest中的dtype与shape，`a=np.frombuffer(raw,dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])`，在manifest记录的group下创建`transition/<字段>`。在HDF5根写metadata_json属性为metadata_selected.json文本，result_json属性为result.json文本。group、字段、dtype、shape和每个元素的位模式均可还原；缺失字段保持不创建。完整原始HDF5其他未消费字段、视觉帧、压缩布局和未选metadata不在此包，不能据此宣称重建了整份原始HDF5。冻结分析及URDF/任务几何定义仍使用仓库已登记源码和资产。\n\n`array_sha256_manifest.json`列出372条记录每数组的原始SHA；`shards.json`给每个gzip大小/SHA/身份；`verification.json`是标准库完整解码核验。\n''')
    write(OUT/'receipt.json',dict(trajectories=len(jobs),cohorts=dict(Counter(j['cohort'] for j in jobs)),arrays=array_count,raw_array_bytes=raw_bytes,
        optional_missing_fields=dict(missing),shards=len(shards),compressed_bytes=sum(s['bytes'] for s in shards),max_shard_bytes=max(s['bytes'] for s in shards),
        frozen_analyze_unchanged=sha(ANALYZE)==FROZEN_ANALYZE_SHA,original_sources_rehashed=False,original_sources_modified=False,terminal_analysis_or_threshold_selection=False,
        new_policy_inference=0,optimizer_updates=0,new_simulation=0,new_dagger=0,export_script_sha256=sha(__file__)))
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,PUBLIC/p.name)
    print(json.dumps(read(OUT/'receipt.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
