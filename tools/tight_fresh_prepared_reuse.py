"""Reuse a completed aggregate and committed frozen cache before any training."""
import argparse,json,shutil,time
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

def reuse(source,root,collection,output):
    read=lambda p:json.loads(p.read_text())
    if output.exists():raise FileExistsError('Never replace prepared or trained output')
    previous=read(source/'prepared.json');session=read(root/'session.json')
    if previous['collection_plan_sha256']!=sha256(collection/'plan.json'):
        raise ValueError('Prepared data belongs to another collection')
    if not read(collection/'result.json')['software_complete']:raise ValueError('Collection incomplete')
    if read(source/'session.json')['deadline']!=session['deadline']:raise ValueError('No deadline extension')
    if sha256(source/'metadata.json')!=sha256(root/'metadata.json'):raise ValueError('Physical task metadata changed')
    for role in ('A','B'):
        if (source/role/'training').exists():raise ValueError('This recovery is only before any optimizer run')
    failure=read(source.with_name(source.name+'_stages')/'failure.json')
    if 'visual_cache' not in str(failure):raise ValueError('Expected interrupted pre-training cache stage')
    output.mkdir();bindings={}
    def copy_file(src,dst):
        digest=sha256(src);shutil.copy2(src,dst)
        if sha256(dst)!=digest:raise ValueError('Prepared copy changed')
        bindings[str(src)]=digest
    for name in ('metadata.json','gate.json'):copy_file(source/name,output/name)
    atomic_json(output/'session.json',dict(session,resource_root=str(root)))
    (output/'shared_resources.json').symlink_to(root/'shared_resources.json')
    for role in ('A','B'):
        dst=output/role;dst.mkdir();(dst/'state_data').mkdir()
        for src in (source/role/'state_data').iterdir():
            if src.is_file():copy_file(src,dst/'state_data'/src.name)
        copy_file(source/role/'continuation.json',dst/'continuation.json')
        for name in ('cache_acceptance_index.json','cache_progress_index.json','cache_index.json'):
            src=source/role/name
            if src.exists():copy_file(src,dst/name)
        atomic_json(dst/'session.json',dict(session,resource_root=str(output),cache_deadline=session['deadline']))
        (dst/'shared_resources.json').symlink_to(root/'shared_resources.json')
    atomic_json(output/'prepared.json',dict(previous,reused_prepared=dict(source=str(source),
        source_prepared_sha256=sha256(source/'prepared.json'),copied_bindings=bindings,utc=time.time(),
        optimizer_updates_repeated=0,frozen_cache_paths_preserved=True,
        original_reserve_bytes=read(source/'session.json')['reserve_bytes'],reserve_bytes=session['reserve_bytes'])))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','root','collection','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();reuse(a.source.resolve(),a.root.resolve(),a.collection.resolve(),a.output.resolve())
