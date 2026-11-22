#!/usr/bin/env python3
"""Optional NumPy/h5py minimal numerical-HDF5 reconstruction (no simulation)."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import h5py
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort');p.add_argument('--identity');a=p.parse_args();written=[]
    for shard in sorted(a.directory.glob('*.jsonl.gz')):
        with gzip.open(shard,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if a.cohort and r['cohort']!=a.cohort:continue
                if a.identity and r['identity']!=a.identity:continue
                path=a.out/r['cohort']/(r['identity']+'.h5');path.parent.mkdir(parents=True,exist_ok=True)
                # Exclusive-create protects any previous reconstruction.
                with h5py.File(path,'x') as h:
                    h.attrs['metadata_json']=r['metadata_selected_json'];h.attrs['result_json']=r['result_json']
                    h.attrs['reconstruction_role']='minimal frozen-analyze numerical inputs; not full original HDF5'
                    tr=h.require_group(r['group']+'/transition')
                    for name,spec in r['datasets'].items():
                        if spec is None:continue
                        raw=base64.b64decode(spec['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==spec['raw_sha256']
                        data=np.frombuffer(raw,dtype=np.dtype(spec['dtype'])).reshape(spec['shape']);tr.create_dataset(name,data=data)
                written.append(str(path))
    if not written:raise ValueError('No matching trajectory')
    print(json.dumps(dict(written=len(written),files=written),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
