"""Measure HDF5 allocation, not decoded-array size. No simulator or training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import h5py
import numpy as np
from .compact import sha256
from .trial_utils import write_json


def category(field):
    """Disjoint allocation groups; old compressed RGB-D chunks cannot be split."""
    if field.endswith('/image'):
        return 'packed_RGB_and_depth_not_separable'
    if field.endswith('/rgb_u8'):
        return 'lossless_RGB_uint8'
    if field.endswith('/depth_m'):
        return 'metric_depth_float32'
    if field.endswith('/depth_valid'):
        return 'depth_mask'
    if 'torch_rng' in field:
        return 'rng_diagnostic'
    if 'contact' in field:
        return 'tactile_and_timestamps'
    if 'finger_' in field or 'fk_' in field:
        return 'FK_and_joint_samples'
    if 'diagnostics_' in field or 'external_' in field or 'applied_wrench' in field or 'actor_input' in field:
        return 'state_and_disturbance_diagnostic'
    if any(key in field for key in ('action','command','target','tracking_error','control_trace','provenance','prior_plan','previous_nominal')):
        return 'action_execution_provenance'
    return 'other_sensor_lifecycle_metadata'


def allocation_groups(fields, overhead):
    result=defaultdict(int)
    for name,value in fields.items():
        result[category(name)]+=value
    result['container_overhead']=overhead
    return dict(result)


def audit_file(path):
    fields = {}
    with h5py.File(path) as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                fields[name] = dict(shape=list(obj.shape),dtype=str(obj.dtype),compression=obj.compression,
                    compression_opts=obj.compression_opts,shuffle=obj.shuffle,chunks=obj.chunks,
                    logical_bytes=int(obj.size*obj.dtype.itemsize),storage_bytes=int(obj.id.get_storage_size()))
        f.visititems(visit)
        g = next(iter(f['episodes'].values()))
        n = int(g.attrs['length'])
        complete = bool(g.attrs['complete'])
        success = bool(complete and g['transition/diagnostics_after/successes'][n-1]>=g['transition/diagnostics_after/max_goals'][n-1]
                       and g['transition/diagnostics_after/retract_succeeded'][n-1])
    size = Path(path).stat().st_size
    allocated = sum(x['storage_bytes'] for x in fields.values())
    return dict(bytes=size,steps=n,complete=complete,success=success,bytes_per_step=size/n,
                container_overhead_bytes=size-allocated,fields=fields)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.root/'hdf5_manifest.json').read_text())
    rows={}
    for name, expected in manifest.items():
        path=a.root/name
        if path.stat().st_size != expected['bytes'] or sha256(path)!=expected['sha256']:
            raise ValueError('Old manifest checksum mismatch: '+name)
        rows[name]=audit_file(path)
    pilot={k:v for k,v in rows.items() if k.startswith('pilot/')}
    sums=defaultdict(int)
    for record in pilot.values():
        for field,x in record['fields'].items():
            sums[field.split('/',2)[2]]+=x['storage_bytes']
    report=dict(files=rows,pilot_bytes=sum(x['bytes'] for x in pilot.values()),
        pilot_steps=sum(x['steps'] for x in pilot.values()),pilot_field_storage_bytes=dict(sums),
        pilot_overhead_bytes=sum(x['container_overhead_bytes'] for x in pilot.values()),
        outcomes={str(s):dict(count=sum(x['success']==s for x in pilot.values()),
            bytes=sum(x['bytes'] for x in pilot.values() if x['success']==s),
            steps=sum(x['steps'] for x in pilot.values() if x['success']==s)) for s in (True,False)})
    report['pilot_allocation_groups_bytes']=allocation_groups(sums,report['pilot_overhead_bytes'])
    write_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','pilot_field_storage_bytes')},indent=2))


if __name__=='__main__':main()
