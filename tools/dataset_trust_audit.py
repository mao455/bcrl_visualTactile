"""Read-only full narrow-data audit and finite expert/replay acceptance. No student."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from tools.runtime import CODE, read, write, sha
from visuotactile.narrow_data import record, initial_row, angle, near
from visuotactile.success_audit import analyze, distribution


def prepare(root):
    parent=(CODE/'outputs/visuotactile/run/narrow_repair_20261106_01').resolve()
    old=read(parent/'session.json');root.mkdir(parents=True,exist_ok=False);now=time.time()
    session=dict(parent=str(parent),started=now,deadline=now+3*3600,collection_deadline=now+3*3600,
        expert_checkpoint=old['expert_checkpoint'],expert_sha256=old['expert_sha256'],
        resource_policy='shared_recent_v1',resource_target_percent=35.,reserve_bytes=60*1024**3,
        maximum_output_bytes=8*1024**3,maximum_kit_starts=60,maximum_hours=3,
        precision_policy='unified_fp32',camera_content_validation=True,
        formal_updates=0,student_rollouts=0,formal_collection=0,owned_worker_roots=[],
        audit_base='84cbb98',audit_head='0c209b9')
    write(root/'session.json',session);write(root/'budget.json',session)
    guards=read(CODE/'outputs/visuotactile/run/state_mechanism_20261106_01/protected.json')
    for name in ('manifest.json','families.json','state_data/data.json','state_data/statistics.pt'):
        guards[str(parent/name)]=sha(parent/name)
    write(root/'protected.json',guards)
    write(root/'status.json',dict(state='prepared',stage='static',owned_workers_exited=True))


def sensor_scan(g):
    n=len(g['obs/time_s']);invalid=np.zeros(3,dtype=int);mins=np.full(3,np.inf);coverage=np.ones(3)
    blank=np.zeros(3,dtype=int);mask_errors=0;rgb_digest=hashlib.sha256();depth_digest=hashlib.sha256()
    # Full pixels, every pre-action frame; no source acceptance files are written.
    for start in range(0,n,8):
        rgb=g['frames/rgb_u8'][start:start+8];depth=g['frames/depth_m'][start:start+8];valid=g['frames/depth_valid'][start:start+8]
        rgb_digest.update(rgb.tobytes());depth_digest.update(depth.tobytes())
        spatial=rgb.astype(np.float32).reshape(len(rgb),3,3,-1).std(-1).max(-1)
        expected=np.isfinite(depth)&(depth>0);fraction=expected.reshape(len(rgb),3,-1).mean(-1)
        mask_errors+=int(np.count_nonzero(valid!=expected));invalid+=((spatial<=1)|(fraction<=.01)).sum(0)
        blank+=(spatial==0).sum(0);mins=np.minimum(mins,spatial.min(0));coverage=np.minimum(coverage,fraction.min(0))
    clocks=float(abs(g['obs/camera_time_s'][:]-g['obs/time_s'][:][:,None]).max())
    frames_monotonic=bool((np.diff(g['obs/camera_frame'][:],axis=0)>0).all())
    tactile=[g[k][:] for k in ('obs/contact_w_N','obs/contact_substeps_w_N','obs/contact_substeps_local_N','transition/contact_substeps_after_w_N')]
    tactile_finite=all(np.isfinite(a).all() for a in tactile);flags=g['obs/contact_valid'][:]
    tactile_valid=bool(tactile_finite and not flags[0].any() and flags[1:].all())
    expected_contact=g['obs/time_s'][1:][:,None]+np.array([-1/120,0.])
    tactile_sync=float(abs(g['obs/contact_time_s'][1:]-expected_contact).max())
    return dict(frames=n,all_pixels_read=True,invalid_frames_by_view=invalid.tolist(),constant_rgb_frames=blank.tolist(),
        rgb_std_min=mins.tolist(),depth_valid_fraction_min=coverage.tolist(),depth_mask_errors=mask_errors,
        camera_clock_max_error_s=clocks,camera_frames_monotonic=frames_monotonic,
        rgbd_valid=bool(not invalid.any() and not mask_errors and clocks<1e-8 and frames_monotonic),
        wrist_valid=bool(invalid[2]==0 and not mask_errors and clocks<1e-8 and frames_monotonic),
        tactile_numeric_valid=tactile_valid,tactile_clock_max_error_s=tactile_sync,
        first_tactile_frame_invalid=True,tactile_nonzero_ticks=int((np.linalg.norm(tactile[0],axis=-1).max(-1)>.1).sum()),
        decoded_rgb_sha256=rgb_digest.hexdigest(),decoded_depth_sha256=depth_digest.hexdigest(),
        tactile_scope='Recorded finite force and clock/validity; not hardware calibration or stable grasp certification')


def flatten(x,prefix=''):
    out={}
    for k,v in x.items():
        path=prefix+k
        if isinstance(v,dict):out.update(flatten(v,path+'.'))
        else:out[path]=v
    return out


def static(root):
    s=read(root/'session.json');parent=Path(s['parent']);entries=read(parent/'manifest.json')['episodes']
    cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
    families={f['identity']:f for f in read(parent/'families.json')};center=read(parent/'session.json')['domain']['center']
    rows=[];config_values=defaultdict(dict);observed=defaultdict(list);errors=[]
    for index,e in enumerate(entries):
        try:
            assert sha(e['path'])==e['sha256'] and Path(e['path']).stat().st_size==e['size']
            key,m,rec=record(e['path']);n=len(rec['time']);cached=cache[e['id']]
            assert all(torch.equal(v,cached[k]) for k,v in rec.items())
            with h5py.File(e['path']) as f:
                g=f['episodes'][key];tr=g['transition'];b=tr['diagnostics_before'];a=tr['diagnostics_after']
                initial=initial_row(e['family'],{k:v[0] for k,v in b.items()},m['env_origins_w'][0])
                assert initial['fingerprint']==families[e['family']]['fingerprint']
                continuity=float(abs(a['joint_position_rad'][:-1]-b['joint_position_rad'][1:]).max());assert continuity==0
                success,_=analyze(g,m);assert success['passed'] and success['reconstructed_success']
                assert success['first_complete_tick']==n and np.count_nonzero(tr['termination_reasons/max_successes'][:])==1
                assert bool(g['terminated'][-1]) and not g['truncated'][:].any()
                speed=m['expert_speed'];assert speed==dict(kind='closed_loop',arm=e['alpha'],hand=e['alpha'],phase='all')
                previous=tr['target_before_rad'][:];native=tr['speed_native_target_rad'][:];target=tr['target_after_rad'][:]
                calculated=native if e['alpha']==1 else previous+e['alpha']*(native-previous)
                speed_error=float(abs(calculated-target).max());assert speed_error<1e-6
                sensory=sensor_scan(g);assert sensory['tactile_numeric_valid'] and sensory['tactile_clock_max_error_s']<1e-8
                assert sensory['rgbd_valid']==e['multi_view_valid'],'Actual visual validity differs from preserved manifest'
                dr={}
                for name in ('external_force_probability','external_torque_probability','external_force_scale','external_torque_scale','external_object_mass_kg'):
                    vals=tr[name][:].reshape(-1);dr[name]=[float(vals.min()),float(vals.max())];observed[name].extend(vals.tolist())
                for name in ('force_w_N','torque_w_Nm'):
                    v=tr['applied_wrench/'+name][:];dr[name+'_peak_norm']=float(np.linalg.norm(v,axis=-1).max())
                    dr[name+'_nonzero_ticks']=int((np.linalg.norm(v,axis=-1).reshape(n,-1).max(-1)>0).sum())
                delay=tr['control_trace/delay_index'][:];dr['delay_counts']=dict(Counter(str(int(v)) for v in delay.reshape(-1)))
                drift=float(np.sqrt(np.mean((target-a['joint_position_rad'][:])**2)))
                reset_match={k:bool(m['config']['reset'].get(k)==v) for k,v in families[e['family']]['spec']['narrow_reset'].items() if not k.startswith('hole_')}
                assert all(reset_match.values())
                assert m['expert_model_sha256']==s['expert_sha256'] and m['initial_family']==e['family'] and m['dataset_split']==e['split']
            for k,v in flatten(m['config']).items():
                token=json.dumps(v,sort_keys=True);config_values[k].setdefault(token,[]).append(e['id'])
            row=dict(id=e['id'],family=e['family'],split=e['split'],origin='new75' if e['multi_view_valid'] else 'old200',
                alpha=e['alpha'],steps=n,duration_s=n/60,sha256=e['sha256'],bytes=e['size'],initial=initial,
                initial_rotation_deg=angle(initial['quaternion'],center['quaternion']),state_valid=True,
                cache_exact=True,q_continuity_max=continuity,slow_formula_max=speed_error,target_vs_q_rmse=drift,
                success=success,sensors=sensory,disturbances=dr,metadata_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest())
            rows.append(row);write(root/'episodes'/f'{e["id"]}.json',row)
        except BaseException as exc:
            errors.append(dict(identity=e['id'],error=repr(exc),traceback=traceback.format_exc()))
            write(root/'static_errors.json',errors)
        write(root/'heartbeat.json',dict(stage='static',checked=index+1,total=len(entries),errors=len(errors),utc=time.time()))
    assert not errors, 'Static audit failures preserved; physical stage must not start'
    configs={k:[dict(value=json.loads(v),count=len(ids),ids=ids) for v,ids in vs.items()] for k,vs in config_values.items()}
    write(root/'configurations.json',configs)
    finish(root, rows)


def finish(root, rows=None):
    parent=Path(read(root/'session.json')['parent']);entries=read(parent/'manifest.json')['episodes']
    families={f['identity']:f for f in read(parent/'families.json')}
    if rows is None:
        progress=read(root/'heartbeat.json');assert progress['checked']==275 and progress['errors']==0
        rows=[read(root/'episodes'/f'{e["id"]}.json') for e in entries]
        assert all(r['sha256']==e['sha256'] for r,e in zip(rows,entries))
    observed=defaultdict(list)
    for row in rows:
        for k,v in row['disturbances'].items():
            if k.startswith('external_'):observed[k].append(v[0])
    byfamily=defaultdict(list)
    for e in rows:byfamily[e['family']].append(e)
    pairs=[];unpaired=[]
    for fam,rr in byfamily.items():
        assert len({x['split'] for x in rr})==1
        native=[x for x in rr if x['alpha']==1]
        for x in rr:
            if x['alpha']==1:continue
            if native:
                # Multiple native captures are all retained; report every pair and median-reference ratio.
                ratios=[x['steps']/a['steps'] for a in native]
                pairs.append(dict(id=x['id'],family=fam,alpha=x['alpha'],native_ids=[a['id'] for a in native],
                    duration_ratios=ratios,median_reference_ratio=x['steps']/np.median([a['steps'] for a in native]),
                    matched_observable_initial=True,hidden_state_exact=False))
            else:unpaired.append(x['id'])
    families_rows=[]
    for fam,rr in byfamily.items():
        families_rows.append(dict(families[fam],episodes=len(rr),alphas=[x['alpha'] for x in rr]))
    write(root/'families_actual.json',families_rows)
    summary=dict(passed=True,episodes=len(rows),families=len(byfamily),ticks=sum(x['steps'] for x in rows),
        state_valid=len(rows),rgbd_valid=sum(x['sensors']['rgbd_valid'] for x in rows),wrist_valid=sum(x['sensors']['wrist_valid'] for x in rows),
        tactile_numeric_valid=sum(x['sensors']['tactile_numeric_valid'] for x in rows),
        duration=distribution([x['duration_s'] for x in rows]),duration_5_to_7=sum(5<=x['duration_s']<=7 for x in rows),
        native_count=sum(x['alpha']==1 for x in rows),slow_count=sum(x['alpha']<1 for x in rows),
        paired_slow=len(pairs),unpaired_slow=unpaired,paired_ratio=distribution([x['median_reference_ratio'] for x in pairs]),
        paired_longer=int(sum(x['median_reference_ratio']>1 for x in pairs)),paired_ge_1p4=int(sum(x['median_reference_ratio']>=1.4 for x in pairs)),
        family_counts=dict(Counter(x['episodes'] for x in families_rows)),split_episodes=dict(Counter(x['split'] for x in rows)),
        split_families=dict(Counter(x['split'] for x in families_rows)),dr_observed={k:distribution(v) for k,v in observed.items()},
        duplicate_decoded_rgb=len(rows)-len({x['sensors']['decoded_rgb_sha256'] for x in rows}),
        grouped_duration={f'{o}/{a}':distribution([x['duration_s'] for x in rows if x['origin']==o and (x['alpha']==1 if a=='native' else x['alpha']<=.65 if a=='slow_le_065' else .65<x['alpha']<1)]) for o in ('old200','new75') for a in ('native','slow_le_065','mid')},
        completed=time.time())
    write(root/'pairs.json',pairs);write(root/'static_summary.json',summary)
    write(root/'status.json',dict(state='static_complete',stage='static',owned_workers_exited=True,utc=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','static','finish'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    try:globals()[a.mode](root)
    except BaseException:
        write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()));raise
