"""Supplemental train/dev-only hand-geometry command proxy.

This does not change the frozen trajectory audit or define a native open action.
The CLI reads only sources already registered by that audit, never terminal data.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_premature_scope_20261122.trajectory_audit import (
    PROJECT, OUT, PUBLIC, read, write, sha, rotate, relative, event, first_run)
from visuotactile.kinematics import EncoderKinematics, FINGERS


def main():
    dest=OUT/'supplemental_command_withdrawal'; dest.mkdir(exist_ok=True)
    pub=PUBLIC/'supplemental_command_withdrawal';pub.mkdir(exist_ok=True)
    registration=dict(registered_at=datetime.now(timezone.utc).isoformat(),sources='already frozen train180/development96 only',
        terminal_reads=0,changes_primary_analyze=False,primary_code_sha256=sha(PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py'),
        reference='same last stable contact frame before first gate as primary audit; no-gate closest-approach fallback labelled separately',
        geometry='fix7 arm joints to actual reference q; fix original object root relative to reference palm; vary ONLY22 final hand target joints; URDF distal link-frame origin distances to fixed object root',
        proxy='mean5 distal-link origin distances to reference object-root grows>=5mm from reference final command and remains>=5mm for3ticks',
        actual_hand_comparator='same fixed-arm/fixed-object construction, substituting actual hand joints and actual-reference-hand baseline',
        meaning='supplemental command hand geometry withdrawal signal; not native open action, object-surface separation, collision simulation, or causal intentional release',
        threshold_mm=5.,sustain_ticks=3.,native_open_target='not found; remains unknown')
    write(dest/'registration.json',registration)
    rows=read(OUT/'trajectories.json'); summaries=[]; csvf=gzip.open(dest/'derived_ticks.csv.gz','wt',newline='');writer=None
    try:
        for i,row in enumerate(rows):
            assert row['cohort'] in ('train','development')
            path=Path(row['path']);ref=row['stable_reference_tick']; entry=row['gate_first_entry_tick']
            out=dict(identity=row['identity'],cohort=row['cohort'],policy=row['policy'],native_success=row['native_success'],
                     native_outcome=row['native_outcome'],category=row['exclusive_category'],reference_tick=ref,
                     reference_scope=row['stable_reference_scope'],known=ref is not None)
            if ref is None:
                out.update(command_proxy=event(None),actual_hand_proxy=event(None),command_after_gate=event(None),actual_hand_after_gate=event(None))
                summaries.append(out);continue
            with h5py.File(path,'r') as f:
                tr=f[row['group']+'/transition'];meta=json.loads(f.attrs['metadata_json'])
                qa=tr['diagnostics_after/joint_position_rad'][:];qc=tr['target_after_rad'][:]
                palm=tr['diagnostics_after/palm_body_pose_w'][ref];obj=tr['diagnostics_after/object_pose_w'][ref]
                n=len(qa);urdf=Path(meta['config']['assets']['robot_urdf'])
                if not urdf.exists():urdf=PROJECT/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf'
                fk=EncoderKinematics(urdf,meta['joint_order'],links=('iiwa14_link_7',)+FINGERS)
                assert fk.sha256==meta['fk_urdf_sha256'] and fk.root==meta['fk_root']
                fixed_c=qc.copy();fixed_a=qa.copy();fixed_c[:,:7]=qa[ref,:7];fixed_a[:,:7]=qa[ref,:7]
                with torch.no_grad():
                    pc,rc=fk(torch.as_tensor(fixed_c,dtype=torch.float32));pa,ra=fk(torch.as_tensor(fixed_a,dtype=torch.float32))
                pc,rc,pa,ra=pc.numpy().astype(float),rc.numpy().astype(float),pa.numpy().astype(float),ra.numpy().astype(float)
                # FK palm fixed by arm; transform the measured held-object root
                # through its recorded relative-palm position into that frame.
                relobj=relative(palm[None],obj[None])[0,:3]
                obj_fixed=pa[ref,0]+ra[ref,0]@relobj
                dc=np.linalg.norm(pc[:,1:]-obj_fixed[None,None],axis=-1)*1000
                da=np.linalg.norm(pa[:,1:]-obj_fixed[None,None],axis=-1)*1000
                growc=dc.mean(-1)-dc[ref].mean();growa=da.mean(-1)-da[ref].mean()
                ticks=np.arange(n);pre=ticks<(row['native_first_subgoal_tick'] if row['native_first_subgoal_tick'] is not None else n)
                mask=pre&(ticks>ref);aftergate=mask&(ticks>=entry) if entry is not None else np.zeros(n,bool)
                ec=event((growc>=5)&mask,3);ea=event((growa>=5)&mask,3)
                gc=event((growc>=5)&aftergate,3);ga=event((growa>=5)&aftergate,3)
                for ev in (ec,ea,gc,ga):
                    for label in ('onset_tick','confirmation_tick'):
                        t=ev[label];ev[label.replace('_tick','_context')]=dict(command_growth_mm=float(growc[t]),actual_hand_growth_mm=float(growa[t]),
                            command_mean_distance_mm=float(dc[t].mean()),actual_hand_mean_distance_mm=float(da[t].mean())) if t is not None else None
                out.update(command_proxy=ec,actual_hand_proxy=ea,command_after_gate=gc,actual_hand_after_gate=ga,
                           maximum_command_growth_before_confirmation_mm=float(growc[mask].max()) if mask.any() else None,
                           maximum_actual_hand_growth_before_confirmation_mm=float(growa[mask].max()) if mask.any() else None,
                           urdf_sha256=fk.sha256)
                summaries.append(out)
                if writer is None:
                    writer=csv.DictWriter(csvf,fieldnames=['identity','tick','reference_tick','native_first_phase','command_mean_dist_fixed_object_mm','actual_hand_mean_dist_fixed_object_mm','command_growth_mm','actual_hand_growth_mm']+[f'command_{f}_dist_fixed_object_mm' for f in FINGERS]+[f'actual_hand_{f}_dist_fixed_object_mm' for f in FINGERS]);writer.writeheader()
                for t in range(n):
                    record=dict(identity=row['identity'],tick=t,reference_tick=ref,native_first_phase=bool(pre[t]),command_mean_dist_fixed_object_mm=float(dc[t].mean()),
                                actual_hand_mean_dist_fixed_object_mm=float(da[t].mean()),command_growth_mm=float(growc[t]),actual_hand_growth_mm=float(growa[t]))
                    record.update({f'command_{f}_dist_fixed_object_mm':float(dc[t,j]) for j,f in enumerate(FINGERS)})
                    record.update({f'actual_hand_{f}_dist_fixed_object_mm':float(da[t,j]) for j,f in enumerate(FINGERS)})
                    writer.writerow(record)
            if i%50==49:print('progress',i+1,flush=True)
    finally:csvf.close()
    write(dest/'trajectories.json',summaries)
    summary={}
    for cohort in ('train','development'):
        pool=[s for s in summaries if s['cohort']==cohort];success=[s for s in pool if s['native_success']]
        summary[cohort]=dict(n=len(pool),known=sum(s['known'] for s in pool),
            command_after_gate=sum(s['command_after_gate']['onset_tick'] is not None for s in pool),
            actual_hand_after_gate=sum(s['actual_hand_after_gate']['onset_tick'] is not None for s in pool),
            native_success_denominator=len(success),command_after_gate_in_success=sum(s['command_after_gate']['onset_tick'] is not None for s in success),
            actual_hand_after_gate_in_success=sum(s['actual_hand_after_gate']['onset_tick'] is not None for s in success))
    write(dest/'summary.json',summary)
    flat=[]
    for r in summaries:
        q={k:r[k] for k in ('identity','cohort','policy','native_outcome','category','reference_tick','reference_scope','known')}
        for name in ('command_proxy','actual_hand_proxy','command_after_gate','actual_hand_after_gate'):
            for key in ('known','onset_tick','confirmation_tick'):q[name+'_'+key]=r[name][key]
        flat.append(q)
    with (dest/'trajectories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(dest/'receipt.json',dict(source_trajectories=len(rows),terminal_reads=0,optimizer_updates=0,new_inference=0,new_simulation=0,new_dagger=0,
         primary_analyze_unchanged=sha(PROJECT/'scripts/tight_premature_scope_20261122/trajectory_audit.py')==registration['primary_code_sha256'],script_sha256=sha(__file__)))
    for p in dest.iterdir():shutil.copy2(p,pub/p.name)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
