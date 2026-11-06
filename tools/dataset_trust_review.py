"""Provenance, selection-bias and coverage receipts for the frozen State dataset."""
import argparse
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch

from tools.runtime import CODE,read,write,sha
from visuotactile.narrow_data import initial_row,quantiles,near
from visuotactile.success_audit import analyze,distribution


def git(*args):return subprocess.check_output(['git',*args],cwd=CODE,text=True).strip()


def functions(source):
    tree=ast.parse(source);out={}
    def visit(nodes,prefix=''):
        for n in nodes:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                out[prefix+n.name]=ast.dump(n,include_attributes=False)
            elif isinstance(n,ast.ClassDef):visit(n.body,prefix+n.name+'.')
    visit(tree.body);return out


def source(root):
    base='84cbb98';head='0c209b9';digest=lambda x:hashlib.sha256(x.encode()).hexdigest()
    names=git('ls-tree','-r','--name-only',head,'--','visuotactile','tools','isaacsimenvs','evaluation','rl_games','assets').splitlines()
    oldnames=set(git('ls-tree','-r','--name-only',base,'--','visuotactile','tools','isaacsimenvs','evaluation','rl_games','assets').splitlines())
    changes=git('diff','--name-status',base,head,'--','visuotactile','tools','isaacsimenvs','evaluation','rl_games','assets','tests').splitlines()
    changed={x.split('\t')[-1] for x in changes};rows=[]
    for path in names:
        # Git blob IDs establish unchanged binary assets without reading large meshes twice.
        before=git('rev-parse',base+':'+path) if path in oldnames else None;after=git('rev-parse',head+':'+path)
        row=dict(path=path,base_blob=before,head_blob=after,unchanged=before==after)
        if path.endswith('.py') and (path.startswith('visuotactile/') or path in changed):
            a=git('show',head+':'+path)+'\n';actual=(CODE/path).read_text()
            row['current_sha256']=sha(CODE/path);row['head_sha256']=hashlib.sha256(subprocess.check_output(['git','show',head+':'+path],cwd=CODE)).hexdigest()
            assert row['current_sha256']==row['head_sha256'],'Audit HEAD drift '+path
            if before:
                b=git('show',base+':'+path)+'\n';old=functions(b);new=functions(a)
                row.update(unchanged_functions=[k for k in old.keys()&new.keys() if old[k]==new[k]],
                    changed_functions=[k for k in old.keys()&new.keys() if old[k]!=new[k]],
                    added_functions=sorted(new.keys()-old.keys()),removed_functions=sorted(old.keys()-new.keys()))
        rows.append(row)
    # Original320 sampler behavior, including RNG stream and checkpoint state.
    namespace={};exec(git('show',base+':visuotactile/anchor_sampling.py'),namespace)
    from visuotactile.anchor_sampling import AnchorSampler
    lengths={f'e{i}':145+i%600 for i in range(320)}
    rg=[torch.Generator().manual_seed(902) for _ in range(2)]
    a=namespace['AnchorSampler'](lengths,rg[0],64);b=AnchorSampler(lengths,rg[1],64)
    for _ in range(160):assert a.draw()==b.draw() and a.state_dict()==b.state_dict() and torch.equal(rg[0].get_state(),rg[1].get_state())
    old_rejected=False
    try:namespace['AnchorSampler'](dict(list(lengths.items())[:148]),torch.Generator(),64)
    except ValueError:old_rejected=True
    assert old_rejected
    runtime=[]
    inventories=[(CODE/'outputs/visuotactile/native_dp_v2/state_random_20261105_02/runtime_freeze.json','100d71e'),
        (CODE/'outputs/visuotactile/run/narrow_state_20261105_02/source_freeze.json','b119470')]
    inventories += [(p,None) for p in (CODE/'outputs/visuotactile/run/narrow_repair_20261106_01/phases').glob('*/source.json')]
    for path,commit in inventories:
        d=read(path);compared=[]
        for name,h in d.items():
            rel=next((name[name.index(tag):] for tag in ('visuotactile/','tools/','isaacsimenvs/','evaluation/','rl_games/') if tag in name),None)
            if rel is None:continue
            item=dict(path=rel,runtime_sha256=h,current_equal=(CODE/rel).exists() and sha(CODE/rel)==h)
            if commit:
                res=subprocess.run(['git','show',commit+':'+rel],cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
                item['pinned_commit_equal']=res.returncode==0 and hashlib.sha256(res.stdout).hexdigest()==h
            compared.append(item)
        runtime.append(dict(inventory=str(path.relative_to(CODE)),sha256=sha(path),expected_commit=commit,files=compared))
    write(root/'source_audit.json',dict(base=git('rev-parse',base),head=git('rev-parse',head),
        baseline_runtime_commit=git('rev-parse','100d71e'),changes=changes,files=rows,runtime_inventories=runtime,
        sampler_original320_exact_updates=160,sampler_rng_exact=True,old148_rejected=True,
        note='84cbb98 is last pre-narrow collection tree;100d71e is verified runtime before opt-in expert slowdown. Correctness scope is contracts, not successful student behavior.',
        native_asset_changed=[x for x in changes if x.split('\t')[-1].startswith(('isaacsimenvs/','evaluation/','rl_games/','assets/'))]))


def attempts(root):
    parent=Path(read(root/'session.json')['parent']);old=CODE/'outputs/visuotactile/run/narrow_state_20261105_02';reports={};failures=[]
    for label,directory in [('old200',old),('new75',parent)]:
        if label=='old200':paths=sorted((directory/'episodes').glob('*/result.json'))
        else:
            paths=[Path(read(parent/'camera_expert.json')['result']['dataset_capture']['path']).parents[1]/'result.json']+sorted((directory/'episodes').glob('*/result.json'))
            paths=list(dict.fromkeys(paths))
        rr=[]
        for path in paths:
            x=read(path);classification='preaction_rejection' if not x['legal'] else 'expert_task_failure' if not x['success'] else 'accepted_success'
            rr.append(dict(identity=x['identity'],family=x['family'],alpha=x.get('alpha',1.),legal=x['legal'],success=x['success'],
                classification=classification,reasons=x.get('reasons'),steps=x.get('steps'),result_sha256=sha(path)))
            if classification=='expert_task_failure':
                f=Path(x['dataset_capture']['path'])
                with h5py.File(f) as h:
                    m=json.loads(h.attrs['metadata_json']);g=next(iter(h['episodes'].values()));a,_=analyze(g,m)
                    tr=g['transition'];initial=initial_row(x['family'],{k:v[0] for k,v in tr['diagnostics_before'].items()},m['env_origins_w'][0])
                    target=tr['target_after_rad'][:];native=tr['speed_native_target_rad'][:];before=tr['target_before_rad'][:]
                    alpha=x.get('alpha',1.);pred=native if alpha==1 else before+alpha*(native-before)
                    evidence=dict(target_formula_max=float(abs(target-pred).max()),actuator_max=float(abs(tr['actuator_substep_target_rad'][:]-target[:,None,:]).max()),
                        continuity_max=float(abs(tr['diagnostics_after/joint_position_rad'][:-1]-tr['diagnostics_before/joint_position_rad'][1:]).max()),
                        delay_values=np.unique(tr['control_trace/delay_index'][:]).tolist(),
                        object_z_min=float(tr['diagnostics_after/object_pose_w'][:,2].min()))
                failures.append(dict(origin=label,identity=x['identity'],family=x['family'],alpha=alpha,source=str(f),
                    result_sha256=sha(path),initial=initial,success_audit=a,control=evidence,initial_legal=read(path.parent/'initial_collision.json')['legal']))
        reports[label]=dict(attempts=len(rr),classification=dict(Counter(x['classification'] for x in rr)),rows=rr)
    write(root/'attempts.json',reports);write(root/'failures.json',failures)
    # Historical screening is separate from the accepted-capture denominator.
    v1=CODE/'outputs/visuotactile/run/narrow_state_20261105_01'
    historical=dict(v1_status=read(v1/'status.json'),v1_files=[str(p.relative_to(v1)) for p in v1.glob('*.json')],
        speed_study_summary=read(CODE/'outputs/visuotactile/native_dp_v2/expert_speed_20261105_01/summary.json'),
        source_selection=read(CODE/'research_docs/evidence/narrow_selection.json'))
    write(root/'historical_screening.json',historical)


def coverage(root):
    s=read(root/'session.json');parent=Path(s['parent']);old=CODE/'outputs/visuotactile/run/narrow_state_20261105_02'
    families=read(parent/'families.json');byid={f['identity']:f for f in families};center=read(parent/'session.json')['domain']['center'];candidates=[]
    for cohort,directory in [('old',old),('new',parent)]:
        for path in sorted((directory/'initials').glob('*/initial_observable.json')):
            obs={k:np.asarray(v)[0] for k,v in read(path).items()};name=path.parent.name
            row=initial_row(name,obs,np.zeros(3));row.update(cohort=cohort,legal=read(path.parent/'initial_collision.json')['legal'],selected=name in byid)
            candidates.append(row)
    lo=np.r_[np.asarray(center['object_xyz'][:2])-.025,np.asarray(center['relative_xyz'][:2])-.035]
    hi=np.r_[np.asarray(center['object_xyz'][:2])+.025,np.asarray(center['relative_xyz'][:2])+.035]
    def feature(f):return np.r_[f['object_xyz'][:2],f['relative_xyz'][:2]]
    def occupancy(rows,bins):
        cells=np.floor((np.array([feature(f) for f in rows])-lo)/(hi-lo)*bins).astype(int).clip(0,bins-1)
        return dict(occupied=len(set(map(tuple,cells))),grid=bins**4,dimensions='objectXY + hole-minus-objectXY, marginal grid diagnostic, not continuous-volume coverage')
    nearest=[]
    for i,a in enumerate(families):
        nearest.append(min(np.linalg.norm((feature(a)-feature(b))/(hi-lo)) for j,b in enumerate(families) if j!=i))
    report=dict(configured_position_box=dict(lower=lo.tolist(),upper=hi.tolist()),quantile_order=[0,.1,.5,.9,1],
        actual_family_quantiles=quantiles(families,center),
        cohorts={c:dict(examined=sum(x['cohort']==c for x in candidates),legal=sum(x['cohort']==c and x['legal'] for x in candidates),
            selected=sum(x['cohort']==c and x['selected'] for x in candidates),
            all_quantiles=quantiles([x for x in candidates if x['cohort']==c],center),
            rejected_quantiles=quantiles([x for x in candidates if x['cohort']==c and not x['legal']],center)) for c in ('old','new')},
        grid2=occupancy(families,2),grid3=occupancy(families,3),grid4=occupancy(families,4),
        nearest_4d_normalized_distance=distribution(nearest),
        exact_initial_duplicates=len(families)-len({x['fingerprint'] for x in families}),
        near_pairs=[[a['identity'],b['identity']] for i,a in enumerate(families) for b in families[:i] if near(a,b)],
        selected_semantics='Old first60 admitted of68 draws; new geometry max-min32 from74 admitted of80. Neither is uniform92 iid successes.',
        candidates=candidates)
    write(root/'coverage.json',report)


def details(root):
    r=root;parent=Path(read(r/'session.json')['parent']);rows=[];exports=[]
    for e in read(parent/'manifest.json')['episodes']:
     with h5py.File(e['path']) as h:
      g=next(iter(h['episodes'].values()));tr=g['transition'];before=tr['diagnostics_before'];after=tr['diagnostics_after']
      q=after['joint_position_rad'][:];q0=before['joint_position_rad'][:];v=abs(q-q0)*60
      palm=np.linalg.norm(after['palm_body_pose_w'][:,:3]-before['palm_body_pose_w'][:,:3],axis=-1)*60
      obj=np.linalg.norm(after['object_pose_w'][:,:3]-before['object_pose_w'][:,:3],axis=-1)*60
      rows.append(dict(id=e['id'],family=e['family'],alpha=e['alpha'],duration_s=len(q)/60,arm_actual_velocity_p95=float(np.quantile(v[:,:7],.95)),hand_actual_velocity_p95=float(np.quantile(v[:,7:],.95)),palm_velocity_p95=float(np.quantile(palm,.95)),object_velocity_p95=float(np.quantile(obj,.95)),low_motion_fraction=float(((v[:,:7].max(-1)<.01)&(v[:,7:].max(-1)<.02)&(palm<.005)).mean()),arm_motion_rad=float(abs(q-q0)[:,:7].sum()),hand_motion_rad=float(abs(q-q0)[:,7:].sum()),palm_path_m=float(palm.sum()/60),object_path_m=float(obj.sum()/60)))
     p=Path(e['path']).parents[1]/'initial_export.pt'
     x=torch.load(p,map_location='cpu',weights_only=False);env=x.get('env_tensors',{})
     fields={k:v.tolist() if torch.is_tensor(v) else v for k,v in env.items() if any(w in k for w in ('noise','force','torque','delay','mass','friction'))}
     exports.append(dict(id=e['id'],sha256=sha(p),top_keys=list(x),missing=x.get('missing'),fields=fields))
    lookup={x['id']:x for x in rows};pairs=[]
    for p in read(r/'pairs.json'):
     a=lookup[p['id']];ratios={}
     for k in ('arm_actual_velocity_p95','hand_actual_velocity_p95','palm_velocity_p95','object_velocity_p95','palm_path_m','object_path_m','low_motion_fraction'):
      base=np.median([lookup[i][k] for i in p['native_ids']]);ratios[k]=float(a[k]/base) if base else None
     pairs.append(dict(id=p['id'],alpha=p['alpha'],duration_ratio=p['median_reference_ratio'],ratios=ratios))
    write(r/'motion.json',dict(rows=rows,pairs=pairs,paired_ratio_summary={k:distribution([p['ratios'][k] for p in pairs if p['ratios'][k] is not None]) for k in pairs[0]['ratios']},low_motion_definition='all arm <.01rad/s, all hand <.02rad/s, palm <.005m/s, finite-difference actual movement per native tick; not task-stage label'))
    write(r/'initial_exports.json',exports)
    ledger=[];speed=CODE/'outputs/visuotactile/native_dp_v2/expert_speed_20261105_01'
    attempts=read(speed/'attempts.json');print('SPEED_ATTEMPTS_TYPE',type(attempts).__name__,list(attempts)[:3] if isinstance(attempts,dict) else len(attempts))
    for p in speed.glob('stages/*/attempt_*/output/result.json'):
     d=read(p)
     for a in d.get('trials',[]):
      ledger.append(dict(stage=p.parents[2].name,identity=a['identity'],legal=a['legal'],success=a['success'],reasons=a.get('reasons'),steps=a.get('steps'),sha256=sha(p)))
    write(r/'historical_attempt_ledger.json',dict(speed_attempts=attempts,speed_numeric_results=ledger,result_counts=dict(Counter('illegal' if not a['legal'] else 'success' if a['success'] else 'task_failure' for a in ledger)),
     v1_results=[dict(path=str(p.relative_to(CODE)),sha256=sha(p),result=read(p)) for p in (CODE/'outputs/visuotactile/run/narrow_state_20261105_01/episodes').glob('*/result.json')],
     v1_selection=read(CODE/'outputs/visuotactile/run/narrow_state_20261105_01/selection.json')))
    # Verify every source SHA against the revision recorded by each actual repair phase.
    receipts=[]
    for p in (parent/'phases').glob('*/source.json'):
     budget=read(p.parent/'budget.json');commit=budget['source_commit'];bad=[];count=0
     for name,digest in read(p).items():
      rel=next((name[name.index(tag):] for tag in ('visuotactile/','tools/','isaacsimenvs/','evaluation/') if tag in name),None)
      if rel is None:continue
      q=subprocess.run(['git','show',commit+':'+rel],cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);count+=1
      if q.returncode or hashlib.sha256(q.stdout).hexdigest()!=digest:bad.append(rel)
     receipts.append(dict(phase=p.parent.name,commit=commit,files=count,mismatches=bad))
    write(r/'runtime_commits.json',receipts)
    print('DONE',len(rows),len(exports),len(ledger),receipts)
    # Earlier gates ran dirty source trees; resolve hashes rather than trusting HEAD.
    commits=git('rev-list','84cbb98..0c209b9').splitlines()
    for receipt in receipts:
        receipt['resolved_runtime_mismatches']=[]
        freeze=read(parent/'phases'/receipt['phase']/'source.json')
        for rel in receipt['mismatches']:
            digest=next(h for p,h in freeze.items() if p.endswith('/'+rel));matches=[]
            for revision in reversed(commits):
                q=subprocess.run(['git','show',revision+':'+rel],cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
                if q.returncode==0 and hashlib.sha256(q.stdout).hexdigest()==digest:matches.append(revision)
            receipt['resolved_runtime_mismatches'].append(dict(path=rel,sha256=digest,
                first_matching_commit=matches[0] if matches else None,current_equal=sha(CODE/rel)==digest))
    write(root/'runtime_commits.json',receipts)


def physics(root):
    r=root;es=read(Path(read(r/'session.json')['parent'])/'manifest.json')['episodes'];values=defaultdict(list);families=defaultdict(list);per=[]
    for e in es:
     x=torch.load(Path(e['path']).parents[1]/'initial_export.pt',map_location='cpu',weights_only=False);v={}
     for body,fields in x['physics'].items():
      for k,a in fields.items():
       if k=='root_state_w':continue
       a=a.numpy();key=body+'/'+k;values[key].append(a);v[key]=hashlib.sha256(a.tobytes()).hexdigest()
     fields=x['env_tensors'];v['cpu_rng_torch']=hashlib.sha256(x['cpu_rng'][2].numpy().tobytes()).hexdigest()
     for k in ('goal_pos_obs_noise','goal_yaw_obs_noise','_random_force_prob','_random_torque_prob','_object_scale_multiplier'):
      values[k].append(fields[k].numpy());v[k]=hashlib.sha256(fields[k].numpy().tobytes()).hexdigest()
     per.append(dict(id=e['id'],family=e['family'],fields=v));families[e['family']].append(v)
    summary={k:dict(shape=list(a[0].shape),unique_values=len({v.tobytes() for v in a}),min=float(np.min(a)),max=float(np.max(a)),quantiles=np.quantile(np.stack(a),[0,.1,.5,.9,1],axis=0).tolist()) for k,a in values.items()}
    write(r/'physics_dr.json',dict(per_episode=per,summary=summary,family_reset_mismatches={f:[k for k in v[0] if len({x[k] for x in v})>1] for f,v in families.items() if any(len({x[k] for x in v})>1 for k in v[0])},hidden_state_missing='PhysX contact/solver/broadphase internal state'))
    print({k:{a:b for a,b in v.items() if a!='quantiles'} for k,v in summary.items()});print('FAMILY_MISMATCHES',read(r/'physics_dr.json')['family_reset_mismatches'])


def physical(root):
    from visuotactile.narrow_data import angle
    from visuotactile.position_control import recorded_target_plan
    r=root
    assert (r/'phases/physical_acceptance/complete.json').exists() and read(r/'status.json')['owned_workers_exited']
    plan={x['identity']:x for x in read(r/'replay_plan.json')['jobs']};results=read(r/'expert_results.json')+read(r/'replay_results.json');details=[]
    for row in results:
     job=plan[row['identity']];out=r/'physical'/row['identity'];item=dict(identity=row['identity'],source_id=job['source_id'],mode=row['mode'],source_alpha=job['alpha'],source_success=job['source_success'],trace_sha256=sha(out/'trace.npz'),numeric_sha256=sha(out/'numeric.h5'))
     old=torch.load(Path(job['source']).parents[1]/'initial_export.pt',map_location='cpu',weights_only=False);new=torch.load(out/'initial_export.pt',map_location='cpu',weights_only=False)
     physical={};env={}
     for body,fields in old['physics'].items():
      for k,a in fields.items():physical[body+'/'+k]=float(abs(a-new['physics'][body][k]).max())
     for k,a in old['env_tensors'].items():
      if k in new['env_tensors'] and torch.is_tensor(a) and a.shape==new['env_tensors'][k].shape:
       env[k]=float(abs(a.to(torch.float64)-new['env_tensors'][k].to(torch.float64)).max())
     item['saved_physical_max_errors']=physical;item['saved_env_differences']={k:v for k,v in env.items() if v}
     item['source_controls_comparable']=job['mode']!='native_expert' or job['alpha']==1
     with h5py.File(job['source']) as h,h5py.File(out/'numeric.h5') as n:
      numeric=n['episode'];meta=json.loads(n.attrs['metadata_json']);item['success_reconstruction'],_=analyze(numeric,meta);assert item['success_reconstruction']['passed']
      a=next(iter(h['episodes'].values()))['transition'];b=n['episode/transition'];length=min(a['target_after_rad'].shape[0],b['target_after_rad'].shape[0]);errs={}
      for name in ('joint_position_rad','joint_velocity_rad_s','object_pose_w','hole_pose_w','palm_body_pose_w','finger_position_w'):
       source=a['diagnostics_after/'+name][:length];actual=b['diagnostics_after/'+name][:length];delta=abs(source-actual).reshape(length,-1).max(-1);nonzero=np.flatnonzero(delta)
       errs[name]=dict(first_nonzero_tick=int(nonzero[0])+1 if len(nonzero) else None,first_tick_max=float(delta[0]),maximum=float(delta.max()),ticks_over_1e_6=int((delta>1e-6).sum()))
      item['actual_state_errors']=errs
      d=np.linalg.norm(a['diagnostics_after/object_pose_w'][:length,:3]-b['diagnostics_after/object_pose_w'][:length,:3],axis=-1);item['object_position_max_mm']=float(d.max()*1000)
      item['object_orientation_max_deg']=max(0. if np.array_equal(x,y) else angle(x.astype(np.float64)/np.linalg.norm(x.astype(np.float64)),y.astype(np.float64)/np.linalg.norm(y.astype(np.float64))) for x,y in zip(a['diagnostics_after/object_pose_w'][:length,3:],b['diagnostics_after/object_pose_w'][:length,3:]))
      if row['mode']=='h24_replay':
       targets=a['target_after_rad'][:];plans=np.load(out/'plans.npz')['plans'];expected=np.stack([recorded_target_plan(targets,t)[0].numpy() for t in range(0,row['result']['steps'],12)]);assert np.array_equal(plans,expected)
       item['all_h24_plans_exact']=True
     if (out/'rollout.mp4').exists():item['video_sha256']=sha(out/'rollout.mp4')
     details.append(item)
    paired=[]
    for sid in read(r/'replay_plan.json')['source_ids']:
     names=[sid+'__'+mode for mode in ('tick_replay','h24_replay')];traces=[np.load(r/'physical'/n/'trace.npz') for n in names];length=min(len(t['joint']) for t in traces)
     paired.append(dict(source_id=sid,ticks_common=length,target_exact=np.array_equal(traces[0]['target'][:length],traces[1]['target'][:length]),actual_joint_max=float(abs(traces[0]['joint'][:length]-traces[1]['joint'][:length]).max()),object_pose_max=float(abs(traces[0]['object'][:length]-traces[1]['object'][:length]).max()),results=[next(x['result'] for x in results if x['identity']==n) for n in names]))
    write(r/'physical_details.json',dict(rows=details,replay_pairs=paired,missing='PhysX contact/solver/broadphase internal state is not exported or forced. World/body composer internal caches are not complete snapshots.'))
    print('PHYSICAL_DETAILS',len(details),'pairs',len(paired),'pair_outcome_disagreements',sum(x['results'][0]['success']!=x['results'][1]['success'] for x in paired))


def finalize(root):
    import shutil
    from tools.narrow_repair import protected
    from tools.runtime import proc
    r=root;s=read(r/'session.json');parent=Path(s['parent']);entries=read(parent/'manifest.json')['episodes'];before=time.time()
    assert (r/'phases/fresh_pair_check/complete.json').exists() and read(r/'status.json')['owned_workers_exited']
    for e in entries:assert sha(e['path'])==e['sha256'] and Path(e['path']).stat().st_size==e['size']
    protected(r)
    original=CODE/'outputs/visuotactile/data/close_v4';old=read(original/'manifest.json')['episodes'];paths=[original/e['directory']/'data.h5' for e in old];assert len(paths)==400 and all(p.exists() for p in paths)
    rows=[json.loads(x) for x in (r/'resources/shared.jsonl').read_text().splitlines()];work=[json.loads(x) for x in (r/'work.jsonl').read_text().splitlines()];pids=sorted({x['pid'] for x in work});assert all(proc(p) is None for p in pids)
    summary=read(r/'monitor_exit.json')['summary'];size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());assert size<s['maximum_output_bytes'];free=shutil.disk_usage(r).free;assert free>s['reserve_bytes']
    report=dict(completed=time.time(),started=s['started'],deadline=s['deadline'],deadline_reset=False,actual_trajectory_starts=60,primary_starts=56,targeted_fresh_starts=4,formal_updates=0,student_rollouts=0,formal_collection=0,source_commits=['4ddb554','f6d36a8'],protected_artifacts=len(read(r/'protected.json')),all275_final_sha_match=True,original400_files_present=True,owned_gpu_workers=pids,owned_gpu_workers_absent=True,services_exit_code=0,output_bytes=size,current_free_disk_bytes=free,resource_summary=summary,minimum_free_vram_gib=min(x['gpu_free_mib'] for x in rows)/1024,minimum_available_ram_gib=min(x['ram_available_bytes'] for x in rows)/1024**3,cpu_audit_reader_fixes=4,cpu_postprocess_observer_initial_exit=143,cpu_postprocess_independent_recheck_exit=0,resource_gap_explanation='231.662s between completed primary phase and fresh-check phase; no owned Kit during CPU review/registration. Not continuous session telemetry.',admission='275 State/recorded tactile numerical-time-valid;75 three-view-valid. No automatic recollection or training. Fresh comparisons verify current tick/H24 parity on two outcome-selected sources, not exact historical-state reconstruction.')
    write(r/'final_audit.json',report);status=read(r/'status.json');status.update(state='audit_complete',stage='await_user_decision',owned_workers_exited=True,finished=time.time(),actual_trajectory_starts=60);write(r/'status.json',status)
    print('FINAL',report['actual_trajectory_starts'],round(size/1024**3,3),round(report['minimum_free_vram_gib'],3),round(report['minimum_available_ram_gib'],3),'hash_seconds',time.time()-before)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('source','attempts','coverage','details','physics','physical','finalize'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.root.resolve())
