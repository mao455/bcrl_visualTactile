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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('source','attempts','coverage'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a.root.resolve())
