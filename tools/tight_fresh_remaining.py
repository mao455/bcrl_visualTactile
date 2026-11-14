"""Prepare only never-started cases after a censored autonomous worker.

This registers a subset of the original opportunities; it launches no worker
and never repeats either a complete or a partially executed case.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def prepare(root, original, previous, output, source):
    if output.exists():raise FileExistsError('Preserve prior remaining-case identity')
    if original.resolve()==previous.resolve():raise ValueError('Original and recovery attempts must be distinct')
    plan_bytes=(original/'plan.json').read_bytes()
    if plan_bytes!=(previous/'plan.json').read_bytes():
        raise ValueError('Original and recovery plans must be bytewise identical')
    old=json.loads(plan_bytes);session=read(root/'session.json')
    if old.get('kind')!='autonomous':raise ValueError('Only autonomous remaining cases are supported')
    if time.time()>=session['deadline']:raise TimeoutError('Original session expired')
    cases=old.get('cases',[])
    if (not cases or len({c['name'] for c in cases})!=len(cases)
            or any(Path(c['name']).name!=c['name'] for c in cases)):
        raise ValueError('Registered cases require unique local names')
    plan_hash=sha256(original/'plan.json');attempts=[]
    for attempt in (original,previous):
        started=read(attempt/'run_started.json')
        if (Path('/proc')/str(started['pid'])).exists():
            raise ValueError('Every original/recovery worker must exit before remaining-case registration')
        if started.get('plan_sha256')!=plan_hash:
            raise ValueError('Prior worker plan binding differs')
        if read(attempt/'session.json')!=session:
            raise ValueError('Original session and deadline must remain unchanged')
        attempts.append(dict(root=str(attempt),run_started_sha256=sha256(attempt/'run_started.json')))
    if set(old.get('pairs',{}))!={'A','B'} or any(c['role'] not in old['pairs'] for c in cases):
        raise ValueError('Registered plan requires complete A/B policy identities')
    if {role:completed_pair(root,role) for role in ('A','B')}!=old['pairs']:
        raise ValueError('Registered complete policy pairs changed')
    hashes=old.get('source_sha256',{})
    actual={str(p.relative_to(source)):p for folder in ('tools','visuotactile')
        for p in (source/folder).rglob('*.py')}
    if not hashes or set(hashes)!=set(actual) or any(sha256(actual[p])!=digest for p,digest in hashes.items()):
        raise ValueError('Frozen source population or hashes changed')
    frozen_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if frozen_commit!=old.get('source_commit'):raise ValueError('Frozen source commit changed')
    remaining=[];excluded=[]
    for case in cases:
        seen=[attempt/case['name'] for attempt in (original,previous)
            if (attempt/case['name']).exists() or (attempt/case['name']).is_symlink()]
        if seen:
            excluded.append(dict(name=case['name'],role=case['role'],started_locations=[str(p) for p in seen],
                completed_locations=[str(p) for p in seen if (p/'result.json').exists()],
                exclusion='Any case directory counts as started; complete and censored cases never repeat'))
        else:remaining.append(case)
    if not remaining:raise ValueError('No never-started registered cases remain')
    for attempt in (original,previous):
        if (attempt/'results.json').exists():
            recorded=read(attempt/'results.json')
            if any(row['case'] in {c['name'] for c in remaining} for row in recorded):
                raise ValueError('A remaining case already has a recorded result')
    plan=copy.deepcopy(old);plan['cases']=remaining
    output.mkdir(parents=True)
    atomic_json(output/'plan.json',plan)
    shutil.copyfile(original/'session.json',output/'session.json')
    owner=Path(__file__).resolve()
    registration=dict(original_plan_sha256=plan_hash,remaining_plan_sha256=sha256(output/'plan.json'),
        original_case_count=len(cases),remaining_cases=[c['name'] for c in remaining],excluded_cases=excluded,
        attempts=attempts,source_pairs=old['pairs'],frozen_execution_source=str(source),
        frozen_execution_source_commit=frozen_commit,verified_source_files=len(hashes),
        owner_script=str(owner),owner_script_sha256=sha256(owner),
        owner_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner.parent.parent,text=True).strip(),
        deadline=session['deadline'],startup_seconds=180,periodic_traceback_dump=False,
        new_optimizer_updates=0,simulation_starts=0,run_invoked=False,
        meaning='Only original registered opportunities with no case directory in either prior attempt; no case replacement or repeat',utc=time.time())
    atomic_json(output/'remaining_registration.json',registration)
    return registration


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','original','previous','output','source'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    prepare(*(getattr(args,name).resolve() for name in ('root','original','previous','output','source')))
