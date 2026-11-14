import copy
import json
import os
from pathlib import Path
import shutil

import pytest

from tools import tight_fresh_remaining as remaining
from visuotactile.compact import sha256
from test_fresh_eval_recovery import registered_plan


@pytest.fixture
def interrupted(registered_plan,monkeypatch):
    x=registered_plan
    x.original=x.previous.parent/'original';shutil.copytree(x.previous,x.original)
    monkeypatch.setattr(remaining,'completed_pair',lambda root,role:copy.deepcopy(x.pairs[role]))
    complete=x.previous/x.cases[0]['name'];complete.mkdir();x.write(complete/'result.json',{'software_complete':True})
    partial=x.previous/x.cases[1]['name'];partial.mkdir();x.write(partial/'started.json',{})
    x.write(x.previous/'results.json',[{'case':x.cases[0]['name']}])
    return x


def test_only_never_created_original_cases_remain(interrupted):
    x=interrupted
    registration=remaining.prepare(x.root,x.original,x.previous,x.output,x.source)
    plan=json.loads((x.output/'plan.json').read_text())
    expected=copy.deepcopy(x.old);expected['cases']=x.cases[2:]
    assert plan==expected
    assert (x.original/'plan.json').read_bytes()==(x.previous/'plan.json').read_bytes()
    assert registration['remaining_cases']==[c['name'] for c in x.cases[2:]]
    assert [c['name'] for c in registration['excluded_cases']]==[c['name'] for c in x.cases[:2]]
    assert registration['excluded_cases'][0]['completed_locations']
    assert registration['excluded_cases'][1]['completed_locations']==[]
    assert registration['deadline']==x.session['deadline']
    assert registration['startup_seconds']==180 and not registration['periodic_traceback_dump']
    assert registration['simulation_starts']==registration['new_optimizer_updates']==0
    assert not registration['run_invoked'] and not x.calls
    with pytest.raises(FileExistsError):remaining.prepare(x.root,x.original,x.previous,x.output,x.source)


def test_empty_case_directory_in_original_is_also_excluded(interrupted):
    x=interrupted;(x.original/x.cases[2]['name']).mkdir()
    result=remaining.prepare(x.root,x.original,x.previous,x.output,x.source)
    assert result['remaining_cases']==[x.cases[3]['name']]


@pytest.mark.parametrize('fault',['plan_bytes','live_original','live_recovery','plan_binding',
    'session','deadline','pair','source','source_population','source_commit','all_started','result_without_directory'])
def test_changed_bindings_or_already_started_cases_cannot_be_registered(interrupted,monkeypatch,fault):
    x=interrupted
    if fault=='plan_bytes':(x.previous/'plan.json').write_text(json.dumps(x.old))
    elif fault in ('live_original','live_recovery'):
        at=x.original if fault=='live_original' else x.previous
        x.write(at/'run_started.json',dict(pid=os.getpid(),plan_sha256=sha256(at/'plan.json')))
    elif fault=='plan_binding':x.write(x.previous/'run_started.json',dict(pid=999999999,plan_sha256='changed'))
    elif fault=='session':x.write(x.previous/'session.json',dict(x.session,deadline=x.session['deadline']+1))
    elif fault=='deadline':x.write(x.root/'session.json',dict(x.session,deadline=0))
    elif fault=='pair':monkeypatch.setattr(remaining,'completed_pair',lambda *args:{'changed':True})
    elif fault=='source':(x.source/'tools/worker.py').write_text('# changed\n')
    elif fault=='source_population':(x.source/'tools/extra.py').write_text('# extra\n')
    elif fault=='source_commit':
        changed=dict(x.old,source_commit='different')
        for at in (x.original,x.previous):
            x.write(at/'plan.json',changed);x.write(at/'run_started.json',dict(pid=999999999,plan_sha256=sha256(at/'plan.json')))
    elif fault=='all_started':
        for case in x.cases[2:]:(x.previous/case['name']).mkdir()
    elif fault=='result_without_directory':x.write(x.previous/'results.json',[{'case':x.cases[2]['name']}])
    with pytest.raises((ValueError,TimeoutError)):
        remaining.prepare(x.root,x.original,x.previous,x.output,x.source)
    assert not x.output.exists() and not x.calls
