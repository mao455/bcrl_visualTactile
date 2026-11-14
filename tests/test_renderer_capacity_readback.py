"""Mock application/environment proves both capacity gates precede expert actions."""
import contextlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize('mismatch_phase',[None,'after_app','after_environment'])
def test_registered_capacity_checked_after_app_and_after_environment(tmp_path,monkeypatch,mismatch_phase):
    import tools.tight_state as state
    import visuotactile.native_evaluation as evaluation
    import visuotactile.expert as expert
    from visuotactile.execution_backend import DESCRIPTOR_VERSION
    settings={'/plugins/carb.tasking.plugin/threadCount':2,'/persistent/physics/numThreads':2,
        '/rtx/descriptorSets':360000,'/rtx/reservedDescriptors':900000,'/rtx/viewTile/limit':8}
    calls=[]
    def get(path):
        calls.append(('read',path));return settings.get(path)
    def launch(**kwargs):
        calls.append(('app',kwargs['kit_args']))
        assert '--/rtx/descriptorSets=360000' in kwargs['kit_args']
        if mismatch_phase=='after_app':settings['/rtx/descriptorSets']=60000
        return SimpleNamespace(app='fake_app')
    app_module=ModuleType('isaaclab.app');app_module.AppLauncher=launch
    lab_module=ModuleType('isaaclab');lab_module.__path__=[];lab_module.app=app_module
    settings_module=ModuleType('carb.settings');settings_module.get_settings=lambda:SimpleNamespace(get=get)
    carb_module=ModuleType('carb');carb_module.__path__=[];carb_module.settings=settings_module
    for name,module in [('isaaclab',lab_module),('isaaclab.app',app_module),('carb',carb_module),
        ('carb.settings',settings_module),('isaacsimenvs',ModuleType('isaacsimenvs'))]:
        monkeypatch.setitem(sys.modules,name,module)
    def make_env(*args,**kwargs):
        calls.append(('environment',None))
        if mismatch_phase=='after_environment':settings['/rtx/reservedDescriptors']=None
        return SimpleNamespace(device='cpu',physics_dt=1/120,step_dt=1/60,num_envs=8,
            cfg=SimpleNamespace(student_obs=SimpleNamespace(camera_backend='tiled')))
    def create_expert(*args,**kwargs):
        calls.append(('expert',None));return 'fake_expert',{}
    monkeypatch.setattr(evaluation,'make_env',make_env)
    monkeypatch.setattr(evaluation,'client_startup_controls',lambda *args:{})
    monkeypatch.setattr(expert,'create_expert',create_expert)
    checkpoint=tmp_path/'expert';checkpoint.write_bytes(b'fixture')
    (tmp_path/'session.json').write_text(json.dumps(dict(expert_checkpoint=str(checkpoint),
        expert_sha256=state.sha(checkpoint))))
    invoke=lambda:state.environment(tmp_path,evaluation=True,num_envs=8,
        lease=SimpleNamespace(block=lambda **kwargs:contextlib.nullcontext()),
        execution_backend=dict(version=DESCRIPTOR_VERSION,camera_backend='tiled'))
    if mismatch_phase is None:
        assert invoke()[0]=='fake_app'
        assert [event[0] for event in calls if event[0]!='read']==['app','environment','expert']
        assert json.loads((tmp_path/'kit_started.json').read_text())['renderer_resources']['matched']
        assert json.loads((tmp_path/'execution_backend_observed.json').read_text())['renderer_resources']['matched']
        assert sum(event==('read','/rtx/descriptorSets') for event in calls)==2
    else:
        with pytest.raises(RuntimeError,match='Renderer resource settings'):invoke()
        assert not any(event[0]=='expert' for event in calls)
        filename='kit_started.json' if mismatch_phase=='after_app' else 'execution_backend_observed.json'
        assert not json.loads((tmp_path/filename).read_text())['renderer_resources']['matched']
        assert any(event[0]=='environment' for event in calls)==(mismatch_phase=='after_environment')
