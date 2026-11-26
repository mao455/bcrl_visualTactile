"""Independent closure review; CPU temporary fixtures, no real lifecycle reads."""
import json
from types import SimpleNamespace
import os
import pytest

from outputs.diagnostics.tight_paired_dagger_20261122_01.test_close_and_certify_watcher15 import (
    env, lifecycle_fixture, put, m,
)


@pytest.mark.parametrize('names', [[], ['summarize_test'],
    ['summarize_test','summarize_test'],
    ['audit_collection_cuda_noise_recovery05','summarize_test']])
def test_exact_two_tail_population_required(env,monkeypatch,names):
    lifecycle_fixture(env,monkeypatch)
    reg=m.A/'TEST_CONTINUATION15_REGISTRATION.json'
    put(reg,dict(tail_stages=[dict(name=name,skip_closed=False) for name in names]))
    activation=m.A/'TEST_CONTINUATION15_ACTIVATED.json'
    row=json.loads(activation.read_bytes());row['registry_sha256']=m.sha(reg);put(activation,row)
    with pytest.raises(ValueError,match='Exact original two tail helpers required'):
        m.lifecycle()


@pytest.mark.parametrize('command', [
    lambda:['python','-m','tools.tight_paired_evaluate','summarize','--output',str(m.R/'evaluation/test')],
    lambda:['python',str(m.A/'audit_collection_recovery05.py'),'noise','--collection',str(m.R/'collection')],
])
def test_fresh_census_refuses_live_original_helper(env,command):
    process=SimpleNamespace(pid=900001,info=dict(cmdline=command(),uids=SimpleNamespace(real=os.getuid())),
                            create_time=lambda:1)
    with pytest.raises(ValueError,match='still live'):
        m.own_census(processes=[process])
