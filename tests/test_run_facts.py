import json
import torch

from visuotactile.compact import sha256
from visuotactile.run_facts import sampler_facts


def test_reporting_uses_loaded_checkpoint_seed_and_actual_offline_configuration(tmp_path):
    child=tmp_path/'A';(child/'training').mkdir(parents=True)
    source=tmp_path/'parent.pt';torch.save(dict(contract={'sampler_seed':8817}),source)
    (child/'continuation.json').write_text(json.dumps(dict(path=str(source),sha256=sha256(source))))
    configuration=dict(offline_planning=dict(seed=9931,max_anchors=64))
    (child/'training/configuration.json').write_text(json.dumps(configuration))
    facts=sampler_facts(tmp_path,'A')
    assert facts['checkpoint_sampler_seed']==8817
    assert facts['offline_planning']==configuration['offline_planning']
    assert 'not recoverable' in facts['episode_rng_position']
