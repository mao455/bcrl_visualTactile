"""Bounded real three-camera causal replay and Jacobian probe; zero updates."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch

from .compact import EpisodeFile
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import CorrectionPolicy
from .rollout import NominalController
from .correction import CorrectionShadow, latest_correction_inputs
from .check_dataset import fingerprint
from .trial_utils import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(134)
    with EpisodeFile(a.data) as f:
        g=next(iter(f['episodes'].items()))[1];views=g['obs/image'].shape[1]
        cfg=PolicyConfig(views=views,inference_steps=1,history=2,stride=1)
        controller=NominalController(NominalDiffusionPolicy(cfg).eval(),1);shadow=CorrectionShadow(controller,1)
        before=[fingerprint(controller.model),fingerprint(shadow.model)]
        for t in range(min(16,int(g.attrs['length']))):
            packet={k:torch.from_numpy(ds[t:t+1]) for k,ds in g['obs'].items()}
            controller.act(packet);shadow.observe(packet,controller)
            shadow.context.commit(torch.from_numpy(g['transition/submitted_action'][t:t+1]))
        inputs=latest_correction_inputs(shadow.last_inputs,2)
        image=inputs['visual_history']['image'].detach().clone().requires_grad_(True)
        inputs['visual_history']['image']=image
        result=shadow.model(**inputs)
        result.sum().backward()
        per_view=image.grad.abs().sum((0,1,3,4,5))
        assert torch.isfinite(image.grad).all() and (per_view>0).all()
        module_grads={}
        for name,module in [('rgb',shadow.model.visual.vision.rgb),('depth',shadow.model.visual.vision.depth),
                            ('tactile',shadow.model.tactile),('motion',shadow.model.motion),('intent',shadow.model.intent)]:
            values=[v.grad for v in module.parameters() if v.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values)
            total=float(sum(v.abs().sum() for v in values));assert total>0;module_grads[name]=total
        condition=controller.model.encoder(inputs['visual_history']);condition.sum().backward()
        saved=dict(config=asdict(cfg),model=controller.model.state_dict(),correction=shadow.model.state_dict(),
                   parameter_updates=0,role='untrained_jacobian_probe_not_rollout_checkpoint')
        torch.save(saved,a.output/'untrained_roundtrip.pt')
        restored=torch.load(a.output/'untrained_roundtrip.pt',weights_only=False)
        other=CorrectionPolicy(PolicyConfig.from_dict(restored['config'])).eval();other.load_state_dict(restored['correction'])
        torch.testing.assert_close(result,other(**inputs),rtol=0,atol=0)
        after=[fingerprint(controller.model),fingerprint(shadow.model)];assert before==after
        write_json(a.output/'report.json',dict(steps_replayed=t+1,views=views,parameter_updates=0,
            correction_executed=False,nominal_plan_source='fresh untrained causal replay, never expert future',
            source_data_unchanged=True,correction_shape=list(result.shape),visual_gradient_by_view=per_view.tolist(),
            module_gradient_l1=module_grads,checkpoint_roundtrip_exact=True,
            hashes_before=before,hashes_after=after,limitation='Jacobian/connectivity only; no learned skill or trained correction claim'))


if __name__=='__main__':main()
