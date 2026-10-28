"""Audited conversion worker; supervisor monitors its complete wall time."""
import argparse
import json
from pathlib import Path
from .compact import convert_episode
from .run_rl_trials import validate_completion
from .contracts import assert_policy_compatible
from .trial_utils import write_json


def finalize(source,destination,report,reference):
    trial=validate_completion(source)
    m,s=trial['metadata'],trial['summary']
    ref=json.loads(Path(reference).read_text())
    assert_policy_compatible(ref,m)
    for section in ('action','assets','domain_randomization','obs','reset','reward','termination','precise_assembly'):
        if m['config'][section]!=ref['config'][section]:raise ValueError('Protocol drift: '+section)
    if m['expert_model_sha256']!=ref['expert_model_sha256']:raise ValueError('Expert drift')
    if not m['initial_geometry']['valid_box_initialization']:raise ValueError('Illegal start')
    if m['camera_ids']!=['fixed_high','fixed_side','wrist_rgbd']:raise ValueError('Missing camera')
    if s['requested_vs_applied_peak_vector_error_N']>1e-5:raise ValueError('Wrench mismatch')
    info=None
    if s['outcome']=='success':
        info=convert_episode(Path(source)/'trajectories.h5',destination,
            result=json.loads((Path(source)/'result.json').read_text()),write_videos=False)
        compact=validate_completion(destination)
        if compact['summary']['outcome']!=s['outcome']:raise ValueError('Outcome mismatch')
    write_json(report,dict(summary=s,conversion=info,metadata=m,
        initial={k:v.tolist() for k,v in trial['arrays']['initial'].items()}))


def main():
    p=argparse.ArgumentParser()
    for name in ('source','destination','report','reference'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();finalize(a.source,a.destination,a.report,a.reference)
if __name__=='__main__':main()
