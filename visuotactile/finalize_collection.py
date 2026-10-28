"""Audited conversion worker; supervisor monitors its complete wall time."""
import argparse
import json
from pathlib import Path
from .compact import convert_episode
from .run_rl_trials import validate_completion
from .contracts import assert_policy_compatible
from .trial_utils import write_json


def finalize_stream(source, reference):
    """Background CPU audit of an already losslessly streamed episode.

    Does not publish/delete data. The single manifest owner does that after
    this worker returns, so a crash cannot invent a committed success.
    """
    import numpy as np
    from .recording import verify_stream_fields
    from .compact import EpisodeFile, sha256
    source=Path(source)
    exact_fields=verify_stream_fields(source/'data.h5')
    trial=validate_completion(source)
    m,s=trial['metadata'],trial['summary']
    assert_policy_compatible(reference,m)
    for section in ('action','assets','domain_randomization','obs','reset','reward','termination','precise_assembly'):
        if m['config'][section]!=reference['config'][section]:raise ValueError('Protocol drift: '+section)
    if m['expert_model_sha256']!=reference['expert_model_sha256']:raise ValueError('Expert drift')
    if not m['initial_geometry']['valid_box_initialization']:raise ValueError('Illegal start')
    if s['requested_vs_applied_peak_vector_error_N']>1e-5:raise ValueError('Wrench mismatch')
    with EpisodeFile(source/'data.h5') as f:
        name=next(iter(f['episodes']));g=f['episodes'][name]
        lag=g['transition/control_trace/effective_lag_steps'][:]
        content_digest=__import__('hashlib').sha256(
            g['transition/submitted_action'][:].tobytes()+
            g['transition/diagnostics_before/joint_position_rad'][:].tobytes()).hexdigest()
    return dict(summary=s,initial={k:v.tolist() for k,v in trial['arrays']['initial'].items()},
        effective_delay_counts=np.bincount(lag.astype(int),minlength=3).tolist(),
        content_digest=content_digest,episode=name,length=s['control_steps'],
        frame_count=s['control_steps']+int(s['outcome']=='censored'),exact_fields=exact_fields,
        storage_version=2,camera_ids=m['camera_ids'],video_files=[],
        files={'data.h5':dict(bytes=(source/'data.h5').stat().st_size,sha256=sha256(source/'data.h5'))})


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
