"""Admit only closed, naturally terminated captures from the interrupted07 queue.

Never repair or open incomplete streams, and never mark the old batch complete.
"""
import json,time
from pathlib import Path
import h5py,numpy as np
from visuotactile.async_recording import _verify_closed_capture
from visuotactile.state_benchmark import dagger_recovery_coverage,DaggerCaptureAuditMemo
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_batch_run import admit_capture

ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_07/new_family_15')
read=lambda p:json.loads(p.read_text())
if __name__=='__main__':
    owner=read(ROOT/'owner_finished.json')
    assert not owner['software_complete']
    plan=read(ROOT/'plan.json');entries={};rows=[];memo=DaggerCaptureAuditMemo(max_payloads=32,max_pairs=4)
    for case in plan['cases']:
        d=ROOT/case['name']
        if not (d/'verification_queued.json').exists():continue
        assert (d/'execution_complete.json').exists()
        result=read(d/'execution_complete.json')
        assert result['software_complete'] and result['success']
        if (d/'capture_complete.json').exists():capture=read(d/'capture_complete.json')
        else:
            capture=_verify_closed_capture(dict(numeric_path=str(d/'numeric.h5'),steps=result['steps'],
                dataset_capture=dict(path=str(d/'demonstration/data.h5'),checkpoint_sha256=result['checkpoint_sha256'],
                    trajectory_kind='expert_takeover_v1',accepted_success=True),
                profile=dict(worker_verify_s=0.,worker_hash_s=0.),
                recording_contract='shared_async_capture_v2',
                recovery=dict(reason='owner_interrupted_after_closed_files',prior_owner=owner,original_writer_times_unavailable=True)))
        with np.load(d/'trace.npz') as archive:arrays={k:archive[k] for k in archive.files}
        with h5py.File(d/'numeric.h5','r') as stream:
            result['dagger']['recovery_coverage']=dagger_recovery_coverage(arrays,stream['episode'])
        result.update(dataset_capture=capture['dataset_capture'],async_capture=capture)
        if not (d/'result.json').exists():atomic_json(d/'result.json',result)
        else:result=read(d/'result.json')
        admit_capture(ROOT,case,result,plan['pairs'][case['role']],plan,entries,audit_memo=memo)
        rows.append(dict(case=case['name'],role=case['role'],result=result))
    atomic_json(ROOT/'closed_capture_recovery.json',dict(utc=time.time(),accepted=len(entries),
        successful_expert_ticks=sum(r['result']['dagger']['expert_valid_ticks'] for r in rows),
        cases=rows,batch_software_complete=False,remaining='Physical prefixes interrupted; no resumed-state claim'))
    print(json.dumps(dict(accepted=len(entries),expert_ticks=sum(r['result']['dagger']['expert_valid_ticks'] for r in rows))))
