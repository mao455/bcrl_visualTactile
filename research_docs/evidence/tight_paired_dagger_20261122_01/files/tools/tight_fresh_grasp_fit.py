"""Selected/source DP errors on latest real early corrections, reusing frozen features."""
import argparse
from pathlib import Path
import time
import torch

from tools import tight_fresh_train as trainer
from visuotactile.native_data import visual_records
from visuotactile.target_windows import with_target_window
from visuotactile.offline_planning import profile
from visuotactile.state_resources import lease_for
from visuotactile.training_resources import atomic_json
from visuotactile.precision import state_fp32


def run(root, role, output):
    if output.exists(): raise FileExistsError('Preserve previous fitting evidence')
    state_fp32();began=time.time();base=root/role;data=trainer.read(base/'state_data/data.json')
    ids=data['aggregation']['new_training_ids'];continuation=trainer.read(base/'continuation.json')
    config=trainer.read(base/'training/configuration.json');done=trainer.read(base/'training/finished.json')
    statistics=torch.load(base/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    records=with_target_window(visual_records(base/'state_data',ids,'G00',device='cuda:0'),continuation.get('target_window_contract'))
    lease=lease_for(root,deadline=trainer.read(root/'session.json')['deadline']-60)
    rows={}
    # No broad source/raw re-audit: the owner already bound the frozen source,
    # and trainer/cache receipts bind the files consumed here. Actual checkpoint
    # config and per-episode visits remain checked against those receipts.
    for label,path in [('source',Path(continuation['path'])),('selected',base/'training/best.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if label=='selected' and saved['configuration']!=config:raise ValueError('Selected frontend config changed')
        with lease.block():
            model=trainer.new_model(role,statistics)
            if model.contract!=saved['contract']:raise ValueError('Frontend semantics differ')
            model.load_state_dict(saved['ema']);model.eval().requires_grad_(False)
            metrics=trainer.metrics(model,records,ids,offline_planning=profile(32),check=lease.check,
                frozen_visual_io=config.get('frozen_visual_io'),grasp_phase_report=True)
        rows[label]=dict(checkpoint=str(path),epoch=saved['epoch'],updates=saved['updates'],metrics=metrics,
            visits={i:saved.get('sampling_exposure',{}).get(i,0) for i in ids})
        del model,saved
        torch.cuda.empty_cache()
    atomic_json(output,dict(complete=True,role=role,results=rows,seconds=time.time()-began,
        selected_checkpoint_sha256_from_training_receipt=done['checkpoint_sha256'],
        semantics='A q[t+1], B expert absolute command; legal expert suffix only, true entire history, no autonomous success inferred'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',choices=['A','B'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.role,a.output)
