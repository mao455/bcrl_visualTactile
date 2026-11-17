"""Matched full-policy command errors on common72 recorded histories."""
import json
import torch,h5py,numpy as np
from tools.tight_fresh_rollouts import completed_pair
from visuotactile.compact import sha256
from visuotactile.models.native_dp import TargetResidual
from visuotactile.anchor_continue import residual_prediction
from visuotactile.bounded_extension import error_bins,summarize
from scripts.tight_balance_20261117.diagnose import BASE,OUT,phases
from scripts.tight_layout_20261116.publication import write,table

def calculate(root,lease):
 rows=[]
 for role in 'AB':
  old=BASE/'round05/learning';common=json.loads((old/role/'state_data/data.json').read_text())['splits']['train']
  for version,source in [('parent72',old),('candidate76',root)]:
   completed_pair(source,role)
   rear=source/role/'backend'/role;index=json.loads((rear/'nominal/index.json').read_text());data=json.loads((rear/'state_data/data.json').read_text())
   with lease.block():
    checkpoint=torch.load(rear/'training/best.pt',map_location='cpu',weights_only=False,mmap=True);cfg=checkpoint['configuration']
    assert cfg['base_sha256']==index['base_sha256']
    stats=torch.load(cfg['residual_statistics_path'],map_location='cpu',weights_only=False)
    scales=json.loads((rear/'nominal/scales.json').read_text())['scales']
    model=TargetResidual(stats,scales,execution_adapter=cfg.get('execution_adapter'),loss_contract=cfg.get('loss_contract'),history_contract=cfg.get('history_contract'),output_contract=cfg.get('output_contract'))
    model.load_state_dict(checkpoint['model']);model.cuda().eval();del checkpoint
   bins=[];newbins=[]
   for identity in data['splits']['train']:
    path=rear/'nominal'/(identity+'.pt');assert sha256(path)==index['episodes'][identity]['sha256']
    rec=torch.load(path,map_location='cpu',weights_only=False);valid=rec['supervision_mask'].cpu().bool();ticks=valid.nonzero().flatten().numpy()
    assert rec['cache_identity']['base_checkpoint_sha256']==index['base_sha256']
    prediction=residual_prediction(model,rec,lease);entry=data['entries'][identity];correction=entry.get('trajectory_kind')=='expert_takeover_v1'
    with h5py.File(entry['path']) as f:phase=phases(f['episodes'][entry['episode']]['transition'])
    onset=np.zeros(len(valid),bool)
    if correction:onset[ticks[:24]]=True
    masks=dict(original_early=(np.arange(len(valid))<150)&(not correction),correction_onset24=onset,near_alignment=phase==4,after_subgoal=phase==5,all=valid.numpy())
    result=error_bins((prediction-rec['target'].cpu())[valid],ticks,masks,model.action_scale)
    (bins if identity in common else newbins).append(result)
    rows.append(dict(role=role,version=version,identity=identity,cohort='common72' if identity in common else 'new4',bins=result))
   report=dict(role=role,version=version,common72=summarize(bins),new4=summarize(newbins) if newbins else None,frontend_sha256=index['base_sha256'],backend_sha256=sha256(rear/'training/best.pt'),scope='Complete composed command at every legal expert tick, corresponding immutable DP plans and actual recorded history. Parent/candidate comparison uses common72 only; not closed-loop imitation error.')
   write(OUT/f'backend_{role}_{version}_phases.json',report)
   del model;torch.cuda.empty_cache()
 table(OUT/'backend_phase_errors.csv',rows)
