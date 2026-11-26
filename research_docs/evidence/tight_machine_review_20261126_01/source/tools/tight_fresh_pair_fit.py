"""Small train-only complete-policy fitting diagnosis on saved true histories.

Uses refreshed commit12 plans and current rear recurrence. No optimizer, no new
expert labels and no claim that teacher-forced errors predict closed-loop success.
"""
import argparse
from pathlib import Path
import time
import torch

from visuotactile.anchor_continue import residual_setup, residual_prediction
from visuotactile.precision import state_fp32
from visuotactile.training_resources import atomic_json


@torch.no_grad()
def run(root, role, output):
    if output.exists(): raise FileExistsError('Preserve prior diagnostic')
    state_fp32();began=time.time()
    _,_,data,records,model,config,lease=residual_setup(root/role/'backend'/role, selected=True)
    ids=data['normalization_train_ids'][:2]+data['aggregation']['new_training_ids'][:2]
    assert set(ids)<=set(data['splits']['train'])
    result={}
    for identity in ids:
        record=records[identity];prediction=residual_prediction(model,record,lease).cpu()
        truth=record['target'].cpu();base=record['base'].cpu();mask=record['supervision_mask'].cpu()
        command_error=prediction-truth
        nominal_label=record.get('nominal_supervision')
        metrics={}
        for name,selection in [('all_legal',mask),('legal_before100',mask&(torch.arange(len(mask))<100)),
                               ('first24_legal',mask & (mask.cumsum(0)<=24))]:
            e=command_error[selection]
            metrics[name]=dict(ticks=len(e),arm_rmse_rad=float(e[:,:7].square().mean().sqrt()) if len(e) else None,
                hand_rmse_rad=float(e[:,7:].square().mean().sqrt()) if len(e) else None)
        result[identity]=dict(metrics=metrics,nominal_label=nominal_label,
            first150_actual_target_error=command_error[:150].square().mean(-1).sqrt().tolist(),
            first150_nominal_to_command_gap=(base[:150]-truth[:150]).square().mean(-1).sqrt().tolist(),
            first150_legal=mask[:150].tolist(),
            comparison_note='A nominal forecasts next q, so nominal-to-command gap is not an A forecasting error; rear final command error has the same physical target for both roles')
    atomic_json(output,dict(complete=True,role=role,seconds=time.time()-began,identities=ids,
        results=result,base_sha256=config['base_sha256'],optimizer_updates=0,test_read=False,
        scope='Two original successful training families and two latest corrections; teacher-forced actual past, refreshed current DP plans, complete selected Transformer output. Autonomous evidence required separately.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--role',choices=['A','B'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.role,args.output.resolve())
