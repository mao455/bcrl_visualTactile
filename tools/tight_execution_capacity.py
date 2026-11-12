"""CPU output-domain infimum error under the unchanged v1 tolerance map."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools.tight_execution_learning import SOURCE,read,write_csv
from tools.tight_gradient_diagnostics import model_and_records,GROUPS
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.anchor_sampling import supervision_rows,recovery_prefix_rows
from visuotactile.training_resources import atomic_json


def error_infimum(base,radius,target,bounds,limits,tolerance=1e-6):
    # Open finite-logit domain has this closure. Distances are infima, not
    # assertions of finite-logit attainment. Strict exterior rays stay strict.
    lower,upper=bounds.unbind(-1);jl,ju=limits.unbind(-1)
    left=(base-radius).clamp(jl,ju);right=(base+radius).clamp(jl,ju)
    il=torch.maximum(left,lower-tolerance);iu=torch.minimum(right,upper+tolerance)
    identity=(target-target.clamp(il,torch.maximum(il,iu))).abs()
    distance=torch.where(il<=iu,identity,torch.full_like(identity,float('inf')))
    distance=torch.where(left<lower-tolerance,torch.minimum(distance,(target-lower).abs()),distance)
    distance=torch.where(right>upper+tolerance,torch.minimum(distance,(target-upper).abs()),distance)
    if not torch.isfinite(distance).all():raise ValueError('Empty modeled execution image')
    return distance


def audit(root):
    manifest=read(root/'manifest.json');rows=[]
    for group in GROUPS:
        model,records,_=model_and_records(manifest,group)
        for select in manifest['selection']:
            record=records[select['identity']];value=np.load(SOURCE/group/'initial'/(select['identity']+'.npz'))
            base=torch.from_numpy(value['base']);target=record['target'];raw=torch.from_numpy(value['raw'])
            executed=torch.from_numpy(value['u_exec']);mask,_=supervision_rows(record)
            early=recovery_prefix_rows(record,48) if select['kind']=='correction' else torch.zeros_like(mask)
            bounds=ideal_execution_bounds(record['robot'][:,58:87],record['limits'],model.execution_adapter)
            floor=error_infimum(base,model.scales/model.action_scale,target,bounds,record['limits'])
            stable=error_infimum(base,.95*model.scales/model.action_scale,target,bounds,record['limits'])
            strata={'core':mask} if select['kind']=='core' else {'initial48':mask&early,'later':mask&~early}
            for stratum,active in strata.items():
                for part,sl,tol in [('Arm',slice(0,7),1e-4),('Hand',slice(7,29),1e-3)]:
                    f=floor[active,sl];err=(executed-target)[active,sl];scale=model.action_scale[sl]
                    weights=(1+2*early[active].float())[:,None];saturation=raw[active,sl].tanh().abs()>=.95
                    mass=err.square()*scale.square()*weights;bound_mass=f.square()*scale.square()*weights
                    wrong=err.abs()>tol;attenuation=1-raw[active,sl].tanh().square()
                    rows.append(dict(group=group,identity=select['identity'],partition=select['partition'],stratum=stratum,part=part,
                        coordinates=f.numel(),floor_over_tolerance=int((f>tol).sum()),floor_max_rad=float(f.max()),
                        floor_sse_rad2=float(f.square().sum()),actual_sse_rad2=float(err.square().sum()),
                        floor_normalized_weighted_mass=float(bound_mass.sum()),actual_normalized_weighted_mass=float(mass.sum()),
                        floor_fraction_of_actual=float(bound_mass.sum()/mass.sum().clamp_min(1e-30)),
                        saturated_errorful=int((saturation&wrong).sum()),saturated_error_mass_fraction=float(mass[saturation&wrong].sum()/mass.sum().clamp_min(1e-30)),
                        severe_attenuation_errorful=int(((attenuation<.01)&wrong).sum()),
                        stable_floor_max_rad=float(stable[active,sl].max())))
    aggregate=[]
    for group in GROUPS:
        rr=[r for r in rows if r['group']==group]
        mass=lambda field:sum(r[field]*.5/(7 if r['part']=='Arm' else 22) for r in rr)
        aggregate.append(dict(group=group,floor_fraction_of_actual=mass('floor_normalized_weighted_mass')/mass('actual_normalized_weighted_mass'),
            floor_max_rad=max(r['floor_max_rad'] for r in rr),floor_over_tolerance=sum(r['floor_over_tolerance'] for r in rr),
            saturated_errorful=sum(r['saturated_errorful'] for r in rr),severe_attenuation_errorful=sum(r['severe_attenuation_errorful'] for r in rr)))
    result=dict(groups=aggregate,rows=rows,scope='fixed DP and six recorded histories; infimum over output domain, not whole architecture limit',
        actual_v1_tolerance_rad=1e-6,test22_read=False)
    atomic_json(root/'capacity.json',result);print(json.dumps(aggregate,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);audit(a.root)
