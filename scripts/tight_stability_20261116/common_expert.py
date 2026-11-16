"""Both rear checkpoints on exactly the same saved legal expert histories."""
import json,time
from pathlib import Path
import torch
from visuotactile.models.native_dp import TargetResidual
from visuotactile.native_data import residual_batch
from visuotactile.bounded_extension import cohorts,phase_masks,error_bins,summarize
from visuotactile.precision import state_fp32
from visuotactile.compact import sha256
from scripts.tight_stability_20261116.analyze_existing import TREE,OLD,OUT,read

def load_model(path,device='cuda'):
    s=torch.load(path,map_location='cpu',mmap=True,weights_only=False);c=s['configuration']
    stats=torch.load(c['residual_statistics_path'],map_location='cpu',weights_only=False)
    m=TargetResidual(stats,c['fixed_residual_scales'],execution_adapter=c['execution_adapter'],
        loss_contract=c['loss_contract'],history_contract=c['history_contract'],output_contract=c['output_contract']).to(device).eval()
    m.load_state_dict(s['model']);return m,s

@torch.no_grad()
def predict(m,record):
    state=None;parts={k:[] for k in ['u_exec','u_pre','delta','raw']};device=next(m.parameters()).device
    for start in range(0,len(record['robot']),32):
        end=min(start+32,len(record['robot']));x=residual_batch([record],start,end,device)
        raw,state=m(x,state)
        chain=m.compose_chain(record['base'][None,start:end].to(device),raw,record['gate'][None,start:end].to(device),record['limits'].to(device),previous=x['robot'][...,58:87])
        for k in parts:parts[k].append(chain[k][0].cpu())
    return {k:torch.cat(v) for k,v in parts.items()}

def main():
    torch.set_num_threads(2);state_fp32();OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'common_expert.json';assert not output.exists();start=time.time();rows=[]
    for role in 'AB':
        data=read(OLD/role/'state_data/data.json');ids=cohorts(data)
        cache=OLD/role/'backend'/role/'nominal';index=read(cache/'index.json')
        models={};saved={}
        for name,file in [('short','best_through_epoch8.pt'),('long','best.pt')]:
            path=OLD/role/'backend'/role/'training'/file;models[name],saved[name]=load_model(path)
            assert saved[name]['configuration']['base_sha256']==index['base_sha256']
        for cohort,identities in ids.items():
            by_model={k:{stage:[] for stage in ['nominal','u_pre','u_exec']} for k in models}
            individual=[]
            for identity in identities:
                r=torch.load(cache/(identity+'.pt'),map_location='cpu',mmap=True,weights_only=False)
                legal=r['supervision_mask'].bool();ticks=legal.nonzero().flatten().numpy();masks=phase_masks(data['entries'][identity])
                values={k:predict(m,r) for k,m in models.items()}
                one=dict(identity=identity,legal_ticks=int(legal.sum()),total_history_ticks=len(legal),models={})
                for name,m in models.items():
                    one['models'][name]={}
                    for stage in ['nominal','u_pre','u_exec']:
                        pred=r['base'] if stage=='nominal' else values[name][stage]
                        # A nominal is next reached q, NOT a command prediction.
                        target=r['q_reached'] if role=='A' and stage=='nominal' else r['target']
                        bins=error_bins((pred-target)[legal],ticks,masks,m.action_scale.cpu())
                        by_model[name][stage].append(bins);one['models'][name][stage]=summarize([bins])
                individual.append(one)
            rows.append(dict(role=role,cohort=cohort,n=len(identities),episodes=individual,
                metrics={name:{stage:summarize(v) for stage,v in stages.items()} for name,stages in by_model.items()},
                base_sha256=index['base_sha256'],epochs={k:v['epoch'] for k,v in saved.items()}))
        del models,saved;torch.cuda.empty_cache()
    result=dict(complete=True,seconds=time.time()-start,rows=rows,updates=0,
        scope='Exactly identical stored real expert/student-prefix sensor+plan inputs. Each rear independently recomputes its own complete causal memory. Only legal expert suffix supervised. A nominal error uses q[t+1]; B and final commands use expert target. Fixed original4/history4/latest4/validation6, not all-original30.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['role'],r['cohort'],{k:{p:v['u_exec'][p]['hand_rmse_rad'] for p in ['all','pre_contact','grasp_contact']} for k,v in r['metrics'].items()},flush=True)

if __name__=='__main__': main()
