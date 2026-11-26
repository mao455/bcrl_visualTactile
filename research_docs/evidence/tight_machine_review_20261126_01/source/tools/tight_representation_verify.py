"""Pin audit sources and verify protected model/data identities without writes to them."""
import csv,gzip,time,subprocess
from pathlib import Path
from tools.tight_representation_audit import PROJECT,RUN,PUBLIC,ROOTS,read,write,sha,load


def main():
    began=time.time();ex=read(PUBLIC/'exposure.json');receipts=[];contracts=[]
    for stage in ex['stages']:
        p=Path(stage['path']);assert sha(p)==stage['sha256'];saved=load(p);c=saved['configuration']
        log=p.parent/('learning.jsonl' if stage['component']=='DP' else 'loss.jsonl')
        receipts.append(dict(path=str(p),sha256=stage['sha256'],log_path=str(log),log_sha256=sha(log),
            selected_updates=saved['updates'],optimizer_present='optimizer' in saved,ema_present='ema' in saved,
            ema_optimization_step=saved.get('ema_optimization_step'),optimizer_lr=[g['lr'] for g in saved['optimizer']['param_groups']],
            state_steps=sorted(set(int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s)),
            checkpoint_counters={k:saved.get(k) for k in ['epoch','updates','supervised_ticks']},
            configuration={k:v for k,v in c.items() if k not in ['source_sha256']},
            core_source_differences={k:dict(recorded=v,current=sha(PROJECT/k) if (PROJECT/k).is_file() else None) for k,v in c.get('source_sha256',{}).items() if (PROJECT/k).is_file() and sha(PROJECT/k)!=v}))
        if stage['path'] in [chain[0] for chain in ex['chains'].values()]:
            contracts.append(dict(path=str(p),contract=saved.get('contract'),residual_version=saved.get('residual_version'),
                parameter_shapes={k:list(v.shape) for k,v in saved['model'].items() if not (k.startswith('denoiser.') or 'transformer.layers' in k)},
                execution_adapter=c.get('execution_adapter'),history_contract=c.get('history_contract'),output_contract=c.get('output_contract'),loss_contract=c.get('loss_contract'),
                normalization_identity=c.get('normalization_identity'),train_ids=saved['train_ids']))
    protected=[]
    for r in read(PROJECT/'research_docs/evidence/project_unification_20261118/protected.json')['files']:
        current=sha(r['path']);assert current==r['sha256'];protected.append(dict(r,current_sha256=current))
    a=ROOTS['accepted72'];b=ROOTS['candidate130']
    for role in 'AB':
        assert sha(a/role/'state_data/statistics.pt')==sha(b/role/'state_data/statistics.pt')
        assert load(a/role/'training/best.pt')['contract']==load(b/role/'training/best.pt')['contract']
    data=read(b/'A/state_data/data.json');assert len(data['splits']['train'])==130 and len(data['splits']['validation'])==6
    assert all(Path(data['entries'][i]['path']).is_file() for i in data['splits']['train']+data['splits']['validation'])
    # Public sufficient statistics must agree exactly with the per-tick receipt.
    tick=list(csv.DictReader(gzip.open(PUBLIC/'tick_exposure.csv.gz','rt')))
    for name,s in ex['summary'].items():
        rr=[r for r in tick if r['model']==name]
        assert len(rr)==s['legal_tick_slots']['n']
        assert sum(int(r['slots'])<50 for r in rr)==s['legal_tick_slots']['below50']
        assert all(int(r['distinct_visits'])<=int(r['slots']) for r in rr)
    for f in read(PUBLIC/'probe_folds.json'):
        assert not set(f['train_families'])&set(f['validation_families'])
        assert f['train_frames']>0 and f['validation_frames']>0
    write(PUBLIC/'source_receipts.json',receipts);write(PUBLIC/'input_contracts.json',contracts)
    write(PUBLIC/'verification.json',dict(seconds=time.time()-began,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protected=protected,
        checkpoints_with_log_hashes=len(receipts),family_fold_checks=len(read(PUBLIC/'probe_folds.json')),data_paths_resolved=136,
        normalizers_byte_identical=True,DP_contracts_identical=True,per_tick_summary_reconciles=True,
        tests=(RUN/'tests.log').read_text(),policy_updates=0,new_simulator_starts=0,new_dagger=0,
        limitations='Protected manifest/weights/statistics fully SHA256 checked;136 raw HDF5 paths and source observations parsed, not another full raw HDF5 SHA sweep. Original400 protection reuses unchanged manifest identity and prior unification full validation.'))

if __name__=='__main__':main()
