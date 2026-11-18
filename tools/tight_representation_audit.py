"""Read-only selected-lineage exposure and family-separated sensor diagnostics.

No policy optimizer, simulator, collection, or deployment changes. Each stage has
an exclusive output identity; raw arrays stay under outputs, small reports public.
"""
import argparse, collections, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
RUN=Path(os.environ.get('TIGHT_AUDIT_RUN', PROJECT/'outputs/visuotactile/run/tight_representation_audit_20261119_01'))
PUBLIC=Path(os.environ.get('TIGHT_AUDIT_PUBLIC', PROJECT/'research_docs/evidence/tight_representation_audit_20261119'))
ROOTS={'accepted72':PROJECT/'outputs/visuotactile/run/tight_long_20261117_01/round05/learning',
       'candidate130':PROJECT/'outputs/visuotactile/run/tight_recovery_20261118_01/learning'}

def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return torch.load(p,map_location='cpu',mmap=True,weights_only=False)
def table(p,rows):
    import gzip
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'xt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quant(x):
    x=np.asarray(x);return dict(n=x.size,minimum=float(x.min()) if x.size else None,
        **{f'p{q}':float(np.percentile(x,q)) if x.size else None for q in (5,25,50,75,95)},
        maximum=float(x.max()) if x.size else None,below50=int((x<50).sum()),zero=int((x==0).sum()))
def check_resources():
    import shutil
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    if available<10*2**30 or shutil.disk_usage(RUN).free<12*2**30:raise RuntimeError('Audit RAM/disk reserve reached')

def lineage():
    chains={};nodes={}
    for cohort,root in ROOTS.items():
        for role in 'AB':
            for component in ('DP','rear'):
                p=root/role/('training/best.pt' if component=='DP' else f'backend/{role}/training/best.pt');chain=[]
                while p is not None:
                    key=str(p.resolve());assert key not in chain
                    if key not in nodes:
                        saved=load(p);c=saved['configuration'];parent=c.get('continuation' if component=='DP' else 'continuation_from')
                        nodes[key]=dict(path=key,sha256=sha(p),role=role,component=component,epoch=saved['epoch'],updates=saved['updates'],
                            train_ids=saved['train_ids'],counts=saved.get('sampling_exposure' if component=='DP' else 'episode_counts'),configuration=c,parent=parent)
                    n=nodes[key];chain.append(key);parent=n['parent']
                    if parent and parent.get('path'):
                        p=Path(parent['path']);assert sha(p)==parent['sha256']
                    else:p=None
                chains[f'{cohort}_{role}_{component}']=chain
    return dict(chains=chains,nodes=nodes)

def exposure():
    from visuotactile.anchor_sampling import labels,sample_training_anchors,supervision_rows
    from visuotactile.target_windows import with_target_window
    from visuotactile.geometry_sampling import STAGES
    from scripts.tight_balance_20261117.diagnose import phases
    import h5py
    began=time.time();L=lineage();data=read(ROOTS['candidate130']/'A/state_data/data.json');ids=data['splits']['train']
    current=load(ROOTS['candidate130']/'A/state_data/numeric.pt')['records'];phase={}
    for i in ids:
        e=data['entries'][i]
        with h5py.File(e['path']) as f:phase[i]=phases(f['episodes'][e['episode']]['transition'])
    bynode={};node_reports=[];learning_curves=[];loss_rows=[]
    for key,node in L['nodes'].items():
        check_resources();p=Path(key);component=node['component'];c=node['configuration'];counts=node['counts'];assert counts is not None
        arrays={i:dict(slots=np.zeros(len(current[i]['time']),np.int64),visits_hit=np.zeros(len(current[i]['time']),np.int64),
            weight=np.zeros(len(current[i]['time']),np.float64),geometry=np.zeros(len(current[i]['time']),np.int64),visits=0) for i in ids}
        local=c.get('source_root') if component=='rear' else str(p.parent.parent)
        records=load(Path(local)/'state_data/numeric.pt')['records']
        records=with_target_window(records,c.get('continuation',{}).get('target_window_contract')) if component=='DP' else records
        for i in node['train_ids']:
            assert torch.equal(supervision_rows(records[i])[0],current[i]['supervision_mask'])
        selected_losses=[];epoch_visits=collections.Counter();stages=collections.defaultdict(lambda:collections.Counter());log_count=0
        log=p.parent/('learning.jsonl' if component=='DP' else 'loss.jsonl')
        if component=='DP':
            prev=0
            with log.open() as f:
                for line in f:
                    row=json.loads(line);u=row['updates'];assert u==prev+1;prev=u;log_count+=1
                    learning_curves.append(dict(node=key,component=component,epoch=row['epoch'],update=u,loss=row.get('loss'),gradient_norm=row.get('gradient_norm'),selected=u<=node['updates']))
                    if u>node['updates']:continue
                    selected_losses.append(row.get('loss',0.))
                    for v in row['visits']:
                        i=v['episode'];r=records[i];a,receipt=sample_training_anchors(r,c['training_sampling']['anchors'],identity=i,epoch=row['epoch'],visit_index=v['episode_index'])
                        assert receipt==v['anchor_sampling']
                        _,valid=labels(r,a,24);ticks=(a[:,None]+torch.arange(24)[None])[valid].numpy();assert len(ticks)==v['target_ticks']
                        z=arrays[i];z['visits']+=1;epoch_visits[row['epoch']]+=1
                        np.add.at(z['slots'],ticks,1);z['visits_hit'][np.unique(ticks)]+=1
                        np.add.at(z['weight'],ticks,v['loss_weight']/len(ticks))
                        for k in np.unique(phase[i][ticks]):stages[STAGES[k]]['target_slots']+=int((phase[i][ticks]==k).sum());stages[STAGES[k]]['normalized_slot_coefficient']+=float((phase[i][ticks]==k).sum()*v['loss_weight']/len(ticks))
                        gs=(v.get('geometry_sampling') or {}).get('selected_ticks',[])
                        for g in gs:z['geometry'][g['tick']]+=1
                        # Actual logged per-visit losses; no fake phase allocation of the base loss.
                        for name,value in v.get('parts',{}).items():
                            if value is not None:stages['_logged_components'][name]+=float(value)*v['loss_weight']
                        for gr in (v.get('geometry') or {}).get('rows',[]):
                            k=gr.get('stage')
                            if k is not None and gr.get('included',True):
                                s=stages[STAGES[k]];s['geometry_evaluations']+=1
                                for n in ('geometry_loss','contact_hand_gradient_norm_per_rad','penetration_hand_gradient_norm_per_rad'):
                                    s[n+'_sum']+=float(gr.get(n,0.))
            assert {i:arrays[i]['visits'] for i in counts}==counts
        else:
            for i,n in counts.items():
                z=arrays[i];mask=current[i]['supervision_mask'].numpy();z['visits']=n;z['slots'][mask]=n;z['visits_hit'][mask]=n;z['weight'][mask]=n/128.
                for k in np.unique(phase[i][mask]):stages[STAGES[k]]['target_slots']+=int(((phase[i]==k)&mask).sum())*n;stages[STAGES[k]]['fixed128_coefficient']+=int(((phase[i]==k)&mask).sum())*n/128.
            with log.open() as f:
                for line in f:
                    row=json.loads(line);log_count+=1
                    u=row['update'];assert u==log_count
                    learning_curves.append(dict(node=key,component=component,epoch=row['epoch'],update=u,loss=row.get('loss'),gradient_norm=row.get('gradient_norm'),selected=u<=node['updates']))
                    if u<=node['updates']:selected_losses.append(row['loss'])
            saved=load(p);assert sum(int(current[i]['supervision_mask'].sum())*n for i,n in counts.items())==saved['supervised_ticks']
        assert log_count>=node['updates'];bynode[key]=arrays
        finished=read(p.parent/'finished.json') if (p.parent/'finished.json').exists() else {}
        restores={f.name:read(f) for f in p.parent.glob('*restore*.json')}
        node_reports.append(dict(path=key,sha256=node['sha256'],role=node['role'],component=component,parent=node['parent'],
            train_trajectories=len(node['train_ids']),selected_epoch=node['epoch'],selected_updates=node['updates'],actual_log_updates=log_count,
            actual_finished_epochs=finished.get('epochs',finished.get('epoch')),selected_visits=sum(counts.values()),episode_visits=counts,
            epoch_deck_visits=dict(epoch_visits),batch=c.get('full_trajectory_batch',c.get('microbatch')),initialization=c.get('initialization'),
            optimizer_restored_from_parent=c.get('continuation_audit',{}).get('optimizer_state_restored',False),
            lr=c.get('learning_rate'),local_restore_receipts=restores,ema=component=='DP',selected_loss_mean=float(np.mean(selected_losses)),
            phase_exposure=dict(stages),sampler=c.get('sampling_contract',c.get('episode_sampling')),normalization=c.get('normalization_identity'),
            logged_gradient_scope='Whole update norm (clipped after measurement); geometry output-coordinate norms where logged. Not per-phase parameter gradients.'))
        print('exposure',key.split('/run/')[-1],node['updates'],flush=True)
    summaries={};episode_rows=[];tick_rows=[];npz={}
    for name,chain in L['chains'].items():
        merged={i:{k:sum((bynode[p][i][k] for p in chain)) for k in ('slots','visits_hit','weight','geometry','visits')} for i in ids}
        allslots=[];allhits=[];allvisits=[];rounds=collections.defaultdict(list);phase_bins=collections.defaultdict(list)
        for i,z in merged.items():
            e=data['entries'][i];mask=current[i]['supervision_mask'].numpy();allslots.extend(z['slots'][mask]);allhits.extend(z['visits_hit'][mask]);allvisits.append(z['visits'])
            label='original30' if i in data['normalization_train_ids'] else f"round{e.get('round_index','unknown')}";rounds[label].append(i)
            for k in np.unique(phase[i][mask]):phase_bins[STAGES[k]].extend(z['slots'][mask&(phase[i]==k)])
            episode_rows.append(dict(model=name,identity=i,family=e['family'],population=label,visits=z['visits'],legal_ticks=int(mask.sum()),
                tick_min=int(z['slots'][mask].min()),tick_median=float(np.median(z['slots'][mask])),tick_p95=float(np.percentile(z['slots'][mask],95)),ticks_below50=int((z['slots'][mask]<50).sum()),tick_zero=int((z['slots'][mask]==0).sum()),geometry_draws=int(z['geometry'].sum())))
            for t in np.flatnonzero(mask):tick_rows.append(dict(model=name,identity=i,tick=int(t),phase=STAGES[phase[i][t]],slots=int(z['slots'][t]),distinct_visits=int(z['visits_hit'][t]),normalized_loss_coefficient=float(z['weight'][t]),geometry_draws=int(z['geometry'][t])))
            npz[name+'__'+i]=np.stack((z['slots'],z['visits_hit'],z['weight'],z['geometry']))
        summaries[name]=dict(trajectory_visits=quant(allvisits),legal_tick_slots=quant(allslots),legal_tick_distinct_visits=quant(allhits),
            rounds={k:dict(trajectories=len(ii),visits=quant([merged[i]['visits'] for i in ii]),slots=quant(np.concatenate([merged[i]['slots'][current[i]['supervision_mask'].numpy()] for i in ii]))) for k,ii in rounds.items()},phases={k:quant(v) for k,v in phase_bins.items()},chain=chain)
    np.savez_compressed(RUN/'exposure_arrays.npz',**npz)
    import gzip
    with gzip.open(PUBLIC/'tick_exposure.csv.gz','wt') as f:
        w=csv.DictWriter(f,fieldnames=list(tick_rows[0]));w.writeheader();w.writerows(tick_rows)
    table(PUBLIC/'episode_exposure.csv',episode_rows);table(PUBLIC/'training_curves.csv.gz',learning_curves)
    write(PUBLIC/'exposure.json',dict(chains=L['chains'],stages=node_reports,summary=summaries,seconds=time.time()-began,policy_updates=0,
        scope='Exact selected checkpoint ancestry only; actual logged starts/masks. DP slots can repeat a tick within a visit; distinct visits reported separately. EMA causal support is not an equal additive influence estimate. Historical whole-update norms cannot identify per-phase parameter-gradient contribution.'))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['exposure']);a=p.parse_args();exposure()
