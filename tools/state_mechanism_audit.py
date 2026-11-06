"""Read-only data/exposure audit for the bounded StateDP mechanism diagnosis."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import time

import h5py
import numpy as np
import torch

from tools.runtime import read, write, sha
from tools.narrow_repair_audit import stages, motion


class WindowSampler:
    """Diagnostic only: shuffled ALL anchors, multiple windows per episode.

    Keep the short final batch. No duplicated windows, dropped tails or fabricated
    history. The formal episode-balanced sampler remains unchanged.
    """
    def __init__(self, lengths, generator, batch_size=16):
        self.population=[(i,t) for i,n in lengths.items() for t in range(0,n,12)]
        if not self.population or batch_size<1:raise ValueError('Nonempty windows and positive batch required')
        self.generator=generator;self.batch_size=batch_size;self.order=[];self.cursor=0
        self.counts=[0]*len(self.population);self.draws=0

    def draw(self):
        if self.cursor==len(self.order):
            self.order=torch.randperm(len(self.population),generator=self.generator).tolist();self.cursor=0
        indices=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=len(indices)
        for k in indices:self.counts[k]+=1
        self.draws+=len(indices)
        return [self.population[k] for k in indices]

    def state_dict(self):
        return {k:getattr(self,k) for k in ('order','cursor','counts','draws')}

    def load_state_dict(self,state):
        if len(state['counts'])!=len(self.population):raise ValueError('Window identity mismatch')
        for k,v in state.items():setattr(self,k,v)


def distribution(x):
    a=np.asarray(x,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),quantiles=dict(zip(('min','p10','median','p90','max'),np.quantile(a,[0,.1,.5,.9,1]).tolist())))


def audit(root):
    from visuotactile.narrow_data import record, quantiles
    from visuotactile.anchor_sampling import labels
    s=read(root/'session.json');parent=Path(s['parent']);old=Path(read(parent/'session.json')['parent'])
    manifest=read(parent/'manifest.json')['episodes'];data=read(parent/'state_data/data.json')
    cache=torch.load(parent/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    assert len(manifest)==275 and cache['data_sha256']==sha(parent/'state_data/data.json')
    families={};rows=[];phase_map={};maximum=defaultdict(float);tick_count=0
    for entry in manifest:
        assert sha(entry['path'])==entry['sha256'],entry['id']
        key,meta,rec=record(entry['path']);n=len(rec['time']);identity=entry['id']
        assert data['entries'][identity]['length']==n
        assert families.setdefault(entry['family'],entry['split'])==entry['split']
        assert meta['initial_family']==entry['family'] and meta['dataset_split']==entry['split']
        for k,v in rec.items():
            e=float((v-cache['records'][identity][k]).abs().max());maximum['cached_'+k]=max(maximum['cached_'+k],e)
            assert e==0.,(identity,k,e)
        a=torch.arange(0,n,12);y,m=labels(rec,a,24)
        for j,t in enumerate(a.tolist()):
            count=min(24,n-t);assert int(m[j].sum())==count
            assert torch.equal(y[j,:count],rec['command'][t:t+count]) and not torch.count_nonzero(y[j,count:])
        with h5py.File(entry['path']) as f:
            g=f['episodes'][key];success=g['transition/termination_reasons/max_successes'][:].reshape(n,-1).any(-1)
            assert success[-1] and success.sum()==1,'Natural complete success at last transition required'
            before=g['transition/diagnostics_before/joint_position_rad'][:];after=g['transition/diagnostics_after/joint_position_rad'][:]
            maximum['actual_joint_next_tick']=max(maximum['actual_joint_next_tick'],float(abs(after[:-1]-before[1:]).max()))
            touch=g['obs/contact_w_N'][:];assert np.isfinite(touch).all()
            phase=stages(g);phase_map[identity]=phase.tolist();mov=motion(g)
        if entry['multi_view_valid']:
            vp=Path(entry['path']).parent/'visual_acceptance.json'
            assert sha(vp)==entry['visual_acceptance_sha256'] and read(vp)['passed']
        rows.append(dict(identity=identity,family=entry['family'],split=entry['split'],steps=n,duration_s=n/60,
            alpha=entry['alpha'],multi_view_valid=entry['multi_view_valid'],state_valid=True,tactile_numeric_valid=True,
            tactile_scope='Finite real recorded force, first contact_valid=false and subsequent true; no hardware calibration claim',
            phase_seconds={k:v['seconds'] for k,v in mov['phases'].items()},anchor_phases=mov['anchor_phase_counts'],
            stationary_target_fraction=sum(v['ticks']*v['target_hold_fraction'] for v in mov['phases'].values())/n))
        tick_count+=n
        write(root/'heartbeat.json',dict(stage='data_audit',episodes=len(rows),utc=time.time()))
    assert maximum['actual_joint_next_tick']<1e-5,maximum
    initial=read(parent/'families.json');domain=read(parent/'session.json')['domain']
    assert len(initial)==92 and len({f['fingerprint'] for f in initial})==92
    for f in initial:
        assert np.max(np.abs(np.asarray(f['object_xyz'][:2])-domain['center']['object_xyz'][:2]))<=.025001
        assert np.max(np.abs(np.asarray(f['relative_xyz'][:2])-domain['center']['relative_xyz'][:2]))<=.035001
        from visuotactile.narrow_data import angle
        assert angle(f['quaternion'],domain['center']['quaternion'])<=20.001
        assert np.max(np.abs(f['velocity']))<=.150001
    old_test={f['identity'] for f in read(old/'families.json') if f['split']=='test'}
    assert not old_test.intersection({e['family'] for e in manifest if e['split']=='train'})
    groups={}
    for label,predicate in [('all',lambda r:True),('original200',lambda r:not r['multi_view_valid']),('new75',lambda r:r['multi_view_valid']),
        ('native_alpha1',lambda r:r['alpha']==1),('alpha_le_065',lambda r:r['alpha']<=.65),('alpha_065_to_1',lambda r:.65<r['alpha']<1)]:
        selected=[r for r in rows if predicate(r)]
        groups[label]=dict(duration_s=distribution([r['duration_s'] for r in selected]),frames=sum(r['steps'] for r in selected),
            phase_duration_s={p:distribution([r['phase_seconds'].get(p,0) for r in selected]) for p in sorted({p for r in rows for p in r['phase_seconds']})})
    exposures={}
    for label,path in [('state200_best5735',old/'training/best.pt'),('state275_best4500',parent/'expanded_training/best.pt'),('state275_last5000',parent/'expanded_training/last.pt')]:
        cp=torch.load(path,map_location='cpu',weights_only=False);ss=cp['anchor_sampler'];ids=cp['train_ids'];counter=np.array([v for values in ss['counts'].values() for v in values])
        logs=[json.loads(x) for x in (path.parent/'training.jsonl').read_text().splitlines() if json.loads(x)['update']<=cp['updates']]
        assert len(logs)==cp['updates'] and sum(x['actual_batch'] for x in logs)==ss['draws']==int(counter.sum())
        phase=Counter();family=Counter();uniform_windows=Counter();uniform_episodes=Counter()
        for i,counts in ss['counts'].items():
            family[data['entries'][i]['family']]+=sum(counts)
            for j,count in enumerate(counts):
                p=phase_map[i][j*12];phase[p]+=count;uniform_windows[p]+=1;uniform_episodes[p]+=1/len(counts)
        exposures[label]=dict(updates=cp['updates'],train_episodes=len(ids),batch_distribution=dict(Counter(x['actual_batch'] for x in logs)),
            mean_batch=ss['draws']/cp['updates'],anchors=len(counter),draws=ss['draws'],unique_anchors=int((counter>0).sum()),
            anchor_equivalent_epochs=ss['draws']/len(counter),anchor_visits=distribution(counter),
            family_exposure=dict(family),phase_exposure=dict(phase),uniform_anchor_phase_proportions={k:v/len(counter) for k,v in uniform_windows.items()},
            episode_balanced_phase_proportions={k:v/len(ids) for k,v in uniform_episodes.items()},
            prefix_ticks=ss['prefix_ticks'],gradient_ticks=ss['gradient_ticks'],checkpoint_sha256=sha(path))
        del cp
    result=dict(passed=True,episodes=len(rows),ticks=tick_count,groups=groups,rows=rows,reader_max_errors=dict(maximum),
        modalities=dict(state_valid=275,multi_view_valid=75,old_fixed_views_invalid=200,tactile_numeric_valid=275),
        family_splits=dict(Counter(families.values())),initial_quantiles=quantiles(initial,domain['center']),
        exposure=exposures,success_sampling_scope='200/201 original captures plus75/76 expansion captures, conditioned on admitted families; not100% over a continuous initial distribution',
        stage_scope='Diagnostic contact/height/native-stage proxies, not new success definitions',
        no_gpu=True,no_data_mutation=True)
    write(root/'data_audit.json',result);write(root/'phases_by_episode.json',phase_map)
    print('Audited',len(rows),'episodes',tick_count,'ticks; mean seconds',groups['all']['duration_s']['mean'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root.resolve())
