"""Independent read-only checks on actual DAgger records and both consumers."""
import hashlib,json,sys
from pathlib import Path
import h5py,numpy as np,torch

CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS')
SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS'
RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.native_data import robot_rows,visual_records,residual_batch,Episode,depth_grid
from visuotactile.position_control import position_robot_history

torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text())
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def maximum(a,b):return float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float))))
def run():
    result={'populations':[],'episodes':[],'source_hashes':[],'violations':[]}
    old=read(CAN/'outputs/visuotactile/run/tight_exposure50_20261119_01/A/state_data/data.json')
    prev=set(old['splits']['train'])
    for index in (1,2):
        root=RUN/f'round{index:02d}/learning'
        data={r:read(root/r/'state_data/data.json') for r in 'AB'}
        assert data['A']['splits']==data['B']['splits'] and data['A']['entries']==data['B']['entries']
        ids=data['A']['splits']['train'];new=[i for i in ids if i not in prev];prev=set(ids)
        trainfamilies={data['A']['entries'][i]['family'] for i in ids}
        valfamilies={data['A']['entries'][i]['family'] for i in data['A']['splits']['validation']}
        assert not trainfamilies&valfamilies
        assert len({(Path(data['A']['entries'][i]['path']).resolve(),data['A']['entries'][i]['episode']) for i in ids})==len(ids)
        records={r:visual_records(root/r/'state_data',new,'G00') for r in 'AB'}
        row=dict(round=index,train=len(ids),validation=len(data['A']['splits']['validation']),new=len(new),train_families=sorted(trainfamilies),validation_families=sorted(valfamilies),legal_ticks=0)
        result['populations'].append(row)
        for role in 'AB':
            cfg=read(root/role/'training/configuration.json')
            diffs=[]
            for name,expected in cfg['source_sha256'].items():
                if hashlib.sha256((SRC/name).read_bytes()).hexdigest()!=expected:diffs.append(name)
            result['source_hashes'].append(dict(round=index,role=role,recorded=cfg['source_sha256'],current_differences=diffs))
        for identity in new:
            entry=data['A']['entries'][identity]
            with h5py.File(entry['path']) as f:
                g=f['episodes'][entry['episode']];o=g['obs'];t=g['transition'];d=t['dagger'];n=len(o['time_s'])
                meta=json.loads(f.attrs['metadata_json']);mask=d['expert_label_valid'][:];legal=np.flatnonzero(mask);switch=int(entry['takeover_tick'])
                before=t['diagnostics_before/joint_position_rad'][:];after=t['diagnostics_after/joint_position_rad'][:];targets=t['target_after_rad'][:]
                errors={}
                for role,target in [('A',after),('B',targets)]:
                    rec=records[role][identity]
                    errors[role+'_clean_sample']=maximum(rec['clean_sample'][mask],target[mask])
                    errors[role+'_mask']=maximum(rec['supervision_mask'],mask)
                    errors[role+'_time']=maximum(rec['time'],o['time_s'][:])
                    errors[role+'_robot_q']=maximum(rec['robot'][:,:29],before)
                    errors[role+'_robot_current_target']=maximum(rec['robot'][:,58:87],t['target_before_rad'][:])
                    errors[role+'_previous_requested']=maximum(rec['robot'][1:,116:145],targets[:-1])
                errors['q_reached_after']=maximum(d['q_reached'][:][mask],after[mask])
                errors['u_expert_executed']=maximum(d['u_expert'][:][mask],targets[mask])
                errors['actuator_targets']=maximum(t['actuator_substep_target_rad'][:],np.repeat(targets[:,None],2,axis=1))
                errors['trace_executed']=maximum(t['control_trace/applied_position_target_rad'][:],targets)
                errors['q_chain']=maximum(after[:-1],before[1:])
                errors['command_chain']=maximum(targets[:-1],t['target_before_rad'][1:])
                errors['camera_time']=maximum(o['camera_time_s'][:],np.repeat(o['time_s'][:][:,None],3,axis=1))
                errors['contact_world_chain']=maximum(o['contact_substeps_w_N'][1:],t['contact_substeps_after_w_N'][:-1])
                errors['contact_clock_chain']=maximum(o['contact_time_s'][1:],t['contact_time_after_s'][:-1])
                errors['substep_spacing']=maximum(np.diff(o['contact_time_s'][1:],axis=1),1/120)
                errors['camera_duplicate_count']=int(np.sum(np.diff(o['camera_frame'][:],axis=0)<=0))
                row['legal_ticks']+=len(legal)
                samples=sorted({max(0,switch-2),max(0,switch-1),switch,min(n-1,switch+1),int(legal[0]),(switch//12)*12,min(n-1,(switch//12+1)*12),n-1})
                detailed=[]
                for tick in samples:
                    detailed.append(dict(tick=tick,source=int(d['control_source'][tick]),mask=bool(mask[tick]),segment=int(d['segment_id'][tick]),queue=int(d['expert_queue_length'][tick]),source_step=int(d['expert_source_step'][tick]),clean=bool(d['switch_clean'][tick]),time=float(o['time_s'][tick]),camera_frame=o['camera_frame'][tick].tolist(),q_before=before[tick].tolist(),q_after=after[tick].tolist(),u_exec=targets[tick].tolist(),q_label=d['q_reached'][tick].tolist(),u_label=d['u_expert'][tick].tolist()))
                # Both actual consumers retain all prefix ticks; only the expert mask trains.
                backend=root/'B/backend/B/nominal'/(identity+'.pt')
                nominal=torch.load(backend,map_location='cpu',weights_only=False,mmap=True)
                errors['rear_target']=maximum(nominal['target'][mask],targets[mask])
                errors['rear_mask']=maximum(nominal['supervision_mask'],mask)
                errors['rear_robot']=maximum(nominal['robot'],records['B'][identity]['robot'])
                errors['rear_previous_nominal']=maximum(nominal['previous_nominal'][1:],nominal['base'][:-1])
                packed=residual_batch([nominal],max(0,switch-2),min(n,switch+4),'cpu')
                assert packed['valid'].all()
                item=dict(round=index,identity=identity,path=entry['path'],source_role=entry['source_group'],length=n,switch=switch,phase=switch%12,first_legal=int(legal[0]),legal_ticks=len(legal),camera_ids=meta.get('camera_ids'),errors=errors,samples=detailed)
                result['episodes'].append(item)
                bad={k:v for k,v in errors.items() if v>1e-6}
                if bad:result['violations'].append(dict(identity=identity,errors=bad))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write('data_chain.json',result)
    print('violations',json.dumps(result['violations'],ensure_ascii=False),flush=True)
if __name__=='__main__':run()
