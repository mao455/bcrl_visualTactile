"""Versioned per-visit training sampling; deployment remains on commit12."""
import hashlib
import json
from pathlib import Path
import torch

VERSION='fresh_random_visit_training_v1'

def validate_profile(value):
    if value is None:return None
    from .anchor_sampling import validate_training_anchor_profile
    from .geometry_sampling import validate_profile as validate_geometry
    if not isinstance(value,dict) or set(value)!={'version','anchors','geometry'} or value['version']!=VERSION:
        raise ValueError('Explicit random visit sampling profile required')
    anchors=validate_training_anchor_profile(value['anchors'])
    geometry=validate_geometry(value['geometry'])
    if anchors is None or geometry is None:raise ValueError('Both planning and geometry sampling must be explicit')
    return dict(version=VERSION,anchors=anchors,geometry=geometry)

def load_profile(root):
    path=Path(root)/'training_sampling.json'
    return validate_profile(json.loads(path.read_text())) if path.exists() else None

def visit(record,profile,*,identity,epoch,visit_index):
    from .anchor_sampling import sample_training_anchors
    profile=validate_profile(profile)
    anchors,receipt=sample_training_anchors(record,profile['anchors'],identity=identity,epoch=epoch,visit_index=visit_index)
    payload=json.dumps(dict(kind='hand_geometry',profile=profile,identity=identity,epoch=epoch,visit_index=visit_index),
        sort_keys=True,separators=(',',':'))
    seed=int(hashlib.sha256(payload.encode()).hexdigest()[:16],16)%(2**63)
    return anchors,receipt,torch.Generator(device='cpu').manual_seed(seed)

class Coverage:
    """Exact union of real episode/tick positions, not repeated window slots."""
    def __init__(self,state=None):
        self.state=json.loads(json.dumps(state)) if state is not None else {}
    def add(self,identity,anchors,geometry):
        row=self.state.setdefault(identity,dict(visits=0,anchor_draws=0,anchor_ticks=[],geometry_draws=0,geometry_ticks=[],geometry_stages={}))
        row['visits']+=1
        selected=anchors['selected_anchors'];row['anchor_draws']+=len(selected)
        row['anchor_ticks']=sorted(set(row['anchor_ticks'])|set(selected))
        if geometry:
            chosen=geometry['selected_ticks']
            if any(v['identity']!=identity for v in chosen):raise ValueError('Geometry exposure mixes trajectory identities')
            ticks=[v['tick'] for v in chosen]
            if len(ticks)!=len(set(ticks)):raise ValueError('Geometry visit repeats the same actual configuration')
            row['geometry_draws']+=len(ticks);row['geometry_ticks']=sorted(set(row['geometry_ticks'])|set(ticks))
            for v in chosen:
                stage=v['stage'];row['geometry_stages'][stage]=sorted(set(row['geometry_stages'].get(stage,[]))|{v['tick']})
    def state_dict(self):return json.loads(json.dumps(self.state))
    def summary(self):
        result=dict(episodes=len(self.state),visits=0,anchor_draws=0,unique_anchor_ticks=0,unique_non_native_grid_ticks=0,
            geometry_draws=0,unique_geometry_ticks=0,geometry_stage_unique_ticks={})
        for row in self.state.values():
            for k in ('visits','anchor_draws','geometry_draws'):result[k]+=row[k]
            result['unique_anchor_ticks']+=len(row['anchor_ticks'])
            result['unique_non_native_grid_ticks']+=sum(t%12!=0 for t in row['anchor_ticks'])
            result['unique_geometry_ticks']+=len(row['geometry_ticks'])
            for stage,ticks in row['geometry_stages'].items():
                result['geometry_stage_unique_ticks'][stage]=result['geometry_stage_unique_ticks'].get(stage,0)+len(ticks)
        return result
