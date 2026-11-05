"""Read-only official-source/checkpoint audit; no Kit or optimizer imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from tools.runtime import CODE,read,write,sha


def audit(reference):
    def git(*args):return subprocess.check_output(['git',*args],cwd=CODE)
    upstream=git('rev-parse','upstream/main').decode().strip()
    base=git('merge-base','HEAD',upstream).decode().strip()
    session=read(reference/'session.json');checkpoint=Path(session['expert_checkpoint'])
    remote=json.load(urllib.request.urlopen('https://huggingface.co/api/models/Kushal20/play2perfect-checkpoints',timeout=30))
    revision=remote['sha']
    url=f'https://huggingface.co/api/models/Kushal20/play2perfect-checkpoints/tree/{revision}/tight_insertion'
    files=json.load(urllib.request.urlopen(url,timeout=30));released=next(x for x in files if x['path']=='tight_insertion/model.pth')
    digest=sha(checkpoint);assert digest==released['lfs']['oid']
    stage=reference/'stages/smoke_expert/attempt_1/output'
    cfg=read(stage/'environment.json')['config'];actor=read(stage/'actor_configuration.json')
    paths=['assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf',
        'assets/urdf/peg_in_hole/lpeg_matchedmass/lpeg_matchedmass.urdf',
        'assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf',
        'isaacsimenvs/tasks/play/utils/action_utils.py','isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py',
        'evaluation/eval_isaacsim.py','evaluation/eval_offline.py']
    source={}
    for name in paths:
        old=git('show',base+':'+name);new=git('show',upstream+':'+name)
        source[name]=dict(local_sha256=sha(CODE/name),official_base_sha256=hashlib.sha256(old).hexdigest(),
            official_latest_sha256=hashlib.sha256(new).hexdigest(),
            equal_to_official_base=(CODE/name).read_bytes()==old,
            equal_to_official_latest=(CODE/name).read_bytes()==new)
    geometries={}
    for name in paths[:3]:
        tree=ET.parse(CODE/name)
        geometries[name]=dict(collision_boxes=[dict(size=x.find('geometry/box').get('size'),
            origin=x.find('origin').get('xyz') if x.find('origin') is not None else '0 0 0')
            for x in tree.findall('.//collision') if x.find('geometry/box') is not None],
            sdf_markers=len(tree.findall('.//collision/sdf')),mass_kg=[float(x.get('value')) for x in tree.findall('.//mass')])
    expert=read(CODE/'research_docs/evidence/state_dp/success_timing_audit.json')
    hole=geometries[paths[2]]['collision_boxes'];peg=geometries[paths[1]]['collision_boxes']
    slot=[]
    for axis in (0,1):
        wall=next(x for x in hole if float(x['origin'].split()[axis])>0)
        slot.append(2*float(wall['origin'].split()[axis])-float(wall['size'].split()[axis]))
    peg_cross=sorted(float(x) for x in peg[0]['size'].split()[1:])
    differences=[a-b for a,b in zip(sorted(slot),peg_cross)]
    assert all(abs(v-.001)<1e-9 for v in differences)
    value=dict(state='static_and_existing_runtime_audit_complete_new_baseline_pending',
        official_repo='https://github.com/kushal2000/play2perfect',official_base=base,official_latest=upstream,
        paper='https://arxiv.org/html/2606.26428v1',checkpoint=dict(path=str(checkpoint),sha256=digest,
            bytes=checkpoint.stat().st_size,official_hf_revision=revision,official_metadata_url=url,official_lfs_sha256=released['lfs']['oid'],assembly_not_play=True),
        source=source,local_native_changes_since_official_base=git('diff','--name-only',base,'HEAD','--','isaacsimenvs','evaluation','assets/urdf').decode().splitlines(),
        runtime_configuration={k:cfg[k] for k in ('assets','action','termination','precise_assembly','domain_randomization','sim','decimation')},
        actuator_configuration=read(stage/'actuator_configuration.json'),actor_network=actor['params']['network'],
        observation_normalization='Released checkpoint running_mean_std; deterministic actor, actions clipped[-1,1], intrinsic coefficient50',
        geometries=geometries,clearance=dict(nominal_per_side_m=.0005,paper_total_dimension_difference_m=.001,
            filename='tol0p5mm denotes per-side margin; not the paper0.5mm total-clearance row',
            slot_cross_section_m=slot,peg_cross_section_m=peg_cross,computed_total_clearance_m=differences,
            object_primary_box_m=[.25,.03,.02],object_secondary_box_m=[.02,.08,.02],
            sdf_for_this_pair=False,scale_6_25_0_75_0_5='Geometry descriptor scaled by25; not a25x USD collider scale'),
        existing400_time_sim_s=expert['expert_duration_sim_s'],existing400_verified=expert['expert_passed'],
        proven_differences=['Official eval defaults disable action/observation delay and external wrenches; current production keeps native DR',
            'Released tol0p5mm asset has1mm total cross-section difference, whereas paper defines clearance as total difference',
            'Current wrapper uses explicit world_hold wrench frames in this installed Lab; not a bitwise claim against bare official SDK behavior',
            'CPU physics single lane for new tests versus source collection four lanes; historical hidden PhysX state cannot be fully reconstructed',
            'Paper describes final1cm/palm0.2m; released native code uses ordered goals and7.5mm/fingertipmean0.1m. Keep released native metric',
            'Paper real-world uses FoundationPose30Hz; simulation reads native state with configured noise/delay'],
        unknowns=['No official typical simulation completion-time distribution found',
            'Hardware-specific PD gains, safety speed filters and exact timing exclusions are not released in minimal deployment reference',
            'Current release mass57.8g matches our assets; exact asset/mass used to train checkpoint is not established by release comments',
            'Measured3.412s is conditional on400 successes, not an unconditional completion-time expectation'])
    write(CODE/'research_docs/evidence/slow_expert/configuration_audit.json',value)
    return value
