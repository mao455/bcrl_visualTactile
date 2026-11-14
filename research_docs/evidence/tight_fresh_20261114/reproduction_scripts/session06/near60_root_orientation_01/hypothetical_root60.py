"""Candidate trigger on the immutable already-completed A52 student prefix only."""
from pathlib import Path
import json,time
import h5py
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import make_trigger,observe_transition,NEVER_FALLBACK
from visuotactile.training_expansion import schedule_trigger

directory=Path(__file__).parent;out=directory/'hypothetical_root60.json'
assert not out.exists()
review=json.loads((directory/'review.json').read_text())
case=next(row for row in review['near60_cases'] if row['held_root_within60mm_ticks'])
source=Path(review['root'])/case['case']/'numeric.h5'
assert sha256(source)==case['numeric_sha256']
trigger=make_trigger(schedule_trigger('held_root60'),NEVER_FALLBACK)
with h5py.File(source,'r') as stream:
    t=stream['episode/transition'];switch=None
    for tick in range(len(t['step_id'])):
        assert int(t['dagger/control_source'][tick])==0
        switch=observe_transition(trigger,tick,t,index=tick)
        if switch is not None:break
    assert switch==265==case['first_held_root60']['completed_tick']
    stage=int(t['diagnostics_after/native_stage_step'][switch-1])
    assert stage==265 and int(t['diagnostics_after/successes'][switch-1])==0
result=dict(version='historical_student_prefix_held_root60_candidate_v1',created_utc=time.time(),
    case=case['case'],source_numeric_sha256=case['numeric_sha256'],review_sha256=sha256(directory/'review.json'),
    script_sha256=sha256(__file__),trigger_source_sha256=sha256(Path('visuotactile/takeover_trigger.py')),
    configuration=trigger.configuration,receipt=trigger.receipt(),
    hypothetical_next_action_takeover_index=switch,completed_real_student_transitions_read=switch,
    original_actual_takeovers=0,original_accepted_trajectories=0,
    native_stage_step=stage,remaining_controls_if_no_subgoal=601-stage,
    goal0_root_above_hole_root_m=case['first_held_root60']['goal_hole_root_offset_m'],
    boundary=case['first_held_root60'],future_transitions_used_to_choose_trigger=False,
    outcome='Counterfactual trigger eligibility only. No expert continuation was executed or data admitted.',
    simulator_starts=0,optimizer_updates=0,expert_actions_generated=0,new_accepted_trajectories=0,
    original_plan_modified=False,rgb_payloads_opened=0)
with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(case=case['case'],hypothetical_switch=switch,receipt=trigger.receipt()),indent=2))
