import copy
import json
import math

import pytest

from tools.tight_fresh_study_plot import autonomous_panels, plot


def evidence():
    result=dict(complete=True,test_used=False,version='matched_anchor_study_report_v1',report_schema_revision=2,
        arm_semantics={arm:dict(label=label) for arm,label in [('source','Source (round_05)'),
            ('control','Nominal anchor'),('previous','Previous-target anchor')]},roles={},
        source_autonomous=dict(cases=[],totals={}),autonomous=dict(cases=[],totals={'control':{},'previous':{}}))
    for role in ('A','B'):
        result['roles'][role]=dict(arms={arm:dict(selected_train_scope={'identities':['train1','train2']},
            validation_identities=['val'+str(i) for i in range(6)],history=[dict(epoch=1,train={'all':{'hand_rmse_rad':.02}},
            validation={'all':{'hand_rmse_rad':.03}})]) for arm in ('control','previous')},
            onset_fitting={arm:dict(all_corrections=dict(episodes=54,bins={'first12':{'hand_rmse_rad':.04}}))
                for arm in ('source','control','previous')})
        for arm in ('source','control','previous'):
            totals=dict(opportunities=2,legal=2,invalid_initializations=0,native_lift_cases=2,settled_stable_cases=1,
                near_hole_cases=0,first_subgoal_cases=0,successes=0)
            cohort=result['source_autonomous'] if arm=='source' else result['autonomous']
            if arm=='source':cohort['totals'][role]=totals
            else:cohort['totals'][arm][role]=totals
            for i in range(2):
                snapshot=dict(root_quaternion_angle_rad=math.pi/2,stable_window=False) if i==0 else None
                cohort['cases'].append(dict(case=role+arm+str(i),role=role,arm=arm,legal=True,
                    phase_snapshots=dict(first_goal_after_hold_best_orientation=snapshot)))
    return result


def test_no_prior_hold_is_missing_not_zero_rotation_and_source_denominator_is_separate():
    data=evidence();panels=autonomous_panels(data,'A')
    assert len(panels)==3 and panels[0]['label']=='Source (round_05)'
    assert all(len(panel['turns'])==1 and panel['turns'][0]['degrees']==90 for panel in panels)
    assert all(panel['totals']['opportunities']==2 for panel in panels)
    altered=copy.deepcopy(data);altered['autonomous']['totals']['control']['A']['opportunities']=3
    with pytest.raises(ValueError,match='denominator'):autonomous_panels(altered,'A')


def test_plot_writes_actual_milestone_and_orientation_panels_from_complete_audited_report(tmp_path):
    source=tmp_path/'report.json';source.write_text(json.dumps(evidence()));output=tmp_path/'figure.png'
    plot(source,output)
    assert output.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n' and output.stat().st_size>10000
