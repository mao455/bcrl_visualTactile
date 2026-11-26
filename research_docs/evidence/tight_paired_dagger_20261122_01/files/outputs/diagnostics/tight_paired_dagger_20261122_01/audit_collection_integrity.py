"""CPU read-only collection audit. Writes only a new AUDIT result directory.

No CUDA/Kit, no retries, no input file changes. Production summarize is reused
with its sole JSON writer intercepted, so collection_summary.json is untouched.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))
from audit_collection_noise import read, require, assignment_checks, planning_scope, numeric_schedule


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def production_summary(module, output):
    writes = []
    def capture(path, value):
        require(Path(path).resolve() == output/'collection_summary.json', 'Unexpected summary write')
        writes.append(value)
    with patch.object(module, 'atomic_json', capture), contextlib.redirect_stdout(io.StringIO()):
        summary = module.summarize(output)
    require(writes == [summary], 'Unexpected production summary side effects')
    return summary


def check_numeric(directory, result, *, compact_path=None):
    """Verify all actual history, masks and labels; no visual payload loading."""
    import numpy as np
    import h5py
    with h5py.File(directory/'numeric.h5') as handle:
        group = handle['episode']; d = group['transition/dagger']; tr = group['transition']
        n = result['steps']
        codes, valid, clean, segment = (d[key][:] for key in ('control_source','expert_label_valid','switch_clean','segment_id'))
        require(all(value.shape == (n,) for value in (codes,valid,clean,segment)), 'Malformed authority/mask/segment')
        require(valid.dtype == np.bool_ and clean.dtype == np.bool_, 'Masks must be boolean')
        require(np.isin(codes,[0,1]).all() and not (codes[1:] < codes[:-1]).any(), 'Noncontiguous student prefix')
        require(not (valid & ((codes != 1) | ~clean)).any() and not valid[0], 'Illegal student/queue-transition action supervision')
        require(int(valid.sum()) == result['dagger']['expert_valid_ticks'], 'Legal tick denominator differs')
        require(not (((codes[1:] != codes[:-1]) | (valid[1:] != valid[:-1])) & (segment[1:] == segment[:-1])).any(), 'Mask/authority switch lacks segment')
        for key in ('actual_issued','actual_applied','u_expert','q_reached','previous_issued'):
            require(d[key].shape == (n,29) and np.isfinite(d[key][:]).all(), 'Nonfinite/malformed '+key)
        issued = d['actual_issued'][:]; applied = d['actual_applied'][:]
        require(np.array_equal(issued,applied) and np.array_equal(applied,tr['target_after_rad'][:]), 'Issued/applied/target chain changed')
        require(np.array_equal(d['previous_issued'][1:],issued[:-1]), 'Historical feedback replaced by predictions')
        require(np.array_equal(d['u_expert'][:][valid],applied[valid]), 'Expert command label differs from executed command')
        require(np.array_equal(d['q_reached'][:],tr['diagnostics_after/joint_position_rad'][:]), 'Reached label differs from actual after-state')
        require(group['obs/step_id'][:].tolist() == list(range(n)), 'Full real prefix history not preserved')
        compact_checked = False
        if compact_path is not None:
            with h5py.File(compact_path) as compact:
                cg = compact['episodes'][result['identity']]
                # Read only numeric robot/time/feedback/label/tactile paths.
                fields = ['obs/step_id','obs/time_s','obs/proprio',
                          'obs/actuator_target_rad','obs/contact_substeps_w_N','obs/contact_substeps_local_N',
                          'obs/contact_time_s','obs/contact_valid']
                fields += ['transition/diagnostics_before/joint_position_rad','transition/diagnostics_before/joint_velocity_rad_s',
                           'transition/diagnostics_after/joint_position_rad','transition/target_before_rad','transition/target_after_rad']
                fields += ['transition/dagger/'+key for key in d.keys()]
                for field in fields:
                    require(field in group and field in cg, 'Missing real compact historical field '+field)
                    require(np.array_equal(group[field][:],cg[field][:]), 'Compact historical/label field differs '+field)
                terminal = cg['terminated'][:].astype(bool) | cg['truncated'][:].astype(bool)
                require(not terminal[:-1].any() and terminal[-1], 'Compact episode crosses reset or is incomplete')
            compact_checked = True
        return dict(legal_expert_ticks=int(valid.sum()), student_prefix_ticks=int((codes==0).sum()),
                    queue_or_invalid_expert_ticks=int(((codes==1)&~valid).sum()),
                    compact_complete_numeric_history_checked=compact_checked)


def audit(output):
    from tools import tight_paired_collection as collection
    output = Path(output).resolve()
    plan = read(output/'plan.json'); entries = read(output/'correction_entries.json')
    registration = read(output/'registration.json')
    require(sha(output/'plan.json') == registration['plan_sha256'], 'Frozen collection plan changed')
    require(sha(PROJECT/'tools/tight_paired_collection.py') == registration['wrapper_sha256'], 'Frozen collector wrapper changed')
    require(sha(plan['parent_pairs_path']) == plan['parent_pairs_sha256'], 'Actual parent pair ledger changed')
    collection.validate_plan(plan)
    assignment_checks(plan,read(output/'paired_queue_assignments.json'),read(output/'paired_initial_check.json'))
    summary = production_summary(collection,output)
    rows = read(output/'results.json'); cases = {c['name']:c for c in plan['cases']}
    require(len(rows)==96 and len({r['case'] for r in rows})==96, 'Distinct96 result denominator required')
    funnels = {role:{op:dict(opportunities=0,invalid_initialization=0,software_incomplete=0,untriggered=0,
                            takeovers=0,expert_recovered=0,valid_trajectories=0,legal_expert_ticks=0,
                            never_entered_at_takeover=0,ever_entered_then_left_at_takeover=0,
                            ever_entered_still_in_gate_at_takeover=0)
                         for op in ('support','alignment')} for role in 'AB'}
    details = []; problems = []; source_hashes = {}
    for row in rows:
        name = row['case']; case = cases[name]; result = row['result']; directory = output/name
        require(read(directory/'result.json') == result, 'Aggregate/individual result differs '+name)
        require(row['role']==case['role'], 'Result role differs')
        role,op = case['role'],case['opportunity']; f=funnels[role][op]; f['opportunities']+=1
        legal = bool(result.get('legal')); complete = bool(result.get('software_complete'))
        f['invalid_initialization']+=not legal; f['software_incomplete']+=not complete
        if not legal or not complete:
            require(name not in entries and not result.get('accepted_for_training'), 'Invalid/incomplete trajectory admitted')
            problems.append(dict(case=name,issue='invalid_initialization' if not legal else 'software_incomplete',retained_in_denominator=True))
            continue
        require(result['complete_student_policy']==plan['pairs'][role], 'Collector not own complete parent pair')
        trigger = collection.audit_trigger(directory,result)  # exact read-only causal replay
        switched = trigger['actual_switch_tick'] is not None
        recovered = bool(result['dagger']['expert_takeover_success'])
        accepted = name in entries
        f['takeovers']+=switched; f['untriggered']+=not switched; f['expert_recovered']+=switched and recovered
        receipt=result['dagger']['trigger_receipt']
        if switched:
            never=bool(receipt['never_entered_15mm']); ever=bool(receipt['ever_entered_15mm']); left=bool(receipt['ever_entered_then_left_15mm'])
            require(never != ever and (not left or ever), 'Inconsistent gate history')
            f['never_entered_at_takeover']+=never; f['ever_entered_then_left_at_takeover']+=left
            f['ever_entered_still_in_gate_at_takeover']+=ever and not left
        require(not recovered or switched, 'Expert recovery incorrectly counted as untriggered autonomous success')
        require(result['dagger']['student_autonomous_success']==bool(result['success'] and not switched), 'Autonomous success definition differs')
        compact = Path(result['dataset_capture']['path']) if result.get('dataset_capture') else None
        numeric = check_numeric(directory,result,compact_path=compact)
        scope=planning_scope(result,read(directory/'native_inference_timing.json'),numeric_schedule(directory))
        f['legal_expert_ticks']+=numeric['legal_expert_ticks'];f['valid_trajectories']+=accepted
        require(accepted == bool(result.get('accepted_for_training')), 'Accepted flag/manifest differs')
        if accepted:
            entry=entries[name]
            require(switched and recovered and result['success'] and numeric['legal_expert_ticks']>0, 'Unsuccessful/nonexpert suffix admitted')
            require(entry['source_group']==role and entry['opportunity']==op and entry['episode']==name
                    and entry['family']==case['spec']['family'] and entry['length']==result['steps'], 'Own-pool correction identity differs')
            require(Path(entry['path']).resolve()==compact.resolve(), 'Admitted compact source differs')
            source_hashes[name]=sha(compact)
            require(source_hashes[name]==entry['source_sha256']==result['dataset_capture']['sha256'], 'Admitted payload SHA differs')
            for field,entryfield in [('never_entered_15mm','never_entered_15mm_at_takeover'),('ever_entered_15mm','ever_entered_15mm_at_takeover'),('ever_entered_then_left_15mm','ever_entered_then_left_15mm_at_takeover')]:
                require(entry[entryfield]==receipt[field], 'Own entry gate history differs')
        details.append(dict(case=name,role=role,opportunity=op,takeover=switched,expert_recovered=recovered,
                            accepted_for_own_pool=accepted,trigger_reason=receipt['reason'],**numeric,**scope))
    require(all(sum(v['opportunities'] for v in funnels[role].values())==48 for role in 'AB'), '48 opportunities per role not preserved')
    require(all(v['opportunities']==24 for by_op in funnels.values() for v in by_op.values()), 'Each registered opportunity needs24 per role')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA')
    return dict(version='paired_collection_integrity_audit_v1',utc=time.time(),passed=True,
                opportunities=96,paired_opportunities=48,production_summary=summary,by_assigned_opportunity=funnels,
                details=details,technical_failures=problems,admitted_payload_sha256=source_hashes,
                cuda_initialized=False,read_only_inputs=True,
                distinctions=['Production stages are actual trigger reasons; assigned support/alignment denominators remain24 each/role',
                              'Expert recovery is conditional on takeover, separate from autonomous success',
                              'Never-entered and ever-entered-then-left describe history only until takeover',
                              'All invalid/software-incomplete opportunities retained in96; no retry or substitution'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,default=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/collection')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        result=audit(args.collection)
        (args.output/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=result['passed'],opportunities=96,paired_opportunities=48,technical_failures=len(result['technical_failures']))))
    except BaseException as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(exc),utc=time.time()),indent=2)+'\n')
        raise


if __name__=='__main__':main()
