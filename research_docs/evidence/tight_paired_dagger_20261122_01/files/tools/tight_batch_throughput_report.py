"""CPU-only comparison of completed collection receipts; no simulator or retries."""
import argparse
import math
from pathlib import Path
import time

from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def positive_seconds(value,name,*,zero=False):
    if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or value<0 or (value==0 and not zero)):
        raise ValueError('Invalid elapsed time: '+name)
    return float(value)


def population(root,plan,rows,entries):
    cases={case['name']:case for case in plan['cases']}
    if len(cases)!=len(plan['cases']) or len(rows)!=len(cases) or {row['case'] for row in rows}!=set(cases):
        raise ValueError('Every registered opportunity must complete exactly once')
    accepted=set();counts=dict(attempts=len(cases),legal_attempts=0,triggered_attempts=0,
        accepted_trajectories=0,all_control_ticks=0,all_valid_expert_ticks=0,accepted_valid_expert_ticks=0,
        student_autonomous_successes=0)
    for row in rows:
        value=row['result'];case=cases[row['case']];dagger=value.get('dagger',{})
        if row['role']!=case['role'] or value.get('software_complete') is not True:
            raise ValueError('Case role or software completion differs from registration')
        steps=value.get('steps',0);valid=dagger.get('expert_valid_ticks',0)
        if any(type(v) is not int or v<0 for v in (steps,valid)) or valid>steps:
            raise ValueError('Malformed control or expert-label denominator')
        counts['legal_attempts']+=bool(value.get('legal'))
        counts['triggered_attempts']+=dagger.get('expert_ticks',0)>0
        counts['all_control_ticks']+=steps;counts['all_valid_expert_ticks']+=valid
        counts['student_autonomous_successes']+=bool(dagger.get('student_autonomous_success'))
        if value.get('accepted_for_training'):
            if not (value.get('legal') and value.get('success') and dagger.get('expert_takeover_success') and valid):
                raise ValueError('Accepted trajectory lacks a successful valid expert suffix')
            if value.get('actual_exposure',{}).get('supervised_ticks')!=valid:
                raise ValueError('Accepted supervision denominator differs from actual exposure')
            accepted.add(row['case']);counts['accepted_valid_expert_ticks']+=valid
    if accepted!=set(entries):raise ValueError('Accepted rows and actual training entries differ')
    for name,entry in entries.items():
        if entry['episode']!=name:raise ValueError('Accepted episode identity differs from its case')
    counts['accepted_trajectories']=len(accepted)
    return counts


def rates(counts,seconds):
    return dict(attempts_per_hour=counts['attempts']*3600/seconds,
        accepted_trajectories_per_hour=counts['accepted_trajectories']*3600/seconds,
        accepted_valid_expert_ticks_per_hour=counts['accepted_valid_expert_ticks']*3600/seconds,
        all_valid_expert_ticks_per_hour=counts['all_valid_expert_ticks']*3600/seconds,
        aggregate_control_ticks_per_second=counts['all_control_ticks']/seconds)


def summarize(root,*,batch):
    root=Path(root).resolve();bindings={}
    def load(name):
        path=root/name
        if not path.exists():raise ValueError('Incomplete collection; missing '+str(path))
        bindings[name]=dict(path=str(path),sha256=sha256(path))
        return read(path)
    owner=load('owner_finished.json');result=load('result.json');plan=load('plan.json')
    if (owner.get('software_complete') is not True or owner.get('returncode')!=0
            or result.get('software_complete') is not True or result.get('worker_error') is not None):
        raise ValueError('Incomplete collection cannot receive a throughput conclusion')
    if plan.get('kind')!='takeover' or plan.get('test_used') is not False:
        raise ValueError('Registered non-test collection required')
    rows=load('results.json');entries=load('correction_entries.json')
    session=load('session.json');initialization=load('initialization.json')
    counts=population(root,plan,rows,entries)
    if result.get('cases')!=counts['attempts']:raise ValueError('Completion case count differs')
    seconds=positive_seconds(owner['seconds'],'owner.seconds')
    startup=positive_seconds(initialization['seconds'],'initialization.seconds',zero=True)
    profile=dict(owner_seconds=seconds,initialization_seconds=startup,
        owner_timing_scope='Owned worker startup, execution, output and admission until owner completion',
        primary_rates=rates(counts,seconds))
    measurement_sources=dict(owner_seconds='owner_finished.json: seconds',
        initialization_seconds='initialization.json: seconds',
        counts='results.json: each result steps, dagger and actual_exposure; accepted set must equal correction_entries.json',
        primary_rates='counts divided by owner_finished.json: seconds, scaled to seconds or hours')
    if batch:
        waves=load('wave_profiles.json')
        if len({wave['wave'] for wave in waves})!=len(waves):raise ValueError('Duplicate wave profile')
        if sum(wave['cases'] for wave in waves)!=counts['attempts']:
            raise ValueError('Wave opportunity counts differ')
        if sum(wave['control_ticks'] for wave in waves)!=counts['all_control_ticks']:
            raise ValueError('Wave control-tick counts differ')
        wave_seconds=sum(positive_seconds(wave['seconds'],'wave.seconds') for wave in waves)
        elapsed=positive_seconds(result['elapsed_seconds'],'result.elapsed_seconds')
        outside=elapsed-startup-wave_seconds
        if outside < -1e-6:raise ValueError('Nested wave/startup timing exceeds worker elapsed time')
        # Wave wall includes recorder waits; worker component times overlap it.
        # Neither lane lifetimes nor async worker timings are added to wall time.
        admission=(sum(positive_seconds(wave['admission_s'],'wave.admission_s',zero=True) for wave in waves)
            if all('admission_s' in wave for wave in waves) else None)
        profile.update(worker_elapsed_seconds=elapsed,worker_rates=rates(counts,elapsed),
            wave_seconds=wave_seconds,wave_rates=rates(counts,wave_seconds),
            wave_timing_scope='Reset/controller/real simulation and recording through lane verification; excludes subsequent admission',
            admission_seconds=admission,
            outside_wave_after_initialization_seconds=max(0.,outside),
            outside_wave_scope='Admission plus model loading, per-wave setup and manifest/report overhead; not pure admission',
            async_profile_sums={key:sum(positive_seconds(row['result'].get('async_capture',{}).get('profile',{}).get(key,0),key,zero=True)
                for row in rows) for key in ('submission_wait_s','copy_s','worker_numeric_append_s',
                'worker_compact_append_s','worker_flush_s','worker_finish_s','worker_verify_s','worker_hash_s',
                'drain_and_finalize_wait_s')},
            async_scope='Attribution only. Worker and producer timings overlap; do not sum them into elapsed time.',
            writers=[dict(wave=wave['wave'],statistics=wave.get('capture_writer')) for wave in waves])
        measurement_sources.update(worker_elapsed_seconds='result.json: elapsed_seconds',
            wave_seconds='wave_profiles.json: sum of seconds',
            admission_seconds='wave_profiles.json: sum of admission_s only when present in every wave',
            async_profile_sums='results.json: sum of each result.async_capture.profile component; attribution only',
            writers='wave_profiles.json: capture_writer snapshots after verification and before close')
    else:
        case_seconds=sum(positive_seconds(row['timing']['total_seconds'],'case.total_seconds') for row in rows)
        profile.update(serial_case_seconds=case_seconds,serial_case_rates=rates(counts,case_seconds),
            serial_case_scope='Sum of sequential case execution; excludes initial Kit startup and outer admission/report overhead',
            synchronous_profile_sums={key:sum(positive_seconds(row['result'].get('wall_profile',{}).get(key,0),key,zero=True)
                for row in rows) for key in ('expert_inference_s','environment_step_render_s','numeric_write_s',
                'compact_append_s','finalize_verify_s','setup_s','through_result_s')})
        measurement_sources.update(serial_case_seconds='results.json: sum of each timing.total_seconds',
            synchronous_profile_sums='results.json: sum of each result.wall_profile component; nested components are not additive')
    case_identity=[dict(role=case['role'],source_identity=case['spec'].get('source_identity'),
        family=case['spec']['family'],narrow_reset=case['spec']['narrow_reset'],
        trigger=case.get('dagger',{}).get('trigger'),takeover_tick=case['spec'].get('takeover_tick'))
        for case in plan['cases']]
    return dict(root=str(root),complete=True,execution='batch' if batch else 'serial',counts=counts,
        profile=profile,measurement_sources=measurement_sources,bindings=bindings,source_commit=plan['source_commit'],
        source_population_sha256=plan.get('source_data_sha256'),
        pairs={role:{key:pair[key] for key in ('frontend_sha256','backend_sha256')} for role,pair in plan['pairs'].items()},
        registered_case_conditions=case_identity,backend=plan.get('execution_backend'),
        batch=plan.get('batch'),first_plan_forensics=plan.get('first_plan_forensics',False),
        session_deadline=session['deadline'])


def report(serial_root,batch_root=None):
    serial=summarize(serial_root,batch=False)
    output=dict(version='actual_collection_throughput_v1',created_utc=time.time(),serial=serial,
        scope='Complete physical collection and accepted training entries; not student autonomous success or optimizer exposure',
        caveats=['Source initials and trigger assignments can match; native batch RNG is not a serial replay.',
            'Concurrency, kernel microbatch, threading, forensic recording and I/O can differ together; ratios are whole-pipeline observations.',
            'Accepted expert-tick rates exclude rejected recoveries; all valid expert ticks are reported separately.',
            'One short batch does not establish stable throughput or autonomous behavior improvement.'])
    if batch_root is not None:
        current=summarize(batch_root,batch=True);output['batch']=current
        key=lambda row:(row['role'],row['source_identity'])
        left=sorted(serial['registered_case_conditions'],key=key);right=sorted(current['registered_case_conditions'],key=key)
        ratios={key:(current['profile']['primary_rates'][key]/value if value else None)
            for key,value in serial['profile']['primary_rates'].items()}
        output['comparison']=dict(same_pairs=serial['pairs']==current['pairs'],same_registered_case_conditions=left==right,
            ratios_batch_over_serial=ratios,
            ratio_scope='Owner-to-owner complete run denominators; descriptive, not causal or exact paired replay')
    return output


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--serial-root',type=Path,required=True)
    parser.add_argument('--batch-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=report(args.serial_root,args.batch_root)
    if args.output.exists():raise FileExistsError('Preserve the previous report; choose a new output identity')
    atomic_json(args.output,value)


if __name__=='__main__':main()
