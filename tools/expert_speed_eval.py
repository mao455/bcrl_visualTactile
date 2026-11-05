"""Bounded expert study, dispatched by the existing anchor_interim_eval CLI.

The original State150 owner/source is never modified. This CPU owner waits for
its completed 200 trials before acquiring the shared simulation resource lease.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import CODE,RUN,read,write,sha,kill_owned,proc
from tools.anchor_continue_queue import Owner
from tools.anchor_interim_eval import compare_distribution,wilson

FACTORS={'native':None,'identity':dict(kind='closed_loop',arm=1.,hand=1.,phase='all'),
    **{f'A{x}':dict(kind='closed_loop',arm=v,hand=v,phase='all') for x,v in [('080',.8),('067',.67),('050',.5)]},
    **{f'B{x}':dict(kind='recorded_retime',stretch=v) for x,v in [('100',1.),('125',1.25),('150',1.5),('200',2.)]},
    'C067':dict(kind='closed_loop',arm=.67,hand=.67,phase='after_first_subgoal')}


def prepare(root,reference):
    from tools.expert_speed_audit import audit
    reference=reference.resolve();evidence=audit(reference)
    old=read(reference/'session.json');data=read(Path(old['source_root'])/'data.json')
    timing=read(CODE/'research_docs/evidence/state_dp/success_timing_audit.json')['experts']
    # Duration-stratified training exemplars, selected before new outcomes.
    eligible=sorted([x for x in timing if x['identity'] in data['splits']['train']],key=lambda x:(x['ticks'],x['identity']))
    indices=[round((len(eligible)-1)*(i+.5)/10) for i in range(10)]
    specs=[dict(identity=eligible[j]['identity'],cohort='train',seed=23010001+i,
        selection_index=i,source_duration_sim_s=eligible[j]['duration_sim_s']) for i,j in enumerate(indices)]
    assert len({x['identity'] for x in specs})==10
    root.mkdir(parents=True,exist_ok=False);now=time.time()
    s=copy.deepcopy(old);s.update(evaluation_kind='expert_speed',run_id=root.name,started_epoch=now,deadline=now+24*3600,
        training_deadline=now+24*3600,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        dependency_root=str(reference),specs=specs,expansion_specs=[dict(identity=f'fresh_{23020001+i}',cohort='fresh',seed=23020001+i) for i in range(20)],
        collection_specs=[dict(identity=f'fresh_{23030001+i}',cohort='fresh',seed=23030001+i) for i in range(6)],
        kit_maximum_starts=155,kit_auto_retry=False,syscall_diagnostic_stages=[],owned_worker_roots=[str(root)],
        factors=FACTORS,reference_baseline_minimum_successes=8,expert_checkpoint_sha256=evidence['checkpoint']['sha256'],
        no_training_until_new_dataset_contract_passes=True)
    # Old evaluation-specific fields cannot silently apply to this study.
    for key in ('prior_software_starts','predecessor','smoke_specs','smoke_fresh','cohort_targets'):s.pop(key,None)
    write(root/'session.json',s);write(root/'sampling_manifest.json',dict(pilot=specs,expansion=s['expansion_specs'],
        capture=s['collection_specs'],selected_before_new_outcomes=True,test40_used=False,baseline='native ordered complete success'))
    (root/'freeze').mkdir();shutil.copyfile(reference/'freeze/user.config.json',root/'freeze/user.config.json')
    (root/'private_runtime').mkdir();(root/'gpu.lock').symlink_to(reference/'gpu.lock')
    live_cfg=read(reference/'stages/smoke_expert/attempt_1/output/environment.json')['config']
    # URDF files are byte-identical; only this worktree's absolute asset paths move.
    for key in ('object_urdf','receptive_urdf'):
        old_path=Path(live_cfg['assets'][key])
        new_path=CODE/'assets'/str(old_path).split('/assets/',1)[1]
        assert sha(old_path)==sha(new_path);live_cfg['assets'][key]=str(new_path)
    digest=hashlib.sha256(json.dumps(live_cfg,sort_keys=True).encode()).hexdigest()
    write(root/'freeze/benchmark.json',dict(specs=specs,configuration_sha256=digest,outcome_selected=False))
    shutil.copyfile(reference/'initialization_distribution.json',root/'initialization_distribution.json')
    shutil.copyfile(reference/'original_initial_signatures.json',root/'original_initial_signatures.json')
    paths=[p for base in ('visuotactile','tools','isaacsimenvs','evaluation') for p in (CODE/base).rglob('*.py')]
    paths.append(CODE/'visuotactile/run_cpu.sh')
    write(root/'runtime_freeze.json',{str(p.relative_to(CODE)):sha(p) for p in paths})
    write(root/'status.json',dict(state='waiting_dependency',stage='existing_State150_200',pid=None))
    write(root/'configuration_audit.json',evidence)
    report=CODE/'research_docs/SLOW_EXPERT_RESULTS.md'
    write(root/'publication_request.json',dict(expected_head=s['source_commit'],body_sha256=report_body_hash(report.read_text())))


def report_body_hash(text):
    start='<!-- SLOW_EXPERT_RESULTS_START -->';end='<!-- SLOW_EXPERT_RESULTS_END -->'
    if start in text:
        a,b=text.split(start,1);_,c=b.split(end,1);text=a+c
    return hashlib.sha256('\n'.join(x.rstrip() for x in text.splitlines() if x.strip()).encode()).hexdigest()


class ExpertSpeedEvaluation(Owner):
    def status(self,**kw):
        state=read(self.root/'status.json');state.update(pid=os.getpid(),heartbeat=time.time());state.update(kw)
        write(self.root/'status.json',state)

    def validate_kit(self,record):
        import h5py
        import numpy as np
        import torch
        from visuotactile.success_audit import analyze
        from visuotactile.video import verify_video
        from tools.trajectory_metrics import compare_initial
        t=record['result']['trials'][0];folder=Path(record['output'])/t['identity']
        if not t['legal']:
            assert t.get('not_started');record.update(initialization_rejected=True,native_outcome_valid=False);return
        assert t['checkpoint_sha256']==self.s['expert_checkpoint_sha256']
        assert not t['action_exhausted'] and t['actuator_target_max_error']==0
        cfg=read(folder/'metadata.json')['config'];expected=read(self.root/'initialization_distribution.json')
        actual={k:cfg[k] for k in expected if k!='physics_dt'};actual['physics_dt']=cfg['sim']['dt']
        record['distribution_contract']=compare_distribution(actual,expected)
        with np.load(folder/'trace.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        with h5py.File(folder/'numeric.h5') as f:
            state=torch.load(folder/'initial_export.pt',map_location='cpu',weights_only=False)['env_tensors']
            audit,_=analyze(f['episode'],read(folder/'metadata.json'),folder/'success_audit.npz',goal_state=state)
            g=f['episode'];before=g['transition/target_before_rad'][:];after=g['transition/target_after_rad'][:]
            assert np.array_equal(before[1:],after[:-1]),'Target history must use actually applied targets'
            assert np.allclose(np.diff(g['obs/time_s'][:].reshape(-1)),1/60,atol=1e-7)
            if 'transition/speed_native_target_rad' in g:
                from visuotactile.expert_speed import progress_target
                speed=t['expert_speed'];native=torch.from_numpy(g['transition/speed_native_target_rad'][:])
                active=np.ones(len(native),dtype=bool) if speed['phase']=='all' else g['transition/diagnostics_before/successes'][:].reshape(-1)>=1
                predicted=progress_target(torch.from_numpy(before),native,speed['arm'],speed['hand']).numpy()
                predicted[~active]=native.numpy()[~active]
                assert np.array_equal(predicted,after),'Progression applied differently or more than once'
            def stats(x):
                if not len(x):return None
                return dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(np.max(x)))
            d=g['transition/diagnostics_after'];dt=float(cfg['sim']['dt'])*cfg['decimation']
            record['physical_motion']=dict(
                object_linear_speed_m_s=stats(np.linalg.norm(d['object_velocity_w_m_s_rad_s'][:,:3],axis=-1)),
                object_angular_speed_rad_s=stats(np.linalg.norm(d['object_velocity_w_m_s_rad_s'][:,3:],axis=-1)),
                palm_step_speed_m_s=stats(np.linalg.norm(np.diff(d['palm_body_pose_w'][:,:3],axis=0)/dt,axis=-1)),
                scope='Actual physics samples at control boundaries; each trial separately summarized')
        assert audit['passed'] and audit['first_complete_tick']==t['first_success_tick']
        assert t['termination_tick']==audit['termination_tick'] and t['post_success_actions']==0
        write(folder/'success_audit.json',audit);record['success_audit']=audit
        job=read(self.root/'jobs'/(record['name']+'.json'))
        if job.get('factor'):
            weights=read(Path(record['output'])/'expert_weight_audit.json')
            assert weights['passed'] and weights['optimizer_updates']==0
            record['frozen_expert_weights']=weights
        if job.get('paired_baseline'):
            initial=compare_initial(folder,Path(job['paired_baseline']))
            assert initial['observable_equal'] and initial['environment_rng_equal'] and not initial['exported_differences'],initial
            record['paired_initial']=initial
        if job.get('identity_gate'):
            with np.load(folder/'trace.npz') as a,np.load(Path(job['paired_baseline'])/'trace.npz') as b:
                comparisons={k:np.array_equal(a[k],b[k]) for k in b.files}
            record['identity_fields']=comparisons;assert all(comparisons.values()),comparisons
        record['video']=verify_video(folder/'rollout.mp4',t['steps'])
        record.update(native_outcome_valid=True,trace_sha256=sha(folder/'trace.npz'),sdk_exit=dict(code=record['exit_code'],
            isolated_exit=(Path(record['output'])/'process_exit_requested.json').exists(),graceful=False))

    def execute(self,name,spec,factor,baseline=None,capture=False):
        speed=self.s['factors'][factor]
        job=dict(model='Frozen RL '+factor,updates=0,mode='expert' if speed is None else
            'expert_retime' if speed['kind']=='recorded_retime' else 'expert_slow',specs=[spec],
            checkpoint=self.s['expert_checkpoint'],checkpoint_sha256=self.s['expert_checkpoint_sha256'],
            video=True,capture_pair=False,capture_dataset=capture,expert_speed=speed,
            lifecycle=dict(fast_shutdown=True,disarm_stop=True),factor=factor)
        if baseline:
            job['paired_baseline']=str(Path(baseline['output'])/spec['identity'])
            if factor=='identity':job['identity_gate']=True
            if speed and speed['kind']=='recorded_retime' and spec['cohort']=='fresh':
                job['replay_numeric']=str(Path(job['paired_baseline'])/'numeric.h5')
        record=self.kit(name,job)
        # Persist the protocol identity before the next stage is launched.
        record['factor']=factor;record['phase']=name.split('_')[0]
        self.attempts=[record if x['name']==name and x['number']==record['number'] else x for x in self.attempts]
        if record.get('native_outcome_valid'):
            t=record['result']['trials'][0];category=(factor,'success' if t['success'] else '+'.join(k for k,v in t['reasons'].items() if v))
            prior=[x for x in self.attempts if x is not record and x.get('video_category')==list(category) and x.get('retained_video')]
            video=Path(record['output'])/t['identity']/'rollout.mp4';keep=len(prior)<2 or factor=='identity'
            record['video_category']=list(category);record['retained_video']=str(video) if keep else None
            if not keep and video.exists():
                record['discarded_video']=dict(path=str(video),sha256=sha(video),reason='verified duplicate viewing category; full numeric evidence retained');video.unlink()
        write(self.root/'attempts.json',self.attempts);write(self.root/'stages'/name/'committed.json',record)
        self.report();return record

    def trials(self,phase,factor):
        return [x['result']['trials'][0] for x in self.attempts if x.get('phase')==phase and x.get('factor')==factor and x.get('native_outcome_valid')]

    def report(self):
        import numpy as np
        rows=[]
        for phase in ('pilot','expansion','capture'):
            for factor in self.s['factors']:
                trials=self.trials(phase,factor)
                if not trials:continue
                ok=[t for t in trials if t['success']];k=len(ok);n=len(trials)
                rows.append(dict(phase=phase,factor=factor,valid=n,success=k,rate=k/n,ci95=wilson(k,n),
                    mean_success_sim_s=float(np.mean([t['duration_sim_s'] for t in ok])) if ok else None,
                    median_success_sim_s=float(np.median([t['duration_sim_s'] for t in ok])) if ok else None,
                    fall=sum(t['reasons']['fall'] for t in trials),timeout=sum(t['reasons']['timeout'] for t in trials)))
        completed=[x for x in self.attempts if 'exit_code' in x]
        summary=dict(status=read(self.root/'status.json'),rows=rows,maximum_starts=self.s['kit_maximum_starts'],
            actual_starts=len(self.attempts),software_missing=sum(not x.get('valid') for x in completed),
            acceptance_failed=sum(bool(x.get('acceptance_error')) for x in completed),
            initialization_rejections=sum(bool(x.get('initialization_rejected')) for x in completed),
            checkpoint_sha256=self.s['expert_checkpoint_sha256'],source_commit=self.s['source_commit'],
            planned_pilot_specs=10,planned_expansion_specs=20,planned_capture_specs=6,
            missing_successful_replay_sources=len(list((self.root/'missing_replay_sources').glob('*'))),
            gates={name:read(self.root/(name+'.json')) for name in ('baseline_gate','pilot_gates','selection','expansion_gate') if (self.root/(name+'.json')).exists()},
            train_optimizer_updates=0,slow_dataset=(read(self.root/'dataset_manifest.json') if (self.root/'dataset_manifest.json').exists() else None),
            resources=(read(self.root/'resources/resources.json') if (self.root/'resources/resources.json').exists() else None),
            trials=[dict(name=x['name'],factor=x.get('factor'),phase=x.get('phase'),trial=x['result']['trials'][0],
                events={k:v for k,v in x['success_audit'].items() if k!='motion'},motion=x['success_audit'].get('motion'),
                physical_motion=x.get('physical_motion'),video=x.get('retained_video'))
                for x in self.attempts if x.get('native_outcome_valid')])
        write(self.root/'summary.json',summary)
        public=CODE/'research_docs/evidence/slow_expert';write(public/'results.json',summary)
        lines=['<!-- SLOW_EXPERT_RESULTS_START -->','## 实测进度','',
            f"状态：`{summary['status']['state']}`，阶段`{summary['status'].get('stage')}`；已尝试{summary['actual_starts']}/{summary['maximum_starts']}。软件缺失{summary['software_missing']}，验收失败{summary['acceptance_failed']}，非法初态{summary['initialization_rejections']}。",
            '', '|阶段|配置|原生成功/有效|95% Wilson区间|成功均值/中位(s)|fall/timeout|','|---|---|---:|---|---|---|']
        for r in rows:
            duration='N/A' if r['mean_success_sim_s'] is None else f"{r['mean_success_sim_s']:.3f}/{r['median_success_sim_s']:.3f}"
            lines.append(f"|{r['phase']}|{r['factor']}|{r['success']}/{r['valid']}|{r['ci95'][0]:.1%}–{r['ci95'][1]:.1%}|{duration}|{r['fall']}/{r['timeout']}|")
        if not rows:lines += ['','尚无本轮物理rollout；CPU测试和等待服务不代表慢速执行成功。']
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plotted=[r for r in rows if r['phase']!='capture'];x=np.arange(len(plotted))
            fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
            rate=np.array([r['rate']*100 for r in plotted]);low=np.array([r['ci95'][0]*100 for r in plotted]);high=np.array([r['ci95'][1]*100 for r in plotted])
            axes[0].bar(x,rate,color='#387e9d');axes[0].errorbar(x,rate,yerr=np.maximum(0,np.stack([rate-low,high-rate])),fmt='none',color='black',capsize=3)
            axes[0].set_ylabel('Native success % (95% CI)');axes[0].set_ylim(0,105)
            axes[1].axhspan(5,7,color='#78ad6b',alpha=.2)
            for j,r in enumerate(plotted):
                if r['median_success_sim_s'] is not None:axes[1].bar(j,r['median_success_sim_s'],color='#bd8545')
            axes[1].set_ylabel('Success median simulation s');axes[1].set_xticks(x,[f"{r['phase']}:{r['factor']}\nn={r['valid']}" for r in plotted],rotation=45,ha='right')
            axes[0].set_title('Frozen expert: all valid failures retained; duration conditional on success')
            fig.tight_layout();fig.savefig(public/'speed_comparison.png',dpi=140);plt.close(fig)
            lines += ['','![原生成功与真实完成时间](evidence/slow_expert/speed_comparison.png)']
            for row in summary['trials']:
                if row['video']:
                    t=row['trial'];lines.append(f"- [{row['name']}：{'成功' if t['success'] else '失败'}，{t['steps']}ticks]({row['video']})")
        lines += ['',f"本地原始记录：`{self.root}`。首次成功与正式终止、阶段时刻、运动统计及视频见[机器结果](evidence/slow_expert/results.json)。",
            '成功耗时仅对成功样本统计，不能忽略失败或据此声称固定倍率。无训练更新；Image未启动。','<!-- SLOW_EXPERT_RESULTS_END -->','']
        report=CODE/'research_docs/SLOW_EXPERT_RESULTS.md'
        if report.exists():
            text=report.read_text();start='<!-- SLOW_EXPERT_RESULTS_START -->';end='<!-- SLOW_EXPERT_RESULTS_END -->'
            if start in text:
                a,b=text.split(start,1);_,c=b.split(end,1);text=a+'\n'.join(lines)+c
            else:text+='\n'+'\n'.join(lines)
            report.write_text(text)

    def wait_dependency(self):
        parent=Path(self.s['dependency_root'])
        while True:
            if time.time()>self.s['deadline'] or (self.root/'STOP').exists():raise RuntimeError('Fixed deadline/STOP while waiting for protected State evaluation')
            s=read(parent/'status.json')
            if (parent/'owner_exit.json').exists() and s.get('pid') is None:
                assert s['state']=='completed' and read(parent/'summary.json')['valid']==200,'Protected queue stopped incomplete; inspect before new Kit work'
                oldmonitor=read(parent/'monitor.json')['pid']
                cmd=Path('/proc',str(oldmonitor),'cmdline')
                assert not cmd.exists() or b'visuotactile.state_resources' not in cmd.read_bytes(),'Original monitor has not exited'
                break
            self.status(state='waiting_dependency',stage='existing_State150_200',worker_pid=None);self.report();time.sleep(30)

    def run(self):
        from visuotactile.expert_speed import reliability
        root=self.root
        with (root/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                for p,d in read(root/'runtime_freeze.json').items():assert sha(CODE/p)==d,'Source drift: '+p
                assert sha(self.s['expert_checkpoint'])==self.s['expert_checkpoint_sha256']
                self.wait_dependency()
                cache=root/'private_runtime/cache'
                if not cache.exists():shutil.copytree(Path(self.s['dependency_root'])/'private_runtime/cache',cache,symlinks=True)
                (root/'STOP_MONITOR').unlink(missing_ok=True)
                with (root/'monitor.log').open('a') as log:
                    self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                baseline=[]
                for i,spec in enumerate(self.s['specs']):
                    baseline.append(self.execute(f'pilot_native_{i:02d}',spec,'native'))
                    if i==0:self.execute('pilot_identity_00',spec,'identity',baseline[0])
                native=self.trials('pilot','native')
                gate=dict(valid=len(native),success=sum(t['success'] for t in native),minimum_success=8,minimum_valid=10)
                gate['passed']=gate['valid']==10 and gate['success']>=8
                write(root/'baseline_gate.json',gate)
                if not gate['passed']:
                    self.status(state='completed_diagnostic',stage='native_baseline_unreliable_no_slow_expansion');return
                for factor in ('A080','A067','A050','B100','B125','B150','B200'):
                    for i,spec in enumerate(self.s['specs']):self.execute(f'pilot_{factor}_{i:02d}',spec,factor,baseline[i])
                if all(sum(t['success'] for t in self.trials('pilot',f))<8 for f in ('A080','A067','A050')):
                    for i,spec in enumerate(self.s['specs']):self.execute(f'pilot_C067_{i:02d}',spec,'C067',baseline[i])
                gates={f:reliability(self.trials('pilot',f),native) for f in ('A080','A067','A050','B125','B150','B200','C067')}
                # Failed unscaled open-loop recovery cannot certify time warping.
                if sum(t['success'] for t in self.trials('pilot','B100'))<8:
                    for f in ('B125','B150','B200'):gates[f]['passed']=False;gates[f]['replay_reference_unreliable']=True
                write(root/'pilot_gates.json',gates)
                candidates=[f for f,g in gates.items() if g['passed']]
                if not candidates:
                    self.status(state='completed_diagnostic',stage='no_reliable_5_to_7_second_candidate_no_data_or_training');return
                chosen=min(candidates,key=lambda f:(-gates[f]['success'],abs(gates[f]['median_success_s']-6.),f))
                write(root/'selection.json',dict(factor=chosen,rule='success first, then median duration closest6s; prespecified, not test40',pilot=gates[chosen]))
                for i,spec in enumerate(self.s['expansion_specs']):
                    base=self.execute(f'expansion_native_{i:02d}',spec,'native')
                    if chosen.startswith('B') and not base['result']['trials'][0].get('success'):
                        write(root/'missing_replay_sources'/spec['identity'],dict(reason='No successful native source; candidate not executed',spec=spec));continue
                    self.execute(f'expansion_{chosen}_{i:02d}',spec,chosen,base)
                gate=reliability(self.trials('expansion',chosen),self.trials('expansion','native'),minimum=20,success_floor=.85)
                if sum(t['success'] for t in self.trials('expansion','native'))<16:gate['passed']=False;gate['native_baseline_unreliable']=True
                write(root/'expansion_gate.json',gate)
                if not gate['passed']:
                    self.status(state='completed_diagnostic',stage='random20_reliability_not_met_no_dataset_or_training');return
                entries=[]
                for i,spec in enumerate(self.s['collection_specs']):
                    base=self.execute(f'capture_native_{i:02d}',spec,'native') if chosen.startswith('B') else None
                    if base and not base['result']['trials'][0].get('success'):continue
                    record=self.execute(f'capture_{chosen}_{i:02d}',spec,chosen,base,capture=True)
                    t=record['result']['trials'][0]
                    if t.get('success'):
                        capture=t['dataset_capture'];assert capture['accepted_success'] and capture['verified_source_fields']>100
                        entries.append(dict(identity=t['identity'],seed=t['seed'],**capture))
                write(root/'dataset_manifest.json',dict(version='slow_expert_pilot_v1',entries=entries,successes=len(entries),attempts=6,
                    original400_unchanged=True,supervision='transition/target_after_rad',training_ready=False,
                    limitation='Pilot source-field reconstruction passed; State offline/live feature pairing and adequate independent train/validation coverage require audit before fine-tuning'))
                self.status(state='dataset_contract_review',stage='real_sensor_pilot_saved_not_yet_training_ready')
            except BaseException:
                write(root/'owner_error.json',dict(utc=time.time(),error=traceback.format_exc()));self.status(state='blocked',worker_pid=None);raise
            finally:
                if self.worker:kill_owned(self.worker)
                if self.monitor:
                    (root/'STOP_MONITOR').touch()
                    try:self.monitor.wait(timeout=15)
                    except subprocess.TimeoutExpired:kill_owned(self.monitor)
                self.status(pid=None,worker_pid=None,finished=time.time());self.report();write(root/'owner_exit.json',read(root/'status.json'))
                try:self.publish()
                except BaseException:write(root/'publication_status.json',dict(state='blocked',error=traceback.format_exc()))

    def publish(self):
        request=read(self.root/'publication_request.json')
        def git(*args):return subprocess.check_output(['git',*args],cwd=CODE,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
        assert git('branch','--show-current')=='feature/expert-slow-execution'
        assert git('remote','get-url','origin')=='https://github.com/mao455/bcrl_visualTactile.git'
        assert git('rev-parse','HEAD')==request['expected_head'],'Intervening commit requires review'
        assert not git('diff','--cached','--name-only'),'Unrelated staged work'
        assert report_body_hash((CODE/'research_docs/SLOW_EXPERT_RESULTS.md').read_text())==request['body_sha256']
        paths=['research_docs/SLOW_EXPERT_RESULTS.md','research_docs/evidence/slow_expert/results.json']
        if (CODE/'research_docs/evidence/slow_expert/speed_comparison.png').exists():paths.append('research_docs/evidence/slow_expert/speed_comparison.png')
        git('add','--',*paths)
        if git('diff','--cached','--name-only'):git('commit','-m','Report measured frozen expert speed diagnostics and data gates')
        git('push','origin','HEAD:feature/expert-slow-execution')
        write(self.root/'publication_status.json',dict(state='published',commit=git('rev-parse','HEAD')))
