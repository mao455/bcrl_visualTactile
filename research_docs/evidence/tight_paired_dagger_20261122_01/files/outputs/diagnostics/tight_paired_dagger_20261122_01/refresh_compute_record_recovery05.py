#!/usr/bin/env python3
"""Small-metadata-only accounting. Never reads tensor, HDF5, visual or JSONL payloads."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[3]
DEFAULT_RUN = BASE / 'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
AUDIT = Path(__file__).resolve().parent
MAX_METADATA_BYTES = 8 * 1024 * 1024


def inventory(run):
    sources, errors = {}, []
    def read(p, default=None):
        p = Path(p)
        if not p.exists():
            return default
        try:
            if p.stat().st_size > MAX_METADATA_BYTES:
                raise ValueError('refusing large metadata')
            raw = p.read_bytes()
            value = json.loads(raw)
            sources[str(p)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            return value
        except (OSError, ValueError) as exc:
            errors.append({'path': str(p), 'error': str(exc)})
            return default
    def rows_stats(rows):
        counts = collections.Counter()
        by_role = {x: collections.Counter() for x in 'AB'}
        for row in rows:
            d = row.get('result', row)
            dagger = d.get('dagger', {})
            role = row.get('role') or d.get('role') or row.get('case', d.get('identity', ''))[:1]
            counters = [counts] + ([by_role[role]] if role in by_role else [])
            for c in counters:
                c['rows'] += 1
                complete = d.get('software_complete') is True
                c['software_complete'] += complete
                c['software_incomplete'] += not complete
                c['takeover'] += complete and dagger.get('takeover_triggered') is True
                c['expert_recovered'] += complete and dagger.get('expert_takeover_success') is True
                c['accepted'] += row.get('accepted') is True or d.get('accepted_for_training') is True
                c['admission_not_finalized'] += 'accepted' not in row and 'accepted_for_training' not in d
                c['recorded_control_ticks_complete'] += int(d.get('steps', 0)) if complete else 0
                c['legal_expert_ticks_complete'] += int(dagger.get('expert_valid_ticks', 0)) if complete else 0
        return {'total': dict(counts), 'by_role': {k: dict(v) for k,v in by_role.items()}}
    now = time.time()
    owners, closed = [], []
    for owner in sorted((run/'owners').glob('*')):
        start = read(owner/'project_job_started.json')
        if not start:
            continue
        end = read(owner/'project_exit.json')
        fail = read(owner/'project_failure.json')
        row = {'owner': owner.name, 'start_utc': start.get('utc'), 'end_utc': (end or {}).get('utc'), 'failure': fail}
        if isinstance(row['start_utc'], (int,float)) and isinstance(row['end_utc'], (int,float)):
            row['closed_owner_wall_s'] = row['end_utc']-row['start_utc']
            closed.append([row['start_utc'], row['end_utc']])
        else:
            row['closed_owner_wall_s'] = None
            row['status'] = 'no_closed_receipt; elapsed since start is not certified active time'
        owners.append(row)
    merged=[]
    for start,end in sorted(closed):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(end,merged[-1][1])
        else: merged.append([start,end])
    parents=[]
    for role in 'AB':
        for component,sub in [('DP','training'),('Transformer',f'backend/{role}/training')]:
            directory=run/'common180_parents'/role/sub
            selected=read(directory/'selected.json',{})
            finished=read(directory/'finished.json',{})
            parents.append({'role':role,'component':component,'selected':selected,
                'finished':{k:finished[k] for k in ['complete','epochs','updates','wall_seconds','wall_s','training_wall_s','setup_wall_s','wait_s','work_s'] if k in finished},
                'timer_limit':'DP A wall_seconds resets at recovery; all component timers include host work/waits as documented and are not GPU-active time'})
    discarded=[]
    for p in sorted((run/'common180_parents').glob('*/training/recovery.*.json')):
        d=read(p,{})
        discarded.append({'path':str(p),'durable_updates':d.get('durable_updates'),
            'known_discarded_update_ids':d.get('known_discarded_update_ids',[]),'abandoned_rows':d.get('abandoned_rows'),
            'selected_age_contribution':0})
    recovered=read(run/'collection_consumed_recovered05/recovered_results.json',[])
    continued=[]
    for p in sorted((run/'collection_recovery05').glob('*/result.json')):
        d=read(p)
        if d: continued.append(d)
    writer=[]
    # Overlay replaces original receipt for recovered case; never count original and overlay twice.
    for attempt in ['collection_consumed_recovered05','collection_recovery05']:
        for p in sorted((run/attempt).glob('*/capture_complete.json')):
            d=read(p,{})
            writer.append({'attempt':attempt,'case':p.parent.name,'steps':d.get('steps'),
                'status':d.get('status'),'profile':d.get('profile',{}),
                'recovered_after_process_loss':d.get('recovered_after_process_loss',False)})
    writer_totals=collections.Counter(); writer_n=collections.Counter()
    for w in writer:
        for k,v in w['profile'].items():
            if k.endswith('_s') and isinstance(v,(int,float)):
                writer_totals[k]+=v;writer_n[k]+=1
    memos=[]
    for attempt in ['collection_attempt01_prephysics','collection','collection_recovery05']:
        p=run/attempt/'capture_audit_memo.json';d=read(p)
        if d:memos.append({'attempt':attempt,**d})
    interruption=read(AUDIT/'interruption_observation_20261123_0935.json',{})
    last = interruption.get('last_watcher_state',{}).get('utc')
    restart = read(run/'owners/collection_recovery05/project_job_started.json',{}).get('utc')
    observed_absent=interruption.get('observed_utc_epoch')
    expert_rows=read(run/'experts_shared/results.json',[])
    expert_summary=read(run/'experts_shared/result.json',{})
    efficiency=read(AUDIT/'efficiency_review.json',{})
    total_selected=sum(x['selected'].get('update',0) for x in parents)
    return {'version':'paired_compute_record_recovery05_v1','snapshot_utc_epoch':now,
        'snapshot_utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
        'scope':'CPU-only small metadata snapshot; no payload hashes, log scans, GPU/physics launches or live source/config changes',
        'final':False,'current_limit':'Original96 collection not yet closed; pending new round training and independent evaluation not represented as completed',
        'parent_pairs':read(run/'parent_pairs.json',{}),'parent_reconstruction':{'reason':read(run/'parent_reconstruction_amendment.json',{}).get('reason'),
            'decision':read(AUDIT/'reconstruction_age_audit.json',{}).get('decision')},
        'parent_components':parents,'selected_parent_new_updates_total':total_selected,
        'known_discarded_updates':discarded,'known_discarded_update_count':sum(len(x['known_discarded_update_ids']) for x in discarded),
        'parent_actual_update_lower_bound':total_selected+sum(len(x['known_discarded_update_ids']) for x in discarded),
        'expert_shared':{'summary':expert_summary,'rows':rows_stats(expert_rows),
            'failed_attempts_retained':len(expert_rows)-sum(x.get('accepted') is True for x in expert_rows)},
        'collection':{'frozen_total':96,'frozen_per_role':48,'consumed_original15':rows_stats(recovered),
            'continued_closed_rows_snapshot':rows_stats(continued),
            'continuation_target':81,'progress':read(run/'collection_recovery05/batch_progress.json',{}),
            'original_progress':read(run/'collection/batch_progress.json',{}),
            'no_retry':'Assigned original15 consumed exactly once; incomplete outcomes excluded from action supervision, retained in96 opportunity denominator',
            'observable_pair_checks_expected_after_closure':{'verified':47,'unverified':1,'full48_verified':False}},
        'owner_wall':{'attempts':owners,'closed_union_s':sum(b-a for a,b in merged),'closed_intervals':merged,
            'sum_before_overlap_removal_s':sum(b-a for a,b in closed),
            'meaning':'Closed owner wall includes setup, cache, waits, validation and failed work; interval union avoids overlap double-count; no certified active wall across missing receipt'},
        'interruption':{'last_observed_original_worker_watch_utc':last,'observed_all_old_pids_absent_utc':observed_absent,'continuation_start_utc':restart,
            'unattributed_span_last_watch_to_restart_s':restart-last if restart and last else None,
            'absence_observation_to_restart_s':restart-observed_absent if restart and observed_absent else None,
            'meaning':'Unknown termination time and cause; unattributed span is neither certified active compute nor exact downtime'},
        'recording':{'closed_capture_profiles':writer,'profile_sum_s':dict(writer_totals),'profile_sample_counts':dict(writer_n),
            'meaning':'Cumulative worker/submission measurements overlap across lanes and main thread; never add to owner wall; recovered verification lacks original write/drain timings'},
        'hash_audit_memos':memos,'hash_memo_limit':'Snapshots may only describe startup audits. Logical SHA bytes avoided, not measured physical disk savings; process-local identity checks and expected SHA comparisons preserved; no throughput promise',
        'gpu_hours':None,'gpu_hours_reason':'No per-job GPU active duration integration; wall and sampled device utilization cannot establish GPU-hours, FLOPs or energy',
        'efficiency_existing_evidence':{'conclusion':efficiency.get('conclusion'),'epoch_samples':efficiency.get('epoch_samples'),
            'limitations':efficiency.get('limitations'),'safe_overlap':efficiency.get('safe_cpu_overlap')},
        'source_small_metadata_sha256':sources,'read_errors':errors}


def render(d):
    p=d['collection'];e=d['expert_shared']['summary']
    lines=['# 配对 DAgger 计算与记录开销：Recovery05 快照','',
        f"快照：{d['snapshot_utc']}。采集96机会尚未闭合；本文件不宣称新轮训练或独立评估完成。",'',
        '实际父本由共同180条、相同初始化规则重建。历史entry当前阶段能对齐，但可核验祖先年龄不能对齐；未把历史entry当作等龄父本。', '',
        '| 组件 | A选中epoch / 更新 | B选中epoch / 更新 |','| --- | ---: | ---: |']
    for comp in ['DP','Transformer']:
        rows=[next(x for x in d['parent_components'] if x['role']==r and x['component']==comp) for r in 'AB']
        lines.append('| '+comp+' | '+' | '.join(f"{x['selected'].get('epoch')} / {x['selected'].get('update')}" for x in rows)+' |')
    lines += ['',f"父本四组件有效更新合计{d['selected_parent_new_updates_total']}；另保留{d['known_discarded_update_count']}次已发生但未进入恢复模型的弃用更新，不计选中年龄或曝光。A DP的finished计时在恢复后重置，不能将1326秒当成全部A训练耗时。",'',
        f"共享专家：{e.get('accepted')}/{e.get('attempts')}次尝试获准，覆盖{e.get('families')}家族；其中{e.get('rejected_before_action')}次在动作前拒绝，全部失败与重试保留。",'',
        f"原中断采集已消费15机会：{p['consumed_original15']['total'].get('software_complete',0)}软件完成，{p['consumed_original15']['total'].get('accepted',0)}入池；剩余81仅执行原未分配机会。当前续采集已关闭{p['continued_closed_rows_snapshot']['total'].get('rows',0)}条；续采集逐case结果先于最终准入登记，当前accepted=0不能解释为无可用纠偏。实时进度另见JSON，各文件原子写时刻可能不同。软件未完成保留在96分母，不能记作未触发。",'',
        '最终可观测初态检查预期47对已核验、1对未核验；跨中断缺少完整前动作目标，不能声明48对均同初态。', '',
        f"已闭合owner时间区间去重并集{d['owner_wall']['closed_union_s']:.1f}秒。该值含准备、等待、缓存、验证与失败；未闭合owner不计。中断前最后监测到重启之间约{(d['interruption'].get('unattributed_span_last_watch_to_restart_s') or 0):.1f}秒归因不明，不能当作有效计算，也不能精确宣称全部为停机。没有可信GPU小时数。",'',
        '逐capture异步writer回执保留复制、提交阻塞、numeric/compact写入、flush、验证与SHA耗时；它们跨lane和主线程重叠，JSON中的累计值不能相加解释为墙钟。恢复出的A alignment只记录新CPU验证耗时，原写入计时缺失。', '',
        '| SHA备忘快照 | 实读逻辑字节 | 避免重复逻辑字节 | SHA秒 |','| --- | ---: | ---: | ---: |']
    for m in d['hash_audit_memos']:
        v=m['payload'];lines.append(f"| {m['attempt']} | {v.get('actual_bytes_read')} | {v.get('avoided_full_hash_bytes')} | {v.get('full_hash_seconds',0):.3f} |")
    lines += ['', '这些备忘快照可能仅覆盖启动审计；不等于整轮汇总。复用仅在进程内按dev/inode/size/mtime/ctime身份成立时有效，每个调用仍核对预期SHA。避免字节是SHA逻辑读量，未测物理磁盘流量或速度提升。', '',
        '已有性能观测提示主机/存储压力：A父本部分epoch读约75–80GiB，设备利用率多约18–20%；不构成瓶颈因果定位。安全并行工作是增量小回执/曲线、分母清单和报告整理。当前续采集采用独立有限systemd服务；本次只读盘中小JSON，不扫描视觉/HDF5/大权重或启动GPU。', '',
        '完整逐项计时、各组件原始timer范围、实际父本SHA、失败回执、当前writer样本及小文件SHA在同名JSON。刷新：`python outputs/diagnostics/tight_paired_dagger_20261122_01/refresh_compute_record_recovery05.py`。刷新器不启动任何训练或仿真；最终closure后仍需主报告合并后续训练/评估开销。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=DEFAULT_RUN);ap.add_argument('--output-dir',type=Path,default=AUDIT)
    args=ap.parse_args();d=inventory(args.run.resolve());args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix,body in [('json',json.dumps(d,ensure_ascii=False,indent=2)+'\n'),('md',render(d))]:
        p=args.output_dir/f'COMPUTE_RECORD_RECOVERY05.{suffix}';temp=p.with_name(p.name+f'.tmp.{os.getpid()}');temp.write_text(body);os.replace(temp,p)
    print(json.dumps({'snapshot':d['snapshot_utc'],'selected_updates':d['selected_parent_new_updates_total'],'discarded':d['known_discarded_update_count'],'collection':d['collection']['continued_closed_rows_snapshot']['total'],'read_errors':d['read_errors']}))
if __name__=='__main__':main()
