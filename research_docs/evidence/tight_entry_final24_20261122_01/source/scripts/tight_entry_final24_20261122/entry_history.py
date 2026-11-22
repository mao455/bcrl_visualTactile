"""Matched real-history entry inference; no training or simulator.

Prepare is CPU-only. Run uses the production GPU controller, preserved dense
microbatch-2 visual cache and actual issued/applied feedback. Each policy sees
the same entire real prefix and an explicitly saved common CUDA noise stream.
"""
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
AUDIT = PROJECT/'outputs/diagnostics/tight_entry_final24_20261122_01'
RUN = PROJECT/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'
NEW = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def prepare():
    from visuotactile.native_data import visual_records
    from visuotactile.entry_exposure import population
    root = NEW/'A/state_data'
    data = read(root/'data.json')
    records = visual_records(root, data['splits']['train'], 'G00',
                             target_window_contract='recovery_onset_commit12_v1')
    entries = population(records)
    new50 = {r['identity'] for r in csv.DictReader((AUDIT/'A_new50_entry_selected_20261122.csv').open())}
    assert len(entries) == 150 and len(new50) == 50 and new50 <= set(entries)
    cache = read(NEW/'A/cache_index.json')
    cases = []
    for identity, e in entries.items():
        entry = data['entries'][identity]
        source = Path(entry['path'])
        assert source.exists()
        with h5py.File(source) as f:
            g = f['episodes'][entry['episode']]
            actual_mask = torch.from_numpy(g['transition/dagger/expert_label_valid'][:]).bool()
            assert torch.equal(actual_mask, records[identity]['supervision_mask'].cpu())
            first = int(actual_mask.nonzero()[0])
            assert e['anchor'] == first//12*12
            assert e['ticks'][0] == first
            assert all(bool(actual_mask[t]) for t in e['ticks'])
            assert first == int(entry['takeover_tick'])+2
            assert 'actual_issued' in g['transition/dagger']
        visual = cache['entries'][identity]
        cases.append(dict(identity=identity, new50=identity in new50, **e,
            first_legal=first, stop=min(entry['length'], e['anchor']+12),
            raw_path=str(source), raw_source_sha256=entry['source_sha256'],
            visual_cache_path=visual['path'], visual_cache_sha256=visual['sha256']))
    pairs = {}
    original = read(RUN/'round01/collection00_recovery02/plan.json')['pairs']
    for role in 'AB':
        pairs[role] = {}
        paths = {'original130':Path(original[role]['backend']),
                 'original_R2':RUN/f'round02/learning/{role}/backend/{role}/training/best.pt',
                 'entry_repaired':NEW/f'{role}/backend/{role}/training/best.pt'}
        for tag, path in paths.items():
            saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            base_path = Path(saved['configuration']['base_checkpoint'])
            base_sha = sha(base_path)
            assert base_sha == saved['configuration']['base_sha256']
            pairs[role][tag] = dict(backend=str(path), backend_sha256=sha(path),
                frontend=str(base_path), frontend_sha256=base_sha,
                checkpoint_sampler_seed=saved['contract']['sampler_seed'])
            del saved
    result = dict(version='matched_real_history_native_entry_v1', cases=cases, pairs=pairs,
        cases_total=len(cases), slots_total=sum(len(c['ticks']) for c in cases),
        new50_cases=len(new50), new50_slots=sum(len(c['ticks']) for c in cases if c['new50']),
        noise=dict(seed=7091, per_episode='reset common CUDA generator to saved seed7091 state',
            shape=[1,24,29], planning_ticks='0,12,...,native entry anchor',
            historical_stream_claim=False, checkpoint_seed_alone_is_not_queue_rng_position=True),
        history='Complete actual tick0 prefix through native entry commit; overwrite requested/applied feedback with recorded real issued/applied acknowledgements after every tick',
        vision=dict(source='existing shared dense frozen production microbatch2 cache',
            profile=read(NEW/'visual_arithmetic.json'),
            evidence=str(AUDIT/'visual_cache_verification.json'),
            evidence_sha256=sha(AUDIT/'visual_cache_verification.json'),
            all_frames_reencoded=False),
        target_spaces=dict(A_nominal='q_reached', B_nominal='u_expert',
            residual='u_expert - bounded nominal, radians',
            final_command='u_expert absolute target after production clip/projection'),
        interpretation='Paired teacher-forced fit on fixed real observations; not autonomous recovery or a counterfactual physical outcome',
        optimizer_updates=0, physics_steps=0)
    write(AUDIT/'entry_same_history_plan.json', result)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','cases_total','slots_total','new50_cases','new50_slots')}), flush=True)


def metrics(prediction, target):
    error = prediction-target
    return dict(arm_rmse_rad=float(error[:7].square().mean().sqrt()),
                hand_rmse_rad=float(error[7:].square().mean().sqrt()),
                equal_mse_rad2=float(.5*(error[:7].square().mean()+error[7:].square().mean())),
                max_abs_rad=float(error.abs().max()))


@torch.no_grad()
def run(args):
    from visuotactile.gpu_batch_controller import GPUBatchedResidualController
    from visuotactile.native_data import Episode, visual_records
    from visuotactile.precision import state_fp32
    torch.set_num_threads(2)
    arithmetic = state_fp32()
    plan = read(AUDIT/'entry_same_history_plan.json')
    session = read(AUDIT/'session.json')
    from visuotactile.state_resources import lease_for
    lease = lease_for(AUDIT, deadline=session['deadline'])
    assert time.time() < session['deadline']
    out = AUDIT/args.output
    if out.exists():
        raise FileExistsError('Use an independent result directory; preserve prior attempts')
    out.mkdir()
    selected = [c for c in plan['cases'] if args.population == 'all150' or c['new50']]
    if args.identities:
        wanted = set(args.identities.split(','))
        selected = [c for c in selected if c['identity'] in wanted]
        assert {c['identity'] for c in selected} == wanted
    if args.limit:
        selected = selected[:args.limit]
    assert selected
    write(out/'registration.json', dict(plan_sha256=sha(AUDIT/'entry_same_history_plan.json'),
        command_arguments=vars(args), cases=[c['identity'] for c in selected], arithmetic=arithmetic,
        started_utc=time.time(), deadline=session['deadline'], optimizer_updates=0, physics_steps=0))
    noise_state = torch.Generator(device=args.device).manual_seed(plan['noise']['seed']).get_state()
    torch.save(noise_state, out/'common_initial_cuda_rng_state.pt')
    initial_rng_sha = tensor_sha(noise_state)
    all_rows = []
    began = time.time()
    for role in args.roles:
        root = NEW/f'{role}/state_data'
        data = read(root/'data.json')
        records = visual_records(root, [c['identity'] for c in selected], 'G00',
                                 target_window_contract='recovery_onset_commit12_v1')
        for tag in args.tags.split(','):
            pair = plan['pairs'][role][tag]
            assert sha(pair['backend']) == pair['backend_sha256']
            assert sha(pair['frontend']) == pair['frontend_sha256']
            with lease.block():
                pass
            saved = torch.load(pair['backend'], map_location='cpu', weights_only=False, mmap=True)
            first_entry = data['entries'][selected[0]['identity']]
            with h5py.File(first_entry['path']) as f:
                metadata = json.loads(f.attrs['metadata_json'])
            metadata['joint_limits_rad'] = metadata['joint_limits_rad'][:1]
            controller = GPUBatchedResidualController(saved, metadata, None, 1,
                device=args.device, neural_batch=True, scheduler_index_mode='host', vision_microbatch=2)
            c = controller.lanes[0]
            for case in selected:
                assert time.time() < session['deadline']
                identity = case['identity']
                entry = data['entries'][identity]
                rec = records[identity]
                ep = Episode(root, identity, 'I-HistoryDP')
                assert tuple(c.base.limits.cpu().shape) == tuple(rec['joint_limits'].shape)
                assert torch.equal(c.base.limits.cpu(), rec['joint_limits'].cpu())
                c.base.episode = c.episode = None
                c.base.generator.set_state(noise_state)
                states = []
                values = []
                case_start = time.time()
                with h5py.File(entry['path']) as f:
                    g = f['episodes'][entry['episode']]
                    tr = g['transition']
                    d = tr['dagger']
                    stop = case['stop']
                    packet_inputs = {k:torch.from_numpy(v[:stop]) for k,v in g['obs'].items()}
                    diag = {k:torch.from_numpy(tr['diagnostics_before/'+k][:stop])
                            for k in ('joint_position_rad','joint_velocity_rad_s')}
                    for tick in range(stop):
                        if tick % 12 == 0:
                            with lease.block():
                                pass
                        packet = {k:v[tick:tick+1] for k,v in packet_inputs.items()}
                        packet.update(depth_m=torch.from_numpy(g['frames/depth_m'][tick:tick+1]),
                                      depth_valid=torch.from_numpy(g['frames/depth_valid'][tick:tick+1]))
                        if tick%12 == 0:
                            before = c.base.generator.get_state()
                            probe = torch.Generator(device=args.device)
                            probe.set_state(before)
                            noise = torch.randn((1,24,29), device=args.device, generator=probe)
                            states.append(dict(tick=tick, rng_before_sha256=tensor_sha(before),
                                noise_sha256=tensor_sha(noise), expected_rng_after_sha256=tensor_sha(probe.get_state())))
                        c.base.dino.set(ep.perception['patches'][tick].to(args.device))
                        command = packet['actuator_target_rad'].clone()
                        controller._act_neural_batch(packet,
                            {k:v[tick:tick+1] for k,v in diag.items()}, [0], command)
                        if tick%12 == 0:
                            assert tensor_sha(c.base.generator.get_state()) == states[-1]['expected_rng_after_sha256']
                        # This proves feedback equality on every prefix tick, including masked prefix.
                        assert torch.equal(c.base.last_robot.cpu(), rec['robot'][tick].cpu())
                        if tick in case['ticks']:
                            nominal = c.nominal[0].cpu()
                            residual = c.residual_diagnostics['proposed_rad'][0].cpu()
                            preprojection = c.residual_diagnostics['execution_preprojection_target_rad'][0].cpu()
                            final = c.command[0].cpu()
                            target = rec['expert_command'][tick].cpu()
                            nominal_target = rec['clean_sample'][tick].cpu()
                            row = dict(role=role, policy=tag, identity=identity, new50=case['new50'],
                                native_anchor=case['anchor'], slot=tick-case['anchor'], absolute_tick=tick,
                                first_legal=tick == case['first_legal'],
                                nominal=metrics(nominal, nominal_target),
                                residual=metrics(residual, target-nominal),
                                preprojection=metrics(preprojection, target), final_command=metrics(final, target),
                                projection_magnitude=metrics(final,preprojection),
                                values=dict(nominal=nominal.tolist(), nominal_target=nominal_target.tolist(),
                                    residual=residual.tolist(), preprojection=preprojection.tolist(),
                                    final_command=final.tolist(), expert_target=target.tolist()))
                            values.append(row)
                        # A candidate's own proposal is never substituted for the actual executed history.
                        c.base.command = torch.from_numpy(d['actual_issued'][tick:tick+1]).to(args.device)
                        c.observe_execution({'position_applied_rad':torch.from_numpy(tr['position_applied_rad'][tick:tick+1])})
                assert [r['absolute_tick'] for r in values] == case['ticks']
                item = dict(role=role, policy=tag, identity=identity, frames=stop,
                    initial_rng_sha256=initial_rng_sha, planning_noise=states, rows=values,
                    prefix_robot178_exact=True, wall_s=time.time()-case_start,
                    source_visual_cache_sha256=case['visual_cache_sha256'], pair=pair)
                write(out/f'{role}/{tag}/{identity}.json', item)
                all_rows.extend(values)
                print(json.dumps(dict(role=role, policy=tag, identity=identity, frames=stop,
                    slots=len(values), wall_s=item['wall_s'])), flush=True)
                del ep
            del controller, c, saved
            gc.collect()
            torch.cuda.empty_cache()
    write(out/'rows.json', all_rows)
    write(out/'summary.json', summarize(all_rows, plan, began))
    write(out/'resources.json', dict(memory=lease.memory(), admission_context_s=lease.work_s, timing_scope='Resource-admission contexts only; actual inference time is recorded per case and by the diagnostic owner'))


def summarize(rows, plan, began):
    summaries = []
    for role in sorted({r['role'] for r in rows}):
        for policy in sorted({r['policy'] for r in rows}):
            for population in ('new50',):
                for slots in ('first_legal','remaining_legal','all_entry_commit'):
                    subset = [r for r in rows if r['role']==role and r['policy']==policy
                        and (population=='all150' or r['new50'])
                        and (slots=='all_entry_commit' or (slots=='first_legal' and r['first_legal']) or (slots=='remaining_legal' and not r['first_legal']))]
                    if not subset:
                        continue
                    s = dict(role=role, policy=policy, population=population, slots=slots,
                        episodes=len({r['identity'] for r in subset}), ticks=len(subset))
                    for stage in ('nominal','residual','preprojection','final_command','projection_magnitude'):
                        s[stage] = {k:float(np.mean([r[stage][k] for r in subset]))
                            for k in ('arm_rmse_rad','hand_rmse_rad','equal_mse_rad2','max_abs_rad')}
                    summaries.append(s)
    paired = []
    for role in sorted({r['role'] for r in rows}):
        for control in ('original130','original_R2'):
            lookup = {(r['identity'],r['absolute_tick']):r for r in rows
                      if r['role']==role and r['policy']==control}
            candidate = [r for r in rows if r['role']==role and r['policy']=='entry_repaired']
            for population in ('new50',):
                for slots in ('first_legal','remaining_legal','all_entry_commit'):
                    subset = [r for r in candidate if (r['identity'],r['absolute_tick']) in lookup
                        and (population=='all150' or r['new50'])
                        and (slots=='all_entry_commit' or (slots=='first_legal' and r['first_legal']) or (slots=='remaining_legal' and not r['first_legal']))]
                    if not subset:
                        continue
                    result = dict(role=role, control=control, candidate='entry_repaired',
                        population=population, slots=slots,
                        episodes=len({r['identity'] for r in subset}), ticks=len(subset))
                    for stage in ('nominal','residual','preprojection','final_command'):
                        before = np.asarray([lookup[(r['identity'],r['absolute_tick'])][stage]['equal_mse_rad2'] for r in subset])
                        after = np.asarray([r[stage]['equal_mse_rad2'] for r in subset])
                        result[stage] = dict(before_mean_equal_mse_rad2=float(before.mean()),
                            after_mean_equal_mse_rad2=float(after.mean()),
                            relative_change=float(after.mean()/before.mean()-1) if before.mean() else None,
                            improved_slots=int((after<before).sum()), tied_slots=int((after==before).sum()),
                            worse_slots=int((after>before).sum()))
                    paired.append(result)
    return dict(complete=True, rows=summaries, paired=paired, wall_s=time.time()-began,
        scope=plan['interpretation'], aggregation='Mean of per-slot metrics, equal weight per legal entry slot; first_legal has one slot per episode',
        noise=plan['noise'], target_spaces=plan['target_spaces'], optimizer_updates=0, physics_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','run'])
    parser.add_argument('--roles', default='AB')
    parser.add_argument('--tags', default='original130,original_R2,entry_repaired')
    parser.add_argument('--population', choices=['all150','new50'], default='all150')
    parser.add_argument('--identities', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='entry_same_history_all150_20261122_01')
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args)


if __name__ == '__main__':
    main()
