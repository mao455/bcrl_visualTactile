"""Small-metadata gates for inactive recovery07 final evidence adapters.

Root reviews FINAL_RECOVERY07_BINDINGS.json and pins its SHA plus these source
bytes in ROOT_EVIDENCE_TOOLS_REVIEW07.json. The binding names only real receipts;
it never substitutes success for old failures. Import performs no experiment
writes, process operations, checkpoint opens, CUDA work or sealed reads.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
BINDING_NAME = 'FINAL_RECOVERY07_BINDINGS.json'
REVIEW_NAME = 'ROOT_EVIDENCE_TOOLS_REVIEW07.json'
FINAL_TOOLS = ('final_recovery07_common.py','extract_selected_ledgers_recovery07.py',
    'final_report_assembler_recovery07.py','final_compute_recovery07.py',
    'export_final_evidence_recovery07.py','test_final_evidence_recovery07.py')
OLD_COMPLETED_STAGES = ('validate_collection_recovery05','audit_collection_integrity_recovery05',
    'prepare_learning_recovery05','preflight_learning','register_learning_jobs','cache_own_observations')
STANDALONE_STAGES = ('summarize_development','summarize_test','audit_collection_cuda_noise_recovery05')
REMAINING_STAGES = ('train_backend_recovery07','verify_final_actual_training','prepare_development',
    'evaluate_development','summarize_development','prepare_test','evaluate_test','summarize_test',
    'audit_collection_cuda_noise_recovery05')
AMENDED_PATHS = {'tools/tight_fresh_backend.py','visuotactile/fresh_contracts.py',
                 'visuotactile/anchor_continue.py'}
RUNTIME_SOURCES = ('run_backend_recovery07.py','parallel_rear_recovery07.py','shared_capacity07.py',
    'parallel_rear_proposal.py','parallel_rear_shared06.py','run_main_recovery05.py',
    'shared_capacity06.py','resource_watch_v4.py','presealed_guard.py',
    'audit_collection_recovery05.py','driver_support.py','driver_support_v4.py')


def require(value, message):
    if not value: raise ValueError(message)


def identity(stat):
    return dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''): h.update(block)
    return h.hexdigest()


def stable_small(path, maximum=16*2**20):
    path = Path(path); before = path.stat()
    require(before.st_size <= maximum, 'Oversized small evidence: '+str(path))
    raw = path.read_bytes(); after = path.stat()
    require(identity(before) == identity(after) and len(raw) == before.st_size,
            'Small evidence changed during read: '+str(path))
    return raw


def read_small(path, maximum=16*2**20):
    return json.loads(stable_small(path, maximum))


def pinned_module(filename, expected, name, replacements=()):
    path = HERE/filename; raw = stable_small(path)
    require(hashlib.sha256(raw).hexdigest() == expected, 'Pinned base source differs: '+filename)
    if not replacements:
        spec = importlib.util.spec_from_file_location('_paired07_'+name, path)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        return value
    # Exact fixed path substitutions in a private module, with base bytes pinned.
    # Actual source files and receipts are not modified or redirected on disk.
    import types
    source = raw.decode()
    for old,new in replacements:
        require(source.count(old) == 1, 'Private path substitution must match exactly once')
        source = source.replace(old,new)
    value = types.ModuleType('_paired07_'+name); value.__file__ = str(path)
    exec(compile(source,str(path),'exec'),value.__dict__)
    return value


def _under(path, project):
    path = Path(path).resolve()
    require(path.is_relative_to(project), 'Binding escapes project: '+str(path))
    return path


def reviewed_binding(root, audit):
    root, audit = Path(root).resolve(), Path(audit).resolve(); project = audit.parents[2]
    binding_path, review_path = audit/BINDING_NAME, audit/REVIEW_NAME
    binding, review = read_small(binding_path), read_small(review_path)
    require(review.get('status') in ('cpu_review_passed','cpu_review_passed_not_activated'),
            'Root review07 must pass')
    require(review.get('binding_sha256') == sha(binding_path), 'Root-reviewed final07 binding changed')
    require(binding.get('version') == 'paired_final_recovery07_bindings_v1'
        and Path(binding.get('root','')).resolve() == root,
        'Final07 binding root/version differs')
    reviewed = review.get('source_sha256',{})
    for name in FINAL_TOOLS:
        require(reviewed.get(name) == sha(audit/name), 'Root-reviewed final07 source differs: '+name)
    session = read_small(root/'session.json')
    require(binding.get('original_deadline') == session.get('deadline') == 1795684571.2261844,
            'Original finite deadline changed')
    files = binding.get('small_file_sha256',{})
    require(isinstance(files,dict) and bool(files), 'Real bounded source/receipt bindings required')
    observed = {}
    for name, expected in files.items():
        path = _under(name,project)
        require(path.suffix.lower() in ('.json','.py','.md','.txt'), 'Only small metadata/source binding allowed')
        digest = hashlib.sha256(stable_small(path)).hexdigest()
        require(expected == digest, 'Final07 bound source/receipt changed: '+str(path))
        observed[str(path)] = digest
    return binding,dict(binding_sha256=sha(binding_path),review_sha256=sha(review_path),small_file_sha256=observed)


def _pinned(path, bindings):
    path = Path(path)
    require(str(path.resolve()) in bindings['small_file_sha256'], 'Required ancestry receipt is not pinned: '+str(path))
    return read_small(path)


def runtime_gate(root,audit,binding,bindings):
    """Bind the actual new registry, source chronology and all rear stage exits."""
    reg = _pinned(audit/'BACKEND_RECOVERY07_REGISTRATION.json',bindings)
    production = binding['production']; before,after = production['before_sha256'],production['after_sha256']
    require(reg.get('version') == 'paired_backend_only_recovery07_v1'
        and reg.get('root') == str(root) and reg.get('original_deadline') == binding['original_deadline']
        and reg.get('source_sha256') == after and reg.get('old_production_sha256') == before
        and reg.get('activated') is False and reg.get('owner') == str(root/'owners/backend_recovery07')
        and reg.get('logs') == str(root/'cycle_logs_recovery07')
        and reg.get('watcher') == str(audit/'watch_shared_capacity07'), 'Actual backend07 registry differs')
    job_path = audit/'proposed_backend_recovery07.json'
    require(reg.get('job') == str(job_path) and reg.get('job_sha256') == sha(job_path),
            'Actual backend07 job SHA/path differs')
    old = _pinned(audit/'driver_main_recovery05/registration.json',bindings)
    require(old.get('source_sha256') == before and old.get('deadline') == binding['original_deadline']
        and reg.get('prior_registration_sha256') == sha(audit/'driver_main_recovery05/registration.json'),
        'Original completed DP source chronology differs')
    activated = _pinned(audit/'driver_backend_recovery07/registration.json',bindings)
    require(activated == dict(reg,activated=True,activated_utc=activated.get('activated_utc'),
        registry_sha256=sha(audit/'BACKEND_RECOVERY07_REGISTRATION.json')),
        'Activated07 runtime differs from its frozen registry')
    closure = _pinned(audit/'driver_backend_recovery07/closure.json',bindings)
    driver_exit = _pinned(audit/'driver_backend_recovery07/driver_exit.json',bindings)
    require(closure.get('stage') == 'main_stages_complete'
        and closure.get('status') == 'backend_and_remaining_main_stages_complete'
        and driver_exit.get('returned_after_owned_helpers') is True
        and driver_exit.get('utc',-1) >= closure.get('utc',0), 'Actual07 driver/helper closure differs')
    expected_sources = {str(audit/name) for name in RUNTIME_SOURCES}
    require(set(reg.get('orchestration_sha256',{})) == expected_sources,
            'Exact12 runtime source population required')
    for path,digest in reg['orchestration_sha256'].items():
        require(bindings['small_file_sha256'].get(str(Path(path).resolve())) == digest,
                'Actual07 orchestration source is not pinned: '+path)
    # Registry also pins the original short failure log and exact source patch.
    # They are checked here but remain local under the bounded exporter policy.
    for path,digest in reg.get('old_terminal_sha256',{}).items():
        path = _under(path,audit.parents[2])
        require(path.suffix.lower() in ('.json','.log','.patch')
            and hashlib.sha256(stable_small(path)).hexdigest() == digest,
            'Original failed06 terminal/patch bytes changed: '+str(path))
    require(bool(reg.get('old_terminal_sha256')), 'Old terminal ancestry cannot be empty')
    changes = {path:dict(old_sha256=before[path],new_sha256=after[path]) for path in AMENDED_PATHS}
    amendment = reg.get('source_amendment',{})
    require(amendment.get('changes') == changes and amendment.get('changed_paths') == sorted(AMENDED_PATHS),
            'Exact3 reviewed source amendments required')
    receipt = _pinned(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',bindings)
    require(amendment.get('receipt_sha256') == sha(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json')
        and receipt.get('changed_paths') == changes and receipt.get('optimizer_updates') == 0
        and receipt.get('physical_steps') == 0 and receipt.get('affine_or_label_changes') is False,
        'Amendment receipt/source scope differs')
    require(_pinned(audit/'driver_backend_recovery07/backend_source_identity.json',bindings).get('sources') == after,
            'New backend source identity differs')
    stages = reg.get('stages',[])
    require([s.get('name') for s in stages] == list(REMAINING_STAGES)
        and all(isinstance(s.get('command'),list) and bool(s['command']) for s in stages)
        and {s['name'] for s in stages if s.get('gpu') is True} == {
            'train_backend_recovery07','evaluate_development','summarize_development',
            'evaluate_test','summarize_test','audit_collection_cuda_noise_recovery05'},
        'Exact missing-only07 ordered stage registry required')
    guard = next(s for s in stages if s['name'] == 'verify_final_actual_training')
    require(guard['command'][-1] == str(audit/'presealed_guard_main_recovery07.json'),
            'Original absent presealed output must remain unused')
    invariants = dict(dp_new_updates=0,dp_reruns=0,collection_retries=0,new_opportunities=0,
        optimizer_epochs=60,deploy_epoch=60,rear_updates_per_role=38400,caches_own_dp=True,
        encoder_frozen=True,development_descriptive_only=True,sealed_test_families=64,
        external_presence_is_veto=False)
    require(all(reg.get('scientific_invariants',{}).get(k) == v for k,v in invariants.items()),
            'Actual07 scientific budget/inference invariants differ')
    runtime = audit/'parallel_rear_recovery07_runtime'; last_end = None
    for phase,population in [('serial_caches',{'A':{0,1},'B':{0,1}}),
                             ('rear_training',{'A':{0},'B':{0}}),('serial_offline',{'A':{0},'B':{0}})]:
        events = _pinned(runtime/phase/'pipeline_events.json',bindings)
        starts = [r for r in events if 'pid' in r]; exits = [r for r in events if 'returncode' in r]
        expected = {(role,stage) for role,stages0 in population.items() for stage in stages0}
        require(len(events) == 2*len(expected) and len(starts) == len(exits) == len(expected)
            and {(r.get('role'),r.get('stage')) for r in starts} == expected
            and {(r.get('role'),r.get('stage')) for r in exits} == expected
            and all(r.get('returncode') == 0 for r in exits), 'Rear07 stage population/exit differs: '+phase)
        for row in starts:
            end = next(e for e in exits if (e['role'],e['stage']) == (row['role'],row['stage']))
            require(isinstance(row.get('command'),list) and end['utc'] >= row['utc'],
                    'Rear07 stage ordering differs')
        require(last_end is None or min(r['utc'] for r in starts) >= last_end,
                'Own caches, rear training and offline phases overlap out of order')
        last_end = max(r['utc'] for r in exits)
    admission = _pinned(runtime/'admission.json',bindings); complete = _pinned(runtime/'complete.json',bindings)
    require((admission.get('pair',{}).get('allowed') is True or admission.get('single',{}).get('allowed') is True)
        and complete.get('parallel') is admission.get('parallel')
        and complete.get('caches_serial') is True and complete.get('offline_serial') is True
        and complete.get('utc',-1) >= last_end, 'Actual07 capacity/pipeline completion differs')
    for stage in stages:
        if stage['gpu']:
            row = _pinned(audit/'driver_backend_recovery07'/(stage['name']+'_shared_capacity06_admission.json'),bindings)
            require(row.get('capacity',{}).get('ready') is True, 'Actual07 capacity admission missing')
    return reg


def closure_gate(root, audit):
    """No checkpoint payloads, model imports or sealed outputs before real closure."""
    root, audit = Path(root).resolve(), Path(audit).resolve()
    driver = audit/'driver_backend_recovery07'
    required = [audit/BINDING_NAME,audit/REVIEW_NAME,driver/'status.json',
        root/'owners/backend_recovery07/project_complete.json',
        root/'owners/backend_recovery07/project_exit.json',
        root/'evaluation/test/registration.json',root/'cycle_logs_recovery07/summarize_test_exit.json']
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            required.append(root/scope/role/'state_data/data.json')
            for sub in ('training',f'backend/{role}/training'):
                required.extend(root/scope/role/sub/name for name in ('best.pt','deployment.json','finished.json'))
    required.extend(root/'evaluation'/cohort/'project_exit.json' for cohort in ('development','test'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing: return dict(status='pending',missing=missing,checkpoint_reads=0)
    binding, bindings = reviewed_binding(root,audit)
    runtime_registration = runtime_gate(root,audit,binding,bindings)
    status = read_small(driver/'status.json')
    require(status.get('stage') == 'main_stages_complete', 'Real recovery07 driver closure required')
    require(isinstance(status.get('utc'),(int,float)) and not isinstance(status['utc'],bool)
        and math.isfinite(status['utc']) and status['utc'] <= binding['original_deadline'],
        'Real07 driver must close under the original finite deadline')
    require(not (driver/'failure.json').exists(), 'Failed recovery07 cannot close final evidence')
    for name in ('BACKEND_RECOVERY07_REGISTRATION.json','proposed_backend_recovery07.json',
                 'run_backend_recovery07.py','parallel_rear_recovery07.py'):
        require(str((audit/name).resolve()) in bindings['small_file_sha256'],
                'Actual07 runtime registration/source/job is not pinned: '+name)
    # Old failed lifecycle is required, not ignored or rewritten into success.
    for directory in (audit/'driver_main_recovery05',audit/'wait_main_shared06'):
        require(_pinned(directory/'status.json',bindings).get('stage') == 'stopped_with_evidence',
                'Truthful stopped06 status must remain')
        require(bool(_pinned(directory/'failure.json',bindings).get('error')), 'Retained06 failure required')
    old_exit = _pinned(audit/'wait_main_shared06/reviewed_main_exit.json',bindings)
    require(old_exit.get('returncode') == 1 and old_exit.get('owned_helper_returned_after_cleanup') is True,
            'Old failed helper exit/cleanup differs')
    require(bool(_pinned(root/'owners/main_train/project_failure.json',bindings).get('error'))
        and _pinned(root/'owners/main_train/project_exit.json',bindings).get('children_reaped') is True
        and _pinned(root/'cycle_logs/train_four_components_parallel_rears_exit.json',bindings).get('returncode') == 1,
        'Old failed training owner/exit must remain')
    for stage in OLD_COMPLETED_STAGES:
        require(_pinned(root/'cycle_logs'/(stage+'_exit.json'),bindings).get('returncode') == 0,
                'Previously completed real stage differs: '+stage)
    expected_exits = binding.get('stage_exit_paths',{})
    require(set(expected_exits) == set(REMAINING_STAGES), 'Exact missing-stage07 population required')
    for stage,path in expected_exits.items():
        expected = root/'cycle_logs_recovery07'/(stage+'_exit.json')
        require(Path(path).resolve() == expected, 'Wrong07 stage exit path: '+stage)
        require(_pinned(expected,bindings).get('returncode') == 0, 'Missing-stage07 did not close: '+stage)
    for owner in (root/'owners/backend_recovery07',root/'evaluation/development',root/'evaluation/test'):
        require(not (owner/'project_failure.json').exists(), 'Failed current07 owner: '+str(owner))
        started, complete, exited = [read_small(owner/name) for name in (
            'project_job_started.json','project_complete.json','project_exit.json')]
        stages = started.get('job',{}).get('stages'); commands = started.get('commands')
        require(isinstance(stages,list) and bool(stages) and isinstance(commands,list)
            and complete.get('stages') == len(stages) == len(commands)
            and Path(started['job'].get('resource_root','')).resolve() == owner
            and exited.get('children_reaped') is True,
            'Real current07 owner stages/cleanup differ: '+str(owner))
        if owner == root/'owners/backend_recovery07':
            require(started['job'] == _pinned(audit/'proposed_backend_recovery07.json',bindings),
                    'Actual backend07 owner job differs from reviewed missing-stage job')
            old_end = _pinned(root/'owners/main_train/project_exit.json',bindings)['utc']
            require(started.get('utc',-1) >= old_end,
                    'New backend owner overlaps unreaped failed06 ancestry')
    preserved = binding.get('preserved_dp',{})
    require(set(preserved) == {'A','B'}, 'Both actual DP60 deployments must be preserved')
    for role,row in preserved.items():
        directory = root/'learning'/role/'training'
        deployment = _pinned(directory/'deployment.json',bindings)
        finished = _pinned(directory/'finished.json',bindings)
        require(deployment.get('epoch') == finished.get('epochs') == 60
            and deployment.get('update') == finished.get('updates') == 10200
            and finished.get('complete') is True
            and deployment.get('checkpoint_sha256') == row.get('checkpoint_sha256'),
            'Actual preserved DP age/SHA differs: '+role)
        require(identity((directory/'best.pt').stat()) == row.get('checkpoint_stat_identity'),
                'Preserved DP metadata changed after actual SHA binding: '+role)
    production = binding.get('production',{})
    before, after = production.get('before_sha256',{}),production.get('after_sha256',{})
    changed = {path for path in before if before[path] != after.get(path)}
    require(len(before) == 295 and set(before) == set(after) and len(changed) == 3
        and changed == set(production.get('changed_paths',[])) == AMENDED_PATHS, 'Exact295-source/3-file amendment required')
    for relative,digest in after.items():
        path = _under(root.parents[3]/relative,root.parents[3])
        require(bindings['small_file_sha256'].get(str(path)) == digest,
                'Current amended production source is not SHA-verified: '+relative)
    snapshots = production.get('snapshot_paths',{})
    require(set(snapshots) == changed, 'Three original changed-source snapshots required')
    for relative,path in snapshots.items():
        require(bindings['small_file_sha256'].get(str(Path(path).resolve())) == before[relative],
                'Original changed-source snapshot not preserved: '+relative)
    require(read_small(root/'learning/training_source_identity.json').get('sources') == before,
            'Completed DP source identity was not preserved')
    for role in 'AB':
        require(runtime_registration['preserved_dp'][role]['checkpoint_sha256'] == preserved[role]['checkpoint_sha256'],
                'Reviewed preservation differs from actual07 DP deployment')
    final = read_small(root/'evaluation/test/registration.json')
    require(final.get('cohort') == 'test' and final.get('n') == 64
        and set(final.get('pairs',{})) == {'A_new','B_new','A_parent','B_parent'}
        and final.get('source_sha256') == after,
        'Sealed64 complete-policy/source freeze differs')
    return dict(status='closed',version='paired_final_recovery07_lifecycle_v1',checkpoint_reads=0,
        failed06_preserved=True,preserved_dp_retrained=False,original_deadline=binding['original_deadline'],
        root_review=bindings,production_amendment_paths=sorted(changed),
        scope='Small-source/receipt SHA bindings plus checkpoint metadata stability; actual checkpoint content is verified only by the separate selected-ledger SHA/mmap pass.')


def watcher_closed(audit):
    audit = Path(audit).resolve(); binding = read_small(audit/BINDING_NAME)
    path = _under(binding['watcher_exit_path'],audit.parents[2])
    require(path.is_relative_to(audit), 'Watcher07 exit must be under AUDIT')
    exited = read_small(path)
    require(exited.get('modified_schedules') == 0 and exited.get('conflicts') == [],
            'Actual watcher07 must restore all schedules and close')
    require(exited.get('exact_original_byte_restoration') is True,
            'Actual watcher07 must certify exact original schedule byte restoration')
    old = read_small(audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json')
    require(old.get('source_pid_reaped') is True, 'Old watcher06 must remain reaped')
    return dict(path=str(path),sha256=sha(path),receipt=exited,
        previous_closed06_sha256=sha(audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json'))


def build_binding(root=ROOT,audit=HERE):
    """Construct a root-review candidate from real closed receipts, never weights.

    This returns a pending list without writing if any receipt is absent. Root
    writes the binding only after closure and separately reviews its SHA/source
    bytes; this function cannot manufacture a root approval.
    """
    root,audit = Path(root).resolve(),Path(audit).resolve();project=audit.parents[2]
    reg=read_small(audit/'BACKEND_RECOVERY07_REGISTRATION.json')
    before,after=reg['old_production_sha256'],reg['source_sha256']
    amendment=read_small(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json')
    # The amendment's snapshot stores AFTER bytes. Original completed DP source
    # identity lives in frozen_source; both directions remain explicit.
    snapshots={relative:str(audit/'frozen_source'/relative) for relative in reg['source_amendment']['changed_paths']}
    amended_snapshots={relative:str(Path(amendment['snapshot'])/relative) for relative in reg['source_amendment']['changed_paths']}
    for relative,path in snapshots.items():
        require(hashlib.sha256(stable_small(path)).hexdigest() == before[relative],
                'Actual BEFORE source snapshot SHA differs: '+relative)
    for relative,path in amended_snapshots.items():
        require(hashlib.sha256(stable_small(path)).hexdigest() == after[relative],
                'Actual AFTER amendment snapshot SHA differs: '+relative)
    stages={stage:str(root/'cycle_logs_recovery07'/(stage+'_exit.json')) for stage in REMAINING_STAGES}
    watcher=audit/'watch_shared_capacity07/resource_watch_v4_exit.json'
    paths={audit/'BACKEND_RECOVERY07_REGISTRATION.json',audit/'proposed_backend_recovery07.json',
        audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json',
        audit/'BACKEND_RECOVERY07_ACTIVATION_RECONCILIATION.json',
        audit/'driver_backend_recovery07/registration.json',audit/'driver_backend_recovery07/status.json',
        audit/'driver_backend_recovery07/closure.json',audit/'driver_backend_recovery07/driver_exit.json',
        audit/'driver_backend_recovery07/backend_source_identity.json',
        audit/'driver_main_recovery05/registration.json',audit/'driver_main_recovery05/status.json',
        audit/'driver_main_recovery05/failure.json',audit/'wait_main_shared06/status.json',
        audit/'wait_main_shared06/failure.json',audit/'wait_main_shared06/reviewed_main_exit.json',
        root/'owners/main_train/project_failure.json',root/'owners/main_train/project_exit.json',
        root/'cycle_logs/train_four_components_parallel_rears_exit.json',root/'learning/training_source_identity.json',watcher}
    paths.update(Path(p) for p in reg['orchestration_sha256'])
    paths.update(Path(p) for p in reg['old_terminal_sha256'] if Path(p).suffix.lower() in ('.json','.py','.md','.txt'))
    paths.update(project/p for p in after);paths.update(Path(p) for p in snapshots.values())
    paths.update(Path(p) for p in amended_snapshots.values())
    paths.update(Path(p) for p in stages.values())
    paths.update(root/'cycle_logs'/(s+'_exit.json') for s in OLD_COMPLETED_STAGES)
    runtime=audit/'parallel_rear_recovery07_runtime'
    paths.update(runtime/phase/'pipeline_events.json' for phase in ('serial_caches','rear_training','serial_offline'))
    paths.update((runtime/'admission.json',runtime/'complete.json'))
    paths.update(audit/'driver_backend_recovery07'/(s['name']+'_shared_capacity06_admission.json') for s in reg['stages'] if s['gpu'])
    for owner in (root/'owners/backend_recovery07',root/'evaluation/development',root/'evaluation/test'):
        paths.update(owner/name for name in ('project_job_started.json','project_complete.json','project_exit.json'))
    preserved={}
    for role in 'AB':
        train=root/'learning'/role/'training'
        paths.update(train/name for name in ('deployment.json','finished.json'))
        checkpoint=train/'best.pt'
        if checkpoint.is_file():
            preserved[role]=dict(checkpoint_sha256=reg['preserved_dp'][role]['checkpoint_sha256'],
                checkpoint_stat_identity=identity(checkpoint.stat()),
                content_identity_source='Actual SHA-verified recovery07 preserved_dp preflight; stat is supplemental only')
        else:paths.add(checkpoint)
    missing=sorted(str(path) for path in paths if not path.is_file())
    if missing:return dict(status='pending',missing=missing,checkpoint_reads=0)
    status=read_small(audit/'driver_backend_recovery07/status.json')
    require(status.get('stage')=='main_stages_complete','Real07 closure is required before root binding')
    return dict(version='paired_final_recovery07_bindings_v1',root=str(root),
        original_deadline=reg['original_deadline'],small_file_sha256={str(p.resolve()):hashlib.sha256(stable_small(p)).hexdigest() for p in sorted(paths)},
        preserved_dp=preserved,production=dict(before_sha256=before,after_sha256=after,
            changed_paths=reg['source_amendment']['changed_paths'],snapshot_paths=snapshots,
            after_snapshot_paths=amended_snapshots),
        stage_exit_paths=stages,watcher_exit_path=str(watcher))


def extracted_component(root,audit,directory,scope,role,component):
    """Reuse one real successful SHA/mmap pass, with stable immutable metadata.

    This does not claim checkpoint content is established from stat alone. The
    independent extractor performed the actual SHA/mmap/count validation; these
    checks bind that pass and reject subsequent file identity changes.
    """
    root,audit,directory = (Path(x).resolve() for x in (root,audit,directory))
    require(directory.is_relative_to(audit) and directory != audit, 'Selected ledger output outside AUDIT')
    status = read_small(directory/'extraction_status.json')
    summary = read_small(directory/'selected_ledger_summary.json')
    for record in (status,summary):
        require(record.get('status') == 'passed' and record.get('checkpoint_reads') == 8
            and record.get('cuda_initialized') is False and record.get('all_selected_epoch') == 60
            and all(record.get(key) is True for key in ('all_legal_ticks_at_least50',
                'all_native_entry_slots_at_least50','paired_actual_updates_equal','own_frontend_rear_mask_and_base_binding')),
            'Successful07 eight-selected-ledger pass required')
    require(status.get('extractor_sha256') == sha(audit/'extract_selected_ledgers_recovery07.py')
        and Path(status.get('root','')).resolve() == root
        and Path(status.get('output','')).resolve() == directory
        and status.get('raw_data_opened') is False and status.get('model_forward_calls') == 0
        and status.get('files') == summary.get('files'), 'Selected07 extraction binding differs')
    expected = {f'{s}_{r}_{c}.json' for s in ('common180_parents','learning') for r in 'AB'
                for c in ('frontend','backend')}
    require(set(summary.get('files',{})) == expected, 'Exactly eight selected ledger components required')
    filename = f'{scope}_{role}_{component}.json'; item = summary['files'][filename]
    raw = stable_small(directory/filename)
    require(item.get('bytes') == len(raw) and item.get('sha256') == hashlib.sha256(raw).hexdigest(),
            'Selected ledger component bytes changed')
    row = json.loads(raw)
    child = root/scope/role/('training' if component == 'frontend' else f'backend/{role}/training')
    require((row.get('scope'),row.get('role'),row.get('component')) == (scope,role,component)
        and Path(row.get('checkpoint','')).resolve() == child/'best.pt'
        and row.get('epoch') == 60 and type(row.get('updates')) is int and row['updates'] > 0,
        'Selected actual component identity differs')
    require(identity((child/'best.pt').stat()) == row.get('checkpoint_stat_identity'),
            'Actual SHA-verified checkpoint metadata changed before ledger reuse')
    deployment = read_small(child/'deployment.json'); finished = read_small(child/'finished.json')
    require(sha(child/'deployment.json') == row.get('deployment_sha256')
        and deployment.get('checkpoint_sha256') == row.get('checkpoint_sha256')
        and deployment.get('epoch') == row['epoch'] and deployment.get('update') == row['updates']
        and finished.get('updates') == row['updates'], 'Selected metadata age/SHA differs')
    data_path = root/scope/role/'state_data/data.json'
    require(sha(data_path) == row.get('data_sha256'), 'Selected actual data binding changed')
    require(row.get('ancestors_counted') is False and row.get('duplicate_within_step') == 'count_once',
            'Actual successful-update counting policy differs')
    counts = []; episodes = row.get('episodes',{}); masks = {}
    require(isinstance(episodes,dict) and len(episodes) == row.get('trajectories') and bool(episodes),
            'Actual selected episode counter population missing')
    for name,episode in episodes.items():
        ticks,values,length = episode['legal_ticks'],episode['successful_optimizer_steps'],episode['complete_history_ticks']
        require(type(length) is int and 0 < length <= 20000 and ticks == sorted(set(ticks))
            and len(ticks) == len(values) > 0 and all(type(t) is int and 0 <= t < length for t in ticks)
            and all(type(n) is int and 50 <= n <= row['updates'] for n in values)
            and type(episode.get('visits')) is int and episode['visits'] >= 50,
            'Actual successful-step count or legal population differs')
        legal = set(ticks)
        require(hashlib.sha256(bytes(i in legal for i in range(length))).hexdigest() == episode['mask_sha256'],
                'Selected actual legal-mask identity differs')
        counts.extend(values); masks[name] = dict(zip(ticks,values))
    require(len(counts) == row.get('legal_ticks') and min(counts) == row.get('minimum_tick_steps')
        and max(counts) == row.get('maximum_tick_steps') and row.get('below50_ticks') == 0
        and min(e['visits'] for e in episodes.values()) == row.get('minimum_visits'),
        'Selected actual counter summary differs')
    if component == 'frontend':
        entries = row.get('native_entry_rows'); require(isinstance(entries,list), 'Native correction entries missing')
        keys = set()
        for entry in entries:
            name,tick = entry['identity'],entry['absolute_tick']; key = (name,tick)
            require(key not in keys and tick in masks[name], 'Duplicate or illegal native-entry row')
            keys.add(key); anchor = min(masks[name])//12*12
            require(entry['native_anchor'] == anchor and entry['slot'] == tick-anchor
                and 0 <= entry['slot'] < 12 and type(entry['successful_optimizer_steps']) is int
                and 50 <= entry['successful_optimizer_steps'] <= masks[name][tick],
                'Actual native-entry phase/count differs')
        expected_keys = {(name,tick) for name,values in masks.items() if 0 not in values
            for tick in values if tick//12 == min(values)//12}
        require(keys == expected_keys, 'Native correction entry population incomplete')
    return row
