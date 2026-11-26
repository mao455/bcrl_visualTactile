"""Inactive admission14 metadata adapter; original10 and12 remain truthful failures.

No weights, sealed outcomes, CUDA, runtime schedules or experiment writes are
opened on import. Root alone reviews actual post-closure bindings. The existing
presealed main guard opens 8 selected payloads before evaluation; the later final
extractor has its own 8-payload detailed-ledger pass. Assembler/exporter reuse it.
"""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ROOT=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
BINDING_NAME='FINAL_RECOVERY14_BINDINGS.json'
REVIEW_NAME='ROOT_EVIDENCE_TOOLS_REVIEW14.json'
FINAL_TOOLS=('final_recovery14_common.py','extract_selected_ledgers_recovery14.py',
 'final_report_assembler_recovery14.py','final_compute_recovery14.py',
 'export_final_evidence_recovery14.py','test_final_evidence_recovery14.py')
OLD_COMPLETED_STAGES=('validate_collection_recovery05','audit_collection_integrity_recovery05',
 'prepare_learning_recovery05','preflight_learning','register_learning_jobs','cache_own_observations')
STANDALONE_STAGES=('summarize_development','summarize_test','audit_collection_cuda_noise_recovery05')
REMAINING_STAGES=('summarize_development','prepare_test','evaluate_test','summarize_test','audit_collection_cuda_noise_recovery05')
COMPLETED10_STAGES=('train_backend_recovery10','verify_final_actual_training','prepare_development')
RUNTIME12_SOURCES=('run_eval_recovery12.py','eval_startup12.py','shared_capacity12.py',
 'run_backend_recovery10.py','run_main_recovery05.py','shared_capacity10.py','shared_capacity06.py',
 'resource_watch_v4.py','coexistence_watch_recovery05.py','driver_support.py','driver_support_v4.py')
BACKEND10_STAGES=('train_backend_recovery10','verify_final_actual_training','prepare_development',
 'evaluate_development','summarize_development','prepare_test','evaluate_test','summarize_test',
 'audit_collection_cuda_noise_recovery05')
AMENDED_PATHS={'tools/tight_fresh_backend.py','visuotactile/fresh_contracts.py','visuotactile/anchor_continue.py'}
RUNTIME_SOURCES=('run_backend_recovery10.py','parallel_rear_recovery10.py','shared_capacity10.py',
 'scoped_resource_wait10.py','run_backend_recovery09.py','run_backend_recovery08.py','run_backend_recovery07.py','parallel_rear_proposal.py',
 'parallel_rear_shared06.py','run_main_recovery05.py','shared_capacity06.py',
 'resource_watch_v4.py','presealed_guard.py','audit_collection_recovery05.py',
 'driver_support.py','driver_support_v4.py')
SAFE_SUFFIXES={'.json','.py','.md','.txt'}
# Exact inherited scientific engines and source-only fixture dependencies.
SOURCE_DEPENDENCIES_SHA256={'export_final_evidence_recovery05.py': '56a8212ac5494a8ccda908fc390e926508d035694fb3a85a9ba0c7c410e89d6b',
 'export_final_evidence_recovery07.py': '7f66438d2c8076d8858847a3fc8028c3ea19d975c5eed6f518a542e81e53b8e3',
 'export_final_evidence_recovery10.py': '4d3a8e326c39c0624f6b0fe98843bbb660f251d9571936871ee30c8ee51d5c2d',
 'extract_selected_ledgers_recovery05.py': '197eb8881358740ecba907042d4dc65ed93b268a70c96d32eb0bc4132b58f678',
 'extract_selected_ledgers_recovery07.py': '1247aa9b70541c4ffad9c5b3428e810cf4cdb56c34f4764fdf9a8bfd9e82df49',
 'extract_selected_ledgers_recovery10.py': '43044874bc51ba6e6e9351071d6a56c20f6b6f34552e0a7a16674f33a71c8021',
 'final_compute_recovery07.py': '8b17ed0c3a07aab2a1edb24cde67418ed30791b25eb94d6e7fa92871d4312996',
 'final_compute_recovery10.py': 'f801d5fa9e7ac1373eb92e5057a96b0f4243339d8bc384786d4bbbd999bd946c',
 'final_compute_shared06.py': '9bd109d3895856ab58e9aa3c67d952985b546b5701e5f7df0e01d82c152e1aae',
 'final_recovery07_common.py': 'a2f05ad9745111d17170aff69ddc4cf4fed8421b1bc41be4009c9b8cd46af23b',
 'final_recovery10_common.py': 'e6e92002f45583e5a21e5f7487be94577cdb22d98b86ea3cab3f56cb163d7227',
 'final_report_assembler.py': '1eca4843d9e7e2d1ac6287b220bf8d429fe36458cb58ddcf63a63e9ab2611b2f',
 'final_report_assembler_recovery07.py': '465d65cd3f33ad1111b197d034fc0bfa6928a88475ea44d25276adb8e18c9c02',
 'final_report_assembler_recovery10.py': '963fff2f67a7ec4094b3811d852c4d6cd2d54c0900cf81c949dff5772ccba61a',
 'test_extract_selected_ledgers_recovery05.py': '3a2118a9c4134736ce6e2d716a7530cf077ef7dc02cdbc66417e6ae52f2405e5',
 'test_final_evidence_recovery07.py': '7c7fc6d11fe394deab1fe8d8bc5a16cad8e2df332456f4d20210b8cb5810caba',
 'test_final_evidence_recovery10.py': 'bbbeeac5dd2006b15d53440567b5ee0a734e5780dbc5232a372b609bbc924dc2'}


# Exact inherited12 adapters and synthetic fixture sources required to reproduce final13 tests.
SOURCE_DEPENDENCIES_SHA256.update({'final_recovery12_common.py': '7d2efbda7928c4fdfcad4b7d76bc7307e37faab3c7139eb5deb8c396d12738b1', 'extract_selected_ledgers_recovery12.py': 'b43f8539903384a991563c8cc5268839ed51ff3598390c1a3d1723980b30dcea', 'final_compute_recovery12.py': 'ec95a90725218f997c94f24d54b4e99c5cd4c7d110fd5f07f32d7f6a639acf11', 'final_report_assembler_recovery12.py': '85810a0f969ec9986ce40667500c3c1cc6a57d860e64330c921c8bda69591ed7', 'export_final_evidence_recovery12.py': '61b71131ad4b92982f6be93e973bd73e378bf7ee593f5b769171509210cc9bea', 'test_final_evidence_recovery12.py': '9545e553af5556b4f7f8f128057ce1c54013700371449eab484783573754c6f1', 'test_test_fragment13.py': 'd3c0f7d4eae7dba9607030182d247f5852dbb28e3b31b1bd9504ee9de8395e3c'})

# Frozen13 evidence sources remain immutable and independently reproducible.
SOURCE_DEPENDENCIES_SHA256.update({'final_recovery13_common.py': 'a4a1916fa8f0d7055459e6dcc76d542e02f4dceb9f7eb10f0623855a2c1a1e9c', 'extract_selected_ledgers_recovery13.py': 'b4a45bd1b9cdf8d8246bc31631636a88a0ed28e4fe4ac2dc322d2b6b82cc1cc9', 'final_compute_recovery13.py': '904f5ae5002e843096e43fa8d5aa979ab4a2c4eddab19b02437113c773849cbc', 'final_report_assembler_recovery13.py': 'b5fb2ffcc9e87cbfc49db16c145215a372c3e072c4002f70bed5795e172107c7', 'export_final_evidence_recovery13.py': '41d0f7b12252c5561ccb71b7b7e202573d5a19e712d2b15f6bc5ea7efb3fcf6b', 'test_final_evidence_recovery13.py': '2be4e460e6cdbfecb5f939b46eaf8262934c0ad58b3a6615d365d3ec9fe1884a'})

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
        spec = importlib.util.spec_from_file_location('_paired10_'+name, path)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        return value
    # Exact fixed path substitutions in a private module, with base bytes pinned.
    # Actual source files and receipts are not modified or redirected on disk.
    import types
    source = raw.decode()
    for old,new in replacements:
        require(source.count(old) == 1, 'Private path substitution must match exactly once')
        source = source.replace(old,new)
    value = types.ModuleType('_paired10_'+name); value.__file__ = str(path)
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
            'Root review14 must pass')
    require(review.get('binding_sha256') == sha(binding_path), 'Root-reviewed final14 binding changed')
    require(binding.get('version') == 'paired_final_recovery14_bindings_v1'
        and Path(binding.get('root','')).resolve() == root,
        'Final14 binding root/version differs')
    reviewed = review.get('source_sha256',{})
    for name in FINAL_TOOLS:
        require(reviewed.get(name) == sha(audit/name), 'Root-reviewed final14 source differs: '+name)
    session = read_small(root/'session.json')
    require(binding.get('original_deadline') == session.get('deadline') == 1795684571.2261844,
            'Original finite deadline changed')
    files = binding.get('small_file_sha256',{})
    require(isinstance(files,dict) and bool(files), 'Real bounded source/receipt bindings required')
    for name,expected in SOURCE_DEPENDENCIES_SHA256.items():
        require(files.get(str((audit/name).resolve()))==expected,'Inherited source/fixture dependency not pinned: '+name)
    observed = {}
    for name, expected in files.items():
        path = _under(name,project)
        require(path.suffix.lower() in ('.json','.py','.md','.txt'), 'Only small metadata/source binding allowed')
        digest = hashlib.sha256(stable_small(path)).hexdigest()
        require(expected == digest, 'Final14 bound source/receipt changed: '+str(path))
        observed[str(path)] = digest
    logs=binding.get('local_retained_log_sha256',{})
    require(set(logs)=={str(root/'cycle_logs_fragment13/evaluate_fragment13.log')},'Exact retained failed13 helper log identity required')
    for path,digest in logs.items():
        target=_under(path,project)
        require(not target.is_symlink() and hashlib.sha256(stable_small(target)).hexdigest()==digest,
            'Retained local failed13 helper log changed')
    return binding,dict(binding_sha256=sha(binding_path),review_sha256=sha(review_path),small_file_sha256=observed,
        local_retained_log_sha256=logs)


def _pinned(path, bindings):
    path = Path(path)
    require(str(path.resolve()) in bindings['small_file_sha256'], 'Required ancestry receipt is not pinned: '+str(path))
    return read_small(path)




def registry_gate(root,audit,binding,bindings):
    """Validate actual10 identity and retain failed06/07/08/09 ancestry."""
    reg=_pinned(audit/'BACKEND_RECOVERY10_REGISTRATION.json',bindings)
    prior=_pinned(audit/'BACKEND_RECOVERY09_REGISTRATION.json',bindings)
    prior08=_pinned(audit/'BACKEND_RECOVERY08_REGISTRATION.json',bindings)
    old=_pinned(audit/'BACKEND_RECOVERY07_REGISTRATION.json',bindings)
    before,after=binding['production']['before_sha256'],binding['production']['after_sha256']
    require(reg.get('version')=='paired_backend_missing_cache_recovery10_v1'
        and reg.get('activated') is False and reg.get('root')==str(root)
        and reg.get('original_deadline')==binding['original_deadline']
        and reg.get('owner')==str(root/'owners/backend_recovery10')
        and reg.get('logs')==str(root/'cycle_logs_recovery10')
        and reg.get('watcher')==str(audit/'watch_shared_capacity10')
        and reg.get('source_sha256')==old.get('source_sha256')==after
        and reg.get('old_production_sha256')==old.get('old_production_sha256')==before
        and reg.get('prior_registration_sha256')==sha(audit/'BACKEND_RECOVERY09_REGISTRATION.json')
        and prior.get('prior_registration_sha256')==sha(audit/'BACKEND_RECOVERY08_REGISTRATION.json')
        and prior08.get('prior_registration_sha256')==sha(audit/'BACKEND_RECOVERY07_REGISTRATION.json'),
        'Actual10 registry/09/08/07 ancestry identity differs')
    job=audit/'proposed_backend_recovery10.json'
    require(reg.get('job')==str(job) and reg.get('job_sha256')==sha(job),'Actual10 job differs')
    activated=_pinned(audit/'driver_backend_recovery10/registration.json',bindings)
    require(activated==dict(reg,activated=True,activated_utc=activated.get('activated_utc'),
        registry_sha256=sha(audit/'BACKEND_RECOVERY10_REGISTRATION.json')),'Activated10 registry differs')
    original=_pinned(audit/'driver_main_recovery05/registration.json',bindings)
    require(original.get('source_sha256')==before and original.get('deadline')==binding['original_deadline']
        and old.get('prior_registration_sha256')==sha(audit/'driver_main_recovery05/registration.json'),
        'Original DP source lineage differs')
    require(set(reg.get('orchestration_sha256',{}))=={str(audit/name) for name in RUNTIME_SOURCES},
        'Exact16 orchestration source population required')
    for path,digest in reg['orchestration_sha256'].items():
        require(bindings['small_file_sha256'].get(str(Path(path).resolve()))==digest,'Actual10 runtime source not pinned')
    for field in ('preserved_receipts_sha256','scientific_configuration_sha256'):
        require(bool(reg.get(field)),'Actual10 preserved receipt/configuration identity missing')
        for path,digest in reg[field].items():
            path=_under(path,audit.parents[2])
            require(hashlib.sha256(stable_small(path)).hexdigest()==digest,'Preserved10 bytes changed: '+str(path))
    for path,digest in old.get('old_terminal_sha256',{}).items():
        path=_under(path,audit.parents[2])
        require(hashlib.sha256(stable_small(path)).hexdigest()==digest,'Retained06 terminal bytes changed')
    require(bool(old.get('old_terminal_sha256')),'Original failure ancestry empty')
    amendment=_pinned(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',bindings)
    changes={path:dict(old_sha256=before[path],new_sha256=after[path]) for path in AMENDED_PATHS}
    require(old.get('source_amendment',{}).get('changes')==changes
        and old['source_amendment'].get('receipt_sha256')==sha(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json')
        and amendment.get('changed_paths')==changes and amendment.get('optimizer_updates')==0
        and amendment.get('physical_steps')==0 and amendment.get('affine_or_label_changes') is False,
        'Original reviewed3 source amendments differ')
    require(_pinned(audit/'driver_backend_recovery10/backend_source_identity.json',bindings).get('sources')==after,
        'Actual10 backend source identity differs')
    invariants=dict(dp_reruns=0,dp_new_updates=0,collection_retries=0,new_opportunities=0,
        deploy_epoch=60,optimizer_epochs=60,rear_updates_per_role=38400,caches_own_dp=True,
        retained_complete_cache_episodes={'A':174,'B':16},prepare_reruns=0,derived_unpersisted_episode_restart_from_tick0=True,
        external_presence_is_veto=False,encoder_frozen=True,sealed_test_families=64)
    require(all(reg.get('scientific_invariants',{}).get(k)==v for k,v in invariants.items()),
        'Actual10 scientific invariants differ')
    stages=reg.get('stages',[])
    require([s.get('name') for s in stages]==list(BACKEND10_STAGES)
        and {s['name'] for s in stages if s.get('gpu') is True}=={
        'train_backend_recovery10','evaluate_development','summarize_development',
        'evaluate_test','summarize_test','audit_collection_cuda_noise_recovery05'},'Exact10 stage population required')
    guard=next(s for s in stages if s['name']=='verify_final_actual_training')
    require(guard['command'][-1]==str(audit/'presealed_guard_main_recovery10.json'),'Actual10 presealed guard path differs')
    for stage in stages:
        if stage['gpu'] and stage['name'] in ('train_backend_recovery10','evaluate_development'):
            row=_pinned(audit/'driver_backend_recovery10'/(stage['name']+'_shared_capacity06_admission.json'),bindings)
            require(row.get('capacity',{}).get('ready') is True,'Actual10 admission missing')
    boundary=reg.get('preserved_cache',{})
    require(set(boundary)=={'A','B'},'Both10 preserved cache boundaries required')
    for role,expected_count,total in (('A',174,234),('B',16,237)):
        retained=boundary[role]
        require(retained.get('count')==expected_count and len(retained.get('episodes',{}))==expected_count,
            'Actual preserved10 cache count differs: '+role)
        index=_pinned(root/'learning'/role/'backend'/role/'nominal/index.json',bindings)
        require(len(index.get('episodes',{}))==total,'Actual own-DP final nominal row population differs: '+role)
        require(all(index['episodes'].get(name,{}).get('sha256')==row['sha256'] for name,row in retained['episodes'].items()),
            'Retained10 cache identity changed: '+role)
        prepared=_pinned(root/'learning'/role/'backend/prepared.json',bindings)
        retained_prepared=reg.get('preserved_prepared_backends',{}).get(role,{})
        require(retained_prepared.get('prepared_sha256')==sha(root/'learning'/role/'backend/prepared.json')
            and all(prepared.get(k)==retained_prepared.get(k) for k in ('source_frontend_sha256','train','validation','epochs')),
            'Prepared10 backend was rebuilt or changed: '+role)
    presealed=_pinned(audit/'presealed_guard_main_recovery10.json',bindings)
    require(presealed.get('status')=='passed' and presealed.get('mode')=='main'
        and presealed.get('checkpoint_reads')==8 and presealed.get('cuda_initialized') is False,
        'Actual pre-evaluation8-checkpoint guard required')
    return reg


def prior_failures(root,audit,bindings):
    for directory in (audit/'driver_main_recovery05',audit/'wait_main_shared06',audit/'driver_backend_recovery07'):
        require(_pinned(directory/'status.json',bindings).get('stage')=='stopped_with_evidence'
            and bool(_pinned(directory/'failure.json',bindings).get('error')),'Truthful failed06/07 lifecycle required')
    old_exit=_pinned(audit/'wait_main_shared06/reviewed_main_exit.json',bindings)
    require(old_exit.get('returncode')==1 and old_exit.get('owned_helper_returned_after_cleanup') is True,
        'Failed06 helper rc1/cleanup required')
    for owner in (root/'owners/main_train',root/'owners/backend_recovery07'):
        require(bool(_pinned(owner/'project_failure.json',bindings).get('error'))
            and _pinned(owner/'project_exit.json',bindings).get('children_reaped') is True,
            'Failed06/07 owner cleanup differs')
    for path in (root/'cycle_logs/train_four_components_parallel_rears_exit.json',
                 root/'cycle_logs_recovery07/train_backend_recovery07_exit.json'):
        require(_pinned(path,bindings).get('returncode')==1,'Original failed stage rc1 required')
    driver_exit=_pinned(audit/'driver_backend_recovery07/driver_exit.json',bindings)
    require(driver_exit.get('returned_after_owned_helpers') is True and not (audit/'driver_backend_recovery07/closure.json').exists(),'Failed07 driver/helpers not reaped or contradictory closure')
    oldwatch=_pinned(audit/'OLD_WATCHER07_CLOSED_FOR_RECOVERY08.json',bindings)
    require(oldwatch.get('source_pid_reaped') is True and oldwatch.get('backend07_failure_preserved') is True
        and len(oldwatch.get('restored_verified',[]))==29 and 'MainPID=0' in oldwatch.get('service','')
        and 'ActiveState=inactive' in oldwatch['service'],'Old07 watcher must be closed/restored')
    old06=_pinned(audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json',bindings)
    require(old06.get('source_pid_reaped') is True,'Old06 watcher remains unreaped')



def prior08_failure(root,audit,bindings):
    driver=audit/'driver_backend_recovery08'
    require(_pinned(driver/'status.json',bindings).get('stage')=='stopped_with_evidence'
        and bool(_pinned(driver/'failure.json',bindings).get('error'))
        and _pinned(driver/'driver_exit.json',bindings).get('returned_after_owned_helpers') is True
        and not (driver/'closure.json').exists(),'Truthful failed08 driver/helper closure required')
    owner=root/'owners/backend_recovery08'
    require(bool(_pinned(owner/'project_failure.json',bindings).get('error'))
        and _pinned(owner/'project_exit.json',bindings).get('children_reaped') is True
        and _pinned(root/'cycle_logs_recovery08/train_backend_recovery08_exit.json',bindings).get('returncode')==1,
        'Failed08 owner/stage was not reaped')
    old=_pinned(audit/'OLD_WATCHER08_CLOSED_FOR_RECOVERY09.json',bindings)
    require(old.get('source_pid_reaped') is True and len(old.get('restored_verified',[]))==32
        and 'MainPID=0' in old.get('service','') and 'ActiveState=inactive' in old['service'],
        'Old08 watcher must be closed with32 exact schedules restored')
    # This boundary is an immutable root receipt, not a synthetic A error trace.
    _pinned(audit/'BACKEND_RECOVERY08_ACTUAL_TERMINAL_BOUNDARY09.json',bindings)


def prior09_failure(root,audit,bindings):
    driver=audit/'driver_backend_recovery09'
    require(_pinned(driver/'status.json',bindings).get('stage')=='stopped_with_evidence'
        and bool(_pinned(driver/'failure.json',bindings).get('error'))
        and _pinned(driver/'driver_exit.json',bindings).get('returned_after_owned_helpers') is True
        and not (driver/'closure.json').exists(),'Truthful failed09 driver/helper closure required')
    owner=root/'owners/backend_recovery09'
    require(bool(_pinned(owner/'project_failure.json',bindings).get('error'))
        and _pinned(owner/'project_exit.json',bindings).get('children_reaped') is True
        and _pinned(root/'cycle_logs_recovery09/train_backend_recovery09_exit.json',bindings).get('returncode')==1,
        'Failed09 owner/stage was not reaped')
    old=_pinned(audit/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',bindings)
    require(old.get('source_pid_reaped') is True and len(old.get('restored_verified',[]))==33
        and 'MainPID=0' in old.get('service','') and 'ActiveState=inactive' in old['service'],
        'Old09 watcher must be closed with33 exact schedules restored')
    _pinned(audit/'BACKEND_RECOVERY09_ACTUAL_TERMINAL_BOUNDARY10.json',bindings)

def pipeline_gate(root,audit,reg,bindings):
    runtime=audit/'parallel_rear_recovery10_runtime';last_end=None
    phases=(('caches',{'A':{0},'B':{0}}),('rear_training',{'A':{0},'B':{0}}),('serial_offline',{'A':{0},'B':{0}}))
    for phase,population in phases:
        events=_pinned(runtime/phase/'pipeline_events.json',bindings)
        starts=[r for r in events if 'pid' in r];exits=[r for r in events if 'returncode' in r]
        expected={(role,stage) for role,values in population.items() for stage in values}
        require(len(events)==2*len(expected) and len(starts)==len(exits)==len(expected)
            and {(r.get('role'),r.get('stage')) for r in starts}==expected
            and {(r.get('role'),r.get('stage')) for r in exits}==expected
            and all(r.get('returncode')==0 for r in exits),'Actual10 phase populations/rc differ: '+phase)
        for row in starts:
            end=next(r for r in exits if (r['role'],r['stage'])==(row['role'],row['stage']))
            key=(row['role']+'_cache' if phase=='caches' else row['role']+'_train' if phase=='rear_training' else row['role']+'_offline')
            command=row.get('command',[])
            require(isinstance(command,list) and '--command-key' in command and key in command
                and end['utc']>=row['utc'],'Exact10 wrapper stage identity/chronology differs')
        require(last_end is None or min(r['utc'] for r in starts)>=last_end,'10 phase ordering differs')
        last_end=max(r['utc'] for r in exits)
    cache=_pinned(runtime/'caches_admission.json',bindings)
    train=_pinned(runtime/'rear_trainers_admission.json',bindings)
    for row in (cache,train):
        require((row.get('pair',{}).get('allowed') is True or row.get('single',{}).get('allowed') is True)
            and row.get('parallel') is row.get('pair',{}).get('allowed'),'Actual10 capacity admission differs')
    complete=_pinned(runtime/'complete.json',bindings)
    require(complete.get('caches_parallel') is cache['parallel']
        and complete.get('rears_parallel') is train['parallel'] and complete.get('offline_serial') is True
        and complete.get('completed_cache_episodes_recomputed')==0 and complete.get('utc',-1)>=last_end,
        'Actual10 pipeline completion differs')
    waits=reg.get('scoped_resource_wait10',{})
    expected_keys={'A_cache','A_train','A_offline','B_cache','B_train','B_offline'}
    require(set(waits.get('commands',{}))==expected_keys
        and waits.get('version')=='scoped_preentry_resource_wait10_atomicfd_nullpause_v1',
        'Exact6 backend preentry-only wait scopes required')
    # Native tools.tight_finish deliberately calls os._exit(code), so runpy's
    # wrapper finally cannot write exit.json. Parent pipeline success, fixed
    # registered argv, completed scientific metadata and reaped owner certify
    # closure instead; absence is explicitly reported, never fabricated.
    wrapper_finalizers={}
    for key,row in waits['commands'].items():
        directory=runtime/('wait_'+key)
        start=_pinned(directory/'registration.json',bindings)
        require(start.get('version')=='scoped_preentry_resource_wait10_atomicfd_nullpause_v1'
            and start.get('command_key')==key and start.get('original_command')==row['argv']
            and start.get('registration_sha256')==sha(audit/'BACKEND_RECOVERY10_REGISTRATION.json')
            and start.get('wrapper_sha256')==waits['wrapper_sha256']
            and start.get('worker_root')==row['worker_root']
            and start.get('retry_scope')=='original_context_manager_enter_only',
            'Actual10 registered preentry-only wait identity differs: '+key)
        wrapper_finalizers[key]=(directory/'exit.json').is_file()
        require(wrapper_finalizers[key] or row['argv'][:2]==['-m','tools.tight_finish'],
            'Missing wrapper finalizer acceptance only supports native tight_finish os._exit')
        if (directory/'exit.json').is_file():
            end=_pinned(directory/'exit.json',bindings)
            require(end.get('version')==start['version'] and end.get('success') is True
                and end.get('body_retries')==0 and end.get('original_command_replayed') is False
                and end.get('method_restored') is True and end.get('source_files_modified')==0,
                'Present10 wrapper finalizer receipt differs: '+key)
    return dict(wrapper_finalizer_receipts_present=wrapper_finalizers,
        closure_proof='Exact once-launched parent pipeline returns0, fixed registered native argv, completed scientific metadata, and reaped owner. Native tools.tight_finish os._exit bypasses runpy finally; absent wrapper finalizer is not synthesized.')


def partial_dedup_gate(root,audit,binding,bindings):
    applied=_pinned(audit/'CLOSED_MAIN_DP_DEDUP08_APPLY01.json',bindings)
    require(applied.get('status')=='failed_with_prior_completed_rows_preserved'
        and len(applied.get('rows',[]))==1 and applied['rows'][0].get('role')=='A'
        and applied.get('checkpoint_payload_reads')==0 and bool(applied.get('error')),
        'Truthful A-only dedup failure required')
    row=applied['rows'][0];before=_pinned(audit/'CLOSED_MAIN_DP_DEDUP08_REVIEW.json',bindings)
    expected={r['role']:r for r in before['rows']}
    for role in 'AB':
        directory=root/'learning'/role/'training'
        for key in ('best','last'):
            current=(directory/(key+'.pt')).stat()
            actual=dict(dev=current.st_dev,ino=current.st_ino,size=current.st_size,nlink=current.st_nlink,
                mtime_ns=current.st_mtime_ns,ctime_ns=current.st_ctime_ns,allocated_bytes=current.st_blocks*512)
            require(actual==(row['after_'+key] if role=='A' else expected['B']['before_'+key]),
                'Partial dedup actual stat differs; no automatic B retry permitted')
    require(row['after_best']==row['after_last'] and row['after_best']['nlink']==2
        and row['before_best']['ino']==row['after_best']['ino'],'Actual retained A inode identity differs')
    return dict(status='A_completed_B_aborted',A_old_last_allocated_bytes=row['potential_reclaimed_after_lease_close_bytes'],
        B_reclaimed_bytes=0,actual_free_space_delta_attributed=None,
        limitation='A completed before failure; B remained unchanged. Potential released allocation is not an isolated free-space delta.')


def checkpoint_proof_gate(root,audit,binding,bindings):
    """Reuse the real10 preparation payload pass; stat only guards later change."""
    path=Path(binding['preserved_dp_preflight_path']);proof=_pinned(path,bindings)
    require(proof.get('registration_sha256')==sha(audit/'BACKEND_RECOVERY10_REGISTRATION.json')
        and proof.get('status')=='new10prepared_preserved_payload_checks_passed_not_activated'
        and proof.get('wrapper_version')=='scoped_preentry_resource_wait10_atomicfd_nullpause_v1'
        and proof.get('source_files')==295
        and proof.get('preserved_cache_counts')=={'A':174,'B':16}
        and proof.get('activated') is False and proof.get('optimizer_updates')==0,
        'Actual10 preparation payload preflight identity/status differs')
    for role,row in binding['preserved_dp'].items():
        payload=proof.get('preserved_dp',{}).get(role,{})
        train=root/'learning'/role/'training'
        require(payload.get('checkpoint_sha256')==row['checkpoint_sha256']
            and payload.get('deployment_sha256')==sha(train/'deployment.json')
            and payload.get('finished_sha256')==sha(train/'finished.json')
            and payload.get('epoch')==60 and payload.get('updates')==10200,
            'Actual10 preparation payload verification differs')
        current=(train/'best.pt').stat()
        require(identity(current)==row['checkpoint_stat_identity']
            and max(current.st_ctime_ns,current.st_mtime_ns)<=int(proof['utc']*1e9),
            'DP changed after10 actual payload verification; stat alone is insufficient')
    return dict(path=str(path),sha256=sha(path),checkpoint_payload_reads=0,
        content_proof='Actual10 registered preparation plus real payload validation pass, deployment/finished SHA and unchanged supplemental stat; no new payload read by lifecycle.')


def failed10_gate(root,audit,bindings):
    """The consumed original development startup is never converted to success."""
    driver=audit/'driver_backend_recovery10';owner=root/'evaluation/development'
    require(_pinned(driver/'status.json',bindings).get('stage')=='stopped_with_evidence'
        and bool(_pinned(driver/'failure.json',bindings).get('error'))
        and _pinned(driver/'driver_exit.json',bindings).get('returned_after_owned_helpers') is True
        and not (driver/'closure.json').exists(),'Original10 must remain failed and reaped')
    require(bool(_pinned(owner/'project_failure.json',bindings).get('error'))
        and _pinned(owner/'project_exit.json',bindings).get('children_reaped') is True
        and not (owner/'project_complete.json').exists()
        and _pinned(root/'cycle_logs_recovery10/evaluate_development_exit.json',bindings).get('returncode')==1,
        'Original development failed owner/stage must remain truthful')
    for stage in COMPLETED10_STAGES:
        require(_pinned(root/'cycle_logs_recovery10'/(stage+'_exit.json'),bindings).get('returncode')==0,
            'Completed original10 work differs')
    for stage in REMAINING_STAGES:
        require(not (root/'cycle_logs_recovery10'/(stage+'_exit.json')).exists(),
            'Original10 must not claim an unlaunched remaining stage')
    started=_pinned(owner/'project_job_started.json',bindings)
    require(len(started.get('commands',[]))==len(started.get('job',{}).get('stages',[]))==8
        and Path(started['job'].get('resource_root','')).resolve()==owner,
        'Original development exact8 launched native stages required')
    termination=_pinned(audit/'EVAL_STARTUP12_OWNED_TERMINATION.json',bindings)
    require(termination.get('version')=='owned_precontrol_startup_termination12_v1'
        and termination.get('student_actions_observed') is False
        and termination.get('completed_wave_result_files')==7 and termination.get('test_prepared') is False
        and termination.get('external_processes_signaled')==0,
        'Original consumed startup boundary differs')
    return dict(failed=True,development_registered_episodes=32,development_consumed_waves=8,
        development_terminal_waves=7,development_software_unfinished_episodes=4,
        reruns=0,physical_steps='Unknown beyond durable records; no zero-physics claim')


def registry12_gate(root,audit,binding,bindings):
    reg=_pinned(audit/'EVALUATION_RECOVERY12_REGISTRATION.json',bindings)
    review=_pinned(audit/'ROOT_EVALUATION_RECOVERY12_REVIEW.json',bindings)
    require(reg.get('version')=='paired_unstarted_test_recovery12_v1'
        and reg.get('root')==str(root) and reg.get('original_deadline')==binding['original_deadline']
        and reg.get('source_sha256')==binding['production']['after_sha256']
        and reg.get('scientific_changes')==reg.get('collection_retries')==reg.get('training_updates')==0
        and reg.get('startup_timeout_seconds')==1800,'Exact evaluation-only12 scope required')
    require(review.get('status')=='source_and_actual_boundary_review_passed_inactive'
        and review.get('registration_sha256')==sha(audit/'EVALUATION_RECOVERY12_REGISTRATION.json'),
        'Root actual12 source/boundary review required')
    require(set(reg.get('orchestration_sha256',{}))=={str(audit/name) for name in RUNTIME12_SOURCES},
        'Exact11 recovery12 source population required')
    for path,digest in reg['orchestration_sha256'].items():
        require(bindings['small_file_sha256'].get(str(Path(path).resolve()))==digest,
            'Recovery12 runtime source not pinned')
    for key,path in [('original_registration_sha256',audit/'BACKEND_RECOVERY10_REGISTRATION.json'),
        ('development_registration_sha256',root/'evaluation/development/registration.json'),
        ('restoration_sha256',audit/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json'),
        ('original_startup_failure_sha256',audit/'EVAL_STARTUP12_OWNED_TERMINATION.json'),
        ('cohorts_sha256',root/'cohorts.json')]:
        require(reg.get(key)==sha(path),'Recovery12 inherited identity differs: '+key)
    census=reg.get('lifecycle',{})
    expected=dict(development_population=8,development_consumed=8,development_closed_wave_receipts=7,
        development_reruns=0,test_prepared=False,test_registered_families=64,test_launched=0,
        lifecycle_only=True,outcome_values_interpreted=False)
    require(all(census.get(k)==v for k,v in expected.items()),'Original8/7 and unstarted64 census differs')
    require(census.get('original_owner_failure_sha256')==sha(root/'evaluation/development/project_failure.json')
        and census.get('original_driver_failure_sha256')==sha(audit/'driver_backend_recovery10/failure.json'),
        'Frozen original10 failures changed')
    rows=census.get('rows',[])
    require(len(rows)==8 and [r.get('index') for r in rows]==list(range(8)),'Exact8 development census rows required')
    for index,row in enumerate(rows):
        wave=root/'evaluation/development'/f'wave{index:02d}'
        require(row.get('wave')==str(wave) and row.get('consumed') is True
            and row.get('eligible_for_rerun') is False and row.get('terminal_receipts_present') is (index<7),
            'Development census opportunity identity differs')
        require((wave/'run_started.json').is_file()
            and ((wave/'results.json').is_file() and (wave/'process_exit_requested.json').is_file()) is (index<7),
            'Original7 closed plus consumed startup wave required')
        inventory=row.get('inventory',{})
        actual={str(p.relative_to(wave)) for p in wave.rglob('*') if p.is_file()}
        require(set(inventory)==actual,'Original consumed wave inventory changed')
        for relative,item in inventory.items():
            path=wave/relative;stat=path.stat()
            require(not path.is_symlink() and stat.st_size==item.get('bytes')
                and stat.st_mtime_ns==item.get('mtime_ns'),'Consumed development file identity changed')
            if 'sha256' in item: require(sha(path)==item['sha256'],'Consumed development bytes changed')
        if index==7:
            require(not (wave/'initialization.json').exists() and not list(wave.glob('*/started.json')),
                'Original failed startup advanced beyond frozen boundary')
    stages=reg.get('stages',[])
    prior_stages=_pinned(audit/'BACKEND_RECOVERY10_REGISTRATION.json',bindings)['stages']
    exact=[dict(name=row['name'],args=row['command'],gpu=row['gpu'])
        for row in prior_stages if row['name'] in REMAINING_STAGES]
    for row in exact:
        if row['name']=='evaluate_test':row['args']=['-m',
            'outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',
            'project','evaluate','--job',str(audit.parents[2]/'configs/experiments/paired_dagger_20261122/evaluate_test.json')]
    require(stages==exact,'Recovery12 must retain exact original5 commands except startup wrapper')
    require([r.get('name') for r in stages]==list(REMAINING_STAGES)
        and {r['name'] for r in stages if r.get('gpu') is True}==set(STANDALONE_STAGES)|{'evaluate_test'},
        'Exact remaining5 recovery12 stages required')
    teststage=next(r for r in stages if r['name']=='evaluate_test')
    require(teststage.get('args')==['-m','outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',
        'project','evaluate','--job',str(audit.parents[2]/'configs/experiments/paired_dagger_20261122/evaluate_test.json')],
        'Only startup diagnostic12 evaluation wrapper allowed')
    activated=_pinned(audit/'driver_eval_recovery12/registration.json',bindings)
    require(activated==dict(reg,activated=True,activated_utc=activated.get('activated_utc')),
        'Activated12 registration differs')
    require(reg.get('frozen_pairs')==_pinned(root/'evaluation/development/registration.json',bindings).get('pairs'),
        'Recovery12 must retain original four complete frozen policies')
    return reg


def evaluation_execution_inventory(root,cohort,registration,bindings=None):
    """Counts durable launch/terminal receipts; full outcome denominator is separate."""
    directory=Path(root)/'evaluation'/cohort;n=registration.get('n')
    require(type(n) is int and n==(8 if cohort=='development' else 64),'Frozen cohort population differs')
    attempts=[];terminal=[];unexecuted=[];unfinished=[]
    for index in range(n):
        wave=directory/f'wave{index:02d}'
        launched=(wave/'run_started.json').is_file()
        done=(wave/'results.json').is_file() and (wave/'process_exit_requested.json').is_file()
        require(not done or launched,'Terminal wave without durable launch identity')
        if bindings is not None:
            if launched:_pinned(wave/'run_started.json',bindings)
            if done:_pinned(wave/'process_exit_requested.json',bindings)
        (attempts if launched else unexecuted).append(index)
        if done:terminal.append(index)
        elif launched:unfinished.append(index)
    return dict(cohort=cohort,registered_waves=n,registered_episodes=4*n,
        attempted_waves=len(attempts),attempted_episodes=4*len(attempts),
        terminal_waves=len(terminal),terminal_episodes=4*len(terminal),
        software_unfinished_waves=len(unfinished),software_unfinished_episodes=4*len(unfinished),
        unexecuted_waves=len(unexecuted),unexecuted_episodes=4*len(unexecuted),
        attempted_indices=attempts,terminal_indices=terminal,unexecuted_indices=unexecuted,
        denominator_does_not_certify_attempts=True)


def closure_gate(root,audit):
    """Truthful failed12/failed13 driver plus completed finite admission14 suffix."""
    root,audit=Path(root).resolve(),Path(audit).resolve()
    driver=audit/'driver_test_admission14';owner=root/'owners/test_fragment13'
    required=[audit/BINDING_NAME,audit/REVIEW_NAME,driver/'closure.json',driver/'driver_exit.json',
        audit/'TEST_FRAGMENT13_REGISTRATION.json',audit/'ROOT_TEST_FRAGMENT13_BOUNDARY.json',
        audit/'WATCHER14_ACTUAL_CLOSED.json',audit/'TEST_ADMISSION14_REGISTRATION.json',
        audit/'ROOT_TEST_ADMISSION14_BOUNDARY.json',root/'evaluation/test/registration.json']
    for directory in (root/'owners/backend_recovery10',owner):
        required.extend(directory/name for name in ('project_job_started.json','project_complete.json','project_exit.json'))
    for scope in ('common180_parents','learning'):
        for role in 'AB':
            required.append(root/scope/role/'state_data/data.json')
            for sub in ('training',f'backend/{role}/training'):
                required.extend(root/scope/role/sub/name for name in ('best.pt','deployment.json','finished.json'))
    missing=[str(p) for p in required if not p.is_file()]
    if missing:return dict(status='pending',missing=missing,checkpoint_reads=0)
    binding,bindings=reviewed_binding(root,audit);watcher=watcher_closed(audit)
    prior_failures(root,audit,bindings);prior08_failure(root,audit,bindings);prior09_failure(root,audit,bindings)
    reg=registry_gate(root,audit,binding,bindings);pipeline=pipeline_gate(root,audit,reg,bindings)
    failed10=failed10_gate(root,audit,bindings);reg12=registry12_gate(root,audit,binding,bindings)
    failed12=failed12_gate(root,audit,binding,bindings)
    admission14,failed13=admission14_gate(root,audit,binding,bindings,failed12)
    fragment=fragment14_gate(root,audit,binding,bindings,failed12,admission14,failed13)
    for stage in OLD_COMPLETED_STAGES:
        require(_pinned(root/'cycle_logs'/(stage+'_exit.json'),bindings).get('returncode')==0,'Prior completed work changed')
    backend=root/'owners/backend_recovery10'
    require(not (backend/'project_failure.json').exists(),'Completed backend10 owner failed')
    began,done,end=[_pinned(backend/name,bindings) for name in ('project_job_started.json','project_complete.json','project_exit.json')]
    require(began.get('job')==_pinned(audit/'proposed_backend_recovery10.json',bindings)
        and len(began.get('commands',[]))==len(began['job']['stages'])==done.get('stages')
        and end.get('children_reaped') is True
        and began['utc']>=_pinned(root/'owners/backend_recovery09/project_exit.json',bindings)['utc'],
        'Completed backend10 native stage/closure identity differs')
    production=binding['production'];before,after=production['before_sha256'],production['after_sha256']
    changed={p for p in before if before[p]!=after.get(p)}
    require(len(before)==295 and set(before)==set(after) and changed==AMENDED_PATHS
        and set(production.get('changed_paths',[]))==changed,'Exact295-source/3-file amendment required')
    for relative,digest in after.items():
        require(bindings['small_file_sha256'].get(str((audit.parents[2]/relative).resolve()))==digest,
            'Production source is not pinned')
    for field,expected0 in (('snapshot_paths',before),('after_snapshot_paths',after)):
        snapshots=production.get(field,{})
        require(set(snapshots)==changed,'Both before/after snapshot populations required')
        for relative,path in snapshots.items():
            require(bindings['small_file_sha256'].get(str(Path(path).resolve()))==expected0[relative],
                'Snapshot direction/SHA differs')
    require(_pinned(root/'learning/training_source_identity.json',bindings).get('sources')==before,
        'Original completed DP source identity changed')
    require(set(binding.get('preserved_dp',{}))=={'A','B'},'Both DP deployments required')
    for role,row in binding['preserved_dp'].items():
        directory=root/'learning'/role/'training'
        deployment=_pinned(directory/'deployment.json',bindings);finished=_pinned(directory/'finished.json',bindings)
        require(deployment.get('epoch')==finished.get('epochs')==60
            and deployment.get('update')==finished.get('updates')==10200 and finished.get('complete') is True
            and deployment.get('checkpoint_sha256')==row['checkpoint_sha256']==reg['preserved_dp'][role]['checkpoint_sha256'],
            'Preserved actual DP60 deployment age/SHA differs')
    dedup=partial_dedup_gate(root,audit,binding,bindings)
    proof=checkpoint_proof_gate(root,audit,binding,bindings)
    final=_pinned(root/'evaluation/test/registration.json',bindings)
    require(final.get('cohort')=='test' and final.get('n')==64
        and set(final.get('pairs',{}))=={'A_new','B_new','A_parent','B_parent'}
        and final.get('source_sha256')==after,'Sealed64 complete policy/source freeze differs')
    require(final.get('pairs')==reg12.get('frozen_pairs'),'Sealed64 changed original frozen policies')
    execution=dict(development=evaluation_execution_inventory(root,'development',
        _pinned(root/'evaluation/development/registration.json',bindings),bindings),test=fragment['execution'])
    require(execution['development']['attempted_waves']==8 and execution['development']['terminal_waves']==7
        and execution['development']['software_unfinished_episodes']==4,'Original development denominator differs')
    return dict(status='closed',version='paired_final_recovery14_lifecycle_v1',checkpoint_reads=0,
        failed06_preserved=True,failed07_preserved=True,failed08_preserved=True,failed09_preserved=True,
        failed10_preserved=True,failed12_preserved=True,failed13_driver_preserved=True,preserved_dp_retrained=False,
        original_deadline=binding['original_deadline'],root_review=bindings,partial_dp_dedup=dedup,
        root10_inherited_payload_proof=proof,watcher_closure=watcher,
        production_amendment_paths=sorted(changed),wrapper_closure=pipeline,
        original10_failure=failed10,original12_failure=failed12,original13_driver_failure=failed13,fragment14=fragment,evaluation_execution=execution,
        payload_read_accounting=dict(presealed_main_guard_checkpoint_reads=8,
            final_detailed_ledger_checkpoint_reads=8,assembler_checkpoint_reads=0,exporter_checkpoint_reads=0,
            scope='Guard and final extractor are distinct8-payload passes; root/runtime preflight checks are additional.'),
        scope='Bounded metadata/source validation. No weight payload read by lifecycle.')


def watcher_restoration_proof(audit,watcher,bindings,*,historical_receipt=None,expected_count=None,verify_current=False,run_root=None):
    """Audit every backed-up identity; current population comes from real backups.

    Historical09/10 schedules are proven against their immutable backup and
    root closure receipts. Current12 additionally checks actual restored schedule
    bytes/absence and its actual service/reaping/root receipt.
    """
    audit=Path(audit).resolve();project=audit.parents[2]
    require(watcher in ('watch_shared_capacity09','watch_shared_capacity10','watch_shared_capacity12','watch_shared_capacity13','watch_shared_capacity14'),'Unsupported restoration scope')
    expected_root=_under(run_root if run_root is not None else project/'outputs/visuotactile/run/tight_paired_dagger_20261122_01',project)
    directory=audit/watcher
    backups=sorted(directory.glob('resource_watch_v4_backups_*'))
    require(len(backups)==1 and backups[0].is_dir() and not backups[0].is_symlink(),
        'Exactly one real watcher backup population required: '+watcher)
    def read(path):
        path=Path(path).resolve();raw=stable_small(path)
        require(bindings['small_file_sha256'].get(str(path))==hashlib.sha256(raw).hexdigest(),
            'Restoration evidence not pinned or changed: '+str(path))
        return json.loads(raw)
    exit_path=directory/'resource_watch_v4_exit.json';exited=read(exit_path)
    require(exited.get('modified_schedules')==0 and exited.get('conflicts')==[]
        and exited.get('exact_original_byte_restoration') is True,
        'Watcher'+watcher[-2:]+' must restore original schedule bytes')
    require(str(backups[0].name).endswith('_'+str(exited.get('pid'))),
        'Backup population PID differs from reaped watcher')
    records={};backup_sources={}
    files=sorted(backups[0].iterdir())
    require(files and all(x.is_file() and not x.is_symlink() and x.suffix=='.json' for x in files),
        'Complete bounded watcher backup metadata population required')
    for path in files:
        row=read(path);target=_under(row.get('path',''),project)
        require(target.is_relative_to(expected_root)
            and target.name=='concurrency.json' and str(target)==row.get('path')
            and path.name==hashlib.sha256(str(target).encode()).hexdigest()+'.json'
            and type(row.get('existed')) is bool and str(target) not in records,
            'Watcher backup identity/path population differs')
        if row['existed']:
            require(isinstance(row.get('base64'),str),'Existing backup bytes required')
            raw=base64.b64decode(row['base64'],validate=True)
            require(hashlib.sha256(raw).hexdigest()==row.get('sha256'),'Backup original byte SHA differs')
        else:
            require(row.get('base64') is None and row.get('sha256') is None,'Absent backup must have null byte identity')
            raw=None
        if verify_current:
            require(not target.is_symlink() and (target.is_file() if row['existed'] else not target.exists()),
                'Restored schedule existence differs: '+str(target))
            if row['existed']:require(stable_small(target)==raw,'Restored schedule bytes differ: '+str(target))
        records[str(target)]={key:row[key] for key in ('path','existed','sha256')}
        backup_sources[str(path)]=sha(path)
    require(expected_count is None or len(records)==expected_count,'Watcher restored backup population count differs')
    restored=exited.get('restored')
    require(isinstance(restored,list) and len(restored)==len(set(restored)) and set(restored)<=set(records),
        'Watcher exit restoration paths differ from backup identities')
    historical=None
    if historical_receipt is not None:
        historical_path=Path(historical_receipt);historical=read(historical_path)
        rows=historical.get('restored_verified',[])
        require(isinstance(rows,list) and len(rows)==len(records)
            and all(isinstance(row,dict) for row in rows)
            and len({row.get('path') for row in rows})==len(records)
            and {row.get('path'):{k:row.get(k) for k in ('path','existed','sha256')} for row in rows}==records,
            'Every historical restoration identity must match its exact backup')
        for row in rows:
            if 'backup_sha256' in row:
                backup=backups[0]/(hashlib.sha256(row['path'].encode()).hexdigest()+'.json')
                require(row['backup_sha256']==sha(backup),'Historical backup SHA differs')
        require(historical.get('source_pid_reaped') is True and historical.get('pid')==exited.get('pid')
            and service_is_closed(historical.get('service'),require_success=False)
            and historical.get('exit_sha256')==sha(exit_path),'Historical root restoration/exit binding differs')
    return dict(watcher=watcher,backup_count=len(records),backup_directory=str(backups[0]),
        backup_file_sha256=backup_sources,exit_sha256=sha(exit_path),
        restoration_identities=records,all_identities_verified=True,current_schedule_bytes_checked=verify_current,
        historical_restoration_receipt_sha256=sha(historical_receipt) if historical_receipt else None,
        population_rule='Actual complete backup metadata population; current watcher count is never assumed.')



def build_binding(root=ROOT,audit=HERE,*,preflight_name='BACKEND_RECOVERY10_ACTUAL_PREFLIGHT.json'):
    """Return pending without weights/sealed reads; root writes/reviews only after closure."""
    root,audit=Path(root).resolve(),Path(audit).resolve();project=audit.parents[2]
    reg=read_small(audit/'BACKEND_RECOVERY10_REGISTRATION.json')
    old=read_small(audit/'BACKEND_RECOVERY07_REGISTRATION.json')
    before,after=reg['old_production_sha256'],reg['source_sha256']
    amendment=read_small(audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json')
    before_paths={p:str(audit/'frozen_source'/p) for p in AMENDED_PATHS}
    after_paths={p:str(Path(amendment['snapshot'])/p) for p in AMENDED_PATHS}
    for paths,expected0 in ((before_paths,before),(after_paths,after)):
        for relative,path in paths.items():require(hashlib.sha256(stable_small(path)).hexdigest()==expected0[relative],'Real snapshot SHA/direction differs')
    stages={}
    watcher=audit/'watch_shared_capacity14/resource_watch_v4_exit.json'
    preflight=audit/preflight_name
    require(preflight.parent==audit and preflight.suffix=='.json','Explicit root preflight must be an audit JSON filename')
    paths={audit/'BACKEND_RECOVERY09_REGISTRATION.json',
        audit/'driver_backend_recovery09/status.json',audit/'driver_backend_recovery09/failure.json',audit/'driver_backend_recovery09/driver_exit.json',
        root/'owners/backend_recovery09/project_failure.json',root/'owners/backend_recovery09/project_exit.json',
        root/'cycle_logs_recovery09/train_backend_recovery09_exit.json',
        audit/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',audit/'BACKEND_RECOVERY09_ACTUAL_TERMINAL_BOUNDARY10.json',
        audit/'BACKEND_RECOVERY10_REGISTRATION.json',audit/'BACKEND_RECOVERY08_REGISTRATION.json',audit/'BACKEND_RECOVERY07_REGISTRATION.json',
        audit/'driver_backend_recovery08/status.json',audit/'driver_backend_recovery08/failure.json',audit/'driver_backend_recovery08/driver_exit.json',
        root/'owners/backend_recovery08/project_failure.json',root/'owners/backend_recovery08/project_exit.json',
        root/'cycle_logs_recovery08/train_backend_recovery08_exit.json',
        audit/'OLD_WATCHER08_CLOSED_FOR_RECOVERY09.json',audit/'BACKEND_RECOVERY08_ACTUAL_TERMINAL_BOUNDARY09.json',
        audit/'proposed_backend_recovery10.json',audit/'proposed_backend_recovery07.json',
        audit/'PAIRED_CONTRACT07_SOURCE_AMENDMENT.json',audit/'OLD_WATCHER06_CLOSED_FOR_RECOVERY07.json',
        audit/'OLD_WATCHER07_CLOSED_FOR_RECOVERY08.json',audit/'CLOSED_MAIN_DP_DEDUP08_APPLY01.json',
        audit/'CLOSED_MAIN_DP_DEDUP08_REVIEW.json',audit/'CLOSED_MAIN_DP_DEDUP08_PARTIAL_DIAGNOSIS01.json',
        audit/'driver_backend_recovery10/registration.json',audit/'driver_backend_recovery10/status.json',
        audit/'driver_backend_recovery10/failure.json',audit/'driver_backend_recovery10/driver_exit.json',
        audit/'driver_backend_recovery10/backend_source_identity.json',
        audit/'driver_backend_recovery07/status.json',audit/'driver_backend_recovery07/failure.json',audit/'driver_backend_recovery07/driver_exit.json',
        audit/'driver_main_recovery05/registration.json',audit/'driver_main_recovery05/status.json',audit/'driver_main_recovery05/failure.json',
        audit/'wait_main_shared06/status.json',audit/'wait_main_shared06/failure.json',audit/'wait_main_shared06/reviewed_main_exit.json',
        root/'owners/main_train/project_failure.json',root/'owners/main_train/project_exit.json',
        root/'owners/backend_recovery07/project_failure.json',root/'owners/backend_recovery07/project_exit.json',
        root/'cycle_logs/train_four_components_parallel_rears_exit.json',root/'cycle_logs_recovery07/train_backend_recovery07_exit.json',
        root/'learning/training_source_identity.json',root/'evaluation/test/registration.json',
        root/'learning/A/backend/A/nominal/index.json',root/'learning/B/backend/B/nominal/index.json',
        root/'learning/A/backend/prepared.json',root/'learning/B/backend/prepared.json',
        audit/'presealed_guard_main_recovery10.json',watcher,preflight}
    for name,expected in SOURCE_DEPENDENCIES_SHA256.items():
        require(hashlib.sha256(stable_small(audit/name)).hexdigest()==expected,
            'Inherited source/fixture dependency changed: '+name)
        paths.add(audit/name)
    paths.update(Path(p) for p in reg['orchestration_sha256'])
    paths.update(Path(p) for field in ('preserved_receipts_sha256','scientific_configuration_sha256')
        for p in reg[field] if Path(p).suffix.lower() in SAFE_SUFFIXES)
    paths.update(Path(p) for p in old['old_terminal_sha256'] if Path(p).suffix.lower() in SAFE_SUFFIXES)
    paths.update(project/p for p in after);paths.update(Path(p) for p in before_paths.values());paths.update(Path(p) for p in after_paths.values())
    paths.update(Path(p) for p in stages.values());paths.update(root/'cycle_logs'/(s+'_exit.json') for s in OLD_COMPLETED_STAGES)
    runtime=audit/'parallel_rear_recovery10_runtime'
    paths.update(runtime/phase/'pipeline_events.json' for phase in ('caches','rear_training','serial_offline'))
    paths.update(runtime/name for name in ('caches_admission.json','rear_trainers_admission.json','complete.json'))
    for key in ('A_cache','A_train','A_offline','B_cache','B_train','B_offline'):
        paths.add(runtime/('wait_'+key)/'registration.json')
        if (runtime/('wait_'+key)/'exit.json').is_file(): paths.add(runtime/('wait_'+key)/'exit.json')
    paths.update(audit/'driver_backend_recovery10'/(s['name']+'_shared_capacity06_admission.json') for s in reg['stages'] if s['gpu'] and s['name'] in ('train_backend_recovery10','evaluate_development'))
    paths.update(root/'cycle_logs_recovery10'/(stage+'_exit.json') for stage in COMPLETED10_STAGES+('evaluate_development',))
    reg12=read_small(audit/'EVALUATION_RECOVERY12_REGISTRATION.json')
    reg13_path=audit/'TEST_FRAGMENT13_REGISTRATION.json'
    if not reg13_path.is_file():return dict(status='pending',missing=[str(reg13_path)],checkpoint_reads=0)
    reg13=read_small(reg13_path)
    paths.update(audit/name for name in ('EVALUATION_RECOVERY12_REGISTRATION.json','ROOT_EVALUATION_RECOVERY12_REVIEW.json',
        'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json','EVAL_STARTUP12_OWNED_TERMINATION.json','WATCHER12_ACTUAL_CLOSED.json',
        'TEST_FRAGMENT13_REGISTRATION.json','ROOT_TEST_FRAGMENT13_REVIEW.json','ROOT_TEST_FRAGMENT13_BOUNDARY.json',
        'test_fragment13_job.json','WATCHER13_ACTUAL_CLOSED.json','TEST_ADMISSION14_REGISTRATION.json',
        'ROOT_TEST_ADMISSION14_REVIEW.json','ROOT_TEST_ADMISSION14_BOUNDARY.json','WATCHER14_ACTUAL_CLOSED.json'))
    paths.update(Path(p) for p in reg12['orchestration_sha256'])
    paths.update(Path(p) for p in reg13['orchestration_sha256'])
    reg14_path=audit/'TEST_ADMISSION14_REGISTRATION.json'
    if not reg14_path.is_file():return dict(status='pending',missing=[str(reg14_path)],checkpoint_reads=0)
    reg14=read_small(reg14_path)
    paths.update(Path(p) for p in reg14['orchestration_sha256'])
    boundary14=read_small(audit/'ROOT_TEST_ADMISSION14_BOUNDARY.json')
    rawlog=root/'cycle_logs_fragment13/evaluate_fragment13.log'
    paths.update(Path(p) for p in boundary14['artifact_sha256'] if Path(p)!=rawlog)
    boundary=read_small(audit/'ROOT_TEST_FRAGMENT13_BOUNDARY.json')
    paths.update(Path(p) for p in boundary['artifact_sha256'])
    paths.update(audit/'driver_eval_recovery12'/name for name in ('registration.json','status.json','failure.json','driver_exit.json'))
    paths.update(audit/'driver_eval_recovery12'/(name+'_capacity_admission.json') for name in ('summarize_development','evaluate_test'))
    paths.update(audit/'driver_test_fragment13'/name for name in ('registration.json','status.json','failure.json','driver_exit.json','evaluate_fragment13_capacity_admission.json'))
    paths.update(audit/'driver_test_admission14'/name for name in ('registration.json','status.json','closure.json','driver_exit.json','native_owner_dispatch14.json','evaluate_fragment14_capacity_admission.json'))
    stages={name:str(root/'cycle_logs_recovery12'/(name+'_exit.json')) for name in ('summarize_development','prepare_test','evaluate_test')}
    stages['evaluate_fragment13_failed']=str(root/'cycle_logs_fragment13/evaluate_fragment13_exit.json')
    stages['evaluate_fragment14']=str(root/'cycle_logs_admission14/evaluate_fragment14_exit.json')
    for row in reg13['tail_stages']:
        directory='cycle_logs_recovery12' if row['skip_closed'] else 'cycle_logs_admission14'
        stages[row['name']]=str(root/directory/(row['name']+'_exit.json'))
        if not row['skip_closed']:paths.add(audit/'driver_test_admission14'/(row['name']+'_capacity_admission.json'))
        else:paths.add(audit/'driver_eval_recovery12'/(row['name']+'_capacity_admission.json'))
    paths.update(Path(p) for p in stages.values())
    paths.update((root/'cohorts.json',root/'evaluation/development/registration.json',
        root/'evaluation/development/project_job_started.json',root/'evaluation/development/project_failure.json',
        root/'evaluation/development/project_exit.json'))
    for owner in (root/'owners/backend_recovery10',root/'owners/test_fragment13'):
        paths.update(owner/name for name in ('project_job_started.json','project_complete.json','project_exit.json'))
    owner=root/'owners/test_fragment13'
    paths.update(owner/name for name in ('session.json','project_status.json','monitor.json'))
    paths.update((root/'evaluation/test')/name for name in ('session.json','project_job_started.json','project_failure.json','project_exit.json','project_status.json','monitor.json'))
    paths.add(Path(read_small(root/'evaluation/test/registration.json')['job']))
    paths.update((owner/'dispatches').glob('*.json'))
    paths.update(Path(p) for p in reg13['original_wave_session_sha256'])
    for cohort,count in (('development',8),('test',64)):
        for index in range(count):
            wave=root/'evaluation'/cohort/f'wave{index:02d}'
            for name in ('plan.json','session.json','run_started.json','process_exit_requested.json'):
                if (wave/name).is_file():paths.add(wave/name)
    preserved={}
    for role in 'AB':
        train=root/'learning'/role/'training';paths.update(train/name for name in ('deployment.json','finished.json'))
        checkpoint=train/'best.pt'
        if checkpoint.is_file():preserved[role]=dict(checkpoint_sha256=reg['preserved_dp'][role]['checkpoint_sha256'],
            checkpoint_stat_identity=identity(checkpoint.stat()),content_identity_source='Actual root10 SHA preflight plus unchanged supplemental stat; no payload read by builder')
        else:paths.add(checkpoint)
    for watch in ('watch_shared_capacity09','watch_shared_capacity10','watch_shared_capacity12','watch_shared_capacity13','watch_shared_capacity14'):
        paths.add(audit/watch/'resource_watch_v4_exit.json')
        paths.update((audit/watch).glob('resource_watch_v4_backups_*/*.json'))
    watcher=audit/'watch_shared_capacity14/resource_watch_v4_exit.json'
    paths.discard(audit/'driver_eval_recovery12/closure.json')
    missing=sorted(str(p) for p in paths if not p.is_file())
    if missing:return dict(status='pending',missing=missing,checkpoint_reads=0)
    require((audit/'driver_test_admission14/closure.json').is_file(),'Real14 closure required before binding')
    require(not (audit/'driver_test_fragment13/closure.json').exists(),'Failed13 driver must remain failed')
    require(rawlog.is_file() and not rawlog.is_symlink(),'Failed13 raw helper log must remain local')
    return dict(version='paired_final_recovery14_bindings_v1',root=str(root),original_deadline=reg['original_deadline'],
        small_file_sha256={str(p.resolve()):hashlib.sha256(stable_small(p)).hexdigest() for p in sorted(paths)},
        local_retained_log_sha256={str(rawlog):hashlib.sha256(stable_small(rawlog)).hexdigest()},
        preserved_dp=preserved,preserved_dp_preflight_path=str(preflight),
        production=dict(before_sha256=before,after_sha256=after,changed_paths=sorted(AMENDED_PATHS),
            snapshot_paths=before_paths,after_snapshot_paths=after_paths),stage_exit_paths=stages,watcher_exit_path=str(watcher))


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
            'Successful10 eight-selected-ledger pass required')
    require(status.get('extractor_sha256') == sha(audit/'extract_selected_ledgers_recovery14.py')
        and Path(status.get('root','')).resolve() == root
        and Path(status.get('output','')).resolve() == directory
        and status.get('raw_data_opened') is False and status.get('model_forward_calls') == 0
        and status.get('files') == summary.get('files'), 'Selected10 extraction binding differs')
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

# Fragment13-specific gates are intentionally metadata-only, never runtime entry points.
RUNTIME13_SOURCES=('test_fragment13.py','shared_capacity13.py','run_main_recovery05.py','shared_capacity06.py',
    'resource_watch_v4.py','coexistence_watch_recovery05.py','resource_watch_parallel_proposal.py',
    'driver_support.py','driver_support_v4.py','audit_collection_recovery05.py',
    'audit_collection_noise.py','audit_collection_integrity.py','collection_recovery05_scope.py')


def service_is_closed(service,*,require_success=True):
    if isinstance(service,str):
        rows=dict(line.split('=',1) for line in service.splitlines() if '=' in line)
    elif isinstance(service,dict):rows=service
    else:return False
    return str(rows.get('MainPID'))=='0' and rows.get('ActiveState')=='inactive' and (not require_success or rows.get('Result')=='success')


def failed12_gate(root,audit,binding,bindings):
    driver=audit/'driver_eval_recovery12';test=root/'evaluation/test'
    failure=_pinned(driver/'failure.json',bindings);end=_pinned(driver/'driver_exit.json',bindings)
    status=_pinned(driver/'status.json',bindings)
    require(bool(failure.get('error')) and status.get('stage')=='stopped_with_evidence'
        and end.get('returned_after_owned_helpers') is True and not (driver/'closure.json').exists(),
        'Original12 truthful failed driver and owned-helper closure required')
    require(bool(_pinned(test/'project_failure.json',bindings).get('error'))
        and _pinned(test/'project_exit.json',bindings).get('children_reaped') is True
        and not (test/'project_complete.json').exists(),'Original12 native test owner must remain failed/reaped')
    boundary=_pinned(audit/'ROOT_TEST_FRAGMENT13_BOUNDARY.json',bindings)
    require(boundary.get('status')=='actual_original12_boundary_closed_for_missing_only_fragment'
        and boundary.get('deadline')==binding['original_deadline']
        and boundary.get('all_original12_actors_reaped') is True and boundary.get('no_external_signals') is True,
        'Root actual failed12 closed-boundary certificate required')
    mandatory=[test/name for name in ('project_failure.json','project_exit.json','monitor.json','project_status.json','project_job_started.json')]
    mandatory += [driver/'failure.json',driver/'driver_exit.json',audit/'WATCHER12_ACTUAL_CLOSED.json']
    proof=boundary.get('artifact_sha256',{})
    require(set(proof)>={str(p) for p in mandatory},'Actual12 boundary omits mandatory original actors')
    for path,digest in proof.items():
        require(bindings['small_file_sha256'].get(str(Path(path).resolve()))==digest,
            'Actual12 root boundary artifact identity changed')
    watch=_pinned(audit/'WATCHER12_ACTUAL_CLOSED.json',bindings)
    actors={_pinned(test/'project_job_started.json',bindings)['pid'],_pinned(test/'monitor.json',bindings)['pid'],
        _pinned(test/'project_status.json',bindings)['pid'],end['pid'],watch['pid']}
    identities=boundary.get('closed_identities',[])
    require(len({row.get('pid') for row in identities})==len(identities)
        and {row.get('pid') for row in identities}>=actors
        and all(type(row.get('pid')) is int and row['pid']>0 and type(row.get('created')) in (int,float)
            and math.isfinite(row['created']) for row in identities),'Complete original12 PID/birth identities required')
    native_start=_pinned(test/'project_job_started.json',bindings);native_end=_pinned(test/'project_exit.json',bindings)
    activated=_pinned(driver/'registration.json',bindings)
    require(activated['activated_utc']<=native_start['utc']<=native_end['utc']<=end['utc']<=binding['original_deadline']
        and _pinned(audit/'presealed_guard_main_recovery10.json',bindings)['observed_utc']<=native_start['utc'],
        'Original12 native owner/driver/guard chronology differs')
    require(_pinned(root/'cycle_logs_recovery12/evaluate_test_exit.json',bindings).get('returncode')==1,
        'Original12 failed test stage rc1 required')
    for name in ('summarize_development','prepare_test'):
        require(_pinned(root/'cycle_logs_recovery12'/(name+'_exit.json'),bindings).get('returncode')==0,
            'Completed pre-failure12 stage differs')
    return dict(failed=True,driver_exit_utc=end['utc'],boundary_sha256=sha(audit/'ROOT_TEST_FRAGMENT13_BOUNDARY.json'),
        no_retry=True,actor_identities=identities,physical_steps='Unknown beyond durable records; no zero-physics claim')


def _launch_prefix(directory,count):
    """An empty stage log consumes its stage, before native run_started exists."""
    directory=Path(directory)
    names={p.name for p in directory.glob('project_stage_*.log')}
    known={f'project_stage_{i:02d}.log' for i in range(count)}
    require(names<=known,'Unknown native launch stage')
    indices=[i for i in range(count) if (directory/f'project_stage_{i:02d}.log').exists()]
    require(indices==list(range(len(indices))) and all(not (directory/f'project_stage_{i:02d}.log').is_symlink()
        for i in indices),'Native stage launch logs must form one unambiguous prefix')
    return indices


def _frozen_inventory(directory,expected):
    directory=Path(directory)
    actual={str(p.relative_to(directory)):p for p in directory.rglob('*')}
    require(set(actual)==set(expected),'Consumed original12 wave inventory changed')
    for relative,row in expected.items():
        path=actual[relative];require(not path.is_symlink(),'Consumed12 symlink is unsafe')
        if row.get('kind')=='directory':
            require(path.is_dir() and path.stat().st_mtime_ns==row['mtime_ns'],'Consumed12 directory identity changed')
        else:
            stat=path.stat();require(path.is_file() and stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'],
                'Consumed12 file identity changed')
            if 'sha256' in row:require(sha(path)==row['sha256'],'Consumed12 bounded metadata changed')


def fragment_execution_inventory(root,audit,reg,bindings):
    """Full64 denominator plus exact, outcome-independent prefix/suffix attempts."""
    test=root/'evaluation/test';owner=root/'owners/test_fragment13'
    census=reg.get('census',{});suffix=reg.get('eligible_indices')
    require(isinstance(suffix,list) and bool(suffix) and all(type(i) is int for i in suffix)
        and suffix==list(range(suffix[0],64)) and 0<suffix[0]<64
        and suffix==census.get('eligible_indices'),'Exact nonempty whole-wave suffix required')
    prefix=list(range(suffix[0]));rows=census.get('rows',[])
    require(census.get('lifecycle_only') is True and census.get('outcome_values_interpreted') is False
        and census.get('registered_waves')==64 and census.get('registered_opportunities')==256
        and census.get('consumed_waves')==len(prefix)
        and len(rows)==64 and [r.get('index') for r in rows]==list(range(64)),
        'Original outcome-independent64 lifecycle census differs')
    require(_launch_prefix(test,64)==prefix,'Original12 attempted prefix changed or unknown dispatch exists')
    require(_launch_prefix(owner,len(suffix))==list(range(len(suffix))),'Admission14 suffix not fully dispatched once')
    known={f'wave{i:02d}' for i in range(64)}
    require({p.name for p in test.iterdir() if p.name.startswith('wave')}==known,'Unknown/missing original wave')
    dispatch=owner/'dispatches'
    require(not dispatch.is_symlink() and dispatch.is_dir()
        and {p.name for p in dispatch.iterdir()}=={f'wave{i:02d}.json' for i in suffix},
        'Exact fragment13 once-only suffix fences required; duplicate/unknown dispatch refused')
    freeze=_pinned(test/'registration.json',bindings)
    require(freeze.get('n')==64 and freeze.get('cohort')=='test','Original64 registration required')
    terminals=[];unfinished=[]
    for i,row in enumerate(rows):
        wave=test/f'wave{i:02d}'
        require(row.get('wave')==str(wave) and row.get('consumed') is (i in prefix)
            and row.get('eligible_for_dispatch') is (i in suffix)
            and not wave.is_symlink() and wave.is_dir(),'Original census wave partition differs')
        require(all(not (wave/name).is_symlink() for name in ('plan.json','session.json','run_started.json','process_exit_requested.json','results.json')),
            'Original wave metadata/terminal receipts cannot be symlinks')
        require(bindings['small_file_sha256'].get(str(wave/'plan.json'))==freeze['wave_plan_sha256'][str(wave)],
            'Original frozen wave plan changed')
        require(reg['original_wave_session_sha256'].get(str(wave/'session.json'))==
            bindings['small_file_sha256'].get(str(wave/'session.json')),'Original frozen wave session changed')
        if i in prefix:
            _frozen_inventory(wave,row['inventory'])
        else:
            require(set(row['inventory'])=={'plan.json','session.json'},'Pre13 suffix was not pristine')
            fence=_pinned(dispatch/f'wave{i:02d}.json',bindings)
            require(fence.get('index')==i and fence.get('wave')==str(wave) and fence.get('dispatch_attempt')==1
                and fence.get('rerun') is False and fence.get('registration_sha256')==sha(audit/'TEST_FRAGMENT13_REGISTRATION.json')
                and fence.get('plan_sha256')==freeze['wave_plan_sha256'][str(wave)]
                and fence.get('admission14_registration_sha256')==sha(audit/'TEST_ADMISSION14_REGISTRATION.json')
                and type(fence.get('pid')) is int and fence['pid']>0,'Fragment14 exclusive dispatch proof differs')
            cooperative_metadata_gate(root,audit,wave,fence.get('cooperative_metadata'),bindings)
        launched=(wave/'run_started.json').is_file()
        done=(wave/'results.json').is_file() and (wave/'process_exit_requested.json').is_file()
        require(not done or launched,'Terminal wave missing actual native launch receipt')
        if launched:
            launch=_pinned(wave/'run_started.json',bindings)
            if i in suffix:require(launch.get('pid')==fence['pid'] and launch.get('plan_sha256')==fence['plan_sha256']
                and launch.get('utc',-1)>=fence['utc'],'Fragment14 native launch/fence PID/time/plan differs')
        if done:
            end=_pinned(wave/'process_exit_requested.json',bindings)
            if i in suffix:require(end.get('pid')==fence['pid'] and end.get('utc',-1)>=launch['utc'],
                'Fragment14 native terminal/fence PID/time differs')
        if i in prefix:
            require(done is row.get('terminal_receipts_present'),'Consumed12 terminal boundary changed')
        else:require(done,'Successful admission14 owner requires terminal receipt for every suffix wave')
        (terminals if done else unfinished).append(i)
    require(any(i in prefix for i in unfinished),'Truthful failed12 consumed wave must remain software unfinished')
    return dict(cohort='test',registered_waves=64,registered_episodes=256,
        original12_attempted_indices=prefix,fragment14_attempted_indices=suffix,
        original13_pre_native_additional_attempts=0,native_owner_path=str(owner),
        attempted_indices=prefix+suffix,attempted_waves=64,attempted_episodes=256,
        terminal_indices=terminals,terminal_waves=len(terminals),terminal_episodes=4*len(terminals),
        software_unfinished_indices=unfinished,software_unfinished_waves=len(unfinished),software_unfinished_episodes=4*len(unfinished),
        unexecuted_indices=[],unexecuted_waves=0,unexecuted_episodes=0,duplicate_attempts=0,
        denominator_does_not_certify_attempts=True,attempt_consumption_rule='Original durable stage log or fragment exclusive dispatch fence consumes all4 opportunities, including pre-native failure.')


def fragment14_gate(root,audit,binding,bindings,failed12,admission14,failed13):
    path=audit/'TEST_FRAGMENT13_REGISTRATION.json';reg=_pinned(path,bindings)
    review=_pinned(audit/'ROOT_TEST_FRAGMENT13_REVIEW.json',bindings)
    test=root/'evaluation/test';owner=root/'owners/test_fragment13';driver=audit/'driver_test_admission14'
    require(reg.get('version')=='test_missing_whole_suffix_fragment13_v1' and reg.get('root')==str(root)
        and reg.get('owner')==str(owner) and reg.get('test')==str(test)
        and reg.get('deadline')==binding['original_deadline'] and reg.get('startup_timeout_seconds')==1800
        and reg.get('scientific_changes')==reg.get('optimizer_updates')==reg.get('collection_opportunities')==0
        and reg.get('external_process_presence_is_not_a_veto') is True and reg.get('original12_remains_failed') is True,
        'Exact finite missing-only fragment13 scope required')
    require(review.get('status')=='source_and_actual_missing_suffix_review_passed_inactive'
        and review.get('registration_sha256')==sha(path),'Root actual missing-suffix source review required')
    expected={str(audit/name) for name in RUNTIME13_SOURCES}
    require(set(reg.get('orchestration_sha256',{}))==expected,'Exact13 fragment orchestration source population required')
    for name,digest in reg['orchestration_sha256'].items():
        require(bindings['small_file_sha256'].get(str(Path(name).resolve()))==digest,'Fragment13 runtime source changed')
    exact={audit/'ROOT_TEST_FRAGMENT13_BOUNDARY.json':reg['boundary_sha256'],
        audit/'test_fragment13_job.json':reg['job_sha256'],test/'registration.json':reg['original_registration_sha256'],
        test/'session.json':reg['original_owner_session_sha256'],owner/'session.json':reg['fragment_owner_session_sha256'],
        root/'cohorts.json':reg['original_cohorts_sha256'],test/'project_job_started.json':reg['census']['original_started_sha256'],
        test/'project_status.json':reg['census']['original_status_sha256']}
    for filename,digest in exact.items():require(bindings['small_file_sha256'].get(str(filename))==digest,'Fragment13 original/boundary binding differs')
    original=_pinned(test/'registration.json',bindings)
    originaljob=_pinned(Path(original['job']),bindings)
    require(original.get('job_sha256')==reg['census'].get('original_job_sha256')==sha(original['job']),
        'Original64 job SHA differs from frozen registration/census')
    originalstarted=_pinned(test/'project_job_started.json',bindings)
    require(originalstarted.get('job')==originaljob and originaljob.get('resource_root')==str(test)
        and originaljob.get('operation')=='evaluate' and originaljob.get('stages')==[
            dict(entry='paired_evaluate',args=['worker','--output',str(test/f'wave{i:02d}')]) for i in range(64)],
        'Original64 native job/stage population differs')
    originalcommands=originalstarted.get('commands',[])
    require(len(originalcommands)==64 and all(isinstance(command,list) and command[1:]==['-m',
        'outputs.diagnostics.tight_paired_dagger_20261122_01.eval_startup12',*stage['args']]
        and Path(command[0]).name.startswith('python') for command,stage in zip(originalcommands,originaljob['stages'])),
        'Original12 exact64 dispatch commands differ')
    require(reg.get('frozen_pairs')==original.get('pairs') and reg.get('source_sha256')==binding['production']['after_sha256'],
        'Fragment13 frozen original4pairs/source295 differs')
    job=_pinned(audit/'test_fragment13_job.json',bindings)
    require(reg.get('job')==str(audit/'test_fragment13_job.json') and job.get('operation')=='evaluate'
        and job.get('resource_root')==str(owner),'Fragment13 native job differs')
    suffix=reg['eligible_indices']
    stages=[dict(entry='paired_evaluate_fragment13',args=['worker','--output',str(test/f'wave{i:02d}')]) for i in suffix]
    require(job.get('stages')==stages,'Fragment13 exact original suffix native stages required')
    begin,done,end=[_pinned(owner/name,bindings) for name in ('project_job_started.json','project_complete.json','project_exit.json')]
    require(not (owner/'project_failure.json').exists() and begin.get('job')==job and done.get('stages')==len(suffix)
        and end.get('children_reaped') is True,'Fragment13 owner failed/incomplete/unreaped')
    commands=begin.get('commands',[])
    require(len(commands)==len(stages) and all(isinstance(c,list) and c[1:]==['-m',
        'outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_admission14',*s['args']]
        and Path(c[0]).name.startswith('python') for c,s in zip(commands,stages)),
        'Fragment13 exact one-worker command shape differs')
    activated=_pinned(driver/'registration.json',bindings)
    require(activated==dict(admission14,activated=True,activated_utc=activated.get('activated_utc'))
        and activated['activated_utc']>=failed13['end_utc'] and begin['utc']>=activated['activated_utc'],
        'Fragment13 overlaps unreaped original12 or activated registry differs')
    closed=_pinned(driver/'closure.json',bindings);exited=_pinned(driver/'driver_exit.json',bindings)
    require(not (driver/'failure.json').exists()
        and closed.get('status')=='original_missing_suffix_and_audits_complete'
        and closed.get('original13_driver_remains_failed') is True and closed.get('native_owner13_used_once') is True
        and exited.get('returned_after_owned_helpers') is True
        and end['utc']<=closed['utc']<=exited['utc']<=binding['original_deadline'],
        'Actual fragment14 closure/helper/deadline differs')
    native_admission=_pinned(driver/'evaluate_fragment14_capacity_admission.json',bindings)
    require(native_admission.get('capacity',{}).get('ready') is True
        and native_admission.get('admission')==ADMISSION14,'Fragment14 actual capacity admission differs')
    native_exit=_pinned(root/'cycle_logs_admission14/evaluate_fragment14_exit.json',bindings)
    require(native_exit.get('returncode')==0
        and activated['activated_utc']<=native_admission['utc']<=begin['utc']<=end['utc']<=native_exit['utc']<=closed['utc'],
        'Fragment14 native evaluate stage exit/admission chronology differs')
    tails=reg.get('tail_stages',[])
    require([r.get('name') for r in tails]==['summarize_test','audit_collection_cuda_noise_recovery05'],
        'Exact original still-unattempted tail helper population required')
    resolved={'summarize_development':str(root/'cycle_logs_recovery12/summarize_development_exit.json'),
        'prepare_test':str(root/'cycle_logs_recovery12/prepare_test_exit.json'),
        'evaluate_test':str(root/'cycle_logs_recovery12/evaluate_test_exit.json'),
        'evaluate_fragment13_failed':str(root/'cycle_logs_fragment13/evaluate_fragment13_exit.json'),
        'evaluate_fragment14':str(root/'cycle_logs_admission14/evaluate_fragment14_exit.json')}
    expected_tail_args={
        'summarize_test':['-m','tools.tight_paired_evaluate','summarize','--output',str(test),'--reconstruct-cuda-noise'],
        'audit_collection_cuda_noise_recovery05':[str(audit/'audit_collection_recovery05.py'),'noise',
            '--collection',str(root/'collection'),'--output',str(audit/'collection_noise_recovery05_actual01'),
            '--reconstruct-cuda-noise']}
    for row in tails:
        require(type(row.get('skip_closed')) is bool and row.get('args')==expected_tail_args[row['name']],
            'Exact original CUDA helper command/skip identity differs')
        name=row['name'];directory='cycle_logs_recovery12' if row['skip_closed'] else 'cycle_logs_admission14'
        receipt=root/directory/(name+'_exit.json');resolved[name]=str(receipt)
        require(_pinned(receipt,bindings).get('returncode')==0,'Tail helper not really closed')
        other=root/('cycle_logs_admission14' if row['skip_closed'] else 'cycle_logs_recovery12')/(name+'_exit.json')
        otherlog=other.with_name(name+'.log')
        require(not other.exists() and not otherlog.exists(),'Tail helper retry/duplicate attempted under both12 and13')
        if row['skip_closed']:
            require(row.get('original_exit_sha256')==sha(receipt),'Inherited rc0 helper SHA changed')
            require(_pinned(audit/'driver_eval_recovery12'/(name+'_capacity_admission.json'),bindings).get('capacity',{}).get('ready') is True,
                'Inherited rc0 helper actual admission missing')
        else:
            admission=_pinned(driver/(name+'_capacity_admission.json'),bindings)
            require(admission.get('capacity',{}).get('ready') is True and admission.get('admission')==ADMISSION14
                and native_exit['utc']<=admission['utc']<=_pinned(receipt,bindings)['utc']<=closed['utc'],
                'Tail helper actual capacity admission/chronology differs')
        require(not (root/'cycle_logs_fragment13'/(name+'_exit.json')).exists()
            and not (root/'cycle_logs_fragment13'/(name+'.log')).exists(),
            'Failed13 pre-native driver cannot have attempted tail helpers')
    require(binding.get('stage_exit_paths')==resolved,'Exact actual12/13 resolved stage receipts differ')
    dispatch14=_pinned(driver/'native_owner_dispatch14.json',bindings)
    require(dispatch14.get('job_sha256')==sha(audit/'test_fragment13_job.json')
        and dispatch14.get('dispatch_attempt')==1 and dispatch14.get('rerun') is False
        and activated['activated_utc']<=dispatch14['utc']<=begin['utc'],
        'Actual14 exclusive native owner dispatch differs')
    execution=fragment_execution_inventory(root,audit,reg,bindings)
    return dict(registration_sha256=sha(path),closed=True,original12_failed=True,original13_driver_failed=True,
        execution=execution,tail_stage_exit_paths={r['name']:resolved[r['name']] for r in tails})


def watcher_closed(audit):
    audit=Path(audit).resolve();binding=read_small(audit/BINDING_NAME)
    bindings={'small_file_sha256':binding['small_file_sha256']};root=Path(binding['root'])
    require(Path(binding['watcher_exit_path'])==audit/'watch_shared_capacity14/resource_watch_v4_exit.json','Admission14 watcher path differs')
    proofs={}
    for name,receipt,current,count in (
        ('watch_shared_capacity14','WATCHER14_ACTUAL_CLOSED.json',True,None),
        ('watch_shared_capacity13','WATCHER13_ACTUAL_CLOSED.json',False,None),
        ('watch_shared_capacity12','WATCHER12_ACTUAL_CLOSED.json',False,None),
        ('watch_shared_capacity10','OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',False,None),
        ('watch_shared_capacity09','OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',False,33)):
        proof=watcher_restoration_proof(audit,name,bindings,historical_receipt=audit/receipt,
            verify_current=current,expected_count=count,run_root=root)
        actual=_pinned(audit/receipt,bindings)
        require(actual.get('source_pid_reaped') is True and actual.get('conflicts',[])==[]
            and service_is_closed(actual.get('service'),require_success=name!='watch_shared_capacity09'),'Actual watcher service/reaping differs: '+name)
        if name in ('watch_shared_capacity12','watch_shared_capacity13','watch_shared_capacity14'):
            marker=root/('STOP_RESOURCE_WATCH14' if name.endswith('14') else 'STOP_RESOURCE_WATCH13' if name.endswith('13') else 'STOP_RESOURCE_WATCH')
            require(actual.get('actual_restoration_identities')==proof['backup_count']
                and actual.get('stop_marker_preserved')==str(marker) and marker.is_file() and not marker.is_symlink(),
                'Actual watcher dynamic restoration/stop identity differs')
        proofs[name]=proof
    return dict(path=binding['watcher_exit_path'],sha256=sha(binding['watcher_exit_path']),
        actual_service_reaping_receipt_sha256=sha(audit/'WATCHER14_ACTUAL_CLOSED.json'),
        current_restoration_audit=proofs['watch_shared_capacity14'],historical_restoration_audits={k:v for k,v in proofs.items() if not k.endswith('14')})


RUNTIME14_SOURCES=('run_test_admission14.py','shared_capacity14.py','test_fragment13.py','shared_capacity13.py',
    'run_main_recovery05.py','shared_capacity06.py','resource_watch_v4.py','driver_support.py','driver_support_v4.py')
ADMISSION14=dict(gpu_resident_mib=11264,ram_resident_bytes=12*2**30,
    gpu_reserve_mib=4096,ram_reserve_bytes=10*2**30,
    cooperative_early_gpu_margin_mib=1024,cooperative_early_ram_margin_bytes=2*2**30)


def admission14_gate(root,audit,binding,bindings,failed12):
    """Retain the failed13 driver; root certifies zero native attempts before14."""
    path=audit/'TEST_ADMISSION14_REGISTRATION.json';reg=_pinned(path,bindings)
    review=_pinned(audit/'ROOT_TEST_ADMISSION14_REVIEW.json',bindings)
    original=_pinned(audit/'TEST_FRAGMENT13_REGISTRATION.json',bindings)
    boundary_path=audit/'ROOT_TEST_ADMISSION14_BOUNDARY.json';boundary=_pinned(boundary_path,bindings)
    require(reg.get('version')=='test_admission14_pre_native_only_v1' and reg.get('deadline')==binding['original_deadline']
        and reg.get('owner')==str(root/'owners/test_fragment13') and reg.get('job')==str(audit/'test_fragment13_job.json')
        and reg.get('admission')==ADMISSION14 and reg.get('scientific_changes')==reg.get('training_updates')==0
        and reg.get('original13_driver_remains_failed') is True and reg.get('native_owner13_reused_once') is True
        and reg.get('no_outcome_retry') is True
        and reg.get('schedule_metadata_policy')=='Only live watcher14-owned exact paused schedule with exact original backup binding; no other runtime metadata',
        'Exact source-only admission14 amendment identity required')
    require(review.get('status')=='source_and_actual_pre_native14_review_passed_inactive'
        and review.get('registration_sha256')==sha(path),'Root actual14 source/pre-native review required')
    require(reg.get('original13_registry_sha256')==sha(audit/'TEST_FRAGMENT13_REGISTRATION.json')
        and reg.get('original13_job_sha256')==sha(audit/'test_fragment13_job.json')
        and reg.get('boundary_sha256')==sha(boundary_path)
        and reg.get('original_test_registration_sha256')==sha(root/'evaluation/test/registration.json'),
        'Original13 registry/job/boundary/test freeze changed')
    for key in ('eligible_indices','tail_stages','source_sha256','frozen_pairs','original_cohorts_sha256'):
        require(reg.get(key)==original.get(key),'Admission14 changed original13 science: '+key)
    require(set(reg.get('orchestration_sha256',{}))=={str(audit/name) for name in RUNTIME14_SOURCES},
        'Exact9 runtime14 amendment source population required')
    for filename,digest in reg['orchestration_sha256'].items():
        require(bindings['small_file_sha256'].get(str(Path(filename).resolve()))==digest,'Runtime14 amendment source changed')
    census=reg.get('native_zero_attempt_census',{})
    expected=dict(native_owner13_started=False,native_stage_logs=0,dispatch_fences=0,
        eligible_indices=original['eligible_indices'],native_attempts=0,lifecycle_only=True,outcome_values_interpreted=False)
    require(census==expected,'Actual14 prerequisite must certify zero prior native13 attempts')
    require(boundary.get('status')=='actual_fragment13_failed_before_native_owner'
        and boundary.get('deadline')==binding['original_deadline']
        and boundary.get('original13_driver_reaped') is True and boundary.get('no_external_signals') is True,
        'Root actual failed13 pre-native closed boundary required')
    driver=audit/'driver_test_fragment13'
    required=[driver/'failure.json',driver/'driver_exit.json',
        root/'cycle_logs_fragment13/evaluate_fragment13.log',root/'cycle_logs_fragment13/evaluate_fragment13_exit.json',
        audit/'WATCHER13_ACTUAL_CLOSED.json']
    artifacts=boundary.get('artifact_sha256',{})
    require(reg.get('original13_driver_artifacts')==artifacts and set(artifacts)>={str(p) for p in required},
        'Exact actual13 failure/watcher artifacts required')
    all_pins={**bindings['small_file_sha256'],**bindings.get('local_retained_log_sha256',{})}
    for filename,digest in artifacts.items():
        require(all_pins.get(str(Path(filename).resolve()))==digest,'Failed13 boundary artifact changed')
    failure=_pinned(driver/'failure.json',bindings);end=_pinned(driver/'driver_exit.json',bindings)
    activated=_pinned(driver/'registration.json',bindings)
    require(activated==dict(original,activated=True,activated_utc=activated.get('activated_utc'))
        and bool(failure.get('error')) and 'Original suffix changed' in failure['error']
        and end.get('returned_after_owned_helpers') is True and not (driver/'closure.json').exists(),
        'Original13 pre-native failed driver must remain truthful')
    require(type(_pinned(root/'cycle_logs_fragment13/evaluate_fragment13_exit.json',bindings).get('returncode')) is int
        and _pinned(root/'cycle_logs_fragment13/evaluate_fragment13_exit.json',bindings)['returncode']!=0,
        'Original13 pre-native evaluate helper must retain nonzero exit')
    identities=boundary.get('closed_identities',[])
    watch=_pinned(audit/'WATCHER13_ACTUAL_CLOSED.json',bindings)
    require(isinstance(identities,list) and bool(identities) and all(
        type(row.get('pid')) is int and row['pid']>0 and type(row.get('created')) in (int,float)
        and math.isfinite(row['created']) and row['created']>0 for row in identities)
        and {end.get('pid'),watch.get('pid')}<={row['pid'] for row in identities},
        'Actual13 driver and watcher finite positive closed identities required')
    actor=next(row for row in identities if row['pid']==end.get('pid'))
    require(actor is not None
        and failed12['driver_exit_utc']<=activated['activated_utc']<=end['utc']<=binding['original_deadline'],
        'Actual failed13 driver PID/birth/closed chronology differs')
    capacity=_pinned(driver/'evaluate_fragment13_capacity_admission.json',bindings)
    require(capacity.get('capacity',{}).get('ready') is True,'Failed13 pre-native actual admission receipt missing')
    return reg,dict(failed=True,native_owner_started=False,additional_wave_attempts=0,
        start_utc=activated['activated_utc'],end_utc=end['utc'],wall_seconds=end['utc']-activated['activated_utc'],
        driver_pid=end['pid'],driver_created=actor['created'],boundary_sha256=sha(boundary_path),
        interval_scope='Actual failed13 driver activation-to-owned-helper-exit wall interval; no native13 interval fabricated.')


def cooperative_metadata_gate(root,audit,wave,metadata,bindings):
    """Allow exactly the runtime14-proven watcher pause; no payload exception."""
    if metadata is None:return
    require(isinstance(metadata,dict) and set(metadata)=={'path','sha256','watcher_pid','backup','backup_sha256'}
        and metadata['path']==str(wave/'concurrency.json'),'Unexpected cooperative metadata identity')
    exit_path=audit/'watch_shared_capacity14/resource_watch_v4_exit.json'
    exited=_pinned(exit_path,bindings);pid=exited['pid']
    require(metadata['watcher_pid']==pid,'Cooperative pause belongs to another watcher')
    backup=Path(metadata['backup'])
    require(backup.parent.parent==audit/'watch_shared_capacity14'
        and backup.parent.name.endswith('_'+str(pid))
        and backup.name==hashlib.sha256(str(wave/'concurrency.json').encode()).hexdigest()+'.json'
        and metadata['backup_sha256']==bindings['small_file_sha256'].get(str(backup.resolve())),
        'Cooperative pause backup identity differs')
    row=_pinned(backup,bindings)
    require(row.get('path')==str(wave/'concurrency.json') and row.get('existed') is False
        and row.get('base64') is None and row.get('sha256') is None,'Cooperative suffix pause must derive from original absence')
    paused=dict(pause_groups=[None,'A','B','G00','G10','G01','G11'],paired_resource_watch_v4_pid=pid)
    require(metadata['sha256']==hashlib.sha256(json.dumps(paused,indent=2,allow_nan=False).encode()).hexdigest(),
        'Cooperative pause bytes differ from exact registered watcher14 schedule')
