"""Inactive recovery12 metadata adapter; original10 remains a truthful failed attempt.

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
BINDING_NAME='FINAL_RECOVERY12_BINDINGS.json'
REVIEW_NAME='ROOT_EVIDENCE_TOOLS_REVIEW12.json'
FINAL_TOOLS=('final_recovery12_common.py','extract_selected_ledgers_recovery12.py',
 'final_report_assembler_recovery12.py','final_compute_recovery12.py',
 'export_final_evidence_recovery12.py','test_final_evidence_recovery12.py')
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
            'Root review12 must pass')
    require(review.get('binding_sha256') == sha(binding_path), 'Root-reviewed final12 binding changed')
    require(binding.get('version') == 'paired_final_recovery12_bindings_v1'
        and Path(binding.get('root','')).resolve() == root,
        'Final12 binding root/version differs')
    reviewed = review.get('source_sha256',{})
    for name in FINAL_TOOLS:
        require(reviewed.get(name) == sha(audit/name), 'Root-reviewed final12 source differs: '+name)
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
        require(expected == digest, 'Final12 bound source/receipt changed: '+str(path))
        observed[str(path)] = digest
    return binding,dict(binding_sha256=sha(binding_path),review_sha256=sha(review_path),small_file_sha256=observed)


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
    root,audit=Path(root).resolve(),Path(audit).resolve()
    driver=audit/'driver_eval_recovery12'
    required=[audit/BINDING_NAME,audit/REVIEW_NAME,driver/'status.json',driver/'closure.json',
        driver/'driver_exit.json',root/'evaluation/test/registration.json',
        root/'cycle_logs_recovery12/summarize_test_exit.json']
    for owner in (root/'owners/backend_recovery10',root/'evaluation/test'):
        required.extend(owner/name for name in ('project_job_started.json','project_complete.json','project_exit.json'))
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
    status=_pinned(driver/'status.json',bindings);closed=_pinned(driver/'closure.json',bindings)
    exited=_pinned(driver/'driver_exit.json',bindings)
    require(not (driver/'failure.json').exists() and status.get('stage')=='remaining_evaluation_complete'
        and closed.get('status')=='original_sealed_test_and_remaining_summaries_complete'
        and closed.get('original10_remains_failed') is True and status.get('original10_remains_failed') is True
        and exited.get('returned_after_owned_helpers') is True
        and exited.get('utc',-1)>=closed.get('utc',0)
        and type(status.get('utc')) in (int,float) and math.isfinite(status['utc'])
        and status['utc']<=binding['original_deadline'],'Actual12 closure/helpers/deadline differs')
    activated=_pinned(driver/'registration.json',bindings)
    require(activated['activated_utc']>=_pinned(audit/'driver_backend_recovery10/driver_exit.json',bindings)['utc'],
        'Recovery12 overlaps unreaped original10 driver')
    require(not list(driver.glob('startup_timeout_wave*.json')) and not (driver/'startup_watchdog_error.json').exists(),
        'Recovery12 timeout/error remains unfinished, never certified closed')
    expected=binding.get('stage_exit_paths',{})
    require(set(expected)==set(REMAINING_STAGES),'Exact5 remaining12 stage exits required')
    for stage,path in expected.items():
        require(Path(path)==root/'cycle_logs_recovery12'/(stage+'_exit.json')
            and _pinned(Path(path),bindings).get('returncode')==0,'Actual12 stage not complete: '+stage)
    for stage in OLD_COMPLETED_STAGES:
        require(_pinned(root/'cycle_logs'/(stage+'_exit.json'),bindings).get('returncode')==0,'Prior completed stage changed')
    for stage in reg12['stages']:
        if stage['gpu']:
            require(_pinned(driver/(stage['name']+'_capacity_admission.json'),bindings).get('capacity',{}).get('ready') is True,
                'Recovery12 stage capacity admission missing')
    for owner in (root/'owners/backend_recovery10',root/'evaluation/test'):
        require(not (owner/'project_failure.json').exists(),'Completed backend10 or new test owner failed')
        began,done,end=[_pinned(owner/name,bindings) for name in ('project_job_started.json','project_complete.json','project_exit.json')]
        require(isinstance(began.get('commands'),list) and bool(began['commands'])
            and len(began['commands'])==len(began.get('job',{}).get('stages',[]))==done.get('stages')
            and Path(began['job'].get('resource_root','')).resolve()==owner and end.get('children_reaped') is True,
            'Completed owner stage population/cleanup differs')
        if owner==root/'evaluation/test':
            require(len(began['commands'])==done['stages']==64
                and began['utc']>=activated['activated_utc'] and end['utc']<=status['utc'],
                'Test exact64 native stages or activation/closure chronology differs')
            require(_pinned(audit/'presealed_guard_main_recovery10.json',bindings)['observed_utc']<=began['utc'],
                'Test began before actual selected-training guard')
        else:
            require(began['job']==_pinned(audit/'proposed_backend_recovery10.json',bindings)
                and began['utc']>=_pinned(root/'owners/backend_recovery09/project_exit.json',bindings)['utc'],
                'Backend10 owner overlaps unreaped09 or differs from job')
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
    execution={cohort:evaluation_execution_inventory(root,cohort,_pinned(root/'evaluation'/cohort/'registration.json',bindings),bindings)
        for cohort in ('development','test')}
    require(execution['development']['attempted_waves']==8 and execution['development']['terminal_waves']==7
        and execution['development']['software_unfinished_episodes']==4,'Consumed original development denominator differs')
    require(execution['test']['attempted_waves']==execution['test']['terminal_waves']==64
        and execution['test']['unexecuted_waves']==0,'Successful12 does not certify all64 durable physical opportunities')
    return dict(status='closed',version='paired_final_recovery12_lifecycle_v1',checkpoint_reads=0,
        failed06_preserved=True,failed07_preserved=True,failed08_preserved=True,failed09_preserved=True,
        failed10_preserved=True,preserved_dp_retrained=False,original_deadline=binding['original_deadline'],
        root_review=bindings,partial_dp_dedup=dedup,root10_inherited_payload_proof=proof,
        watcher_closure=watcher,production_amendment_paths=sorted(changed),wrapper_closure=pipeline,
        original10_failure=failed10,evaluation_execution=execution,
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
    require(watcher in ('watch_shared_capacity09','watch_shared_capacity10','watch_shared_capacity12'),'Unsupported restoration scope')
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
            and 'MainPID=0' in historical.get('service','') and 'ActiveState=inactive' in historical['service']
            and historical.get('exit_sha256')==sha(exit_path),'Historical root restoration/exit binding differs')
    return dict(watcher=watcher,backup_count=len(records),backup_directory=str(backups[0]),
        backup_file_sha256=backup_sources,exit_sha256=sha(exit_path),
        restoration_identities=records,all_identities_verified=True,current_schedule_bytes_checked=verify_current,
        historical_restoration_receipt_sha256=sha(historical_receipt) if historical_receipt else None,
        population_rule='Actual complete backup metadata population; current watcher count is never assumed.')


def watcher_closed(audit):
    audit=Path(audit).resolve();binding=read_small(audit/BINDING_NAME)
    path=_under(binding['watcher_exit_path'],audit.parents[2])
    require(path==audit/'watch_shared_capacity12/resource_watch_v4_exit.json','12 watcher path differs')
    bindings={'small_file_sha256':binding['small_file_sha256']}
    current=watcher_restoration_proof(audit,'watch_shared_capacity12',bindings,
        historical_receipt=audit/'WATCHER12_ACTUAL_CLOSED.json',verify_current=True,run_root=binding['root'])
    actual=_pinned(audit/'WATCHER12_ACTUAL_CLOSED.json',bindings)
    marker=Path(binding['root'])/'STOP_RESOURCE_WATCH'
    require(actual.get('conflicts')==[] and 'Result=success' in actual.get('service','')
        and actual.get('actual_restoration_identities')==current['backup_count']
        and actual.get('stop_marker_preserved')==str(marker) and marker.is_file() and not marker.is_symlink(),
        'Actual12 service/reaping/restoration population/retained stop differs')
    prior10=watcher_restoration_proof(audit,'watch_shared_capacity10',bindings,
        historical_receipt=audit/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',run_root=binding['root'])
    prior09=watcher_restoration_proof(audit,'watch_shared_capacity09',bindings,
        historical_receipt=audit/'OLD_WATCHER09_CLOSED_FOR_RECOVERY10.json',expected_count=33,run_root=binding['root'])
    historical=_pinned(audit/'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json',bindings)
    require(historical.get('conflicts')==[] and 'Result=success' in historical.get('service',''),
        'Original10 watcher service/restoration conflicts remain')
    return dict(path=str(path),sha256=sha(path),receipt=read_small(path),
        current_restoration_audit=current,previous10_restoration_audit=prior10,previous09_restoration_audit=prior09,
        actual_service_reaping_receipt_sha256=sha(audit/'WATCHER12_ACTUAL_CLOSED.json'))


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
    stages={s:str(root/'cycle_logs_recovery12'/(s+'_exit.json')) for s in REMAINING_STAGES}
    watcher=audit/'watch_shared_capacity12/resource_watch_v4_exit.json'
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
    paths.update(audit/name for name in ('EVALUATION_RECOVERY12_REGISTRATION.json','ROOT_EVALUATION_RECOVERY12_REVIEW.json',
        'OLD_WATCHER10_CLOSED_FOR_RECOVERY12.json','EVAL_STARTUP12_OWNED_TERMINATION.json',
        'WATCHER12_ACTUAL_CLOSED.json'))
    paths.update(Path(p) for p in reg12['orchestration_sha256'])
    paths.update(audit/'driver_eval_recovery12'/name for name in ('registration.json','status.json','closure.json','driver_exit.json'))
    paths.update(audit/'driver_eval_recovery12'/(stage['name']+'_capacity_admission.json') for stage in reg12['stages'] if stage['gpu'])
    paths.update((root/'cohorts.json',root/'evaluation/development/registration.json',
        root/'evaluation/development/project_job_started.json',root/'evaluation/development/project_failure.json',
        root/'evaluation/development/project_exit.json'))
    for owner in (root/'owners/backend_recovery10',root/'evaluation/test'):
        paths.update(owner/name for name in ('project_job_started.json','project_complete.json','project_exit.json'))
    for cohort,count in (('development',8),('test',64)):
        for index in range(count):
            wave=root/'evaluation'/cohort/f'wave{index:02d}'
            for name in ('run_started.json','process_exit_requested.json'):
                if (wave/name).is_file():paths.add(wave/name)
    preserved={}
    for role in 'AB':
        train=root/'learning'/role/'training';paths.update(train/name for name in ('deployment.json','finished.json'))
        checkpoint=train/'best.pt'
        if checkpoint.is_file():preserved[role]=dict(checkpoint_sha256=reg['preserved_dp'][role]['checkpoint_sha256'],
            checkpoint_stat_identity=identity(checkpoint.stat()),content_identity_source='Actual root10 SHA preflight plus unchanged supplemental stat; no payload read by builder')
        else:paths.add(checkpoint)
    for watch in ('watch_shared_capacity09','watch_shared_capacity10','watch_shared_capacity12'):
        paths.add(audit/watch/'resource_watch_v4_exit.json')
        paths.update((audit/watch).glob('resource_watch_v4_backups_*/*.json'))
    missing=sorted(str(p) for p in paths if not p.is_file())
    if missing:return dict(status='pending',missing=missing,checkpoint_reads=0)
    require(read_small(audit/'driver_eval_recovery12/status.json').get('stage')=='remaining_evaluation_complete','Real12 closure required before root binding')
    return dict(version='paired_final_recovery12_bindings_v1',root=str(root),original_deadline=reg['original_deadline'],
        small_file_sha256={str(p.resolve()):hashlib.sha256(stable_small(p)).hexdigest() for p in sorted(paths)},
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
    require(status.get('extractor_sha256') == sha(audit/'extract_selected_ledgers_recovery12.py')
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
