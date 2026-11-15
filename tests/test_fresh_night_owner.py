import json
from tools import tight_fresh_night_owner as night


def test_small_cohort_cannot_adopt_on_loss_or_equal_holds(tmp_path, monkeypatch):
    def behavior(directory, role=None, horizon=None):
        new=directory==tmp_path/'development'
        rows=[]
        for group in ([role] if role else 'AB'):
            for seed in range(4):
                held=seed<2 if group=='A' or not new else True
                rows.append(dict(role=group,seed=seed,legal=True,hold60=dict(sustained=held)))
        return dict(rows=rows)
    monkeypatch.setattr(night,'behavior',behavior)
    d=night.source_decision(tmp_path,dict(A='oldA',B='oldB'),dict(A='evalA',B='evalB'))
    assert not d['A']['use_candidate'] and d['B']['use_candidate']
    assert d['B']['source_hold60']==2 and d['B']['candidate_hold60']==4


def test_only_transient_prephysical_stage_can_retry(tmp_path):
    candidate=tmp_path/'round';candidate.mkdir();ledger=tmp_path/'round_stages';ledger.mkdir()
    (ledger/'failure.json').write_text(json.dumps(dict(error='RAM reserve')))
    assert night.recoverable_training_failure(candidate)
    (ledger/'frontend_A.log').write_text('nonfinite gradient')
    assert not night.recoverable_training_failure(candidate)
    (ledger/'frontend_A.log').unlink();(candidate/'development').mkdir()
    assert not night.recoverable_training_failure(candidate)


def test_hard_review_includes_registered_recovered_families(tmp_path,monkeypatch):
    from tools import tight_fresh_grasp_review as review
    from tools import tight_fresh_batch_run as batch
    from visuotactile import training_expansion as expansion
    root=tmp_path/'root';(root/'A/state_data').mkdir(parents=True)
    (root/'A/state_data/data.json').write_text(json.dumps(dict(training_expansion_registries=[{}])))
    collection=tmp_path/'collection';collection.mkdir();(collection/'plan.json').write_text('{"cases":[]}')
    monkeypatch.setattr(expansion,'read_registry',lambda _:dict(families={'train_family':dict(source_identity='train',seed=12,narrow_reset={})}))
    def prepare(root,output,**kw):output.mkdir();return {}
    monkeypatch.setattr(batch,'prepare',prepare)
    cfg=dict(lanes=4,development_count_per_role=2,development_seed=100,execution_backend={},vision_microbatch=4,
        queued_gpu=True,hard_families=['train_family'],grasp_evaluation_ticks=300)
    plan=review.prepare_cohort(root,collection,tmp_path/'review',cfg,'hard')
    assert {c['role'] for c in plan['cases']}=={'A','B'}
    assert all(c['spec']['split']=='train' and c['spec']['seed']==12 for c in plan['cases'])
