import torch
from visuotactile.checkpointing import save,promote_snapshot


def test_best_retains_previous_epoch_when_last_is_atomically_replaced(tmp_path):
    last,best=tmp_path/'last.pt',tmp_path/'best.pt'
    state=dict(model=torch.arange(9),optimizer=dict(moment=torch.ones(3)),rng=torch.get_rng_state(),epoch=1)
    save(last,state);promote_snapshot(last,best)
    assert last.stat().st_ino==best.stat().st_ino
    before=best.read_bytes();save(last,dict(state,epoch=2,model=torch.zeros(9)))
    assert best.read_bytes()==before and last.stat().st_ino!=best.stat().st_ino
    assert torch.load(best,weights_only=False)['epoch']==1
    promote_snapshot(last,best)
    assert best.read_bytes()==last.read_bytes() and best.stat().st_ino==last.stat().st_ino
    restored=torch.load(best,weights_only=False)
    assert restored['epoch']==2 and torch.equal(restored['optimizer']['moment'],state['optimizer']['moment'])
    assert torch.equal(restored['rng'],state['rng'])
    assert not list(tmp_path.glob('*.partial'))
