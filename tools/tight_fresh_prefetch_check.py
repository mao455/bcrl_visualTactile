"""Single train-only CPU check of B2 read-ahead data/order/cleanup; no new simulation."""
import argparse,json,time,traceback,hashlib
from pathlib import Path
import torch
from visuotactile.frozen_visual_io import prefetch_profile
from visuotactile.native_data import Episode
from visuotactile.trajectory_prefetch import TrajectoryPrefetch


def run(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(2)
    state=root/'A/state_data';data=json.loads((state/'data.json').read_text())
    train=data['splits']['train'];cache=json.loads((root/'A/cache_index.json').read_text())
    assert cache['complete'] and json.loads((root/'A/training/finished.json').read_text())['complete']
    # Largest two train shards exercise the memory bound; no held-out images.
    ids=sorted(train,key=lambda i:(state/'perception'/(i+'.pt')).stat().st_size,reverse=True)[:2]
    groups=[ids,ids[::-1]];profile=prefetch_profile()
    factory=lambda i:Episode(state,i,'I-HistoryDP',frozen_visual_io=profile)
    initial_rng=torch.get_rng_state().clone();loader=TrajectoryPrefetch(groups,factory,state,profile)
    rows=[]
    try:
        for index,identities in enumerate(groups):
            episodes=loader.take(index)
            for identity,episode in zip(identities,episodes):
                reference=Episode(state,identity,'I-HistoryDP')
                assert all(t.device.type=='cpu' for d in [episode.record,episode.perception] for t in d.values())
                size=0
                for key in ('record','perception'):
                    a,b=getattr(episode,key),getattr(reference,key)
                    assert a.keys()==b.keys()
                    for name,x in a.items():
                        y=b[name];assert x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride()
                        assert torch.equal(x.view(torch.uint8),y.view(torch.uint8)),(index,identity,key,name)
                        size+=x.numel()*x.element_size()
                rows.append(dict(batch=index,identity=identity,tensor_bytes=size,bytewise_equal=True))
                del reference
            loader.finish();assert not episodes
        loader.close();assert loader.current is None and loader.future is None
        assert torch.equal(initial_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
        # Tiny lifecycle cases use no source payload, model, clock wait or GPU.
        class Dummy:
            def __init__(self):
                self.record={};self.perception={};self.frozen_visual_io_receipt={'loaded_storage_bytes':0}
        low=TrajectoryPrefetch(groups,lambda i:Dummy(),state,profile,memory=lambda:0)
        for j in range(2):low.take(j);low.finish()
        low.close();assert low.summary()['admission_fallbacks']==1
        early=TrajectoryPrefetch(groups,lambda i:Dummy(),state,profile)
        early.take(0);early.close();assert early.current is None and early.future is None
        result=dict(complete=True,seconds=time.time()-started,profile=profile,rows=rows,
            prefetch=loader.summary(),low_memory_fallback_passed=True,early_close_passed=True,
            torch_cpu_rng_unchanged=True,cuda_initialized=False,models_opened=0,optimizer_updates=0,
            source=str(root),source_data_sha256=hashlib.sha256((state/'data.json').read_bytes()).hexdigest(),
            code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('visuotactile/trajectory_prefetch.py'),Path('visuotactile/frozen_visual_io.py'),Path('tools/tight_fresh_train.py')]},
            limitation='Only immutable loading changed. Actual tensors/order/lifecycle checked; no model update comparison or formal-cycle speed claim. Largest two shards repeated; OS cache is warm.')
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['complete','seconds','prefetch','low_memory_fallback_passed','early_close_passed']}))
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),rows=rows),indent=2)+'\n');raise
    finally:loader.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.output.resolve())
