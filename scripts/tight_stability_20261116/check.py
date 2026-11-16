"""One focused real-record check for the read-only input-capture addition."""
import time
import torch
from scripts.tight_stability_20261116.common_expert import load_model
from scripts.tight_stability_20261116.capture_eval import cpu_tree,slice_tree,cat_tree
from scripts.tight_stability_20261116.analyze_existing import OLD,OUT,read
from visuotactile.native_data import residual_batch
from visuotactile.precision import state_fp32
from visuotactile.training_resources import atomic_json

def main():
    torch.set_num_threads(2);state_fp32();began=time.time()
    p=OLD/'B/backend/B/training/best.pt';m,s=load_model(p)
    data=read(OLD/'B/state_data/data.json');identity=data['normalization_train_ids'][0]
    r=torch.load(OLD/'B/backend/B/nominal'/(identity+'.pt'),map_location='cpu',mmap=True,weights_only=False)
    x=residual_batch([r],0,32,'cuda');captured=[]
    with torch.no_grad():
        raw,state=m(x)
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        hook=m.register_forward_pre_hook(lambda module,args:captured.append(cpu_tree(args[0])))
        raw2,state2=m(x);hook.remove()
        assert torch.equal(raw,raw2)
        y=slice_tree(captured[0],0)
        assert set(y)==m.input_keys
        # Capture's time concatenation retains all 120 Hz tactile samples.
        def part(v,a,b,key=None):
            if isinstance(v,dict):return {k:part(w,a,b,k) for k,w in v.items()}
            return v[a*2:b*2] if len(v)==64 else v[a:b]
        restored=cat_tree([part(y,0,12),part(y,12,32)])
        def equal(a,b):
            return all(equal(a[k],b[k]) for k in a) if isinstance(a,dict) else torch.equal(a,b)
        assert equal(y,restored)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    atomic_json(OUT/'capture_check.json',dict(complete=True,seconds=time.time()-began,real_record=identity,
        raw_output_exactly_equal=True,input_roundtrip_exactly_equal=True,updates=0,simulation_starts=0,
        scope='Actual B22 model and real first32 sensor/plan inputs; capture hook does not change forward output; tactile2x temporal concatenation exact. Runtime replay checks each actual execution separately.'))
if __name__=='__main__':main()
