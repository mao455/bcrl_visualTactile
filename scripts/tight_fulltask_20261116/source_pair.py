"""Finite matched-reset source A/B reference while independent rear training finishes."""
import time,traceback
from scripts.tight_fulltask_20261116 import run as driver
from scripts.tight_fulltask_20261116.run import ROOT,DIAG,read,atomic_json,DEADLINE
from scripts.tight_fulltask_20261116.train import eval_prepare
from pathlib import Path
orig=driver.atomic_json
def write(path,value):
    if Path(path)==ROOT/'status.json':path=ROOT/'status_source_pair.json'
    orig(path,value)
driver.atomic_json=write

def main():
    resource=read(ROOT/'shared_resources.json')
    assert time.time()<DEADLINE-180 and resource['gpu_free_mib']>10*1024 and resource['ram_available_bytes']>16*1024**3
    output=ROOT/'paired_sources';eval_prepare(output,read(DIAG/'candidates/plan.json')['pairs'])
    plan=read(output/'plan.json');plan['cases']=[c for c in plan['cases'] if c['spec']['seed']==28500100];plan['batch']['lanes']=2
    plan['evaluation_scope']=dict(protocol='Complete native termination; matched native reset RNG, source models only',seeds=[28500100],
        reason='Independent source reference with actual paired q0; updated rear training still running, never mislabeled as updated performance',
        concurrent_work='One Kit only; independent backend training/cache. Memory admission >10GiB VRAM/>16GiB RAM plus live ordinary reserves. Model files immutable.')
    plan['paired_native_reset']=dict(version='private_per_family_native_reset_v1',seed_offset=500000)
    atomic_json(output/'plan.json',plan)
    driver.execute('paired_sources',['scripts.tight_fulltask_20261116.paired_eval','--output',output])
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(ROOT/'source_pair_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
