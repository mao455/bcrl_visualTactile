"""Join independently accepted tick receipts to original physical diagnostic phases."""
import argparse
import collections
import csv
import gzip

import numpy as np

from tools.tight_adaptive_dagger import ROOT, PUBLIC, read, write
from tools.tight_frozen_comparison import table
from tools.tight_representation_probe import source
from visuotactile.geometry_sampling import STAGES


def export(round_index):
    public=PUBLIC/f'round{round_index:02d}'
    target=ROOT/f'round{round_index:02d}/learning'
    assert read(target/'exposure_acceptance.json')['passed']
    data=read(target/'A/state_data/data.json')
    original=set(data['normalization_train_ids'])
    latest=set(data['aggregation']['new_training_ids'])
    raw={identity:source(data['entries'][identity]) for identity in data['splits']['train']}
    groups=collections.defaultdict(list);totals=collections.Counter()
    with gzip.open(public/'selected_tick_exposure.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            identity=row['identity'];tick=int(row['tick']);count=int(row['optimizer_steps'])
            legal=row['legal']=='True'
            assert legal==bool(raw[identity]['legal'][tick])
            if not legal:
                assert count==0
                continue
            assert count>=50
            cohort='original30' if identity in original else 'this_round' if identity in latest else 'historical_correction'
            stage=STAGES[int(raw[identity]['phase'][tick])]
            for label in (stage,'all_legal'):
                groups[row['model'],cohort,label].append(count)
            totals[row['model']]+=1
    rows=[]
    for (model,cohort,stage),values in sorted(groups.items()):
        x=np.asarray(values)
        rows.append(dict(model=model,cohort=cohort,stage=stage,legal_ticks=len(x),
            minimum=int(x.min()),median=float(np.median(x)),mean=float(x.mean()),
            p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),maximum=int(x.max()),
            below50=int((x<50).sum())))
    expected=sum(int(r['legal'].sum()) for r in raw.values())
    assert len(totals)==4 and all(v==expected for v in totals.values())
    table(public/'selected_phase_exposure.csv',rows)
    write(public/'selected_phase_exposure.json',dict(rows=rows,legal_ticks_per_component=expected,
        interpretation='Distinct newly committed optimizer steps at the actual selected checkpoint, joined to recorded physical phase proxies; padding/prefix/gaps excluded. Phases are not a strict success funnel.',
        trajectory_population=len(raw),ancestors_counted=False,all_phases_passed=True))
    print(dict(round=round_index,trajectories=len(raw),legal_ticks=expected,rows=len(rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True)
    export(p.parse_args().round)
