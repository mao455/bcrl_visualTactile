"""Compare recorded command errors on the same validation contact bins."""
import argparse
import csv
import math
from pathlib import Path
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def report(roots, output):
    cases=[];reference=None
    for root in roots:
        for role in ['A','B']:
            base=root/role/'backend'/role/'offline'
            path=base/'contact_phase_metrics.csv';result=read(base/'result.json')
            if not result.get('complete') or result.get('test_used') is not False:
                raise ValueError('Require complete non-test offline evidence')
            with path.open() as stream:
                rows=[r for r in csv.DictReader(stream) if r['split']=='validation']
            identities=sorted({r['identity'] for r in rows});phases={}
            population=sorted((r['identity'],r['contact_phase'],int(r['supervised_ticks'])) for r in rows)
            if not identities:raise ValueError('Empty validation population')
            if reference is None:reference=population
            if population!=reference:raise ValueError('Validation families or phase exposure differ')
            for phase in ['precontact','initialcontact','sustained','lostcontact']:
                selected=[r for r in rows if r['contact_phase']==phase and int(r['supervised_ticks'])]
                n=sum(int(r['supervised_ticks']) for r in selected)
                phases[phase]=dict(ticks=n,**{part+'_rmse_rad':math.sqrt(sum(
                    int(r['supervised_ticks'])*float(r['final_command_'+part+'_rmse_rad'])**2
                    for r in selected)/n) if n else None for part in ['arm','hand']})
            cases.append(dict(run=root.name,role=role,validation_ids=identities,phases=phases,
                checkpoint_sha256=result['checkpoint_sha256'],contact_metrics_sha256=sha256(path)))
    atomic_json(output,dict(cases=cases,test_used=False,
        interpretation='Same held-out demonstration families and phase exposure; tick-weighted squared-error aggregation of actual composed commands. Contact bins are observed contact only, not task-success stages. Decreased expert-history error does not establish autonomous improvement.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.roots,a.output)
