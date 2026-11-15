"""Compact existing round12 review/receipts and retain actual mixed source roles."""
import argparse,hashlib,json,sys
from pathlib import Path

TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))


def run(root,output):
    def read(p):return json.loads(p.read_text())
    private=root/'completed_review';cycle=read(private/'cycle.json')
    previous=cycle.pop('source');previous.pop('cases',None)
    cycle['previous_round']=previous
    a=read(TREE/'research_docs/evidence/tight_fresh_20261114/collection11_recovery/completed_cycle.json')['candidate']
    b=read(TREE/'research_docs/evidence/tight_fresh_20261114/round09_behavior/report.json')['candidate']
    from tools.tight_fresh_round_compare import matched_legal_comparison
    # The older report also stores an observed-file hash; it is provenance,
    # not an execution setting. Keep comparing every actual execution field.
    assert a['execution']=={k:b['execution'][k] for k in a['execution']}==cycle['candidate']['execution']
    source=dict(execution=a['execution'],cases=[x for x in a['cases'] if x['role']=='A']+[x for x in b['cases'] if x['role']=='B'],
        pairs={'A':a['pairs']['A'],'B':b['pairs']['B']})
    session=read(root/'session.json');assert source['pairs']==session['followup']['selected_pairs']
    cycle['actual_sampling_sources']=dict(roles={'A':a['development'],'B':b['development']},
        pairs=source['pairs'],totals={'A':a['totals']['A'],'B':b['totals']['B']})
    cycle['compared_to_actual_sampling_sources']=matched_legal_comparison(source,cycle['candidate'])
    collection=read(private/'collection.json');collection.update(training_status_at_collection_close='not_started_then_completed',student_autonomous_benefit='No overall benefit: holding lower, zero alignment/insertion; native lift higher')
    front=read(private/'frontends.json')
    data=read(root/'round_12/A/state_data/data.json')
    io={role:read(private/(role+'_io_profile.json')) for role in ('A','B')}
    result=dict(version='completed12_outcomes_and_efficiency_v1',cycle=cycle,collection=collection,frontends=front,
        train_trajectories=len(data['splits']['train']),training_families=len({data['entries'][i]['family'] for i in data['splits']['train']}),
        validation_trajectories=len(data['splits']['validation']),frontend_actual_io=io,
        owner={k:v for k,v in read(root/'research_loop/owner_finished.json').items() if k not in ('rounds','stages')},
        interpretation='Round12 not adopted. Retain A11/B09. 21 holding-related successes/4286 valid expert steps are useful new data but do not establish autonomous benefit. Read-ahead candidate addresses measured I/O; formal speed outcome pending.',
        private_evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in private.glob('*.json')})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(train=result['train_trajectories'],families=result['training_families'],collection=collection['total'],seconds=cycle['seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.output.resolve())
