"""Display only recorded same-input B diagnostics; no interpolation."""
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tight_stability_20261116.analyze_existing import OUT

def main():
    p=OUT/'b_feedback.json';data=json.loads(p.read_text());maps={d:{r['seed']:r for r in data['rows'] if r.get('driven')==d} for d in ['short','long']};common=sorted(set(maps['short'])&set(maps['long']))
    paired=[]
    for driven in maps:
        selected=[maps[driven][s] for s in common];metrics={}
        for tested in maps:
            vals=[r['metrics']['first48'][tested] for r in selected]
            metrics[tested]=float(np.sqrt(np.mean([v['hand']**2 for v in vals])))
        paired.append(dict(driven=driven,n=len(common),hold60=sum(r['hold60'] for r in selected),first48_hand=metrics))
    data['paired_common']=paired;data['paired_seeds']=common;p.write_text(json.dumps(data,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(2)
    for offset,key in [(-.18,'short'),(.18,'long')]:ax[0].bar(x+offset,[r['first48_hand'][key] for r in paired],.36,label=key+' rear replay')
    ax[0].set_xticks(x,['Short-driven history','Long-driven history']);ax[0].set_ylabel('First48 hand RMSE vs diagnostic expert, rad');ax[0].legend()
    ax[1].bar(x,[r['hold60']/r['n'] for r in paired]);ax[1].set_xticks(x,['Short rear controls','Long rear controls']);ax[1].set_ylim(0,1)
    for i,r in enumerate(paired):ax[1].text(i,r['hold60']/r['n']+.03,f"{r['hold60']}/{r['n']}",ha='center')
    ax[1].set_ylabel('Autonomous ever hold60 / paired legal starts');fig.suptitle('B23 DP, rear6 vs rear22: same seven legal development starts, 5s diagnostic')
    fig.tight_layout();fig.savefig(OUT/'b_feedback.png',dpi=150);print(json.dumps(paired,indent=2))
if __name__=='__main__':main()
