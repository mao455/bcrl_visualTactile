"""Optional conservative paired intervals; original registered bootstrap is unchanged.

This source was prepared before opening sealed test outcomes. Each discordant
category has a marginal binomial probability. Two 97.5% Clopper-Pearson intervals
give simultaneous coverage >=95% by the union bound, without independence of
the two category counts. Subtracting their bounds encloses p10-p01. This is
conservative; it remains informative about uncertainty when bootstrap is [0,0].
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def paired_bounds(only_left, only_right, n, alpha=.05):
    from scipy.stats import beta
    if (type(n) is not int or n <= 0 or type(only_left) is not int
            or type(only_right) is not int or min(only_left, only_right) < 0
            or only_left+only_right > n or not 0 < alpha < 1):
        raise ValueError('Valid mutually exclusive paired counts required')
    def marginal(k):
        return (0. if k == 0 else float(beta.ppf(alpha/4, k, n-k+1)),
                1. if k == n else float(beta.ppf(1-alpha/4, k+1, n-k)))
    a, b = marginal(only_left), marginal(only_right)
    return [100*(a[0]-b[1]), 100*(a[1]-b[0])]


def build(path):
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 64 or len({r['family'] for r in rows}) != 64:
        raise ValueError('Exactly the original64 paired families required')
    def bit(value):
        if value not in ('True','False'): raise ValueError('Boolean outcome required')
        return value == 'True'
    contrasts = []
    for left, right in [('A_new','B_new'),('A_new','A_parent'),('B_new','B_parent')]:
        a = [bit(r[left+'_success']) for r in rows]
        b = [bit(r[right+'_success']) for r in rows]
        p10 = sum(x and not y for x,y in zip(a,b))
        p01 = sum(y and not x for x,y in zip(a,b))
        contrasts.append(dict(left=left,right=right,n=len(rows),only_left=p10,only_right=p01,
            paired_difference_pp=100*(p10-p01)/len(rows),
            conservative_exact95_pp=paired_bounds(p10,p01,len(rows))))
    return dict(method='Paired discordance probabilities; two97.5% Clopper-Pearson marginal intervals; Bonferroni simultaneous coverage>=95%; bounds on p10-p01',
        primary_registered_bootstrap_unchanged=True,software_failures_keep_original_primary_coding=True,
        input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),contrasts=contrasts)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value=build(args.pairs)
    with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
