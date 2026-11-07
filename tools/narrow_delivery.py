"""Retained historical decision/compatibility helpers for existing controls.

Collection, training and publication owners are retired; use tools.tight_finish.
Full historical implementation remains in Git b85738e.
"""
import hashlib
import math
MARKER="NARROW_FOLLOWUP"


def outside(text):
    start=f'<!-- {MARKER}_START -->';end=f'<!-- {MARKER}_END -->'
    if start in text:
        a,rest=text.split(start,1);_,b=rest.split(end,1);text=a+b
    return hashlib.sha256(text.encode()).hexdigest()


def replace(text, block):
    start=f'<!-- {MARKER}_START -->';end=f'<!-- {MARKER}_END -->'
    generated=start+'\n'+block+'\n'+end
    if start in text:
        a,rest=text.split(start,1);_,b=rest.split(end,1);return a+generated+b
    title,sep,rest=text.partition('\n')
    return title+'\n\n'+generated+'\n'+rest


def wilson(success,total):
    if not total:return None
    p=success/total;z=1.959963984540054;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [mid-half,mid+half]
