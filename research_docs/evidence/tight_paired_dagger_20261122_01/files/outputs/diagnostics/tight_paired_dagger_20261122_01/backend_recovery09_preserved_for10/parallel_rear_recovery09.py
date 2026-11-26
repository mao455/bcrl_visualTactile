"""Exact new owner identity; no DP, physical or completed-cache replay."""
import argparse
from pathlib import Path
import sys
from outputs.diagnostics.tight_paired_dagger_20261122_01 import run_backend_recovery09 as driver
def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':return driver.project()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['rear-pair'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--registration',type=Path,default=driver.REGISTRATION)
    a=p.parse_args();driver.rear_pair(a.root,a.registration)
if __name__=='__main__':main()
