"""Unactivated v4 adapter recognizing only the exact proposed main owner argv.

Preserves v4 external predicates, telemetry binding, all-group pause schedules,
byte restoration and singleton lock. Old v4 must exit before this can apply.
"""
from pathlib import Path
from outputs.diagnostics.tight_paired_dagger_20261122_01 import resource_watch_v4 as base
from outputs.diagnostics.tight_paired_dagger_20261122_01 import parallel_rear_proposal as proposal

original_specs = base.specs


def specs(root, config):
    rows=original_specs(root,config)
    for row in rows:
        if row['name']=='main_train':
            row.update(module=proposal.MODULE,job=proposal.JOB)
    return rows


def main():
    base.specs=specs
    base.main()


if __name__=='__main__': main()
