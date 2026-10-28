import os
from pathlib import Path
import pytest
from visuotactile.collection_resources import output_inventory, time_mean


def test_inventory_counts_other_batch_and_temporary_never_links(tmp_path):
    for name, size in [('outputs/old/a',13),('outputs/new/temporary/a',17),('runs/a',19)]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x'*size)
    outside=tmp_path/'unrelated';outside.mkdir();(outside/'large').write_bytes(b'x'*100)
    (tmp_path/'outputs/link').symlink_to(outside,target_is_directory=True)
    (tmp_path/'outputs/linkfile').symlink_to(outside/'large')
    assert output_inventory(tmp_path)=={'outputs':30,'runs':19}


def test_irregular_samples_are_weighted_and_missing_not_zero():
    assert time_mean([(0,20),(9,100)],10)==28
    assert time_mean([(0,20),(9,100)],10,2)==60
    assert time_mean([(0,None),(9,0)],10)==90
