import json
from unittest.mock import patch
import numpy as np
import pytest

from visuotactile.compact import sha256
from visuotactile.initial_geometry import rotation, overlaps
from visuotactile.manifest import ManifestReader, ManifestSequenceDataset, assert_group_disjoint
from visuotactile.run_insertion_dataset import design
from visuotactile.storage_audit import allocation_groups


def test_box_geometry_rotation_touch_and_penetration():
    r=rotation([np.sqrt(.5),0,0,np.sqrt(.5)])
    np.testing.assert_allclose(r @ [1,0,0],[0,1,0],atol=1e-14)
    box=(np.zeros(3),np.eye(3),np.array([1,.2,.3]))
    assert not overlaps(box,(np.array([2,0,0]),np.eye(3),box[2]))
    assert overlaps(box,(np.array([1.99,0,0]),np.eye(3),box[2]))
    assert not overlaps(box,(np.array([1.5,0,0]),r,box[2]))
    assert overlaps(box,(np.array([1.1,0,0]),r,box[2]))


def manifest_fixture(root):
    rows=[]
    for key,outcome,pair in [('a','success','seed1'),('b','failure','seed1'),('c','success','seed2')]:
        directory=root/key;directory.mkdir()
        files={}
        for filename in ('data.h5','camera_0.mp4','camera_1.mp4'):
            file=directory/filename;file.write_bytes(b'fixture')
            files[filename]=dict(bytes=file.stat().st_size,sha256=sha256(file))
        rows.append(dict(id=key,outcome=outcome,pair_id=pair,directory=key,files=files,valid=True,episode='0'))
    path=root/'manifest.json';path.write_text(json.dumps(dict(manifest_version=1,episodes=rows)))
    return path,rows


def test_manifest_checksum_missing_files_and_group_split(tmp_path):
    path,rows=manifest_fixture(tmp_path)
    reader=ManifestReader(path,verify=True)
    assert_group_disjoint(reader,['a','b'],['c'])
    with pytest.raises(ValueError,match='leak'):assert_group_disjoint(reader,['a'],['b'])
    (tmp_path/'a/data.h5').write_bytes(b'altered')
    with pytest.raises(ValueError,match='checksum'):ManifestReader(path,verify=True)
    del rows[0]['files']['camera_1.mp4']
    path.write_text(json.dumps(dict(manifest_version=1,episodes=rows)))
    with pytest.raises(ValueError,match='three files'):ManifestReader(path)


def test_manifest_success_filter_and_self_plan_guard(tmp_path):
    path,rows=manifest_fixture(tmp_path)
    with patch('visuotactile.manifest.EpisodeSequenceDataset') as inner:
        inner.return_value.metadata={'nominal_plan_source':'unavailable'}
        dataset=ManifestSequenceDataset(path)
        assert dataset.episodes==['a','c']
        with pytest.raises(ValueError,match='causal model replay'):dataset.require_self_plan_training()
        assert ManifestSequenceDataset(path,allow_unsuccessful=True).episodes==['a','b','c']
        with pytest.raises(ValueError,match='Unknown'):ManifestSequenceDataset(path,ids=['missing'])
        rows[0].update(stage='acceptance',wrench_frame_mode='legacy')
        rows[2].update(stage='scan',wrench_frame_mode='world_hold')
        path.write_text(json.dumps(dict(manifest_version=1,episodes=rows)))
        assert ManifestSequenceDataset(path).episodes==['c']
        with pytest.raises(ValueError,match='mix'):ManifestSequenceDataset(path,stage=None)


def test_frozen_design_and_storage_accounting():
    with patch('visuotactile.run_insertion_dataset.code_identity',return_value={}):plan=design()
    assert len(plan['scan'])==52 and len(plan['acceptance'])==4
    assert len({x['pair_id'] for x in plan['scan']})==4
    for seed in plan['seeds']:
        rows=[r for r in plan['scan'] if r['seed']==seed]
        assert len({r['condition'] for r in rows})==13
        assert rows[0]['offset_xy']==[0,0]
        assert max(np.linalg.norm(r['offset_xy']) for r in rows)==.05
    fields={'obs/image':300,'obs/depth_m':50,'obs/depth_valid':4,'transition/raw_action':20}
    groups=allocation_groups(fields,100)
    assert sum(groups.values())==474
    assert groups['packed_RGB_and_depth_not_separable']==300
