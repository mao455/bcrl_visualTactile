import numpy as np
import torch

from visuotactile.camera_validity import content_valid, frame_metrics


def test_blank_colour_is_not_scene_texture():
    rgb=torch.ones(1,3,10,12)*.9
    rgb[:,0]=.2
    depth=torch.ones(1,1,10,12);mask=torch.ones_like(depth,dtype=torch.bool)
    assert not content_valid(rgb,depth,mask).item()
    assert not frame_metrics((rgb.numpy()*255).astype('uint8'),depth.numpy(),mask.numpy())['content_valid'][0]


def test_real_texture_requires_valid_metric_depth():
    rgb=torch.arange(120).reshape(1,1,10,12).repeat(1,3,1,1)/120
    depth=torch.ones(1,1,10,12);mask=torch.ones_like(depth,dtype=torch.bool)
    assert content_valid(rgb,depth,mask).item()
    assert not content_valid(rgb,depth*0,mask).item()
    row=frame_metrics((rgb.numpy()*255).astype('uint8'),depth.numpy(),np.zeros_like(mask.numpy()))
    assert not row['mask_exact']


def test_each_lane_checked_independently():
    rgb=torch.zeros(2,3,10,12);rgb[1,:,:,6:]=1
    depth=torch.ones(2,1,10,12);mask=torch.ones_like(depth,dtype=torch.bool)
    assert content_valid(rgb,depth,mask).tolist()==[False,True]
