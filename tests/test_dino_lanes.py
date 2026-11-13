import torch
import pytest
from torch import nn
from visuotactile.models.native_vision import NativeDinoSpatial


class IndependentBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward_features(self, images):
        self.batch_sizes.append(len(images))
        return dict(x_norm_patchtokens=images.mean((1,2,3))[:,None,None].expand(-1,576,384))


def fixture():
    model=NativeDinoSpatial.__new__(NativeDinoSpatial)
    nn.Module.__init__(model)
    model.backbone=IndependentBackbone()
    model.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
    model.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    return model


def test_lanes_are_independent_and_keep_exact_six_image_batches():
    model=fixture()
    rgb=torch.stack([torch.full((3,3,252,448),v,dtype=torch.uint8) for v in (30,70,140)])
    batch=model.forward_lanes(rgb)
    serial=torch.stack([model(frame) for frame in rgb])
    assert torch.equal(batch,serial)
    changed=rgb.clone();changed[1]=220
    other=model.forward_lanes(changed)
    assert torch.equal(batch[0],other[0]) and torch.equal(batch[2],other[2])
    assert not torch.equal(batch[1],other[1])
    assert set(model.backbone.batch_sizes)=={6}
    assert torch.equal(batch,model.forward_lanes(rgb.float()/255))


def test_invalid_layout_and_nonreversible_pixels_are_rejected():
    model=fixture()
    with pytest.raises(ValueError):model.forward_lanes(torch.zeros(0,3,3,252,448,dtype=torch.uint8))
    with pytest.raises(ValueError):model.forward_lanes(torch.full((1,3,3,252,448),.123456))
