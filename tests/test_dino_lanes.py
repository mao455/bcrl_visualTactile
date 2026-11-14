import torch
import pytest
from torch import nn
from visuotactile.models.native_vision import NativeDinoSpatial,vision_lane_budget,EXACT_TAIL_PROFILE


class IndependentBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.image_tags = []

    def forward_features(self, images):
        self.batch_sizes.append(len(images))
        self.image_tags.append(images[:,0,0,0].clone())
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


@pytest.mark.parametrize('size',[1,2,4,8])
def test_candidate_kernel_budget_padding_and_lane_order_are_explicit(size):
    model=fixture()
    rgb=torch.stack([torch.full((3,3,252,448),v,dtype=torch.uint8) for v in (30,70,140)])
    result=model.forward_lanes(rgb,vision_microbatch=size)
    budget=vision_lane_budget(3,size)
    assert model.backbone.batch_sizes==[3*size]*budget['forward_features_calls']
    tags=torch.cat(model.backbone.image_tags).reshape(-1,3)
    expected=(torch.tensor([30,70,140])/255-.485)/.229
    # Inspect the real mock backbone inputs, not a CUDA numerical parity claim.
    torch.testing.assert_close(tags[:3],expected[:,None].expand(-1,3))
    if budget['padding_lanes']:
        torch.testing.assert_close(tags[3:],expected[-1:].expand(budget['padding_lanes'],3))
    assert result.shape==(3,3,384,18,32)
    assert budget['processed_images']==sum(model.backbone.batch_sizes)
    assert budget['real_images']==9 and budget['real_lanes']==3
    # Opt-in candidate calls never mutate the ordinary single-observation path.
    model(rgb[0]);assert model.backbone.batch_sizes[-1]==6


@pytest.mark.parametrize('size',[0,-1,True,3.5,'8',None])
def test_invalid_microbatch_rejected_before_backbone(size):
    model=fixture()
    with pytest.raises(ValueError,match='positive integer'):
        model.forward_lanes(torch.zeros(1,3,3,252,448,dtype=torch.uint8),vision_microbatch=size)
    assert model.backbone.batch_sizes==[]


def test_budget_has_no_four_lane_cap_and_counts_no_work_for_empty_active_set():
    assert vision_lane_budget(200,128)['forward_features_calls']==2
    assert vision_lane_budget(200,128)['padding_lanes']==56
    assert vision_lane_budget(0,128)['processed_images']==0


@pytest.mark.parametrize('lanes',[1,3,8,9])
def test_exact_tail_backbone_receives_only_real_images_and_preserves_lane_values(lanes):
    model=fixture()
    rgb=torch.stack([torch.full((3,3,252,448),15+i*20,dtype=torch.uint8) for i in range(lanes)])
    before=torch.get_rng_state().clone()
    result=model.forward_lanes(rgb,vision_microbatch=8,vision_tail_profile=EXACT_TAIL_PROFILE)
    expected_sizes=[24]*(lanes//8)+([3*(lanes%8)] if lanes%8 else [])
    assert model.backbone.batch_sizes==expected_sizes
    assert torch.equal(before,torch.get_rng_state())
    work=model.last_lane_execution
    assert work==vision_lane_budget(lanes,8,vision_tail_profile=EXACT_TAIL_PROFILE)
    assert work['processed_images']==work['real_images']==3*lanes and work['padding_lanes']==0
    tags=torch.cat(model.backbone.image_tags)
    expected=(torch.arange(lanes).repeat_interleave(3)*20+15)/255
    torch.testing.assert_close(tags,(expected-.485)/.229)
    # The mock is explicitly image-independent; real CUDA/full-controller parity
    # remains a separate gate and is not inferred from this CPU assertion.
    fixed=fixture();baseline=fixed.forward_lanes(rgb,vision_microbatch=8)
    assert torch.equal(result,baseline)
    assert set(fixed.backbone.batch_sizes)=={24}
    assert not hasattr(fixed,'last_lane_execution')
    model(rgb[0]);assert model.backbone.batch_sizes[-1]==6


@pytest.mark.parametrize('bad',[{},True,'exact',{'version':'unregistered'},dict(EXACT_TAIL_PROFILE,enabled=True)])
def test_unknown_tail_identity_rejected_without_inference(bad):
    model=fixture()
    with pytest.raises(ValueError,match='vision_tail_profile'):
        model.forward_lanes(torch.zeros(1,3,3,252,448,dtype=torch.uint8),vision_tail_profile=bad)
    assert model.backbone.batch_sizes==[]
    with pytest.raises(ValueError,match='vision_tail_profile'):vision_lane_budget(1,8,vision_tail_profile=bad)


def test_old_default_work_report_has_exact_original_schema():
    assert vision_lane_budget(3,8)==dict(version='current_observation_lane_microbatch_v1',vision_microbatch=8,
        images_per_kernel=24,forward_features_calls=1,real_lanes=3,padding_lanes=5,real_images=9,processed_images=24)
    assert vision_lane_budget(0,8,vision_tail_profile=EXACT_TAIL_PROFILE)['kernel_image_batch_counts']=={}
