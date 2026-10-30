"""CPU/meta parameter census of local checkpoints and pinned public examples."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn

from .compact import sha256
from .training_resources import atomic_json


def count(module):
    parameters = list(module.parameters())
    return dict(total=sum(value.numel() for value in parameters),
                requires_grad=sum(value.numel() for value in parameters if value.requires_grad),
                frozen=sum(value.numel() for value in parameters if not value.requires_grad))


def describe(module):
    result = count(module)
    result['components'] = {name: count(child) for name, child in module.named_children()}
    result['fp32_parameter_mib'] = result['total']*4/1024**2
    return result


def meta_factory(constructor):
    def create(*args, **kwargs):
        with torch.device('meta'):
            return constructor(*args, **kwargs)
    return create


def own_models(root):
    from .rollout import load_command_controller
    from .models.residual import ResidualCorrection
    rows = {}
    for variant in ('b0', 'b1'):
        path = Path('outputs/visuotactile/run/dp_command_20261029') / variant / 'best.pt'
        controller = load_command_controller(path, 1, device='cpu')
        visual = controller.model.encoder if variant == 'b0' else controller.model.visual
        saved_base = torch.load(path, map_location='cpu', weights_only=False)
        rows[variant] = dict(**describe(controller.model), checkpoint_sha256=sha256(path),
            config=vars(controller.model.cfg), selected_epoch=controller.checkpoint_epoch,
            selected_checkpoint_updates=saved_base['optimizer_updates'],
            optimizer_groups=[{name:value for name,value in group.items() if name!='params'}
                              for group in saved_base['optimizer']['param_groups']],
            encoder_components={name:count(child) for name,child in visual.named_children()},
            rgbd_components={name:count(child) for name,child in visual.vision.named_children()})
        del saved_base
        if variant == 'b0':
            for group in ('b', 'c'):
                checkpoint = root / 'training_full' / (group+'_best.pt')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model = ResidualCorrection(controller.model.encoder, controller.model.cfg,
                    saved['scales'], force_masked=saved['force_masked'])
                model.load_state_dict(saved['model'])
                rows['residual_'+group] = dict(**describe(model), checkpoint_sha256=sha256(checkpoint),
                    base_sha256=saved['base_sha256'], optimizer_updates=saved['optimizer_updates'],
                    input_version=saved['input_version'], action_version=saved['action_version'],
                    optimizer_parameter_count=sum(value.numel() for value in model.parameters() if value.requires_grad))
                del model, saved
        del controller
        gc.collect()
    rows['deployment'] = dict(a=rows['b0']['total'],
        b=rows['b0']['total']+rows['residual_b']['total'],
        c=rows['b0']['total']+rows['residual_c']['total'],
        note='A mathematically uses B0 only; matched runner also loads and evaluates bypassed residual B. '
             'All deployment parameters are frozen; requires_grad columns above describe training construction.')
    return rows


def public_models(references):
    from diffusers import DDIMScheduler
    from omegaconf import OmegaConf
    import torchvision
    if 'peft' not in sys.modules:
        optional = ModuleType('peft')
        def unavailable(name):
            if name.startswith('__'):
                raise AttributeError(name)
            raise RuntimeError('LoRA is outside this census; optional PEFT must not be used')
        optional.__getattr__ = unavailable
        sys.modules['peft'] = optional
    dp = references / 'diffusion_policy-5ba07ac6661db573af695b419a7947ecb704690f'
    for path in (dp, references / 'rdp', references / 'irdp'):
        sys.path.insert(0, str(path.resolve()))
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2')
    rows = {}

    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
    shape = OmegaConf.to_container(OmegaConf.load(dp / 'diffusion_policy/config/task/lift_image_abs.yaml').shape_meta)
    hybrid = DiffusionUnetHybridImagePolicy(shape_meta=shape, noise_scheduler=scheduler,
        horizon=16, n_action_steps=8, n_obs_steps=2, num_inference_steps=8, crop_shape=None,
        diffusion_step_embed_dim=128, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        cond_predict_scale=True, obs_encoder_group_norm=True, eval_fixed_crop=True)
    rows['dp_hybrid_lift'] = dict(**describe(hybrid), shape_meta=shape,
        obs_feature_dim=hybrid.obs_encoder.output_shape()[0],
        denoiser=describe(hybrid.model), encoder=describe(hybrid.obs_encoder),
        method='Unmodified pinned full policy constructor on CPU, robomimic 0.2.0; no checkpoint or rollout')
    del hybrid
    gc.collect()

    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder as DPEncoder
    from diffusion_policy.model.vision.model_getter import get_resnet as dp_resnet
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from diffusion_policy.policy import diffusion_unet_image_policy as dp_policy_module
    shape = dict(obs={name:dict(shape=[3, 240, 320], type='rgb') for name in ('camera_1', 'camera_3')},
                 action=dict(shape=[2]))
    shape['obs']['robot_eef_pose'] = dict(shape=[2], type='low_dim')
    encoder = DPEncoder(shape_meta=shape, rgb_model=dp_resnet('resnet18'),
        resize_shape=[240, 320], crop_shape=[216, 288], random_crop=True,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    with patch.object(dp_policy_module, 'ConditionalUnet1D', meta_factory(dp_policy_module.ConditionalUnet1D)):
        policy = DiffusionUnetImagePolicy(shape_meta=shape, noise_scheduler=scheduler, obs_encoder=encoder,
            horizon=16, n_action_steps=8, n_obs_steps=2, num_inference_steps=16,
            diffusion_step_embed_dim=128, down_dims=[512, 1024, 2048], kernel_size=5,
            n_groups=8, cond_predict_scale=True)
    rows['dp_real_pusht'] = dict(**describe(policy), shape_meta=shape,
        obs_feature_dim=policy.obs_feature_dim, denoiser=describe(policy.model), encoder=describe(encoder),
        method='Pinned full policy, topology unmodified; denoiser factory uses meta device, '
               'CPU encoder, deployment DDIM16 override')
    del policy, encoder
    gc.collect()

    from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder as RDPEncoder
    from reactive_diffusion_policy.model.vision.model_getter import get_resnet as rdp_resnet
    from reactive_diffusion_policy.model.vae.model import VAE
    from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import LatentDiffusionUnetImagePolicy
    from reactive_diffusion_policy.policy import diffusion_unet_image_policy as rdp_policy_module
    shape = dict(obs=dict(left_wrist_img=dict(shape=[3, 240, 320], type='rgb'),
        left_robot_tcp_pose=dict(shape=[3], type='low_dim'),
        left_robot_gripper_width=dict(shape=[1], type='low_dim'),
        left_robot_tcp_wrench=dict(shape=[6], type='low_dim')),
        extended_obs=dict(left_robot_tcp_wrench=dict(shape=[6], type='low_dim')), action=dict(shape=[4]))
    tokenizer = VAE(shape_meta=shape, horizon=32, n_latent_dims=4, mlp_layer_num=1,
        use_conv_encoder=True, conv_latent_dims=32, conv_layer_num=1,
        use_rnn_decoder=True, rnn_latent_dims=32, rnn_layer_num=1,
        use_vq=False, n_embed=16, vqvae_groups=4, eval=True, device='cpu', act_scale=1.)
    tokenizer_counts = {name:count(getattr(tokenizer, name)) for name in ('encoder','decoder','quant','post_quant')}
    encoder = RDPEncoder(shape_meta=shape, rgb_model=rdp_resnet('resnet18'), resize_shape=[240, 320],
        random_transforms=OmegaConf.create([dict(type='RandomCrop', ratio=.9)]), use_group_norm=True,
        share_rgb_model=False, imagenet_norm=True)
    with patch.object(rdp_policy_module, 'ConditionalUnet1D', meta_factory(rdp_policy_module.ConditionalUnet1D)):
        policy = LatentDiffusionUnetImagePolicy(at=tokenizer, use_latent_action_before_vq=False,
            shape_meta=shape, noise_scheduler=scheduler, obs_encoder=encoder,
            horizon=32, n_action_steps=29, n_obs_steps=2, num_inference_steps=8,
            diffusion_step_embed_dim=128, down_dims=[512, 1024, 2048], kernel_size=5,
            n_groups=8, cond_predict_scale=True)
    tokenizer_total = sum(row['total'] for row in tokenizer_counts.values())
    rows['rdp_peel_wrench'] = dict(**describe(policy), mutated_model_shape_meta=shape,
        obs_feature_dim=policy.obs_feature_dim, latent_horizon=policy.horizon,
        latent_channels=policy.action_dim, effective_kernel=3, tokenizer=tokenizer_counts,
        tokenizer_total=tokenizer_total, complete_including_tokenizer=tokenizer_total+count(policy)['total'],
        deployed_tokenizer_decoder_only=tokenizer_counts['decoder']['total']+tokenizer_counts['post_quant']['total'],
        denoiser=describe(policy.model), encoder=describe(encoder),
        method='Pinned full LDP constructor with meta-device denoiser factory; topology unmodified. '
               'CPU encoder/AT. AT is not an nn.Module '
               'and is counted separately; actual non-VQ latent channels=16, latent horizon=8, kernel=3.')
    del policy, encoder, tokenizer
    gc.collect()

    from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder
    from ImplicitRDP.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
    shape = dict(obs=dict(right_wrist_img=dict(shape=[3, 360, 640], type='rgb'),
        right_robot_tcp_pose=dict(shape=[6], type='low_dim'), right_robot_tcp_wrench=dict(shape=[6], type='low_dim')),
        extended_obs=dict(right_robot_tcp_wrench=dict(shape=[6], type='low_dim')), action=dict(shape=[13]))
    transforms = [OmegaConf.create(dict(type='CenterCrop')),
                  torchvision.transforms.RandomRotation(degrees=(-5., 5.)),
                  torchvision.transforms.ColorJitter(brightness=.3, contrast=.4, saturation=.5, hue=.08)]
    encoder = TransformerObsEncoder(shape_meta=shape, obs_horizon=2, model_name='resnet34',
        n_emb=768, transforms=transforms, global_pool='', pretrained=False, frozen=False,
        use_group_norm=True, share_rgb_model=False, feature_aggregation=None, downsample_ratio=32)
    with torch.device('meta'):
        denoiser = TransformerForDiffusion(shape_meta=shape, action_horizon=16, n_obs_steps=2,
            cond_dim=768, n_layer=7, n_head=4, n_emb=768, p_drop_emb=0., p_drop_attn=.1,
            time_as_cond=True, obs_as_cond=True, n_cond_layers=0)
    policy = DiffusionTransformerImagePolicy(shape_meta=shape, noise_scheduler=scheduler, obs_encoder=encoder,
        model=denoiser, horizon=16, n_action_steps=15, n_obs_steps=2, num_inference_steps=5,
        n_emb=768, use_rnn_obs_encoder=True, rnn_obs_encoder_n_layer=1, rnn_obs_encoder_hidden_dim=512)
    rows['implicitrdp_flip_wrench'] = dict(**describe(policy), shape_meta=shape,
        obs_token_shape=list(encoder.output_shape()), denoiser=describe(denoiser), encoder=describe(encoder),
        method='Unmodified full policy constructor; CPU timm ResNet34 encoder/force GRU, meta transformer; '
               'no official checkpoint; 224-square transform from stored 360x640 RGB')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--isolated-dependencies', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.isolated_dependencies.resolve()))
    torch.set_num_threads(2)
    result = dict(schema_version=1, scope='Neural parameters only, unique tensors; no normalizer statistics, '
        'optimizer state, EMA duplicate, activation memory or measured latency. External counts describe '
        'explicit pinned example constructors, not trained author checkpoints.', own=own_models(args.training_root))
    result['public'] = public_models(args.references)
    result['revisions'] = dict(dp='real-stanford/diffusion_policy@5ba07ac6661db573af695b419a7947ecb704690f',
        rdp='xiaoxiaoxh/reactive_diffusion_policy@824c5e8de1fd1811106907a04b5f0186e0138c0b',
        implicitrdp='Chen-Wendi/ImplicitRDP@112d6857d914c5dee4c4c172790088adc6bdc3a5')
    result['dependencies'] = {name:importlib.metadata.version(name)
        for name in ('torch','torchvision','robomimic','timm','diffusers','einops')}
    result['import_limitations'] = ['Unused optional peft import replaced by a sentinel that raises on use; '
        'none of the selected architectures uses LoRA. No upstream source or shared installation was edited.',
        'timm is unpinned upstream; this census fixes 1.0.15. DP robomimic uses its declared 0.2.0. '
        'PyTorch/torchvision are the installed versions, not author training environments.']
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(args.references.resolve()) and path.suffix=='.py':
                sources[str(path.relative_to(args.references.resolve()))] = sha256(path)
    result['loaded_source_sha256'] = sources
    result['census_source_sha256'] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({name:row['total'] for name,row in result['public'].items()}, indent=2))


if __name__ == '__main__':
    main()
