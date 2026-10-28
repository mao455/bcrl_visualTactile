"""Render-only diagnostic after a rollout. No physics steps or task observation calls."""
import json
import time
import subprocess
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch


def capture_static(env,output,frames=24):
    import omni.usd
    import carb
    from pxr import Usd,UsdPhysics
    stage=omni.usd.get_context().get_stage()
    robot=stage.GetPrimAtPath('/World/envs/env_0/Robot')
    tree=[dict(path=str(p.GetPath()),type=p.GetTypeName(),rigid=p.HasAPI(UsdPhysics.RigidBodyAPI))
          for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.GetName() in
          ('iiwa14_link_ee','sharpa_mount','left_hand_C_MC','iiwa14_link_7')]
    before=(env.robot.data.joint_pos.clone(),env.object.data.root_state_w.clone(),torch.random.get_rng_state().clone())
    images=[];poses=[];times=[]
    for i in range(frames):
        t=time.perf_counter();env.sim.render()
        for c in env.research_cameras:c.update(env.physics_dt,force_recompute=True)
        images.append(np.stack([c.data.output['rgb'][0,:,:,:3].cpu().numpy() for c in env.research_cameras]))
        if env._camera_pose_annotators:
            from .camera_rigs import renderer_pose_ros
            poses.append(np.stack([renderer_pose_ros(a[0].get_data()['cameraViewTransform']) for a in env._camera_pose_annotators]))
        else:
            poses.append(np.stack([torch.cat((c.data.pos_w,c.data.quat_w_ros),-1)[0].cpu().numpy() for c in env.research_cameras]))
        times.append(time.perf_counter()-t)
    after=(env.robot.data.joint_pos,env.object.data.root_state_w,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,after)), 'Static renderer changed state/RNG'
    data=np.stack(images);np.savez_compressed(output/'static_render.npz',rgb=data,pose_w=np.stack(poses))
    settings=carb.settings.get_settings()
    keys=['/rtx/rendermode','/rtx/post/aa/op','/rtx/post/dlss/execMode','/rtx/directLighting/sampledLighting/enabled',
          '/rtx/raytracing/fractionalCutoutOpacity','/rtx/raytracing/cached/enabled','/rtx/pathtracing/spp','/rtx/sceneDb/ambientLightIntensity',
          '/rtx/directLighting/sampledLighting/denoisingTechnique','/rtx-transient/dldenoiser/enabled',
          '/rtx/indirectDiffuse/enabled','/rtx/raytracing/subpixel/mode']
    record=dict(physical_state_and_torch_rng_unchanged=True,frames=frames,physics_steps=0,
        temporal_std_rgb_u8_by_view=data.astype(float).std(axis=0).mean(axis=(1,2,3)).tolist(),
        consecutive_mae_rgb_u8_by_view=np.abs(np.diff(data.astype(float),axis=0)).mean(axis=(0,2,3,4)).tolist(),
        render_seconds=times,usd_robot_tree=tree,robot_body_names=env.robot.body_names,
        camera_prim_paths=[c.cfg.prim_path for c in env.research_cameras],
        render_settings={k:settings.get(k) for k in keys},render_cfg=env.cfg.sim.render.to_dict(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
    # Controlled render-only ablation: same physical state, no task DR changes.
    key = '/rtx/directLighting/sampledLighting/enabled'
    spp = '/rtx/directLighting/sampledLighting/samplesPerPixel'
    original, original_spp = settings.get(key), settings.get(spp)
    record['lighting_ablation'] = {}
    try:
        for label, sampled, samples in [('sampled_off',False,original_spp), ('sampled_16',True,16)]:
            settings.set(key,sampled)
            if samples is not None: settings.set(spp,samples)
            for _ in range(8): env.sim.render()
            frames_rgb=[]; elapsed=[]
            for _ in range(frames):
                start=time.perf_counter(); env.sim.render()
                for camera in env.research_cameras: camera.update(env.physics_dt,force_recompute=True)
                frames_rgb.append(np.stack([c.data.output['rgb'][0,:,:,:3].cpu().numpy().copy() for c in env.research_cameras]))
                elapsed.append(time.perf_counter()-start)
            array=np.stack(frames_rgb)
            np.savez_compressed(output/('static_'+label+'.npz'),rgb=array)
            record['lighting_ablation'][label] = dict(sampled=settings.get(key),spp=settings.get(spp),
                temporal_std_rgb_u8_by_view=array.astype(float).std(0).mean((1,2,3)).tolist(),
                consecutive_mae_rgb_u8_by_view=np.abs(np.diff(array.astype(float),axis=0)).mean((0,2,3,4)).tolist(),
                render_seconds=elapsed)
    finally:
        settings.set(key,original)
        if original_spp is not None: settings.set(spp,original_spp)
    after=(env.robot.data.joint_pos,env.object.data.root_state_w,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,after)), 'Ablation changed state/RNG'
    record['static_pose_source']='render_product CameraParams' if env._camera_pose_annotators else 'fixed USD camera'
    record['camera_pose_backend'] = [dict(update_latest=c.cfg.update_latest_camera_pose,
         fabric_view=c._view._use_fabric) for c in env.research_cameras]
    gpu_xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    record['own_gpu_process']=[{x.tag:x.text for x in p} for p in gpu_xml.findall('.//process_info')
                               if p.findtext('pid')==str(os.getpid())]
    (output/'camera_diagnostics.json').write_text(json.dumps(record,default=str,indent=2))
