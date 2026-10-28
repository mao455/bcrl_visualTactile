"""Serializable sensor/execution contracts, independent of Isaac Lab imports."""
import hashlib
import json
import numpy as np


def policy_input_contract(metadata):
    version = metadata.get("perception_version",1)
    student = metadata["student_config"]
    fields = ("proprio_list", "image_modality", "image_width", "image_height",
              "image_input_width", "image_input_height", "crop_enabled",
              "crop_top_left", "crop_bottom_right", "depth_preprocess_mode",
              "depth_min_m", "depth_max_m", "camera_backend",
              "camera_update_period_s", "use_camera_delay", "camera_delay_max",
              "use_student_obs_delay", "student_obs_delay_max",
              "use_depth_aug", "use_camera_pose_rand", "hide_goal_viz")
    result = {"problem": metadata["problem"], "camera_mode": metadata["camera_mode"],
              "joint_order": metadata["joint_order"],
              "physics_dt": metadata["physics_dt"], "control_dt": metadata["control_dt"],
              "student": {k: student[k] for k in fields},
              "nominal_proprio": metadata["nominal_proprio"]}
    result["perception_version"] = version
    if version == 2:
        calibration = []
        for camera in metadata["cameras"]:
            mount=metadata.get('camera_rig',{}).get('wrist') if camera['id']=='wrist_rgbd' else None
            for field in ("pos_env","quat_ros_wxyz","intrinsics"):
                value=([mount['pos_parent_m']] if mount and field=='pos_env' else
                       [mount['quat_parent_ros_wxyz']] if mount and field=='quat_ros_wxyz' else camera[field])
                values = np.asarray(value,dtype=float)
                if field == "quat_ros_wxyz":
                    values = np.where(values[:,:1]<0,-values,values)
                if not np.allclose(values,values[0],atol=1e-5,rtol=0):
                    raise ValueError("Per-environment camera calibration differs; variable mounts need an explicit model contract")
                calibration.extend(values[0].reshape(-1).tolist())
        result.update(camera_ids=metadata["camera_ids"],
                      resolutions=[c["resolution"] for c in metadata["cameras"]],
                      depth_encoding=metadata["depth_encoding"],
                      task_vocabulary=metadata["task_command"]["vocabulary"],
                      goal_frame=metadata["task_command"]["frame"],
                      calibration=calibration, fk_urdf_sha256=metadata["fk_urdf_sha256"])
        rig=metadata.get('camera_rig',{})
        if rig.get('version')==3:
            result['camera_layout_version']=3
            result['render_config']=metadata['config']['sim']['render']
            result['camera_mount']=rig.get('wrist',{'type':'all_fixed'})
        return result
    if version != 1:
        raise ValueError("Unsupported perception version")
    # Every current view shares the same extrinsics relative to its environment.
    # Legacy schema-1 captures lack env origins; the explicit named oblique mount
    # was fixed by the collector, not inferred from a simulator object pose.
    position = metadata.get("camera_pos_env", [[.6, -.8, 1.1]] if metadata["camera_mode"] == "oblique"
                            else [student["camera_pos"]])[0]
    quat = np.asarray(metadata["camera_quat_ros_wxyz"][0], dtype=float)
    if quat[0] < 0:
        quat = -quat
    result["calibration"] = np.concatenate((position, quat, np.asarray(
        metadata["camera_intrinsics"][0]).reshape(-1))).tolist()
    return result


def assert_policy_compatible(checkpoint_metadata, current_metadata):
    a, b = policy_input_contract(checkpoint_metadata), policy_input_contract(current_metadata)
    ca, cb = a.pop("calibration"), b.pop("calibration")
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        raise ValueError("Checkpoint sensor/action input contract differs from current environment")
    if not np.allclose(ca, cb, atol=1e-5, rtol=0):
        raise ValueError("Checkpoint camera calibration differs from current environment")


def evaluation_contract(metadata):
    """Compare experiments only within one recorded physical/success protocol."""
    cfg = metadata["config"]
    sections = ("action", "assets", "domain_randomization", "obs", "reset",
                "reward", "termination", "precise_assembly")
    sensor = policy_input_contract(metadata)
    sensor["calibration"] = np.round(sensor["calibration"], 5).tolist()
    result = {"task": {k: cfg[k] for k in sections}, "sensors": sensor,
              "collection": {"wrench_frame_mode": metadata.get("wrench_frame_mode","legacy"),
                             "initial_offset_xy_m": metadata.get("initial_offset",{}).get("offset_xy_m",[0.,0.])},
              "sim": {k: cfg["sim"][k] for k in ("device","dt","gravity","physx","physics_material","use_fabric")},
              "num_envs": cfg["scene"]["num_envs"], "seed": metadata["seed"],
              "success": "all goals AND retract_succeeded, completed episodes only"}
    return result


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
