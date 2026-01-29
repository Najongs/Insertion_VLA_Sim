#!/usr/bin/env python3
"""
Calibrate orientation transformation using optimization
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
from scipy.optimize import least_squares

MODEL_PATH = "meca_scene22.xml"

# Training data: (joint_angles, expected_ee_orientation)
training_data = [
    ([0, 0, 0, 0, 0, 0], [0, 90, 0]),
    ([30, 0, 0, 0, 0, 0], [-90, 60, 90]),
    ([45, 0, 0, 0, 0, 0], [-90, 45, 90]),
    ([0, 30, 0, 0, 0, 0], [-180, 60, 180]),
    ([0, 45, 0, 0, 0, 0], [-180, 45, 180]),
    ([0, 0, 30, 0, 0, 0], [-180, 60, 180]),
    ([0, 0, 45, 0, 0, 0], [-180, 45, 180]),
    ([0, 0, 0, 30, 0, 0], [30, 90, 0]),
    ([0, 0, 0, 45, 0, 0], [45, 90, 0]),
    ([0, 0, 0, 0, 30, 0], [-180, 60, 180]),
    ([0, 0, 0, 0, 45, 0], [-180, 45, 180]),
    ([0, 0, 0, 0, 0, 30], [30, 90, 0]),
    ([0, 0, 0, 0, 0, 45], [45, 90, 0]),
    ([30, -20, 20, 0, 30, 60], [-139.106605, 48.590378, -169.106605]),
]

def get_sim_orientation(model, data, link6_id, joints):
    """Get simulation orientation"""
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = np.deg2rad(joints)
    mujoco.mj_forward(model, data)

    link6_mat_world = data.xmat[link6_id].reshape(3, 3)

    sy = np.sqrt(link6_mat_world[0,0]**2 + link6_mat_world[1,0]**2)
    if sy > 1e-6:
        rx = np.arctan2(link6_mat_world[2,1], link6_mat_world[2,2])
        ry = np.arctan2(-link6_mat_world[2,0], sy)
        rz = np.arctan2(link6_mat_world[1,0], link6_mat_world[0,0])
    else:
        rx = np.arctan2(-link6_mat_world[1,2], link6_mat_world[1,1])
        ry = np.arctan2(-link6_mat_world[2,0], sy)
        rz = 0

    return np.array([np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)])

def normalize_angle(angle):
    """Normalize angle to [-180, 180]"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def residual_function(params, sim_ori_list, real_ori_list):
    """Residual function for optimization"""
    # params = [R00, R01, R02, R10, R11, R12, R20, R21, R22, offset_x, offset_y, offset_z]
    R = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], params[8]]
    ])
    offsets = np.array([params[9], params[10], params[11]])

    residuals = []
    for sim_ori, real_ori in zip(sim_ori_list, real_ori_list):
        predicted = R @ sim_ori + offsets
        error = predicted - real_ori
        # Handle angle wrapping
        error = np.array([normalize_angle(e) for e in error])
        residuals.extend(error)

    return residuals

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")

    # Collect simulation orientations
    sim_ori_list = []
    real_ori_list = []

    for joints, expected_ori in training_data:
        sim_ori = get_sim_orientation(model, data, link6_id, joints)
        sim_ori_list.append(sim_ori)
        real_ori_list.append(np.array(expected_ori))

    print("Collected simulation orientations:")
    for i, (joints, _) in enumerate(training_data):
        print(f"  {joints} -> sim: {sim_ori_list[i]}, real: {real_ori_list[i]}")

    # Initial guess: identity rotation + zero offset
    initial_params = np.array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0
    ])

    print("\nOptimizing transformation...")
    result = least_squares(
        residual_function,
        initial_params,
        args=(sim_ori_list, real_ori_list),
        verbose=2,
        max_nfev=1000
    )

    print("\n" + "="*80)
    print("Optimization Result:")
    print("="*80)

    R_opt = result.x[:9].reshape(3, 3)
    offsets_opt = result.x[9:]

    print("Rotation Matrix:")
    print(R_opt)
    print("\nOffsets:")
    print(offsets_opt)

    print("\n" + "="*80)
    print("Verification:")
    print("="*80)

    for i, (joints, expected_ori) in enumerate(training_data):
        sim_ori = sim_ori_list[i]
        predicted = R_opt @ sim_ori + offsets_opt
        error = predicted - real_ori_list[i]
        error = np.array([normalize_angle(e) for e in error])
        error_norm = np.linalg.norm(error)

        print(f"\n{joints}")
        print(f"  Expected: {expected_ori}")
        print(f"  Predicted: [{predicted[0]:.1f}, {predicted[1]:.1f}, {predicted[2]:.1f}]")
        print(f"  Error: {error_norm:.2f}°")

if __name__ == "__main__":
    main()
