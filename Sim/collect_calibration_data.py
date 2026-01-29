#!/usr/bin/env python3
"""
Step 1: Collect calibration data from simulation
Outputs simulation EE poses for predefined joint angles
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import json

MODEL_PATH = "meca_scene22.xml"

# Calibration joint angles (degrees)
CALIBRATION_JOINTS = [
    [0, 0, 0, 0, 0, 0],           # Home
    [30, 0, 0, 0, 0, 0],          # J1 = 30
    [45, 0, 0, 0, 0, 0],          # J1 = 45
    [0, 30, 0, 0, 0, 0],          # J2 = 30
    [0, 45, 0, 0, 0, 0],          # J2 = 45
    [0, 0, 30, 0, 0, 0],          # J3 = 30
    [0, 0, 45, 0, 0, 0],          # J3 = 45
    [0, 0, 0, 30, 0, 0],          # J4 = 30
    [0, 0, 0, 45, 0, 0],          # J4 = 45
    [30, -20, 20, 0, 30, 60],     # Combined pose
]

def get_ee_pose_raw(data, link6_id):
    """
    Get raw EE pose from simulation (6_Link body)
    Returns: [x, y, z, rx, ry, rz] in mm and degrees
    """
    # Get position (convert to mm)
    link6_pos_world = data.xpos[link6_id].copy()
    pos_mm = link6_pos_world * 1000.0

    # Get orientation (ZYX Euler angles)
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

    ori_deg = np.rad2deg([rx, ry, rz])

    return np.concatenate([pos_mm, ori_deg])

def main():
    print("="*80)
    print("Calibration Data Collection - Simulation Side")
    print("="*80)

    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")

    # Collect simulation poses
    calibration_data = {
        "joints_deg": [],
        "sim_poses": [],
        "real_poses": []  # To be filled manually
    }

    print("\nCollecting simulation EE poses...")
    print("-"*80)
    print(f"{'Joint Angles (deg)':<30} {'Sim EE Pose (mm, deg)'}")
    print("-"*80)

    for joints in CALIBRATION_JOINTS:
        # Set joint angles
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = np.deg2rad(joints)
        mujoco.mj_forward(model, data)

        # Get EE pose
        ee_pose = get_ee_pose_raw(data, link6_id)

        # Store
        calibration_data["joints_deg"].append(joints)
        calibration_data["sim_poses"].append(ee_pose.tolist())
        calibration_data["real_poses"].append([0, 0, 0, 0, 0, 0])  # Placeholder

        # Display
        j_str = str(joints)
        pose_str = f"[{ee_pose[0]:7.2f}, {ee_pose[1]:7.2f}, {ee_pose[2]:7.2f}, " \
                   f"{ee_pose[3]:7.2f}, {ee_pose[4]:7.2f}, {ee_pose[5]:7.2f}]"
        print(f"{j_str:<30} {pose_str}")

    print("-"*80)

    # Save to JSON
    output_file = "calibration_data.json"
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\n✅ Simulation data saved to: {output_file}")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run the SAME joint angles on your REAL ROBOT")
    print("2. Record the EE poses from the robot's teach pendant or API")
    print("3. Edit 'calibration_data.json' and fill in the 'real_poses' values")
    print("4. Then run: python compute_transformation.py")
    print("="*80)

if __name__ == "__main__":
    main()
