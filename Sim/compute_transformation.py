#!/usr/bin/env python3
"""
Step 2: Compute transformation matrix from sim to real robot
Uses Kabsch algorithm (SVD) for optimal rigid transformation
"""
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def normalize_angle(angle):
    """Normalize angle to [-180, 180]"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def kabsch_algorithm(P_sim, P_real):
    """
    Compute optimal rotation and translation using Kabsch algorithm

    Args:
        P_sim: (N, 3) array of simulation positions
        P_real: (N, 3) array of real robot positions

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    # Center the point clouds
    centroid_sim = np.mean(P_sim, axis=0)
    centroid_real = np.mean(P_real, axis=0)

    P_sim_centered = P_sim - centroid_sim
    P_real_centered = P_real - centroid_real

    # Compute covariance matrix
    H = P_sim_centered.T @ P_real_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R_opt = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # Compute translation
    t_opt = centroid_real - R_opt @ centroid_sim

    return R_opt, t_opt

def compute_orientation_transformation(ori_sim_list, ori_real_list):
    """
    Compute orientation transformation matrix

    Args:
        ori_sim_list: List of simulation orientations [[rx,ry,rz], ...] in degrees
        ori_real_list: List of real robot orientations [[rx,ry,rz], ...] in degrees

    Returns:
        R_ori: (3, 3) rotation matrix for orientation transformation
    """
    # Convert Euler angles to rotation matrices
    R_sim_list = []
    R_real_list = []

    for ori_sim, ori_real in zip(ori_sim_list, ori_real_list):
        # ZYX convention
        r_sim = R.from_euler('ZYX', ori_sim[::-1], degrees=True)
        r_real = R.from_euler('ZYX', ori_real[::-1], degrees=True)

        R_sim_list.append(r_sim.as_matrix())
        R_real_list.append(r_real.as_matrix())

    # Find optimal R_ori such that: R_real = R_ori @ R_sim
    # Formulate as least squares problem
    A = []
    b = []

    for R_sim, R_real in zip(R_sim_list, R_real_list):
        # R_real = R_ori @ R_sim
        # Vectorize: vec(R_real) = (R_sim^T ⊗ I) vec(R_ori)
        for i in range(3):
            for j in range(3):
                row = np.zeros(9)
                for k in range(3):
                    row[i*3 + k] = R_sim[k, j]
                A.append(row)
                b.append(R_real[i, j])

    A = np.array(A)
    b = np.array(b)

    # Solve least squares
    R_ori_vec, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    R_ori = R_ori_vec.reshape(3, 3)

    # Orthogonalize using SVD
    U, S, Vt = np.linalg.svd(R_ori)
    R_ori = U @ Vt

    # Ensure det = 1
    if np.linalg.det(R_ori) < 0:
        U[:, -1] *= -1
        R_ori = U @ Vt

    return R_ori

def main():
    print("="*80)
    print("Computing Transformation Matrix (Sim → Real)")
    print("="*80)

    # Load calibration data
    with open("calibration_data.json", 'r') as f:
        data = json.load(f)

    joints = np.array(data["joints_deg"])
    sim_poses = np.array(data["sim_poses"])
    real_poses = np.array(data["real_poses"])

    # Check if real poses are filled
    if np.all(real_poses == 0):
        print("\n❌ ERROR: 'real_poses' in calibration_data.json are not filled!")
        print("Please run the same joint angles on your REAL ROBOT and update the file.")
        return

    # Extract positions and orientations
    pos_sim = sim_poses[:, :3]   # (N, 3)
    ori_sim = sim_poses[:, 3:]   # (N, 3)
    pos_real = real_poses[:, :3]
    ori_real = real_poses[:, 3:]

    print(f"\nUsing {len(pos_sim)} calibration points")

    # ===== Position Transformation =====
    print("\n" + "-"*80)
    print("Computing Position Transformation (Kabsch Algorithm)")
    print("-"*80)

    R_pos, t_pos = kabsch_algorithm(pos_sim, pos_real)

    print("\nRotation Matrix R:")
    print(R_pos)
    print("\nTranslation Vector t (mm):")
    print(t_pos)

    # Verify position transformation
    print("\nPosition Transformation Verification:")
    print(f"{'Joint Angles':<25} {'Sim Pos':<30} {'Real Pos':<30} {'Predicted':<30} {'Error (mm)'}")
    print("-"*80)

    pos_errors = []
    for i in range(len(pos_sim)):
        pos_pred = R_pos @ pos_sim[i] + t_pos
        error = np.linalg.norm(pos_pred - pos_real[i])
        pos_errors.append(error)

        j_str = str(joints[i].tolist())
        sim_str = f"[{pos_sim[i,0]:6.1f},{pos_sim[i,1]:6.1f},{pos_sim[i,2]:6.1f}]"
        real_str = f"[{pos_real[i,0]:6.1f},{pos_real[i,1]:6.1f},{pos_real[i,2]:6.1f}]"
        pred_str = f"[{pos_pred[0]:6.1f},{pos_pred[1]:6.1f},{pos_pred[2]:6.1f}]"

        print(f"{j_str:<25} {sim_str:<30} {real_str:<30} {pred_str:<30} {error:6.2f}")

    print(f"\nPosition RMSE: {np.sqrt(np.mean(np.array(pos_errors)**2)):.2f} mm")
    print(f"Position Max Error: {np.max(pos_errors):.2f} mm")

    # ===== Orientation Transformation =====
    print("\n" + "-"*80)
    print("Computing Orientation Transformation")
    print("-"*80)

    R_ori = compute_orientation_transformation(ori_sim.tolist(), ori_real.tolist())

    print("\nOrientation Rotation Matrix R_ori:")
    print(R_ori)

    # Verify orientation transformation
    print("\nOrientation Transformation Verification:")
    print(f"{'Joint Angles':<25} {'Sim Ori':<30} {'Real Ori':<30} {'Predicted':<30} {'Error (deg)'}")
    print("-"*80)

    ori_errors = []
    for i in range(len(ori_sim)):
        # Convert to rotation matrices
        r_sim = R.from_euler('ZYX', ori_sim[i][::-1], degrees=True)
        r_real = R.from_euler('ZYX', ori_real[i][::-1], degrees=True)

        # Apply transformation
        R_pred = R_ori @ r_sim.as_matrix()
        r_pred = R.from_matrix(R_pred)
        ori_pred = r_pred.as_euler('ZYX', degrees=True)[::-1]

        # Compute error (angular distance)
        R_error = r_real.as_matrix().T @ R_pred
        angle_error = np.rad2deg(np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1)))
        ori_errors.append(angle_error)

        j_str = str(joints[i].tolist())
        sim_str = f"[{ori_sim[i,0]:6.1f},{ori_sim[i,1]:6.1f},{ori_sim[i,2]:6.1f}]"
        real_str = f"[{ori_real[i,0]:6.1f},{ori_real[i,1]:6.1f},{ori_real[i,2]:6.1f}]"
        pred_str = f"[{ori_pred[0]:6.1f},{ori_pred[1]:6.1f},{ori_pred[2]:6.1f}]"

        print(f"{j_str:<25} {sim_str:<30} {real_str:<30} {pred_str:<30} {angle_error:6.2f}")

    print(f"\nOrientation RMSE: {np.sqrt(np.mean(np.array(ori_errors)**2)):.2f} deg")
    print(f"Orientation Max Error: {np.max(ori_errors):.2f} deg")

    # ===== Save Transformation Matrix =====
    print("\n" + "="*80)
    print("Saving Transformation Matrices")
    print("="*80)

    # Save as 4x4 homogeneous transformation matrix for position
    T_pos = np.eye(4)
    T_pos[:3, :3] = R_pos
    T_pos[:3, 3] = t_pos

    np.save("transformation_position.npy", T_pos)
    np.save("transformation_orientation.npy", R_ori)

    print(f"\n✅ Saved transformation_position.npy (4x4 matrix)")
    print(f"✅ Saved transformation_orientation.npy (3x3 matrix)")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the error metrics above")
    print("2. If errors are acceptable, run: python update_save_dataset.py")
    print("3. This will integrate the transformation into Save_dataset.py")
    print("="*80)

if __name__ == "__main__":
    main()
