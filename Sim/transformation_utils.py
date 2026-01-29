"""
Transformation utilities for converting simulation EE poses to real robot coordinates
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

class SimToRealTransform:
    """Transform simulation EE poses to real robot coordinate system"""

    def __init__(self, pos_transform_file="transformation_position.npy",
                 ori_transform_file="transformation_orientation.npy"):
        """
        Load transformation matrices

        Args:
            pos_transform_file: Path to 4x4 position transformation matrix
            ori_transform_file: Path to 3x3 orientation transformation matrix
        """
        self.T_pos = np.load(pos_transform_file)  # 4x4 matrix
        self.R_pos = self.T_pos[:3, :3]  # 3x3 rotation
        self.t_pos = self.T_pos[:3, 3]   # 3 translation

        self.R_ori = np.load(ori_transform_file)  # 3x3 orientation transform

        print(f"✅ Loaded transformation matrices")
        print(f"   Position RMSE: (see compute_transformation.py output)")
        print(f"   Orientation RMSE: (see compute_transformation.py output)")

    def transform_pose(self, sim_pose):
        """
        Transform simulation EE pose to real robot coordinates

        Args:
            sim_pose: [x, y, z, rx, ry, rz] in simulation frame
                     Position in mm, orientation in degrees (or radians)

        Returns:
            real_pose: [x, y, z, rx, ry, rz] in real robot frame
                      Same units as input
        """
        pos_sim = sim_pose[:3]
        ori_sim = sim_pose[3:]

        # Check if orientation is in radians or degrees
        is_radians = np.abs(ori_sim).max() < 10  # Heuristic

        if is_radians:
            ori_sim_deg = np.rad2deg(ori_sim)
        else:
            ori_sim_deg = ori_sim

        # Transform position
        pos_real = self.R_pos @ pos_sim + self.t_pos

        # Transform orientation
        # Convert Euler angles to rotation matrix (ZYX convention)
        r_sim = R.from_euler('ZYX', ori_sim_deg[::-1], degrees=True)
        R_sim = r_sim.as_matrix()

        # Apply orientation transformation: R_real = R_ori @ R_sim
        R_real = self.R_ori @ R_sim

        # Convert back to Euler angles
        r_real = R.from_matrix(R_real)
        ori_real_deg = r_real.as_euler('ZYX', degrees=True)[::-1]

        # Return in same format as input
        if is_radians:
            ori_real = np.deg2rad(ori_real_deg)
        else:
            ori_real = ori_real_deg

        return np.concatenate([pos_real, ori_real])

    def transform_action(self, delta_action):
        """
        Transform simulation delta action to real robot frame

        Args:
            delta_action: [dx, dy, dz, drx, dry, drz] in simulation frame
                         Position delta in mm, orientation delta in degrees (or radians)

        Returns:
            real_delta: [dx, dy, dz, drx, dry, drz] in real robot frame
                       Same units as input
        """
        delta_pos = delta_action[:3]
        delta_ori = delta_action[3:]

        # Check if orientation is in radians or degrees
        is_radians = np.abs(delta_ori).max() < 1  # Heuristic for deltas

        # Transform position delta (only rotation, no translation)
        delta_pos_real = self.R_pos @ delta_pos

        # Transform orientation delta
        # For small deltas, linear approximation: delta_ori_real ≈ R_ori @ delta_ori
        delta_ori_real = self.R_ori @ delta_ori

        return np.concatenate([delta_pos_real, delta_ori_real])

# Singleton instance (will be loaded once)
_transform_instance = None

def get_transform():
    """Get singleton transform instance"""
    global _transform_instance
    if _transform_instance is None:
        _transform_instance = SimToRealTransform()
    return _transform_instance
