#!/usr/bin/env python3
"""
Collect real robot EE poses for calibration
Connects to real Mecademic robot and moves to predefined joint angles
Records actual EE poses and updates calibration_data.json
"""
import sys
import time
import json
import numpy as np
from termcolor import colored
import logging

# Add parent directory to path to import mecademicpy
sys.path.append('/home/najo/NAS/VLA/Insertion_VLA_Sim/digital_twin')
import mecademicpy.robot as mdr

# === Configuration ===
ROBOT_ADDRESS = "192.168.0.100"
CALIBRATION_DATA_FILE = "calibration_data.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("="*80)
    print(colored("Real Robot Calibration Data Collection", "cyan"))
    print("="*80)

    # Load calibration data
    try:
        with open(CALIBRATION_DATA_FILE, 'r') as f:
            calib_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ {CALIBRATION_DATA_FILE} not found!")
        logger.error("   Run: python collect_calibration_data.py first")
        return

    joint_angles = calib_data["joints_deg"]
    num_poses = len(joint_angles)

    print(f"\n📋 Will collect {num_poses} calibration poses:")
    for i, joints in enumerate(joint_angles):
        print(f"  {i+1}. {joints}")

    input(f"\n⚠️  Make sure robot workspace is clear. Press ENTER to connect...")

    # Connect to robot
    try:
        logger.info(f"🔌 Connecting to robot at {ROBOT_ADDRESS}...")
        robot = mdr.Robot()
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error("❌ Failed to connect to robot")
            return

        logger.info("✅ Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.WaitIdle()
        time.sleep(0.5)

        logger.info("✅ Robot ready!")

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return

    # Collect poses
    real_poses = []

    print("\n" + "="*80)
    print("Starting Data Collection")
    print("="*80)

    for i, joints in enumerate(joint_angles):
        print(f"\n[{i+1}/{num_poses}] Moving to: {joints}")

        try:
            # Move to joint position
            robot.MoveJoints(*joints)
            robot.WaitIdle()
            time.sleep(0.5)  # Allow settling

            # Read EE pose
            pose = robot.GetPose()
            if not pose or len(pose) < 6:
                logger.error(f"   ❌ Failed to read pose!")
                real_poses.append([0, 0, 0, 0, 0, 0])
                continue

            ee_pose = list(pose[:6])
            real_poses.append(ee_pose)

            print(f"   ✅ EE Pose: [{ee_pose[0]:.2f}, {ee_pose[1]:.2f}, {ee_pose[2]:.2f}, "
                  f"{ee_pose[3]:.2f}, {ee_pose[4]:.2f}, {ee_pose[5]:.2f}]")

        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            real_poses.append([0, 0, 0, 0, 0, 0])

        # Safety pause between movements
        time.sleep(0.3)

    # Update calibration data
    calib_data["real_poses"] = real_poses

    # Save updated file
    with open(CALIBRATION_DATA_FILE, 'w') as f:
        json.dump(calib_data, f, indent=2)

    print("\n" + "="*80)
    print("✅ Data Collection Complete!")
    print("="*80)
    print(f"Updated: {CALIBRATION_DATA_FILE}")
    print("\nSummary:")
    print("-"*80)
    print(f"{'Joint Angles':<30} {'Real EE Pose (mm, deg)'}")
    print("-"*80)

    for joints, pose in zip(joint_angles, real_poses):
        j_str = str(joints)
        pose_str = f"[{pose[0]:7.2f}, {pose[1]:7.2f}, {pose[2]:7.2f}, " \
                   f"{pose[3]:7.2f}, {pose[4]:7.2f}, {pose[5]:7.2f}]"
        print(f"{j_str:<30} {pose_str}")

    print("-"*80)

    # Return to home
    try:
        logger.info("\n🏠 Returning to home position...")
        robot.MoveJoints(0, 0, 0, 0, 0, 0)
        robot.WaitIdle()
    except Exception as e:
        logger.warning(f"⚠️ Home move failed: {e}")

    # Cleanup
    try:
        robot.DeactivateRobot()
        robot.Disconnect()
        logger.info("✅ Robot disconnected")
    except:
        pass

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the real_poses data above")
    print("2. If data looks good, run: python compute_transformation.py")
    print("3. Check RMSE metrics (should be < 10mm, < 10°)")
    print("4. Then run: python Save_dataset.py")
    print("="*80)

if __name__ == "__main__":
    main()
