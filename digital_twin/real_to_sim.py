#!/usr/bin/env python
"""
Digital Twin: Real-to-Sim Synchronization
Streams Real Mecademic Robot State → MuJoCo Simulation

Architecture:
- Real robot state sampler in separate thread
- Simulation mirror runs in main thread (MuJoCo viewer)
- State synchronization via thread-safe queue
- Gamepad control of real robot → visualization in simulation
"""

import os
import sys
import time
import threading
import logging
import numpy as np
import queue
from termcolor import colored
import cv2
import contextlib
import pathlib

import mujoco
import mujoco.viewer
import mecademicpy.robot as mdr
import depthai as dai
import pygame

# === Configuration ===
ROBOT_ADDRESS = "192.168.0.100"
MODEL_PATH = "../Sim/meca_scene22.xml"
HOME_JOINTS = (30, -20, 20, 0, 30, 60)
CONTROL_FREQUENCY = 15  # Hz

# Gamepad settings (same as Robot_action.py)
SCALE_POS = 0.8
SCALE_HAT = 0.3
SCALE_Z = 0.5
SCALE_ROT = 0.3
DEADZONE = 0.2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Real Robot State Sampler (from Robot_action.py)
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.dt = 1.0 / float(rate_hz)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = np.zeros(6)
        self.latest_p = np.zeros(6)

    def stop(self):
        self.stop_evt.set()

    def get_latest_data(self):
        with self.lock:
            return self.latest_q.copy(), self.latest_p.copy()

    def run(self):
        logger.info("🤖 Starting robot state sampler...")

        # Initial read
        initial_success = False
        for _ in range(10):
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())
                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
                    logger.info(f"✅ Initial robot state acquired")
                    initial_success = True
                    break
            except Exception as e:
                logger.debug(f"Initial read failed: {e}")
            time.sleep(0.1)

        if not initial_success:
            logger.warning("⚠️ Failed to get initial robot state!")

        next_t = time.time()
        while not self.stop_evt.is_set():
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())

                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
            except Exception as e:
                pass

            next_t += self.dt
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

# ============================================================
# Gamepad Controller (from Robot_action.py)
# ============================================================
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.control_mode = 1
        self.mode_switch_cooldown = 0
        self.smoothing_enabled = False
        self.smoothing_cooldown = 0
        self.current_action = np.zeros(6)
        self.acceleration_rate = 0.2
        self.deceleration_rate = 0.7

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"🎮 Gamepad connected: {self.joystick.get_name()}")
        else:
            logger.warning("⚠️ No gamepad found (robot will stay in home position)")

    def get_action(self):
        if not self.joystick:
            return np.zeros(6), False, False, False

        pygame.event.pump()

        # Mode switching
        btn_mode_switch = self.joystick.get_button(6) if self.joystick.get_numbuttons() > 6 else False
        if btn_mode_switch and self.mode_switch_cooldown == 0:
            self.control_mode = (self.control_mode % 3) + 1
            logger.info(colored(f"🔧 Control Mode: {self.control_mode}", "yellow"))
            self.mode_switch_cooldown = 10
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1

        # Smoothing toggle
        btn_smoothing = self.joystick.get_button(2) if self.joystick.get_numbuttons() > 2 else False
        if btn_smoothing and self.smoothing_cooldown == 0:
            self.smoothing_enabled = not self.smoothing_enabled
            status = "ON" if self.smoothing_enabled else "OFF"
            logger.info(colored(f"🌊 Smoothing: {status}", "cyan"))
            self.smoothing_cooldown = 10
        if self.smoothing_cooldown > 0:
            self.smoothing_cooldown -= 1

        # Analog sticks
        y_stick_raw = self.joystick.get_axis(1)
        x_stick_raw = -self.joystick.get_axis(0)
        rs_x_raw = self.joystick.get_axis(3)
        rs_y_raw = -self.joystick.get_axis(4)

        # Deadzone
        y_stick = y_stick_raw if abs(y_stick_raw) > DEADZONE else 0.0
        x_stick = x_stick_raw if abs(x_stick_raw) > DEADZONE else 0.0
        rs_x = rs_x_raw if abs(rs_x_raw) > DEADZONE else 0.0
        rs_y = rs_y_raw if abs(rs_y_raw) > DEADZONE else 0.0

        y_stick *= SCALE_POS
        x_stick *= SCALE_POS

        # Triggers and bumpers
        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)

        # D-Pad
        hat_x, hat_y = self.joystick.get_hat(0)
        y_hat = -hat_y * SCALE_HAT
        x_hat = -hat_x * SCALE_HAT

        # Combine movement
        y = y_stick + y_hat
        x = x_stick + x_hat

        # Tool frame compensation (60° rotation)
        angle = np.radians(60)
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        x, y = x_rotated, y_rotated

        # Rotation mapping based on control mode
        if self.control_mode == 1:
            rx = rs_y * SCALE_ROT
            ry = rs_x * SCALE_ROT
            rz = (rb - lb) * SCALE_ROT * 1.5
            z = (rt - lt) * SCALE_Z
        elif self.control_mode == 2:
            rx = rs_y * SCALE_ROT
            rz = rs_x * SCALE_ROT * 1.5
            ry = (rb - lb) * SCALE_ROT
            z = (rt - lt) * SCALE_Z
        else:  # Mode 3
            rx = rs_y * SCALE_ROT
            ry = (rb - lb) * SCALE_ROT
            rz = (rt - lt) * SCALE_ROT * 2.0
            z = 0

        # Rotation compensation
        angle = np.radians(60)
        rx_rotated = rx * np.cos(angle) - ry * np.sin(angle)
        ry_rotated = rx * np.sin(angle) + ry * np.cos(angle)
        rx, ry = rx_rotated, ry_rotated

        target_action = np.array([y, x, z, rx, ry, rz])

        # Smoothing
        if self.smoothing_enabled:
            if np.linalg.norm(target_action) < 0.01:
                self.current_action += (target_action - self.current_action) * self.deceleration_rate
            else:
                self.current_action += (target_action - self.current_action) * self.acceleration_rate
            action = np.where(np.abs(self.current_action) < 0.001, 0, self.current_action)
        else:
            action = target_action
            self.current_action = target_action

        # Buttons
        btn_home = self.joystick.get_button(3)  # Y
        btn_exit = self.joystick.get_button(7) if self.joystick.get_numbuttons() > 7 else False  # START

        return action, btn_home, btn_exit

# ============================================================
# Camera Manager (from Robot_action.py)
# ============================================================
class OAKCameraManager:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.stack = contextlib.ExitStack()
        self.queues = []

    def initialize_cameras(self):
        infos = dai.Device.getAllAvailableDevices()
        if not infos:
            logger.warning("⚠️ No OAK cameras found")
            return 0

        for info in infos:
            p = dai.Pipeline()
            c = p.create(dai.node.ColorCamera)
            c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            c.setPreviewSize(self.width, self.height)
            c.setInterleaved(False)

            # Manual focus for camera ID starting with "19"
            camera_id = info.getMxId()
            if camera_id.startswith("19"):
                c.initialControl.setManualFocus(101)

            out = p.create(dai.node.XLinkOut)
            out.setStreamName("rgb")
            c.preview.link(out.input)
            d = self.stack.enter_context(dai.Device(p, info, dai.UsbSpeed.SUPER))
            self.queues.append(d.getOutputQueue("rgb", 4, False))

        return len(self.queues)

    def get_frames(self):
        frames = {}
        for i, q in enumerate(self.queues):
            f = q.tryGet()
            if f:
                frames[f"camera{i+1}"] = f.getCvFrame()
        return frames

    def close(self):
        self.stack.close()

# ============================================================
# Main
# ============================================================
def main():
    logger.info(colored("=== Digital Twin: Real-to-Sim ===", "cyan"))

    # Load MuJoCo model
    logger.info(f"📦 Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Get site IDs for visualization
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    except:
        logger.warning("⚠️ Needle sites not found")
        tip_id = -1
        back_id = -1

    # Initialize gamepad and cameras
    gamepad = GamepadController()
    camera_mgr = OAKCameraManager()

    try:
        # Connect to robot
        logger.info(f"🔌 Connecting to robot at {ROBOT_ADDRESS}...")
        robot = mdr.Robot()
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error("❌ Failed to connect to robot")
            return

        logger.info("✅ Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)

        logger.info("🏠 Moving to home position...")
        robot.MoveJoints(*HOME_JOINTS)
        robot.WaitIdle()
        logger.info("✅ Robot ready!")

        # Start state sampler
        sampler = RtSampler(robot, rate_hz=100)
        sampler.start()

        # Initialize cameras
        logger.info("📷 Initializing cameras...")
        num_cameras = camera_mgr.initialize_cameras()
        if num_cameras > 0:
            logger.info(f"✅ {num_cameras} camera(s) initialized")
        else:
            logger.warning("⚠️ No cameras available")

        logger.info(colored("\n=== CONTROLS ===", "cyan"))
        logger.info(" [Gamepad]    Control real robot (see Robot_action.py for details)")
        logger.info(" [Y Button]   Go to home position")
        logger.info(" [START]      Exit program")
        logger.info(" Real robot movements will be mirrored in simulation!\n")

        # Set initial simulation pose
        data.qpos[:6] = np.deg2rad(HOME_JOINTS)
        mujoco.mj_forward(model, data)

        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            last_control_time = time.time()
            control_dt = 1.0 / CONTROL_FREQUENCY

            while viewer.is_running():
                step_start = time.time()

                # Get real robot state
                robot_q, robot_p = sampler.get_latest_data()

                # Update simulation to match robot
                # Convert degrees to radians for MuJoCo
                sim_qpos = np.deg2rad(robot_q)
                data.qpos[:6] = sim_qpos
                mujoco.mj_forward(model, data)

                # Get camera frames
                frames = camera_mgr.get_frames()

                # Control robot with gamepad
                current_time = time.time()
                if current_time - last_control_time >= control_dt:
                    action, btn_home, btn_exit = gamepad.get_action()

                    if btn_exit:
                        logger.info(colored("🛑 Exit button pressed", "red"))
                        break

                    if btn_home:
                        logger.info(colored("🏠 Going home...", "yellow"))
                        try:
                            robot.MoveJoints(*HOME_JOINTS)
                            robot.WaitIdle()
                        except Exception as e:
                            logger.error(f"Home failed: {e}")

                    # Send movement command to robot
                    if np.any(np.abs(action) > 0.001):
                        try:
                            robot.MoveLinRelTrf(*[float(x) for x in action])
                        except Exception as e:
                            logger.debug(f"Move command failed: {e}")

                    last_control_time = current_time

                # Display camera feeds
                if frames and step_count % 2 == 0:
                    sorted_keys = sorted(frames.keys())
                    img_list = []
                    for key in sorted_keys:
                        img = frames[key].copy()
                        cv2.putText(img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (255, 255, 0), 2)
                        img_list.append(img)

                    if img_list:
                        try:
                            # Combine horizontally
                            if len(img_list) > 1:
                                heights = [img.shape[0] for img in img_list]
                                if len(set(heights)) > 1:
                                    target_h = min(heights)
                                    resized = []
                                    for img in img_list:
                                        if img.shape[0] != target_h:
                                            aspect = img.shape[1] / img.shape[0]
                                            target_w = int(target_h * aspect)
                                            resized.append(cv2.resize(img, (target_w, target_h)))
                                        else:
                                            resized.append(img)
                                    combined = np.hstack(resized)
                                else:
                                    combined = np.hstack(img_list)
                            else:
                                combined = img_list[0]

                            # Add status text
                            cv2.putText(combined, "Real-to-Sim Mirror",
                                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                       (0, 255, 0), 2)

                            # Display joint positions
                            joint_text = f"Joints: [{', '.join([f'{x:.1f}' for x in robot_q])}]"
                            cv2.putText(combined, joint_text,
                                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (255, 255, 255), 1)

                            cv2.imshow("Real Robot Cameras", combined)

                            if cv2.waitKey(1) == ord('q'):
                                break

                        except Exception as e:
                            logger.debug(f"Display error: {e}")

                # Sync viewer
                viewer.sync()
                step_count += 1

                # Rate limiting
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'sampler' in locals():
            sampler.stop()
            sampler.join(timeout=2.0)

        if 'robot' in locals() and robot.IsConnected():
            robot.DeactivateRobot()
            robot.Disconnect()

        if 'camera_mgr' in locals():
            camera_mgr.close()

        cv2.destroyAllWindows()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
