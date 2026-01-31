"""

python save_data_with_real.py --episodes 1 --no-randomize-phantom-pos

"""


import os
os.environ['MUJOCO_GL'] = 'egl'

import time
import threading
import pathlib
import datetime
import logging
import argparse
import numpy as np
import cv2
import h5py
import mujoco
import mecademicpy.robot as mdr

# === Configuration ===
ROBOT_ADDRESS = "192.168.0.100"
MODEL_PATH = "meca_add.xml"
SAVE_DIR = "collected_data_with_real"
IMG_WIDTH = 640
IMG_HEIGHT = 480
CONTROL_HZ = 15
TARGET_INSERTION_DEPTH = 0.0275
TRAJ_DURATION = 15.0
HOME_JOINTS_DEG = (30, -20, 20, 0, 30, 60)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Recorder Class ===
class SimRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.recording = False
        self.is_saving = False

    def start(self):
        if self.is_saving:
            return
        self.buffer = []
        self.recording = True
        logger.info("\ud83d\udd34 Recording started")

    def add(self, frames, qpos, ee_pose, action, timestamp, phase, sensor_dist):
        if not self.recording:
            return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "phase": phase,
            "sd": sensor_dist,
        })

    def save_async(self, episode_idx=None):
        if not self.buffer:
            return
        data_snapshot = self.buffer
        self.buffer = []
        self.recording = False
        self.is_saving = True

        def worker(data, filename):
            try:
                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")

                    q_data = np.array([x['q'] for x in data], dtype=np.float32)
                    p_data = np.array([x['p'] for x in data], dtype=np.float32)
                    act_data = np.array([x['act'] for x in data], dtype=np.float32)
                    ts_data = np.array([x['ts'] for x in data], dtype=np.float32)
                    phase_data = np.array([x['phase'] for x in data], dtype=np.int32)
                    sensor_data = np.array([x['sd'] for x in data], dtype=np.float32)

                    obs.create_dataset("qpos", data=q_data, compression="gzip")
                    obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                    obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip")
                    f.create_dataset("action", data=act_data, compression="gzip")
                    f.create_dataset("timestamp", data=ts_data, compression="gzip")
                    f.create_dataset("phase", data=phase_data, compression="gzip")

                    first_imgs = data[0]["imgs"]
                    for cam_name in first_imgs.keys():
                        jpeg_list = []
                        for step in data:
                            img = step["imgs"][cam_name]
                            success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success:
                                jpeg_list.append(buf.flatten())
                            else:
                                jpeg_list.append(np.zeros(1, dtype=np.uint8))

                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                        for i, code in enumerate(jpeg_list):
                            dset[i] = code
            except Exception as e:
                logger.error(f"\u274c Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if episode_idx is None:
            fname = self.out / f"episode_{timestamp}.h5"
        else:
            fname = self.out / f"episode_{episode_idx:03d}_{timestamp}.h5"
        t = threading.Thread(target=worker, args=(data_snapshot, fname))
        t.start()

    def discard(self):
        self.buffer = []
        self.recording = False
        logger.warning("\ud83d\uddd1\ufe0f Recording discarded")

# === Real Robot Sampler ===
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
        logger.info("\ud83e\udd16 Starting robot state sampler...")
        next_t = time.time()
        while not self.stop_evt.is_set():
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())
                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
            except Exception:
                pass
            next_t += self.dt
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

# === Helper Functions ===
def _pose_deg_to_rad(pose_deg):
    pose = np.array(pose_deg, dtype=np.float32)
    pose[3:6] = np.deg2rad(pose[3:6])
    return pose


def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def randomize_phantom_pos(model, data, phantom_id, rot_id):
    if phantom_id == -1 or rot_id == -1:
        return
    offset_x = np.random.uniform(-0.1, 0.1)
    offset_y = np.random.uniform(-0.05, 0.03)
    model.body_pos[phantom_id] = np.array([offset_x, offset_y, 0.0])

    random_angle_deg = np.random.uniform(-30, 30)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    mujoco.mj_forward(model, data)


def _parse_args():
    parser = argparse.ArgumentParser(description="Record real robot + simulation dataset.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument(
        "--start-delay-sec",
        type=float,
        default=2.0,
        help="Delay before each episode starts recording (seconds).",
    )
    parser.add_argument(
        "--randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_true",
        default=True,
        help="Enable phantom position randomization.",
    )
    parser.add_argument(
        "--no-randomize-phantom-pos",
        dest="randomize_phantom_pos",
        action="store_false",
        help="Disable phantom position randomization.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    logger.info(f"\ud83d\udd04 Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        n_motors = model.nu
        dof = model.nv
    except Exception as e:
        logger.warning(f"\u26a0\ufe0f Warning: Some IDs not found: {e}")
        tip_id = -1
        back_id = -1
        target_entry_id = -1
        target_depth_id = -1
        phantom_body_id = -1
        rotating_id = -1
        link6_id = -1
        n_motors = model.nu
        dof = model.nv

    recorder = SimRecorder(SAVE_DIR)
    home_pose_rad = np.deg2rad(np.array(HOME_JOINTS_DEG, dtype=np.float32))

    try:
        logger.info(f"\ud83d\udd0c Connecting to robot at {ROBOT_ADDRESS}...")
        robot = mdr.Robot()
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error("\u274c Failed to connect to robot")
            return

        logger.info("\u2705 Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)

        logger.info("\ud83c\udfe0 Moving to home position...")
        robot.MoveJoints(*HOME_JOINTS_DEG)
        robot.WaitIdle()
        logger.info("\u2705 Robot ready!")

        sampler = RtSampler(robot, rate_hz=100)
        sampler.start()

        last_record_time = 0.0
        last_real_ee = None
        record_dt = 1.0 / float(CONTROL_HZ)

        for episode_idx in range(int(args.episodes)):
            while recorder.is_saving:
                time.sleep(0.1)

            if args.start_delay_sec > 0:
                logger.info(f"⏱️ Waiting {args.start_delay_sec:.2f}s before episode {episode_idx}...")
                time.sleep(args.start_delay_sec)

            recorder.start()
            last_real_ee = None

            mujoco.mj_resetData(model, data)
            data.qpos[:6] = home_pose_rad
            if args.randomize_phantom_pos:
                randomize_phantom_pos(model, data, phantom_body_id, rotating_id)
            mujoco.mj_forward(model, data)

            task_state = 1
            traj_start_time = data.time
            insertion_started = False
            accumulated_depth = 0.0
            align_timer = 0
            traj_initialized = False
            hold_start_time = None

            p_entry = data.site_xpos[target_entry_id].copy() if target_entry_id >= 0 else np.zeros(3)
            p_depth = data.site_xpos[target_depth_id].copy() if target_depth_id >= 0 else np.zeros(3)
            start_tip = data.site_xpos[tip_id].copy() if tip_id >= 0 else np.zeros(3)
            start_back = data.site_xpos[back_id].copy() if back_id >= 0 else np.zeros(3)
            needle_len = np.linalg.norm(start_tip - start_back) if tip_id >= 0 and back_id >= 0 else 0.0

            target_tip_pos = start_tip.copy()
            target_back_pos = start_back.copy()

            success = False
            steps_per_cycle = max(1, int(round(record_dt / model.opt.timestep)))

            while True:
                loop_start = time.time()

                for _ in range(steps_per_cycle):
                    t_curr = data.time
                    curr_tip = data.site_xpos[tip_id].copy() if tip_id >= 0 else target_tip_pos
                    curr_back = data.site_xpos[back_id].copy() if back_id >= 0 else target_back_pos

                    if task_state == 1:
                        if not traj_initialized:
                            traj_start_time = t_curr
                            start_tip_pos = curr_tip.copy()
                            start_back_pos = curr_back.copy()
                            traj_initialized = True
                        progress = smooth_step((t_curr - traj_start_time) / TRAJ_DURATION)
                        axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                        goal_tip = p_entry - (axis_dir * 0.0001)
                        goal_back = p_entry - (axis_dir * (0.0001 + needle_len))
                        target_tip_pos = (1 - progress) * start_tip_pos + progress * goal_tip
                        target_back_pos = (1 - progress) * start_back_pos + progress * goal_back
                        if progress >= 1.0:
                            if np.linalg.norm(curr_tip - goal_tip) < 0.002:
                                align_timer += 1
                            else:
                                align_timer = 0
                            if align_timer > 20:
                                task_state = 2
                                insertion_started = False
                    elif task_state == 2:
                        if not insertion_started:
                            phase3_base_tip = curr_tip.copy()
                            insertion_started = True
                            accumulated_depth = 0.0
                            hold_start_time = None

                        axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
                        if accumulated_depth < TARGET_INSERTION_DEPTH:
                            accumulated_depth += 0.0000025
                            target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                            target_back_pos = target_tip_pos - (axis_dir * needle_len)
                            if accumulated_depth >= TARGET_INSERTION_DEPTH:
                                hold_start_time = data.time
                        else:
                            if hold_start_time is None:
                                hold_start_time = data.time
                            target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                            target_back_pos = target_tip_pos - (axis_dir * needle_len)
                            if data.time - hold_start_time >= 1.0:
                                success = True
                                break

                    if tip_id >= 0 and back_id >= 0:
                        err_tip = target_tip_pos - curr_tip
                        err_back = target_back_pos - curr_back
                        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)

                        offset_angle = np.deg2rad(180 + 30)
                        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
                        current_side_vec = tip_rot_mat @ offset_local_vec

                        needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
                        target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
                        if np.linalg.norm(target_side_vec) > 1e-3:
                            target_side_vec = target_side_vec / np.linalg.norm(target_side_vec)
                        else:
                            target_side_vec = np.array([1, 0, 0])
                        err_roll = np.cross(current_side_vec, target_side_vec)

                        jac_tip_full = np.zeros((6, dof))
                        jac_back = np.zeros((3, dof))
                        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
                        mujoco.mj_jacSite(model, data, jac_back, None, back_id)

                        J_p1 = jac_tip_full[:3, :n_motors]
                        e_p1 = (err_tip * 50.0)
                        if np.linalg.norm(e_p1) > 1.0:
                            e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
                        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
                        dq_p1 = J_p1_pinv @ e_p1

                        P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
                        J_p2_proj = jac_back[:, :n_motors] @ P_null_1
                        dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)

                        P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
                        J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
                        dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))

                        data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * 0.5

                    mujoco.mj_step(model, data)

                if success:
                    break

                # Send joint targets to real robot and record
                target_qpos_deg = np.rad2deg(data.qpos[:n_motors].copy()).astype(np.float32)
                try:
                    robot.MoveJoints(*[float(x) for x in target_qpos_deg[:6]])
                except Exception as e:
                    logger.debug(f"MoveJoints failed: {e}")

                robot_q, robot_p = sampler.get_latest_data()
                real_ee = _pose_deg_to_rad(robot_p)

                frames = {}
                for cam_name in ["side_camera", "tool_camera", "top_camera"]:
                    renderer.update_scene(data, camera=cam_name)
                    frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

                if last_real_ee is None:
                    last_real_ee = real_ee.copy()
                delta_ee_action = real_ee - last_real_ee

                current_sensor_dist = -1.0
                if tip_id >= 0 and back_id >= 0 and link6_id >= 0:
                    curr_tip = data.site_xpos[tip_id].copy()
                    curr_back = data.site_xpos[back_id].copy()
                    needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
                    dist_to_surface = mujoco.mj_ray(model, data, curr_tip, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
                    current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0

                recorder.add(
                    frames,
                    target_qpos_deg[:6],
                    real_ee,
                    delta_ee_action,
                    time.time(),
                    task_state,
                    current_sensor_dist,
                )
                last_real_ee = real_ee.copy()

                elapsed = time.time() - loop_start
                if elapsed < record_dt:
                    time.sleep(record_dt - elapsed)

            if success:
                recorder.save_async(episode_idx=episode_idx)
            else:
                recorder.discard()

            # Return to home between episodes, then wait before next recording
            if episode_idx < int(args.episodes) - 1:
                try:
                    logger.info("🏠 Returning to home position between episodes...")
                    robot.MoveJoints(*HOME_JOINTS_DEG)
                    robot.WaitIdle()
                except Exception as e:
                    logger.warning(f"Failed to return home between episodes: {e}")

    except KeyboardInterrupt:
        logger.info("\n\u23f9\ufe0f Interrupted by user")
    except Exception as e:
        logger.error(f"\u274c Error: {e}", exc_info=True)
    finally:
        if 'sampler' in locals():
            sampler.stop()
            sampler.join(timeout=2.0)

        if 'robot' in locals() and robot.IsConnected():
            robot.DeactivateRobot()
            robot.Disconnect()

        logger.info("\u2705 Cleanup complete")


if __name__ == "__main__":
    main()
