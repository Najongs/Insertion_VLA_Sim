import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

# 1. Load Model
model_path = "meca_scene22.xml"
print(f"Loading Model: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# 2. Get IDs
try:
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    viz_tip_tgt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "viz_target_tip")

    phantom_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly") # 이동용
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")   # 회전용

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link") 

    dof = model.nv
except Exception as e:
    print(f"[Error] XML ID Load Failed: {e}")
    raise e

# === Control Parameters ===
damping = 1e-3
base_speed = 1.0 # 기본 속도

TARGET_DISTANCE_FROM_ENTRY = 0.0001
TARGET_INSERTION_DEPTH = 0.022
COAXIAL_TOLERANCE = 50e-6
SENSOR_THRESHOLD = 0.010  # 10mm 

# State Variables
task_state = 1
align_timer = 0
insertion_started = False
accumulated_depth = 0.0
phase3_base_tip = np.zeros(3)

# === Trajectory Variables (부드러운 움직임을 위한 변수) ===
traj_initialized = False
traj_start_time = 0.0
TRAJ_DURATION = 2.0  # 정렬까지 걸리는 시간 (초)
start_tip_pos = np.zeros(3)
start_back_pos = np.zeros(3)

def randomize_phantom_pos(model, data):
    # 1. 위치 이동 (Translation)
    offset_x = np.random.uniform(-0.1, 0.1)
    offset_y = np.random.uniform(-0.05, 0.03)
    offset_z = 0.0 
    new_pos = np.array([offset_x, offset_y, offset_z])
    model.body_pos[phantom_pos_id] = new_pos
    
    # 2. 회전 (Rotation)
    random_angle_deg = np.random.uniform(-30, 30)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rotating_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    

# === Helper: SmoothStep Function ===
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

randomize_phantom_pos(model, data)

print("Started: Smooth Trajectory Mode with Ray-Casting Sensor")

home_pose = np.array([0.5, 0.0, 0.0, 0.0, -0.5, 0.0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    data.qpos[:6] = home_pose  # 1. 관절 값 대입
    mujoco.mj_forward(model, data)
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])

    step_count = 0
    
    # Init Viz
    target_tip_pos = data.site_xpos[tip_id].copy()
    
    while viewer.is_running():
        step_start = time.time()
        curr_time = data.time
        
        # Current States
        p_entry = data.site_xpos[target_entry_id].copy()
        p_depth = data.site_xpos[target_depth_id].copy()
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        p_sensor = data.site_xpos[sensor_id].copy()

        # Target Vectors
        axis_vec = p_depth - p_entry
        axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)

        # ==========================================
        # [NEW] Ray-Casting Sensor Logic
        # ==========================================
        # 1. 바늘의 진행 방향 벡터 계산 (Back -> Tip)
        needle_dir = (curr_tip - curr_back)
        needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)


        geom_id_out = np.zeros(1, dtype=np.int32) # 결과(부딪힌 지오메트리 ID)를 받을 변수
        dist_to_surface = mujoco.mj_ray(model, data, p_sensor, needle_dir, 
                                        None, 1, link6_id, geom_id_out)

        # 3. 감지 결과 처리
        obstacle_detected = False
        sensor_msg = "Sensor: Clear"
        
        # dist_to_surface가 -1이면 아무것도 없는 것, 0 이상이면 거리
        if 0 <= dist_to_surface <= SENSOR_THRESHOLD:
            obstacle_detected = True
            sensor_msg = f"OBSTACLE! ({dist_to_surface*1000:.1f}mm)"
        
        # ==========================================

        # 기본 상태 메시지 및 색상
        if task_state == 1:
            status_color = (255, 0, 255)
            msg = "Aligning..."
        elif task_state == 2:
            status_color = (0, 255, 0)
            msg = "Inserting..."
        else:
            status_color = (0, 0, 255)
            msg = "Finished"

        # 장애물 감지 시 경고 메시지로 덮어쓰기
        if obstacle_detected:
            status_color = (0, 165, 255) # Orange Warning
            msg = sensor_msg

        # 속도 설정 (장애물 감지 시 감속)
        current_speed = 0.1 if obstacle_detected else base_speed

        # === State Machine ===
        if task_state == 1:
            # [Phase 1: Smooth Approach & Align]
            
            if not traj_initialized:
                traj_start_time = curr_time
                start_tip_pos = curr_tip.copy()
                start_back_pos = curr_back.copy()
                traj_initialized = True
                print(">>> Trajectory Generated.")

            elapsed_t = curr_time - traj_start_time
            raw_progress = elapsed_t / TRAJ_DURATION
            alpha = smooth_step(raw_progress)
            
            goal_tip = p_entry - (axis_dir * TARGET_DISTANCE_FROM_ENTRY)
            goal_back = goal_tip - (axis_dir * needle_len)
            
            # Interpolation
            target_tip_pos = (1 - alpha) * start_tip_pos + alpha * goal_tip
            target_back_pos = (1 - alpha) * start_back_pos + alpha * goal_back
            
            if not obstacle_detected: # 장애물이 없을 때만 진행 상태 메시지 업데이트
                msg = f"Aligning... {alpha*100:.1f}%"

            if raw_progress >= 1.0:
                dist_error = np.linalg.norm(curr_tip - goal_tip)
                
                vec_entry_to_tip = curr_tip - p_entry
                proj_point = p_entry + (np.dot(vec_entry_to_tip, axis_dir) * axis_dir)
                dist_coaxial = np.linalg.norm(curr_tip - proj_point)
                
                if dist_error < 0.002 and dist_coaxial < COAXIAL_TOLERANCE:
                    align_timer += 1
                    if not obstacle_detected: msg = "Holding Alignment..."
                else:
                    align_timer = 0
                
                if align_timer > 20:
                    task_state = 2
                    insertion_started = False
                    print(">>> Alignment Complete. Starting Insertion.")

        elif task_state == 2:
            # [Phase 2: Insertion]
            if not obstacle_detected: msg = f"Inserting ({accumulated_depth*1000:.1f}mm)"
            
            if not insertion_started:
                phase3_base_tip = p_entry - (axis_dir * TARGET_DISTANCE_FROM_ENTRY)
                insertion_started = True
                accumulated_depth = 0.0

            step_z = 0.000025
            
            # 장애물이 감지되어도 멈추지 않고 천천히 가려면 아래 주석 해제, 멈추려면 조건문 사용
            # if not obstacle_detected: 
            accumulated_depth += step_z
            
            target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
            target_back_pos = target_tip_pos - (axis_dir * needle_len)
            
            if accumulated_depth >= TARGET_INSERTION_DEPTH:
                task_state = 3
                print(">>> Finished.")

        elif task_state == 3:
            msg = "FINISHED"

        # === Stacked IK Solver ===
        data.site_xpos[viz_tip_tgt_id] = target_tip_pos
        
        # Position Errors
        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back
        
        # Roll Correction
        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
        current_side_vec = tip_rot_mat @ np.array([1, 0, 0])
        needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
        if np.linalg.norm(target_side_vec) < 1e-3: target_side_vec = np.array([1, 0, 0])
        else: target_side_vec /= np.linalg.norm(target_side_vec)
        err_roll = np.cross(current_side_vec, target_side_vec)
        
        # Jacobian
        jac_tip_full = np.zeros((6, dof))
        jac_back = np.zeros((3, dof))
        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
        mujoco.mj_jacSite(model, data, jac_back, None, back_id)
        
        # Stacked Calculation
        J_p1 = jac_tip_full[:3] # Tip Pos
        e_p1 = err_tip * 50.0   # Tip Error Gain
        
        if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
            
        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
        dq_p1 = J_p1_pinv @ e_p1
        P_null_1 = np.eye(dof) - (J_p1_pinv @ J_p1)
        
        J_p2 = jac_back # Back Pos
        e_p2 = err_back * 50.0
        J_p2_proj = J_p2 @ P_null_1
        J_p2_pinv = np.linalg.pinv(J_p2_proj, rcond=1e-4)
        dq_p2 = J_p2_pinv @ (e_p2 - J_p2 @ dq_p1)
        P_null_2 = P_null_1 - (J_p2_pinv @ J_p2_proj)
        
        J_p3 = jac_tip_full[3:] # Roll
        e_p3 = err_roll * 10.0
        J_p3_proj = J_p3 @ P_null_2
        J_p3_pinv = np.linalg.pinv(J_p3_proj, rcond=1e-4)
        dq_p3 = J_p3_pinv @ (e_p3 - J_p3 @ (dq_p1 + dq_p2))
        
        dq = dq_p1 + dq_p2 + dq_p3
        
        data.ctrl[:] = data.qpos[:dof] + (dq * current_speed)
        mujoco.mj_step(model, data)
        step_count += 1
        
        if step_count % 67 == 0:
            frames = []
            for cam in ["side_camera", "tool_camera", "top_camera"]:
                renderer.update_scene(data, camera=cam)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            combined = np.hstack(frames)
            
            # 텍스트 오버레이
            cv2.putText(combined, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Smooth Trajectory & Ray Sensor", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                mujoco.mj_resetData(model, data)
                data.qpos[:6] = home_pose
                randomize_phantom_pos(model, data)
                mujoco.mj_forward(model, data)
                task_state = 1
                traj_initialized = False
                insertion_started = False
                accumulated_depth = 0.0

        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
 
cv2.destroyAllWindows()