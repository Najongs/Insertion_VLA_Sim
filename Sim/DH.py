import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

model = mujoco.MjModel.from_xml_path("meca_scene22.xml")

def extract_dh_from_xml(model, body_name):
    # 1. XML에 적힌 상대 위치(pos)와 회전(quat) 가져오기
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = model.body_pos[body_id]
    quat = model.body_quat[body_id] # [w, x, y, z]

    # 2. 4x4 변환 행렬 구성
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    
    # 3. DH 파라미터 매핑 (간소화 버전)
    d = pos[2] 
    a = np.sqrt(pos[0]**2 + pos[1]**2)
    
    # 4. Alpha(축 비틀림) 계산 및 Degree 변환
    new_z = rot @ np.array([0, 0, 1])
    alpha_rad = np.arctan2(new_z[1], new_z[2])
    alpha_deg = np.rad2deg(alpha_rad) # Radian -> Degree 변환

    return {"alpha_deg": alpha_deg, "a": a, "d": d}

# 각 링크별로 루프를 돌며 수치 추출
print(f"{'Link Name':<12} | {'Alpha (deg)':<12} | {'a (mm)':<10} | {'d (mm)':<10}")
print("-" * 55)

for i in range(1, 7):
    name = f"{i}_Link"
    params = extract_dh_from_xml(model, name)
    print(f"{name:<12} | {params['alpha_deg']:>11.2f}° | {params['a']*1000:>8.2f} | {params['d']*1000:>8.2f}")