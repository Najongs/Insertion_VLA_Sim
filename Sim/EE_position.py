import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. 모델 로드
model = mujoco.MjModel.from_xml_path('meca_add.xml')
data = mujoco.MjData(model)

# 2. 순방향 운동학(Forward Kinematics) 업데이트
mujoco.mj_forward(model, data)

def get_info(model, data, name):
    try:
        # 바디(Body)에서 이름 찾기
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            pos = data.xpos[bid]
            mat = data.xmat[bid].reshape(3, 3)
            
            # 오일러 각도 추출
            r = R.from_matrix(mat)
            euler = r.as_euler('xyz', degrees=True)
            
            # [수정됨] 보정 로직 제거됨 (Raw Euler Angle 출력)
            
            # 0.0000001과 같은 부동소수점 노이즈만 0으로 처리 (가독성 위함)
            euler = np.where(np.abs(euler) < 1e-10, 0, euler)
            return pos, euler
            
    except Exception:
        pass
    
    # 사이트(Site)에서 이름 찾기 (바늘 끝 등)
    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid != -1:
            pos = data.site_xpos[sid]
            mat = data.site_xmat[sid].reshape(3, 3)
            r = R.from_matrix(mat)
            euler = r.as_euler('xyz', degrees=True)
            
            # [수정됨] 보정 로직 제거됨 (Raw Euler Angle 출력)
            
            euler = np.where(np.abs(euler) < 1e-10, 0, euler)
            return pos, euler
    except Exception:
        pass

    return None, None

# 대상 리스트
target_bodies = ["base_link", "1_Link", "2_Link", "3_Link", "4_Link", "5_Link", "6_Link"]

print(f"{'Target Name':<20} | {'Position (x, y, z) [mm]':<30} | {'Rotation (rx, ry, rz)':<20}")
print("-" * 85)

for name in target_bodies:
    pos, ori = get_info(model, data, name)
    
    if pos is not None:
        p_mm = pos * 1000
        print(f"{name:<20} | {p_mm[0]:8.2f}, {p_mm[1]:8.2f}, {p_mm[2]:8.2f} | {ori[0]:6.1f}, {ori[1]:6.1f}, {ori[2]:6.1f}")
    else:
        # base_link는 world와 합쳐져 에러가 날 수 있으므로 건너뜀
        if name != "base_link":
            print(f"{name:<20} | [Error] Name not found")