import h5py
import numpy as np

# 데이터 파일 로드
file_path = "/home/najo/NAS/VLA/Insertion_VLA_Sim/Sim/collected_data_sim_clean/episode_20260129_214849.h5"

with h5py.File(file_path, 'r') as f:
    print("=== HDF5 파일 구조 ===")
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

    print("\n=== Joint Position (qpos) 확인 ===")
    qpos = f['observations/qpos'][:]
    print(f"Shape: {qpos.shape}")
    print(f"Unit: degrees (저장 시 rad2deg 적용)")
    print(f"\nFirst 5 timesteps:")
    print(qpos[:5])

    print(f"\n=== Joint Position 통계 ===")
    print(f"Min values per joint: {np.min(qpos, axis=0)}")
    print(f"Max values per joint: {np.max(qpos, axis=0)}")
    print(f"Mean values per joint: {np.mean(qpos, axis=0)}")
    print(f"Std values per joint: {np.std(qpos, axis=0)}")

    print(f"\n=== Joint Position 변화량 확인 ===")
    qpos_diff = np.diff(qpos, axis=0)
    print(f"Max change per step (degrees): {np.max(np.abs(qpos_diff), axis=0)}")
    print(f"Mean change per step (degrees): {np.mean(np.abs(qpos_diff), axis=0)}")

    print(f"\n=== 예상 범위와 비교 ===")
    print("일반적인 로봇 joint 범위: -180° ~ 180°")
    print(f"현재 데이터가 이 범위 내에 있는지: {np.all(qpos >= -180) and np.all(qpos <= 180)}")

    # Home pose 확인 (코드에서 home_pose = [30, -20, 20, 0, 30, 60] degrees)
    print(f"\n=== Home Pose와 비교 ===")
    expected_home = np.array([30.0, -20.0, 20.0, 0.0, 30.0, 60.0])
    print(f"Expected home pose (degrees): {expected_home}")
    print(f"First recorded qpos (degrees): {qpos[0]}")
    print(f"Difference: {qpos[0] - expected_home}")
