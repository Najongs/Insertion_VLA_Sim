# Simulation to Real Robot Coordinate Calibration

## Overview

This calibration system uses **Point Cloud Matching** to find the transformation between simulation and real robot coordinates. It computes a single 4×4 transformation matrix that accounts for:
- Base frame position differences
- DH parameter mismatches
- TCP (needle tip) length differences
- Orientation convention differences

## 🚀 Quick Start

### Step 1: Collect Simulation Data

```bash
python collect_calibration_data.py
```

This generates `calibration_data.json` with 10 predefined joint angles and their corresponding simulation EE poses.

**Output example:**
```
Joint Angles: [0, 0, 0, 0, 0, 0]
Sim EE Pose: [15.54, 179.00, 222.38, 90.00, -60.00, 0.00]
```

### Step 2: Collect Real Robot Data

**On your REAL ROBOT**, move to the SAME joint angles and record the EE poses:

```python
# Example joint angles to test:
[0, 0, 0, 0, 0, 0]           # Home
[30, 0, 0, 0, 0, 0]          # J1 = 30°
[45, 0, 0, 0, 0, 0]          # J1 = 45°
[0, 30, 0, 0, 0, 0]          # J2 = 30°
... (see calibration_data.json for full list)
```

**Edit `calibration_data.json`** and fill in the `real_poses` field:

```json
{
  "real_poses": [
    [190.0, 0.0, 308.0, 0.0, 90.0, 0.0],    # Home pose from real robot
    [164.5, 95.0, 308.0, -90.0, 60.0, 90.0], # J1=30 from real robot
    ...
  ]
}
```

### Step 3: Compute Transformation Matrix

```bash
python compute_transformation.py
```

This uses **Kabsch algorithm (SVD)** to compute the optimal transformation.

**Output:**
- `transformation_position.npy`: 4×4 position transformation matrix
- `transformation_orientation.npy`: 3×3 orientation transformation matrix
- Error metrics (RMSE, max error)

**Expected output:**
```
Position RMSE: 3.5 mm
Position Max Error: 7.2 mm
Orientation RMSE: 2.1 deg
Orientation Max Error: 5.8 deg
```

### Step 4: Use Transformed Coordinates

The transformation is **automatically applied** in `Save_dataset.py`:

```python
from transformation_utils import get_transform

transform = get_transform()

# EE poses are automatically transformed to real robot coordinates
ee_pose_real = transform.transform_pose(ee_pose_sim)

# Actions (deltas) are also transformed
action_real = transform.transform_action(action_sim)
```

**Now run data collection:**

```bash
python Save_dataset.py
```

All saved data will be in **real robot coordinate frame**!

### Step 5: Validation

Use `data_replay.py` to visualize collected data:

```bash
python data_replay.py
```

**Verify:**
- EE positions match real robot teach pendant values
- Orientation values are consistent with real robot conventions
- Actions (deltas) produce correct movements

## 📊 Mathematical Background

### Position Transformation

Uses **Kabsch algorithm**:

```
P_real = R @ P_sim + t

Where:
- R: 3×3 rotation matrix
- t: 3×1 translation vector
- Computed via SVD to minimize: ||P_real - (R @ P_sim + t)||²
```

### Orientation Transformation

Transforms rotation matrices:

```
R_real = R_ori @ R_sim

Where:
- R_ori: 3×3 orientation transformation matrix
- Computed via least-squares optimization
```

### Combined 4×4 Transformation

```
T = [R  t]  (4×4 homogeneous matrix)
    [0  1]
```

## 🔧 Files Created

| File | Purpose |
|------|---------|
| `collect_calibration_data.py` | Collect simulation data |
| `calibration_data.json` | Store sim/real pose pairs |
| `compute_transformation.py` | Calculate transformation matrices |
| `transformation_utils.py` | Transform functions |
| `transformation_position.npy` | Position transform matrix (4×4) |
| `transformation_orientation.npy` | Orientation transform matrix (3×3) |
| `Save_dataset.py` | **Modified** - uses transformation |

## ⚠️ Important Notes

1. **Joint angles must match exactly** between simulation and real robot
2. **Use at least 5-10 calibration points** covering the workspace
3. **Check error metrics** after computing transformation:
   - Position RMSE < 10 mm is acceptable
   - Orientation RMSE < 10° is acceptable
4. **Re-calibrate** if you change:
   - Robot base position
   - TCP/tool geometry
   - Simulation model parameters

## 🎯 Benefits

✅ **No manual tuning** - automatic optimization via SVD
✅ **Single source of truth** - one transformation matrix for all
✅ **Accounts for all errors** - base frame, DH params, TCP length
✅ **Easy to validate** - clear error metrics
✅ **Production ready** - data in real robot coordinates

## 📝 Troubleshooting

### Error: "Transformation matrices not found"

Run calibration steps 1-3 first.

### High RMSE (>20mm or >20°)

- Check if real_poses in calibration_data.json are correct
- Verify joint angles match between sim and real robot
- Add more calibration points
- Check for gimbal lock in orientations

### Data looks wrong in replay

- Verify transformation matrices were loaded (check console output)
- Re-run compute_transformation.py and check error metrics
- Ensure calibration_data.json has correct real robot values

## 📚 References

- **Kabsch algorithm**: W. Kabsch, "A solution for the best rotation to relate two sets of vectors", Acta Crystallographica, 1976
- **Rigid body transformation**: Horn, "Closed-form solution of absolute orientation using unit quaternions", JOSA, 1987
