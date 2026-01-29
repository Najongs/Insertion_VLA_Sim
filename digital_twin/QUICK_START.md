# Digital Twin Quick Start Guide

## 30-Second Overview

**Sim-to-Real**: Simulation drives robot → Test policies safely
**Real-to-Sim**: Robot drives simulation → Visualize real movements

## Quick Launch

### Sim-to-Real (Simulation Controls Robot)

```bash
cd /home/najo/NAS/VLA/Insertion_VLA_Sim/digital_twin
python sim_to_real.py
```

- Robot follows simulation automatically
- Use for: Testing algorithms, autonomous control
- Safety: Keep E-stop ready!

### Real-to-Sim (Robot Controls Simulation)

```bash
cd /home/najo/NAS/VLA/Insertion_VLA_Sim/digital_twin
python real_to_sim.py
```

- Use gamepad to control real robot
- Simulation mirrors movements
- Use for: Visualization, data collection

## Key Differences

| Feature | Sim-to-Real | Real-to-Sim |
|---------|-------------|-------------|
| **Control Source** | Simulation (MuJoCo) | Gamepad |
| **Motion Driver** | Autonomous/Manual in sim | Gamepad → Real robot |
| **Primary Use** | Test policies | Visualize operations |
| **Robot Behavior** | Follows sim commands | Controlled by user |
| **Sync Direction** | Sim → Robot | Robot → Sim |
| **Camera Display** | No | Yes (real cameras) |
| **Data Collection** | Not implemented | Can be added |
| **Risk Level** | Medium (robot moves) | Low (user controls) |

## Pre-Flight Checklist

- [ ] Robot powered on and connected (192.168.0.100)
- [ ] Workspace clear of obstacles
- [ ] E-stop accessible
- [ ] Network connection stable
- [ ] (Real-to-Sim only) Gamepad connected
- [ ] (Real-to-Sim only) OAK cameras connected

## Common Commands

### Both Modes
- `Q` - Quit program
- Mouse - Rotate simulation view

### Sim-to-Real Specific
- `R` - Reset to home position
- Arrow keys - Move simulation (robot follows)

### Real-to-Sim Specific
- `Y button` - Go home
- `START button` - Exit
- `Left stick` - X-Y movement
- `Right stick` - Rotation
- `LT/RT triggers` - Z-axis
- `BACK/SELECT` - Switch control mode

## Troubleshooting (10-Second Fixes)

| Problem | Solution |
|---------|----------|
| Can't connect | `ping 192.168.0.100` |
| Robot won't move | Check E-stop, reset errors |
| Lag/delay | Reduce `SYNC_FREQUENCY = 10` |
| No gamepad | Connect before running |
| No cameras | Script continues without them |
| Black screen | `export MUJOCO_GL=egl` |

## Configuration Files

### Change Robot IP
```python
# In sim_to_real.py or real_to_sim.py
ROBOT_ADDRESS = "192.168.0.100"  # ← Change this
```

### Change Sync Speed
```python
# In sim_to_real.py
SYNC_FREQUENCY = 15  # Hz (lower = more stable)
```

### Change Model
```python
MODEL_PATH = "../Sim/meca_scene22.xml"  # ← Your model
```

## Expected Console Output

### Successful Startup
```
=== Digital Twin: Sim-to-Real ===
📦 Loading model: ../Sim/meca_scene22.xml
🔌 Connecting to robot at 192.168.0.100...
✅ Robot connected! Activating and homing...
✅ Robot ready for sync!
```

### Running Status
```
🔄 Sync Status | Sim Joints: [30.0, -20.0, ...] | Error: 0.15mm
```

### Clean Shutdown
```
✅ Shutdown complete
```

## Safety Rules (Read This!)

1. ⚠️ **Always** keep E-stop accessible
2. ⚠️ **Never** run unattended
3. ⚠️ **Start** with small movements
4. ⚠️ **Test** in simulation first
5. ⚠️ **Clear** workspace before starting

## Next Steps

1. **Read full README.md** for detailed explanation
2. **Test with small movements** to verify calibration
3. **Adjust gains** if robot is too fast/slow
4. **Integrate your policy** (for sim-to-real)
5. **Add data recording** (for real-to-sim)

## File Structure

```
digital_twin/
├── sim_to_real.py      # Simulation → Robot
├── real_to_sim.py      # Robot → Simulation
├── Robot_action.py     # Original teleoperation
├── README.md           # Full documentation
└── QUICK_START.md      # This file
```

## Performance Targets

- **Latency**: < 100ms
- **Update Rate**: 15 Hz control, 100 Hz state
- **Position Error**: < 1mm
- **Rotation Error**: < 0.5°

## Getting Help

1. Check `README.md` for detailed info
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Monitor sync error in console output
4. Test robot connection separately first

---

**Ready?** Run one of the commands above and start your digital twin! 🚀
