#!/usr/bin/env python
"""
Digital Twin Setup Verification Script
Tests all dependencies and connections before running the main programs
"""

import sys
import importlib
from termcolor import colored

def print_status(test_name, passed, message=""):
    if passed:
        print(colored(f"✅ {test_name}", "green"), message)
    else:
        print(colored(f"❌ {test_name}", "red"), message)
    return passed

def test_imports():
    """Test all required Python packages"""
    print("\n" + "="*50)
    print("Testing Python Dependencies")
    print("="*50)

    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'h5py': 'HDF5',
        'mujoco': 'MuJoCo',
        'pygame': 'Pygame',
        'mecademicpy.robot': 'Mecademic API',
        'depthai': 'DepthAI (OAK cameras)',
        'termcolor': 'Termcolor',
    }

    all_passed = True
    for module_name, display_name in packages.items():
        try:
            importlib.import_module(module_name)
            print_status(display_name, True)
        except ImportError as e:
            print_status(display_name, False, f"- Install: pip install {module_name.split('.')[0]}")
            all_passed = False

    return all_passed

def test_mujoco_model():
    """Test MuJoCo model loading"""
    print("\n" + "="*50)
    print("Testing MuJoCo Model")
    print("="*50)

    try:
        import mujoco
        import os

        model_path = "../Sim/meca_scene22.xml"

        if not os.path.exists(model_path):
            print_status("Model file exists", False, f"- File not found: {model_path}")
            return False

        print_status("Model file exists", True, f"- {model_path}")

        model = mujoco.MjModel.from_xml_path(model_path)
        print_status("Model loads successfully", True, f"- DOF: {model.nv}, Bodies: {model.nbody}")

        # Check for required sites
        try:
            tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
            back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
            print_status("Required sites found", True, "- needle_tip, needle_back")
        except:
            print_status("Required sites found", False, "- Missing needle_tip or needle_back sites")
            return False

        return True

    except Exception as e:
        print_status("MuJoCo model test", False, f"- Error: {e}")
        return False

def test_robot_connection():
    """Test connection to Mecademic robot"""
    print("\n" + "="*50)
    print("Testing Robot Connection")
    print("="*50)

    try:
        import mecademicpy.robot as mdr
        import socket

        ROBOT_ADDRESS = "192.168.0.100"

        # First test network reachability
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((ROBOT_ADDRESS, 10000))
            sock.close()

            if result == 0:
                print_status("Network reachable", True, f"- {ROBOT_ADDRESS} is accessible")
            else:
                print_status("Network reachable", False, f"- Cannot reach {ROBOT_ADDRESS}")
                print("  ℹ️  Make sure robot is powered on and connected to network")
                return False
        except Exception as e:
            print_status("Network test", False, f"- Error: {e}")
            return False

        # Try to connect to robot
        print("  ⏳ Attempting robot connection (this may take a few seconds)...")
        robot = mdr.Robot()
        robot.Connect(address=ROBOT_ADDRESS, enable_synchronous_mode=False, timeout=5)

        if robot.IsConnected():
            print_status("Robot connection", True, f"- Connected to {ROBOT_ADDRESS}")

            # Get robot info
            try:
                # Don't activate, just test communication
                model = robot.GetRobotInfo()
                print(f"  ℹ️  Robot Model: {model}")
            except:
                pass

            robot.Disconnect()
            return True
        else:
            print_status("Robot connection", False, "- Failed to connect")
            print("  ℹ️  Check if robot is activated and no other software is connected")
            return False

    except Exception as e:
        print_status("Robot connection test", False, f"- Error: {e}")
        print("  ℹ️  Make sure mecademicpy is installed: pip install mecademicpy")
        return False

def test_gamepad():
    """Test gamepad connection"""
    print("\n" + "="*50)
    print("Testing Gamepad (Optional)")
    print("="*50)

    try:
        import pygame

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()

        if count > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print_status("Gamepad detected", True, f"- {joystick.get_name()}")

            # Test button/axis count
            print(f"  ℹ️  Buttons: {joystick.get_numbuttons()}, Axes: {joystick.get_numaxes()}")
            return True
        else:
            print_status("Gamepad detected", False, "- No gamepad found (required for real_to_sim.py)")
            print("  ℹ️  This is optional - connect gamepad if you want to use real-to-sim mode")
            return True  # Not critical

    except Exception as e:
        print_status("Gamepad test", False, f"- Error: {e}")
        return True  # Not critical

def test_cameras():
    """Test OAK camera connection"""
    print("\n" + "="*50)
    print("Testing OAK Cameras (Optional)")
    print("="*50)

    try:
        import depthai as dai

        devices = dai.Device.getAllAvailableDevices()

        if len(devices) > 0:
            print_status("OAK cameras detected", True, f"- Found {len(devices)} camera(s)")
            for i, device in enumerate(devices):
                print(f"  ℹ️  Camera {i+1}: {device.getMxId()}")
            return True
        else:
            print_status("OAK cameras detected", False, "- No cameras found")
            print("  ℹ️  This is optional - cameras are used in real-to-sim mode for visualization")
            return True  # Not critical

    except Exception as e:
        print_status("Camera test", False, f"- Error: {e}")
        return True  # Not critical

def main():
    print(colored("\n" + "="*50, "cyan"))
    print(colored("  Digital Twin Setup Verification", "cyan", attrs=['bold']))
    print(colored("="*50, "cyan"))

    results = {}

    # Run all tests
    results['imports'] = test_imports()
    results['model'] = test_mujoco_model()
    results['robot'] = test_robot_connection()
    results['gamepad'] = test_gamepad()
    results['cameras'] = test_cameras()

    # Summary
    print("\n" + colored("="*50, "cyan"))
    print(colored("  Test Summary", "cyan", attrs=['bold']))
    print(colored("="*50, "cyan"))

    critical_tests = ['imports', 'model', 'robot']
    optional_tests = ['gamepad', 'cameras']

    critical_passed = all(results[test] for test in critical_tests if test in results)

    if critical_passed:
        print(colored("\n✅ ALL CRITICAL TESTS PASSED!", "green", attrs=['bold']))
        print("\nYou can now run:")
        print(colored("  python sim_to_real.py", "cyan"))

        if results.get('gamepad', False):
            print(colored("  python real_to_sim.py", "cyan"))
        else:
            print(colored("  python real_to_sim.py", "yellow"), "- Connect gamepad first")
    else:
        print(colored("\n❌ SOME CRITICAL TESTS FAILED", "red", attrs=['bold']))
        print("\nPlease fix the issues above before running the digital twin system.")

        if not results.get('imports', False):
            print("\n📦 Install missing packages:")
            print("  pip install mujoco numpy opencv-python h5py pygame mecademicpy depthai termcolor")

        if not results.get('robot', False):
            print("\n🤖 Robot connection checklist:")
            print("  1. Is the robot powered on?")
            print("  2. Is it connected to network?")
            print("  3. Can you ping 192.168.0.100?")
            print("  4. Is any other software connected to it?")

    print("\n" + colored("="*50, "cyan"))

    # Exit code
    sys.exit(0 if critical_passed else 1)

if __name__ == "__main__":
    main()
